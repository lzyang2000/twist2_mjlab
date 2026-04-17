"""G1 real-hardware node — drop-in replacement for sim_node.py.

Reads IMU and joint state from the G1 via unitree_interface, sends it to
the policy node over UDP, receives joint-position targets, and applies them
to the motors.  Same UDP protocol as sim_node; no ROS dependency.

Startup sequence (mirrors g1_wrapper.py from TWIST2):
  1. Press START on the wireless remote  →  interpolate to default pose (2 s)
  2. Press A                             →  enter 50 Hz policy loop
  3. Press B during loop                 →  graceful stop + damp

Usage:
    uv run python deploy/real/hardware_node.py [--net eth0] [--policy-ip 127.0.0.1]
"""

import argparse
import re
import socket
import time

import numpy as np

import unitree_interface
from deploy.common.udp_sync import (
    ACTION_BYTES, UDP_HOST, UDP_POLICY_PORT, UDP_SIM_PORT,
    pack_state, unpack_action,
)
from deploy.real.g1_robot_constants import (
    KNEES_BENT_KEYFRAME,
    STIFFNESS_5020, DAMPING_5020,
    STIFFNESS_7520_14, DAMPING_7520_14,
    STIFFNESS_7520_22, DAMPING_7520_22,
    STIFFNESS_4010, DAMPING_4010,
)

# Wireless controller button map (mirrors TWIST2 g1_wrapper.py ContollerMapping).
# WirelessController.keys is a raw bitmask; use btn(ctrl, "name") to check.
CONTROLLER_MAPPING = {
    "R1": 0x0001, "L1": 0x0002, "start": 0x0004, "select": 0x0008,
    "R2": 0x0010, "L2": 0x0020, "F1":    0x0040, "F2":     0x0080,
    "A":  0x0100, "B":  0x0200, "X":     0x0400, "Y":      0x0800,
    "up": 0x1000, "right": 0x2000, "down": 0x4000, "left":  0x8000,
}


def btn(ctrl, name: str) -> bool:
    return bool(ctrl.keys & CONTROLLER_MAPPING[name])


# ---------------------------------------------------------------------------
# Constants (mirrors sim_node.py)
# ---------------------------------------------------------------------------
POLICY_JOINT_NAMES = [
    "left_hip_pitch_joint",    "left_hip_roll_joint",    "left_hip_yaw_joint",
    "left_knee_joint",         "left_ankle_pitch_joint", "left_ankle_roll_joint",
    "right_hip_pitch_joint",   "right_hip_roll_joint",   "right_hip_yaw_joint",
    "right_knee_joint",        "right_ankle_pitch_joint","right_ankle_roll_joint",
    "waist_yaw_joint",         "waist_roll_joint",       "waist_pitch_joint",
    "left_shoulder_pitch_joint","left_shoulder_roll_joint","left_shoulder_yaw_joint",
    "left_elbow_joint",        "left_wrist_roll_joint",  "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint","right_shoulder_roll_joint","right_shoulder_yaw_joint",
    "right_elbow_joint",       "right_wrist_roll_joint", "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
]
NUM_JOINTS = len(POLICY_JOINT_NAMES)   # 29
CONTROL_DT = 0.02                      # 50 Hz

# ---------------------------------------------------------------------------
# PD gains — mjlab gains per POLICY_JOINT_NAMES order
#
# Motor type per joint index:
#   7520_14 : hip_pitch (0,6), hip_yaw (2,8), waist_yaw (12)
#   7520_22 : hip_roll (1,7), knee (3,9)
#   2×5020  : ankle (4,5,10,11), waist_roll/pitch (13,14)
#   5020    : shoulder (15-17,22-24), elbow (18,25), wrist_roll (19,26)
#   4010    : wrist_pitch (20,27), wrist_yaw (21,28)
# ---------------------------------------------------------------------------
_KP = np.array([
    STIFFNESS_7520_14,   # 0  left_hip_pitch
    STIFFNESS_7520_22,   # 1  left_hip_roll
    STIFFNESS_7520_14,   # 2  left_hip_yaw
    STIFFNESS_7520_22,   # 3  left_knee
    2*STIFFNESS_5020,    # 4  left_ankle_pitch
    2*STIFFNESS_5020,    # 5  left_ankle_roll
    STIFFNESS_7520_14,   # 6  right_hip_pitch
    STIFFNESS_7520_22,   # 7  right_hip_roll
    STIFFNESS_7520_14,   # 8  right_hip_yaw
    STIFFNESS_7520_22,   # 9  right_knee
    2*STIFFNESS_5020,    # 10 right_ankle_pitch
    2*STIFFNESS_5020,    # 11 right_ankle_roll
    STIFFNESS_7520_14,   # 12 waist_yaw
    2*STIFFNESS_5020,    # 13 waist_roll
    2*STIFFNESS_5020,    # 14 waist_pitch
    STIFFNESS_5020,      # 15 left_shoulder_pitch
    STIFFNESS_5020,      # 16 left_shoulder_roll
    STIFFNESS_5020,      # 17 left_shoulder_yaw
    STIFFNESS_5020,      # 18 left_elbow
    STIFFNESS_5020,      # 19 left_wrist_roll
    STIFFNESS_4010,      # 20 left_wrist_pitch
    STIFFNESS_4010,      # 21 left_wrist_yaw
    STIFFNESS_5020,      # 22 right_shoulder_pitch
    STIFFNESS_5020,      # 23 right_shoulder_roll
    STIFFNESS_5020,      # 24 right_shoulder_yaw
    STIFFNESS_5020,      # 25 right_elbow
    STIFFNESS_5020,      # 26 right_wrist_roll
    STIFFNESS_4010,      # 27 right_wrist_pitch
    STIFFNESS_4010,      # 28 right_wrist_yaw
], dtype=np.float64)

_KD = np.array([
    DAMPING_7520_14,     # 0  left_hip_pitch
    DAMPING_7520_22,     # 1  left_hip_roll
    DAMPING_7520_14,     # 2  left_hip_yaw
    DAMPING_7520_22,     # 3  left_knee
    2*DAMPING_5020,      # 4  left_ankle_pitch
    2*DAMPING_5020,      # 5  left_ankle_roll
    DAMPING_7520_14,     # 6  right_hip_pitch
    DAMPING_7520_22,     # 7  right_hip_roll
    DAMPING_7520_14,     # 8  right_hip_yaw
    DAMPING_7520_22,     # 9  right_knee
    2*DAMPING_5020,      # 10 right_ankle_pitch
    2*DAMPING_5020,      # 11 right_ankle_roll
    DAMPING_7520_14,     # 12 waist_yaw
    2*DAMPING_5020,      # 13 waist_roll
    2*DAMPING_5020,      # 14 waist_pitch
    DAMPING_5020,        # 15 left_shoulder_pitch
    DAMPING_5020,        # 16 left_shoulder_roll
    DAMPING_5020,        # 17 left_shoulder_yaw
    DAMPING_5020,        # 18 left_elbow
    DAMPING_5020,        # 19 left_wrist_roll
    DAMPING_4010,        # 20 left_wrist_pitch
    DAMPING_4010,        # 21 left_wrist_yaw
    DAMPING_5020,        # 22 right_shoulder_pitch
    DAMPING_5020,        # 23 right_shoulder_roll
    DAMPING_5020,        # 24 right_shoulder_yaw
    DAMPING_5020,        # 25 right_elbow
    DAMPING_5020,        # 26 right_wrist_roll
    DAMPING_4010,        # 27 right_wrist_pitch
    DAMPING_4010,        # 28 right_wrist_yaw
], dtype=np.float64)

# ---------------------------------------------------------------------------
# Default pose (same source as sim_node.py)
# ---------------------------------------------------------------------------
def _resolve_keyframe(joint_names, keyframe):
    vals = np.zeros(len(joint_names), dtype=np.float32)
    for i, name in enumerate(joint_names):
        for pattern, v in keyframe.joint_pos.items():
            if re.fullmatch(pattern, name):
                vals[i] = v
                break
    return vals


DEFAULT_POS = _resolve_keyframe(POLICY_JOINT_NAMES, KNEES_BENT_KEYFRAME)


# ---------------------------------------------------------------------------
# Motor helpers
# ---------------------------------------------------------------------------
def _send_pd(robot, q_target, kp, kd):
    cmd = robot.create_zero_command()
    cmd.q_target = list(q_target)
    cmd.dq_target = [0.0] * NUM_JOINTS
    cmd.kp = list(kp)
    cmd.kd = list(kd)
    cmd.tau_ff = [0.0] * NUM_JOINTS
    robot.write_low_command(cmd)


# ---------------------------------------------------------------------------
# Startup sequence (mirrors g1_wrapper.py remote logic)
# ---------------------------------------------------------------------------
def startup(robot):
    # Step 1: damp and wait for START to trigger stand-up
    print("Press START on the wireless remote to move to default position ...")
    while True:
        ctrl = robot.read_wireless_controller()
        if btn(ctrl, "start"):
            break
        state = robot.read_low_state()
        current = np.array(state.motor.q[:NUM_JOINTS], dtype=np.float32)
        _send_pd(robot, current, np.zeros(NUM_JOINTS), _KD)
        time.sleep(CONTROL_DT)

    # Step 2: 2-second linear interpolation to DEFAULT_POS
    print("Moving to default position ...")
    state = robot.read_low_state()
    q_start = np.array(state.motor.q[:NUM_JOINTS], dtype=np.float32)
    n_steps = int(2.0 / CONTROL_DT)
    for i in range(n_steps):
        alpha = (i + 1) / n_steps
        q_target = (1.0 - alpha) * q_start + alpha * DEFAULT_POS
        _send_pd(robot, q_target, _KP, _KD)
        time.sleep(CONTROL_DT)
    print("Default position reached.")

    # Step 3: hold default, wait for A to start policy
    print("Press A to start the policy loop ...")
    while True:
        ctrl = robot.read_wireless_controller()
        if btn(ctrl, "A"):
            break
        _send_pd(robot, DEFAULT_POS, _KP, _KD)
        time.sleep(CONTROL_DT)
    print("Starting policy loop.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="G1 real-hardware node")
    parser.add_argument("--net", default="eth0",
                        help="Network interface for robot DDS (default: eth0)")
    parser.add_argument("--policy-ip", default=UDP_HOST,
                        help="IP of the policy node (default: 127.0.0.1)")
    args = parser.parse_args()

    robot = unitree_interface.UnitreeInterface.create_g1(args.net)
    robot.set_control_mode(unitree_interface.ControlMode.PR)

    startup(robot)

    udp_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    udp_sock.bind((UDP_HOST, UDP_SIM_PORT))
    udp_sock.setblocking(False)
    policy_addr = (args.policy_ip, UDP_POLICY_PORT)
    print(f"UDP: hardware={UDP_HOST}:{UDP_SIM_PORT}  policy={args.policy_ip}:{UDP_POLICY_PORT}")

    zeros3 = np.zeros(3, dtype=np.float32)
    last_target = DEFAULT_POS.copy()
    step_count = 0

    try:
        while True:
            t0 = time.perf_counter()

            # Read hardware state
            state = robot.read_low_state()
            quat    = np.array(state.imu.quat,       dtype=np.float32)  # wxyz
            ang_vel = np.array(state.imu.omega,       dtype=np.float32)  # body frame
            jpos    = np.array(state.motor.q[:NUM_JOINTS],  dtype=np.float32)
            jvel    = np.array(state.motor.dq[:NUM_JOINTS], dtype=np.float32)

            # Send state to policy (body_lin_vel and root_pos → zeros: no odometry)
            udp_sock.sendto(
                pack_state(step_count, quat, zeros3, zeros3, ang_vel, jpos, jvel),
                policy_addr,
            )

            # Drain UDP to latest action packet
            latest_raw = None
            try:
                while True:
                    latest_raw, _ = udp_sock.recvfrom(ACTION_BYTES + 64)
            except BlockingIOError:
                pass
            if latest_raw is not None:
                # ref_root_pos / ref_root_quat / ref_joint_pos are ghost-overlay
                # fields used only by the sim viewer — ignore on hardware.
                _, last_target, _, _, _ = unpack_action(latest_raw)

            _send_pd(robot, last_target, _KP, _KD)

            # select → emergency stop: damp at current position, exit immediately
            # B → graceful stop: interpolate to rest in finally block
            ctrl = robot.read_wireless_controller()
            if btn(ctrl, "select"):
                print("SELECT pressed — emergency stop (damp mode).")
                state = robot.read_low_state()
                current = np.array(state.motor.q[:NUM_JOINTS], dtype=np.float32)
                _send_pd(robot, current, np.zeros(NUM_JOINTS), _KD)
                return
            if btn(ctrl, "B"):
                print("B pressed — exiting policy loop.")
                break

            step_count += 1
            elapsed = time.perf_counter() - t0
            time.sleep(max(0.0, CONTROL_DT - elapsed))

    finally:
        udp_sock.close()
        # Return to damped hold at current position
        state = robot.read_low_state()
        current = np.array(state.motor.q[:NUM_JOINTS], dtype=np.float32)
        _send_pd(robot, current, np.zeros(NUM_JOINTS), _KD)


if __name__ == "__main__":
    main()
