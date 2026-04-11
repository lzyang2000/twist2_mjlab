"""TWIST2 motion-tracking policy — UDP control loop.

Receives state from sim_node via UDP, loads motion reference from PKL,
runs ONNX inference, sends action back.  Pure UDP at 50 Hz.

Unlike the wbc hand_policy (which gets mimic from velocity commands),
the twist2 policy needs a motion library to construct the 35D mimic
observation containing reference joint positions and root state.

Motion playback includes 3-second blend-in/blend-out transitions
to/from the default standing pose (KNEES_BENT_KEYFRAME).
"""

import argparse
import math
import re
import socket
import time
from collections import deque

import numpy as np
import onnxruntime as ort
import torch

from mjlab.asset_zoo.robots import G1_ACTION_SCALE
from mjlab.asset_zoo.robots.unitree_g1.g1_constants import KNEES_BENT_KEYFRAME

from twist2_mjlab.pkl_motion_lib import PklMotionLib

from deploy.common.udp_sync import (
    UDP_HOST, UDP_SIM_PORT, UDP_POLICY_PORT,
    STATE_BYTES, unpack_state, pack_action,
)

# ---------------------------------------------------------------------------
# Constants (must exactly match training)
# ---------------------------------------------------------------------------
POLICY_JOINT_NAMES = [
    "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint",
    "left_knee_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint",
    "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint",
    "right_knee_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint",
    "waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint",
    "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint",
    "left_elbow_joint", "left_wrist_roll_joint", "left_wrist_pitch_joint", "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint",
    "right_elbow_joint", "right_wrist_roll_joint", "right_wrist_pitch_joint", "right_wrist_yaw_joint",
]

# Body names in training order (pelvis=index 0 is root).
MOTION_BODY_NAMES = (
    "pelvis",
    "left_hip_roll_link",
    "left_knee_link",
    "left_ankle_roll_link",
    "right_hip_roll_link",
    "right_knee_link",
    "right_ankle_roll_link",
    "torso_link",
    "left_shoulder_roll_link",
    "left_elbow_link",
    "left_wrist_yaw_link",
    "right_shoulder_roll_link",
    "right_elbow_link",
    "right_wrist_yaw_link",
)

NUM_JOINTS = 29
_ANKLE_DOF_INDICES = {4, 5, 10, 11}

BASE_ANG_VEL_SCALE = 0.25
JOINT_POS_SCALE = 1.0
JOINT_VEL_SCALE = 0.05

ACTOR_MIMIC_DIM = 35
ACTOR_PROPRIO_DIM = 92
ACTOR_CURRENT_DIM = ACTOR_MIMIC_DIM + ACTOR_PROPRIO_DIM  # 127
ACTOR_HISTORY_LENGTH = 11
ACTOR_OBS_DIM = ACTOR_CURRENT_DIM * (1 + ACTOR_HISTORY_LENGTH)  # 1524

BLEND_DURATION = 3.0  # seconds for blend-in / blend-out
DEFAULT_HEIGHT = 0.76


def _resolve_keyframe(joint_names, keyframe):
    vals = np.zeros(len(joint_names), dtype=np.float32)
    for i, name in enumerate(joint_names):
        for pattern, value in keyframe.joint_pos.items():
            if re.fullmatch(pattern, name):
                vals[i] = value
                break
    return vals


def _resolve_scales(joint_names, scale_dict):
    scales = np.zeros(len(joint_names), dtype=np.float32)
    for i, name in enumerate(joint_names):
        for pattern, scale in scale_dict.items():
            if re.match(pattern, name):
                scales[i] = scale
                break
    return scales


DEFAULT_POS = _resolve_keyframe(POLICY_JOINT_NAMES, KNEES_BENT_KEYFRAME)
JOINT_SCALES = _resolve_scales(POLICY_JOINT_NAMES, G1_ACTION_SCALE)
JOINT_VEL_SCALES = np.array(
    [0.0 if i in _ANKLE_DOF_INDICES else JOINT_VEL_SCALE for i in range(NUM_JOINTS)],
    dtype=np.float32,
)

# Default standing mimic: robot at rest in KNEES_BENT pose.
DEFAULT_STANDING_MIMIC = np.zeros(ACTOR_MIMIC_DIM, dtype=np.float32)
DEFAULT_STANDING_MIMIC[2] = DEFAULT_HEIGHT  # motion_root_z
DEFAULT_STANDING_MIMIC[6:] = DEFAULT_POS    # motion_joint_pos (29D at offset 6)

# Default standing ghost pose (for blend interpolation).
DEFAULT_ROOT_POS = np.array([0.0, 0.0, DEFAULT_HEIGHT], dtype=np.float32)
DEFAULT_ROOT_QUAT = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)  # wxyz identity


# ---------------------------------------------------------------------------
# Math
# ---------------------------------------------------------------------------
def euler_roll_pitch_from_quat(q):
    """Extract roll, pitch from wxyz quaternion."""
    w, x, y, z = q
    roll = math.atan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
    pitch = math.asin(max(-1.0, min(1.0, 2.0 * (w * y - z * x))))
    return np.array([roll, pitch], dtype=np.float32)


def quat_rotate_inverse(q, v):
    """Inverse-rotate vector v by wxyz quaternion q."""
    w, x, y, z = q
    q_vec = np.array([x, y, z])
    a = v * (2.0 * w * w - 1.0)
    b = np.cross(q_vec, v) * w * 2.0
    c = q_vec * np.dot(q_vec, v) * 2.0
    return a - b + c


def quat_slerp(q0, q1, t):
    """Spherical linear interpolation between two wxyz quaternions."""
    dot = np.dot(q0, q1)
    if dot < 0:
        q1 = -q1
        dot = -dot
    dot = min(dot, 1.0)
    if dot > 0.9995:
        result = q0 + t * (q1 - q0)
    else:
        theta = math.acos(dot)
        sin_theta = math.sin(theta)
        result = (math.sin((1 - t) * theta) / sin_theta) * q0 + (math.sin(t * theta) / sin_theta) * q1
    return (result / np.linalg.norm(result)).astype(np.float32)


def quat_mul(q1, q2):
    """Hamilton product of two wxyz quaternions."""
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
    ], dtype=np.float32)


def yaw_correction_from_quat(q):
    """Return (correction_quat, cos, sin) that rotates initial yaw to face +X."""
    w, x, y, z = q
    yaw = math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    neg_yaw = -yaw
    half = neg_yaw / 2.0
    corr_quat = np.array([math.cos(half), 0.0, 0.0, math.sin(half)], dtype=np.float32)
    return corr_quat, math.cos(neg_yaw), math.sin(neg_yaw)


# ---------------------------------------------------------------------------
# Motion library wrapper
# ---------------------------------------------------------------------------
class DeployMotionLib:
    """Wraps PklMotionLib for single-env numpy deployment."""

    def __init__(self, motion_file: str, motion_index: int = 0):
        self._lib = PklMotionLib(motion_file, MOTION_BODY_NAMES, device="cpu")
        self._motion_id = motion_index % self._lib.num_motions()
        self._motion_length = float(
            self._lib.get_motion_length(torch.tensor([self._motion_id]))[0].item()
        )
        print(
            f"Motion {self._motion_id}/{self._lib.num_motions()}: "
            f"length={self._motion_length:.2f}s"
        )

    @property
    def motion_length(self) -> float:
        return self._motion_length

    def get_frame(self, motion_time: float) -> dict[str, np.ndarray]:
        """Get interpolated frame at given time, returned as numpy arrays."""
        t = min(motion_time, self._motion_length)
        mid = torch.tensor([self._motion_id], dtype=torch.long)
        mt = torch.tensor([t], dtype=torch.float32)
        frame = self._lib.get_frame(mid, mt)
        return {
            "joint_pos": frame.joint_pos[0].numpy(),
            "body_pos_w": frame.body_pos_w[0].numpy(),
            "body_quat_w": frame.body_quat_w[0].numpy(),
            "body_lin_vel_w": frame.body_lin_vel_w[0].numpy(),
            "body_ang_vel_w": frame.body_ang_vel_w[0].numpy(),
        }


# ---------------------------------------------------------------------------
# Observation building
# ---------------------------------------------------------------------------
def build_mimic_from_frame(frame: dict[str, np.ndarray]) -> np.ndarray:
    """Build 35D mimic observation from a motion library frame."""
    root_pos_w = frame["body_pos_w"][0]
    root_quat_w = frame["body_quat_w"][0]
    root_lin_vel_w = frame["body_lin_vel_w"][0]
    root_ang_vel_w = frame["body_ang_vel_w"][0]

    root_lin_vel_b = quat_rotate_inverse(root_quat_w, root_lin_vel_w)
    root_ang_vel_b = quat_rotate_inverse(root_quat_w, root_ang_vel_w)

    mimic = np.concatenate([
        root_lin_vel_b[:2],                          # motion_root_vel_xy_b (2D)
        [root_pos_w[2]],                             # motion_root_z (1D)
        euler_roll_pitch_from_quat(root_quat_w),     # motion_root_roll_pitch (2D)
        [root_ang_vel_b[2]],                         # motion_root_yaw_ang_vel_b (1D)
        frame["joint_pos"],                          # motion_joint_pos (29D)
    ], dtype=np.float32)
    return mimic


def build_proprio(joint_pos, joint_vel, root_quat, body_ang_vel, last_action):
    """Build 92D proprioceptive observation from robot state."""
    base_ang_vel = body_ang_vel * BASE_ANG_VEL_SCALE
    imu_rp = euler_roll_pitch_from_quat(root_quat)
    joint_pos_rel = (joint_pos - DEFAULT_POS) * JOINT_POS_SCALE
    joint_vel_scaled = joint_vel * JOINT_VEL_SCALES
    proprio = np.concatenate([
        base_ang_vel, imu_rp, joint_pos_rel, joint_vel_scaled, last_action,
    ], dtype=np.float32)
    return proprio


# ---------------------------------------------------------------------------
# Playback state machine
# ---------------------------------------------------------------------------
class PlaybackState:
    """Manages motion playback with blend-in/blend-out phases."""

    BLEND_IN = "blend_in"
    PLAYING = "playing"
    BLEND_OUT = "blend_out"
    IDLE = "idle"

    def __init__(self, motion_lib: DeployMotionLib, blend_sec: float = BLEND_DURATION):
        self.motion_lib = motion_lib
        self.blend_sec = blend_sec
        self.phase = self.BLEND_IN
        self.phase_time = 0.0
        self.motion_time = 0.0
        self._start_frame = motion_lib.get_frame(0.0)
        self._start_mimic = build_mimic_from_frame(self._start_frame)
        self._end_frame: dict[str, np.ndarray] | None = None
        self._end_mimic: np.ndarray | None = None

        # Yaw correction so the ghost always starts facing +X.
        start_quat = self._start_frame["body_quat_w"][0]
        self._yaw_corr_q, self._yaw_cos, self._yaw_sin = yaw_correction_from_quat(start_quat)

    def _correct_root_pose(self, root_pos, root_quat):
        """Rotate root pose by the yaw correction (ghost faces +X at t=0)."""
        p = root_pos.copy()
        x, y = p[0], p[1]
        p[0] = self._yaw_cos * x - self._yaw_sin * y
        p[1] = self._yaw_sin * x + self._yaw_cos * y
        q = quat_mul(self._yaw_corr_q, root_quat)
        return p, q

    def _frame_pose(self, frame: dict[str, np.ndarray]):
        """Extract yaw-corrected (root_pos, root_quat, joint_pos) for ghost."""
        rp, rq = self._correct_root_pose(frame["body_pos_w"][0], frame["body_quat_w"][0])
        return rp, rq, frame["joint_pos"].copy()

    def get_mimic_and_ref_pose(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Return (mimic_35D, ref_root_pos, ref_root_quat, ref_joint_pos)."""
        if self.phase == self.BLEND_IN:
            alpha = min(self.phase_time / self.blend_sec, 1.0)
            mimic = (1.0 - alpha) * DEFAULT_STANDING_MIMIC + alpha * self._start_mimic
            sp, sq, sj = self._frame_pose(self._start_frame)
            ref_pos = (1.0 - alpha) * DEFAULT_ROOT_POS + alpha * sp
            ref_quat = quat_slerp(DEFAULT_ROOT_QUAT, sq, alpha)
            ref_jpos = (1.0 - alpha) * DEFAULT_POS + alpha * sj
            return mimic, ref_pos, ref_quat, ref_jpos

        if self.phase == self.PLAYING:
            frame = self.motion_lib.get_frame(self.motion_time)
            mimic = build_mimic_from_frame(frame)
            self._end_mimic = mimic
            self._end_frame = frame
            rp, rq, rj = self._frame_pose(frame)
            return mimic, rp, rq, rj

        if self.phase == self.BLEND_OUT:
            assert self._end_mimic is not None and self._end_frame is not None
            alpha = min(self.phase_time / self.blend_sec, 1.0)
            mimic = (1.0 - alpha) * self._end_mimic + alpha * DEFAULT_STANDING_MIMIC
            ep, eq, ej = self._frame_pose(self._end_frame)
            ref_pos = (1.0 - alpha) * ep + alpha * DEFAULT_ROOT_POS
            ref_quat = quat_slerp(eq, DEFAULT_ROOT_QUAT, alpha)
            ref_jpos = (1.0 - alpha) * ej + alpha * DEFAULT_POS
            return mimic, ref_pos, ref_quat, ref_jpos

        # IDLE
        return DEFAULT_STANDING_MIMIC.copy(), DEFAULT_ROOT_POS.copy(), DEFAULT_ROOT_QUAT.copy(), DEFAULT_POS.copy()

    def step(self, dt: float) -> None:
        """Advance playback by dt seconds."""
        self.phase_time += dt

        if self.phase == self.BLEND_IN:
            if self.phase_time >= self.blend_sec:
                self.phase = self.PLAYING
                self.phase_time = 0.0
                self.motion_time = 0.0

        elif self.phase == self.PLAYING:
            self.motion_time += dt
            if self.motion_time >= self.motion_lib.motion_length:
                # Cache end frame before transitioning
                end_t = self.motion_lib.motion_length - dt
                self._end_frame = self.motion_lib.get_frame(max(end_t, 0.0))
                self._end_mimic = build_mimic_from_frame(self._end_frame)
                self.phase = self.BLEND_OUT
                self.phase_time = 0.0

        elif self.phase == self.BLEND_OUT:
            if self.phase_time >= self.blend_sec:
                # Restart with blend-in for next loop
                self.phase = self.BLEND_IN
                self.phase_time = 0.0
                self.motion_time = 0.0
                self._start_frame = self.motion_lib.get_frame(0.0)
                self._start_mimic = build_mimic_from_frame(self._start_frame)


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("onnx_path", help="Path to twist2 .onnx model")
    parser.add_argument("--motion-file", required=True, help="Path to .pkl or .yaml motion file")
    parser.add_argument("--motion-index", type=int, default=0, help="Motion index in dataset")
    args = parser.parse_args()

    # Load ONNX
    session = ort.InferenceSession(args.onnx_path, providers=["CPUExecutionProvider"])
    inp_name = session.get_inputs()[0].name
    expected_shape = (1, ACTOR_OBS_DIM)
    actual_shape = tuple(session.get_inputs()[0].shape)
    if actual_shape != expected_shape:
        print(f"WARNING: ONNX input {actual_shape} != expected {expected_shape}")

    # Load motion library
    motion_lib = DeployMotionLib(args.motion_file, args.motion_index)
    playback = PlaybackState(motion_lib)

    # Policy state
    last_action = np.zeros(NUM_JOINTS, dtype=np.float32)
    dt = 1.0 / 50.0
    history: deque[np.ndarray] = deque(maxlen=ACTOR_HISTORY_LENGTH)
    history_initialized = False

    # UDP socket
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind((UDP_HOST, UDP_POLICY_PORT))
    sock.settimeout(1.0)
    sim_addr = (UDP_HOST, UDP_SIM_PORT)
    print(f"TWIST2 policy UDP: listening on {UDP_HOST}:{UDP_POLICY_PORT}")

    next_tick = time.perf_counter()

    try:
        while True:
            # Real-time 50 Hz pacing
            now = time.perf_counter()
            sleep_time = next_tick - now
            if sleep_time > 0:
                time.sleep(sleep_time)
            next_tick += dt

            # 1. Receive state (drain to latest)
            latest_data = None
            sock.setblocking(False)
            try:
                while True:
                    latest_data, _ = sock.recvfrom(STATE_BYTES + 64)
            except BlockingIOError:
                pass
            sock.setblocking(True)
            sock.settimeout(1.0)

            if latest_data is None:
                continue

            step_id, root_quat, root_pos, body_lin_vel, body_ang_vel, \
                joint_pos, joint_vel = unpack_state(latest_data)

            # 2. Build observation
            mimic, ref_root_pos, ref_root_quat, ref_joint_pos = (
                playback.get_mimic_and_ref_pose()
            )
            proprio = build_proprio(
                joint_pos, joint_vel, root_quat, body_ang_vel, last_action,
            )
            actor_current = np.concatenate([mimic, proprio], dtype=np.float32)

            if not history_initialized:
                for _ in range(ACTOR_HISTORY_LENGTH):
                    history.append(actor_current.copy())
                history_initialized = True
            else:
                history.append(actor_current.copy())

            actor_history = np.concatenate(list(history), dtype=np.float32)
            obs = np.concatenate([actor_current, actor_history], dtype=np.float32).reshape(1, -1)

            # 3. Inference
            raw_action = session.run(None, {inp_name: obs})[0][0]

            target_pos = DEFAULT_POS + raw_action * JOINT_SCALES
            last_action = raw_action.copy()

            # 4. Send action + reference pose (for ghost rendering)
            sock.sendto(
                pack_action(
                    step_id,
                    target_pos.astype(np.float32),
                    ref_root_pos.astype(np.float32),
                    ref_root_quat.astype(np.float32),
                    ref_joint_pos.astype(np.float32),
                ),
                sim_addr,
            )

            # 5. Advance motion playback
            playback.step(dt)

    except KeyboardInterrupt:
        pass
    finally:
        sock.close()
        print("TWIST2 policy stopped.")


if __name__ == "__main__":
    main()
