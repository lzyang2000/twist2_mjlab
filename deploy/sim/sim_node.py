"""TWIST2 sim2sim node — UDP-synchronous policy exchange.

MuJoCo physics runs locally.  The policy node runs in a separate process
and communicates via UDP (see deploy/common/udp_sync.py).  No ROS dependency.
The reference motion pose sent back by the policy is rendered as a
semi-transparent green ghost overlay in the viewer.
"""

import copy
import math
import os
import re
import socket
import time
from pathlib import Path

import numpy as np
import mujoco
import mujoco.viewer

from mjlab.asset_zoo.robots import get_g1_robot_cfg, G1_ACTION_SCALE
from mjlab.asset_zoo.robots.unitree_g1.g1_constants import KNEES_BENT_KEYFRAME
from mjlab.entity import Entity

from deploy.common.udp_sync import (
    UDP_HOST, UDP_SIM_PORT, UDP_POLICY_PORT,
    ACTION_BYTES, pack_state, unpack_action,
)

# ---------------------------------------------------------------------------
# Constants
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

DECIMATION = 4
TWIST2_G1_XML_ENV = "TWIST2_MJLAB_G1_XML"


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
# Math
# ---------------------------------------------------------------------------
def quat_rotate_inverse(q, v):
    w, x, y, z = q
    q_vec = np.array([x, y, z])
    a = v * (2.0 * w * w - 1.0)
    b = np.cross(q_vec, v) * w * 2.0
    c = q_vec * np.dot(q_vec, v) * 2.0
    return a - b + c


# ---------------------------------------------------------------------------
# MuJoCo model builder
# ---------------------------------------------------------------------------
def build_model():
    xml_override = os.environ.get(TWIST2_G1_XML_ENV, "").strip()
    if xml_override:
        xml_path = Path(xml_override).expanduser()
        if not xml_path.exists():
            raise FileNotFoundError(f"{TWIST2_G1_XML_ENV} -> {xml_path} not found")
        print(f"XML override: {xml_path}")
        robot_cfg = get_g1_robot_cfg()
        robot_cfg.spec_fn = lambda xml_path=xml_path: mujoco.MjSpec.from_file(str(xml_path))
        robot_cfg.articulation = None
        robot_cfg.collisions = ()
        robot_cfg.init_state.joint_pos = None
        return Entity(robot_cfg).spec.compile()

    spec = mujoco.MjSpec()
    spec.option.timestep = 0.005
    spec.option.solver = mujoco.mjtSolver.mjSOL_NEWTON
    spec.option.gravity[:] = [0.0, 0.0, -9.81]

    sky = spec.add_texture()
    sky.type = mujoco.mjtTexture.mjTEXTURE_SKYBOX
    sky.builtin = mujoco.mjtBuiltin.mjBUILTIN_GRADIENT
    sky.rgb1[:] = [0.3, 0.5, 0.7]
    sky.rgb2[:] = [0.0, 0.0, 0.0]
    sky.width = sky.height = 512

    tex = spec.add_texture(name="texplane")
    tex.type = mujoco.mjtTexture.mjTEXTURE_2D
    tex.builtin = mujoco.mjtBuiltin.mjBUILTIN_CHECKER
    tex.rgb1[:] = [0.2, 0.3, 0.4]
    tex.rgb2[:] = [0.1, 0.15, 0.2]
    tex.width = tex.height = 512
    tex.mark = mujoco.mjtMark.mjMARK_CROSS
    tex.markrgb[:] = [0.8, 0.8, 0.8]

    mat = spec.add_material(name="matplane")
    mat.reflectance = 0.3
    mat.textures[mujoco.mjtTextureRole.mjTEXROLE_RGB.value] = tex.name
    mat.texrepeat[:] = [1.0, 1.0]
    mat.texuniform = True

    spec.worldbody.add_light(
        type=mujoco.mjtLightType.mjLIGHT_DIRECTIONAL, castshadow=False,
        pos=(0, 0, 5), dir=(0, 0, -1), diffuse=(0.8, 0.8, 0.8), specular=(0.2, 0.2, 0.2),
    )
    floor = spec.worldbody.add_geom(name="floor")
    floor.type = mujoco.mjtGeom.mjGEOM_PLANE
    floor.size[:] = [0, 0, 0.05]
    floor.material = mat.name

    robot = Entity(get_g1_robot_cfg())
    frame = spec.worldbody.add_frame()
    spec.attach(robot.spec, prefix="", frame=frame)
    return spec.compile()


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------
def main():
    model = build_model()
    data = mujoco.MjData(model)

    # Joint index maps
    qpos_idx, qvel_idx, ctrl_idx = [], [], []
    for name in POLICY_JOINT_NAMES:
        j = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        a = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
        if a == -1:
            raise ValueError(f"Missing actuator for joint {name}")
        qpos_idx.append(model.jnt_qposadr[j])
        qvel_idx.append(model.jnt_dofadr[j])
        ctrl_idx.append(a)
    qpos_idx = np.array(qpos_idx)
    qvel_idx = np.array(qvel_idx)
    ctrl_idx = np.array(ctrl_idx)

    pelvis_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
    assert pelvis_body_id >= 0

    # Ghost model for reference motion visualization
    ghost_model = copy.deepcopy(model)
    ghost_model.geom_rgba[:] = [0.5, 0.7, 0.5, 0.5]  # semi-transparent green
    ghost_data = mujoco.MjData(ghost_model)
    ghost_vopt = mujoco.MjvOption()
    ghost_pert = mujoco.MjvPerturb()

    # Initial pose
    _init_yaw = math.radians(float(os.environ.get("TWIST2_INIT_YAW_DEG", "0")))
    data.qpos[2] = 0.76
    data.qpos[3] = math.cos(_init_yaw / 2)
    data.qpos[6] = math.sin(_init_yaw / 2)
    for i, qi in enumerate(qpos_idx):
        data.qpos[qi] = DEFAULT_POS[i]
    data.ctrl[ctrl_idx] = DEFAULT_POS
    mujoco.mj_forward(model, data)

    # UDP socket
    udp_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    udp_sock.bind((UDP_HOST, UDP_SIM_PORT))
    udp_sock.settimeout(0.05)
    policy_addr = (UDP_HOST, UDP_POLICY_PORT)
    print(f"UDP: sim={UDP_HOST}:{UDP_SIM_PORT} -> policy={UDP_HOST}:{UDP_POLICY_PORT}")

    step_count = 0
    control_dt = model.opt.timestep * DECIMATION

    def _do_reset():
        data.qpos[2] = 0.76
        data.qpos[3] = math.cos(_init_yaw / 2)
        data.qpos[4] = 0.0
        data.qpos[5] = 0.0
        data.qpos[6] = math.sin(_init_yaw / 2)
        for i, qi in enumerate(qpos_idx):
            data.qpos[qi] = DEFAULT_POS[i]
        data.ctrl[ctrl_idx] = DEFAULT_POS
        mujoco.mj_forward(model, data)
        print("Simulation reset.")

    # Viewer + real-time loop
    print("Launching MuJoCo viewer...")
    with mujoco.viewer.launch_passive(model, data) as viewer:
        start_wall = time.perf_counter() - data.time
        prev_sim_time = data.time

        while viewer.is_running():
            # Real-time pacing
            target_wall = start_wall + data.time + control_dt
            sleep_time = target_wall - time.perf_counter()
            if sleep_time > 0:
                time.sleep(sleep_time)

            # Detect GUI reset
            if data.time < prev_sim_time - control_dt * 0.5:
                _do_reset()
                step_count = 0
                start_wall = time.perf_counter() - data.time

            # Pack & send state via UDP
            root_quat = data.qpos[3:7].astype(np.float32)
            root_pos = data.qpos[0:3].astype(np.float32)
            cvel = data.cvel[pelvis_body_id]
            ang_vel_w = cvel[0:3].astype(np.float32)
            lin_vel_c = cvel[3:6].astype(np.float32)
            pos_f = data.xpos[pelvis_body_id].astype(np.float32)
            stcom = data.subtree_com[pelvis_body_id].astype(np.float32)
            lin_vel_w = lin_vel_c - np.cross(ang_vel_w, stcom - pos_f)
            body_ang_vel = quat_rotate_inverse(root_quat, ang_vel_w)
            body_lin_vel = quat_rotate_inverse(root_quat, lin_vel_w)
            joint_pos = data.qpos[qpos_idx].astype(np.float32)
            joint_vel = data.qvel[qvel_idx].astype(np.float32)

            udp_sock.sendto(
                pack_state(step_count, root_quat, root_pos,
                           body_lin_vel, body_ang_vel,
                           joint_pos, joint_vel),
                policy_addr,
            )

            # Recv action + reference pose via UDP
            ref_root_pos = ref_root_quat = ref_joint_pos = None
            try:
                action_data, _ = udp_sock.recvfrom(ACTION_BYTES + 64)
                _, target_pos, ref_root_pos, ref_root_quat, ref_joint_pos = (
                    unpack_action(action_data)
                )
                data.ctrl[ctrl_idx] = target_pos
            except socket.timeout:
                pass

            # Step physics
            for _ in range(DECIMATION):
                mujoco.mj_step(model, data)

            # Render ghost overlay of reference motion
            if ref_root_pos is not None:
                ghost_data.qpos[:] = 0
                ghost_data.qpos[0:3] = ref_root_pos      # root position
                ghost_data.qpos[3:7] = ref_root_quat      # root quaternion (wxyz)
                ghost_data.qpos[qpos_idx] = ref_joint_pos  # joint positions
                mujoco.mj_forward(ghost_model, ghost_data)

                with viewer.lock():
                    viewer.user_scn.ngeom = 0
                    mujoco.mjv_addGeoms(
                        ghost_model, ghost_data, ghost_vopt, ghost_pert,
                        mujoco.mjtCatBit.mjCAT_ALL.value, viewer.user_scn,
                    )

            prev_sim_time = data.time
            step_count += 1
            viewer.sync()


if __name__ == "__main__":
    main()
