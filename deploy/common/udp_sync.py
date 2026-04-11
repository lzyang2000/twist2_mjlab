"""UDP-based synchronous state/action exchange for twist2 sim2sim.

Protocol (all values float32, little-endian):

  State packet (sim -> policy):  72 floats = 288 bytes
    [step_id(1), quat_wxyz(4), pos_xyz(3), body_lin_vel(3), body_ang_vel(3),
     joint_pos(29), joint_vel(29)]

  Action packet (policy -> sim):  66 floats = 264 bytes
    [step_id(1), target_pos(29),
     ref_root_pos(3), ref_root_quat_wxyz(4), ref_joint_pos(29)]

The action packet includes the reference motion pose so the sim node
can render a ghost overlay without loading the motion library itself.

Ports differ from wbc_mjlab (9870/9871) so both can run simultaneously.
"""

import numpy as np

UDP_SIM_PORT = 9880
UDP_POLICY_PORT = 9881
UDP_HOST = "127.0.0.1"

NUM_JOINTS = 29
STATE_FLOATS = 1 + 4 + 3 + 3 + 3 + NUM_JOINTS + NUM_JOINTS  # 72
ACTION_FLOATS = 1 + NUM_JOINTS + 3 + 4 + NUM_JOINTS  # 66
STATE_BYTES = STATE_FLOATS * 4   # 288
ACTION_BYTES = ACTION_FLOATS * 4  # 264


def pack_state(
    step_id: int,
    root_quat: np.ndarray,
    root_pos: np.ndarray,
    body_lin_vel: np.ndarray,
    body_ang_vel: np.ndarray,
    joint_pos: np.ndarray,
    joint_vel: np.ndarray,
) -> bytes:
    buf = np.empty(STATE_FLOATS, dtype=np.float32)
    buf[0] = float(step_id)
    buf[1:5] = root_quat
    buf[5:8] = root_pos
    buf[8:11] = body_lin_vel
    buf[11:14] = body_ang_vel
    buf[14:43] = joint_pos
    buf[43:72] = joint_vel
    return buf.tobytes()


def unpack_state(
    data: bytes,
) -> tuple[int, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    buf = np.frombuffer(data, dtype=np.float32)
    step_id = int(buf[0])
    root_quat = buf[1:5].copy()
    root_pos = buf[5:8].copy()
    body_lin_vel = buf[8:11].copy()
    body_ang_vel = buf[11:14].copy()
    joint_pos = buf[14:43].copy()
    joint_vel = buf[43:72].copy()
    return step_id, root_quat, root_pos, body_lin_vel, body_ang_vel, joint_pos, joint_vel


def pack_action(
    step_id: int,
    target_pos: np.ndarray,
    ref_root_pos: np.ndarray,
    ref_root_quat: np.ndarray,
    ref_joint_pos: np.ndarray,
) -> bytes:
    buf = np.empty(ACTION_FLOATS, dtype=np.float32)
    buf[0] = float(step_id)
    buf[1:30] = target_pos
    buf[30:33] = ref_root_pos
    buf[33:37] = ref_root_quat
    buf[37:66] = ref_joint_pos
    return buf.tobytes()


def unpack_action(
    data: bytes,
) -> tuple[int, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    buf = np.frombuffer(data, dtype=np.float32)
    step_id = int(buf[0])
    target_pos = buf[1:30].copy()
    ref_root_pos = buf[30:33].copy()
    ref_root_quat = buf[33:37].copy()
    ref_joint_pos = buf[37:66].copy()
    return step_id, target_pos, ref_root_pos, ref_root_quat, ref_joint_pos
