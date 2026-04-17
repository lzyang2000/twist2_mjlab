"""Robot-safe G1 constants copied from the pinned mjlab revision.

These values mirror:
https://github.com/mujocolab/mjlab/blob/60eca4afee7fd6c2c5da55f6f1943bb4dd41b292/src/mjlab/asset_zoo/robots/unitree_g1/g1_constants.py

They are kept here so the real-hardware deploy path can run without importing
mjlab (which imports warp at module import time).
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class InitialStateCfg:
    pos: tuple[float, float, float]
    joint_pos: dict[str, float]
    joint_vel: dict[str, float]


KNEES_BENT_KEYFRAME = InitialStateCfg(
    pos=(0.0, 0.0, 0.76),
    joint_pos={
        ".*_hip_pitch_joint": -0.312,
        ".*_knee_joint": 0.669,
        ".*_ankle_pitch_joint": -0.363,
        ".*_elbow_joint": 0.6,
        "left_shoulder_roll_joint": 0.2,
        "left_shoulder_pitch_joint": 0.2,
        "right_shoulder_roll_joint": -0.2,
        "right_shoulder_pitch_joint": 0.2,
    },
    joint_vel={".*": 0.0},
)

# Physics-derived PD gains copied from mjlab/docs and the pinned upstream source.
STIFFNESS_5020 = 14.25062309787429
DAMPING_5020 = 0.907222843292423

STIFFNESS_7520_14 = 40.17923863450712
DAMPING_7520_14 = 2.557889775413375

STIFFNESS_7520_22 = 99.09842777666111
DAMPING_7520_22 = 6.308801853496639

STIFFNESS_4010 = 16.77832748089279
DAMPING_4010 = 1.06814150219

G1_ACTION_SCALE: dict[str, float] = {
    ".*_elbow_joint": 0.43857731392336724,
    ".*_shoulder_pitch_joint": 0.43857731392336724,
    ".*_shoulder_roll_joint": 0.43857731392336724,
    ".*_shoulder_yaw_joint": 0.43857731392336724,
    ".*_wrist_roll_joint": 0.43857731392336724,
    ".*_hip_pitch_joint": 0.5475464629911068,
    ".*_hip_yaw_joint": 0.5475464629911068,
    "waist_yaw_joint": 0.5475464629911068,
    ".*_hip_roll_joint": 0.35066146637882434,
    ".*_knee_joint": 0.35066146637882434,
    ".*_wrist_pitch_joint": 0.07450087032950714,
    ".*_wrist_yaw_joint": 0.07450087032950714,
    "waist_pitch_joint": 0.43857731392336724,
    "waist_roll_joint": 0.43857731392336724,
    ".*_ankle_pitch_joint": 0.43857731392336724,
    ".*_ankle_roll_joint": 0.43857731392336724,
}
