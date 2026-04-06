"""Observation helpers for TWIST2 tasks."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import torch

from mjlab.envs import mdp as env_mdp
from mjlab.tasks.velocity import mdp as velocity_mdp
from mjlab.utils.lab_api.math import (
	axis_angle_from_quat,
	euler_xyz_from_quat,
	quat_apply_inverse,
	quat_inv,
	quat_mul,
	wrap_to_pi,
	yaw_quat,
)

from twist2_mjlab.commands import PklMotionCommand

if TYPE_CHECKING:
	from mjlab.envs import ManagerBasedRlEnv

PRIV_FUTURE_STEP_OFFSETS: tuple[int, ...] = (
	1,
	5,
	10,
	15,
	20,
	25,
	30,
	35,
	40,
	45,
	50,
	55,
	60,
	65,
	70,
	75,
	80,
	85,
	90,
	95,
)

KEY_BODY_NAMES: tuple[str, ...] = (
	"left_wrist_yaw_link",
	"right_wrist_yaw_link",
	"left_ankle_roll_link",
	"right_ankle_roll_link",
	"left_knee_link",
	"right_knee_link",
	"left_elbow_link",
	"right_elbow_link",
	"torso_link",
)
FOOT_GEOM_PATTERN = r"^(left|right)_foot[1-7]_collision$"
FEET_BODY_NAMES: tuple[str, ...] = ("left_ankle_roll_link", "right_ankle_roll_link")
ACTOR_HISTORY_LENGTH = 11
NUM_G1_JOINTS = 29
ACTOR_MIMIC_DIM = 35
ACTOR_PROPRIO_DIM = 92
ACTOR_HISTORY_FEATURE_DIM = ACTOR_MIMIC_DIM + ACTOR_PROPRIO_DIM
CRITIC_PRIV_STEP_DIM = 21 + NUM_G1_JOINTS + 3 * len(KEY_BODY_NAMES)


def get_motion_command(env: ManagerBasedRlEnv, command_name: str) -> PklMotionCommand:
	return cast(PklMotionCommand, env.command_manager.get_term(command_name))


def tracked_body_indices(command: PklMotionCommand) -> torch.Tensor:
	return torch.tensor(
		[command.cfg.body_names.index(name) for name in KEY_BODY_NAMES],
		device=command.device,
		dtype=torch.long,
	)


def _reference_root_state(
	command: PklMotionCommand,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
	return (
		command.body_pos_w[:, 0],
		command.body_quat_w[:, 0],
		command.body_lin_vel_w[:, 0],
		command.body_ang_vel_w[:, 0],
	)


def _robot_root_state(command: PklMotionCommand) -> tuple[torch.Tensor, torch.Tensor]:
	return command.robot_body_pos_w[:, 0], command.robot_body_quat_w[:, 0]


def _get_robot(env: ManagerBasedRlEnv, command_name: str = "motion"):
	return get_motion_command(env, command_name).robot


def _torso_body_index(env: ManagerBasedRlEnv, command_name: str = "motion") -> int:
	robot = _get_robot(env, command_name)
	torso_body_ids, _ = robot.find_bodies(("torso_link",), preserve_order=True)
	return int(robot.indexing.body_ids[torso_body_ids[0]].item())


def _foot_geom_indices(
	env: ManagerBasedRlEnv, command_name: str = "motion"
) -> torch.Tensor:
	robot = _get_robot(env, command_name)
	foot_geom_ids, _ = robot.find_geoms(FOOT_GEOM_PATTERN)
	return robot.indexing.geom_ids[foot_geom_ids]


def _ctrl_ids(env: ManagerBasedRlEnv, command_name: str = "motion") -> torch.Tensor:
	return _get_robot(env, command_name).indexing.ctrl_ids


def motion_root_vel_xy_b(
	env: ManagerBasedRlEnv, command_name: str = "motion"
) -> torch.Tensor:
	command = get_motion_command(env, command_name)
	_, root_quat_w, root_lin_vel_w, _ = _reference_root_state(command)
	root_lin_vel_b = quat_apply_inverse(root_quat_w, root_lin_vel_w)
	return root_lin_vel_b[:, :2]


def motion_root_z(
	env: ManagerBasedRlEnv, command_name: str = "motion"
) -> torch.Tensor:
	command = get_motion_command(env, command_name)
	root_pos_w, _, _, _ = _reference_root_state(command)
	return root_pos_w[:, 2:3]


def motion_root_roll_pitch(
	env: ManagerBasedRlEnv, command_name: str = "motion"
) -> torch.Tensor:
	command = get_motion_command(env, command_name)
	_, root_quat_w, _, _ = _reference_root_state(command)
	roll, pitch, _ = euler_xyz_from_quat(root_quat_w)
	return torch.stack((roll, pitch), dim=-1)


def motion_root_yaw_ang_vel_b(
	env: ManagerBasedRlEnv, command_name: str = "motion"
) -> torch.Tensor:
	command = get_motion_command(env, command_name)
	_, root_quat_w, _, root_ang_vel_w = _reference_root_state(command)
	root_ang_vel_b = quat_apply_inverse(root_quat_w, root_ang_vel_w)
	return root_ang_vel_b[:, 2:3]


def motion_joint_pos(env: ManagerBasedRlEnv, command_name: str = "motion") -> torch.Tensor:
	return get_motion_command(env, command_name).joint_pos


def imu_roll_pitch(env: ManagerBasedRlEnv) -> torch.Tensor:
	robot = get_motion_command(env, "motion").robot
	roll, pitch, _ = euler_xyz_from_quat(robot.data.root_link_quat_w)
	return torch.stack((roll, pitch), dim=-1)


def critic_root_pos_w(
	env: ManagerBasedRlEnv, command_name: str = "motion"
) -> torch.Tensor:
	command = get_motion_command(env, command_name)
	root_pos_w, _ = _robot_root_state(command)
	return root_pos_w


def critic_root_quat_w(
	env: ManagerBasedRlEnv, command_name: str = "motion"
) -> torch.Tensor:
	command = get_motion_command(env, command_name)
	_, root_quat_w = _robot_root_state(command)
	return root_quat_w


def critic_key_body_pos_b(
	env: ManagerBasedRlEnv, command_name: str = "motion"
) -> torch.Tensor:
	command = get_motion_command(env, command_name)
	key_body_indices = tracked_body_indices(command)
	root_pos_w, root_quat_w = _robot_root_state(command)
	key_body_pos_w = command.robot_body_pos_w[:, key_body_indices]
	delta_w = key_body_pos_w - root_pos_w[:, None, :]
	root_quat_repeat = root_quat_w[:, None, :].expand(-1, len(KEY_BODY_NAMES), -1)
	key_body_pos_b = quat_apply_inverse(
		root_quat_repeat.reshape(-1, 4), delta_w.reshape(-1, 3)
	).reshape(env.num_envs, len(KEY_BODY_NAMES), 3)
	return key_body_pos_b.reshape(env.num_envs, -1)


def critic_foot_contact(
	env: ManagerBasedRlEnv,
	sensor_name: str = "feet_ground_contact",
) -> torch.Tensor:
	return velocity_mdp.foot_contact(env, sensor_name)


def critic_base_com_offset(
	env: ManagerBasedRlEnv, command_name: str = "motion"
) -> torch.Tensor:
	torso_body_idx = _torso_body_index(env, command_name)
	current = env.sim.model.body_ipos[:, torso_body_idx, :]
	default = env.sim.get_default_field("body_ipos")[torso_body_idx].unsqueeze(0)
	return current - default


def critic_foot_friction(
	env: ManagerBasedRlEnv, command_name: str = "motion"
) -> torch.Tensor:
	foot_geom_indices = _foot_geom_indices(env, command_name)
	friction = env.sim.model.geom_friction[:, foot_geom_indices, 0]
	return friction.mean(dim=1, keepdim=True)


def critic_added_mass(
	env: ManagerBasedRlEnv, command_name: str = "motion"
) -> torch.Tensor:
	torso_body_idx = _torso_body_index(env, command_name)
	current = env.sim.model.body_mass[:, torso_body_idx].unsqueeze(-1)
	default = env.sim.get_default_field("body_mass")[torso_body_idx].view(1, 1)
	return current - default


def critic_motor_scales(
	env: ManagerBasedRlEnv, command_name: str = "motion"
) -> torch.Tensor:
	ctrl_ids = _ctrl_ids(env, command_name)
	current_kp = env.sim.model.actuator_gainprm[:, ctrl_ids, 0]
	current_kd = -env.sim.model.actuator_biasprm[:, ctrl_ids, 2]
	default_kp = env.sim.get_default_field("actuator_gainprm")[ctrl_ids, 0].unsqueeze(0)
	default_kd = -env.sim.get_default_field("actuator_biasprm")[ctrl_ids, 2].unsqueeze(0)
	return torch.cat((current_kp / default_kp - 1.0, current_kd / default_kd - 1.0), dim=-1)


def critic_encoder_bias(
	env: ManagerBasedRlEnv, command_name: str = "motion"
) -> torch.Tensor:
	del command_name
	robot = _get_robot(env)
	return robot.data.encoder_bias


def privileged_future_sequence(
	env: ManagerBasedRlEnv,
	command_name: str = "motion",
	step_offsets: tuple[int, ...] = PRIV_FUTURE_STEP_OFFSETS,
) -> torch.Tensor:
	command = get_motion_command(env, command_name)
	offsets = torch.tensor(step_offsets, device=command.device, dtype=torch.float32)
	num_envs = env.num_envs
	num_steps = len(step_offsets)

	motion_ids = command.motion_ids[:, None].expand(-1, num_steps).reshape(-1)
	motion_times = command.motion_times[:, None] + offsets[None, :] * env.step_dt
	frame = command.motion_lib.get_frame(motion_ids, motion_times.reshape(-1))

	body_pos_w = frame.body_pos_w.reshape(num_envs, num_steps, -1, 3)
	body_quat_w = frame.body_quat_w.reshape(num_envs, num_steps, -1, 4)
	body_lin_vel_w = frame.body_lin_vel_w.reshape(num_envs, num_steps, -1, 3)
	body_ang_vel_w = frame.body_ang_vel_w.reshape(num_envs, num_steps, -1, 3)
	joint_pos = frame.joint_pos.reshape(num_envs, num_steps, -1)

	root_pos_w = body_pos_w[:, :, 0]
	root_quat_w = body_quat_w[:, :, 0]
	root_lin_vel_w = body_lin_vel_w[:, :, 0]
	root_ang_vel_w = body_ang_vel_w[:, :, 0]

	flat_root_quat = root_quat_w.reshape(-1, 4)
	root_lin_vel_b = quat_apply_inverse(flat_root_quat, root_lin_vel_w.reshape(-1, 3)).reshape(
		num_envs, num_steps, 3
	)
	root_ang_vel_b = quat_apply_inverse(flat_root_quat, root_ang_vel_w.reshape(-1, 3)).reshape(
		num_envs, num_steps, 3
	)

	roll, pitch, yaw = euler_xyz_from_quat(flat_root_quat)
	root_rpy = torch.stack((roll, pitch, wrap_to_pi(yaw)), dim=-1).reshape(num_envs, num_steps, 3)

	robot_root_pos_w, _ = _robot_root_state(command)
	root_pos_distance_to_target = root_pos_w - robot_root_pos_w[:, None, :]

	current_ref_root_pos_w, current_ref_root_quat_w, _, _ = _reference_root_state(command)
	current_ref_root_pos_w = current_ref_root_pos_w[:, None, :]
	current_ref_root_quat_w = current_ref_root_quat_w[:, None, :]

	root_pos_delta_w = root_pos_w - current_ref_root_pos_w
	root_pos_delta_b = quat_apply_inverse(
		current_ref_root_quat_w.expand(-1, num_steps, -1).reshape(-1, 4),
		root_pos_delta_w.reshape(-1, 3),
	).reshape(num_envs, num_steps, 3)

	root_rot_delta = quat_mul(
		quat_inv(current_ref_root_quat_w.expand(-1, num_steps, -1).reshape(-1, 4)),
		flat_root_quat,
	)
	root_rot_delta_b = axis_angle_from_quat(root_rot_delta).reshape(num_envs, num_steps, 3)

	key_body_indices = tracked_body_indices(command)
	key_body_pos_w = body_pos_w[:, :, key_body_indices]
	key_body_delta_w = key_body_pos_w - root_pos_w[:, :, None, :]
	key_body_quat = root_quat_w[:, :, None, :].expand(-1, -1, len(KEY_BODY_NAMES), -1)
	key_body_pos_b = quat_apply_inverse(
		key_body_quat.reshape(-1, 4), key_body_delta_w.reshape(-1, 3)
	).reshape(num_envs, num_steps, len(KEY_BODY_NAMES) * 3)

	return torch.cat(
		(
			root_pos_w,
			root_pos_distance_to_target,
			root_rpy,
			root_lin_vel_b,
			root_ang_vel_b,
			root_pos_delta_b,
			root_rot_delta_b,
			joint_pos,
			key_body_pos_b,
		),
		dim=-1,
	)


def critic_dr_dim() -> int:
	return 3 + 1 + 1 + 2 * NUM_G1_JOINTS + NUM_G1_JOINTS


def critic_extras_dim() -> int:
	return 3 + 3 + 4 + 3 * len(KEY_BODY_NAMES) + 2 + critic_dr_dim()


def critic_priv_step_dim() -> int:
	return CRITIC_PRIV_STEP_DIM
