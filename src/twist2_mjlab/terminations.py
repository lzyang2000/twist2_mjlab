"""Termination helpers for TWIST2 tasks."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.utils.lab_api.math import euler_xyz_from_quat, quat_apply_inverse

from twist2_mjlab.observations import KEY_BODY_NAMES, get_motion_command

if TYPE_CHECKING:
	from mjlab.envs import ManagerBasedRlEnv


def twist2_motion_end(
	env: ManagerBasedRlEnv, command_name: str = "motion"
) -> torch.Tensor:
	command = get_motion_command(env, command_name)
	motion_lengths = command.motion_lib.get_motion_length(command.motion_ids)
	motion_times = env.episode_length_buf.to(dtype=torch.float32) * env.step_dt
	return motion_times >= motion_lengths


def twist2_root_height_diff(
	env: ManagerBasedRlEnv,
	command_name: str = "motion",
	threshold: float = 0.3,
) -> torch.Tensor:
	command = get_motion_command(env, command_name)
	ref_root_z = command.body_pos_w[:, 0, 2]
	robot_root_z = command.robot.data.root_link_pos_w[:, 2]
	return torch.abs(ref_root_z - robot_root_z) > threshold


def twist2_roll_limit(env: ManagerBasedRlEnv, threshold: float = 4.0) -> torch.Tensor:
	command = get_motion_command(env, "motion")
	roll, _, _ = euler_xyz_from_quat(command.robot.data.root_link_quat_w)
	return torch.abs(roll) > threshold


def twist2_pitch_limit(env: ManagerBasedRlEnv, threshold: float = 4.0) -> torch.Tensor:
	command = get_motion_command(env, "motion")
	_, pitch, _ = euler_xyz_from_quat(command.robot.data.root_link_quat_w)
	return torch.abs(pitch) > threshold


def twist2_velocity_too_large(
	env: ManagerBasedRlEnv, threshold: float = 5.0
) -> torch.Tensor:
	command = get_motion_command(env, "motion")
	return torch.norm(command.robot.data.root_link_lin_vel_w, dim=-1) > threshold


def twist2_pose_fail(
	env: ManagerBasedRlEnv,
	command_name: str = "motion",
	threshold: float = 0.7,
	track_root: bool = False,
	root_tracking_threshold: float = 2.0,
) -> torch.Tensor:
	command = get_motion_command(env, command_name)
	key_body_ids = command.robot.find_bodies(KEY_BODY_NAMES, preserve_order=True)[0]
	key_body_ids = torch.tensor(key_body_ids, device=command.device, dtype=torch.long)

	robot_root_pos = command.robot.data.root_link_pos_w
	robot_root_quat = command.robot.data.root_link_quat_w
	robot_body_pos = command.robot.data.body_link_pos_w[:, key_body_ids]
	robot_body_delta = robot_body_pos - robot_root_pos[:, None, :]
	robot_body_local = quat_apply_inverse(
		robot_root_quat[:, None, :].expand(-1, len(KEY_BODY_NAMES), -1).reshape(-1, 4),
		robot_body_delta.reshape(-1, 3),
	).reshape(env.num_envs, len(KEY_BODY_NAMES), 3)

	ref_root_pos = command.body_pos_w[:, 0]
	ref_root_quat = command.body_quat_w[:, 0]
	body_name_to_idx = {name: i for i, name in enumerate(command.cfg.body_names)}
	ref_key_ids = torch.tensor(
		[body_name_to_idx[name] for name in KEY_BODY_NAMES],
		device=command.device,
		dtype=torch.long,
	)
	ref_body_pos = command.body_pos_w[:, ref_key_ids]
	ref_body_delta = ref_body_pos - ref_root_pos[:, None, :]
	ref_body_local = quat_apply_inverse(
		ref_root_quat[:, None, :].expand(-1, len(KEY_BODY_NAMES), -1).reshape(-1, 4),
		ref_body_delta.reshape(-1, 3),
	).reshape(env.num_envs, len(KEY_BODY_NAMES), 3)

	body_pos_diff = ref_body_local - robot_body_local
	body_pos_dist = torch.sum(torch.square(body_pos_diff), dim=-1)
	pose_fail = torch.max(body_pos_dist, dim=-1).values > threshold**2

	if track_root:
		root_pos_diff = ref_root_pos[:, :2] - robot_root_pos[:, :2]
		root_pos_dist = torch.sum(torch.square(root_pos_diff), dim=-1)
		pose_fail |= root_pos_dist > root_tracking_threshold**2

	return pose_fail & (env.episode_length_buf > 0)
