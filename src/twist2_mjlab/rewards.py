"""Reward helpers for TWIST2 tasks."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.entity import Entity
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import ContactSensor
from mjlab.utils.lab_api.math import quat_apply_inverse, quat_error_magnitude, yaw_quat

from twist2_mjlab.observations import FEET_BODY_NAMES, KEY_BODY_NAMES, get_motion_command, tracked_body_indices

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")
_ANKLE_JOINT_NAMES = (
  "left_ankle_pitch_joint",
  "left_ankle_roll_joint",
  "right_ankle_pitch_joint",
  "right_ankle_roll_joint",
)


def _get_robot(env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg) -> Entity:
  return env.scene[asset_cfg.name]


def tracking_joint_dof(
	env: ManagerBasedRlEnv,
	command_name: str = "motion",
	dof_err_w: tuple[float, ...] | None = None,
) -> torch.Tensor:
	command = get_motion_command(env, command_name)
	dof_diff = command.joint_pos - command.robot_joint_pos
	if dof_err_w is None:
		weights = torch.ones(dof_diff.shape[-1], device=env.device, dtype=dof_diff.dtype)
	else:
		weights = torch.tensor(dof_err_w, device=env.device, dtype=dof_diff.dtype)
	dof_err = torch.sum(weights * torch.square(dof_diff), dim=-1)
	return torch.exp(-0.15 * dof_err)


def tracking_joint_vel(
	env: ManagerBasedRlEnv,
	command_name: str = "motion",
	dof_err_w: tuple[float, ...] | None = None,
) -> torch.Tensor:
	command = get_motion_command(env, command_name)
	vel_diff = command.joint_vel - command.robot_joint_vel
	if dof_err_w is None:
		weights = torch.ones(vel_diff.shape[-1], device=env.device, dtype=vel_diff.dtype)
	else:
		weights = torch.tensor(dof_err_w, device=env.device, dtype=vel_diff.dtype)
	vel_err = torch.sum(weights * torch.square(vel_diff), dim=-1)
	return torch.exp(-0.01 * vel_err)


def tracking_root_translation_z(
	env: ManagerBasedRlEnv,
	command_name: str = "motion",
) -> torch.Tensor:
	command = get_motion_command(env, command_name)
	z_err_sq = torch.square(command.body_pos_w[:, 0, 2] - command.robot_body_pos_w[:, 0, 2])
	return torch.exp(-5.0 * z_err_sq)


def tracking_root_rotation(
	env: ManagerBasedRlEnv,
	command_name: str = "motion",
) -> torch.Tensor:
	command = get_motion_command(env, command_name)
	quat_err_sq = torch.square(
		quat_error_magnitude(command.robot_body_quat_w[:, 0], command.body_quat_w[:, 0])
	)
	return torch.exp(-5.0 * quat_err_sq)


def tracking_root_linear_vel(
	env: ManagerBasedRlEnv,
	command_name: str = "motion",
	asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
	del asset_cfg
	command = get_motion_command(env, command_name)
	ref_lin_vel_b = quat_apply_inverse(command.body_quat_w[:, 0], command.body_lin_vel_w[:, 0])
	robot_lin_vel_b = command.robot.data.root_link_lin_vel_b
	vel_err_sq = torch.sum(torch.square(ref_lin_vel_b - robot_lin_vel_b), dim=-1)
	return torch.exp(-1.0 * vel_err_sq)


def tracking_root_angular_vel(
	env: ManagerBasedRlEnv,
	command_name: str = "motion",
	asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
	del asset_cfg
	command = get_motion_command(env, command_name)
	ref_ang_vel_b = quat_apply_inverse(command.body_quat_w[:, 0], command.body_ang_vel_w[:, 0])
	robot_ang_vel_b = command.robot.data.root_link_ang_vel_b
	vel_err_sq = torch.sum(torch.square(ref_ang_vel_b - robot_ang_vel_b), dim=-1)
	return torch.exp(-1.0 * vel_err_sq)


def tracking_keybody_pos(
	env: ManagerBasedRlEnv,
	command_name: str = "motion",
) -> torch.Tensor:
	command = get_motion_command(env, command_name)
	key_body_indices = tracked_body_indices(command)

	robot_root_pos_w = command.robot_body_pos_w[:, 0]
	robot_root_quat_w = command.robot_body_quat_w[:, 0]
	ref_root_pos_w = command.body_pos_w[:, 0]
	ref_root_quat_w = command.body_quat_w[:, 0]

	robot_delta_w = command.robot_body_pos_w[:, key_body_indices] - robot_root_pos_w[:, None, :]
	ref_delta_w = command.body_pos_w[:, key_body_indices] - ref_root_pos_w[:, None, :]

	robot_yaw_quat = yaw_quat(robot_root_quat_w)[:, None, :].expand(-1, len(KEY_BODY_NAMES), -1)
	ref_yaw_quat = yaw_quat(ref_root_quat_w)[:, None, :].expand(-1, len(KEY_BODY_NAMES), -1)

	robot_delta_b = quat_apply_inverse(
		robot_yaw_quat.reshape(-1, 4), robot_delta_w.reshape(-1, 3)
	).reshape(env.num_envs, len(KEY_BODY_NAMES), 3)
	ref_delta_b = quat_apply_inverse(
		ref_yaw_quat.reshape(-1, 4), ref_delta_w.reshape(-1, 3)
	).reshape(env.num_envs, len(KEY_BODY_NAMES), 3)

	key_err_sq = torch.sum(torch.square(robot_delta_b - ref_delta_b), dim=-1).sum(dim=-1)
	return torch.exp(-10.0 * key_err_sq)


def tracking_keybody_pos_global(
	env: ManagerBasedRlEnv,
	command_name: str = "motion",
) -> torch.Tensor:
	command = get_motion_command(env, command_name)
	key_body_indices = tracked_body_indices(command)
	key_err_sq = torch.sum(
		torch.square(
			command.robot_body_pos_w[:, key_body_indices] - command.body_pos_w[:, key_body_indices]
		),
		dim=-1,
	).sum(dim=-1)
	return torch.exp(-10.0 * key_err_sq)


def feet_contact_forces(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  max_contact_force: float = 500.0,
) -> torch.Tensor:
  sensor: ContactSensor = env.scene[sensor_name]
  force = sensor.data.force
  assert force is not None
  vertical_force = torch.abs(force[..., 2])
  return torch.clamp(vertical_force - max_contact_force, min=0.0).sum(dim=1)


def feet_stumble(env: ManagerBasedRlEnv, sensor_name: str) -> torch.Tensor:
  sensor: ContactSensor = env.scene[sensor_name]
  force = sensor.data.force
  assert force is not None
  horizontal = torch.norm(force[..., :2], dim=-1)
  vertical = torch.abs(force[..., 2])
  return torch.any(horizontal > 4.0 * vertical, dim=1).float()


def feet_slip(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
  contact_force_threshold: float = 5.0,
) -> torch.Tensor:
  asset = _get_robot(env, asset_cfg)
  sensor: ContactSensor = env.scene[sensor_name]
  force = sensor.data.force
  assert force is not None
  contact = torch.abs(force[..., 2]) > contact_force_threshold
  foot_ids, _ = asset.find_bodies(FEET_BODY_NAMES, preserve_order=True)
  foot_vel_xy = asset.data.body_link_lin_vel_w[:, foot_ids, :2]
  foot_speed_norm = torch.norm(foot_vel_xy, dim=-1)
  slip = torch.sqrt(torch.clamp(foot_speed_norm, min=0.0))
  return torch.sum(slip * contact.float(), dim=1)


def ang_vel_xy(
	env: ManagerBasedRlEnv,
	asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
	asset = _get_robot(env, asset_cfg)
	return torch.sum(torch.square(asset.data.root_link_ang_vel_b[:, :2]), dim=1)


class dof_torque_limits:
	"""TWIST2 normalized actuator-force-over-limit penalty."""

	def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv):
		self._asset_cfg: SceneEntityCfg = cfg.params.get("asset_cfg", _DEFAULT_ASSET_CFG)
		asset = _get_robot(env, self._asset_cfg)
		actuator_ids = asset.find_actuators((".*",), preserve_order=True)[0]
		self._actuator_ids = torch.tensor(
			actuator_ids, device=env.device, dtype=torch.long
		)

	def __call__(
		self,
		env: ManagerBasedRlEnv,
		soft_torque_limit: float = 0.95,
		asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
	) -> torch.Tensor:
		del asset_cfg
		asset = _get_robot(env, self._asset_cfg)
		actuator_force = torch.abs(asset.data.actuator_force[:, self._actuator_ids])
		force_range = env.sim.model.actuator_forcerange
		if force_range.ndim == 2:
			max_force = force_range[self._actuator_ids, 1].unsqueeze(0)
		else:
			max_force = force_range[:, self._actuator_ids, 1]
		max_force = torch.clamp(max_force, min=1.0e-6)
		over_limit = torch.clamp(actuator_force / max_force - soft_torque_limit, min=0.0)
		return torch.sum(over_limit, dim=1)


class ankle_dof_acc:
	"""TWIST2 ankle-only acceleration penalty."""

	def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv):
		self._asset_cfg: SceneEntityCfg = cfg.params.get("asset_cfg", _DEFAULT_ASSET_CFG)
		asset = _get_robot(env, self._asset_cfg)
		joint_ids = asset.find_joints(_ANKLE_JOINT_NAMES, preserve_order=True)[0]
		self._joint_ids = torch.tensor(joint_ids, device=env.device, dtype=torch.long)

	def __call__(
		self,
		env: ManagerBasedRlEnv,
		asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
	) -> torch.Tensor:
		del asset_cfg
		asset = _get_robot(env, self._asset_cfg)
		return torch.sum(torch.square(asset.data.joint_acc[:, self._joint_ids]), dim=1)


class ankle_dof_vel:
	"""TWIST2 ankle-only velocity penalty."""

	def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv):
		self._asset_cfg: SceneEntityCfg = cfg.params.get("asset_cfg", _DEFAULT_ASSET_CFG)
		asset = _get_robot(env, self._asset_cfg)
		joint_ids = asset.find_joints(_ANKLE_JOINT_NAMES, preserve_order=True)[0]
		self._joint_ids = torch.tensor(joint_ids, device=env.device, dtype=torch.long)

	def __call__(
		self,
		env: ManagerBasedRlEnv,
		asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
	) -> torch.Tensor:
		del asset_cfg
		asset = _get_robot(env, self._asset_cfg)
		return torch.sum(torch.square(asset.data.joint_vel[:, self._joint_ids]), dim=1)


class feet_air_time:
	"""TWIST2 landing-time reward gated by reference motion speed."""

	def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv):
		del cfg
		self.step_dt = env.step_dt

	def __call__(
		self,
		env: ManagerBasedRlEnv,
		sensor_name: str,
		command_name: str = "motion",
		feet_air_time_target: float = 0.5,
	) -> torch.Tensor:
		sensor: ContactSensor = env.scene[sensor_name]
		last_air_time = sensor.data.last_air_time
		assert last_air_time is not None
		first_contact = sensor.compute_first_contact(dt=self.step_dt).float()
		air_time = torch.clamp(last_air_time - feet_air_time_target, max=0.0)
		reward = torch.sum(air_time * first_contact, dim=1)
		command = get_motion_command(env, command_name)
		active = torch.norm(command.body_lin_vel_w[:, 0, :2], dim=1) > 0.05
		return reward * active.float()

	def reset(self, env_ids: torch.Tensor | slice | None) -> None:
		del env_ids

