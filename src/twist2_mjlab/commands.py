"""PKL motion commands for TWIST2 tasks."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from mjlab.managers import CommandTerm, CommandTermCfg
from mjlab.tasks.tracking.mdp.commands import MotionCommandCfg
from mjlab.utils.lab_api.math import (
	quat_apply,
	quat_error_magnitude,
	quat_from_euler_xyz,
	quat_inv,
	quat_mul,
	sample_uniform,
	yaw_quat,
)

from twist2_mjlab.pkl_motion_lib import PklMotionLib

if TYPE_CHECKING:
	from mjlab.entity import Entity
	from mjlab.envs import ManagerBasedRlEnv


class PklMotionCommand(CommandTerm):
	"""Motion command that loads multi-motion PKL datasets via PklMotionLib."""

	cfg: PklMotionCommandCfg
	_env: ManagerBasedRlEnv

	def __init__(self, cfg: PklMotionCommandCfg, env: ManagerBasedRlEnv):
		super().__init__(cfg, env)

		self.robot: Entity = env.scene[cfg.entity_name]
		self.robot_anchor_body_index = self.robot.body_names.index(
			self.cfg.anchor_body_name
		)
		self.motion_anchor_body_index = self.cfg.body_names.index(
			self.cfg.anchor_body_name
		)
		self.body_indexes = torch.tensor(
			self.robot.find_bodies(self.cfg.body_names, preserve_order=True)[0],
			dtype=torch.long,
			device=self.device,
		)

		self.motion_lib = PklMotionLib(
			self.cfg.motion_file, self.cfg.body_names, device=self.device
		)

		self.motion_ids = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
		self.motion_times = torch.zeros(self.num_envs, device=self.device)

		n_joints = self.robot.data.joint_pos.shape[-1]
		n_bodies = len(cfg.body_names)
		self._joint_pos = torch.zeros(self.num_envs, n_joints, device=self.device)
		self._joint_vel = torch.zeros(self.num_envs, n_joints, device=self.device)
		self._body_pos_w = torch.zeros(self.num_envs, n_bodies, 3, device=self.device)
		self._body_quat_w = torch.zeros(self.num_envs, n_bodies, 4, device=self.device)
		self._body_quat_w[:, :, 0] = 1.0
		self._body_lin_vel_w = torch.zeros(self.num_envs, n_bodies, 3, device=self.device)
		self._body_ang_vel_w = torch.zeros(self.num_envs, n_bodies, 3, device=self.device)

		self.body_pos_relative_w = torch.zeros(
			self.num_envs, n_bodies, 3, device=self.device
		)
		self.body_quat_relative_w = torch.zeros(
			self.num_envs, n_bodies, 4, device=self.device
		)
		self.body_quat_relative_w[:, :, 0] = 1.0

		max_motion_length = self.motion_lib._motion_lengths.max().item()
		self.bin_count = max(int(max_motion_length / self._env.step_dt) + 1, 1)
		self.bin_failed_count = torch.zeros(
			self.bin_count, dtype=torch.float, device=self.device
		)
		self._current_bin_failed = torch.zeros(
			self.bin_count, dtype=torch.float, device=self.device
		)
		self.kernel = torch.tensor(
			[self.cfg.adaptive_lambda**i for i in range(self.cfg.adaptive_kernel_size)],
			device=self.device,
		)
		self.kernel = self.kernel / self.kernel.sum()

		self.metrics["error_anchor_pos"] = torch.zeros(self.num_envs, device=self.device)
		self.metrics["error_anchor_rot"] = torch.zeros(self.num_envs, device=self.device)
		self.metrics["error_anchor_lin_vel"] = torch.zeros(
			self.num_envs, device=self.device
		)
		self.metrics["error_anchor_ang_vel"] = torch.zeros(
			self.num_envs, device=self.device
		)
		self.metrics["error_body_pos"] = torch.zeros(self.num_envs, device=self.device)
		self.metrics["error_body_rot"] = torch.zeros(self.num_envs, device=self.device)
		self.metrics["error_joint_pos"] = torch.zeros(self.num_envs, device=self.device)
		self.metrics["error_joint_vel"] = torch.zeros(self.num_envs, device=self.device)
		self.metrics["sampling_entropy"] = torch.zeros(self.num_envs, device=self.device)
		self.metrics["sampling_top1_prob"] = torch.zeros(self.num_envs, device=self.device)
		self.metrics["sampling_top1_bin"] = torch.zeros(self.num_envs, device=self.device)

	@property
	def command(self) -> torch.Tensor:
		return torch.cat([self._joint_pos, self._joint_vel], dim=1)

	@property
	def joint_pos(self) -> torch.Tensor:
		return self._joint_pos

	@property
	def joint_vel(self) -> torch.Tensor:
		return self._joint_vel

	@property
	def body_pos_w(self) -> torch.Tensor:
		return self._body_pos_w + self._env.scene.env_origins[:, None, :]

	@property
	def body_quat_w(self) -> torch.Tensor:
		return self._body_quat_w

	@property
	def body_lin_vel_w(self) -> torch.Tensor:
		return self._body_lin_vel_w

	@property
	def body_ang_vel_w(self) -> torch.Tensor:
		return self._body_ang_vel_w

	@property
	def anchor_pos_w(self) -> torch.Tensor:
		return (
			self._body_pos_w[:, self.motion_anchor_body_index]
			+ self._env.scene.env_origins
		)

	@property
	def anchor_quat_w(self) -> torch.Tensor:
		return self._body_quat_w[:, self.motion_anchor_body_index]

	@property
	def anchor_lin_vel_w(self) -> torch.Tensor:
		return self._body_lin_vel_w[:, self.motion_anchor_body_index]

	@property
	def anchor_ang_vel_w(self) -> torch.Tensor:
		return self._body_ang_vel_w[:, self.motion_anchor_body_index]

	@property
	def robot_joint_pos(self) -> torch.Tensor:
		return self.robot.data.joint_pos

	@property
	def robot_joint_vel(self) -> torch.Tensor:
		return self.robot.data.joint_vel

	@property
	def robot_body_pos_w(self) -> torch.Tensor:
		return self.robot.data.body_link_pos_w[:, self.body_indexes]

	@property
	def robot_body_quat_w(self) -> torch.Tensor:
		return self.robot.data.body_link_quat_w[:, self.body_indexes]

	@property
	def robot_body_lin_vel_w(self) -> torch.Tensor:
		return self.robot.data.body_link_lin_vel_w[:, self.body_indexes]

	@property
	def robot_body_ang_vel_w(self) -> torch.Tensor:
		return self.robot.data.body_link_ang_vel_w[:, self.body_indexes]

	@property
	def robot_anchor_pos_w(self) -> torch.Tensor:
		return self.robot.data.body_link_pos_w[:, self.robot_anchor_body_index]

	@property
	def robot_anchor_quat_w(self) -> torch.Tensor:
		return self.robot.data.body_link_quat_w[:, self.robot_anchor_body_index]

	@property
	def robot_anchor_lin_vel_w(self) -> torch.Tensor:
		return self.robot.data.body_link_lin_vel_w[:, self.robot_anchor_body_index]

	@property
	def robot_anchor_ang_vel_w(self) -> torch.Tensor:
		return self.robot.data.body_link_ang_vel_w[:, self.robot_anchor_body_index]

	def _update_metrics(self) -> None:
		self.metrics["error_anchor_pos"] = torch.norm(
			self.anchor_pos_w - self.robot_anchor_pos_w, dim=-1
		)
		self.metrics["error_anchor_rot"] = quat_error_magnitude(
			self.anchor_quat_w, self.robot_anchor_quat_w
		)
		self.metrics["error_anchor_lin_vel"] = torch.norm(
			self.anchor_lin_vel_w - self.robot_anchor_lin_vel_w, dim=-1
		)
		self.metrics["error_anchor_ang_vel"] = torch.norm(
			self.anchor_ang_vel_w - self.robot_anchor_ang_vel_w, dim=-1
		)
		self.metrics["error_body_pos"] = torch.norm(
			self.body_pos_relative_w - self.robot_body_pos_w, dim=-1
		).mean(dim=-1)
		self.metrics["error_body_rot"] = quat_error_magnitude(
			self.body_quat_relative_w, self.robot_body_quat_w
		).mean(dim=-1)
		self.metrics["error_joint_pos"] = torch.norm(
			self._joint_pos - self.robot_joint_pos, dim=-1
		)
		self.metrics["error_joint_vel"] = torch.norm(
			self._joint_vel - self.robot_joint_vel, dim=-1
		)

	def _write_reference_state_to_sim(
		self,
		env_ids: torch.Tensor,
		root_pos: torch.Tensor,
		root_ori: torch.Tensor,
		root_lin_vel: torch.Tensor,
		root_ang_vel: torch.Tensor,
		joint_pos: torch.Tensor,
		joint_vel: torch.Tensor,
	) -> None:
		soft_limits = self.robot.data.soft_joint_pos_limits[env_ids]
		joint_pos = torch.clip(joint_pos, soft_limits[:, :, 0], soft_limits[:, :, 1])
		self.robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)

		root_state = torch.cat([root_pos, root_ori, root_lin_vel, root_ang_vel], dim=-1)
		self.robot.write_root_state_to_sim(root_state, env_ids=env_ids)
		self.robot.reset(env_ids=env_ids)

	def _resample_command(self, env_ids: torch.Tensor) -> None:
		n = len(env_ids)

		self.motion_ids[env_ids] = self.motion_lib.sample_motions(n)

		if self.cfg.sampling_mode == "start":
			self.motion_times[env_ids] = 0.0
		elif self.cfg.sampling_mode == "uniform":
			self.motion_times[env_ids] = self.motion_lib.sample_time(self.motion_ids[env_ids])
		else:
			assert self.cfg.sampling_mode == "adaptive"
			self._adaptive_sampling(env_ids)

		frame = self.motion_lib.get_frame(
			self.motion_ids[env_ids], self.motion_times[env_ids]
		)

		root_pos = frame.body_pos_w[:, 0].clone() + self._env.scene.env_origins[env_ids]
		root_ori = frame.body_quat_w[:, 0].clone()
		root_lin_vel = frame.body_lin_vel_w[:, 0].clone()
		root_ang_vel = frame.body_ang_vel_w[:, 0].clone()

		range_list = [
			self.cfg.pose_range.get(key, (0.0, 0.0))
			for key in ["x", "y", "z", "roll", "pitch", "yaw"]
		]
		ranges = torch.tensor(range_list, device=self.device)
		rand_samples = sample_uniform(
			ranges[:, 0], ranges[:, 1], (n, 6), device=self.device
		)
		root_pos += rand_samples[:, 0:3]
		orientations_delta = quat_from_euler_xyz(
			rand_samples[:, 3], rand_samples[:, 4], rand_samples[:, 5]
		)
		root_ori = quat_mul(orientations_delta, root_ori)

		range_list = [
			self.cfg.velocity_range.get(key, (0.0, 0.0))
			for key in ["x", "y", "z", "roll", "pitch", "yaw"]
		]
		ranges = torch.tensor(range_list, device=self.device)
		rand_samples = sample_uniform(
			ranges[:, 0], ranges[:, 1], (n, 6), device=self.device
		)
		root_lin_vel += rand_samples[:, :3]
		root_ang_vel += rand_samples[:, 3:]

		joint_pos = frame.joint_pos.clone()
		joint_vel = frame.joint_vel

		joint_pos += sample_uniform(
			lower=self.cfg.joint_position_range[0],
			upper=self.cfg.joint_position_range[1],
			size=joint_pos.shape,
			device=str(joint_pos.device),
		)

		self._write_reference_state_to_sim(
			env_ids,
			root_pos,
			root_ori,
			root_lin_vel,
			root_ang_vel,
			joint_pos,
			joint_vel,
		)

	def _adaptive_sampling(self, env_ids: torch.Tensor) -> None:
		episode_failed = self._env.termination_manager.terminated[env_ids]
		if torch.any(episode_failed):
			motion_lengths = self.motion_lib.get_motion_length(self.motion_ids[env_ids])
			phase = (self.motion_times[env_ids] / motion_lengths.clamp(min=1e-6)).clamp(
				0, 1
			)
			current_bin_index = (phase * (self.bin_count - 1)).long().clamp(
				0, self.bin_count - 1
			)
			fail_bins = current_bin_index[episode_failed]
			self._current_bin_failed[:] = torch.bincount(
				fail_bins, minlength=self.bin_count
			).float()

		sampling_probabilities = (
			self.bin_failed_count
			+ self.cfg.adaptive_uniform_ratio / float(self.bin_count)
		)
		sampling_probabilities = torch.nn.functional.pad(
			sampling_probabilities.unsqueeze(0).unsqueeze(0),
			(0, self.cfg.adaptive_kernel_size - 1),
			mode="replicate",
		)
		sampling_probabilities = torch.nn.functional.conv1d(
			sampling_probabilities, self.kernel.view(1, 1, -1)
		).view(-1)
		sampling_probabilities = sampling_probabilities / sampling_probabilities.sum()

		sampled_bins = torch.multinomial(
			sampling_probabilities, len(env_ids), replacement=True
		)
		phase = (
			sampled_bins + sample_uniform(0.0, 1.0, (len(env_ids),), device=self.device)
		) / self.bin_count

		self.motion_ids[env_ids] = self.motion_lib.sample_motions(len(env_ids))
		motion_lengths = self.motion_lib.get_motion_length(self.motion_ids[env_ids])
		self.motion_times[env_ids] = phase * motion_lengths

		H = -(sampling_probabilities * (sampling_probabilities + 1e-12).log()).sum()
		H_norm = H / math.log(self.bin_count) if self.bin_count > 1 else 1.0
		pmax, imax = sampling_probabilities.max(dim=0)
		self.metrics["sampling_entropy"][:] = H_norm
		self.metrics["sampling_top1_prob"][:] = pmax
		self.metrics["sampling_top1_bin"][:] = imax.float() / self.bin_count

	def update_relative_body_poses(self) -> None:
		n_bodies = len(self.cfg.body_names)
		anchor_pos_w_repeat = self.anchor_pos_w[:, None, :].repeat(1, n_bodies, 1)
		anchor_quat_w_repeat = self.anchor_quat_w[:, None, :].repeat(1, n_bodies, 1)
		robot_anchor_pos_w_repeat = self.robot_anchor_pos_w[:, None, :].repeat(
			1, n_bodies, 1
		)
		robot_anchor_quat_w_repeat = self.robot_anchor_quat_w[:, None, :].repeat(
			1, n_bodies, 1
		)

		delta_pos_w = robot_anchor_pos_w_repeat.clone()
		delta_pos_w[..., 2] = anchor_pos_w_repeat[..., 2]
		delta_ori_w = yaw_quat(
			quat_mul(robot_anchor_quat_w_repeat, quat_inv(anchor_quat_w_repeat))
		)

		self.body_quat_relative_w = quat_mul(delta_ori_w, self.body_quat_w)
		self.body_pos_relative_w = delta_pos_w + quat_apply(
			delta_ori_w, self.body_pos_w - anchor_pos_w_repeat
		)

	def _update_command(self) -> None:
		self.motion_times += self._env.step_dt

		motion_lengths = self.motion_lib.get_motion_length(self.motion_ids)
		env_ids = torch.where(self.motion_times >= motion_lengths)[0]
		if env_ids.numel() > 0:
			self._resample_command(env_ids)

		frame = self.motion_lib.get_frame(self.motion_ids, self.motion_times)
		self._joint_pos = frame.joint_pos
		self._joint_vel = frame.joint_vel
		self._body_pos_w = frame.body_pos_w
		self._body_quat_w = frame.body_quat_w
		self._body_lin_vel_w = frame.body_lin_vel_w
		self._body_ang_vel_w = frame.body_ang_vel_w

		self.update_relative_body_poses()

		if self.cfg.sampling_mode == "adaptive":
			self.bin_failed_count = (
				self.cfg.adaptive_alpha * self._current_bin_failed
				+ (1 - self.cfg.adaptive_alpha) * self.bin_failed_count
			)
			self._current_bin_failed.zero_()


@dataclass(kw_only=True)
class PklMotionCommandCfg(MotionCommandCfg):
	"""Configuration for PKL motion command."""

	def build(self, env: ManagerBasedRlEnv) -> PklMotionCommand:
		return PklMotionCommand(self, env)
