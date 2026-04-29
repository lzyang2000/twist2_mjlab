"""Multi-motion library for enriched PKL files.

Loads enriched PKL files (with body_pos_w, body_quat_w) from a YAML dataset
or single PKL. Supports multi-motion sampling, frame interpolation, and
optional CPU offload with a GPU-resident LRU cache.
"""

from __future__ import annotations

import os
import pickle
import sys
import types
from dataclasses import dataclass

import numpy as np
import torch
import yaml
from tqdm import tqdm


if "numpy._core" not in sys.modules:
	sys.modules["numpy._core"] = types.ModuleType("numpy._core")
if "numpy._core.multiarray" not in sys.modules:
	_mc = types.ModuleType("numpy._core.multiarray")
	if hasattr(np.core, "multiarray"):
		_mc._reconstruct = np.core.multiarray._reconstruct  # type: ignore[attr-defined]
	sys.modules["numpy._core.multiarray"] = _mc


@dataclass
class FrameData:
	"""Interpolated frame data for a batch of environments."""

	joint_pos: torch.Tensor
	joint_vel: torch.Tensor
	body_pos_w: torch.Tensor
	body_quat_w: torch.Tensor
	body_lin_vel_w: torch.Tensor
	body_ang_vel_w: torch.Tensor


def _batched_slerp(q0: torch.Tensor, q1: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
	while t.dim() < q0.dim():
		t = t.unsqueeze(-1)

	dot = (q0 * q1).sum(dim=-1, keepdim=True)
	q1 = torch.where(dot < 0, -q1, q1)
	dot = dot.abs().clamp(max=1.0 - 1e-6)

	omega = torch.acos(dot)
	sin_omega = torch.sin(omega)

	small = sin_omega.abs() < 1e-6
	s0 = torch.where(small, 1.0 - t, torch.sin((1.0 - t) * omega) / sin_omega)
	s1 = torch.where(small, t, torch.sin(t * omega) / sin_omega)

	return s0 * q0 + s1 * q1


def _compute_ang_vel_from_quat(quats: torch.Tensor, dt: float) -> torch.Tensor:
	from mjlab.utils.lab_api.math import quat_box_minus

	T = quats.shape[0]
	orig_shape = quats.shape[1:-1]
	flat_q = quats.reshape(T, -1, 4)
	B = flat_q.shape[1]

	omega = torch.zeros(T, B, 3, device=quats.device, dtype=quats.dtype)

	if T < 2:
		return omega.reshape(T, *orig_shape, 3)

	if T == 2:
		omega[0] = quat_box_minus(flat_q[1], flat_q[0]) / dt
		omega[1] = omega[0]
		return omega.reshape(T, *orig_shape, 3)

	omega[1:-1] = (
		quat_box_minus(flat_q[2:].reshape(-1, 4), flat_q[:-2].reshape(-1, 4)).reshape(T - 2, B, 3)
		/ (2.0 * dt)
	)
	omega[0] = quat_box_minus(flat_q[1], flat_q[0]) / dt
	omega[-1] = quat_box_minus(flat_q[-1], flat_q[-2]) / dt

	return omega.reshape(T, *orig_shape, 3)


# Names for the 6 per-frame data arrays, used to iterate uniformly.
_TENSOR_NAMES = (
	"joint_pos", "joint_vel",
	"body_pos_w", "body_quat_w",
	"body_lin_vel_w", "body_ang_vel_w",
)


class PklMotionLib:
	"""Multi-motion library that loads enriched PKL files."""

	def __init__(
		self,
		motion_file: str,
		body_names: tuple[str, ...],
		device: str = "cpu",
		offload_to_cpu: bool = False,
		gpu_cache: bool = False,
		cache_capacity_gb: float = 8.0,
	) -> None:
		self._device = device
		self._offload = offload_to_cpu or gpu_cache
		self._use_cache = gpu_cache
		self._data_device = "cpu" if self._offload else device
		self._cache_capacity_gb = cache_capacity_gb
		self._body_names = body_names

		self._acc_joint_pos: list[torch.Tensor] = []
		self._acc_joint_vel: list[torch.Tensor] = []
		self._acc_body_pos_w: list[torch.Tensor] = []
		self._acc_body_quat_w: list[torch.Tensor] = []
		self._acc_body_lin_vel_w: list[torch.Tensor] = []
		self._acc_body_ang_vel_w: list[torch.Tensor] = []
		self._acc_num_frames: list[int] = []
		self._acc_lengths: list[float] = []
		self._acc_weights: list[float] = []
		self._acc_descriptions: list[str] = []

		self._load_motions(motion_file)

	def _load_motions(self, motion_file: str) -> None:
		files, weights, descriptions = self._parse_motion_file(motion_file)

		for i in tqdm(range(len(files)), desc="[PklMotionLib] Loading motions"):
			path = files[i]
			if not os.path.exists(path):
				print(f"[PklMotionLib] Skipping missing file: {path}")
				continue

			try:
				with open(path, "rb") as f:
					data = pickle.load(f)
			except Exception as e:
				print(f"[PklMotionLib] Error loading {path}: {e}")
				continue

			self._add_motion(data, weights[i], descriptions[i])

		self._finalize()
		print(f"[PklMotionLib] Loaded {self._num_motions} motions, {self._total_frames} total frames.")

	def _parse_motion_file(self, motion_file: str) -> tuple[list[str], list[float], list[str]]:
		if motion_file.endswith(".yaml") or motion_file.endswith(".yml"):
			with open(motion_file) as f:
				config = yaml.safe_load(f)
			root_path = config["root_path"]
			files = [os.path.join(root_path, m["file"]) for m in config["motions"]]
			weights = [m.get("weight", 1.0) for m in config["motions"]]
			descriptions = [m.get("description", "") for m in config["motions"]]
			return files, weights, descriptions
		return [motion_file], [1.0], [""]

	def _add_motion(self, data: dict, weight: float, description: str = "") -> None:
		fps = data["fps"]
		dt = 1.0 / fps

		joint_pos = torch.tensor(np.asarray(data["dof_pos"]), dtype=torch.float32, device=self._data_device)
		T = joint_pos.shape[0]
		if T < 2:
			return

		joint_vel = torch.gradient(joint_pos, spacing=(dt,), dim=0)[0]

		if "body_pos_w" not in data or "body_quat_w" not in data:
			raise ValueError(
				"PKL missing 'body_pos_w' or 'body_quat_w'. Run enrich_pkl.py first to add world-frame body data."
			)

		link_body_list = data["link_body_list"]
		tracked_indices = self._get_tracked_indices(link_body_list)

		all_body_pos_w = torch.tensor(np.asarray(data["body_pos_w"]), dtype=torch.float32, device=self._data_device)
		all_body_quat_w = torch.tensor(np.asarray(data["body_quat_w"]), dtype=torch.float32, device=self._data_device)

		body_pos_w = all_body_pos_w[:, tracked_indices]
		body_quat_w = all_body_quat_w[:, tracked_indices]

		body_lin_vel_w = torch.gradient(body_pos_w, spacing=(dt,), dim=0)[0]
		body_ang_vel_w = _compute_ang_vel_from_quat(body_quat_w, dt)

		motion_length = dt * (T - 1)

		self._acc_joint_pos.append(joint_pos)
		self._acc_joint_vel.append(joint_vel)
		self._acc_body_pos_w.append(body_pos_w)
		self._acc_body_quat_w.append(body_quat_w)
		self._acc_body_lin_vel_w.append(body_lin_vel_w)
		self._acc_body_ang_vel_w.append(body_ang_vel_w)
		self._acc_num_frames.append(T)
		self._acc_lengths.append(motion_length)
		self._acc_weights.append(weight)
		self._acc_descriptions.append(description)

	def _get_tracked_indices(self, link_body_list: list[str]) -> list[int]:
		indices = []
		for name in self._body_names:
			if name not in link_body_list:
				raise ValueError(
					f"Tracked body '{name}' not found in PKL link_body_list: {link_body_list}"
				)
			indices.append(link_body_list.index(name))
		return indices

	# ------------------------------------------------------------------
	# Finalize: build flat arrays and (optionally) GPU cache
	# ------------------------------------------------------------------

	def _finalize(self) -> None:
		if not self._acc_weights:
			raise RuntimeError("[PklMotionLib] No motions loaded!")

		self._num_motions = len(self._acc_weights)
		self._total_frames = sum(self._acc_num_frames)

		if self._offload:
			# CPU pinned only — process one tensor at a time to keep peak memory at ~2x
			# instead of ~3x dataset size. Free each accumulator list before pinning so
			# only the concatenated tensor and its pinned copy coexist briefly.
			for name in _TENSOR_NAMES:
				acc_list = getattr(self, f"_acc_{name}")
				concatenated = torch.cat(acc_list, dim=0)
				setattr(self, f"_acc_{name}", None)
				del acc_list
				setattr(self, f"_all_{name}", concatenated.pin_memory())
				del concatenated
			all_tensors = None
		else:
			# Concatenate per-motion lists into flat arrays (on _data_device).
			all_tensors = {
				"joint_pos": torch.cat(self._acc_joint_pos, dim=0),
				"joint_vel": torch.cat(self._acc_joint_vel, dim=0),
				"body_pos_w": torch.cat(self._acc_body_pos_w, dim=0),
				"body_quat_w": torch.cat(self._acc_body_quat_w, dim=0),
				"body_lin_vel_w": torch.cat(self._acc_body_lin_vel_w, dim=0),
				"body_ang_vel_w": torch.cat(self._acc_body_ang_vel_w, dim=0),
			}

			if self._use_cache:
				# CPU pinned copies + GPU cache.
				for name in _TENSOR_NAMES:
					setattr(self, f"_cpu_{name}", all_tensors[name].pin_memory())
				self._init_cache(all_tensors)
			else:
				# Everything on GPU.
				for name in _TENSOR_NAMES:
					setattr(self, f"_all_{name}", all_tensors[name])

			del self._acc_joint_pos, self._acc_joint_vel
			del self._acc_body_pos_w, self._acc_body_quat_w
			del self._acc_body_lin_vel_w, self._acc_body_ang_vel_w

		# Small metadata — always on GPU.
		self._motion_num_frames = torch.tensor(self._acc_num_frames, dtype=torch.long, device=self._device)
		self._motion_lengths = torch.tensor(self._acc_lengths, dtype=torch.float32, device=self._device)

		weights = torch.tensor(self._acc_weights, dtype=torch.float32, device=self._device)
		self._motion_weights = weights / weights.sum()

		shifted = self._motion_num_frames.roll(1)
		shifted[0] = 0
		self._motion_start_idx = shifted.cumsum(0)

		# CPU copies of metadata for cache miss handling (avoid GPU sync).
		if self._use_cache:
			self._motion_num_frames_cpu = self._motion_num_frames.cpu()
			self._motion_start_idx_cpu = self._motion_start_idx.cpu()

		_ground_keywords = ("crawl", "kneel", "on hands and knees", "on all fours")
		is_ground = [any(kw in desc.lower() for kw in _ground_keywords) for desc in self._acc_descriptions]
		self._is_ground_motion = torch.tensor(is_ground, dtype=torch.bool, device=self._device)

		del self._acc_num_frames, self._acc_lengths, self._acc_weights
		del self._acc_descriptions

	def _init_cache(self, all_tensors: dict[str, torch.Tensor]) -> None:
		"""Pre-allocate GPU cache buffers."""
		# Compute bytes per frame from the actual tensor shapes.
		bytes_per_frame = 0
		self._cache_shapes: dict[str, tuple[int, ...]] = {}
		for name in _TENSOR_NAMES:
			t = all_tensors[name]
			frame_shape = t.shape[1:]  # drop the T dimension
			self._cache_shapes[name] = frame_shape
			bytes_per_frame += int(np.prod(frame_shape)) * t.element_size()

		self._cache_capacity = int(self._cache_capacity_gb * 1e9) // bytes_per_frame
		print(f"[PklMotionLib] GPU cache: {self._cache_capacity} frames "
			  f"({self._cache_capacity_gb:.1f} GB), {bytes_per_frame} B/frame")

		# Pre-allocate GPU cache tensors.
		for name in _TENSOR_NAMES:
			shape = (self._cache_capacity,) + self._cache_shapes[name]
			setattr(self, f"_cache_{name}", torch.zeros(shape, dtype=torch.float32, device=self._device))

		# Cache bookkeeping.
		self._cache_map = torch.full(
			(self._num_motions,), -1, dtype=torch.long, device=self._device,
		)
		self._cache_frames_used: int = 0

	# ------------------------------------------------------------------
	# Cache management
	# ------------------------------------------------------------------

	def _ensure_cached(self, motion_ids: torch.Tensor) -> None:
		"""Load any uncached motions into the GPU cache."""
		unique_ids = motion_ids.unique()
		cached_starts = self._cache_map[unique_ids]
		missing_mask = cached_starts < 0

		if not missing_mask.any():
			return  # 100% cache hit — no CPU sync

		missing_ids_cpu = unique_ids[missing_mask].cpu()
		new_ids: list[int] = []
		new_starts: list[int] = []

		for mid_item in missing_ids_cpu.tolist():
			mid = int(mid_item)
			n_frames = int(self._motion_num_frames_cpu[mid].item())
			src_start = int(self._motion_start_idx_cpu[mid].item())

			# If no room at the end, reset the entire cache.
			if self._cache_frames_used + n_frames > self._cache_capacity:
				# Pre-reset entries are discarded: _reset_cache fills _cache_map
				# with -1 anyway, and their H2D transfers are ordered before the
				# post-reset writes in the same CUDA stream so data is correct.
				self._reset_cache()
				new_ids.clear()
				new_starts.clear()

			dst_start = self._cache_frames_used
			src_end = src_start + n_frames
			dst_end = dst_start + n_frames

			for name in _TENSOR_NAMES:
				src = getattr(self, f"_cpu_{name}")
				dst = getattr(self, f"_cache_{name}")
				dst[dst_start:dst_end] = src[src_start:src_end].to(self._device, non_blocking=True)

			new_ids.append(mid)
			new_starts.append(dst_start)
			self._cache_frames_used += n_frames

		# One batch scatter instead of N scalar writes.
		# Per-element tensor[i]=v on a CUDA tensor dispatches a tiny kernel each
		# time; 4096 of them after a cache reset flood the CUDA command queue and
		# cause every subsequent .any() sync to stall for the full backlog.
		if new_ids:
			self._cache_map[
				torch.tensor(new_ids, dtype=torch.long, device=self._device)
			] = torch.tensor(new_starts, dtype=torch.long, device=self._device)

	def _reset_cache(self) -> None:
		"""Clear the entire cache. Active motions will be re-cached on next access."""
		self._cache_map.fill_(-1)
		self._cache_frames_used = 0

	# ------------------------------------------------------------------
	# Public API
	# ------------------------------------------------------------------

	def num_motions(self) -> int:
		return self._num_motions

	def get_motion_length(self, motion_ids: torch.Tensor) -> torch.Tensor:
		return self._motion_lengths[motion_ids]

	def get_ground_motion_mask(self, motion_ids: torch.Tensor) -> torch.Tensor:
		"""Return a bool tensor [N] that is True for motions involving crawl/kneel/crouch."""
		return self._is_ground_motion[motion_ids]

	def sample_motions(self, n: int) -> torch.Tensor:
		return torch.multinomial(self._motion_weights, num_samples=n, replacement=True)

	def sample_time(self, motion_ids: torch.Tensor) -> torch.Tensor:
		phase = torch.rand(motion_ids.shape, device=self._device)
		lengths = self._motion_lengths[motion_ids]
		return lengths * phase

	def get_frame(self, motion_ids: torch.Tensor, motion_times: torch.Tensor) -> FrameData:
		lengths = self._motion_lengths[motion_ids]
		num_frames = self._motion_num_frames[motion_ids]
		start_idx = self._motion_start_idx[motion_ids]

		motion_times = motion_times % lengths.clamp(min=1e-6)

		phase = (motion_times / lengths).clamp(0.0, 1.0)
		frame_idx_f = phase * (num_frames - 1).float()
		frame_idx0 = frame_idx_f.long()
		frame_idx1 = torch.min(frame_idx0 + 1, num_frames - 1)
		blend = frame_idx_f - frame_idx0.float()

		# Global indices into the flat CPU array.
		idx0 = frame_idx0 + start_idx
		idx1 = frame_idx1 + start_idx

		if self._use_cache:
			# GPU cache path: ensure motions cached, remap indices, gather from cache.
			self._ensure_cached(motion_ids)
			offsets = self._cache_map[motion_ids] - start_idx
			return self._get_frame_gpu(
				self._cache_joint_pos, self._cache_joint_vel,
				self._cache_body_pos_w, self._cache_body_quat_w,
				self._cache_body_lin_vel_w, self._cache_body_ang_vel_w,
				idx0 + offsets, idx1 + offsets, blend,
			)

		if self._offload:
			# Pure CPU offload: gather + blend on CPU, transfer result to GPU.
			return self._get_frame_offload(idx0, idx1, blend)

		# All on GPU.
		return self._get_frame_gpu(
			self._all_joint_pos, self._all_joint_vel,
			self._all_body_pos_w, self._all_body_quat_w,
			self._all_body_lin_vel_w, self._all_body_ang_vel_w,
			idx0, idx1, blend,
		)

	def _get_frame_offload(
		self, idx0: torch.Tensor, idx1: torch.Tensor, blend: torch.Tensor,
	) -> FrameData:
		"""Gather + blend on CPU, transfer results to GPU."""
		dev = self._device
		idx0_c = idx0.cpu()
		idx1_c = idx1.cpu()
		b = blend.cpu().unsqueeze(-1)
		b2 = b.unsqueeze(-1)

		joint_pos = ((1.0 - b) * self._all_joint_pos[idx0_c] + b * self._all_joint_pos[idx1_c]).to(dev, non_blocking=True)
		joint_vel = ((1.0 - b) * self._all_joint_vel[idx0_c] + b * self._all_joint_vel[idx1_c]).to(dev, non_blocking=True)
		body_pos_w = ((1.0 - b2) * self._all_body_pos_w[idx0_c] + b2 * self._all_body_pos_w[idx1_c]).to(dev, non_blocking=True)
		body_quat_w = _batched_slerp(
			self._all_body_quat_w[idx0_c], self._all_body_quat_w[idx1_c], blend.cpu(),
		).to(dev, non_blocking=True)
		body_lin_vel_w = ((1.0 - b2) * self._all_body_lin_vel_w[idx0_c] + b2 * self._all_body_lin_vel_w[idx1_c]).to(dev, non_blocking=True)
		body_ang_vel_w = ((1.0 - b2) * self._all_body_ang_vel_w[idx0_c] + b2 * self._all_body_ang_vel_w[idx1_c]).to(dev, non_blocking=True)

		return FrameData(
			joint_pos=joint_pos,
			joint_vel=joint_vel,
			body_pos_w=body_pos_w,
			body_quat_w=body_quat_w,
			body_lin_vel_w=body_lin_vel_w,
			body_ang_vel_w=body_ang_vel_w,
		)

	@staticmethod
	def _get_frame_gpu(
		all_joint_pos: torch.Tensor,
		all_joint_vel: torch.Tensor,
		all_body_pos_w: torch.Tensor,
		all_body_quat_w: torch.Tensor,
		all_body_lin_vel_w: torch.Tensor,
		all_body_ang_vel_w: torch.Tensor,
		idx0: torch.Tensor,
		idx1: torch.Tensor,
		blend: torch.Tensor,
	) -> FrameData:
		b = blend.unsqueeze(-1)

		joint_pos = (1.0 - b) * all_joint_pos[idx0] + b * all_joint_pos[idx1]
		joint_vel = (1.0 - b) * all_joint_vel[idx0] + b * all_joint_vel[idx1]

		b2 = b.unsqueeze(-1)
		body_pos_w = (1.0 - b2) * all_body_pos_w[idx0] + b2 * all_body_pos_w[idx1]
		body_quat_w = _batched_slerp(all_body_quat_w[idx0], all_body_quat_w[idx1], blend)
		body_lin_vel_w = (1.0 - b2) * all_body_lin_vel_w[idx0] + b2 * all_body_lin_vel_w[idx1]
		body_ang_vel_w = (1.0 - b2) * all_body_ang_vel_w[idx0] + b2 * all_body_ang_vel_w[idx1]

		return FrameData(
			joint_pos=joint_pos,
			joint_vel=joint_vel,
			body_pos_w=body_pos_w,
			body_quat_w=body_quat_w,
			body_lin_vel_w=body_lin_vel_w,
			body_ang_vel_w=body_ang_vel_w,
		)
