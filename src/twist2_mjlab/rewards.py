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
_HIP_PITCH_JOINT_NAMES = (
  "left_hip_pitch_joint",
  "right_hip_pitch_joint",
)
_HIP_ROLL_JOINT_NAMES = (
  "left_hip_roll_joint",
  "right_hip_roll_joint",
)
_G = 9.81  # m/s²


def _get_robot(env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg) -> Entity:
  return env.scene[asset_cfg.name]


# ---------------------------------------------------------------------------
# Stability metric helpers (whole-body CoM, capture point, momentum)
# ---------------------------------------------------------------------------


def _body_masses(asset: Entity) -> torch.Tensor:
  """Per-entity-body masses. Shape [num_envs, n_bodies]."""
  return asset.data.model.body_mass[:, asset.data.indexing.body_ids]


def _whole_body_com_pos(asset: Entity) -> torch.Tensor:
  """Whole-body CoM position in world frame. Shape [num_envs, 3]."""
  masses = _body_masses(asset)                            # [N, B]
  total_mass = masses.sum(dim=1, keepdim=True).clamp_min(1e-8)
  return (masses.unsqueeze(-1) * asset.data.body_com_pos_w).sum(dim=1) / total_mass


def _whole_body_com_vel(asset: Entity) -> torch.Tensor:
  """Whole-body CoM velocity in world frame. Shape [num_envs, 3]."""
  masses = _body_masses(asset)                            # [N, B]
  total_mass = masses.sum(dim=1, keepdim=True).clamp_min(1e-8)
  return (masses.unsqueeze(-1) * asset.data.body_com_lin_vel_w).sum(dim=1) / total_mass


def _capture_point(asset: Entity) -> torch.Tensor:
  """LIPM capture point in world XY. Shape [num_envs, 2].

  CP = CoM_xy + CoM_vel_xy / sqrt(g / h) where h = CoM height.
  Valid for flat ground; use a ray-caster for uneven terrain.
  """
  com_pos = _whole_body_com_pos(asset)                    # [N, 3]
  com_vel = _whole_body_com_vel(asset)                    # [N, 3]
  h = com_pos[:, 2].clamp(min=1e-3)
  omega = torch.sqrt(torch.tensor(_G, device=h.device, dtype=h.dtype) / h)
  return com_pos[:, :2] + com_vel[:, :2] / omega.unsqueeze(-1)  # [N, 2]


def _linear_momentum(asset: Entity) -> torch.Tensor:
  """Total linear momentum p = Σ m_i * v_i. Shape [num_envs, 3]."""
  masses = _body_masses(asset)                            # [N, B]
  return (masses.unsqueeze(-1) * asset.data.body_com_lin_vel_w).sum(dim=1)


def _angular_momentum(asset: Entity) -> torch.Tensor:
  """Orbital angular momentum L = Σ (r_i − r_CoM) × (m_i v_i).

  Spin term (I_i ω_i) is omitted for GPU performance (~10–20 % error vs full
  CAM). Use Pinocchio offline if exact centroidal angular momentum is needed.
  Shape [num_envs, 3].
  """
  masses = _body_masses(asset)                            # [N, B]
  total_mass = masses.sum(dim=1, keepdim=True).clamp_min(1e-8)
  body_pos = asset.data.body_com_pos_w                    # [N, B, 3]
  body_vel = asset.data.body_com_lin_vel_w                # [N, B, 3]
  com_pos = (masses.unsqueeze(-1) * body_pos).sum(dim=1) / total_mass  # [N, 3]
  rel_pos = body_pos - com_pos.unsqueeze(1)               # [N, B, 3]
  return torch.linalg.cross(
    rel_pos, masses.unsqueeze(-1) * body_vel, dim=-1
  ).sum(dim=1)                                            # [N, 3]


def _support_polygon_dist(
  query_xy: torch.Tensor,       # [N, 2]
  contact_pos_xy: torch.Tensor, # [N, M, 2]
  contact_mask: torch.Tensor,   # [N, M] bool
  tolerance: float,
) -> torch.Tensor:
  """Distance from query_xy to tolerance-expanded bounding box of active contacts.

  Returns 0 when query_xy is inside the bounding box + tolerance margin.
  """
  _large = 1e6
  # x bounds
  x = contact_pos_xy[..., 0]
  x_min = torch.where(contact_mask, x, torch.full_like(x,  _large)).min(dim=1).values
  x_max = torch.where(contact_mask, x, torch.full_like(x, -_large)).max(dim=1).values
  # y bounds
  y = contact_pos_xy[..., 1]
  y_min = torch.where(contact_mask, y, torch.full_like(y,  _large)).min(dim=1).values
  y_max = torch.where(contact_mask, y, torch.full_like(y, -_large)).max(dim=1).values
  # signed distance outside box (negative = inside)
  dx = torch.clamp_min(
    torch.maximum(x_min - query_xy[:, 0] - tolerance, query_xy[:, 0] - x_max - tolerance), 0.0
  )
  dy = torch.clamp_min(
    torch.maximum(y_min - query_xy[:, 1] - tolerance, query_xy[:, 1] - y_max - tolerance), 0.0
  )
  return torch.sqrt(dx**2 + dy**2)


# ---------------------------------------------------------------------------
# Stability metric rewards
# ---------------------------------------------------------------------------


def simplified_com_in_support_polygon(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
  feet_sensor_cfg: SceneEntityCfg = SceneEntityCfg("feet_ground_contact"),
  hands_sensor_cfg: SceneEntityCfg | None = None,
  tolerance: float = 0.05,
  std: float = 0.1,
) -> torch.Tensor:
  """Reward whole-body CoM staying inside the convex hull of active contacts.

  Adapted from IHMC IsaacLab for mjlab / Unitree G1.
  The support polygon is approximated by the axis-aligned bounding box of
  active contact points (fast, GPU-friendly, negligible error for 2-foot stance).

  ``hands_sensor_cfg`` extends the support polygon when hands are in contact
  (e.g. during fall recovery).  Pass ``SceneEntityCfg("contact_sensor_hands")``
  and add the corresponding sensor to the scene config to enable this.
  """
  asset = _get_robot(env, asset_cfg)
  feet_sensor: ContactSensor = env.scene[feet_sensor_cfg.name]

  force = feet_sensor.data.force            # [N, n_feet, 3]
  assert force is not None
  foot_mask = force.norm(dim=-1) > 20.0     # [N, n_feet]

  foot_ids, _ = asset.find_bodies(FEET_BODY_NAMES, preserve_order=True)
  foot_pos_xy = asset.data.body_link_pos_w[:, foot_ids, :2]  # [N, n_feet, 2]

  contact_pos = foot_pos_xy
  contact_mask = foot_mask

  if hands_sensor_cfg is not None:
    hand_sensor: ContactSensor = env.scene[hands_sensor_cfg.name]
    hand_force = hand_sensor.data.force
    if hand_force is not None:
      _HAND_BODY_NAMES = ("left_wrist_yaw_link", "right_wrist_yaw_link")
      hand_ids, _ = asset.find_bodies(_HAND_BODY_NAMES, preserve_order=True)
      hand_pos_xy = asset.data.body_link_pos_w[:, hand_ids, :2]
      hand_mask = hand_force.norm(dim=-1) > 20.0
      contact_pos = torch.cat([contact_pos, hand_pos_xy], dim=1)
      contact_mask = torch.cat([contact_mask, hand_mask], dim=1)

  com_xy = _whole_body_com_pos(asset)[:, :2]
  has_contact = contact_mask.any(dim=1)
  dist = _support_polygon_dist(com_xy, contact_pos, contact_mask, tolerance)
  return torch.exp(-(dist**2) / (std**2)) * has_contact.float()


def simplified_capture_point_in_support_polygon(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
  feet_sensor_cfg: SceneEntityCfg = SceneEntityCfg("feet_ground_contact"),
  wall_sensor_cfg: SceneEntityCfg | None = None,
  table_sensor_cfg: SceneEntityCfg | None = None,
  tolerance: float = 0.05,
  std: float = 0.1,
) -> torch.Tensor:
  """Reward LIPM capture point staying inside the active foot support polygon.

  Adapted from IHMC IsaacLab for mjlab / Unitree G1.

  When the robot has environmental contact (wall or table), bypasses the
  foot-polygon check and returns max reward — those envs are legitimately
  stable through multi-surface support.

  To enable the wall/table bypass:
    - Add ``contact_sensor_wall`` / ``contact_sensor_table`` sensors to the scene
    - Pass ``SceneEntityCfg("contact_sensor_wall")`` / ``SceneEntityCfg("contact_sensor_table")``
  """
  asset = _get_robot(env, asset_cfg)
  feet_sensor: ContactSensor = env.scene[feet_sensor_cfg.name]

  force = feet_sensor.data.force            # [N, n_feet, 3]
  assert force is not None
  foot_mask = force.norm(dim=-1) > 50.0     # [N, n_feet]

  foot_ids, _ = asset.find_bodies(FEET_BODY_NAMES, preserve_order=True)
  foot_pos_xy = asset.data.body_link_pos_w[:, foot_ids, :2]  # [N, n_feet, 2]

  cp_xy = _capture_point(asset)             # [N, 2]
  has_contact = foot_mask.any(dim=1)
  dist = _support_polygon_dist(cp_xy, foot_pos_xy, foot_mask, tolerance)
  reward = torch.exp(-(dist**2) / (std**2)) * has_contact.float()

  # Environmental contact bypass — robot is stable without foot-only support
  if wall_sensor_cfg is not None:
    wall_sensor: ContactSensor = env.scene[wall_sensor_cfg.name]
    wf = wall_sensor.data.force
    if wf is not None:
      wall_active = wf.norm(dim=-1).sum(dim=-1) > 10.0
      reward = torch.where(wall_active, torch.ones_like(reward), reward)

  if table_sensor_cfg is not None:
    table_sensor: ContactSensor = env.scene[table_sensor_cfg.name]
    tf = table_sensor.data.force
    if tf is not None:
      table_active = tf.norm(dim=-1).sum(dim=-1) > 10.0
      reward = torch.where(table_active, torch.ones_like(reward), reward)

  return reward


def _cop_penalty(sat: torch.Tensor, alpha: float) -> torch.Tensor:
  """Reciprocal CoP-style penalty: peaks at saturation=1 (friction cone limit)."""
  return alpha / ((sat - 1.0) ** 2 + alpha)


class AnkleHipStepReward:
  """Ankle → Hip → Step strategy reward hierarchy for the G1.

  Adapted from IHMC IsaacLab for mjlab.  Joint IDs and actuator limits are
  cached at ``__init__`` to avoid per-step string matching across 4096 envs.

  Reward levels (all additive):
    • **Ankle** – torque opposes CoM drift and stays within friction cone.
    • **Hip**   – activated when ankle torque saturates.
    • **Step**  – swing foot velocity toward capture point when both ankle + hip
      strategies are saturated.
  """

  def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv) -> None:
    self._asset_cfg: SceneEntityCfg = cfg.params.get("asset_cfg", _DEFAULT_ASSET_CFG)
    asset = _get_robot(env, self._asset_cfg)

    # Cache DOF indices (joint space, used with qfrc_actuator)
    self._ap_ids = asset.find_joints(".*ankle_pitch.*")[0]
    self._ar_ids = asset.find_joints(".*ankle_roll.*")[0]
    self._hp_ids = asset.find_joints(".*hip_pitch.*")[0]
    self._hr_ids = asset.find_joints(".*hip_roll.*")[0]

    # Cache actuator indices for effort limits
    self._ankle_act_ids = asset.find_actuators(".*ankle.*")[0]
    self._hip_p_act_ids = asset.find_actuators(".*hip_pitch.*")[0]
    self._hip_r_act_ids = asset.find_actuators(".*hip_roll.*")[0]

    # Cache foot body indices for step-level capture-point check
    self._foot_ids, _ = asset.find_bodies(FEET_BODY_NAMES, preserve_order=True)

    feet_sensor_cfg: SceneEntityCfg = cfg.params["feet_sensor_cfg"]
    self._sensor_name: str = feet_sensor_cfg.name
    self._friction_coeff: float = cfg.params.get("friction_coeff", 0.7)
    self._cop_alpha: float = cfg.params.get("cop_alpha", 1e-3)
    self._contact_threshold: float = cfg.params.get("contact_force_threshold", 20.0)

  def _effort_limit(self, env: ManagerBasedRlEnv, act_ids: list[int]) -> torch.Tensor:
    """Max actuator force for the given actuator IDs. Returns scalar or [N] tensor."""
    fr = env.sim.model.actuator_forcerange
    if fr.ndim == 2:
      return fr[act_ids, 1].abs().max()
    return fr[:, act_ids, 1].abs().max(dim=-1).values  # [N]

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
    feet_sensor_cfg: SceneEntityCfg = SceneEntityCfg("feet_ground_contact"),
    friction_coeff: float = 0.7,
    cop_alpha: float = 1e-3,
    contact_force_threshold: float = 20.0,
  ) -> torch.Tensor:
    asset = _get_robot(env, self._asset_cfg)
    feet_sensor: ContactSensor = env.scene[self._sensor_name]

    # --- Effort limits (actuator space) -------------------------------------
    ankle_limit = self._effort_limit(env, self._ankle_act_ids)
    hip_p_limit = self._effort_limit(env, self._hip_p_act_ids)
    hip_r_limit = self._effort_limit(env, self._hip_r_act_ids)

    # --- Joint-space torques (qfrc_actuator: DOF forces from actuators) -----
    tau = asset.data.qfrc_actuator        # [N, nv]

    ap_tau = tau[:, self._ap_ids].sum(dim=1)
    ar_tau = tau[:, self._ar_ids].sum(dim=1)
    hp_tau = tau[:, self._hp_ids].sum(dim=1)
    hr_tau = tau[:, self._hr_ids].sum(dim=1)

    # --- Contact forces for friction-cone saturation ------------------------
    force = feet_sensor.data.force        # [N, n_feet, 3]
    assert force is not None
    fx, fy = force[..., 0], force[..., 1]
    fz = force[..., 2].clamp(min=0.0)
    tangential = torch.sqrt(fx**2 + fy**2)
    contact_mask = fz > self._contact_threshold

    # Friction cone saturation: 0 = well inside, 1 = at limit
    margin = self._friction_coeff * fz - tangential
    cone_sat = (
      1.0 - (margin / (self._friction_coeff * fz + 1e-6)).clamp(0.0, 1.0)
    ).max(dim=1).values                   # [N] worst foot

    # Normalize torques and apply friction cone saturation cap
    ankle_ps = torch.clamp(ap_tau.abs() / ankle_limit, 0, 1)
    ankle_rs = torch.clamp(ar_tau.abs() / ankle_limit, 0, 1)
    hip_ps   = torch.clamp(hp_tau.abs() / hip_p_limit, 0, 1)
    hip_rs   = torch.clamp(hr_tau.abs() / hip_r_limit, 0, 1)
    ankle_ps = torch.maximum(ankle_ps, cone_sat)
    ankle_rs = torch.maximum(ankle_rs, cone_sat)
    hip_ps   = torch.maximum(hip_ps,   cone_sat)
    hip_rs   = torch.maximum(hip_rs,   cone_sat)

    # --- CoM drift direction ------------------------------------------------
    com_vel_xy = asset.data.root_com_lin_vel_w[:, :2]    # [N, 2]
    speed = com_vel_xy.norm(dim=1).clamp(min=1e-6)       # [N]
    com_dir = com_vel_xy / speed.unsqueeze(1)             # [N, 2]

    # --- Level 1: Ankle strategy -------------------------------------------
    r_ankle = (
      torch.tanh(-com_dir[:, 0] * ap_tau / ankle_limit).clamp(min=0)
      * (1 - _cop_penalty(ankle_ps, self._cop_alpha))
      + torch.tanh(-com_dir[:, 1] * ar_tau / ankle_limit).clamp(min=0)
      * (1 - _cop_penalty(ankle_rs, self._cop_alpha))
    )

    # --- Level 2: Hip strategy (gated by ankle saturation) -----------------
    r_hip = (
      torch.tanh(-com_dir[:, 0] * hp_tau / hip_p_limit).clamp(min=0)
      * ankle_ps**2
      * (1 - _cop_penalty(hip_ps, self._cop_alpha))
      + torch.tanh(-com_dir[:, 1] * hr_tau / hip_r_limit).clamp(min=0)
      * ankle_rs**2
      * (1 - _cop_penalty(hip_rs, self._cop_alpha))
    )

    # --- Level 3: Step strategy (gated by ankle+hip saturation) -------------
    cp_xy = _capture_point(asset)                         # [N, 2]
    foot_pos_xy = asset.data.body_link_pos_w[:, self._foot_ids, :2]   # [N, 2, 2]
    foot_vel_xy = asset.data.body_link_lin_vel_w[:, self._foot_ids, :2]  # [N, 2, 2]

    to_cp = cp_xy.unsqueeze(1) - foot_pos_xy             # [N, 2, 2]
    to_cp_dir = to_cp / to_cp.norm(dim=-1, keepdim=True).clamp(min=1e-6)
    vel_toward = (foot_vel_xy * to_cp_dir).sum(dim=-1).clamp(min=0.0)  # [N, 2]

    swing_mask = (~contact_mask).float()                  # [N, n_feet]
    step_quality = torch.tanh(
      (vel_toward * swing_mask).max(dim=1).values / 0.5
    )
    step_urgency = torch.minimum(ankle_ps * hip_ps, ankle_rs * hip_rs)
    r_step = step_quality * step_urgency

    return r_ankle + r_hip + r_step

  def reset(self, env_ids: torch.Tensor | slice | None) -> None:
    pass


class LinearMomentumChangePenalty:
  """Penalize rate of change of whole-body linear momentum (F = dp/dt).

  Adapted from IHMC IsaacLab for mjlab.  Returns a *negative* squared
  penalty; pair with a positive weight in the reward config.

  Spin angular momentum is excluded (orbital term only) for GPU performance.
  """

  def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv) -> None:
    self._asset_cfg: SceneEntityCfg = cfg.params.get("asset_cfg", _DEFAULT_ASSET_CFG)
    self._env = env
    asset = _get_robot(env, self._asset_cfg)
    self._prev_lin_mom: torch.Tensor = _linear_momentum(asset).clone()

  def __call__(self, env: ManagerBasedRlEnv) -> torch.Tensor:
    asset = _get_robot(env, self._asset_cfg)
    cur = _linear_momentum(asset)
    net_force = (cur - self._prev_lin_mom) / env.step_dt
    self._prev_lin_mom = cur.clone()
    return -torch.sum(net_force**2, dim=-1)

  def reset(self, env_ids: torch.Tensor | slice | None) -> None:
    asset = _get_robot(self._env, self._asset_cfg)
    full = _linear_momentum(asset)
    if env_ids is None:
      self._prev_lin_mom = full.clone()
    else:
      self._prev_lin_mom[env_ids] = full[env_ids]


class AngularMomentumChangePenalty:
  """Penalize rate of change of whole-body angular momentum (τ = dL/dt).

  Adapted from IHMC IsaacLab for mjlab.  Uses orbital angular momentum only
  (no body spin term) — see ``_angular_momentum`` for details.
  Returns a *negative* squared penalty; pair with a positive weight.
  """

  def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv) -> None:
    self._asset_cfg: SceneEntityCfg = cfg.params.get("asset_cfg", _DEFAULT_ASSET_CFG)
    self._env = env
    asset = _get_robot(env, self._asset_cfg)
    self._prev_ang_mom: torch.Tensor = _angular_momentum(asset).clone()

  def __call__(self, env: ManagerBasedRlEnv) -> torch.Tensor:
    asset = _get_robot(env, self._asset_cfg)
    cur = _angular_momentum(asset)
    net_torque = (cur - self._prev_ang_mom) / env.step_dt
    self._prev_ang_mom = cur.clone()
    return -torch.sum(net_torque**2, dim=-1)

  def reset(self, env_ids: torch.Tensor | slice | None) -> None:
    asset = _get_robot(self._env, self._asset_cfg)
    full = _angular_momentum(asset)
    if env_ids is None:
      self._prev_ang_mom = full.clone()
    else:
      self._prev_ang_mom[env_ids] = full[env_ids]


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


def feet_stumble(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  command_name: str | None = None,
) -> torch.Tensor:
  sensor: ContactSensor = env.scene[sensor_name]
  force = sensor.data.force
  assert force is not None
  horizontal = torch.norm(force[..., :2], dim=-1)
  vertical = torch.abs(force[..., 2])
  result = torch.any(horizontal > 4.0 * vertical, dim=1).float()
  if command_name is not None:
    command = get_motion_command(env, command_name)
    ground_mask = command.motion_lib.get_ground_motion_mask(command.motion_ids)
    result = result * (~ground_mask).float()
  return result


def feet_slip(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
  contact_force_threshold: float = 5.0,
  command_name: str | None = None,
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
  result = torch.sum(slip * contact.float(), dim=1)
  if command_name is not None:
    command = get_motion_command(env, command_name)
    ground_mask = command.motion_lib.get_ground_motion_mask(command.motion_ids)
    result = result * (~ground_mask).float()
  return result


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
		# Reward landing after sufficient air time (positive), penalize landing too early
		# (negative). The old formula (clamp to max=0) could only return 0 or negative,
		# which gave the same 0 reward for not lifting as for a perfect lift — so the
		# policy learned to never lift (avoiding the penalty entirely) rather than step well.
		air_time = torch.clamp(last_air_time, max=feet_air_time_target) - feet_air_time_target * 0.5
		reward = torch.sum(air_time * first_contact, dim=1)
		command = get_motion_command(env, command_name)
		active = torch.norm(command.body_lin_vel_w[:, 0, :2], dim=1) > 0.05
		return reward * active.float()

	def reset(self, env_ids: torch.Tensor | slice | None) -> None:
		del env_ids

