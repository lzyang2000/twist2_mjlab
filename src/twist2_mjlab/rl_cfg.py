"""RL configuration for the TWIST2 task.

TWIST2 uses the same structured observation method as the WBC reference,
but keeps the implementation local to this package.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass

from mjlab.rl import RslRlModelCfg
from mjlab.rl import RslRlOnPolicyRunnerCfg
from mjlab.tasks.tracking.config.g1.rl_cfg import unitree_g1_tracking_ppo_runner_cfg

from twist2_mjlab import observations as twist2_obs


@dataclass
class Twist2FutureModelCfg(RslRlModelCfg):
  class_name: str = "twist2_mjlab.rl.models:ActorCriticFuture"
  current_mimic_dim: int = 0
  current_proprio_dim: int = 0
  history_feature_dim: int = 0
  history_length: int = 0
  privileged_future_step_dim: int = 0
  privileged_future_steps: int = 0
  critic_current_dim: int = 0
  critic_extras_dim: int = 0
  num_motion_observations: int = 0
  num_motion_steps: int = 1
  num_priop_observations: int = 0
  num_history_steps: int = 0
  num_future_observations: int = 0
  num_future_steps: int = 0
  motion_latent_dim: int = 64
  history_latent_dim: int = 64
  future_latent_dim: int = 64
  future_encoder_dims: tuple[int, ...] = (256, 128)
  future_dropout: float = 0.1
  layer_norm: bool = False


def unitree_g1_twist2_flat_runner_cfg() -> RslRlOnPolicyRunnerCfg:
  base = unitree_g1_tracking_ppo_runner_cfg()
  cfg = deepcopy(base)
  cfg.experiment_name = "g1_twist2_flat"
  cfg.run_name = "g1_twist2_flat"
  cfg.wandb_project = "twist2_mjlab"
  cfg.save_interval = 1000
  cfg.obs_groups = {
    "actor": ("actor_current", "actor_history"),
    "critic": (
      "critic_priv_future_sequence",
      "critic_current",
      "critic_extras",
    ),
  }
  cfg.actor = Twist2FutureModelCfg(
    hidden_dims=base.actor.hidden_dims,
    activation=base.actor.activation,
    obs_normalization=base.actor.obs_normalization,
    distribution_cfg=base.actor.distribution_cfg,
    current_mimic_dim=twist2_obs.ACTOR_MIMIC_DIM,
    current_proprio_dim=twist2_obs.ACTOR_PROPRIO_DIM,
    history_feature_dim=twist2_obs.ACTOR_HISTORY_FEATURE_DIM,
    history_length=twist2_obs.ACTOR_HISTORY_LENGTH,
    history_latent_dim=64,
    motion_latent_dim=64,
    num_motion_observations=twist2_obs.ACTOR_MIMIC_DIM,
    num_motion_steps=1,
    num_priop_observations=twist2_obs.ACTOR_PROPRIO_DIM,
    num_history_steps=twist2_obs.ACTOR_HISTORY_LENGTH,
    num_future_observations=0,
    num_future_steps=0,
  )
  cfg.critic = Twist2FutureModelCfg(
    hidden_dims=base.critic.hidden_dims,
    activation=base.critic.activation,
    obs_normalization=base.critic.obs_normalization,
    distribution_cfg=base.critic.distribution_cfg,
    privileged_future_step_dim=twist2_obs.CRITIC_PRIV_STEP_DIM,
    privileged_future_steps=len(twist2_obs.PRIV_FUTURE_STEP_OFFSETS),
    critic_current_dim=twist2_obs.ACTOR_PROPRIO_DIM,
    critic_extras_dim=twist2_obs.critic_extras_dim(),
    motion_latent_dim=64,
    history_latent_dim=64,
    num_motion_observations=
      twist2_obs.CRITIC_PRIV_STEP_DIM * len(twist2_obs.PRIV_FUTURE_STEP_OFFSETS),
    num_motion_steps=len(twist2_obs.PRIV_FUTURE_STEP_OFFSETS),
    num_priop_observations=twist2_obs.ACTOR_PROPRIO_DIM,
    num_history_steps=0,
    num_future_observations=0,
    num_future_steps=0,
  )
  return cfg


__all__ = ["Twist2FutureModelCfg", "unitree_g1_twist2_flat_runner_cfg"]
