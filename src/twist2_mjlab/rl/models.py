"""TWIST2-style observation models adapted to rsl_rl 5.x."""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass

import torch
import torch.nn as nn
from tensordict import TensorDict

from rsl_rl.modules import EmpiricalNormalization, HiddenState
from rsl_rl.modules.distribution import Distribution
from rsl_rl.utils import resolve_callable, resolve_nn_activation, unpad_trajectories

from twist2_mjlab.rl.encoders import FutureMotionEncoder, TemporalConvEncoder


@dataclass(slots=True)
class _ObsGroupSpec:
  name: str
  shape: tuple[int, ...]

  @property
  def flat_dim(self) -> int:
    return math.prod(self.shape)

  @property
  def rank(self) -> int:
    return len(self.shape)


@dataclass(slots=True)
class _FlatObsLayout:
  motion_dim: int
  proprio_dim: int
  history_dim: int
  future_dim: int

  @property
  def total_dim(self) -> int:
    return self.motion_dim + self.proprio_dim + self.history_dim + self.future_dim


def _build_mlp(
  input_dim: int,
  output_dim: int,
  hidden_dims: tuple[int, ...] | list[int],
  activation: str,
  layer_norm: bool,
) -> nn.Sequential:
  layers: list[nn.Module] = []
  in_dim = input_dim
  for index, hidden_dim in enumerate(hidden_dims):
    layers.append(nn.Linear(in_dim, hidden_dim))
    if layer_norm and index == len(hidden_dims) - 1:
      layers.append(nn.LayerNorm(hidden_dim))
    layers.append(resolve_nn_activation(activation))
    in_dim = hidden_dim
  layers.append(nn.Linear(in_dim, output_dim))
  return nn.Sequential(*layers)


class _ObsModelBase(nn.Module):
  """Base model that supports flat or explicit sequence-shaped observation groups."""

  is_recurrent: bool = False

  def __init__(
    self,
    obs: TensorDict,
    obs_groups: dict[str, list[str]],
    obs_set: str,
    output_dim: int,
    hidden_dims: tuple[int, ...] | list[int] = (256, 256, 256),
    activation: str = "elu",
    obs_normalization: bool = False,
    distribution_cfg: dict | None = None,
    num_motion_observations: int = 0,
    num_motion_steps: int = 1,
    num_priop_observations: int = 0,
    num_history_steps: int = 0,
    num_future_observations: int = 0,
    num_future_steps: int = 0,
    motion_latent_dim: int = 64,
    history_latent_dim: int = 64,
    future_latent_dim: int = 64,
    future_encoder_dims: tuple[int, ...] | list[int] = (256, 128),
    future_dropout: float = 0.1,
    layer_norm: bool = False,
    tanh_encoder_output: bool = False,
    current_mimic_dim: int = 0,
    current_proprio_dim: int = 0,
    history_feature_dim: int = 0,
    history_length: int = 0,
    privileged_future_step_dim: int = 0,
    privileged_future_steps: int = 0,
    critic_current_dim: int = 0,
    critic_extras_dim: int = 0,
    **_: object,
  ) -> None:
    super().__init__()
    self.obs_set = obs_set
    self.obs_groups, self.obs_specs, self.obs_dim = self._get_obs_specs(
      obs, obs_groups, obs_set
    )
    self.output_dim = output_dim
    self.activation = activation
    self.tanh_encoder_output = tanh_encoder_output

    if obs_normalization:
      self.obs_normalizer: nn.Module = EmpiricalNormalization(self.obs_dim)
    else:
      self.obs_normalizer = nn.Identity()

    if distribution_cfg is not None:
      dist_cfg = copy.deepcopy(distribution_cfg)
      dist_class: type[Distribution] = resolve_callable(dist_cfg.pop("class_name"))  # type: ignore[assignment]
      self.distribution: Distribution | None = dist_class(output_dim, **dist_cfg)
      mlp_output_dim = self.distribution.input_dim
    else:
      self.distribution = None
      mlp_output_dim = output_dim

    self.layout = _FlatObsLayout(
      motion_dim=num_motion_observations,
      proprio_dim=num_priop_observations,
      history_dim=num_priop_observations * num_history_steps,
      future_dim=num_future_observations,
    )
    self.num_motion_steps = max(num_motion_steps, 1)
    self.num_history_steps = max(num_history_steps, 0)
    self.num_future_steps = max(num_future_steps, 0)

    self.mode = self._resolve_mode()

    self.motion_encoder: TemporalConvEncoder | None = None
    self.history_encoder: TemporalConvEncoder | None = None
    self.future_encoder: FutureMotionEncoder | None = None

    if self.mode == "actor_structured":
      self.actor_current_group = self.obs_specs[0].name
      self.actor_history_group = self.obs_specs[1].name
      self.actor_current_dim = self.obs_specs[0].flat_dim
      self.actor_history_length = self.obs_specs[1].shape[0]
      self.actor_history_feature_dim = math.prod(self.obs_specs[1].shape[1:])
      self._validate_expected_dim(
        "actor current",
        self.actor_current_dim,
        current_mimic_dim + current_proprio_dim,
      )
      self._validate_expected_dim(
        "actor history feature",
        self.actor_history_feature_dim,
        history_feature_dim,
      )
      self._validate_expected_dim(
        "actor history length",
        self.actor_history_length,
        history_length,
      )
      self.history_encoder = TemporalConvEncoder(
        input_size=self.actor_history_feature_dim,
        tsteps=self.actor_history_length,
        output_size=history_latent_dim,
        activation=activation,
      )
      feature_dim = self.actor_current_dim + history_latent_dim
    elif self.mode == "critic_structured":
      self.critic_priv_group = self.obs_specs[0].name
      self.critic_current_group = self.obs_specs[1].name
      self.critic_extras_group = self.obs_specs[2].name
      self.critic_priv_steps = self.obs_specs[0].shape[0]
      self.critic_priv_step_dim = math.prod(self.obs_specs[0].shape[1:])
      self.critic_current_dim = self.obs_specs[1].flat_dim
      self.critic_extras_dim = self.obs_specs[2].flat_dim
      self._validate_expected_dim(
        "critic privileged future step",
        self.critic_priv_step_dim,
        privileged_future_step_dim,
      )
      self._validate_expected_dim(
        "critic privileged future length",
        self.critic_priv_steps,
        privileged_future_steps,
      )
      self._validate_expected_dim(
        "critic current",
        self.critic_current_dim,
        critic_current_dim,
      )
      self._validate_expected_dim(
        "critic extras",
        self.critic_extras_dim,
        critic_extras_dim,
      )
      self.motion_encoder = TemporalConvEncoder(
        input_size=self.critic_priv_step_dim,
        tsteps=self.critic_priv_steps,
        output_size=motion_latent_dim,
        activation=activation,
      )
      feature_dim = motion_latent_dim + self.critic_current_dim + self.critic_extras_dim
    else:
      if self.layout.total_dim > self.obs_dim:
        raise ValueError(
          "Configured flat observation layout exceeds the concatenated TensorDict "
          f"size: layout={self.layout.total_dim}, obs_dim={self.obs_dim}"
        )
      if self.layout.motion_dim > 0:
        if self.layout.motion_dim % self.num_motion_steps != 0:
          raise ValueError(
            "num_motion_observations must divide evenly by num_motion_steps."
          )
        self.motion_encoder = TemporalConvEncoder(
          input_size=self.layout.motion_dim // self.num_motion_steps,
          tsteps=self.num_motion_steps,
          output_size=motion_latent_dim,
          activation=activation,
        )
      if self.layout.history_dim > 0 and self.num_history_steps > 0:
        self.history_encoder = TemporalConvEncoder(
          input_size=self.layout.proprio_dim,
          tsteps=self.num_history_steps,
          output_size=history_latent_dim,
          activation=activation,
        )
      if self.layout.future_dim > 0:
        self.future_encoder = FutureMotionEncoder(
          input_size=self.layout.future_dim,
          output_size=future_latent_dim,
          activation=activation,
          hidden_dims=future_encoder_dims,
          dropout=future_dropout,
        )
      feature_dim = self._get_flat_feature_dim(
        motion_latent_dim=motion_latent_dim,
        history_latent_dim=history_latent_dim,
        future_latent_dim=future_latent_dim,
      )

    self.mlp = _build_mlp(
      feature_dim,
      int(mlp_output_dim),
      hidden_dims,
      activation,
      layer_norm,
    )
    if self.distribution is not None:
      self.distribution.init_mlp_weights(self.mlp)

  def forward(
    self,
    obs: TensorDict | torch.Tensor,
    masks: torch.Tensor | None = None,
    hidden_state: HiddenState = None,
    stochastic_output: bool = False,
  ) -> torch.Tensor:
    if isinstance(obs, TensorDict) and masks is not None:
      obs = unpad_trajectories(obs, masks)
    latent = self.get_latent(obs, masks, hidden_state)
    output = self.mlp(latent)
    if self.distribution is None:
      return output
    self.distribution.update(output)
    if stochastic_output:
      return self.distribution.sample()
    return self.distribution.deterministic_output(output)

  def get_latent(
    self,
    obs: TensorDict | torch.Tensor,
    masks: torch.Tensor | None = None,
    hidden_state: HiddenState = None,
  ) -> torch.Tensor:
    del masks, hidden_state
    flat = self._flatten_obs(obs)
    flat = self.obs_normalizer(flat)
    if self.mode == "flat":
      return self._encode_flat(flat)
    obs_td = self._unflatten_obs(flat)
    return self._encode_structured(obs_td)

  def reset(
    self,
    dones: torch.Tensor | None = None,
    hidden_state: HiddenState = None,
  ) -> None:
    del dones, hidden_state

  def get_hidden_state(self) -> HiddenState:
    return None

  def detach_hidden_state(self, dones: torch.Tensor | None = None) -> None:
    del dones

  @property
  def output_mean(self) -> torch.Tensor:
    if self.distribution is None:
      raise AttributeError("Deterministic model has no output_mean.")
    return self.distribution.mean

  @property
  def output_std(self) -> torch.Tensor:
    if self.distribution is None:
      raise AttributeError("Deterministic model has no output_std.")
    return self.distribution.std

  @property
  def output_entropy(self) -> torch.Tensor:
    if self.distribution is None:
      raise AttributeError("Deterministic model has no output_entropy.")
    return self.distribution.entropy

  @property
  def output_distribution_params(self) -> tuple[torch.Tensor, ...]:
    if self.distribution is None:
      raise AttributeError("Deterministic model has no distribution parameters.")
    return self.distribution.params

  def get_output_log_prob(self, outputs: torch.Tensor) -> torch.Tensor:
    if self.distribution is None:
      raise AttributeError("Deterministic model has no log probabilities.")
    return self.distribution.log_prob(outputs)

  def get_kl_divergence(
    self,
    old_params: tuple[torch.Tensor, ...],
    new_params: tuple[torch.Tensor, ...],
  ) -> torch.Tensor:
    if self.distribution is None:
      raise AttributeError("Deterministic model has no KL divergence.")
    return self.distribution.kl_divergence(old_params, new_params)

  def as_jit(self) -> nn.Module:
    return _TorchObsModel(self)

  def as_onnx(self, verbose: bool) -> nn.Module:
    return _OnnxObsModel(self, verbose)

  def update_normalization(self, obs: TensorDict | torch.Tensor) -> None:
    if isinstance(self.obs_normalizer, EmpiricalNormalization):
      self.obs_normalizer.update(self._flatten_obs(obs))  # type: ignore[arg-type]

  def _resolve_mode(self) -> str:
    if (
      self.obs_set == "actor"
      and len(self.obs_specs) == 2
      and self.obs_specs[0].rank == 1
      and self.obs_specs[1].rank == 2
    ):
      return "actor_structured"
    if (
      self.obs_set == "critic"
      and len(self.obs_specs) == 3
      and self.obs_specs[0].rank == 2
      and self.obs_specs[1].rank == 1
      and self.obs_specs[2].rank == 1
    ):
      return "critic_structured"
    return "flat"

  def _get_obs_specs(
    self,
    obs: TensorDict,
    obs_groups: dict[str, list[str]],
    obs_set: str,
  ) -> tuple[list[str], list[_ObsGroupSpec], int]:
    active_obs_groups = obs_groups[obs_set]
    specs: list[_ObsGroupSpec] = []
    total_dim = 0
    for obs_group in active_obs_groups:
      shape = tuple(obs[obs_group].shape[1:])
      spec = _ObsGroupSpec(name=obs_group, shape=shape)
      specs.append(spec)
      total_dim += spec.flat_dim
    return active_obs_groups, specs, total_dim

  def _flatten_obs(self, obs: TensorDict | torch.Tensor) -> torch.Tensor:
    if isinstance(obs, torch.Tensor):
      return obs
    return torch.cat(
      [obs[group].reshape(obs[group].shape[0], -1) for group in self.obs_groups],
      dim=-1,
    )

  def _unflatten_obs(self, flat: torch.Tensor) -> TensorDict:
    batch = flat.shape[0]
    items: dict[str, torch.Tensor] = {}
    cursor = 0
    for spec in self.obs_specs:
      next_cursor = cursor + spec.flat_dim
      items[spec.name] = flat[:, cursor:next_cursor].reshape(batch, *spec.shape)
      cursor = next_cursor
    return TensorDict(items, batch_size=[batch], device=flat.device)

  def _encode_structured(self, obs: TensorDict) -> torch.Tensor:
    if self.mode == "actor_structured":
      current = obs[self.actor_current_group]
      history = obs[self.actor_history_group]
      if current.dim() != 2 or history.dim() != 3:
        raise ValueError(
          "Actor structured observations must be [B, D] + [B, H, D], got "
          f"{tuple(current.shape)} and {tuple(history.shape)}"
        )
      history_latent = self.history_encoder(history)
      latent = torch.cat((history_latent, current), dim=-1)
    elif self.mode == "critic_structured":
      priv_future = obs[self.critic_priv_group]
      current = obs[self.critic_current_group]
      extras = obs[self.critic_extras_group]
      if priv_future.dim() != 3 or current.dim() != 2 or extras.dim() != 2:
        raise ValueError(
          "Critic structured observations must be [B, T, D] + [B, D] + [B, D], got "
          f"{tuple(priv_future.shape)}, {tuple(current.shape)}, {tuple(extras.shape)}"
        )
      priv_latent = self.motion_encoder(priv_future)
      latent = torch.cat((priv_latent, current, extras), dim=-1)
    else:
      raise RuntimeError(f"Structured encoder called in unsupported mode: {self.mode}")

    if self.tanh_encoder_output:
      latent = torch.tanh(latent)
    return latent

  def _encode_flat(self, flat: torch.Tensor) -> torch.Tensor:
    motion_end = self.layout.motion_dim
    proprio_end = motion_end + self.layout.proprio_dim
    history_end = proprio_end + self.layout.history_dim
    future_end = history_end + self.layout.future_dim

    motion = flat[:, :motion_end]
    proprio = flat[:, motion_end:proprio_end]
    history = flat[:, proprio_end:history_end]
    future = flat[:, history_end:future_end]
    remainder = flat[:, future_end:]

    features: list[torch.Tensor] = []
    if motion.numel() > 0 and self.motion_encoder is not None:
      features.append(self.motion_encoder(motion))
      single_motion = motion[:, : self.layout.motion_dim // self.num_motion_steps]
      features.append(single_motion)
    if history.numel() > 0 and self.history_encoder is not None:
      features.append(self.history_encoder(history))
    if future.numel() > 0 and self.future_encoder is not None:
      features.append(self.future_encoder(future))
    if proprio.numel() > 0:
      features.append(proprio)
    if remainder.numel() > 0:
      features.append(remainder)
    latent = torch.cat(features, dim=-1) if features else flat
    if self.tanh_encoder_output:
      latent = torch.tanh(latent)
    return latent

  def _get_flat_feature_dim(
    self,
    motion_latent_dim: int,
    history_latent_dim: int,
    future_latent_dim: int,
  ) -> int:
    dim = self.obs_dim - self.layout.total_dim
    if self.layout.motion_dim > 0:
      dim += motion_latent_dim + self.layout.motion_dim // self.num_motion_steps
    if self.layout.history_dim > 0:
      dim += history_latent_dim
    if self.layout.future_dim > 0:
      dim += future_latent_dim
    dim += self.layout.proprio_dim
    return dim

  @staticmethod
  def _validate_expected_dim(
    name: str,
    actual: int,
    expected: int,
  ) -> None:
    if expected > 0 and actual != expected:
      raise ValueError(f"Unexpected {name} dim: expected {expected}, got {actual}")


class _TorchObsModel(nn.Module):
  """TorchScript-friendly wrapper for custom models with flat external inputs."""

  def __init__(self, model: _ObsModelBase) -> None:
    super().__init__()
    cached_distribution = None
    if model.distribution is not None:
      cached_distribution = model.distribution._distribution
      model.distribution._distribution = None
    try:
      self.model = copy.deepcopy(model)
    finally:
      if model.distribution is not None:
        model.distribution._distribution = cached_distribution
    self.input_size = self.model.obs_dim

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    return self.model(x, stochastic_output=False)

  @torch.jit.export
  def reset(self) -> None:
    pass


class _OnnxObsModel(_TorchObsModel):
  """ONNX-friendly wrapper for custom models with flat external inputs."""

  is_recurrent: bool = False

  def __init__(self, model: _ObsModelBase, verbose: bool) -> None:
    del verbose
    super().__init__(model)

  def get_dummy_inputs(self) -> tuple[torch.Tensor]:
    return (torch.zeros(1, self.input_size),)

  @property
  def input_names(self) -> list[str]:
    return ["obs"]

  @property
  def output_names(self) -> list[str]:
    return ["actions"]


class ActorCriticFuture(_ObsModelBase):
  """TWIST2 future-motion model adapted to the rsl_rl 5.x model API."""


__all__ = ["ActorCriticFuture"]
