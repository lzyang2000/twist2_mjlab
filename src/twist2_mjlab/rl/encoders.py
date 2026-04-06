"""Temporal encoders for TWIST2-style observation layouts."""

from __future__ import annotations

import torch
from torch import nn

from rsl_rl.utils import resolve_nn_activation


def _build_temporal_conv(
  activation: nn.Module,
  tsteps: int,
  channel_size: int,
) -> nn.Module:
  if tsteps == 1:
    return nn.Flatten()
  if tsteps in (10, 11):
    return nn.Sequential(
      nn.Conv1d(3 * channel_size, 2 * channel_size, kernel_size=4, stride=2),
      activation,
      nn.Conv1d(2 * channel_size, channel_size, kernel_size=2, stride=1),
      activation,
      nn.Flatten(),
    )
  if tsteps == 20:
    return nn.Sequential(
      nn.Conv1d(3 * channel_size, 2 * channel_size, kernel_size=6, stride=2),
      activation,
      nn.Conv1d(2 * channel_size, channel_size, kernel_size=4, stride=2),
      activation,
      nn.Flatten(),
    )
  if tsteps == 50:
    return nn.Sequential(
      nn.Conv1d(3 * channel_size, 2 * channel_size, kernel_size=8, stride=4),
      activation,
      nn.Conv1d(2 * channel_size, channel_size, kernel_size=5, stride=1),
      activation,
      nn.Conv1d(channel_size, channel_size, kernel_size=5, stride=1),
      activation,
      nn.Flatten(),
    )
  raise ValueError(f"Unsupported temporal encoder steps: {tsteps}")


class TemporalConvEncoder(nn.Module):
  """1D-conv temporal encoder for flat or sequence-shaped temporal observations."""

  def __init__(
    self,
    input_size: int,
    tsteps: int,
    output_size: int,
    activation: str,
  ) -> None:
    super().__init__()
    self.tsteps = tsteps
    self.input_size = input_size
    act = resolve_nn_activation(activation)
    channel_size = 20

    self.proj = nn.Sequential(nn.Linear(input_size, 3 * channel_size), act)
    self.temporal = _build_temporal_conv(act, tsteps, channel_size)
    self.out = nn.Linear(channel_size * 3, output_size)

  def forward(self, obs: torch.Tensor) -> torch.Tensor:
    if obs.dim() == 2:
      batch = obs.shape[0]
      obs = obs.reshape(batch, self.tsteps, self.input_size)
    elif obs.dim() == 3:
      batch = obs.shape[0]
      if obs.shape[1] != self.tsteps or obs.shape[2] != self.input_size:
        raise ValueError(
          "TemporalConvEncoder received a sequence with unexpected shape: "
          f"expected [B, {self.tsteps}, {self.input_size}], got {tuple(obs.shape)}"
        )
    else:
      raise ValueError(
        "TemporalConvEncoder expects a [B, T*D] or [B, T, D] tensor, "
        f"got {tuple(obs.shape)}"
      )

    projected = self.proj(obs)
    latent = self.temporal(projected.transpose(1, 2))
    return self.out(latent)


class FutureMotionEncoder(nn.Module):
  """Simple MLP encoder over flattened future motion observations."""

  def __init__(
    self,
    input_size: int,
    output_size: int,
    activation: str,
    hidden_dims: tuple[int, ...] | list[int] = (256, 128),
    dropout: float = 0.1,
  ) -> None:
    super().__init__()
    act_name = activation
    layers: list[nn.Module] = []
    in_dim = input_size
    for hidden_dim in hidden_dims:
      layers.append(nn.Linear(in_dim, hidden_dim))
      layers.append(resolve_nn_activation(act_name))
      if dropout > 0:
        layers.append(nn.Dropout(dropout))
      in_dim = hidden_dim
    layers.append(nn.Linear(in_dim, output_size))
    self.encoder = nn.Sequential(*layers)

    for layer in self.encoder:
      if isinstance(layer, nn.Linear):
        nn.init.xavier_uniform_(layer.weight, gain=0.5)
        nn.init.zeros_(layer.bias)

  def forward(self, obs: torch.Tensor) -> torch.Tensor:
    return self.encoder(obs)
