"""TWIST2 runner shim.

The base MJLab runner already provides the PPO training and export flow.
This thin subclass only preserves the optional `registry_name` argument that
MJLab's training script passes for tracking tasks.
"""

from __future__ import annotations

from mjlab.rl import MjlabOnPolicyRunner


class Twist2OnPolicyRunner(MjlabOnPolicyRunner):
  def __init__(
    self,
    env,
    train_cfg: dict,
    log_dir: str | None = None,
    device: str = "cpu",
    registry_name: str | None = None,
  ) -> None:
    super().__init__(env, train_cfg, log_dir, device)
    self.registry_name = registry_name
