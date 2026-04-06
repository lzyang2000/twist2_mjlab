"""TWIST2 task package for MJLab."""

from mjlab.tasks.registry import register_mjlab_task

from twist2_mjlab.config import unitree_g1_twist2_flat_env_cfg
from twist2_mjlab.rl.runner import Twist2OnPolicyRunner
from twist2_mjlab.rl_cfg import unitree_g1_twist2_flat_runner_cfg

register_mjlab_task(
  task_id="Twist2-Flat-Unitree-G1",
  env_cfg=unitree_g1_twist2_flat_env_cfg(),
  play_env_cfg=unitree_g1_twist2_flat_env_cfg(play=True),
  rl_cfg=unitree_g1_twist2_flat_runner_cfg(),
  runner_cls=Twist2OnPolicyRunner,
)
