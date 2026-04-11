"""Export a TWIST2 checkpoint to ONNX for sim2sim deployment.

Usage:
    python deploy/export_onnx.py <checkpoint_path>

The script loads the Twist2-Flat-Unitree-G1 task, builds an environment
(with a dummy single-motion file), loads the checkpoint, and exports
the actor network to ONNX.
"""

import os
import sys
from dataclasses import asdict
from pathlib import Path


TASK_ID = "Twist2-Flat-Unitree-G1"

# A single PKL file used to satisfy the env init (any enriched PKL works).
_DEFAULT_MOTION_FILE = os.environ.get(
    "TWIST2_MOTION_FILE",
    "/home/yangl/twist2/wbc_twist2_data/OMOMO_g1_GMR/sub1_clothesstand_000.pkl",
)


def main():
    if len(sys.argv) < 2:
        print(f"Usage: python {sys.argv[0]} <checkpoint_path>")
        sys.exit(1)

    checkpoint_path = sys.argv[1]
    if not os.path.exists(checkpoint_path):
        print(f"Error: Checkpoint not found at {checkpoint_path}")
        sys.exit(1)

    print(f"Loading task: {TASK_ID}")
    from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls

    env_cfg = load_env_cfg(TASK_ID, play=True)
    rl_cfg = load_rl_cfg(TASK_ID)
    runner_cls = load_runner_cls(TASK_ID)

    if hasattr(rl_cfg, "__dataclass_fields__"):
        rl_dict = asdict(rl_cfg)
    else:
        rl_dict = rl_cfg

    print(f"Loading checkpoint: {checkpoint_path}")
    device = "cpu"

    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.rl import RslRlVecEnvWrapper

    print("Creating environment to resolve observation space...")
    env_cfg.scene.num_envs = 1
    if hasattr(env_cfg, "viewer"):
        env_cfg.viewer.viewer = "auto"

    if "motion" in env_cfg.commands:
        motion_cmd = env_cfg.commands["motion"]
        if hasattr(motion_cmd, "motion_file"):
            motion_cmd.motion_file = _DEFAULT_MOTION_FILE

    env = ManagerBasedRlEnv(cfg=env_cfg, device=device)
    env = RslRlVecEnvWrapper(env)

    print(f"Instantiating runner: {runner_cls.__name__}")
    runner = runner_cls(env, rl_dict, device=device)

    print("Loading weights into runner...")
    runner.load(checkpoint_path)

    export_dir = Path(checkpoint_path).parent
    filename = f"{export_dir.name}.onnx"

    print(f"Exporting to: {export_dir / filename}")
    runner.export_policy_to_onnx(str(export_dir), filename=filename)
    print("Done!")


if __name__ == "__main__":
    main()
