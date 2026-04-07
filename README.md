# TWIST2 MJLab — Usage Guide

<div align="center">
  <img src="imgs/example.gif" alt="TWIST2 motion tracking example" width="180" />
</div>

## Overview

`twist2_mjlab` is a standalone MJLab task package for Unitree G1 motion tracking based on [TWIST2](https://github.com/amazon-far/TWIST2) in order to enable further development on a supported physics engine (mjwarp) and training framework. The registered task is `Twist2-Flat-Unitree-G1`, and all task-specific logic lives locally under `src/twist2_mjlab/`.

The package loads motion references through a PKL motion library, so the normal workflow is:

1. enrich raw TWIST2 PKLs with MuJoCo forward kinematics (mainly for compatibility with the original motion commands of MJLab’s motion tracking task),
2. point the task at the resulting motion file,
3. train with `train_twist2.sh`, and
4. visualize with `play_twist2.sh`.

## TODOs
- [ ] Add hardware deployment instructions and scripts that use the MJLab G1 definitions.

## What’s in the package

```
twist2_mjlab/
├── pyproject.toml              # MJLab task package + MJLab entry point
├── train_twist2.sh             # Train `Twist2-Flat-Unitree-G1`
├── play_twist2.sh              # Play the latest or a chosen checkpoint
└── src/twist2_mjlab/
    ├── __init__.py             # Task registration
    ├── commands.py             # PKL motion command and resampling
    ├── config.py               # Observations, rewards, terminations, DR
    ├── observations.py         # Actor / critic observation terms
    ├── pkl_motion_lib.py       # Enriched PKL loader + interpolation
    ├── rewards.py              # Tracking and regularization rewards
    ├── terminations.py         # Failure / timeout conditions
    ├── rl_cfg.py               # Runner and model config
    └── scripts/enrich_pkl.py   # Add world-frame body data to PKLs
```

## Quick start

### 1) Install the package

Run everything from `twist2_mjlab/`:

```bash
cd /home/yangl/twist2/twist2_mjlab
uv sync
```

### 2) Prepare motion data

`PklMotionLib` expects PKLs that follows the format of BeyondMimic. If you start from raw [TWIST2 motions](https://drive.google.com/file/d/1JbW_InVD0ji5fvsR5kz7nbsXSXZQQXpd/view), run the enrichment script first:

```bash
uv run python -m twist2_mjlab.scripts.enrich_pkl \
  --dataset /path/to/twist2_dataset.yaml \
  --output-dir /path/to/enriched/ \
  --workers 8
```

This reads a dataset YAML, runs MuJoCo forward kinematics for each PKL, writes enriched PKLs with `body_pos_w` and `body_quat_w`, and saves a new `dataset.yaml` inside the output directory.

### 3) Train

Use a motion file via `TWIST2_MOTION_FILE`. It can be either:

- a single enriched `.pkl`, or
- a dataset `.yaml` with multiple motions.

```bash
TWIST2_MOTION_FILE=/path/to/enriched/dataset.yaml bash train_twist2.sh 0
```

Notes:

- the first positional argument is the GPU id (`0` by default),
- extra CLI flags are forwarded to MJLab’s `train` command,
- training logs are written under `logs/rsl_rl/g1_twist2_flat/`, and
- the W&B project is `twist2_mjlab`.

Useful training environment variables:

| Variable | Default | Purpose |
|---|---:|---|
| `TWIST2_MOTION_FILE` | required | Motion file passed to the task |
| `TWIST2_NUM_ENVS` | `4096` | Number of parallel environments |
| `TWIST2_VIDEO_INTERVAL` | `48000` | Video capture interval |
| `TWIST2_VIDEO_LENGTH` | `500` | Video length in steps |

#### Note: W&B setup and opt-out

This package logs training runs to Weights & Biases by default.

Before your first run, authenticate with W&B:

```bash
wandb login
```

You can also set the API key explicitly with `WANDB_API_KEY` if you prefer not to use the interactive login prompt.

Default W&B values for this task:

- project: `twist2_mjlab`
- experiment name: `g1_twist2_flat`
- run name: `g1_twist2_flat`

If you do not want W&B logging, switch the logger to TensorBoard when you launch training:

```bash
TWIST2_MOTION_FILE=/path/to/enriched/dataset.yaml bash train_twist2.sh 0 --agent.logger tensorboard
```

If you only want to disable W&B at the environment level, you can also set `WANDB_MODE=disabled`.

### 4) Play / visualize

```bash
TWIST2_MOTION_FILE=/path/to/enriched/dataset.yaml bash play_twist2.sh
```

If you do not pass a checkpoint path, `play_twist2.sh` automatically selects the latest `model_*.pt` from the most recent run directory under `logs/rsl_rl/g1_twist2_flat/`.

You can also pass a checkpoint explicitly:

```bash
TWIST2_MOTION_FILE=/path/to/enriched/dataset.yaml bash play_twist2.sh /path/to/model_12345.pt
```

Notes:

- the play script defaults to `--device cpu` and `--viewer native`,
- extra CLI flags are forwarded to MJLab’s `play` command, and
- the script prompts for `TWIST2_MOTION_FILE` if it is not set and the terminal is interactive.

## Motion file format

### Raw PKL input

The enrichment script expects each PKL to contain at least:

- `fps`
- `root_pos`
- `root_rot` in `[x, y, z, w]` order
- `dof_pos`
- `link_body_list`

### Enriched PKL output

After enrichment, the PKL also contains:

- `body_pos_w`
- `body_quat_w`

Those fields are required by the local motion library when the task samples motion frames, computes tracking observations, and builds privileged critic features.

### Dataset YAML example

```yaml
root_path: /path/to/enriched/pkls
motions:
  - file: walk_forward.pkl
    weight: 1.0
  - file: wave_hands.pkl
    weight: 0.5
```

The motion file path passed to `TWIST2_MOTION_FILE` can point to this YAML, or directly to a single enriched PKL.

## Where to tweak behavior

If you want to modify the task, these files are the main ones to look at:

- `src/twist2_mjlab/config.py` — observation groups, rewards, terminations, defaults
- `src/twist2_mjlab/observations.py` — observation building blocks
- `src/twist2_mjlab/rewards.py` — tracking and regularization terms
- `src/twist2_mjlab/terminations.py` — episode failure conditions
- `src/twist2_mjlab/commands.py` — motion command loading and resampling
- `src/twist2_mjlab/pkl_motion_lib.py` — motion loading, interpolation, and sampling
- `src/twist2_mjlab/rl_cfg.py` — runner and model configuration

## Troubleshooting

- **`TWIST2_MOTION_FILE is required`**: set the environment variable before running the script in a non-interactive shell.
- **`No runs found` in play**: train at least once, or pass an explicit checkpoint path.
- **Missing `body_pos_w` / `body_quat_w`**: rerun `enrich_pkl.py` on the raw PKLs.
- **Unexpected log location**: run the scripts from `twist2_mjlab/` so the relative `logs/` path matches the project layout.

## One-line summary

`twist2_mjlab` is the self-contained MJLab package for TWIST2 motion tracking on Unitree G1: enrich the PKLs, train with `Twist2-Flat-Unitree-G1`, then play the latest checkpoint when the robot inevitably refuses to be boring.
