# TWIST2 MJLab — Usage Guide

<div align="center">
    <img src="resources/hello.gif" alt="TWIST2 hello gif" width="360" />
  <img src="resources/example.gif" alt="TWIST2 motion tracking example" width="360" />
</div>

## Overview

`twist2_mjlab` is a standalone MJLab task package for Unitree G1 motion tracking based on [TWIST2](https://github.com/amazon-far/TWIST2) in order to enable further development on a supported physics engine (mjwarp) and training framework. The registered task is `Twist2-Flat-Unitree-G1`, and all task-specific logic lives locally under `src/twist2_mjlab/`.

The package loads motion references through a PKL motion library, so the normal workflow is:

1. enrich raw TWIST2 PKLs with MuJoCo forward kinematics (mainly for compatibility with the original motion commands of MJLab’s motion tracking task),
2. point the task at the resulting motion file,
3. train with `train_twist2.sh`, and
4. visualize with `play_twist2.sh`.

## TODOs
- [x] Decoupled sim2sim pipeline (sim node + policy node over UDP at 50 Hz, real-time MuJoCo viewer with ghost overlay).
- [ ] Add hardware deployment instructions and scripts that use the MJLab G1 definitions (gains, action scale, etc.).

## What’s in the package

```
twist2_mjlab/
├── pyproject.toml              # MJLab task package + MJLab entry point
├── train_twist2.sh             # Train `Twist2-Flat-Unitree-G1`
├── play_twist2.sh              # Play the latest or a chosen checkpoint
├── play_twist2_pretrained.sh   # Play with the checked-in pretrained checkpoint
├── sim2sim_pretrained.sh       # One-line sim2sim with pretrained ONNX
├── resources/
│   ├── pretrained.pt           # Pretrained checkpoint (30K iterations)
│   ├── pretrained.onnx         # Pretrained ONNX model (for sim2sim)
│   ├── hello.gif               # README demo asset
│   ├── example.gif             # README demo asset
│   └── readme_zh.md            # Chinese usage guide
├── deploy/                     # Sim2sim deployment pipeline
│   ├── play_sim_twist2.sh      # Orchestration script
│   ├── export_onnx.py          # Checkpoint -> ONNX export
│   ├── common/udp_sync.py      # UDP state/action protocol
│   ├── sim/sim_node.py         # MuJoCo physics + ghost overlay viewer
│   └── policy/twist2_policy.py # ONNX inference + motion library
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
cd /path/to/twist2_mjlab
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

**Note:** If you want to try playing first, this package already comes with a pretrained checkpoint at 30K iterations; just run `play_twist2_pretrained.sh` directly.

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
- training logs are written under `logs/rsl_rl/g1_twist2_flat/`.

#### Note: W&B setup and what gets saved

This package logs training runs to Weights & Biases by default.

The W&B defaults for this task are:

- project: `twist2_mjlab`
- experiment name: `g1_twist2_flat`
- run name: `g1_twist2_flat`

Before your first run, authenticate with W&B:

```bash
wandb login
```

You can also set the API key explicitly with `WANDB_API_KEY` if you prefer not to use the interactive login prompt.

By default, W&B stores:

- training scalars such as episode statistics, losses, learning rate, action standard deviation, and FPS/performance
- the training and environment configs (`agent.yaml` and `env.yaml`)
- git state for the local repos used in the run, including commit hash, status, and diff
- logged videos (`*.mp4`) found under the run directory
- model checkpoints and exported policy files when `upload_model` is enabled, which is the default

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

Or run the pretrained checkpoint script directly:

```bash
TWIST2_MOTION_FILE=/path/to/enriched/dataset.yaml bash play_twist2_pretrained.sh
```

Running `play_twist2_pretrained.sh` directly uses the pretrained checkpoint at 30K iterations.

Notes:

- the play script defaults to `--device cpu` and `--viewer native`,
- extra CLI flags are forwarded to MJLab’s `play` command, and
- the script prompts for `TWIST2_MOTION_FILE` if it is not set and the terminal is interactive.

### 5) Sim2sim deployment

The sim2sim pipeline runs the trained policy in a hardware-like decoupled two-process architecture: a **sim node** (MuJoCo physics + viewer) and a **policy node** (ONNX inference + motion library), communicating asynchronously over UDP. Both processes run their own real-time clocks independently — if the policy is slow, the sim keeps running with the last command, just like real actuators. A semi-transparent green ghost shows the reference motion the policy is tracking.

**Quickest way — pretrained model:**

```bash
bash sim2sim_pretrained.sh
```

This uses the bundled ONNX model and a sample motion clip; no training or export step needed.

**With your own checkpoint:**

Pass a `.pt` checkpoint (auto-exports to ONNX) or a `.onnx` file directly:

```bash
# From a .pt checkpoint (exports ONNX automatically)
TWIST2_MOTION_FILE=/path/to/enriched/motion.pkl \
  ./deploy/play_sim_twist2.sh /path/to/model_29999.pt

# From a pre-exported .onnx
TWIST2_MOTION_FILE=/path/to/enriched/motion.pkl \
  ./deploy/play_sim_twist2.sh /path/to/model.onnx

# No model arg: auto-selects the latest checkpoint from logs/
TWIST2_MOTION_FILE=/path/to/enriched/motion.pkl \
  ./deploy/play_sim_twist2.sh
```

**How it works:**

The pipeline decouples physics simulation from neural network inference into two fully independent real-time processes, mirroring how a real robot works — actuators hold the last command while the next one is being computed:

```
sim_node (MuJoCo, 1000 Hz)        policy_node (ONNX, 50 Hz)
  Own real-time clock                Own real-time clock
  Step physics (20 × 0.001s)        Load motion library (PKL)
  Pack robot state ──UDP──>          Drain to latest state
                                     Build observations:
                                       mimic (35D) from motion ref
                                       proprio (92D) from robot state
                                       history (11 × 127D)
  Drain to latest action <──UDP──   Run ONNX inference → 29D action
  Hold last action if none arrived   Send action + reference pose
  Render viewer + green ghost
```

- Both processes run their own independent real-time clocks. Neither ever blocks on the other — UDP is fire-and-forget. If either side is momentarily slow, the other keeps running with the latest available data.
- The **sim node** (`deploy/sim/sim_node.py`) runs MuJoCo G1 physics at 1000 Hz (timestep 0.001s, 20x decimation → 50 Hz control rate). Each control cycle it sends state and drains any new action from the policy. If no new action has arrived, `data.ctrl` holds the previous command — exactly like real actuators.
- The **policy node** (`deploy/policy/twist2_policy.py`) runs at 50 Hz. It loads the motion library to construct the 35D mimic observation (reference joint positions + root state), maintains an 11-frame observation history, and runs the exported ONNX actor network.
- Each motion plays with a **3-second blend-in** from the default standing pose and a **3-second blend-out** back to standing, then loops.
- The green ghost orientation is corrected to always start facing the +X direction.

**Environment variables:**

| Variable | Description |
|----------|-------------|
| `TWIST2_MOTION_FILE` | Path to an enriched `.pkl` or a dataset `.yaml` (required) |
| `TWIST2_MOTION_INDEX` | Index of the motion to play from a multi-motion dataset (default `0`) |
| `TWIST2_INIT_YAW_DEG` | Initial robot yaw in degrees (default `0`) |

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
- **Complaining about display, rendering, video**: set `--video False` in the training script.
  
## One-line summary

`twist2_mjlab` is the self-contained MJLab package for TWIST2 motion tracking on Unitree G1: enrich the PKLs, train with `Twist2-Flat-Unitree-G1`, then play the latest checkpoint when the robot inevitably refuses to be boring.

## 中文版

[中文说明](resources/readme_zh.md)
