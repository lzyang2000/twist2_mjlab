# TWIST2 MJLab — Usage Guide

<div align="center">
    <img src="resources/hello.gif" alt="TWIST2 hello gif" width="360" />
  <img src="resources/real.gif" alt="TWIST2 hello in real world gif" width="360" />
</div>

## Overview

`twist2_mjlab` is a standalone MJLab task package for Unitree G1 motion tracking based on [TWIST2](https://github.com/amazon-far/TWIST2) in order to enable further development on a supported physics engine (mjwarp) and training framework. The registered task is `Twist2-Flat-Unitree-G1`, and all task-specific logic lives locally under `src/twist2_mjlab/`.

The package loads motion references through a PKL motion library, and now supports two independent workflows. You can use either one or both, depending on which dataset you want to work with:

1. TWIST2: enrich raw [TWIST2 PKLs](https://drive.google.com/file/d/1JbW_InVD0ji5fvsR5kz7nbsXSXZQQXpd/view) with MuJoCo forward kinematics,
2. SEED: convert the [SEED dataset](https://huggingface.co/datasets/bones-studio/seed) G1 CSV motions into enriched TWIST2 PKLs,
3. point the task at the resulting dataset YAML,
4. train with `train_twist2.sh` or `train_seed.sh`, and
5. visualize with `play_twist2.sh` or `play_seed.sh`.

## TODOs
- [x] Decoupled sim2sim pipeline (sim node + policy node over UDP at 50 Hz, real-time MuJoCo viewer with ghost overlay).
- [x] Hardware deployment on Unitree G1 (shared policy node, hardware state node using the vendored Unitree SDK2 and the MJLab G1 gains/action scale).

## What’s in the package

```
twist2_mjlab/
├── pyproject.toml              # MJLab task package + MJLab entry point
├── train_twist2.sh             # Train `Twist2-Flat-Unitree-G1`
├── train_seed.sh               # Train on the enriched SEED G1 dataset
├── play_twist2.sh              # Play the latest or a chosen checkpoint
├── play_twist2_pretrained.sh   # Play with the checked-in pretrained checkpoint
├── play_seed.sh                # Play the latest or a chosen SEED checkpoint
├── play_seed_pretrained.sh     # Play with the checked-in SEED pretrained checkpoint
├── sim2sim_pretrained.sh       # One-line sim2sim with pretrained ONNX
├── sim2sim_seed.sh             # One-line sim2sim for SEED runs
├── sim2sim_seed_pretrained.sh  # One-line sim2sim with the SEED pretrained ONNX
├── resources/
│   ├── pretrained.pt           # Pretrained checkpoint (30K iterations)
│   ├── pretrained.onnx         # Pretrained ONNX model (for sim2sim)
│   ├── pretrained_seed.pt      # Pretrained SEED checkpoint (30K iterations)
│   ├── pretrained_seed.onnx    # Pretrained SEED ONNX model (for sim2sim)
│   ├── hello.gif               # README demo asset
│   ├── example.gif             # README demo asset
│   └── readme_zh.md            # Chinese usage guide
├── deploy/                     # Sim2sim + real-hardware deployment
│   ├── play_sim_twist2.sh      # Sim2sim orchestration (MuJoCo + policy)
│   ├── play_real_twist2.sh     # Real-hardware orchestration (G1 + policy)
│   ├── install_unitree_sdk.sh  # One-time Unitree SDK2 build + install
│   ├── export_onnx.py          # Checkpoint -> ONNX export
│   ├── common/udp_sync.py      # UDP state/action protocol
│   ├── sim/sim_node.py         # MuJoCo physics + ghost overlay viewer
│   ├── policy/twist2_policy.py # ONNX inference + motion library
│   └── real/                   # Real-hardware deployment
│       ├── hardware_node.py        # 50 Hz G1 loop via unitree_interface
│       ├── g1_robot_constants.py   # Frozen PD gains / default pose
│       └── unitree_sdk2_wrapper/   # Git submodule (SDK2 C++ + pybind11)
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

#### SEED dataset support

The repo also includes a dedicated SEED pipeline for the Hugging Face dataset [`bones-studio/seed`](https://huggingface.co/datasets/bones-studio/seed). Access requires accepting the dataset terms on Hugging Face first. This is optional; you can use the TWIST2 pipeline alone, the SEED pipeline alone, or both.

The SEED enricher expects the G1 CSV motions plus the metadata CSV. The default paths in `seed_enrich.py` are:

- `~/twist2/seed/g1/csv`
- `~/twist2/seed/seed_metadata_v003.csv`

If you keep the Hugging Face folder layout, either pass `--metadata` explicitly or symlink/copy `metadata/seed_metadata_v003.csv` to that default location.

To convert the SEED G1 CSVs into enriched PKLs and dataset YAMLs:

```bash
uv run python -m twist2_mjlab.scripts.seed_enrich \
  --csv-dir ~/twist2/seed/g1/csv \
  --metadata ~/twist2/seed/metadata/seed_metadata_v003.csv \
  --output-dir ~/twist2/seed_g1_enriched_pkl \
  --fps 30 \
  --workers 8
```

This writes enriched PKLs under `~/twist2/seed_g1_enriched_pkl/`, plus:

- `seed_dataset.yaml` — all motions
- `seed_dataset_filtered.yaml` — quality-filtered motions only

If you prefer the script defaults and already mirrored the metadata CSV into `~/twist2/seed/seed_metadata_v003.csv`, you can omit the `--metadata` flag.

**Note:** If you want to try playing first, this package already comes with a pretrained checkpoint at 30K iterations; just run `play_twist2_pretrained.sh` directly.

### 3) Train

For the original TWIST2 motions, use `TWIST2_MOTION_FILE` to point at either a single enriched `.pkl` or a dataset `.yaml` with multiple motions:

```bash
TWIST2_MOTION_FILE=/path/to/enriched/dataset.yaml bash train_twist2.sh 0
```

For the SEED workflow, `train_seed.sh` is wired to the default output from `seed_enrich.py`:

```bash
bash train_seed.sh 0
```

Notes:

- the first positional argument is the GPU id (`0` by default),
- extra CLI flags are forwarded to MJLab’s `train` command,
- `train_twist2.sh` writes logs under `logs/rsl_rl/g1_twist2_flat/`,
- `train_seed.sh` writes logs under `logs/rsl_rl/g1_twist2_seed_flat/`.

If you change the SEED output directory, update the `MOTION_FILE` path inside `train_seed.sh` or create a symlink to `~/twist2/seed_g1_enriched_pkl/seed_dataset.yaml`.

If you want to work with both datasets, just run the two pipelines separately; they use different motion files and log directories, so they won’t step on each other’s toes.

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

For the SEED workflow, use the equivalent launcher and point it at the same enriched motion file or a single motion PKL:

```bash
TWIST2_MOTION_FILE=/path/to/enriched/seed_dataset.yaml bash play_seed.sh
```

To try the bundled SEED checkpoint instead:

```bash
TWIST2_MOTION_FILE=/path/to/enriched/seed_motion.pkl bash play_seed_pretrained.sh
```

`play_seed.sh` automatically selects the latest checkpoint from `logs/rsl_rl/g1_twist2_seed_flat/` when you do not pass one explicitly.

You can freely use just the TWIST2 play scripts, just the SEED play scripts, or both, depending on which checkpoints you have trained.

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

For SEED, the matching launcher is:

```bash
TWIST2_MOTION_FILE=/path/to/enriched/seed_motion.pkl bash sim2sim_seed_pretrained.sh
```

`sim2sim_seed.sh` works the same way as the TWIST2 version, but it looks in `logs/rsl_rl/g1_twist2_seed_flat/` for checkpoints and expects `TWIST2_MOTION_FILE` to point to a single enriched `.pkl` motion.

As with training and playback, the TWIST2 and SEED sim2sim launchers are independent: use whichever one matches the model and motion file you want, or both if you’re comparing results.

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
  Pack robot state ──50Hz UDP──>     Drain to latest state (50 Hz)
                                     Build observations:
                                       mimic (35D) from motion ref
                                       proprio (92D) from robot state
                                       history (11 × 127D)
  Drain to latest action <──50Hz UDP──  Run ONNX inference → 29D action
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

### 6) Hardware deployment

The real-hardware path reuses the same policy node and UDP protocol as sim2sim, and swaps the MuJoCo simulation for a 50 Hz loop that reads IMU + joint state from a Unitree G1 via a vendored SDK2 wrapper and applies the policy's joint targets with PD gains matched to the MJLab G1 definitions.

**Prerequisites:**

- Ubuntu host wired to the G1 (default interface `eth0`),
- Python 3.10 (already pinned by `pyproject.toml`) so the prebuilt `.cpython-310` binding is valid,
- `sudo` access on first run to install `build-essential`, `cmake`, `python3-dev`, `pybind11-dev`.

**One-time install of the Unitree SDK2 binding:**

```bash
git submodule update --init deploy/real/unitree_sdk2_wrapper
./deploy/install_unitree_sdk.sh
```

This builds the C++ wrapper and drops `unitree_interface.so` into the twist2_mjlab uv environment. Verify with:

```bash
uv run python -c "import unitree_interface; print('ok')"
```

**Run on the robot:**

```bash
TWIST2_MOTION_FILE=/path/to/enriched/motion.pkl \
  TWIST2_REAL_NET=eth0 \
  ./deploy/play_real_twist2.sh /path/to/model_29999.pt
```

Same model-argument conventions as `play_sim_twist2.sh`: pass a `.pt` (auto-exports to ONNX), a `.onnx` directly, or no arg to auto-select the latest checkpoint from `logs/rsl_rl/g1_twist2_flat/`.

**Wireless-remote startup sequence** (exactly as `hardware_node.py` prints):

1. **START** — release damped hold and interpolate to the default standing pose over 2 s.
2. **A** — enter the 50 Hz policy loop.
3. **B** — graceful exit (damped hold at the current pose).
4. **SELECT** — emergency damp-stop; use this first if anything looks wrong.

**What's different from sim2sim:**

- No MuJoCo viewer, no ghost overlay. The hardware node still *receives* the reference-motion fields the policy sends back, but discards them since there is nothing to render.
- The robot has no odometry, so the state packet carries zeros for `root_pos` and `body_lin_vel`. The policy's proprioception only uses `body_ang_vel`, the IMU quaternion, and joint state, so this matches training.
- No ROS. The hardware node is pure UDP + DDS (the DDS side is handled inside the vendored SDK wrapper).

**Environment variables:**

| Variable | Description |
|----------|-------------|
| `TWIST2_MOTION_FILE` | Motion reference (same as sim2sim) |
| `TWIST2_MOTION_INDEX` | Motion index inside a dataset YAML (default `0`) |
| `TWIST2_REAL_NET` | DDS network interface for the G1 (default `eth0`) |

**Safety notes:**

- Always keep a hand on the wireless remote; **SELECT** is the fastest way out.
- Start with the robot suspended or with a second person holding it. The first press of **A** hands control to the policy immediately.
- On Ctrl-C the launcher's cleanup trap `pkill`s both nodes, and the hardware node damps at the current pose before exiting.

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

## Large motion datasets: CPU offload & GPU cache

By default, `PklMotionLib` concatenates all motion frames into six tensors on the GPU. With ~14 tracked bodies, each frame takes ~960 B, so the GPU memory roughly equals `total_frames × 960 B`. Datasets like the full SEED library (~142K motions at 120 fps) exceed even a single H100's VRAM. Three opt-in modes handle this.

### Mode 1 — Downsampling (simplest, highest speed)

Load only a subset of the full motion dataset by filtering trajectories or frame decimation at load time.

```bash
# Example: load every Nth motion or subsample frames when loading the PKL
# This is configured at the dataset/PKL preparation stage, not at runtime.
```

- **Pro:** all data fits in GPU memory; no pipeline stalls; full training speed.
- **Con:** loses motion diversity compared to the full dataset.
- **When to use:** you can afford to lose some trajectories or frame resolution, and speed is critical.
- **Practical note (SEED):** by default, SEED enrichment downsamples from 120 fps to 30 fps, which reduces frame count by ~4× and allows the full dataset to fit on a single H100 without any offload or cache overhead.

### Mode 2 — CPU offload (simple, no cache)

Keeps the six motion tensors in pinned CPU RAM and transfers the blended frames back to the GPU each step.

```bash
--env.commands.motion.offload-to-cpu True
```

- **Trade-off:** each env step calls `.cpu()` on the frame indices, which forces a CUDA sync and drains the GPU pipeline. Expect roughly **2× slower** training loops in exchange for fitting arbitrarily large datasets in host RAM.
- **When to use:** dataset does not fit in VRAM and you do not want the complexity of a cache; also useful as a baseline to compare against.

### Mode 3 — GPU cache (recommended for large datasets)

Keeps all data on pinned CPU RAM and maintains a GPU-resident cache that holds the active working set. With ~4096 envs each playing one motion at a time, the working set is only ~4 GB (well under the default 8 GB cache). Cache hits take the same GPU gather path as the no-offload mode; misses load the full motion from CPU in frame-contiguous chunks.

```bash
--env.commands.motion.gpu-cache True \
--env.commands.motion.cache-capacity-gb 80.0
```

- **Performance:** on a 142K-motion dataset with a well-sized cache, ~98 % of `get_frame` calls are pure GPU gathers, so throughput is very close to the all-on-GPU baseline.
- **Cache sizing rule of thumb:** `num_envs × avg_frames_per_motion × 960 B × 2`. For the full SEED dataset with 4096 envs we run at 80 GB on a 96 GB H100.
- **Replacement policy:** append-only cache with a full reset on overflow (not per-motion LRU eviction).
- **When cache fills up:** if the next motion would exceed remaining capacity, the cache map is cleared in O(1) and active motions are reloaded on subsequent accesses.
- **Practical note (SEED dataset):** larger cache is usually better; too small a cache can hurt throughput because frequent full-cache resets force repeated reloading from CPU.

### Which mode to pick

| Situation | Recommendation |
|-----------|----------------|
| Dataset fits comfortably in VRAM | Leave both flags off (default) |
| Dataset is very large, can subsample motions/frames | Downsample at dataset preparation time (Mode 1) |
| Dataset is too large, prefer simplicity, accept slowdown | `--env.commands.motion.offload-to-cpu True` (Mode 2) |
| Dataset is too large, want near-baseline speed | `--env.commands.motion.gpu-cache True` (Mode 3) |

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
