# TWIST2 MJLab — 使用指南

<div align="center">
  <img src="hello.gif" alt="TWIST2 问候 gif" width="360" />
  <img src="example.gif" alt="TWIST2 动作跟踪示例" width="360" />
</div>

## 概览

`twist2_mjlab` 是一个独立的 MJLab 任务包，用于基于 [TWIST2](https://github.com/amazon-far/TWIST2) 的 Unitree G1 动作跟踪，目的是在受支持的物理引擎（mjwarp）和训练框架上继续开发。注册的任务名称是 `Twist2-Flat-Unitree-G1`，所有任务相关逻辑都保存在本地的 `src/twist2_mjlab/` 下。

该包通过 PKL 动作库加载动作参考，因此标准流程如下：

1. 使用 MuJoCo 前向运动学补全原始 TWIST2 PKL 文件（主要是为了兼容 MJLab 动作跟踪任务的原始 motion commands），
2. 将任务指向生成后的动作文件，
3. 使用 `train_twist2.sh` 训练，
4. 使用 `play_twist2.sh` 可视化。

## TODO
- [ ] 添加硬件部署说明，以及使用 MJLab G1 定义（增益、动作缩放等）的脚本。

## 包含内容

```
twist2_mjlab/
├── pyproject.toml              # MJLab 任务包 + MJLab 入口点
├── train_twist2.sh             # 训练 `Twist2-Flat-Unitree-G1`
├── play_twist2.sh              # 播放最新或指定检查点
├── play_twist2_pretrained.sh   # 直接播放内置的预训练检查点
├── resources/
│   ├── pretrained.pt           # 预训练检查点（30K iterations）
│   ├── hello.gif               # README 演示资源
│   ├── example.gif             # README 演示资源
│   └── readme_zh.md            # 中文使用说明
└── src/twist2_mjlab/
    ├── __init__.py             # 任务注册
    ├── commands.py             # PKL 动作命令与重采样
    ├── config.py               # 观测、奖励、终止条件、域随机化
    ├── observations.py         # Actor / critic 观测项
    ├── pkl_motion_lib.py       # 补全后的 PKL 读取与插值
    ├── rewards.py              # 跟踪与正则化奖励
    ├── terminations.py         # 失败 / 超时条件
    ├── rl_cfg.py               # 运行器与模型配置
    └── scripts/enrich_pkl.py   # 给 PKL 添加世界系身体数据
```

## 快速开始

### 1) 安装包

请在 `twist2_mjlab/` 目录下运行所有命令：

```bash
cd /path/to/twist2_mjlab
uv sync
```

### 2) 准备动作数据

`PklMotionLib` 期望 PKL 文件符合 BeyondMimic 的格式。如果你是从原始 [TWIST2 motions](https://drive.google.com/file/d/1JbW_InVD0ji5fvsR5kz7nbsXSXZQQXpd/view) 开始，请先运行补全脚本：

```bash
uv run python -m twist2_mjlab.scripts.enrich_pkl \
  --dataset /path/to/twist2_dataset.yaml \
  --output-dir /path/to/enriched/ \
  --workers 8
```

该脚本会读取 dataset YAML，为每个 PKL 运行 MuJoCo 前向运动学，写出包含 `body_pos_w` 和 `body_quat_w` 的补全版 PKL，并在输出目录中保存新的 `dataset.yaml`。

**注意:** 如果你想先直接体验一下，这个包已经自带了一个训练到 30K iterations 的预训练 checkpoint，直接运行 `play_twist2_pretrained.sh` 即可。

### 3) 训练

通过 `TWIST2_MOTION_FILE` 指定动作文件。它可以是：

- 单个补全后的 `.pkl`，或
- 包含多个动作的 dataset `.yaml`。

```bash
TWIST2_MOTION_FILE=/path/to/enriched/dataset.yaml bash train_twist2.sh 0
```

说明：

- 第一个位置参数是 GPU 编号（默认是 `0`），
- 额外的 CLI 参数会继续传递给 MJLab 的 `train` 命令，
- 训练日志会写入 `logs/rsl_rl/g1_twist2_flat/`。

#### 关于 W&B 以及保存内容

这个包默认会把训练记录到 Weights & Biases。

该任务的 W&B 默认值是：

- project：`twist2_mjlab`
- experiment name：`g1_twist2_flat`
- run name：`g1_twist2_flat`

首次运行前，请先完成 W&B 登录：

```bash
wandb login
```

如果你不想使用交互式登录，也可以直接设置 `WANDB_API_KEY`。

默认情况下，W&B 会保存：

- 训练标量，例如 episode 统计、loss、学习率、action 标准差，以及 FPS / 性能指标
- 训练与环境配置（`agent.yaml` 和 `env.yaml`）
- 本次运行所用本地仓库的 git 状态，包括 commit hash、status 和 diff
- 在运行目录下找到的日志视频（`*.mp4`）
- 当启用 `upload_model` 时，模型 checkpoint 和导出的 policy 文件；该选项默认开启

如果你不想使用 W&B，可以在启动训练时将 logger 切换为 TensorBoard：

```bash
TWIST2_MOTION_FILE=/path/to/enriched/dataset.yaml bash train_twist2.sh 0 --agent.logger tensorboard
```

如果你只想在环境层面禁用 W&B，也可以设置 `WANDB_MODE=disabled`。

### 4) 播放 / 可视化

```bash
TWIST2_MOTION_FILE=/path/to/enriched/dataset.yaml bash play_twist2.sh
```

如果你不传入 checkpoint 路径，`play_twist2.sh` 会自动从 `logs/rsl_rl/g1_twist2_flat/` 下最新的 run 目录里选择最新的 `model_*.pt`。

你也可以显式指定 checkpoint：

```bash
TWIST2_MOTION_FILE=/path/to/enriched/dataset.yaml bash play_twist2.sh /path/to/model_12345.pt
```

你也可以直接运行预训练模型脚本：

```bash
TWIST2_MOTION_FILE=/path/to/enriched/dataset.yaml bash play_twist2_pretrained.sh
```

直接运行 `play_twist2_pretrained.sh` 时，会使用训练到 30K 步的预训练模型。

说明：

- play 脚本默认使用 `--device cpu` 和 `--viewer native`，
- 额外的 CLI 参数会继续传递给 MJLab 的 `play` 命令，
- 如果没有设置 `TWIST2_MOTION_FILE` 且终端是交互式的，脚本会提示输入。

## 动作文件格式

### 原始 PKL 输入

补全脚本至少需要每个 PKL 包含以下字段：

- `fps`
- `root_pos`
- `root_rot`，顺序为 `[x, y, z, w]`
- `dof_pos`
- `link_body_list`

### 补全后的 PKL 输出

完成补全后，PKL 还会新增：

- `body_pos_w`
- `body_quat_w`

当任务采样动作帧、计算跟踪观测以及构建特权 critic 特征时，本地 motion library 需要这些字段。

### Dataset YAML 示例

```yaml
root_path: /path/to/enriched/pkls
motions:
  - file: walk_forward.pkl
    weight: 1.0
  - file: wave_hands.pkl
    weight: 0.5
```

传给 `TWIST2_MOTION_FILE` 的路径既可以指向这个 YAML，也可以直接指向单个补全后的 PKL。

## 在哪里调整行为

如果你想修改任务，重点查看这些文件：

- `src/twist2_mjlab/config.py` — 观测组、奖励、终止条件、默认值
- `src/twist2_mjlab/observations.py` — 观测构建模块
- `src/twist2_mjlab/rewards.py` — 跟踪与正则化项
- `src/twist2_mjlab/terminations.py` — episode 失败条件
- `src/twist2_mjlab/commands.py` — 动作加载与重采样
- `src/twist2_mjlab/pkl_motion_lib.py` — 动作加载、插值与采样
- `src/twist2_mjlab/rl_cfg.py` — 运行器与模型配置

## 排查问题

- **`TWIST2_MOTION_FILE is required`**：在非交互式 shell 中运行脚本前，请先设置该环境变量。
- **play 时提示 `No runs found`**：至少训练一次，或者显式传入 checkpoint 路径。
- **缺少 `body_pos_w` / `body_quat_w`**：请对原始 PKL 重新运行 `enrich_pkl.py`。
- **日志路径不符合预期**：请在 `twist2_mjlab/` 下运行脚本，这样相对路径 `logs/` 才会和项目结构一致。
- **关于显示、渲染或视频的报错**：训练脚本里设置 `--video False`。

## 一句话总结

`twist2_mjlab` 是 TWIST2 在 Unitree G1 上进行动作跟踪的自包含 MJLab 包：先补全 PKL，再用 `Twist2-Flat-Unitree-G1` 训练，最后播放最新 checkpoint——毕竟机器人最擅长的事情之一，就是认真拒绝无聊。
