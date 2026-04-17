# TWIST2 MJLab — 使用指南

<div align="center">
  <img src="hello.gif" alt="TWIST2 问候 gif" width="360" />
  <img src="real.gif" alt="TWIST2 真实世界动作 gif" width="360" />
</div>

## 概览

`twist2_mjlab` 是一个独立的 MJLab 任务包，用于基于 [TWIST2](https://github.com/amazon-far/TWIST2) 的 Unitree G1 动作跟踪，目的是在受支持的物理引擎（mjwarp）和训练框架上继续开发。注册的任务名称是 `Twist2-Flat-Unitree-G1`，所有任务相关逻辑都保存在本地的 `src/twist2_mjlab/` 下。

该包通过 PKL 动作库加载动作参考，现在支持两条彼此独立的流程。你可以只用其中一条，也可以两条都用，取决于你要使用哪个数据集：

1. TWIST2：使用 MuJoCo 前向运动学补全原始 TWIST2 PKL 文件，
2. SEED：把 SEED 数据集里的 G1 CSV 动作转换成补全后的 TWIST2 PKL，
3. 将任务指向生成后的动作文件或 dataset YAML，
4. 使用 `train_twist2.sh` 或 `train_seed.sh` 训练，
5. 使用 `play_twist2.sh` 或 `play_seed.sh` 可视化。

## TODO
- [x] 解耦式 sim2sim 流水线（sim 节点 + policy 节点通过 UDP 50 Hz 通信，实时 MuJoCo 查看器带参考动作绿色影子叠加）。
- [x] 在 Unitree G1 上的硬件部署（与 sim2sim 共用 policy 节点，硬件状态节点通过内置的 Unitree SDK2 读写机器人，PD 增益与动作缩放沿用 MJLab G1 定义）。

## 包含内容

```
twist2_mjlab/
├── pyproject.toml              # MJLab 任务包 + MJLab 入口点
├── train_twist2.sh             # 训练 `Twist2-Flat-Unitree-G1`
├── train_seed.sh               # 训练补全后的 SEED G1 数据集
├── play_twist2.sh              # 播放最新或指定检查点
├── play_twist2_pretrained.sh   # 直接播放内置的预训练检查点
├── play_seed.sh                # 播放最新或指定的 SEED 检查点
├── play_seed_pretrained.sh     # 直接播放内置的 SEED 预训练检查点
├── sim2sim_pretrained.sh       # 一键启动预训练模型 sim2sim
├── sim2sim_seed.sh             # SEED 版本的一键 sim2sim
├── sim2sim_seed_pretrained.sh  # 使用 SEED 预训练 ONNX 的一键 sim2sim
├── resources/
│   ├── pretrained.pt           # 预训练检查点（30K iterations）
│   ├── pretrained.onnx         # 预训练 ONNX 模型（用于 sim2sim）
│   ├── pretrained_seed.pt      # SEED 预训练检查点（30K iterations）
│   ├── pretrained_seed.onnx    # SEED 预训练 ONNX 模型（用于 sim2sim）
│   ├── hello.gif               # README 演示资源
│   ├── example.gif             # README 演示资源
│   └── readme_zh.md            # 中文使用说明
├── deploy/                     # Sim2sim + 真机部署流水线
│   ├── play_sim_twist2.sh      # Sim2sim 启动脚本（MuJoCo + policy）
│   ├── play_real_twist2.sh     # 真机启动脚本（G1 + policy）
│   ├── install_unitree_sdk.sh  # 一次性安装 Unitree SDK2 绑定
│   ├── export_onnx.py          # 检查点 -> ONNX 导出
│   ├── common/udp_sync.py      # UDP 状态/动作协议
│   ├── sim/sim_node.py         # MuJoCo 物理仿真 + 参考动作半透明绿色叠加
│   ├── policy/twist2_policy.py # ONNX 推理 + 动作库
│   └── real/                   # 真机部署
│       ├── hardware_node.py        # 通过 unitree_interface 运行 50 Hz G1 控制循环
│       ├── g1_robot_constants.py   # 冻结的 PD 增益 / 默认姿态
│       └── unitree_sdk2_wrapper/   # Git 子模块（SDK2 C++ 源码 + pybind11）
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

该脚本会读取 dataset YAML，为每个 PKL 运行 MuJoCo 前向运动学，写出包含 `body_pos_w` 和 `body_quat_w` 的补全版 PKL，并在输出目录中生成新的 `dataset.yaml`。

#### SEED 数据集支持

仓库里还提供了面向 Hugging Face 数据集 [`bones-studio/seed`](https://huggingface.co/datasets/bones-studio/seed) 的专用 SEED 流程。访问前需要先在 Hugging Face 页面接受数据集条款。这个流程是可选的；你可以只用 TWIST2 流程、只用 SEED 流程，或者两个都用。

SEED 补全脚本需要 G1 CSV 动作文件和 metadata CSV。`seed_enrich.py` 的默认路径是：

- `~/twist2/seed/g1/csv`
- `~/twist2/seed/seed_metadata_v003.csv`

如果你沿用 Hugging Face 的目录结构，可以显式传 `--metadata`，或者把 `metadata/seed_metadata_v003.csv` 复制 / 软链接到默认位置。

把 SEED G1 CSV 转成补全后的 PKL 和 dataset YAML：

```bash
uv run python -m twist2_mjlab.scripts.seed_enrich \
  --csv-dir ~/twist2/seed/g1/csv \
  --metadata ~/twist2/seed/metadata/seed_metadata_v003.csv \
  --output-dir ~/twist2/seed_g1_enriched_pkl \
  --fps 30 \
  --workers 8
```

这会把补全后的 PKL 写到 `~/twist2/seed_g1_enriched_pkl/`，并生成：

- `seed_dataset.yaml` —— 全部动作
- `seed_dataset_filtered.yaml` —— 过滤后的动作

如果你已经把 metadata CSV 放到了默认路径 `~/twist2/seed/seed_metadata_v003.csv`，那就可以省略 `--metadata`。

**注意:** 如果你想先直接体验一下，这个包已经自带了一个训练到 30K iterations 的预训练 checkpoint，直接运行 `play_twist2_pretrained.sh` 即可。

### 3) 训练

对于原始 TWIST2 动作，`TWIST2_MOTION_FILE` 可以指向单个补全后的 `.pkl`，也可以指向包含多个动作的 dataset `.yaml`：

```bash
TWIST2_MOTION_FILE=/path/to/enriched/dataset.yaml bash train_twist2.sh 0
```

对于 SEED 流程，`train_seed.sh` 默认使用 `seed_enrich.py` 的输出：

```bash
bash train_seed.sh 0
```

说明：

- 第一个位置参数是 GPU 编号（默认是 `0`），
- 额外的 CLI 参数会继续传递给 MJLab 的 `train` 命令，
- `train_twist2.sh` 的训练日志会写入 `logs/rsl_rl/g1_twist2_flat/`，
- `train_seed.sh` 的训练日志会写入 `logs/rsl_rl/g1_twist2_seed_flat/`。

如果你修改了 SEED 输出目录，请同步更新 `train_seed.sh` 里的 `MOTION_FILE`，或者创建一个指向 `~/twist2/seed_g1_enriched_pkl/seed_dataset.yaml` 的软链接。

如果你想同时使用 TWIST2 和 SEED，两条流程分别运行即可；它们使用不同的动作文件和日志目录，不会互相影响。

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

也可以直接运行预训练脚本：

```bash
TWIST2_MOTION_FILE=/path/to/enriched/dataset.yaml bash play_twist2_pretrained.sh
```

直接运行 `play_twist2_pretrained.sh` 时，会使用训练到 30K 步的预训练模型。

对于 SEED 流程，可以使用对应的脚本，并指向同一个补全后的动作文件或单个动作 PKL：

```bash
TWIST2_MOTION_FILE=/path/to/enriched/seed_dataset.yaml bash play_seed.sh
```

如果想直接试用内置的 SEED 预训练模型：

```bash
TWIST2_MOTION_FILE=/path/to/enriched/seed_motion.pkl bash play_seed_pretrained.sh
```

`play_seed.sh` 在不显式传入检查点时，会自动从 `logs/rsl_rl/g1_twist2_seed_flat/` 下选择最新的检查点。

说明：

- play 脚本默认使用 `--device cpu` 和 `--viewer native`，
- 额外的 CLI 参数会继续传递给 MJLab 的 `play` 命令，
- 如果没有设置 `TWIST2_MOTION_FILE` 且终端是交互式的，脚本会提示输入。

你可以只用 TWIST2 的播放脚本，只用 SEED 的播放脚本，或者两者都用，具体取决于你训练了哪些检查点。

### 5) Sim2sim 部署

sim2sim 流水线采用类硬件的解耦双进程架构运行训练好的策略：**sim 节点**（MuJoCo 物理仿真 + 查看器）和 **policy 节点**（ONNX 推理 + 动作库），通过 UDP 异步通信。两个进程各自维护独立的实时时钟——如果 policy 稍慢，sim 会继续使用上一条指令运行，就像真实执行器一样。查看器中会显示一个半透明的绿色"影子"机器人，表示策略正在跟踪的参考动作。

**最快体验——使用预训练模型：**

```bash
bash sim2sim_pretrained.sh
```

这会使用内置的 ONNX 模型和示例动作片段，无需训练或导出步骤。

对于 SEED，对应脚本是：

```bash
TWIST2_MOTION_FILE=/path/to/enriched/seed_motion.pkl bash sim2sim_seed_pretrained.sh
```

`sim2sim_seed.sh` 和 TWIST2 版本的用法一样，只是它会去 `logs/rsl_rl/g1_twist2_seed_flat/` 查找检查点，并且要求 `TWIST2_MOTION_FILE` 指向单个补全后的 `.pkl` 动作文件。

**使用自己训练的检查点：**

传入 `.pt` 检查点（自动导出为 ONNX）或直接传入 `.onnx` 文件：

```bash
# 从 .pt 检查点启动（自动导出 ONNX）
TWIST2_MOTION_FILE=/path/to/enriched/motion.pkl \
  ./deploy/play_sim_twist2.sh /path/to/model_29999.pt

# 从已导出的 .onnx 启动
TWIST2_MOTION_FILE=/path/to/enriched/motion.pkl \
  ./deploy/play_sim_twist2.sh /path/to/model.onnx

# 不传模型参数：自动从 logs/ 选择最新检查点
TWIST2_MOTION_FILE=/path/to/enriched/motion.pkl \
  ./deploy/play_sim_twist2.sh
```

和训练、播放一样，TWIST2 与 SEED 的 sim2sim 启动器也是彼此独立的：你可以按需使用任意一个，或者都用来对比结果。

**工作原理：**

该流水线将物理仿真与神经网络推理解耦为两个完全独立的实时进程，模拟真实机器人的工作方式——执行器在等待下一条指令时保持上一条指令：

```
sim_node (MuJoCo, 1000 Hz)        policy_node (ONNX, 50 Hz)
  独立实时时钟                        独立实时时钟
  步进物理 (20 × 0.001s)             加载动作库 (PKL)
  打包机器人状态 ──50Hz UDP──>        获取最新状态 (50 Hz)
                                     构建观测：
                                       mimic (35D) 来自动作参考
                                       proprio (92D) 来自机器人状态
                                       history (11 × 127D)
  获取最新动作 <──50Hz UDP──           运行 ONNX 推理 → 29D 动作
  无新动作则保持上一条指令             发送动作 + 参考姿态
  渲染查看器 + 绿色影子
```

- 两个进程各自运行独立的实时时钟，互不阻塞——UDP 采用发射后不管模式。如果任一方暂时偏慢，另一方继续使用最新可用数据运行。
- **sim 节点** (`deploy/sim/sim_node.py`) 以 1000 Hz 运行 MuJoCo G1 物理仿真（时间步长 0.001s，20 倍降采样 → 50 Hz 控制频率）。每个控制周期发送状态并获取 policy 的最新动作。如果没有新动作到达，`data.ctrl` 保持上一条指令——和真实执行器行为一致。
- **policy 节点** (`deploy/policy/twist2_policy.py`) 以 50 Hz 运行。加载动作库以构建 35D mimic 观测（参考关节位置 + 根状态），维护 11 帧的观测历史，并运行导出的 ONNX actor 网络。
- 每段动作播放时会有 **3 秒的渐入**（从默认站立姿态过渡）和 **3 秒的渐出**（过渡回站立姿态），然后循环。
- 绿色影子的朝向会被校正为始终面向 +X 方向。

**环境变量：**

| 变量 | 说明 |
|------|------|
| `TWIST2_MOTION_FILE` | 补全后的 `.pkl` 或 dataset `.yaml` 的路径（必填） |
| `TWIST2_MOTION_INDEX` | 多动作数据集中要播放的动作索引（默认 `0`） |
| `TWIST2_INIT_YAW_DEG` | 机器人初始偏航角，单位度（默认 `0`） |

### 6) 硬件部署

真机部署与 sim2sim 共用同一个 policy 节点和 UDP 协议，只是把 MuJoCo 仿真换成了一个 50 Hz 的硬件循环：通过内置的 Unitree SDK2 包装层读取 G1 的 IMU 与关节状态，再把 policy 输出的关节目标以 PD 控制下发到电机，PD 增益与 MJLab G1 定义保持一致。

**前置条件：**

- Ubuntu 主机，通过有线网络连接 G1（默认接口 `eth0`），
- Python 3.10（已由 `pyproject.toml` 固定），以便使用预编译的 `.cpython-310` 绑定，
- 首次运行需要 `sudo` 权限以安装 `build-essential`、`cmake`、`python3-dev`、`pybind11-dev`。

**一次性安装 Unitree SDK2 绑定：**

```bash
git submodule update --init deploy/real/unitree_sdk2_wrapper
./deploy/install_unitree_sdk.sh
```

脚本会编译 C++ 包装层，并把 `unitree_interface.so` 安装到 twist2_mjlab 的 uv 环境里。可以用下面的命令验证：

```bash
uv run python -c "import unitree_interface; print('ok')"
```

**在机器人上运行：**

```bash
TWIST2_MOTION_FILE=/path/to/enriched/motion.pkl \
  TWIST2_REAL_NET=eth0 \
  ./deploy/play_real_twist2.sh /path/to/model_29999.pt
```

模型参数约定与 `play_sim_twist2.sh` 完全相同：可以传入 `.pt`（自动导出为 ONNX）、直接传入 `.onnx`，或不传任何参数以自动选择 `logs/rsl_rl/g1_twist2_flat/` 下最新的 checkpoint。

**手柄启动流程**（与 `hardware_node.py` 控制台提示一致）：

1. **START** — 解除阻尼保持，在 2 秒内平滑过渡到默认站立姿态。
2. **A** — 进入 50 Hz policy 控制循环。
3. **B** — 优雅退出（在当前姿态上进入阻尼保持）。
4. **SELECT** — 紧急阻尼停止，发生任何异常情况时优先按这个键。

**与 sim2sim 的差异：**

- 没有 MuJoCo 查看器，也没有绿色影子叠加。硬件节点仍会**接收** policy 发回的参考姿态字段，但直接丢弃，因为没有渲染目标。
- 机器人没有里程计，所以状态包里 `root_pos` 和 `body_lin_vel` 填 0。Policy 的本体感知只使用 `body_ang_vel`、IMU 四元数和关节状态，这与训练时一致。
- 不依赖 ROS。硬件节点只用 UDP + DDS（DDS 部分由内置 SDK 包装层处理）。

**环境变量：**

| 变量 | 说明 |
|------|------|
| `TWIST2_MOTION_FILE` | 动作参考文件（与 sim2sim 相同） |
| `TWIST2_MOTION_INDEX` | dataset YAML 中的动作索引（默认 `0`） |
| `TWIST2_REAL_NET` | 连接 G1 的 DDS 网络接口（默认 `eth0`） |

**安全建议：**

- 始终保持手握手柄，**SELECT** 是最快的应急出口。
- 启动前请把机器人悬吊起来，或者由另一人扶住。按下 **A** 的瞬间控制权就交给 policy 了。
- Ctrl-C 时启动脚本的清理钩子会 `pkill` 掉两个节点，硬件节点退出前会先在当前姿态阻尼保持。

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

## 大规模动作数据集：CPU 卸载与 GPU 缓存

默认情况下，`PklMotionLib` 会把所有动作帧拼接成六个张量直接放到 GPU 上。追踪 14 个 body 时每一帧约 960 B，所以 GPU 显存占用大致等于 `total_frames × 960 B`。完整的 SEED 数据集（约 14.2 万条动作，120 fps）已经超出单张 H100 的显存容量。为此提供三种可选模式。

### 模式 1 — 下采样（最简单，最快）

在加载时通过筛选轨迹或对帧进行下采样来加载数据集的子集。

```bash
# 例：加载每 N 个动作或在加载 PKL 时对帧下采样
# 这在数据集/PKL 准备阶段配置，不在运行时配置。
```

- **优点：** 所有数据都装入 GPU 显存；没有流水线停顿；完整训练速度。
- **缺点：** 相比完整数据集损失动作多样性。
- **适用场景：** 可以接受减少一些轨迹或帧分辨率，且对速度有要求。
- **实践建议（SEED）：** 默认情况下，SEED 补全会从 120 fps 下采样到 30 fps，帧数减少约 4 倍，使得完整数据集可以装入单张 H100 而无需任何卸载或缓存开销。

### 模式 2 — 纯 CPU 卸载（简单，无缓存)

把六个动作张量放在 pinned CPU 内存里，每一步将混合后的帧回传到 GPU。

```bash
--env.commands.motion.offload-to-cpu True
```

- **代价：** 每个 env step 都要对帧索引调用 `.cpu()`，触发 CUDA 同步并清空 GPU 流水线。训练循环大约会变慢 **2 倍**，但可以容纳任意大小的数据集（只要 host RAM 够）。
- **适用场景：** 数据集装不下显存，同时不想引入缓存带来的额外复杂度；也适合作为与其他模式对比的基线。

### 模式 3 — GPU 缓存（大规模数据集推荐)

所有数据仍然放在 pinned CPU 内存，但额外维护一块 GPU 常驻缓存，用来存放“活跃工作集”。4096 个 env 各自只播放一条动作，工作集只有约 4 GB（远小于默认 8 GB 缓存）。缓存命中时走的是和纯 GPU 模式相同的 gather 路径；未命中时会按连续帧块从 CPU 加载该动作到缓存。

```bash
--env.commands.motion.gpu-cache True \
--env.commands.motion.cache-capacity-gb 70.0   # 可选，默认 70 GB
```

- **性能：** 在 14.2 万条动作的数据集上，只要缓存尺寸合理，约 98% 的 `get_frame` 调用都是纯 GPU gather，吞吐非常接近“全部放 GPU”时的基线。
- **缓存容量经验公式：** `num_envs × avg_frames_per_motion × 960 B × 2`。在 96 GB H100 上跑完整 SEED 数据集时我们使用 80 GB 缓存。
- **替换策略：** 追加写入，空间不足时整块重置（不是按动作的 LRU 淘汰）。
- **缓存填满时：** 如果下一个动作会超过剩余容量，缓存映射会被 O(1) 清空，活跃动作会在后续访问时重新加载。
- **实践建议（SEED 数据集）：** 缓存越大通常越好；缓存太小反而可能拖慢速度，因为整块缓存会更频繁重置并反复从 CPU 回填。

### 如何选择

| 场景 | 推荐设置 |
|------|----------|
| 数据集可以轻松装入显存 | 两个开关都保持默认关闭 |
| 数据集非常大，可以下采样动作/帧 | 在数据集准备阶段下采样（模式 1） |
| 数据集太大，偏好简单方案，接受变慢 | `--env.commands.motion.offload-to-cpu True`（模式 2） |
| 数据集太大，希望接近基线速度 | `--env.commands.motion.gpu-cache True`（模式 3） |

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
