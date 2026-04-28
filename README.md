# Spider RL MJLab

基于 [mjlab](https://mujocolab.github.io/mjlab/)（MuJoCo Warp GPU 仿真）的 PHYBot C2 人形机器人**四肢 spine 爬墙**强化学习环境。

## 任务

- **Task ID**: `Mjlab-Climb-Wall-Phybot-C2`
- **机器人**: PHYBot C2（23 DOF 人形 + 4 个 fixed spine：左右 hand_spine + 左右 foot_spine）
- **目标**: 4 个 spine 贴墙向上爬
- **Adhesion**: 微刺式吸附（MuJoCo adhesion actuator + 自定义 engage/disengage 逻辑），每个 spine 4 个接触点（共 16）

> Big V2 双手交替爬墙任务仍保留，task id `Mjlab-Climb-Wall-Phybot-Big-V2`。

详细 reward / curriculum 设计见 [`docs/rewards.md`](docs/rewards.md) 和
[`docs/curriculum.md`](docs/curriculum.md)（`docs/gait_rewards.md` 是早期
设计稿，已不对齐当前实现）。

## 安装

```bash
conda activate env_mjlab
cd /home/sxuan/projects/spider_rl_mjlab
pip install -e .
```

## 常用命令

### 列出已注册的任务
```bash
python scripts/list_envs.py
```

### 训练
```bash

ssh -L 8080:localhost:8080 spider@bdml-server.stanford.edu

# 多 GPU
python scripts/train.py Mjlab-Climb-Wall-Phybot-C2 --gpu-ids "[0, 1]"

# headless
nohup python scripts/train.py Mjlab-Climb-Wall-Phybot-C2 \
  --env.scene.num-envs 4096 \
  --agent.max-iterations 8000 \
  > train.log 2>&1 &

ps -fu $USER | grep -E 'train|python' | grep -v grep
kill <PID>
```

### 可视化 / 评估

```bash
# Zero 动作（机器人放松，看物理）—— 用来调 curriculum / 检查初始姿态
python scripts/play.py Mjlab-Climb-Wall-Phybot-C2 --agent zero --num-envs 1

# 加载训练好的 checkpoint
python scripts/play.py Mjlab-Climb-Wall-Phybot-C2 \
  --checkpoint-file logs/rsl_rl/<run>/model_<iter>.pt --num-envs 1

# 录像
python scripts/play.py Mjlab-Climb-Wall-Phybot-C2 \
  --checkpoint-file <path> --video
```

**Play 模式**会自动开启 `microspine` 的 debug：
- 终端打印 env 0 的每次 engage/disengage 转换
- Viser viewer 里画一根橙色箭头指向当前重力方向

### TensorBoard

```bash
tensorboard --logdir logs/rsl_rl --bind_all
# 远程访问: ssh -L 6006:localhost:6006 user@host
```
浏览器打开 `http://localhost:6006`。

### 交互式姿态调参（viser 滑块）

调初始姿态时用：
```bash
python scripts/tune_pose.py                    # 默认 c2（含墙）
python scripts/tune_pose.py --robot phybot_big_v2
```
浏览器打开 `http://localhost:8080`（远程: `ssh -L 8080:localhost:8080 user@host`）。
- 每个关节一个滑块
- Base x/y/z 滑块
- 点 **Print qpos / joint dict** 按钮把当前姿态 dump 到终端，复制回 `env_cfgs.py`

### 其他

```bash
# 单独查看机器人模型（不含墙，无 RL）
python -m spider_rl_mjlab.assets.phybot_c2.phybot_c2_constants
python -m spider_rl_mjlab.assets.phybot_big_v2.phybot_big_v2_constants
```

## 项目结构

```
spider_rl_mjlab/
├── README.md                                    # 本文件
├── docs/
│   ├── rewards.md                               # 当前 reward 设计（权威）
│   ├── curriculum.md                            # 当前 curriculum + termination
│   ├── observations.md                          # 当前 obs 设计
│   └── gait_rewards.md                          # 早期方案（历史参考）
├── scripts/
│   ├── train.py                                 # 训练入口
│   ├── play.py                                  # 评估 / 可视化
│   ├── tune_pose.py                             # 交互式姿态调参（viser）
│   └── list_envs.py                             # 列出注册任务
└── src/spider_rl_mjlab/
    ├── assets/
    │   ├── phybot_big_v2/                       # 机器人：MJCF + actuator/keyframe
    │   ├── wall/                                # 墙 entity（10m × 3m × 1m）
    │   └── microspine/                          # 微刺吸附控制器
    │       ├── config.py                        # MicrospineCfg
    │       └── controller.py                    # engage / disengage 逻辑 + debug_vis
    └── tasks/climb_wall/
        ├── climb_wall_env_cfg.py                # 核心: obs/action/reward/curriculum
        ├── mdp/
        │   ├── observations.py                  # gait_clock / height_above_spawn / velocity_command ...
        │   ├── rewards.py                       # hand_grip_time / hand_slide / climb_vz_gauss ...
        │   ├── terminations.py                  # static_hang / both_hands_off / excessive_tilt ...
        │   └── curriculums.py                   # gravity_angle / reward_weight
        └── config/phybot_big_v2/
            ├── env_cfgs.py                      # 任务级 override（姿态、墙位置、adhesion）
            └── rl_cfg.py                        # PPO 超参
```

## Curriculum 概要

一条统一时间轴。机器人出生就贴墙，所有 reward 从 iter 0 全量激活，
curriculum 只调 gravity 倾角：

| Phase | iter | 重力 |
|---|---|---|
| 1 | `0 → 15k` | 10° → 30°（低载荷 motor pattern 发现） |
| 2 | `15k → 45k` | 30° → 90°（线性升到垂直） |
| 3 | `45k → ∞` | 90°（标准垂直爬墙） |

核心 reward 集合（全程激活）：
- `hand_grip_time` — strict single-stance 版迁自 G1 `feet_air_time_positive_biped`
- `track_vz` — 非对称高斯竖直爬升驱动（target 0.08 m/s）
- `hand_orientation` — 手面朝墙（两相位都激活）
- `flat_orientation` / `torso_orientation` — pelvis + 腰绕 body-x 侧翻罚

Regulation（对齐 G1 walking 模板，权重全部 G1 原样）：
- `hand_slide` (−0.1) 迁自 G1 `feet_slide`
- `termination_penalty` (−200)
- `joint_deviation_hip` (−0.1，hip_yaw + hip_roll)
- `joint_deviation_torso` (−0.1，waist_yaw + waist_roll)
- `dof_pos_limits` (−1.0，ankles)
- `action_rate_l2` (−0.005)
- `dof_acc_l2` (−1.25e-7，hip + knee)
- `dof_torques_l2` (−1.5e-7，hip + knee + ankle)
- 故意不迁 `joint_deviation_arms`（手臂是爬墙主动执行器）

Termination（极简，对齐 G1）：
- `time_out` / `base_height`（climb 里 `base_contact` 的对偶，robot 掉下墙）
 / `bad_state`（NaN 数值守护）

详见 [`docs/rewards.md`](docs/rewards.md) 和 [`docs/curriculum.md`](docs/curriculum.md)。

## 主要依赖

- `mjlab`：GPU 仿真 + 管理器式 RL 环境
- `mujoco` / `mujoco-warp`：物理引擎
- `rsl-rl-lib`：PPO 训练
- `viser`：Web 可视化（play & tune_pose）
