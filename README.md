# Spider RL MJLab

基于 [mjlab](https://mujocolab.github.io/mjlab/)（MuJoCo Warp GPU 仿真）的 PHYBot Big V2 人形机器人**双手交替爬墙**强化学习环境。

## 任务

- **Task ID**: `Mjlab-Climb-Wall-Phybot-Big-V2`
- **机器人**: PHYBot Big V2（24 DOF 人形，腿 12 + 腰 2 + 双臂 10）
- **目标**: 两手交替贴墙向上爬（trot-like gait）
- **Adhesion**: 微刺式吸附（MuJoCo adhesion actuator + 自定义 engage/disengage 逻辑）

详细 reward / curriculum 设计见 [`docs/gait_rewards.md`](docs/gait_rewards.md)。

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
# 默认训练（128 envs）
python scripts/train.py Mjlab-Climb-Wall-Phybot-Big-V2

# 自定义环境数 / 迭代数
python scripts/train.py Mjlab-Climb-Wall-Phybot-Big-V2 \
  --env.scene.num-envs 256 \
  --agent.max-iterations 60000

# 多 GPU
python scripts/train.py Mjlab-Climb-Wall-Phybot-Big-V2 --gpu-ids "[0, 1]"
```

### 可视化 / 评估

```bash
# Zero 动作（机器人放松，看物理）—— 用来调 curriculum / 检查初始姿态
python scripts/play.py Mjlab-Climb-Wall-Phybot-Big-V2 --agent zero --num-envs 1

# 加载训练好的 checkpoint
python scripts/play.py Mjlab-Climb-Wall-Phybot-Big-V2 \
  --checkpoint-file logs/rsl_rl/<run>/model_<iter>.pt --num-envs 1

# 录像
python scripts/play.py Mjlab-Climb-Wall-Phybot-Big-V2 \
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
python scripts/tune_pose.py
```
浏览器打开 `http://localhost:8080`（远程: `ssh -L 8080:localhost:8080 user@host`）。
- 每个关节一个滑块
- Base x/y/z 滑块
- 点 **Print qpos / joint dict** 按钮把当前姿态 dump 到终端，复制回 `env_cfgs.py`

### 其他

```bash
# 单独查看机器人模型（不含墙，无 RL）
python -m spider_rl_mjlab.assets.phybot_big_v2.phybot_big_v2_constants
```

## 项目结构

```
spider_rl_mjlab/
├── README.md                                    # 本文件
├── docs/gait_rewards.md                         # Reward / curriculum 详细设计
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
        │   ├── observations.py                  # gait_clock (sin/cos)
        │   ├── rewards.py                       # gait_match / swing_hand_* / climb_vz ...
        │   └── curriculums.py                   # gravity_angle / reward_weight
        └── config/phybot_big_v2/
            ├── env_cfgs.py                      # 任务级 override（姿态、墙位置、adhesion）
            └── rl_cfg.py                        # PPO 超参
```

## Curriculum 概要

一条统一时间轴，4 个 phase：

| Phase | iter | 做什么 | 重力 |
|---|---|---|---|
| 0 | `0 → 5k` | 抓住墙（单手 + 双手接触） | 30° |
| 1 | `5k → 15k` | 学步态（两手交替 swing/stance） | 30° |
| 2 | `15k → 45k` | 重力从 30° 升到 90° | 30° → 90° |
| 3 | `45k → ∞` | 稳定垂直爬墙 | 90° |

关键 reward（Phase 1+ 打开）：
- `gait_match`（±1 节律匹配）
- `swing_hand_height`（swing 手抬高到 base+50cm）
- `swing_hand_clearance`（别在墙上蹭）
- `stance_hand_anchor`（支撑手防滑）
- `climb_vz`（向上速度）

详见 [`docs/gait_rewards.md`](docs/gait_rewards.md)。

## 主要依赖

- `mjlab`：GPU 仿真 + 管理器式 RL 环境
- `mujoco` / `mujoco-warp`：物理引擎
- `rsl-rl-lib`：PPO 训练
- `viser`：Web 可视化（play & tune_pose）
