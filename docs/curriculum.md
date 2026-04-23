# 课程学习（Curriculum）

配置：`src/spider_rl_mjlab/tasks/climb_wall/climb_wall_env_cfg.py:147`。
函数：`src/spider_rl_mjlab/tasks/climb_wall/mdp/curriculums.py`。

单一统一时间轴。机器人初始位姿就在墙边（`(-0.7, 0, 2)`），手臂朝墙伸出，
不需要"先学走路"阶段。贴墙 + 爬升 reward **从 iter 0 就全部激活**，curriculum
只做一件事：

1. 把 gravity 从 10°（缓坡）线性插值到 30° 再到 90°（完全垂直墙）。

所有 reward weight 都是单 stage 常量（iter 0 开始就是目标值）。

## 阶段边界

| 常量 | 值 | 含义 |
|---|---:|---|
| `P1_END` | 15000 | 30° 重力下基本学会爬；开始倾斜重力 |
| `P2_END` | 45000 | 重力完全垂直，进入正式攀爬 |

迭代计数：`env.common_step_counter // steps_per_iter`，其中
`steps_per_iter = 24`（必须与 `rl_cfg.py` 里的 `num_steps_per_env` 一致）。

## 阶段总览

| Phase | iter 区间 | 重力倾角 | 发生什么 |
|---|---|---:|---|
| 1 | `0 → P1_END` | 10° → 30° | 从"几乎水平"线性升到 30°，让 policy 早期在低重力载荷下发现 climbing motor pattern |
| 2 | `P1_END → P2_END` | 30° → 90° | 重力线性升角，30k 迭代缓慢增加"向下"分量 |
| 3 | `> P2_END` | 90° | 完全垂直墙，稳定 climbing gait |

## Reward 权重调度

`reward_weight(reward_name, weight_stages, steps_per_iter=24)` 在 stage 列表
里选**最后一个 `iter <= current_iter` 的 stage**。权重是阶跃切换（非插值）。

### 按项展开

**任务奖励**（curriculum 只对这几项开权重 stage；regulation 在 iter 0 就是常量）

| Reward | iter 0 | 备注 |
|---|---:|---|
| `hand_orientation` | +15.0 | 两相位都激活 |
| `hand_grip_time` | +30.0 | 迁自 G1 `feet_air_time_positive_biped`（strict ∑e == 1） |
| `track_vz` | +40.0 | 迁自 G1 `track_lin_vel_xy_exp`（vz-only 非对称高斯） |
| `flat_orientation` | +10.0 | pelvis roll-only, σ=0.3 |
| `torso_orientation` | +10.0 | `waist_roll` body roll-only, σ=0.3 |
| `hand_slide` | −0.1 | 迁自 G1 `feet_slide`（engaged-gated） |

**Regulation**（对齐 G1 walking，无 curriculum 调度，全程常量）

| Reward | weight | joint scope / 备注 |
|---|---:|---|
| `termination_penalty` | −200.0 | `is_terminated`（不含 timeout） |
| `dof_pos_limits` | −1.0 | ankles（G1 原样） |
| `joint_deviation_hip` | −0.1 | hip_yaw + hip_roll（G1 原样） |
| `joint_deviation_torso` | −0.1 | waist_yaw + waist_roll |
| `action_rate_l2` | −0.005 | 全 action dim |
| `dof_acc_l2` | −1.25e-7 | hip + knee |
| `dof_torques_l2` | −1.5e-7 | hip + knee + ankle |

> **故意不迁**：G1 的 `joint_deviation_arms` / `joint_deviation_fingers`。
> 肩、肘在爬墙时是主动执行器（必须大幅偏离 keyframe 才能抓上去），再
> 加约束会把任务禁掉；PHYBot Big V2 也没有 fingers。

## 重力 curriculum

`gravity_angle(angle_stages, gravity_magnitude=9.81)` 通过 torch view 原地
写 `env.sim.wp_model.opt.gravity`（不重新分配 warp 数组——否则 CUDA graph 会
继续读旧 buffer）。

**墙几何始终保持竖直**，只让重力向量倾斜。设倾角为 $\theta$（度），重力
向量为：
$$
\mathbf{g}(\theta) = \big(g \cos\theta,\ 0,\ -g \sin\theta\big),\quad g = 9.81
$$

- $\theta = 90°$：$\mathbf{g} = (0, 0, -9.81)$，标准垂直墙攀爬。
- $\theta = 30°$：$\mathbf{g} = (8.50, 0, -4.91)$，重力主要把机器人**压向**
  墙面（+x 方向），只有 $\sin 30° = 50\%$ 沿墙"上方向"——物理上等价于 30° 缓坡。
  在 Phase 1 让 policy 先学"怎么贴墙并蹬上"，然后再缓慢加大沿墙下拽。

`gravity_vec_w`（每个 entity 上的）**不更新**，所以姿态类 reward 继续相对
世界竖直方向测量——对"沿墙向上"这个目标来说这是正确的参考方向。

### Stage 表

| Stage | iter | 倾角 |
|---|---:|---:|
| 0 | 0 | 10° |
| 1 | `P1_END`（15000） | 30° |
| 2 | `P2_END`（45000） | 90° |

### 线性插值

`_interp_angle`（`curriculums.py:48`）在相邻 stage 之间线性插值。设当前
迭代 $i$ 落在 $[i_a, i_b]$ 区间，对应倾角 $\theta_a, \theta_b$，则：
$$
\alpha = \frac{i - i_a}{\max(i_b - i_a,\ 1)},\qquad
\theta(i) = \theta_a + \alpha (\theta_b - \theta_a)
$$
区间外取最近端点的角度。

## Termination 兜底

对齐 G1 walking 的极简集合——G1 只有 `time_out + base_contact`，我们对应
把"base 掉下墙"替换 base_contact，再加一条纯数值安全网：

- `time_out`: episode_length 到上限（30 s）
- `base_height`: base z < 0.3 m（**掉下墙**，爬墙里 G1 `base_contact` 的
  对偶——没有地面碰撞可用，用高度下限替代）
- `bad_state`: 速度 > 20 m/s 或 NaN/Inf 等数值发散守护

**已删除的 termination**（从上一版保留）：`excessive_tilt`、`torso_roll_excessive`、
`static_hang`、`both_hands_off`。这些都是为了堵特定退化解手工加的硬约束；
切换到 G1-style 后改为靠 reward 层（`flat_orientation` + `torso_orientation`
+ `joint_deviation_*`）软塑形 + 加强 `termination_penalty`，让 policy 在
连续梯度里学，而不是碰到边界硬 reset。

这和 G1 walking 的哲学一致——只保留最小必要的硬 termination，其余全部
交给 shaped reward。如果发现 policy 仍然钻新空子，首选是调 reward
权重，其次才是加回硬约束。
