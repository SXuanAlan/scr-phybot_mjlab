# 课程学习（Curriculum）

配置：`src/spider_rl_mjlab/tasks/climb_wall/climb_wall_env_cfg.py:147`。
函数：`src/spider_rl_mjlab/tasks/climb_wall/mdp/curriculums.py`。

单一统一时间轴。机器人初始位姿就在墙边（`(-0.7, 0, 2)`），手臂朝墙伸出，
不需要"先学走路"阶段。贴墙 + 爬升 reward **从 iter 0 就全部激活**，curriculum
只做一件事：

1. 在 30° 保持较长时间，然后线性插值到 60°、90°（完全垂直墙）。

所有 reward weight 都是单 stage 常量（iter 0 开始就是目标值）。

## 阶段边界

| 常量 | 值 | 含义 |
|---|---:|---|
| C2 stage 1 | 1500 | 30° 固定低载荷阶段，先学稳定抓墙 / swing pattern |
| C2 stage 2 | 2500 | 从 30° 过渡到 60° |
| C2 stage 3 | 3500 | 从 60° 过渡到 90° |

迭代计数：`env.common_step_counter // steps_per_iter`，其中
`steps_per_iter = 24`（必须与 `rl_cfg.py` 里的 `num_steps_per_env` 一致）。

## 阶段总览

| Phase | iter 区间 | 重力倾角 | 发生什么 |
|---|---|---:|---|
| 1 | `0 → 1500` | 30° | 延长低载荷阶段，先学稳定抓墙 / gait shaping |
| 2 | `1500 → 2500` | 30° → 60° | 增加沿墙下拽分量 |
| 3 | `2500 → 3500` | 60° → 90° | 过渡到完全垂直墙 |
| 4 | `> 3500` | 90° | 完全垂直墙，稳定 climbing gait |

## Reward 权重

C2 climb-wall 当前只保留 gravity-angle curriculum；reward 权重不再随 iter
调度，全部从 iter 0 开始使用常量。`reward_weight(...)` helper 仍保留给
base/legacy 配置使用。

### 按项展开

**任务奖励**

| Reward | iter 0 | 备注 |
|---|---:|---|
| `ee_grip_time` | +10.0 | 14:51:31 版 dense grip/swing timer；要求正好 2 个 EE engaged |
| `climb_height_progress` | +10.0 | 只奖励 episode 内 base z 新高，抑制原地顶胯 |
| `track_lin_vel_xy` | +4.0 | 世界系 x/y 速度 tracking，软约束 lateral drift |
| `track_lin_vel_z` | +8.0 | 世界系 z 速度 tracking，辅助爬升速度 |
| `ee_face_parallel_to_wall` | +2.0 | spine 接触面贴墙 |
| `ee_slide_penalty` | −0.1 | 迁自 feet/slip 思路（engaged-gated） |

**Regulation**（对齐 G1 walking，无 curriculum 调度，全程常量）

| Reward | weight | joint scope / 备注 |
|---|---:|---|
| `termination_penalty` | −200.0 | `is_terminated`（不含 timeout） |
| `ang_vel_xy_l2` | −0.05 | base roll/pitch angular velocity |
| `action_rate_l2` | −0.01 | 全 action dim |
| `dof_acc_l2` | −1.0e-5 | 全 robot actuator scope |
| `dof_torques_l2` | −1.0e-5 | 全 robot actuator scope |

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

`gravity_vec_w`（每个 entity 上的）**不更新**，所以姿态类 reward 可以继续相对
世界竖直方向测量——对"沿墙向上"这个目标来说这是正确的参考方向。policy 的
`projected_gravity` 观测则直接读取 `wp_model.opt.gravity`，会跟随真实重力倾角变化。

### Stage 表

PHYBot C2 当前使用更长的 30° 固定阶段，速度 command 保持固定
$(0, 0, 0.08)$，不做 `vz` 随机化。

| Stage | iter | 倾角 |
|---|---:|---:|
| 0 | 0 | 30° |
| 1 | 1500 | 30° |
| 2 | 2500 | 60° |
| 3 | 3500 | 90° |

### 插值

`_interp_angle` 在相邻 stage 之间按 $\sin(\theta)$ 线性插值，因为沿墙重力
负载正比于 $g\sin(\theta)$。设当前迭代 $i$ 落在 $[i_a, i_b]$ 区间，则：
$$
\alpha = \frac{i - i_a}{\max(i_b - i_a,\ 1)},\qquad
\sin\theta(i) = \sin\theta_a + \alpha(\sin\theta_b - \sin\theta_a)
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
切换到 G1-style 后改为靠 reward 层软塑形 + 加强 `termination_penalty`，让 policy 在
连续梯度里学，而不是碰到边界硬 reset。

这和 G1 walking 的哲学一致——只保留最小必要的硬 termination，其余全部
交给 shaped reward。如果发现 policy 仍然钻新空子，首选是调 reward
权重，其次才是加回硬约束。
