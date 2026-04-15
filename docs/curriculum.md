# 课程学习（Curriculum）

配置：`src/spider_rl_mjlab/tasks/climb_wall/climb_wall_env_cfg.py:212`。
函数：`src/spider_rl_mjlab/tasks/climb_wall/mdp/curriculums.py`。

单一统一时间轴。机器人初始位姿就在墙边（`(-0.7, 0, 2)`），手臂朝墙伸出，
不需要"先学走路"阶段。

## 阶段边界

| 常量 | 值 | 含义 |
|---|---:|---|
| `P0_END` | 5000 | 双手能稳定贴墙 |
| `P1_END` | 15000 | 30° 倾斜下学会步态 |
| `P2_END` | 45000 | 重力到达完全垂直 |

迭代计数：`env.common_step_counter // steps_per_iter`，其中
`steps_per_iter = 24`（必须与 `rl_cfg.py` 里的 `num_steps_per_env` 一致）。

## 阶段总览

| Phase | iter 区间 | 发生什么 | 重力倾角 |
|---|---|---|---:|
| 0 | `0 → P0_END` | 抓墙类 reward 饱和；both-hands bonus 把"双手压墙"压实 | 30° |
| 1 | `P0_END → P1_END` | 步态类 reward 打开（gait_match、swing_height、swing_clearance、stance_anchor、climb_vz）；both-hands bonus 从 30 衰减到 5 | 30° |
| 2 | `P1_END → P2_END` | 重力 **30° → 90°** 线性插值（30 k 迭代）；步态继续训 | 30° → 90° |
| 3 | `> P2_END` | 完全垂直墙，交替步态稳定 | 90° |

## Reward 权重调度

`reward_weight(reward_name, weight_stages, steps_per_iter=24)` 在 stage 列表
里选**最后一个 `iter < current_iter` 的 stage**。所以 stages 是阶跃切换，
不做插值。

### 抓墙类（早期为主，后期衰减）

| Reward | iter 0 | P0_END | P1_END |
|---|---:|---:|---:|
| `hands_toward_wall` | 15.0 | 15.0 | 5.0 |
| `hand_orientation` | 15.0 | 15.0 | 5.0 |
| `hands_on_wall` | 20.0 | （不变） | （不变） |
| `com_near_wall` | 5.0 | （不变） | （不变） |
| `both_hands_bonus` | 30.0 | 15.0 | 5.0 |

### 步态（Phase 1 之前关闭）

`P0_END` 之前都是 `0.0`，之后阶跃到目标值：

| Reward | iter 0 | P0_END 之后 |
|---|---:|---:|
| `gait_match` | 0 | +8.0 |
| `swing_hand_height` | 0 | +10.0 |
| `swing_hand_clearance` | 0 | −50.0 |
| `stance_hand_anchor` | 0 | −2.0 |
| `climb_vz` | 0 | +20.0 |

### 稳定性（常数）

| Reward | 权重 |
|---|---:|
| `torso_orientation` | +5.0 |
| `flat_orientation` | −2.0 |
| `base_height_penalty` | −10.0 |
| `posture` | +1.0 |
| `torso_ang_vel` | −1.0 |
| `forward_velocity`、`lin_vel_xy` | 0（关闭） |

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

`gravity_vec_w`（每个 entity 上的）**不更新**，所以姿态类 reward
（`torso_orientation`、`flat_orientation_l2`）继续相对世界竖直方向测量——
对"沿墙向上"这个目标来说这是正确的参考方向。

### Stage 表

| Stage | iter | 倾角 |
|---|---:|---:|
| 0 | 0 | 30° |
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

### Reward weight 调度

`reward_weight(weight_stages)` **不**做插值，而是**阶跃**切换：选最后一个
$i_k < i$ 的 stage 的权重。也即权重函数：
$$
w(i) = w_{k^*},\quad k^* = \max\{k : i_k < i\}
$$
