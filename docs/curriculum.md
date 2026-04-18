# 课程学习（Curriculum）

配置：`src/spider_rl_mjlab/tasks/climb_wall/climb_wall_env_cfg.py:147`。
函数：`src/spider_rl_mjlab/tasks/climb_wall/mdp/curriculums.py`。

单一统一时间轴。机器人初始位姿就在墙边（`(-0.7, 0, 2)`），手臂朝墙伸出，
不需要"先学走路"阶段。贴墙 + 爬升 reward **从 iter 0 就全部激活**，curriculum
只做两件事：

1. 把 `hands_toward_wall` 的权重从 15 阶跃降到 5（P1_END 后）——gait 建立后
   削弱"靠近墙"这个过度 shaping。
2. 把 gravity 从 30°（倾斜，等价缓坡）线性插值到 90°（完全垂直墙）。

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

| Reward | iter 0 | P1_END | 备注 |
|---|---:|---:|---|
| `hands_toward_wall` | 8.0 | **3.0** | gait 建立后衰减 shaping |
| `hand_orientation` | 15.0 | （不变） | stance 相位门控 |
| `posture` | 3.0 | （不变） | Gaussian kernel per joint group |
| `hands_on_wall` | 8.0 | （不变） | 接触密度 |
| `swing_hand_height` | **10.0** | （不变） | paper sine arc 水平分量（离墙法向距离） |
| `swing_hand_reach_up` | **15.0** | （不变） | paper sine arc 竖直分量（hand_z - root_z）|
| `track_vx` | 6.0 | （不变） | $v_x$ 命令跟踪（cmd=0，σ=0.5 宽容伸手扰动） |
| `track_vy` | 6.0 | （不变） | $v_y$ 命令跟踪（cmd=0，σ=0.5） |
| `track_vz` | **40.0** | （不变） | **主目标**，$v_z$ 命令跟踪（cmd=0.3 m/s，σ=0.25 强信号） |
| `stance_phase_velocity` | 3.0 | （不变） | 反滑动 |

> 注：`airborne_penalty`、`both_hands_unlatched` termination、`climb_vz` (clamp)、
> `climb_height`、`swing_hand_reach_higher` **全部禁用**，纯靠 `track_vx/vy/vz`
> 三个高斯速度跟踪项驱动 policy 自由探索。`height_above_spawn` obs 仍保留
> 供 critic 描述 state。

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

Curriculum 不控制 termination——极端状态由三个 termination 项兜底：
- `time_out`：episode_length 到上限（30s）
- `base_height`：base z < 0.3 m（掉下墙）
- `bad_state`：速度 > 20 m/s 或 NaN/Inf（物理爆炸）

注：`both_hands_unlatched` termination 当前**禁用**，让 policy 自由探索 overlap
gait 而不是被硬约束卡住。函数保留在 `terminations.py` 里做 fallback。
