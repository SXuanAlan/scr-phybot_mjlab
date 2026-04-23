# 观测（Observations）

只有一个观测组 `actor`（actor 和 critic 共用一组观测，没有独立的特权 critic）。
配置位置：`src/spider_rl_mjlab/tasks/climb_wall/climb_wall_env_cfg.py:35`。
自定义观测函数：`src/spider_rl_mjlab/tasks/climb_wall/mdp/observations.py`。

## 符号约定

- $\mathbf{R}_b \in SO(3)$：base 在世界系下的旋转矩阵（root link）。
- $\mathbf{v}_w, \boldsymbol{\omega}_w$：base 在世界系下的线/角速度。
- $\mathbf{g}_w = (0, 0, -9.81)$：世界系重力向量（curriculum 倾斜时已经写到
  `wp_model.opt.gravity`，但 `projected_gravity_b` 用每个 entity 的本地
  `gravity_vec_w`，**不**跟 curriculum 倾，仍指向世界 −z）。
- $\mathbf{q} \in \mathbb{R}^{24}$：24 DOF 的关节角；$\mathbf{q}^{\text{default}}$
  默认姿态。
- $T_{\text{gait}} = 4.0\,\text{s}$，$dt$：单步仿真步长（policy step）。

## 观测项

下面 12 项按顺序在 last 维拼接成 `(E, 139)` 的张量送进 actor。`E = num_envs`。

### `base_lin_vel` — `(E, 3)`
$$
o[e] = \mathbf{R}_{b,e}^\top \mathbf{v}_{w,e}
$$
来源 `asset.data.root_link_lin_vel_b`，shape `(E, 3)`。base **本体系**下的
线速度 $(v_x^b, v_y^b, v_z^b)$。

### `base_ang_vel` — `(E, 3)`
$$
o[e] = \mathbf{R}_{b,e}^\top \boldsymbol{\omega}_{w,e}
$$
来源 `asset.data.root_link_ang_vel_b`，shape `(E, 3)`。base 本体系下的
角速度 $(\omega_x^b, \omega_y^b, \omega_z^b)$。

### `projected_gravity` — `(E, 3)`
$$
o[e] = \mathbf{R}_{b,e}^\top \hat{\mathbf{g}}_w,\quad \hat{\mathbf{g}}_w = (0, 0, -1)
$$
来源 `asset.data.projected_gravity_b`，shape `(E, 3)`。机器人完全竖直时
$o \approx (0, 0, -1)$；前倾时 $o_x > 0$，侧倾时 $o_y \neq 0$。**不随
curriculum 重力倾角变化**——由设计如此，姿态类 reward 仍以世界竖直为参考。

### `joint_pos_rel` — `(E, 24)`
$$
o[e, j] = q_{e, j} - q_j^{\text{default}},\quad j = 0, \dots, 23
$$
24 DOF：腿 12 + 腰 2 + 双臂 10。

### `joint_vel_rel` — `(E, 24)`
$$
o[e, j] = \dot{q}_{e, j} - \dot{q}_j^{\text{default}} = \dot{q}_{e, j}
$$
（默认速度全为 0，所以等于关节绝对速度。）

### `last_action` — `(E, 24)`
$$
o[e] = \mathbf{a}_{t, e}
$$
当前步策略输出（缩放前的 raw action，由 action_manager 维护）。

### `prev_action` — `(E, 24)`
$$
o[e] = \mathbf{a}_{t-1, e}
$$
来自 `action_manager.prev_action`，env reset 时清零。

### `prev_prev_action` — `(E, 24)`
$$
o[e] = \mathbf{a}_{t-2, e}
$$
来自 `action_manager.prev_prev_action`，env reset 时清零。

### `gait_clock` — `(E, 4)`
相位（标量按 env 广播）：
$$
p_e(t) = \left(\frac{t_e \cdot dt}{T_{\text{gait}}}\right) \bmod 1,\qquad
p_{L,e} = p_e,\quad p_{R,e} = (p_e + 0.5) \bmod 1
$$
观测：
$$
o[e] = \big[\sin(2\pi p_{L,e}),\ \cos(2\pi p_{L,e}),\ \sin(2\pi p_{R,e}),\ \cos(2\pi p_{R,e})\big]
$$
$(\sin, \cos)$ 编码避免 $p = 1 \to 0$ 处的跳变。

> ⚠️ 当前 gait reward (`hand_grip_time`) 已经不使用 gait clock——步态节奏由
> 事件驱动（`engaged` 状态转变）而非时钟。`gait_clock` 仍保留在 obs 中作为
> 辅助节奏参考：policy 可以选用这 4 维帮助学出周期性动作，也可以完全忽略。
> 未来如果确认 policy 不需要，可以一并删掉。

### `height_above_spawn` — `(E, 1)`
$$
o[e] = z_{b,e} - z_{\text{spawn}},\quad z_{\text{spawn}} = 2.0
$$

机器人 base 相对出生点 z 高度。即便当前没用位置式 reward，这个观测仍有
价值——critic 能据此判断 state value 如何随高度变化（爬高离掉地 terminate
就越远、episode 更稳定等）。未来若加回 `climb_height` 也无需再改 obs。

### `velocity_command` — `(E, 3)`
$$
o[e] = (v_{x,\text{cmd}},\ v_{y,\text{cmd}},\ v_{z,\text{cmd}})
$$

世界系命令速度，目前固定为 $(0, 0, 0.08)$。policy 看到这个命令就知道期望
的运动方向和速率，配合 `track_vz` 奖励做标准的 commanded-velocity policy。
`hand_grip_time` 也读 $v_z^{\text{cmd}}$ 作为 move-gate 的判据。未来可以接
command manager 做 per-env 命令随机化。

### `hand_engaged_state` — `(E, 2)`
$$
o[e] = \big[\mathbb{1}[c_{L,e} = 1],\ \mathbb{1}[c_{R,e} = 1]\big]
$$

microspine 左/右手当前 latch 状态，直接从 `event_manager.get_term_cfg("microspine_adhesion").func.engaged`
读（bool → float）。让 policy 能和抓墙状态协调动作（已抓住的手别再"试抓"）。

## 维度小结

```
concat([(E,3), (E,3), (E,3), (E,24), (E,24), (E,24), (E,24), (E,24), (E,4), (E,2), (E,1), (E,3)], dim=-1)
= (E, 139)
```

十二项依次：base_lin_vel、base_ang_vel、projected_gravity、joint_pos_rel、
joint_vel_rel、last_action、prev_action、prev_prev_action、gait_clock、
hand_engaged_state、height_above_spawn、velocity_command。

## EMA 低通滤波（备用，当前未启用）

`observations.py` 里保留了 `EMAObsFilter(base_func, alpha=0.35)` 工具函数。
需要抑制 CCD 抖动 / 接触噪声时再启用，包装方式：
```python
ObservationTermCfg(func=mdp.EMAObsFilter(mdp.base_lin_vel, alpha=0.35))
```
公式（参考 arXiv:2510.20174 §IV-a）：
$$
o_t^{\text{filt}} = (1 - \alpha) \cdot o_t^{\text{raw}} + \alpha \cdot o_{t-1}^{\text{filt}},
\quad \alpha = 0.35
$$
返回的可调用对象带 `.reset` 属性，env reset 时观测管理器把对应 env 的 buffer
标记为 fresh，下一帧用原始值（不跨 reset 平滑）。

## 备注

- 没有特权 critic 组。要走 sim-to-real 时应该升级成 asymmetric actor-critic
  + state estimator（参考 arXiv:2510.20174）。
- `height_above_spawn` 是**绝对位置信息**——真机部署需要外部测量（视觉定位
  / 起点 anchor）。若不可观测，替代方案是去掉 `climb_height` reward、只保留
  velocity tracking。
- `velocity_command` 目前是全 env 共享常量；后续可以换成 CommandManager
  做随机化，让 policy 学出 "任意 vz 目标 → 对应爬升速度" 的条件策略。
