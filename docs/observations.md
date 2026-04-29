# 观测（Observations）

只有一个观测组 `actor`（actor 和 critic 共用一组观测，没有独立的特权 critic）。
配置位置：`src/spider_rl_mjlab/tasks/climb_wall/climb_wall_env_cfg.py:35`。
自定义观测函数：`src/spider_rl_mjlab/tasks/climb_wall/mdp/observations.py`。

## 符号约定

- $\mathbf{R}_b \in SO(3)$：base 在世界系下的旋转矩阵（root link）。
- $\mathbf{v}_w, \boldsymbol{\omega}_w$：base 在世界系下的线/角速度。
- $\mathbf{g}_w$：世界系重力向量。curriculum 倾斜时写到
  `wp_model.opt.gravity`；`projected_gravity` 读取该真实重力方向并归一化。
- $\mathbf{q} \in \mathbb{R}^{24}$：24 DOF 的关节角；$\mathbf{q}^{\text{default}}$
  默认姿态。
- $T_{\text{gait}} = 4.0\,\text{s}$，$dt$：单步仿真步长（policy step）。

## 观测项

base/Big-V2 配置下面 12 项按顺序在 last 维拼接成 `(E, 139)` 的张量送进 actor。
C2 climb-wall 去掉 `gait_clock`，但 4 个 spine EE 的 `hand_engaged_state`
从 2 维变成 4 维。
`E = num_envs`。

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
o[e] = \mathbf{R}_{b,e}^\top \hat{\mathbf{g}}_w,\quad
\hat{\mathbf{g}}_w = \frac{\mathbf{g}_w}{\|\mathbf{g}_w\|}
$$
来源 `env.sim.wp_model.opt.gravity` 和 base quaternion sensor，shape `(E, 3)`。
机器人完全竖直且重力为标准竖直向下时 $o \approx (0, 0, -1)$；curriculum
倾斜时该观测会随真实重力方向变化。姿态类 reward 若需要世界竖直参考，则单独读
entity 的 nominal `gravity_vec_w`。

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

> 当前 C2 climb-wall 配置已移除 `gait_clock`，和 MJLab/IsaacLab velocity
> tracking 保持一致：步态不由 phase observation 指定，而是由 velocity tracking
> + `ee_grip_time` adhesion shaping 自发形成。base/Big-V2 配置仍保留该项。

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

世界系命令速度由 `ClimbVelocityCommand` 提供，当前范围固定为
$(0, 0, 0.08)$。policy 看到这个命令就知道期望的运动方向和速率，
`track_lin_vel_xy` / `track_lin_vel_z` reward 也读同一个 command，因此 observation
和 reward 的目标一致。
当前没有启用 per-env `vz` 随机化。

### `hand_engaged_state` — `(E, 2)`
$$
o[e] = \big[\mathbb{1}[c_{L,e} = 1],\ \mathbb{1}[c_{R,e} = 1]\big]
$$

microspine 左/右手当前 latch 状态，直接从 `event_manager.get_term_cfg("microspine_adhesion").func.engaged`
读（bool → float）。让 policy 能和抓墙状态协调动作（已抓住的手别再"试抓"）。

> C2 climb-wall 使用 4 个 spine EE，所以该项是 `(E, 4)`。

## 维度小结

```
concat([(E,3), (E,3), (E,3), (E,24), (E,24), (E,24), (E,24), (E,24), (E,4), (E,2), (E,1), (E,3)], dim=-1)
= (E, 139)  # base/Big-V2
```

base/Big-V2 十二项依次：base_lin_vel、base_ang_vel、projected_gravity、joint_pos_rel、
joint_vel_rel、last_action、prev_action、prev_prev_action、gait_clock、
hand_engaged_state、height_above_spawn、velocity_command。

C2 climb-wall 去掉 `gait_clock`，但 `hand_engaged_state` 是 4 个 spine EE，
所以 actor obs 维度为 `(E, 137)`。

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
- `velocity_command` 现在来自 CommandManager，但当前 command range 是固定的；
  之后若重新启用速度 curriculum，需要同时确认 gait reward 的 move-gate 是否也要读 command。
