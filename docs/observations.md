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
- $T_{\text{gait}} = 2.0\,\text{s}$，$dt$：单步仿真步长（policy step）。

## 观测项

下面 9 项按顺序拼接，**总维度 133**。

### `base_lin_vel`（3 维）
$$
o = \mathbf{R}_b^\top \mathbf{v}_w \in \mathbb{R}^3
$$
来源 `asset.data.root_link_lin_vel_b`。base **本体系**下的线速度
$(v_x^b, v_y^b, v_z^b)$。

### `base_ang_vel`（3 维）
$$
o = \mathbf{R}_b^\top \boldsymbol{\omega}_w \in \mathbb{R}^3
$$
来源 `asset.data.root_link_ang_vel_b`。base 本体系下的角速度
$(\omega_x^b, \omega_y^b, \omega_z^b)$。

### `projected_gravity`（3 维）
$$
o = \mathbf{R}_b^\top \hat{\mathbf{g}}_w,\quad \hat{\mathbf{g}}_w = (0, 0, -1)
$$
来源 `asset.data.projected_gravity_b`。机器人完全竖直时 $o \approx (0, 0, -1)$；
前倾时 $o_x > 0$，侧倾时 $o_y \neq 0$。**不随 curriculum 重力倾角变化**——
由设计如此，姿态类 reward 仍以世界竖直为参考。

### `joint_pos_rel`（24 维）
$$
o_j = q_j - q_j^{\text{default}},\quad j = 1, \dots, 24
$$
24 DOF：腿 12 + 腰 2 + 双臂 10。

### `joint_vel_rel`（24 维）
$$
o_j = \dot{q}_j - \dot{q}_j^{\text{default}} = \dot{q}_j
$$
（默认速度全为 0，所以等于关节绝对速度。）

### `last_action`（24 维）
$$
o = \mathbf{a}_t
$$
当前步策略输出（缩放前的 raw action，由 action_manager 维护）。

### `prev_action`（24 维）
$$
o = \mathbf{a}_{t-1}
$$
来自 `action_manager.prev_action`，env reset 时清零。

### `prev_prev_action`（24 维）
$$
o = \mathbf{a}_{t-2}
$$
来自 `action_manager.prev_prev_action`，env reset 时清零。

### `gait_clock`（4 维）
相位：
$$
p(t) = \left(\frac{t \cdot dt}{T_{\text{gait}}}\right) \bmod 1,\qquad
p_L = p,\quad p_R = (p + 0.5) \bmod 1
$$
观测：
$$
o = \big[\sin(2\pi p_L),\ \cos(2\pi p_L),\ \sin(2\pi p_R),\ \cos(2\pi p_R)\big]
$$
$(\sin, \cos)$ 编码避免 $p = 1 \to 0$ 处的跳变。

## 维度小结

$$
3 + 3 + 3 + 24 + 24 + 24 + 24 + 24 + 4 = 133
$$

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
- microspine 的 `engaged`（每手吸附状态）**没有放进 obs**，而是 reward 直接读取。
  把它（2 维）加进观测是性价比最高的下一步——能直接告诉 policy "我吸住了吗"，
  对学步态有立竿见影的帮助。
