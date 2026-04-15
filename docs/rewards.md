# 奖励（Rewards）

配置：`src/spider_rl_mjlab/tasks/climb_wall/climb_wall_env_cfg.py:70`。
自定义 reward 函数：`src/spider_rl_mjlab/tasks/climb_wall/mdp/rewards.py`。

下表里的"默认权重"是 `RewardTermCfg` 里声明的初始值，**curriculum 会在运行时
覆盖**（详见 `curriculum.md`），所以这一列只是 curriculum 启动前的初值。

## 符号约定

- $E$ 环境数；$h \in \{L, R\}$ 左/右手；$s \in \{1,\dots,16\}$ 16 个吸附球（每手 8）。
- $p \in [0, 1)$ 步态相位，$p_L = p$，$p_R = (p + 0.5) \bmod 1$。
- $c_h \in \{0, 1\}$：第 $h$ 手的 `engaged` 锁定状态。
- $\mathbf{n}_w$：墙的 inward normal（指向机器人方向，世界系）。
- $d_s$：第 $s$ 个球面到墙面的有符号距离（正=有缝，零=贴墙，负=穿透）。
- $\mathbf{g}_b$：base 系下重力方向。
- $z_b$：base 在世界系下的 z 坐标；$z_h$：第 $h$ 手在世界系下的 z 坐标。

---

## 抓墙类（Phase 0+，从 iter 0 就生效）

### `hands_toward_wall`（默认权重 +15.0）
$$
R = -\min\!\left(\overline{|d_s|},\ 0.5\right)
$$
对 16 个球的 $|d_s|$ 取均值并 clip，靠近=负值更小=奖励更高。任意距离都有
连续梯度。

### `hand_orientation`（默认权重 +15.0）
设 $\mathbf{f}_h$ 为第 $h$ 手 local $-x$ 面在世界系下的方向向量（接触面法线），
墙的 inward 方向为 $\mathbf{t} = -\mathbf{n}_w^{\text{outward}}$，则：
$$
R = \frac{1}{2}\sum_{h \in \{L,R\}} \frac{\mathbf{f}_h \cdot \mathbf{t} + 1}{2}
\in [0, 1]
$$

### `hands_on_wall`（默认权重 +20.0）
$$
R = \frac{1}{16}\sum_{s=1}^{16} \mathbb{1}\!\left[|d_s| < 0.02\,\text{m}\right]
$$
16 个球面距墙 < 20 mm 的占比。

### `both_hands_bonus`（默认权重 +30.0）
$$
R = \left(\tfrac{1}{8}\sum_{s \in L} \mathbb{1}[|d_s| < 0.02]\right) \cdot
    \left(\tfrac{1}{8}\sum_{s \in R} \mathbb{1}[|d_s| < 0.02]\right)
$$
两手接触率的乘积。任一手脱离 → 0。

### `com_near_wall`（默认权重 +5.0）
设 base 在世界系下到墙平面的有符号垂直距离为 $\delta_\perp$，则：
$$
R = \exp\!\left(-\frac{|\delta_\perp|}{0.5}\right)
$$
墙平面位姿来自 wall mocap，curriculum 改墙倾角时自动跟随。

---

## 稳定性 / 姿态（常数权重）

### `torso_orientation`（默认权重 +5.0）
$$
R = \exp\!\left(-\|\mathbf{g}_b^{xy}\|^2\right)
$$
完全竖直时 $R=1$，倾斜后衰减。

### `flat_orientation`（默认权重 −5.0；curriculum −2.0）
$$
R = \|\mathbf{g}_b^{xy}\|^2
$$
mjlab 内置 `flat_orientation_l2`，配负权重当罚项。

### `base_height_penalty`（默认权重 −10.0）
$$
R = \max(0,\ z_{\min} - z_b),\quad z_{\min} = 0.9\,\text{m}
$$
低于阈值时按差值线性罚。

### `posture`（默认权重 +5.0；curriculum +1.0）
$$
R = \exp\!\left(-\sum_{j=1}^{24}\frac{(q_j - q_j^{\text{default}})^2}{2\sigma_j^2}\right),\quad \sigma_j = 0.5
$$

### `torso_ang_vel`（默认权重 −1.0）
$$
R = \omega_x^2 + \omega_y^2
$$

### `lin_vel_xy`（默认权重 −1.0）
$$
R = v_x^2 + v_y^2
$$

### `forward_velocity`（默认权重 +10.0；curriculum 0.0，关闭）
$$
R = \mathrm{clip}\!\left(-\mathbf{v}_b \cdot \mathbf{n}_w^{\text{outward}},\ 0,\ 1\right)
$$
朝墙方向速度，clip 到 [0, 1]。

---

## 平滑 / 代价

| 项 | 公式 | 默认权重 |
|---|---|---:|
| `action_rate` | $R = \|\mathbf{a}_t - \mathbf{a}_{t-1}\|^2$ | −0.1 |
| `joint_torques` | $R = \sum_j \tau_j^2$ | −1e-6 |
| `dof_pos_limits` | $R = \sum_j \max(0,\ q_j - q_j^{\max}) + \max(0,\ q_j^{\min} - q_j)$ | −5.0 |

---

## 步态类（Phase 1+，curriculum 在 `P0_END = 5000` iter 之后打开）

参考 Um 等人 2025（arXiv:2510.20174）Table I。两手 trot：左手在 $p \in [0, 0.5)$
摆动，右手相位偏 0.5。

下面五项默认权重都是 `0.0`，curriculum 切到目标值。

### `gait_match`（curriculum +8.0）
软正弦版（与论文硬 ±1 阶梯不同，允许 overlap 自然出现）：
$$
s_h(p) = -\sin(2\pi p_h),\quad
g_h = s_h \cdot (2 c_h - 1),\quad
R = \tfrac{1}{2}(g_L + g_R) \in [-1, 1]
$$
- $p_h$ 中段（摆动峰 $p=0.25$ 或支撑峰 $p=0.75$）惩罚峰值 ±1；
- $p_h = 0, 0.5$ 边界 $s_h \approx 0$，策略可自由选择是否提前抓住下一手。

### `swing_hand_height`（curriculum +10.0）
$$
\Delta z_h = z_h - z_b - 0.5,\qquad
R = 0.5 \cdot \exp\!\left(-\sum_{h:\text{swing}} \Delta z_h^2\right)
$$
摆动手相对 base 抬高目标 0.5 m，高斯 shaping，仅 swing 手贡献。

### `swing_hand_clearance`（curriculum −50.0，惩罚项）
$$
\text{deficit}_h = \max(0,\ 0.5 - (z_h - z_b))
$$
$$
R = \sum_{h \in \{L,R\}} (1 - c_h) \cdot \text{deficit}_h^2 \cdot \sqrt{\max(|v_{z,h}|, 10^{-3})}
$$
未脱开（$c_h=0$）且抬不起来还在墙上"蹭"的手（$|v_z|$ 越大越罚）。

### `stance_hand_anchor`（curriculum −2.0，惩罚项）
$$
R = \sum_{h \in \{L,R\}} c_h \cdot \|\mathbf{v}_h\|^2
$$
支撑手（$c_h=1$）速度越大越罚——惩罚滑动。

### `climb_vz`（curriculum +20.0）
$$
R = \mathrm{clip}(v_{z,\text{base}},\ 0,\ 0.5)
$$
真正的爬升目标，clip 到 0.5 m/s 防止策略用突发速度刷分。

---

## 吸附状态信号

`gait_match`、`swing_hand_clearance`、`stance_hand_anchor` 都通过
`_get_engaged()`（`rewards.py:280`）读取 microspine 控制器的 `engaged`
（每手 bool）。如果 microspine event 没挂，函数会静默返回 0。

## 墙几何辅助

`_SphereGeomCache`（`rewards.py:101`）懒加载缓存：
- 16 个球 geom 的 ID 和半径（每手 8 个 × 2）
- 墙的半厚（来自 `wall/wall_geom`）
- 每步从 wall mocap 拿墙位姿和内法线（curriculum 改变墙倾角时自动跟随）

球面到墙面的有符号距离：
$$
d_s = (\mathbf{p}_s^{\text{center}} - \mathbf{p}_{\text{wall}}) \cdot \mathbf{n}_w^{\text{outward}}
      - \tfrac{1}{2}\text{wall\_thickness} - r_s
$$
其中 $r_s$ 是球半径。正=有缝，零=触墙，负=穿透。
