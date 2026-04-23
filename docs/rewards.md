# 奖励（Rewards）

配置：`src/spider_rl_mjlab/tasks/climb_wall/climb_wall_env_cfg.py`。
自定义 reward 函数：`src/spider_rl_mjlab/tasks/climb_wall/mdp/rewards.py`。

## 整体设计

**任务奖励（task-specific）** 管爬墙动作；**regulation** 部分从 IsaacLab
G1 双足行走模板（`g1_minimal_env_cfg.py`）整体迁移过来，权重、joint-set
scope 按照爬墙场景做了最小必要的改动：

- **G1 的 `feet_air_time_positive_biped` → `hand_grip_time`**（两手版，strict ∑e == 1）
- **G1 的 `feet_slide` → `hand_slide`**（engaged-gated）
- **G1 的 `track_lin_vel_xy_exp` → `track_vz`**（非对称高斯）
- **G1 的 `joint_deviation_hip` / `joint_deviation_torso`** 原样搬过来
- **`joint_deviation_arms` 故意不迁**（手臂是爬墙主动执行器，必须大幅偏离 keyframe）
- **`joint_deviation_fingers` 不适用**（PHYBot Big V2 没有 fingers）
- `dof_pos_limits` / `dof_acc_l2` / `dof_torques_l2` / `action_rate_l2` 权重和 joint scope 都对齐 G1
- `termination_penalty` 权重 −200 与 G1 一致

## 当前启用状态总览

| 分组 | 项 | 权重 | 来源 |
|---|---|---:|---|
| **任务** | `hand_orientation` | +15.0 | 原创（手朝墙） |
| **任务·gait** | `hand_grip_time` | +30.0 | 迁自 G1 `feet_air_time_positive_biped`（strict ∑e == 1） |
| **任务·跟踪** | `track_vz` | +40.0 | 迁自 G1 `track_lin_vel_xy_exp`（非对称高斯，仅 z 轴） |
| **任务·姿态** | `flat_orientation` / `torso_orientation` | +10 / +10 | 自定义 Gaussian，roll-only σ=0.3 |
| **Regulation** | `termination_penalty` | **−200.0** | G1 `termination_penalty` 原样 |
| **Regulation** | `hand_slide` | **−0.1** | 迁自 G1 `feet_slide`（engaged-gated） |
| **Regulation** | `dof_pos_limits`（踝） | **−1.0** | G1 `dof_pos_limits`（踝）原样 |
| **Regulation** | `joint_deviation_hip`（髋 yaw + roll） | **−0.1** | G1 原样 |
| **Regulation** | `joint_deviation_torso`（腰 yaw + roll） | **−0.1** | G1 原样 |
| **Regulation** | `action_rate_l2` | **−0.005** | G1 原样 |
| **Regulation** | `dof_acc_l2`（髋 + 膝） | **−1.25e-7** | G1 原样 |
| **Regulation** | `dof_torques_l2`（髋 + 膝 + 踝） | **−1.5e-7** | G1 原样 |

**已移除**：
- `alive`（G1 没有 alive bonus）
- `leg_posture`（被 `joint_deviation_hip/torso` + `dof_acc_l2` / `dof_torques_l2` 三件套取代，语义更接近 G1）
- `excessive_tilt` / `torso_roll_excessive` / `static_hang` / `both_hands_off` terminations（G1 只用 `base_contact` 一条硬 termination，其余交给 reward 软塑形）
- `swing_hand_height`、`stance_phase_velocity`、`climb_height`、`track_lin_vel` 等更早期的 clock-based / legacy 项

## 符号约定

- $E$ 环境数；$h \in \{L, R\}$ 左/右手。
- $e_h \in \{0, 1\}$ 第 $h$ 手的 `microspine.engaged`。
- $t_{\text{grip}, h}$ / $t_{\text{swing}, h}$ 每手的抓/摆持续时间。
- $\tau$ 时间封顶；$v_z^b$ base 世界 z 速度；$v_z^{\text{cmd}} = 0.08$。
- $\mathbf{n}_w$ 墙外法向；$\mathbf{v}_{h,\tan}$ 第 $h$ 手墙平面切向速度。
- $q_j, q_j^*, \dot{q}_j, \ddot{q}_j, \tau_j$ 第 $j$ 个关节的位置、默认位置（keyframe）、速度、加速度、力矩。

---

## 任务奖励

### `hand_grip_time`（weight +30.0）

G1 `feet_air_time_positive_biped` 的两手爬墙版本。

$$
t_{\text{mode},h} = e_h \cdot t_{\text{grip},h} + (1 - e_h) \cdot t_{\text{swing},h}
$$

$$
\mathbb{1}_{\text{safe}} = \mathbb{1}\!\left[\sum_h e_h = 1\right],\qquad
\mathbb{1}_{\text{move}} = \mathbb{1}\!\left[v_z^{\text{cmd}} > \varepsilon\right]
$$

$$
R_{\text{grip}} = \mathbb{1}_{\text{move}} \cdot \min\!\left(\tau,\ \mathbb{1}_{\text{safe}} \cdot \min_h t_{\text{mode},h}\right)
$$

当前参数：$\tau = 0.5$ s，$v_z^{\text{cmd}} = 0.08$ m/s，$\varepsilon = 0$。

| 状态 | $e_L$ | $e_R$ | $\sum e$ | $R_{\text{grip}}$ |
|---|:---:|:---:|:---:|---|
| 双手都松 | 0 | 0 | 0 | 0 |
| 左抓右摆 / 左摆右抓 | 1/0 | 0/1 | 1 | $\min(\tau, \min_h t_{\text{mode},h})$ |
| 双手都抓 | 1 | 1 | 2 | 0 |

### `track_vz`（weight +40.0）

G1 `track_lin_vel_xy_exp` 的 vz-only 非对称版本：

$$
R_{vz} = \exp\!\left(-\frac{(\max(v_z^{\text{cmd}} - v_z^b,\ 0))^2}{\sigma^2}\right),\ \sigma = 0.05,\ v_z^{\text{cmd}} = 0.08
$$

- $v_z^b \ge 0.08$：$R = 1$（爬得更快不扣分）
- $v_z^b < 0.08$：Gaussian 衰减

### `hand_orientation`（weight +15.0）

手的 local $-x$ 面方向和墙内法向对齐：

$$
R = \sum_{h} \frac{\mathbf{f}_h \cdot (-\mathbf{n}_w) + 1}{2} \in [0, 2]
$$

两相位都激活，防止 swing 中手反转。

### `flat_orientation` / `torso_orientation`（weight +10 / +10）

Gaussian-kernel roll-only（$\sigma = 0.3$ rad）：

$$
R = \exp\!\left(-\frac{s_p \cdot g_x^2 + s_r \cdot g_y^2}{\sigma^2}\right)
$$

当前 `pitch_scale = 0, roll_scale = 1`——只惩罚绕 body-x 侧翻，pitch
留给 regulation 层软管。`flat_orientation` 看 base，`torso_orientation`
看 `waist_roll` body（封住 waist-bend exploit）。

| roll 角 | reward（σ=0.3） |
|---|---:|
| 5° | 0.95 |
| 10° | 0.80 |
| 20° | 0.42 |
| 30° | 0.16 |

---

## Regulation（G1 walking 迁移）

### `termination_penalty`（weight −200.0）

$$
R = -\mathbb{1}[\text{terminated}]
$$

G1 原样。`mdp.is_terminated` 不含 `time_out`，只在真正失败的那一步发火。
配合 `scale_by_dt`，每次失败瞬时贡献 $-200 \times 0.02 = -4$，外加损失
未来所有 task reward 的机会成本。

### `hand_slide`（weight −0.1）

G1 `feet_slide` 的爬墙版：

$$
R_{\text{slide}} = \sum_h e_h \cdot \|\mathbf{v}_{h,\tan}\|
$$

**差异**：G1 用"过去 K 帧接触力幅 > 1 N"判断 contact；我们直接读
`microspine.engaged` latch flag（爬墙 adhesion 的对应语义）。$\mathbf{v}_{h,\tan}$
是手在墙平面内的切向速度（法向分量被减掉），对应 G1 中的水平速度。

### `dof_pos_limits`（weight −1.0）

$$
R = \sum_{j \in \text{ankles}} \bigl(\max(0,\, q_j^{\text{low}} - q_j) + \max(0,\, q_j - q_j^{\text{high}})\bigr)
$$

G1 原样——只惩罚 ankle 超软限位。踝是"容易顶限位的关节"，其他关节
（髋/膝）有足够活动余量，不专门罚。

### `joint_deviation_hip`（weight −0.1）

$$
R = \sum_{j \in \{hip\_yaw_{L,R},\, hip\_roll_{L,R}\}} |q_j - q_j^*|
$$

G1 原样——L1 偏差惩罚。约束腿不向外张开（hip_yaw）或扭转（hip_roll）。
hip_pitch 不在内（爬墙时膝盖靠墙有屈伸余地）。

### `joint_deviation_torso`（weight −0.1）

$$
R = \sum_{j \in \{waist\_yaw,\, waist\_roll\}} |q_j - q_j^*|
$$

G1 用了单个 `torso_joint`；PHYBot Big V2 对应的是 `waist_yaw + waist_roll`
两个 DoF，一起约束。`torso_orientation_gauss` 只看 roll 轴的世界几何，
yaw 靠这一项钉死。

### **故意不迁：`joint_deviation_arms` / `joint_deviation_fingers`**

G1 双足行走会用 `joint_deviation_l1` 罚肩、肘（arms）和手指（fingers）不能
偏离默认姿态。**爬墙任务完全不迁这两项**：
- 肩、肘是爬墙的主动执行器——偏离 keyframe 才能抓高、蹬腿推进，加约束
  会把任务本身禁掉
- PHYBot Big V2 没有手指关节

### `action_rate_l2`（weight −0.005）

$$
R = \|\mathbf{a}_t - \mathbf{a}_{t-1}\|^2
$$

G1 原样。所有 action 维度参与，抑制高频颤振。

### `dof_acc_l2` / `dof_torques_l2`（weight −1.25e-7 / −1.5e-7）

$$
R_{\text{acc}} = \sum_{j \in \text{hip} \cup \text{knee}} \ddot{q}_j^{\,2}, \qquad
R_{\tau} = \sum_{j \in \text{hip} \cup \text{knee} \cup \text{ankle}} \tau_j^{\,2}
$$

G1 权重原样。只算腿段——这些 joint 是被动的，不应该有大加速度/力矩。
上肢（肩、肘）有大力矩是合理的，所以**不纳入**。权重非常小（~1e-7 级）
是 L2-squared 典型量级，配合 `scale_by_dt` 每步贡献 ~1e-5，主要作为
"避免疯狂抽搐"的软 regulation。

---

## Termination

对应见 [curriculum.md](./curriculum.md#termination-兜底)。G1 walking 只用
`time_out + base_contact` 两条硬终止；我们的爬墙版本是
`time_out + base_height + bad_state`——`base_height` 是"robot fell off
wall"（climb 里 base_contact 的语义对偶），`bad_state` 是纯数值安全网。

---

## 墙几何辅助

`_SphereGeomCache`（`rewards.py`）懒加载缓存 16 球 geom ID、半径、墙半厚；
每步从 wall mocap 拿墙位姿和 outward normal（curriculum 改墙倾角自动跟随）。
被 `hand_orientation` 和 `hand_slide` 复用。
