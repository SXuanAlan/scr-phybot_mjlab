# Climb-Wall Reward 与 Curriculum 设计

`Mjlab-Climb-Wall-Phybot-Big-V2` 任务的奖励和课程设计说明。**双手交替式垂直爬墙**。

---

## 1. 总体思路

机器人出生就已经贴近墙面（位置 `(-0.7, 0, 2.0)`，手臂以 `shoulder_pitch=-2.79` 已经朝墙伸出），配合初始 30° 倾斜重力（水平分量把机器人推向墙），第一帧起就在接触墙面附近。**不需要学站立和走路**，直接从"抓住墙"开始。

训练按**一条时间轴**分 4 个 phase：

| Phase | iter 区间 | 任务 | 重力 |
|---|---|---|---|
| 0 | `0 → 5000`      | **抓住墙**（单手/双手接触） | 30° |
| 1 | `5000 → 15000`  | **学步态**（交替 swing/stance） | 30° |
| 2 | `15000 → 45000` | 重力从 30° 渐进到 90° | 30°→90° |
| 3 | `45000 → ∞`     | 稳定的垂直爬墙步态 | 90° |

Phase 0 和 Phase 1 共享"容易"的 30° 重力（机器人被水平推向墙），让机器人先把基本技能学会。Phase 2 开始才逐步加难度。

---

## 2. Phase Clock（步态时钟）

### 2.1 定义

每个 env 维护一个相位 φ ∈ [0, 1)：
```
φ(t) = (episode_length_buf · step_dt / GAIT_PERIOD_S) mod 1
```
其中 `GAIT_PERIOD_S = 2.0` 秒。每次 episode reset 时 φ 归零。

### 2.2 两手错相（trot）

```
φ_left  = φ
φ_right = (φ + 0.5) mod 1
```
两手相差半个周期，保证**任何时刻只有一只手在 swing**。

### 2.3 Swing/Stance 期望（正弦软信号）

核心目标由**正弦期望值** `s_i(φ) = −sin(2π·φ_i)` 定义：
```
φ_i = 0.00 (边界)   →  s_i =  0   # 自由
φ_i = 0.25 (swing)  →  s_i = −1   # 强烈期望抬起
φ_i = 0.50 (边界)   →  s_i =  0   # 自由
φ_i = 0.75 (stance) →  s_i = +1   # 强烈期望抓住
```
硬 0/1 的 swing 指示 `f_i = (φ_i < 0.5)` 还在 `swing_hand_height` 等 reward 里作为 mask 用。

**overlap 在边界 (φ≈0, 0.5) 附近压力接近 0，policy 自由选择早松/晚松——overlap 由 policy 自学。**

时间轴示意（T = 2s）：
```
time(s)  0.0   0.5   1.0   1.5   2.0
left     ░░░░░░░░░░░░░░████████████████
right    ████████████████░░░░░░░░░░░░░░

░ = swing   █ = stance
```

### 2.4 观测

policy 看到 4 维 gait_clock：
```
[sin(2πφ_L), cos(2πφ_L), sin(2πφ_R), cos(2πφ_R)]
```
sin/cos 组合避开 0↔1 跳变。

---

## 3. Reward 清单

记号约定：
- `f_L, f_R` ∈ {0, 1}：swing 指示（见 2.3）
- `c_L, c_R` ∈ {0, 1}：`MicrospineController.engaged`（左/右手是否吸附）

### 3.1 手臂接触/姿态类（Phase 0 主力）

| 名称 | 公式概要 | 含义 |
|---|---|---|
| `hands_toward_wall` | `−mean(\|d_sphere\|)` | 手上 16 个 sphere 到墙面的平均距离，距离越小越好 |
| `hand_orientation` | `(dot(−x_hand, n_wall) + 1) / 2` | 手掌法向对齐墙 inward normal |
| `hands_on_wall` | `fraction(\|d_sphere\| < 2cm)` | 16 个 sphere 中接触墙的比例 |
| `both_hands_bonus` | `ratio_L · ratio_R` | 两手都在接触时给的乘性奖励 |
| `com_near_wall` | `exp(−\|base_to_wall_plane\| / 0.5)` | 身体靠近墙面 |

### 3.2 Gait 类（Phase 1+ 打开）

#### **`gait_match` (R_g)** — 正弦软节律匹配

**期望值 s_i** 用正弦函数平滑地指示"该 stance 还是该 swing"：
```
s_i(φ) = −sin(2π · φ_i)        ∈ [−1, +1]
```
- `φ_i = 0.25`（swing 峰值）：`s_i = −1`，强烈期望 **不接触**
- `φ_i = 0.75`（stance 峰值）：`s_i = +1`，强烈期望 **接触**
- `φ_i = 0` 或 `0.5`（切换边界）：`s_i ≈ 0`，**接触/不接触都不罚** ← 自由区

奖励：
```
g_i = s_i · (2·c_i − 1)        # c_i ∈ {0,1} 映射到 {−1,+1}
R_g = 0.5 · (g_L + g_R)         ∈ [−1, 1]
```

**关键特性：overlap 可学**
- swing / stance 切换边界附近 `|s| ≈ 0` → policy 可以**提前抓下一只手**或**延迟松开当前手**而几乎不扣分
- 切换中心附近 `|s| ≈ 1` → 强约束
- 配合 `climb_vz` 和 `swing_hand_height`（它们持续要求上升 + 抬手），policy 不会一直双手抓着，会在有利时机自然松开
- 最终涌现的 overlap 时长由 policy 自己在"失手风险 vs 上升速度"之间权衡

#### **`swing_hand_height` (R_fh)** — 抬手高斯 shaping

```
Δz_i = hand_z_i − base_z
err  = Σ_i f_i · (Δz_i − lift_target)²           lift_target = 0.5 m
R_fh = 0.5 · exp(−err)                            ∈ (0, 0.5]
```
只对**正在 swing 的手**计误差；目标是 swing 手比 base 高 50cm。高斯形状处处可导，是**教抬手的主要连续信号**。

#### **`swing_hand_clearance` (R_fc)** — "别蹭墙"重罚

```
deficit_i = max(0, lift_target − Δz_i)
R_fc = Σ_i (1 − c_i) · deficit_i² · √|v_{z,i}|
```
只在**手未接触**且**没抬够高**且**正在移动**时激活。系数 `−50`。惩罚"手在墙上横着蹭"这种退化行为——swing 就该真抬起来，不是贴着墙挪。

#### **`stance_hand_anchor` (R_fs)** — stance 手防滑

```
R_fs = Σ_i c_i · ‖v_i‖²
```
系数 `−2`。支撑手应该锚定不动；任何残余速度都罚。

#### **`climb_vz`** — 真正的任务目标

```
R_climb = clip(base_lin_vel_z, 0, 0.5)    m/s
```
基座向上速度（截断到 0.5 防"爆冲"）。**上升是最终目标**，但不能单独用——会退化成抽搐、蠕动之类的丑陋方案，所以必须配合 gait 结构。

### 3.3 姿态/稳定类（全程常量权重）

| 名称 | 权重 | 作用 |
|---|---|---|
| `torso_orientation` | +5 | 身体保持直立（朝世界 +z） |
| `flat_orientation` | −2 | 惩罚非竖直倾角 |
| `base_height_penalty` | −10 | 基座低于 0.9m 罚 |
| `posture` | +1 | 轻度引导姿态接近默认 |
| `torso_ang_vel` | −1 | 防抖 |

### 3.4 基础惩罚（全程不变）

| 名称 | 权重 |
|---|---|
| `action_rate` | −0.1 |
| `joint_torques` | −1e-6 |
| `dof_pos_limits` | −5 |

---

## 4. 各 Phase 权重一览

| Reward | Phase 0 `[0, 5k)` | Phase 1 `[5k, 15k)` | Phase 2+ `[15k, ∞)` |
|---|---|---|---|
| `hands_toward_wall` | **+15** | +15 → +5 (at 15k) | +5 |
| `hand_orientation`  | **+15** | +15 → +5 (at 15k) | +5 |
| `hands_on_wall`     | **+20** | +20 | +20 |
| `both_hands_bonus`  | **+30** | +30 → +15 (at 5k) → +5 (at 15k) | +5 |
| `com_near_wall`     | +5 | +5 | +5 |
| `gait_match`        | 0 | **+8** | +8 |
| `swing_hand_height` | 0 | **+10** | +10 |
| `swing_hand_clearance` | 0 | **−50** | −50 |
| `stance_hand_anchor` | 0 | **−2** | −2 |
| `climb_vz`          | 0 | **+20** | +20 |
| `torso_orientation` | +5 | +5 | +5 |
| `flat_orientation`  | −2 | −2 | −2 |
| `base_height_penalty` | −10 | −10 | −10 |
| `posture`           | +1 | +1 | +1 |
| `torso_ang_vel`     | −1 | −1 | −1 |

权重设计说明：
- **Phase 0**：`both_hands_bonus = 30` 鼓励把两只手都抓到墙上；gait 奖励全关。
- **Phase 1**：gait 五件套一次打开；`both_hands_bonus` 从 30 降到 15（再 15k 后降到 5），**让位给交替步态**（否则"永远两只手抓着"跟 swing 冲突）。
- **Phase 2+**：权重基本不变，只是重力开始倾斜到垂直；已经学会的节律要在更难的物理下保持。

---

## 5. Gravity Curriculum

重力方向按"等效墙角度" α 随 iter 变化：
```
gravity = |g| · (cos α, 0, −sin α)
```
- `α = 30°`（Phase 0, 1）→ `g = (8.5, 0, −4.9)`：主要压向墙
- `α = 90°`（Phase 3 后）→ `g = (0, 0, −9.81)`：标准垂直

```
"angle_stages": [
  {"iter": 0,     "angle_deg": 30.0},
  {"iter": 15000, "angle_deg": 30.0},   # Phase 0+1 一直保持 30°
  {"iter": 45000, "angle_deg": 90.0},   # Phase 2 线性升到 90°
]
```

重力修改通过 `wp.to_torch(env.sim.wp_model.opt.gravity)` 的 in-place 修改，不重新分配 warp 数组（保持 CUDA graph 有效）。每个 env 的 `gravity_vec_w` 故意**不**跟着变——这样 `torso_orientation` 等奖励仍以"世界竖直方向"为参考，对应"沿墙向上"的正确语义。

---

## 6. 关键实现点

### 6.1 获取 engaged 状态
```python
term = env.event_manager.get_term_cfg("microspine_adhesion")
engaged = term.func.engaged   # (E, 2) bool, 列序 = (left_hand, right_hand)
```
MicrospineController 是 step 模式的 event term，每步在 reward 计算前刷新。

### 6.2 没有地板
`config/phybot_big_v2/env_cfgs.py` 的 `_spec_fn` 里把 `terrain/terrain` geom 的 `contype = conaffinity = 0`——视觉上还有参考平面，但碰撞关闭。这样机器人没抓住墙就真的会掉下去，提供学习压力。

### 6.3 Termination
- `time_out`：episode 超时
- `base_height < 0.3m`：掉下去
- `bad_state`：关节/身体速度过大

---

## 6.4 Reward 数学定义（全量）

本节把**所有 reward 的精确数学公式**列齐。实现在 `src/spider_rl_mjlab/tasks/climb_wall/mdp/rewards.py`。所有 tensor 形状都是 `(E,)`，E = num_envs。

### 通用符号

| 符号 | 含义 |
|---|---|
| `i ∈ {L, R}` | 左手 / 右手下标 |
| `x_base, v_base ∈ ℝ³` | 机器人 base link 的世界位姿 / 速度 |
| `x_i, v_i ∈ ℝ³` | 第 i 只手 body 的世界位姿 / 速度 |
| `q_i ∈ ℝ⁴` | 第 i 只手 body 的世界四元数 (wxyz) |
| `x_wall, q_wall` | 墙 body 的世界位姿 / 四元数 |
| `n_w ∈ ℝ³` | 墙的外法向（世界系）= `R(q_wall) · (−1, 0, 0)` |
| `h_wall` | 墙厚度半值（`wall_half_x = 0.5 m`） |
| `{g_{i,k}}_{k=1}^{16}` | 第 i 只手上 8 个 sphere 的世界坐标（共 16 个，左右各 8） |
| `r_k` | 第 k 个 sphere 的半径 |
| `d_k` | 第 k 个 sphere 表面到墙面的有符号距离，正 = 没到墙，负 = 穿透 |
| `c_i ∈ {0, 1}` | 第 i 只手的 engaged 状态（MicrospineController.engaged） |
| `φ ∈ [0, 1)` | 全局相位 = `(episode_length_buf · step_dt / 2.0) mod 1` |
| `φ_L, φ_R` | `φ_L = φ`，`φ_R = (φ + 0.5) mod 1` |
| `f_i` | swing mask：`f_i = 1 if φ_i < 0.5 else 0` |

**Sphere 到墙面的有符号距离：**
$$
d_k = (g_{i,k} - x_\text{wall}) \cdot n_w - h_\text{wall} - r_k
$$
`d_k > 0`：sphere 尚未接触墙。`d_k < 0`：穿透。`d_k = 0`：刚好触墙。

---

### 接触/姿态类

**`hands_toward_wall`** — 接近墙的连续 shaping：
$$
R_\text{toward} = -\min\!\Big(\,\overline{|d_k|_{k=1\ldots 16}}\,,\ 0.5\Big)
$$
16 个 sphere 到墙面距离绝对值取均值，再 clamp 到 0.5，取负。越接近 0 越大。

**`hand_orientation`** — 手掌法向对齐：
$$
R_\text{orient} = \frac{1}{2|H|}\sum_{i \in \{L,R\}}\bigl(\hat{n}_i \cdot (-n_w) + 1\bigr)
$$
其中 `\hat{n}_i = R(q_i) · (−1, 0, 0)` 是第 i 只手 local −x 的世界方向（即手掌触面法向），与墙内法向 `−n_w` 对齐时值为 1。输出 `∈ [0, 1]`。

**`hands_on_wall`** — 接触比例：
$$
R_\text{on} = \frac{1}{16}\sum_{k=1}^{16}\mathbb{1}\!\bigl(|d_k| < 0.02\bigr)
$$
2 cm 以内算接触。

**`both_hands_bonus`** — 双手同时接触乘积：
$$
R_\text{both} = \underbrace{\frac{1}{8}\sum_{k=1}^{8}\mathbb{1}(|d_k|<0.02)}_{\text{left ratio}} \cdot \underbrace{\frac{1}{8}\sum_{k=9}^{16}\mathbb{1}(|d_k|<0.02)}_{\text{right ratio}}
$$
两手接触比例相乘——任何一只手不接触都是 0。

**`com_near_wall`** — 身体靠墙：
$$
R_\text{com} = \exp\!\left(-\frac{|s_\text{base}|}{0.5}\right),\quad s_\text{base} = (x_\text{base} - x_\text{wall}) \cdot n_w - h_\text{wall}
$$
`s_\text{base}` 是 base 到墙面的有符号距离。0.5 m 是衰减尺度。

---

### Gait 类

**`gait_match`（R_g）** — 软正弦节律匹配：
$$
s_i(\varphi_i) = -\sin(2\pi\varphi_i),\quad
g_i = s_i \cdot (2c_i - 1),\quad
R_g = \tfrac{1}{2}(g_L + g_R)
$$
- `s_i ∈ [-1, +1]`，swing 峰 `-1`（想 ¬contact），stance 峰 `+1`（想 contact）
- 在 `φ = 0, 0.5` 边界 `s_i = 0` → overlap 自由区
- 输出 `R_g ∈ [-1, +1]`

**`swing_hand_height`（R_fh）** — swing 手高度高斯 shaping：
$$
\Delta z_i = (x_i)_z - (x_\text{base})_z,\ \ 
\text{err} = \sum_{i\in\{L,R\}} f_i \cdot (\Delta z_i - z^*)^2,\ \ 
R_{fh} = 0.5\exp(-\text{err})
$$
目标 `z^* = 0.5 m`（swing 手比 base 高 50 cm）。swing mask `f_i` 把 stance 侧过滤掉。输出 `∈ (0, 0.5]`。

**`swing_hand_clearance`（R_fc）** — "别蹭墙"惩罚：
$$
\delta_i = \max(0,\ z^* - \Delta z_i),\quad
R_{fc} = \sum_{i\in\{L,R\}} (1 - c_i) \cdot \delta_i^2 \cdot \sqrt{\max(|v_{i,z}|,\,10^{-3})}
$$
只在"没接触 (`1-c_i`) + 没抬够高 (`δ_i > 0`) + 在移动"同时满足时激活。`√|v_z|` 让它随速度 sub-linear 增大。输出正值，env_cfg 里用 **−50** 权重。

**`stance_hand_anchor`（R_fs）** — 防滑：
$$
R_{fs} = \sum_{i\in\{L,R\}} c_i \cdot \|v_i\|^2
$$
只对接触侧 (`c_i = 1`) 算速度平方。用 **−2** 权重。

**`climb_vz`** — 向上速度：
$$
R_\text{climb} = \text{clip}\!\bigl((v_\text{base})_z,\ 0,\ 0.5\bigr)
$$
截断到 `[0, 0.5] m/s`——只奖励上升，且防止爆冲。

---

### 姿态稳定类

**`torso_orientation`** — 用 `projected_gravity_b = R(q_\text{base})^{-1} \cdot (0,0,-1)`：
$$
R_\text{torso} = \exp\!\bigl(-\|g_{b,x}, g_{b,y}\|^2\bigr)
$$
body-frame 重力只有 z 分量时 `= 1`（直立），越歪越小。

**`flat_orientation`（mjlab 内建）**：
$$
R_\text{flat} = g_{b,x}^2 + g_{b,y}^2
$$
直接用 body-frame 重力的 xy 平方和，上式 `-log` 版。权重 **−2**。

**`base_height_penalty`（mjlab 内建）**：
$$
R_\text{height} = \max(0,\ 0.9 - (x_\text{base})_z)
$$
base 低于 0.9 m 线性增长。权重 **−10**。

**`posture`（mjlab 内建）**：
$$
R_\text{posture} = \exp\!\left(-\sum_j \frac{(q_j - q_j^*)^2}{\sigma_j^2}\right)
$$
`q_j^*` 是每个关节的默认姿态（`CLIMB_WALL_KEYFRAME`），`σ_j = 0.5` rad。

**`torso_ang_vel`**：
$$
R_{\omega} = \omega_x^2 + \omega_y^2
$$
base 世界角速度 xy 平方和，惩罚晃动。权重 **−1**。

---

### 基础惩罚

**`action_rate`（mjlab）**：
$$
R_\text{rate} = \|a_t - a_{t-1}\|^2
$$

**`joint_torques`（mjlab）**：
$$
R_{\tau} = \sum_j \tau_j^2
$$

**`dof_pos_limits`（mjlab）**：
$$
R_\text{lim} = \sum_j \max(0,\ q_j^\text{low} - q_j) + \max(0,\ q_j - q_j^\text{high})
$$
超过软限位线性惩罚。权重 **−5**。

---

### 总 reward

所有项以 weighted sum 合成（mjlab 默认加法聚合）：
$$
R_\text{total}(t) = \sum_\text{term} w_\text{term}(\text{iter}) \cdot R_\text{term}(s_t, a_t)
$$
权重 `w_\text{term}(\text{iter})` 由 curriculum（见 §4）按训练 iter 分段线性/阶跃更新。每个 term 的当前权重打印在训练日志 `Curriculum/*` 下。

---

## 7. 调参对照表

| 症状 | 调整 |
|---|---|
| Phase 0 一直不接触墙 | 提高 `hands_on_wall`/`both_hands_bonus` 权重；或延长 Phase 0 时长 |
| 两手抓紧不松（不爬） | 提前 `both_hands_bonus` 降权点；或增大 `climb_vz` |
| 抬不起手 | 提高 `swing_hand_height`（+10 → +20） |
| 手在墙上蹭 | 加大 `swing_hand_clearance` 绝对值（−50 → −100） |
| stance 手滑 | 加大 `stance_hand_anchor`（−2 → −10） |
| 节奏僵硬 | 调整 `GAIT_PERIOD_S`（`mdp/observations.py`，当前 2.0s） |
| Overlap 太多（两手老是一起抓） | 加大 `gait_match` 权重（+8 → +15），或把 `s_i` 公式的 sin 换成更陡峭的幂函数 `sign(sin)·|sin|^0.3` |
| Overlap 太少（容易失手掉下去） | 降 `gait_match` 权重，或把 sin 替换成 `sin × 0.7`（softer） |
| 爬得慢 | 加大 `climb_vz`（+20 → +40）或放宽 clip（0.5 → 1.0） |
| Phase 2 重力升角度时崩 | 延后 `P2_END`（45k → 80k），给更慢的重力坡度 |

---

## 8. 可选扩展

1. **Policy 自控步频**：把 `dφ/dt` 做成 action 输出，让 policy 决定爬墙速度。
2. **4-EE 步态**：手 + 脚都算 swing/stance，相位差 0.25 × 4，更鲁棒（需要给脚也加 microspine）。
3. **乘性 reward 聚合**：改成 `R = positive · exp(−0.2·penalty)`，对"做错任何一项"惩罚更重，参考 Um et al. 2025。
4. **Adhesion 随机失败**：Phase 3 加 `Prob_attach` 从 1.0 降到 0.85，学 adhesion 不可靠时的恢复。
