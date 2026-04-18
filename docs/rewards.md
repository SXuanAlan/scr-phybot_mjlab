# 奖励（Rewards）

配置：`src/spider_rl_mjlab/tasks/climb_wall/climb_wall_env_cfg.py:72`。
自定义 reward 函数：`src/spider_rl_mjlab/tasks/climb_wall/mdp/rewards.py`。

## 当前启用状态总览

| 分组 | 项 | 角色 |
|---|---|---|
| 贴墙 shaping | `hands_toward_wall`, `hand_orientation`, `hands_on_wall`, `posture` | 引导贴墙，权重适中避免静态白嫖 |
| **速度命令跟踪** | `track_vx`, `track_vy`, `track_vz` | base world velocity 高斯核跟踪 |
| **Swing sine-arc shaping** | `swing_hand_height`, `swing_hand_reach_up` | paper arXiv:2204.04157：swing 手 task-space 正弦轨迹（水平+竖直） |
| 反滑动 | `stance_phase_velocity` | 支撑相位手静止 |
| 关闭 | `climb_height` | 保留 fallback，weight=0 |

**当前版本不限制"两手同时脱粘"**——`both_hands_unlatched` termination 已禁用、
`airborne_penalty` reward 也删了。让 policy 纯靠 velocity tracking 自己探索，
看能否自然涌现出 overlap gait。函数都还在 codebase 里做 fallback。

## 符号约定

- $E$ 环境数；$h \in \{L, R\}$ 左/右手；$s \in \{1,\dots,16\}$ 16 个吸附球。
- $p$ 步态相位；$c_h$ 第 $h$ 手的 `engaged`；$C_h^{\text{cmd}}(p)$ 软 gait schedule。
- $\mathbf{n}_w$ 墙 outward normal；$d_s$ 球面到墙面有符号距离。
- $z_b$ base 世界 z；$\mathbf{v}_b$ base 世界系线速度；$\mathbf{v}_{\text{cmd}}$ 命令速度。

---

## 贴墙 shaping

### `hands_toward_wall`（weight +8.0 → +3.0 at P1_END）

$$
R = -\min\!\left(\overline{|d_s|},\ 0.5\right)
$$

16 个球 $|d_s|$ 均值并 clip。P1_END 后衰减到 3。

### `hand_orientation`（weight +15.0，stance 相位门控）

$$
R = \sum_{h} C_h^{\text{cmd}}(p) \cdot \frac{\mathbf{f}_h \cdot (-\mathbf{n}_w) + 1}{2} \in [0, 2]
$$

$\mathbf{f}_h$ 是第 $h$ 手 local $-x$ 面在世界系下的方向。只 stance 相位奖励
对齐，swing 相位不罚。

### `hands_on_wall`（weight +8.0）

16 个球面距墙 < 20 mm 的占比。

### `posture`（weight +3.0）

mjlab 内置 Gaussian kernel，按关节组分别设 std：waist 0.3、hip 0.5、knee/ankle
0.6、shoulder/elbow 1.2。腰/腿严，臂松。

---

## 速度命令跟踪（主驱动）

所有三项共享高斯核：

$$
R_{\text{axis}} = \exp\!\left(-\frac{(v_{\text{cmd}, i} - v_{b, i})^2}{\sigma^2}\right)
$$

$v_b$ 是 base 世界系线速度（`root_link_lin_vel_w`）。命令来自
`velocity_command` 观测（目前固定 $(0, 0, 0.3)$，之后可换 command manager）。

| 项 | weight | $v_{\text{cmd}}$ | $\sigma$ | 作用 |
|---|---:|---:|---:|---|
| `track_vx` | 6 | 0 m/s | 0.5 m/s | 不沿墙面 x 方向漂；sigma 宽避免严惩伸手时的小扰动 |
| `track_vy` | 6 | 0 m/s | 0.5 m/s | 不沿墙面 y 方向漂 |
| `track_vz` | **40** | **+0.3 m/s** | 0.25 m/s | **主目标：稳定向上爬**；sigma 窄强信号 |

权重比：`vz (40) > vx + vy (12) > posture/stance (3)`。vz 单项就比 vx+vy 合计
还高，确保 "爬升" 明确优于 "完全静止"。vx/vy 的 sigma 放宽到 0.5 是关键：
早期训练里用 0.3 sigma 会严惩伸手时的微小 body 漂移（~0.2 m/s），导致
policy 学到 "完全不动 > 爬升"；0.5 sigma 允许 reach 动作造成的暂时扰动不
被惩罚太狠。

**高斯核对称**——robot vz = 0.3 拿满分，vz = 0（静止）和 vz = 0.6（冲刺）都
惩罚。这天然消除"冲刺"激励而不需要额外 clamp。

---

## 反滑动

### `stance_phase_velocity`（weight +3.0）

$$
R = \sum_h C_h^{\text{cmd}}(p) \cdot \exp\!\left(-\frac{\|\mathbf{v}_{h,\tan}\|^2}{\sigma^2}\right),\quad \sigma = 0.15\,\text{m/s}
$$

stance 相位时手切向速度接近 0 → 满奖励。

---

## Swing sine-arc shaping（paper arXiv:2204.04157）

Wu 等人 2022 "Custom Sine Waves Are Enough" 的 task-space 正弦参考。对每只手
的 swing 相位（$p_h \in [0, 0.5)$），给出世界系两条 sine 参考曲线：

### `swing_hand_height`（weight +10.0，水平分量）

$$
d_{\text{ref}, h}(p) = \text{lift\_amp} \cdot \max(0,\ \sin(2\pi p_h)),\quad \text{lift\_amp} = 0.1\,\text{m}
$$

$d_h$ 是第 $h$ 手 8 个球到墙面的均值法向距离。swing 中段（$p_h = 0.25$）
峰值 0.1 m 离墙；stance 阶段 sine 负半波被 clamp 到 0（手应贴墙）。高斯
tracking：$R = \exp(-\sum_h (d_h - d_{\text{ref}})^2 / \sigma^2)$，$\sigma = 0.18$。

### `swing_hand_reach_up`（weight +15.0，竖直分量——**hand-to-hand 差分**）

$$
\Delta z_{\text{ref}}(p) = \text{reach\_peak} \cdot \sin(2\pi p),\quad \text{reach\_peak} = 0.15\,\text{m}
$$

$$
\Delta z = z_L - z_R,\quad R = \exp\!\left(-\frac{(\Delta z - \Delta z_{\text{ref}})^2}{\sigma^2}\right),\quad \sigma = 0.20
$$

**不 clamp**，单条对称正弦波同时编码两手交替：
- $p = 0.25$（左手 swing peak）：$\Delta z_{\text{ref}} = +0.15$ → 左手高 15cm
- $p = 0.75$（右手 swing peak）：$\Delta z_{\text{ref}} = -0.15$ → 右手高 15cm
- $p \in \{0, 0.5\}$（换手瞬间）：$\Delta z_{\text{ref}} = 0$ → 两手平

**为什么不用 hand-vs-root**：初版用 `z_h - z_{\text{root}}` 跟 clamped sine
出过坑——climbing 的手长期在 root 上方（手臂伸向墙），clamped sine
stance ref=0 会要求"把手拉到 root 水平"，和 climbing 物理相反，policy
学成了"拱背把手往下拉"（posture 指标持续衰减）。
差分公式 anatomy-invariant，支撑手 baseline 位置由物理决定而非 reward。

**两项合一起 = 完整的 3D swing 弧**：手先拉离墙（horizontal +）同时抬高
（vertical +），在中段达到"悬臂最远最高"的 reach 姿态，然后弧线回落到
下一个 stance 贴墙点。这正是 paper 对 Cassie swing foot 的设计迁移到
climbing 场景。

**幅度选择**：初版用 0.2 m / 0.3 m（对应完整的 reach 姿态）训练时 policy 难
触及，只能偶尔 spike。按 paper 对 Cassie clearance ~15 cm 的量级缩到
**0.1 m / 0.15 m**，并把 sigma 放宽到 **0.18 / 0.20**，这样 5–10 cm 的小幅
swing 也能拿到显著 reward，early-learning basin 更宽、更易触发。policy 学
出初步 swing 行为后可以再用 curriculum 把幅度阶跃放大到真实 climbing 需要
的尺度。

---

## 关闭项

### `climb_height`（函数保留，未启用）
需要配合 `height_above_spawn` 观测才学得出位置式信号，曾有"挂在 baseline
附近白嫖"的局部最优问题；现在靠 velocity tracking 替代。
`height_above_spawn` 观测留着给 critic 做 state 描述用。

### 原 `climb_vz`（clamp 版本）、`swing_hand_reach_higher`、`airborne_penalty`、
### `both_hands_unlatched` termination

全部禁用。`climb_vz` 被 `track_vz` 取代（对称高斯核）；`swing_hand_reach_higher`
过度约束 gait 反而干扰探索；`airborne_penalty` + 硬 termination 组合在训练
早期打断 policy 探索（observed: entropy 反而上升，value loss 不降）。当前版本
**完全不约束手的 latch 状态**，看 policy 能否靠纯 velocity tracking 自己学出
合理的爬升。

---

## 权重量纲对比

vz_cmd = 0.3 m/s。假设 shaping 都接近满分：

| 场景 | $v_z$ | swing sine arc | 其他分解 | 总和 |
|---|---:|:---:|---|---:|
| 静态双 stance，不动 | 0 | 两项 ≈ 0（stance ref=0，手确实贴墙）| $8 + 15 + 8 + 3 + 6 + 6 + 9.5 + 3 = 58.5$ | **~59** |
| 匀爬 vz=0.3，不 swing | 0.3 | 两项 ≈ 0 | $8 + 15 + 8 + 3 + 6 + 6 + 40 + 3 = 89$ | **~89** |
| **匀爬 vz=0.3 + 正确 swing** | 0.3 | swing_height +10，swing_reach +15 | $89 + 10 + 15 = 114$ | **~114** |
| 匀爬但 swing 方向错（反贴墙）| 0.3 | 两项 ≈ 0 | $89$ | **~89** |
| 冲刺 vz=0.6 | 0.6 | ≈ 0 | $40 e^{-1.44} \approx 9.5$ | **~59** |

关键梯度：
- 静止 59 → 匀爬 89：每步 **+30**，vz tracking 驱动
- 匀爬 89 → 匀爬+正确 swing 114：每步 **+25**，sine arc 额外驱动 swing 动作
- 完整从静止到"匀爬 + 正确 swing"：**+55/step**，非常清晰
- 冲刺过速和静止一样低（对称高斯天然压制）
- regularization（posture + stance ≈ 6/step）不 dominant，只约束姿态

sine arc 在 gait 周期里 **mid-swing 时给出最强信号**，专门引导手的 task-space
轨迹——之前 policy 卡在"双手 stance 不动"的根本原因就是 swing 没有任何激励项，
加上这两项后 swing 有明确 +25/step 的 reward 目标。

## 墙几何辅助

`_SphereGeomCache` 懒加载缓存 16 球 geom ID、半径、墙半厚；每步从 wall
mocap 拿墙位姿和 outward normal（curriculum 改墙倾角自动跟随）。
