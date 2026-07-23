# motionref1 (2026-07-21_15-48-14_motionref1-mj36) 奖励与观测公式

- 代码版本:commit `11fe46d`(运行开始于提交后 5 分钟,logs 内 env.yaml 与该版本一致)
- 时序:sim dt 0.002 s × decimation 10 → 策略步 Δt = 0.02 s(50 Hz),episode 30 s
- 命令爬升速度:`climb_velocity.lin_vel_z` = 0.01395 m/s(= 动作参考一个循环的平均爬升率),
  归一化步进 `target_step` = v_z·Δt = 2.79×10⁻⁴ m/步
- 本版 limit surface 为 **wedge 线性锥**(wang2017 于 07-23 落地,不属于此 run)
- RewardManager 语义:总奖励 = Σᵢ wᵢ·rᵢ(s)·Δt。事件型罚分在函数内 ÷Δt,使 wᵢ 成为一次性事件成本。

## 1. 奖励项

### 1.1 DeepMimic 模仿项(正收入,exp 核)

参考量 `ref_*` 来自手写运动参考 clip 在当前相位帧 k 的值;机器人量取同一关节/EE 子集。

| 项 | w | 公式 |
|---|---|---|
| `motion_joint_pos` | 2.0 | exp( − mean_j (q̂ⱼ − qⱼ)² / σ² ), σ = 0.2 rad |
| `motion_joint_vel` | 0.5 | exp( − mean_j (q̂̇ⱼ − q̇ⱼ)² / σ² ), σ = 1.5 rad/s |
| `motion_ee_pos` | 1.5 | exp( − mean_h ‖p̂ʳₕ − pʳₕ‖² / σ² ), σ = 0.1 m;p ʳ = EE 在**根系**的位置(两侧都根相对 → 只塑形肢体几何,不锚定全局轨迹) |
| `motion_root_ori` | 0.5 | exp( − θ² / σ² ), θ = 四元数测地角误差 rad(仅姿态,位置自由), σ = 0.3 |

### 1.2 相位啮合调度

| 项 | w | 公式 |
|---|---|---|
| `motion_engage_mismatch` | −0.25 | Σₕ 1[ êₕ(k) ≠ eₕ ] — clip 啮合调度与实际 latch 不一致的 EE 计数(每步连续计费) |

### 1.3 任务驱动(爬升收入)

| 项 | w | 公式 |
|---|---|---|
| `climb_height_progress` | 2.0 | 高水位线:r = clamp( (max(z_best, z) − z_best)/target_step, 0, **12** );settle 0.5 s 内高水位跟随不计费;仅 \|cmd_z\|>0.01 时激活 |
| `signed_height_progress` | 0.5 | r = clamp( (z_t − z_{t−1})/target_step, −12, +12 );向下走负分,提供高水位下方的恢复梯度 |

> 注:max_normalized_progress=12 即允许瞬时 12× 参考速度的收入,且无逐 episode 预算帽——
> 这是 motionref1 审计发现"撕爪狂奔"(height:模仿收入 15:1、20× 超速)的直接原因;
> 预算帽(1.25× 参考里程)与 ref-capped signed progress 在 71c1cd9(motionref2)才落地。

### 1.4 姿态/几何护栏

| 项 | w | 公式 |
|---|---|---|
| `lateral_velocity_y_l2` | −0.2 | min( (v_y/0.1)², 4 ),v_y = 基座世界系侧向速度 |
| `wall_distance_band_l2` | −0.2 | min( ( (\|d − 0.196\| − 0.08)⁺ / 0.1 )², 4 ),d = 基座到墙面距离;±8 cm 死区 |
| `ee_face_parallel_error` | −0.5 | mean_{h∈M} (1 − sₕ),sₕ = (1 + cos γₕ)/2,γₕ = EE face 轴(0,−1,0)与墙内法线夹角;掩码 M = engaged ∪ (接触球数>0),悬空摆动肢不计 |

### 1.5 Microspine 物理契约

| 项 | w | 公式 |
|---|---|---|
| `microspine_support_loss` | −4.0 | max(0, 2 − #engaged) — 支撑数低于 2 每步连续计费 |
| `unsafe_microspine_release` | −1.0 | [ Σₕ relₕ·c(reasonₕ) + 1[any rel ∧ #engaged<2]·1.0 ] / Δt;c = {reverse:0, tension:0.25, lateral:0.25, ultimate:1.0};settle 0.5 s 免罚。÷Δt 抵消 manager 的 ×Δt → 每事件一次性成本(reverse 脱钩免费=预期主动释放路径) |
| `ee_slide_penalty` | −0.1 | Σₕ eₕ·‖v_{h,tan}‖ — 已啮合 EE 的墙面切向速度(m/s),法向不罚 |

### 1.6 正则化

| 项 | w | 公式 |
|---|---|---|
| `ang_vel_xy_l2` | −0.05 | ω_x² + ω_y²(基座体坐标系,yaw 不罚) |
| `dof_torques_l2` | −1×10⁻⁵ | Σⱼ τⱼ² |
| `bounded_dof_acc_l2` | −0.02 | min( mean_j (q̈ⱼ/100)², 10 ) — 有界防单样本炸批 |
| `action_rate_l2` | −0.01 | Σⱼ (aⱼ,t − aⱼ,t−1)² |
| `dof_pos_limits` | −5.0 | Σⱼ 软限位越界量(下越 + 上越) |
| `termination_penalty` | −500 | 1[非超时终止] |

## 2. 观测

Actor / critic 均单帧(history 0)、**全部无噪声**(noise: null;唯一注入噪声是 RSI 复位关节噪声 ±0.02 rad)。

### 2.1 Actor(16 项,拼接)

| 项 | 维度 | 公式/来源 |
|---|---|---|
| `base_lin_vel` | 3 | `robot/baselink-velocity` 传感器(基座线速度) |
| `base_ang_vel` | 3 | `robot/baselink-gyro` |
| `projected_gravity` | 3 | 由 `baselink-quat` 传感器旋转重力方向到基座系 |
| `joint_pos_rel` | n_j | qⱼ − q_default,policy 关节序 |
| `joint_vel_rel` | n_j | q̇ⱼ |
| `last_action` / `prev_action` / `prev_prev_action` | 3×n_j | aₜ₋₁, aₜ₋₂, aₜ₋₃ |
| `height_above_spawn` | 1 | z_base − 2.0(常数基准;让 V(s) 看得见高度,height 收入才不是隐藏态噪声) |
| `velocity_command` | 3 | (0, 0, 0.01395) |
| `microspine_engaged_state` | 4 | eₕ ∈ {0,1}(affordance latch,真值源) |
| `microspine_contact_fraction` | 4 | clamp(接触球数ₕ/4, 0, 1) |
| `microspine_disengage_time` | 4 | min(t_disₕ, 1.5)/1.5 — 连续脱开时长(啮合时归零) |
| `microspine_cooldown` | 4 | cooldownₕ/10 — 再啮合冷却归一化 |
| `motion_command` | 6 | [cos φ, sin φ, ê₁..ê₄],φ = 2π·k/K 循环相位,ê = clip 当前帧啮合调度 |

### 2.2 Critic(= actor + 特权项,非对称 AC)

| 项 | 维度 | 公式/来源 |
|---|---|---|
| `microspine_hold_margin` | 4 | wedge 四面归一化最小裕度:m = min( (f_t+f_rev)/f_rev, (cap_n−f_n)/cap_n, (cap_l−\|f_l\|)/cap_l, (f_tmax−f_t)/f_tmax ),cap_n=μ_adh·f_t⁺+f_n0、cap_l=μ_lat·f_t⁺+f_l0;未啮合=1 |
| `microspine_anchor_load` | 12 | (f_t, f_l, f_n)ₕ 锚点载荷(啮合系),×0.01 缩放到 O(1) |
| `microspine_forced_release` | 4 | relₕ ∈ {0,1} 本步 limit-surface 强制释放标志 |

## 3. 与后续版本的差异备忘

motionref1 审计(撕爪狂奔)后,motionref2(71c1cd9)起变更:
`climb_height_progress` 加逐 episode 预算帽(1.25× 参考里程)、signed progress 改 ref-capped、
`motion_engage_mismatch` −0.25→−1.0(加宽限)、`unsafe_microspine_release` −1.0→−2.5、
性能门控换回固定 iteration 重力排程(45/60/75/90° @ 0/3k/6k/9k);
motionref5(bae909d)再加 seat-phase press credit 与更高 reach 收入;
07-23 起 limit surface 换 wang2017 闭合曲面(critic 的 hold_margin 语义随之改为径向裕度)。
