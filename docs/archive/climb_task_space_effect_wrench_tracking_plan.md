# Task-Space Effect + Wrench Tracking 的务实落地方案

## 面向随机 Microspine 攀爬机器人 —— OSC 蒸馏母文档的裁剪执行版

> 文档状态：执行计划（决策已定，待实现）
> 关系：本文是 [`floating_base_osc_wrench_intent_rl_distillation.md`](floating_base_osc_wrench_intent_rl_distillation.md) 的**落地裁剪版**。母文档是终局架构图纸；本文只回答"用现有资产、以最低成本，把 task-space effect 与 wrench 跟踪先跑起来"。
> 核心目标：让 position-only policy 复现 **teacher 的任务空间运动效果（effect）与载荷分配（wrench intent）**，但**不建 OSC teacher**。

---

## 0. 与母文档的关系（一段话）

母文档提出 `Planner → OSC Teacher → Wrench-Guided RL Student → Safety Projection → Position Servo` 的完整体系。它物理判断正确、坑位预判准（尤其 §11 phase/event reference 和 §10.4 硬终止两处与本项目实战教训吻合），但它回答的是"怎么爬得**安全、对**"，主体机械（OSC teacher、在线 DAgger、belief、safety QP）解决的是**载荷正确性与安全性**，工程量以月计，且不直接解当前"手脚接续换爪探索撞不出来"的死结。

本文的立场：**OSC 在母文档里是可替换零件，"蒸馏"（跟踪 effect 与 wrench intent）才是主体。** 我们保留蒸馏框架，把最贵的零件（OSC teacher）替换成廉价来源，分阶段兑现。

---

## 1. 一句话方案

> **effect 管"动"的相位，wrench（其中只有载荷分配 α 有信息量）管"不动"的相位；两者的参考不靠 OSC，而是靠"手写运动学参考 + 一个离线小 QP"生成。**

控制逻辑（对比母文档 §22）：

```
母文档：  OSC 现算 w_G*, α*, x*  →  effect/wrench reward  →  position policy  →  安全 QP
本方案：  手写运动学参考 x*(φ)                                  ┐
          离线小 QP 算 α*(φ)      →  effect/α/margin reward  →  position policy（现有动作空间）
```

---

## 2. 四个关键认识

**① teacher 需要的是"输出"，不是"程序"。**
policy 训练需要的是每个相位下的参考值 `x_B*(φ), x_ee*(φ), α*(φ), γ*`，不是产生它们的那个控制器。谁提供都行——手写、小 QP、甚至 AMP 成功 rollout 的统计。母文档 §6.5/§22 自己也强调 student 从不模仿 OSC 的 torque 或逐爪力，只跟踪合成效果。

**② 准静态攀爬里，OSC 只有约 30% 有用，且那 30% = 一个小 QP。**
OSC 的看家本领是动态解耦（高速前馈）。攀爬是准静态的（母文档 MVP §19 自己限定"慢速准静态"），动力学那一半用不上。真正需要的只是"平衡约束下的载荷分配"，这就是一个静力学小凸 QP，不带时序决策、不带动力学、不上 GPU、可离线跑。它同样能在任意状态查询（state-aligned 性质保留），只是没有动力学。

**③ effect 与 wrench 互补，各补对方的盲区。**
- effect reward（跟踪基座/末端速度与位姿）在**运动的相位**有效；但攀爬最要命的 seat/transfer 相位几乎零运动，effect 在这里完全失明。
- wrench/载荷分配跟踪在**零运动的相位**才有内容（载荷怎么从旧爪转到新爪）。
- 两者叠加才覆盖全相位。母文档 §7.2+§7.3 的组合正是为此。

**④ effect 参考可手写，α 参考必须小 QP；w_G 平凡。**
- 末端三段轨迹（离墙→沿墙上移→压回）可以直接写脚本生成，无需 IK、无需动力学。
- α*(φ) 必须满足静力平衡 + 落在各爪 limit surface 内，手写极易物理不可行 → 必须用小 QP 解。
- **w_G（合成 wrench）在准静态下几乎没有信息量**：身体不动时它恒等于"托住自重"，任何没掉下来的策略自动满足。所以"加 task-space wrench"实操上 = **加一项 α 跟踪**，w_G 可加但权重给低，聊胜于无。

---

## 3. 相位 → reward 职责映射

现有相位机 `ClimbGaitPhase`（`tasks/climb_wall/mdp/commands.py`）：`RELEASE=0, SWING=1, SEAT=2, TRANSFER=3`，由 `resolve_climb_gait_phase` **接触驱动**推进（"without advancing timers"）——即母文档 §11 要的事件门控**已经做好了**。

| 相位 | 物理动作 | 是否有运动 | 主导 reward |
|---|---|---|---|
| SWING (1) | 自由爪离墙、沿墙上移、压回 | 有（大） | **effect：末端三段轨迹跟踪**（+ 基座保持） |
| SEAT (2) | 压墙、等待啮合 | 几乎零 | **按压 shaping**（现有）+ 后期 α 起步；effect 在此失明 |
| TRANSFER (3) | 载荷从旧爪转到新爪 | 几乎零 | **α 载荷分配跟踪**（新建）+ margin |
| RELEASE (0) | 旧爪卸载脱离 | 小 | α 卸载节奏 + **margin** |

三路分工：**effect 管动的相位、按压+α 管不动的相位、margin 全程兜安全底。** 这样每一相位都有稠密信号，且各自可单独画曲线验收。

---

## 4. 与现有代码资产对照

**已经在跑的（effect 侧）——比预想的多：**

| 能力 | 现有实现 | 权重 |
|---|---|---|
| 基座速度跟踪 | `climb_velocity_z_error_l2`, `climb_lateral_velocity_y_l2`, `climb_vz_gauss` | -0.5 / -0.2 |
| 末端目标误差 | `commanded_ee_target_error_l2`, `ee_slide_penalty` | -0.2 / -0.1 |
| 摆动进度 | `climb_swing_progress_shaping`（标量，易被"半空摆爪"刷） | 2.0 |
| 墙距/朝向 | `wall_distance_band_l2`, `ee_face_parallel_error`, `hand_orientation_toward_wall` | -0.2 / -0.5 |
| 相位机（事件门控） | `ClimbGaitPhase` + `resolve_climb_gait_phase` | — |

**已经在跑的（安全/wrench 侧）：**

| 能力 | 现有实现 |
|---|---|
| 载荷读数 λ（= 母文档 λ_i） | `microspine.constraint.compute_anchor_loads` → obs `microspine_anchor_load` |
| 安全裕度 γ_i（= 母文档 γ） | `microspine_hold_margin`（obs + metric `microspine_hold_margin_engaged`） |
| 释放/失稳罚 | `microspine_support_loss`(-3.0), `climb_gait_support_shortfall`(-2.0), `unsafe_microspine_release_penalty`(-1.0), `unscheduled_microspine_release`(-2.0), `microspine_forced_release_penalty` |
| seat 按压 shaping | `climb_seat_press_shaping`(1.0)，affordance 暴露 `pad_press_force` |
| 相位超时=硬终止（§10.4 纪律） | termination `gait_phase_timeout` + penalty `climb_gait_phase_timeout`(-5.0) |
| 特权 critic obs | `microspine_hold_margin`, `microspine_anchor_load`, `microspine_forced_release` |

**缺的（真正要建的只有三块）：**

1. **连贯的相位索引任务空间参考源**——尤其 SWING 的三段末端轨迹。现在 `commanded_ee_target_error_l2` 跟的是命令端点，不是**随相位推进的移动路点**，因此允许任意路径（含可刷分路径）。
2. **margin 正奖励 + 容量罚**——读数全有，缺激励项：`softmin_i(γ_i)` 正奖励 + `Σ_i[u_i−β]_+²` 罚。
3. **α 载荷分配跟踪 + α*(φ) 参考**——sim 侧 α 只需把 `microspine_anchor_load` 归一化；参考 α*(φ) 需离线小 QP。（w_G 跟踪低价值，可选，暂不做。）

结论：**"effect + wrench tracking" 的大部分机械已经接好；剩的是"给它们一个连贯参考 + 补一个 margin 项 + 补一个 α 项（背后一个小 QP）"。**

---

## 5. 分阶段落地

### 阶段 A —— 参考源 + effect + margin（不碰 QP）

**目标**：第一次跑通完整换爪循环（解 07-17 的探索死结）。

**动作**：
1. **建相位索引任务空间参考源**（见 §6 的两个选项，推荐"扩展 command"）。核心是把 SWING 的末端目标从**端点**改成**沿三段路径、按 swing 进度参数化的移动路点**——这把 `commanded_ee_target_error_l2` 从"到终点的误差（允许任意路径）"变成"到当前相位应在位置的误差（强制抬手越过再压回）"，既是真 effect 信号，又天然抗刷分（对移动 setpoint 的 L2 本身双向计费，不像标量 `climb_swing_progress_shaping` 会被半空摆爪攻破）。
2. **提高 SWING 段 EE 跟踪权重**，相应把易刷分的 `climb_swing_progress_shaping` 降权或退役。
3. **加 margin 正奖励**：`r_margin = softmin_i(γ_i)`（只对已啮合爪），γ_i 取自 `microspine_hold_margin`；配 `c_cap = Σ_i[u_i−β]_+²`（β≈0.8）。作用：策略不再贴着拉脱边界干活，缓解释放风暴。

**验收**：完整循环 success 首次稳定 >0；侧位抽帧目视确认手臂多次离墙重挂 + 身体沿墙上移（沿用 07-17 的 `play --video True`, `MUJOCO_GL=egl`, 相机 lookat(-0.6,0,1.2)+azimuth 90）。

**不做**：QP、α、w_G、动作空间改造。

### 阶段 B —— 小 QP → α 载荷分配跟踪

**前置**：阶段 A 循环已跑通（手里有成功 rollout 可交叉验证 α*）。

**目标**：换爪时载荷转移平滑、不把单爪压爆——sim-to-real 前必补的"载荷正确性"课。

**动作**：
1. **建离线小 QP**（规格见 §7），按 (阶段, φ) 生成 α*(φ) 表。
2. **加载荷分配跟踪** `r_share = exp(−‖α − α*(φ)‖²_Qα)`，**phase-gated 只在 SEAT/TRANSFER/RELEASE 开**；α 由 `microspine_anchor_load` 归一化得到。
3. 阶段 A 的粗按压 shaping 可降权——α 跟踪是它的精细化替代。

**验收**（母文档 §17.3 子集）：TRANSFER 期最差 margin ↑、α 误差 ↓、释放瞬间载荷突变 ↓，且循环 success 不掉。

**不做**：w_G 跟踪（除非发现确有需要）、在线 QP。

### 阶段 C —— 推迟到实机 / 恢复行为阶段（现在一律不碰）

- OSC/WBC teacher、DAgger、state-aligned 在线查询（4096 env 每步解 QP，mjwarp 跑不动，工程量巨大）；
- contact belief / observer（母文档 §12，依赖实机传感器配置，现在定不了）；
- safety projection QP（母文档 §15，部署期单机安全层，训练期用不上）；
- 单爪失效后的载荷转移恢复（AMP/手写参考没去过的状态没有参考——**这才是模型基 teacher 真正的强项，也是届时才值得建准静态 QP 做 DAgger 的地方**）；
- 母文档 §17 的全套 baseline 与消融（写论文用）。

---

## 6. 参考源的两个选项

**选项 1（推荐）：扩展 `ClimbGaitCommand`。**
现有 `ClimbGaitCommand` 已经发出"命令末端目标 + 相位 one-hot"（`_write_command`）。把它扩展为发出**相位参数化的任务空间 setpoint**（SWING 段沿三段路径的移动路点 + 基座目标），现有跟踪 reward 直接消费。**无 clip 文件、无 AMP、无判别器黑盒，每项误差可直接画曲线调试。** 这是"我要 track task-space effect"的最直接表达，也最便宜。

**选项 2（备胎）：`gen_climb_regrip_clip` + AMP 判别器。**
按 `scripts/gen_squat_clip.py` / `gen_stand_clip.py` 模式写参考剪辑，走 walk 线已有的 AMP 基建（`amp_rsl_rl/runner.py`, walk `amp_rl_cfg.py`, `walk/mdp/amp_reset.py`）。

**取舍**：你们的参考剪辑反正是脚本合成的（自带相位标签），AMP 最大的优势（不需相位对齐）用不上，却要吃它最大的劣势（判别器不稳、难调试）。**对本场景，选项 1 的稠密可控 tracking 很可能比 AMP 更合适。** AMP 留作"选项 1 的稠密 tracking 仍撞不破探索"时的备胎。

> 注：稠密 effect tracking 直接攻击探索问题——它在每一步给出朝参考的梯度，正是纯稀疏 reward 所缺。动作是关节残差、参考是任务空间，中间仍有 IK gap 要策略探索跨越，但稠密的任务空间误差把这个梯度从稀疏变平滑，强于现有标量 progress shaping。

---

## 7. 离线小 QP 规格（阶段 B 用）

```
输入（按相位 φ 与运动学参考给定）：
  x_B*(φ)         基座位姿（来自 §6 的运动学参考）
  p_i             各承载爪接触点（正运动学得到）
  S(φ)            承载爪集合（此刻挂着的爪）
  Θ_i             各爪 limit surface 参数 —— 用【标称值】，不是 DR 的随机值
                  （参考是目标，鲁棒性由 DR 兜底）

决策变量：λ_i,  i ∈ S(φ)         各爪接触 wrench

目标：   min Σ_i ‖λ_i‖²
   或    max t   s.t. margin_i ≥ t   （最大化最差爪裕度，更贴 §9.2 softmin 口味）

约束：
   Σ_i G_i(φ) λ_i = w_grav(φ)        静力平衡（准静态，惯性≈0）
   ρ_{W_i(Θ_i)}(λ_i) ≤ β,  β≈0.8      收缩后的 limit surface（留 20% 裕度）

输出：λ_i*(φ)  →  归一化  α_i*(φ) = ‖P_s λ_i*‖ / Σ_j ‖P_s λ_j*‖
存储：按 (阶段, φ) 索引成表，离线跑一次即可
```

小凸 QP，接触点/G/limit surface 参数在 `microspine` 模块里都是现成的。几天工作量，母文档 OSC teacher 的三个贵点（会爬的时序、扛随机接触、批量上 GPU）一个不沾。

---

## 8. 与 AMP 的关系（一句话）

同一个"引导探索"坑位的两种做法，**二选一或叠加**。07-17 定案的"转 AMP"没有作废，而是被**降级为备胎**：先试选项 1 的稠密 task-space tracking（更可控、可调试、直接表达"track effect"的意图），撞不破再上 AMP。两者可叠加——AMP 管"动作分布像"，tracking 管"任务空间效果对"。

---

## 9. 风险与纪律

1. **reward 电池膨胀 = 攻击面膨胀。** 你们的历史（伸懒腰农场、半空摆爪刷分、挂死不动）证明每加一项 shaping 就多一个被攻破的口子。**A、B 分两步上，每项新 reward 单独验指纹**，不要一次全开。母文档训练流程（阶段 2→3 先 motion 后 wrench）也是这个节奏。
2. **指纹判读纪律不变**（见 [[climb-pd-regression]]）：`margin 转正 + 超时率高 + 释放率跌 = 挂死躺平`；`margin 偏负 + 释放率高 = 积极干活`；`mean_std 持续上升不回头 = 止损`。
3. **相位门控**：参考按 φ 走，φ 只有挂稳才前进（现有 `resolve_climb_gait_phase` 已保证）——参考永不拖着策略在没挂稳时强行移动/加载，绕开母文档 §11 的坑。
4. **α 参考的 limit surface 用标称值**，不是每 episode 随机的那个——参考是目标，鲁棒性交给 DR。teacher 与 reward 的风险假设要一致（母文档 §9.3）。
5. **不可恢复越界 = 硬终止**（现有 `gait_phase_timeout` 终止已落地），不靠有限罚款（母文档 §10.4 + 07-15 罚款经济学两次失败教训）。

---

## 10. 起手动作 checklist

- [ ] **阶段 A-1**：在 `ClimbGaitCommand._write_command` 里，把 SWING 末端目标改为"沿三段路径按 swing 进度参数化的移动路点"（选项 1）。
- [ ] **阶段 A-2**：提高 SWING 段 `commanded_ee_target_error_l2` 权重，降权/退役 `climb_swing_progress_shaping`。
- [ ] **阶段 A-3**：在 `tasks/climb_wall/mdp/rewards.py` 加 `climb_margin_softmin`（正）+ `climb_capacity_penalty`，读 `microspine_hold_margin`；在 `env_cfgs.py` 注册。
- [ ] **阶段 A 验收**：完整循环 success 稳定 >0 + 侧位视频目视确认。
- [ ] **阶段 B-1**：写离线小 QP（§7），产出 α*(φ) 表。
- [ ] **阶段 B-2**：加 `climb_load_share_track`（phase-gated SEAT/TRANSFER/RELEASE），α 由 `microspine_anchor_load` 归一化。
- [ ] **阶段 B 验收**：TRANSFER 最差 margin↑ / α 误差↓ / 释放载荷突变↓ / success 不掉。

---

## 11. 最终定义

> **以"手写运动学参考 + 离线小 QP"替代 OSC teacher，让 position-only policy 跟踪其任务空间运动效果（effect，管动相位）与载荷分配意图（α，管静相位），在现有 limit-surface 安全裕度（margin）兜底下，先跑通完整换爪循环，再精修载荷正确性。**

保留了母文档蒸馏框架的全部方法论内核（set-valued 跟踪、phase-dependent reward、事件门控、位置伺服建模、特权 critic），去掉了它最贵的零件（OSC teacher、在线 DAgger、belief、safety QP），并把这些推迟到实机/恢复行为阶段按需兑现。
