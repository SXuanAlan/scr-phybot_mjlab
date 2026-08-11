# 统一 Mount+Climb 策略训练方案

## Specialist 老师 → 统一特权老师（按段 KL 路由）→ RMA 压缩出货

> 文档状态：执行计划（路线已定，参数待跑）
> 日期：2026-08-10
> 已定决策：统一老师从 **climb-90 teacher 热启动**；mount 技能靠 KL 注入
> 方法来源：LadderMan（`docs/reference/LadderMan/`，arXiv 2606.05873）的 DAgger+RL 退火 KL；HOVER（arXiv 2410.21229）的条件化统一策略；Ji et al. RA-L 2022 的 policy+estimator 并发训练
> 前置条件：climb-cfg-restructure 分支 B 系列已把两个任务的世界统一（真重力/真地板/墙倾斜，B8 "terminal world equals climb_transition's"）；mount specialist v4 训练进行中

---

## 0. 一句话方案

> **特权和"统一"都发生在老师层；压缩到纯本体感知只做一次、放在最后、复用已验证的 RMA 配方；RL 全程在场，模仿只当缰绳不当上限。**

```
阶段 0                阶段 1               阶段 2                      阶段 3
climb 老师(已有0.955)  链式考卷            统一特权老师 π_uni^T          RMA 压缩
mount 老师(v4 收尾)  → mount→冻结climb   → 全程episode + PPO          → φ_uni 混合分布并发训练
                      出货对切换,散布网格    + λ·KL(按段路由两个冻结老师)   → 出货对 (π_uni, φ_uni)
                      ↓                    λ 退火,接缝淡出带             契约不变,C++ 单网络
                      接缝基线 + 方案C保底   硬门: 90° 不许掉              + mount表‖cycle表拼接 LUT
```

每一步产物冻结、都有考卷；任何一步失败，上一步产物按方案 C（双策略切换）照常出货。全程无单点风险。

---

## 1. 为什么合并、为什么是现在

**为什么合并：**

1. **接缝是当前唯一没被闭环验证的环节。** mount 的 handoff 帧对 cycle 帧 0 只做过静态审计（关节 delta 4e-8 rad）；带投递残差的开局 cycle 态两个策略都没在训练里见过。统一策略让训练每天穿越接缝，问题从"要验证"变成"不存在"。
2. **部署简化。** 08-04 部署审计留下的"上墙流程"悬案直接消掉：C++ ClimbTask 只装一个网络，命令查表从 mount 表续到 cycle 表就是模式切换，不需要双策略切换状态机。此决策应在动笔写 C++ 之前定下（本文即为定案）。
3. **架子可复用。** "多段剧本 × 按段路由老师 × 一个学生"搭好后，descent、station-keep、dismount 都是往命令表加段、往老师库加专家的事。

**为什么是现在能做：** B 系列已把 climb 世界改成与 transition 终态世界相同；transition 契约按构造与 climb 同构（obs/rewards/PD/DR 逐位继承，`climb_transition/config/phybot_c2/env_cfgs.py` 文档字符串）；两任务的差异全部走 motion_command 命令通道内容。合并所需的所有零件已在场。

---

## 2. 方法来源与关键认识

| 来源 | 取什么 | 弃什么 |
|---|---|---|
| LadderMan | 单跑 `L = L_RL + λ·KL(学生‖老师)`，λ 退火；多老师按情况路由；"RL 填老师之间的空隙"（其 w/o RL 真机消融 9/6/7→2/0/0） | 感知栈（VFM/RFM）现阶段无关：我们是盲策略+平面墙 |
| HOVER | 条件化统一策略不掉点、可运行中切模式的存在性证明；我们的 motion_command 就是它的 masked command（且更丰富） | 学生纯 DAgger 无 RL——只适用自由空间跟踪，接触密集任务纯模仿会崩（LadderMan 消融 + 我们 BC 天花板 0.835 双重证据） |
| Ji et al. 2022 | policy+estimator 同批 rollout 并发训练（阶段 3 用，替代原 2a/2b/2c/Phase3 四段） | — |
| 我们自己 | RMA 配方（teacher→φ→ê→RL finetune，0.969 已验证）；考卷文化（每阶段冻结+门控） | — |

**关键认识（决定初始化策略的依据）：真正干活的是 KL 监督，不是初始化。** mount v4 从 climb ckpt 热启动照样在 reach-up/press 上点不着火，靠 spawn-bias + press-bootstrap 硬打——说明热启动不解决新技能点火，点火难是因为没有 mount 老师可抄。第一个 specialist 必须由 RL 生出来；生出来之后，点火战的成果封装在老师里，统一训练靠 KL 逐状态抄答案，不再重打。

三种组合的排序：从零+纯奖励（重打所有仗，禁止）＜ 从零+双路 KL（可行但烧钱，LadderMan 被迫如此因为观测空间不同没法热启动）＜ **热启动+双路 KL（我们的选择：初始化管起点，KL 管全程，两者不互斥）**。

---

## 3. 设计原则

1. **特权信息只进老师和考卷，永不进出货物**（RMA 骨架不变）。
2. **统一在老师层做**：两个老师同为特权 obs、同一契约，KL 干净无歧义；学生也用特权 obs，一份观测喂三个网络。
3. **压缩只做一次、放最后**：那是我们管线最熟的一段。
4. **模仿是初始化+防遗忘缰绳，不是能力上限**：λ 必须退火，RL 必须能超越老师。
5. **每阶段有冻结产物和考卷**；方案 C 全程兜底。

---

## 4. 阶段 0：把两个 specialist 练完（现行工作，不变）

- climb 老师：已有（teacher 0.955@90° 全装备血统）。
- mount 老师：v4 照常收尾。**重新定位：它从此只当老师、永不部署**，因此可以放开手脚——特权观测随便用；奖励侧可加两件 LadderMan 战术件：
  - **宽盆接触吸引**：现有 `motion_swing_approach` std=0.05（15 cm 外梯度死区），LadderMan contact reward τ=0.3（等效 σ≈0.55 m，半米外仍有 0.43 付酬）。若 v4 reach-up 点火仍有残留平台期，在 transition 包内加 mount 专用双尺度核（远端 σ≈0.3–0.5 m tow + 现有近端 funnel）。
  - **非对称臂 tracking**：盲投递散布下 clip 手臂路径带 offset 是"错"的、wall anchor 是环境真值——臂侧 `motion_joint_pos`/`motion_ee_pos` 降权（×0.5）放宽（std×2），腿与 root 保持严格，anchor/approach/engage 项全额。对应 LadderMan Eq 1 上半身 ω/2、σ/2。
- **产出：两个冻结的特权老师。**

## 5. 阶段 1：链式考卷（量化接缝，锁定保底）

- 内容：mount 跑完，在 `mount_handoff_ready` 触发时切入冻结的 climb 出货对（estft-nc/model_599 + phi_2b_nc），散布网格 × 存活/爬升 N 秒判分。
- 两个用途：给统一策略定"必须打赢"的基线；同时验证方案 C 作为全程可用的保底出货路径。
- **判读：若网格全绿，统一工程从"救命"降级为"部署简化"，优先级可重排；若漏，阶段 2 就是修复手段。**

## 6. 阶段 2：统一特权老师（核心新工作）

### 6.1 配置

- **初始化：climb-90 teacher 热启动（已定）。** 90° 全装备技能是最难挣的资产，起点站在它上面；mount 技能靠 KL 注入。从零+KL 仅在两种情况下启用（见 §9 风险表）：换架构（更大网络/加记忆——那时热启动物理上不可能，KL 是唯一搬运通道）、或负迁移实锤且课程掰不过来。
- **环境**：B8 统一世界。episode = 站立散布出生 → mount 表 → handoff → cycle 表 N 周期。命令流拼接即模式切换，无显式 switch。
- **目标函数**：

```
L = L_PPO(现有奖励，按段取用：mount 段用 transition 奖励，cycle 段用 climb 奖励)
  + λ(t) · KL( π_uni(·|s) ‖ π_expert(段)(·|s) )
```

### 6.2 "问老师"的机械（实现要点）

1. 学生 rollout 每步：同一仿真状态组装**一份** obs（契约同构；命令通道 handoff 前装 mount 表值、后装 cycle 表值——正好各是当段老师训练时见的东西，无需观测适配器），同时喂学生和当段老师。
2. 冻结老师前向，存动作分布（均值+方差）进 rollout buffer；老师只推理不更新。
3. PPO update 逐样本加 λ·KL。对角高斯 KL 有闭式解；实现可先用对老师均值的 MSE 打通（HOVER 做法），再换 KL（额外好处仅是老师 std 自带置信加权）。
4. **必须一致的是动作空间**（维度/顺序/缩放/分布族）——三个网络同一 cfg 继承，天然满足。观测空间不要求一致（本例恰好一致是特例，不是要求）。
5. 历史通道（关节史/prev-action）装的是**学生**的历史——这是 DAgger 语义（问老师"从这个处境你会怎么办"），轻微分布错位是已知代价，LadderMan 老师 obs 含 a_{t-3:t-1} 同样如此。

### 6.3 λ 日程与接缝淡出带

- λ 初值压住（模仿主导）至学生任务分接近老师，随后指数退火趋零。具体数字待调（LadderMan 未公布其日程），在 λ 档位留 checkpoint 挂考卷。
- **接缝淡出带（主动设计，非自动发生）**：老师是函数，任何状态都答得出，但带 mount 残差的 cycle 开局态两个老师都没见过，答案是噪声，KL 往噪声上拉是负资产。handoff ±T 帧内 KL 权重线性降到 0，该区间 RL 独占。T 初值建议 0.3–0.5 s，待调。

### 6.4 episode 开局混采

三种开局按比例混采（初值建议 0.3/0.5/0.2，待调）：

| 开局 | 用途 |
|---|---|
| 整链（站立散布起） | 训接缝穿越与全程一致性 |
| 墙上 RSI 直接进 cycle | 保 cycle 经验份额；R 系恢复训练（DR 强拆/冻钟/reseat）照旧在此运行 |
| 接缝附近 RSI（handoff±） | v4 spawn-bias 教训的管线级复用：把最难的区域喂足经验 |

### 6.5 硬门与可选件

- **硬门：90° 考卷不许掉**（match ≥0.95 双绿零摔 + 逐爪 occ 偏差 gate，沿用 v4 标准）。cycle 段 KL 缰绳是防遗忘机制，考卷是验证。
- mount raster 散布网格照考。
- 接缝链式考卷 ≥ 阶段 1 基线。
- 可选：contact-loss termination（LadderMan：必需接触连续丢失 >30 sim step 终止）仅在统一训练早期开，进 R 系恢复课程前**必须关**（与冻钟/reseat 机制冲突）。

### 6.6 奖励设计（学生用哪套 reward）

**核心事实：我们的奖励项是读剧本的函数，不是任务专属的函数。** `motion_joint_pos/ee_pos/engage_mismatch/anchor_landing/toe_brace/ref_height` 全部经 `_motion(env, command_name)` 读**当前表**的参考值；transition cfg 由 climb builder 变异而来，奖励项集合逐位相同。时钟从 mount 表滚进 cycle 表，同一组奖励项自动改按 cycle 参考评分——**奖励的按段切换与 KL 的按段路由是同一个开关（时钟在哪张表），天然对齐；同一时刻只有当段参考在场，两套奖励不存在打架**。

统一学生用**一套静态权重 = climb 平价版**。与两个 specialist 的差异只有三处：

| 差异项 | specialist 里 | 统一阶段处理 | 理由 |
|---|---|---|---|
| `motion_scheduled_press` | transition bootstrap 3.0→0.4 课表 | 恒 0.4（climb 平价） | 点火职责已由 KL 接管（mount 老师会压，学生抄作业）。**原则：凡点火用塑形，统一阶段一律回平价——历史使命已封装进老师** |
| random-release/冻钟（R 系） | climb 有，transition v1 无 | 按 episode 类型门控：仅墙上 RSI 开局的 episode 开；整链 episode v1 关；mount 段 release drill 留作后续 R 级 | 与 transition v1 决策一致 |
| `climb_velocity` 命令 | transition 钉 (0,0)，climb 采样 | 改 schedule-aware：mount 段恒 0，handoff 后按 cycle 采样 | 命令项，影响 obs 与进度类奖励 |

三个补充判断：

- **接缝不是奖励沙漠**：handoff 帧==cycle 帧 0，拼接后 cycle 参考立即在场，跟踪/接触项连续。KL 淡出撤掉的是"抄作业"，不是"评分"。
- **反赖账机制自动生效**：时钟推着走，赖在 mount 段=跟踪收入清零+mismatch 计费；statue 史的全套反赖账机件对拼接表照常工作。
- **跨段不需配平收入**：两段每步收入率不同没关系，critic 学的就是分段价值。

理论边界一句话：学生 L_RL 不必逐位等于老师的训练奖励（KL 不关心 reward），但必须**背书同一种好行为**——否则 λ 退火到零时策略往奖励漂、漂离老师。climb 平价集恰是两个老师奖励的公共核，退火安全。LadderMan 的选择（"L_RL uses the same rewards as the experts"）同理。

**真正的新工程在表而不在奖励**：engage 三张表须做 acyclic 前奏 + cyclic 后续的拼接，08-09 幻影边沿的教训会在拼接点重演——mount 命令的 terminal clamp 行为要从"钳死在 handoff 帧"改为"滚入 cycle 表"，哨兵行为重写 + 反幻影合同测试再加一轮。表对了，读表的奖励全对。

## 7. 阶段 3：RMA 压缩到部署形态

- **φ_uni 在混合分布上训练**：mount 段的隐变量是投递偏差，cycle 段的是 DR/挂爪状态，都从同一条 50 帧本体感知历史读出。
- **合跑升级**（替代原 2a/2b/2c/Phase3 四段）：同批 on-policy rollout 同时喂两个 loss——π 走 PPO finetune、φ 走对特权真值的监督回归。防早期 ê 垃圾带偏 π 的保险（二选一或叠加）：按概率混喂真值 e 与 ê、概率 1→0 退火（scheduled sampling）；前 N iter 冻 π 只训 φ（π 反正是热启动的）。
- **出货对 (π_uni, φ_uni)**：学生契约维度逐位不变（850 维 + 命令查表 sidecar）；C++ 单网络 + mount 表‖cycle 表拼接 LUT；estfreeze 状态机参数不变（arm3/clear5/超时 75）。
- 开放问题：ê 9 维是否需要扩维容纳 mount 段的投递偏差信息（扩维=契约变更，须在 C++ 落地前定）。倾向先试 9 维不动——投递偏差可能已可从历史隐式读出，考卷说话。

## 8. 考卷与验收门汇总

| 门 | 指标 | 阈值 |
|---|---|---|
| cycle 不回退 | 90° 确定性考卷 | match ≥0.95 ×2 零摔；逐爪 occ 偏差 <+0.04（沿用 v4 门） |
| mount | mount raster 散布网格 | ≥ specialist v4 终版 |
| 接缝 | 链式全程存活率（散布网格） | ≥ 阶段 1 双策略基线 |
| 恢复 | R 系恢复考卷（DR 强拆格） | ≥ 现役 estft-nc 水准 |
| 压缩后 | 在环 ê 全套（超真值喂养对照） | ≥0.96 零摔，恢复 2/2 |

## 9. 风险与回退

| 风险 | 信号 | 对策 |
|---|---|---|
| cycle 技能磨损 | 90° 考卷掉线 | 提高 cycle 段 λ / 提高墙上 RSI 混采比例；仍不行则回退该 λ 档 checkpoint |
| 负迁移（climb 先验拒学 mount 段） | mount 段任务分长期平台 + KL loss 不降 | 提高 mount 段 λ；实锤后启用 plan B：从零+双路 KL |
| 网络容量不足 | 两段技能此消彼长、λ 调不平 | 换大网络，从零+KL（热启动不可用，KL 成唯一通道） |
| 接缝 RL 填不上 | 接缝链考卷低于阶段 1 基线 | 加大接缝附近 RSI 比例/淡出带调宽；最终回退方案 C 双策略出货 |
| φ_uni 混合分布学不好 | 压缩后在环掉线集中于 mount 段 | ê 扩维试投递偏差通道；或 mount 段临时保留双策略、cycle 用 π_uni |

## 10. 开放问题（待跑出答案）

1. λ 初值与退火日程；KL vs MSE 实测差异。
2. 三种开局混采比例。
3. 接缝淡出带宽度 T。
4. ê 是否扩维（§7，契约敏感，C++ 前定）。
5. descent：同配方加段（倒放 clip + 反转 engage 表 + 老师库加 descent 专家）。`climb_velocity.lin_vel_z` 命令通道已在契约里，大概率不破 850 维。单独立项。
6. 统一 v1 的墙角度=固定 90°（mount clip 目前只有垂直墙版本）。分角度出货组合（75°/85° 用不同 ckpt）是否纳入统一策略、mount clip 是否按角度再生，待定。

## 11. 参考

- LadderMan: *Learning Humanoid Perceptive Ladder Climbing*, arXiv 2606.05873。本仓 `docs/reference/LadderMan/`（含精读笔记）。取用：Eq 3 退火 KL、多老师路由、w/o RL 消融、接触丢失终止、非对称 tracking（Eq 1）、宽盆接触奖励（Eq 2, τ=0.3）。
- HOVER: *Versatile Neural Whole-Body Controller for Humanoid Robots*, arXiv 2410.21229 (ICRA 2025)。取用：条件化统一策略存在性证明、运行中模式切换、mask/命令条件≈我们的 motion_command。
- Ji et al.: *Concurrent Training of a Control Policy and a State Estimator for Dynamic and Robust Legged Locomotion*, RA-L 2022。取用：阶段 3 合跑形式。
- 本项目：RMA Route-B 全链（commit 8e16846 起）、BC 天花板 0.835 记录、v4 90° 战役、mount v1–v4 点火史（`climb_transition/config/phybot_c2/env_cfgs.py` 注释）。
