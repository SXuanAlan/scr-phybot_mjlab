# 分角度全链管线：六阶 teacher 课程 + 三角度带"地面→墙全程"特权学生 + 三对分角度部署物

> 状态：已批准（2026-08-11）；阶段一（代码全量落地）当日实施，状态见文末
> 关系：本文把 [`2026-08-10_unified_mount_climb_policy_plan.md`](2026-08-10_unified_mount_climb_policy_plan.md) 的统一策略配方从单 90° 升级为三角度带矩阵（其开放问题 6 在此关闭；其 §6 配方被逐带实例化引用）
> 已定决策（用户拍板）：
> 1. teacher 课程 = 45, 75±12, 80±5, 85±3, 88±2, 90（六阶）
> 2. transition（mount）motion 只做三个角度带：80±5, 88±2, 90
> 3. 每带各训一个"地面→墙全程"统一特权学生（按段 KL 配方沿用 docs/plans/2026-08-10_unified_mount_climb_policy_plan.md §6）
> 4. **部署形态 = 三个分角度 deploy policy**（各自 RMA 压缩，无跨角度合并；C++ 按实测墙角选权重对）
> 5. **80°/88° mount specialist = 从 90° specialist（v4/v5 血统）短程 finetune**

## Context

08-10 的统一 mount+climb 方案原本只覆盖 90°。本计划把它升级为三角度带矩阵：每带产出一对部署物 (π_b, φ_b)，覆盖站立→mount→周期攀爬全程。方案 C（双策略切换）仍是每带的保底。两个代码探索已完成，关键结论：①摆墙是运行时旋转、命令表天然角度无关（利好）；②mount clip 的地板锚定前奏使纯旋转复用不可行，88° 是 5cm 级微扰、80° 是 27cm 级新 reach 问题（决定 L-B 设计）；③生成器与 validator 均硬编码垂直墙，需角度参数化。

## 管线总览（六层）

```
L-A  cycle 带教师       六阶课程跑 per-unit teacher 线，沿途铸 T_cyc(80±5)/T_cyc(88±2)/T_cyc(90)
L-B  分角度 mount clip   生成器/validator 角度参数化；带内离散 clip 族+小角旋转补隙
L-C  分角度 mount 老师   M(90)=v4/v5 收尾；M(88)/M(80) 从 M(90) 短程 finetune
L-D  分角度统一特权学生  U(b)：warm-start T_cyc(b)+按段 KL{M(b),T_cyc(b)}+接缝淡出带；90 探路
L-E  分角度 RMA 压缩     每带合跑(φ 回归+π PPO 并发)→ 出货对 (π_b,φ_b) ×3
L-F  部署工件           C++ 按墙角选带；每带 LUT=mount表‖cycle表；golden replay diff ×3
```

任何带在任何层卡住：该带回退方案 C（M(b)+T_cyc(b) 双策略）出货，不阻塞其他带。

## L-A cycle 带教师

- 现行 `WALL_ANGLE_STAGES`（`climb_wall/mdp/curriculums.py:149-155`）已是五阶（45@0 / 75±12@1500 / 85±3@3550 / 88±2@4200 / 90@4800）。**新增 80±5 = 插入 `{"iter": N, "angle_deg": 80.0, "jitter_deg": 5.0}` 并重排后续 iter**。doctrine 约束：tilt_peel 退火在首陡档前完成、档边界距退火步 ≥300 iter；80±5 池(75-85)与邻档重叠合规。
- **教师血统**：现役 = per-unit 计费 from-0 线（perunit-v5/v6、ign45*，已打到 45/75 段）。六阶课程即该线的续跑：resume 进新 wall_angle 栈（先重考当前段），跑 80±5→85±3→88±2→90，各带段末铸带教师并重考（match/逐爪 occ 门沿用 v4 标准）。带内 soak/pin 旋钮 = `FROM0_ANGLE_STAGES` JSON 环境变量（`train_climb_from0.py:31-44`）。
- 产出：三个冻结特权带教师。

**代码落点**：`curriculums.py:149-155`（加档）；`curriculums.py:184-201` `pin_curricula_to_final_stage` 丢 jitter → 加"带 pin"变体（分带 distill/eval 用），顺修 `env_cfgs.py:858-861` 失实 docstring；`periodic_eval.sh:25-27` probe_angles 加 80；raster 角度是位置参数（`engage_raster.py:217`）无需新 token；测试 `test_climb_task_config.py:317-326,346-350`、`test_deploy_variants.py:142`；文档 `climb_env_cfg_notes.md:574-620` doctrine 表、`plot_curriculum.py:50-51`。

## L-B 分角度 mount clip（供给设计 + 工具角度化）

**供给方案：带内离散 clip 族 + ≤2° 运行时旋转补隙。**
- 90：现役 clip（1 条）。88±2：native-88 clip 1 条，±2° 用旋转补（地板帧误差 ~1cm，与既有 spawn 噪声同量级）。80 带：native {80, 84} 两条。
- **实测修订（2026-08-11 生成报告）：a76 物理不可行**——站立两步 mount 的平衡边界 ≈80°（默认参数 CoM 越趾线 99mm，参数组合拳拉到 −55mm 仍无平衡解）。故 80 带的 **mount 训练窗钳到 [78, 85]**（wall_angle 阶段 81.5±3.5），cycle 带教师照常训全池 75-85；**<78° 墙的 mount 属未来"斜板专用编排"立项**（贴墙倒入/脚先上板，结构级改动）。
- 原理：native clip 以"去旋转到 canonical 帧"存储，运行时摆墙机制原样复用；旋转角=采样角−native 角，限 |Δ|≤2°，地板段误差 0.23·sin(2°)≈8mm 可接受。raster 显示带缘弱则加密网格。
- 生成留档：a88 全帧审计 PASS 但 worst anchor 0.9998（0.0% 薄余量，binding=右摆尾窗）——a88 若训弱，第一旋钮=`--rake-right-s 0.6`；a84/a80 用 `--standoff 0.19 --place-bulge -0.015`（余量 1.3%/1.8%）；全族 `--raise-s 3.2`、780 帧、handoff delta 4.2e-8。

**生成器角度化**（`gen_mount_clip.py`，探索报告 Option A 清单）：
- 加 `--wall-angle-deg`；墙面从两个标量（`_WALL_SURFACE_X=1.5`/`_PAD_TOUCH_X`）改为过基线 `(x=1.5,z=0)` 的斜平面（n̂=Rot_y(90−a)·(−x̂)）。触点：`:270` 站立 base x（改垂距 standoff）、`:404-410/:430-435/:1135-1139` 趾抵墙、`:1158` 墙净空扫描、`:429/:667/:677` 世界轴权重向量（语义=法向/侧向/沿墙）。
- state-1 快照绕基线预旋转（`:248-265`：`state1_anchor/state1_pad_rot/final_qpos` 乘 R_y(90−a)，锚梯 `[0,0,dz]`→`R·[0,0,dz]`）；`hand_z/hand_z_low` 改沿墙坐标。
- **80° 需重探 reach frontier**（1.47 天花板分析失效：手锚外移 26.6cm、侧举编排的肘净空问题变号）——这是 L-B 最大风险项，预算按"编排参数重调"留；88° 为 5.3cm 级微扰，现编排大概率直接可行。
- 地板侧（`ground_z :302-317`、CoM 支撑裕度 `:1173-1184`）保持世界 z，不改。
- clip 元数据加 `wall_angle` 键（现无，`:1186-1209`）。

**Validator 角度化**（`validate_climb_reference.py`）：`:73` 法向常数改角度推导；`:305-319` engage 基改斜面基；`:802-823` x 阈值接触判定改斜面有符号距离；读 clip 的 `wall_angle` 键。地板支路不变。每条 native clip 逐帧审计绿 + 合同测试。

**运行侧**（`climb_transition/`）：
- `MountMotionCommand` 加 clip 族支持（加载 K 条、按采样角选最近 native、逐 env 帧表/engage 表）——现单 clip 单路径（`env_cfgs.py:47,:135`），无此机制，属新造但范围可控。
- 分带 transition env 变体：撤销 `env_cfgs.py:165` 的 wall_angle pop，改装"带内网格采样+小角旋转"的静态角度项；站立 init pose `(1.27,0,0.726)` 按角度参数化。
- spawn 散布保持世界 x/y/yaw（站立机器人在水平地板上，散布语义=落地投递误差，与墙倾角无关；代码注释更新即可）。
- 注册三个带任务 id（镜像现有 config/__init__ 模式）。

## L-C 分角度 mount specialist（**训练首发层**）

**首发实验：M(90) 宽盆 σ 疗法**（用户指定训练从这里开始——climb 已被验证可训出，mount 尚未学成过）：

- 现状：v4 双旋钮（挂手窗+press 加价）已打通"按压→咬合→按日程落锚"，latch 事件率 7 倍、landing 项目史首次破零，但 landing 量级仍 0.0001-2、未到 handoff；病根之一是窄核奖励沙漠（`motion_swing_approach` std=0.05，15cm 外无梯度；v4 注释实锤"narrow exp kernels pay ~0 beyond ~1 rad"）。
- 实验：在 transition 包内加**双尺度接触吸引**（LadderMan contact reward τ=0.3 等效 σ≈0.55m 的启发）——远端 tow σ≈0.3–0.5m 新增项 + 现有近端 funnel（std 0.05/0.03）保留不动；从 v4/model_5850 续训（注意 press 退坡课表按进程内迭代计，重启需手动压首档权重）。
- 观测车道：landing 收入量级、LH_occ、press/latch_hz、mount_handoff_ready；成功判据=landing 上量并推进到 handoff 帧，mount raster 出基线。
- 备用第二旋钮：非对称臂 tracking（臂侧 w×0.5、std×2，腿/root 严格）——σ 疗法单独不够时叠加。

M(88)、M(80)：M(90) 学成后从其终版热启动短程 finetune（数百~千 iter 量级），换各自 clip 族；spawn-bias 窗/press-bootstrap 机制原样复用（`resolve_segment_window` 按段标签解析，免疫重定时，跨 clip 直接可用）。80° 若 finetune 不收敛（编排差异过大），回退为短点火战。考卷：各带 mount raster。定位不变：只当老师、永不部署。

## L-D 分角度统一特权学生

沿用 2026-08-10 文档 §6 配方逐带实例化：

- U(b) warm-start 自 T_cyc(b)；PPO（climb 平价奖励一套，§6.6）+ λ·KL 按段路由（mount 段←M(b)，cycle 段←T_cyc(b)）+ 接缝 handoff±T 淡出带；三开局混采（整链/墙上 RSI/接缝 RSI ≈ 0.3/0.5/0.2）。
- 角度供给：整链与接缝 episode 用该带 mount clip 族网格角（±2° 旋转）；墙上 RSI cycle episode 用连续带内 jitter（旋转精确）。
- **U(90) 先行探路**：λ 日程、混采比例、淡出带宽 T 在 90 调通，80/88 继承常数并行。
- 硬门（每带）：带内 cycle 考卷不掉（match ≥0.95 双绿零摔+逐爪 occ 门）；mount raster ≥ M(b)；链式全程 ≥ 方案 C 基线（链式考卷先行）。
- engage 表拼接（mount acyclic 前奏‖cycle 环形后续）：terminal clamp 改"滚入 cycle 表"，反幻影哨兵重写+合同测试（2026-08-10 文档 §6.6 已记）。

## L-E 分角度 RMA 压缩（三对部署物）

- 每带独立合跑：同批 rollout 并发训 π_b（PPO finetune）与 φ_b（特权真值回归），scheduled sampling 真值→ê 退火，前 N iter 可冻 π。φ_b 混合分布 = mount 段（投递偏差）+ cycle 段（DR/挂爪态）。ê 9 维先不动，考卷说话。
- estfreeze 常数分带重标（含 mount 段行为）。90 先压缩探路，80/88 复用配方。
- 考卷（每带）：在环 ê 全套 ≥0.96 零摔 + 恢复 2/2 + 超真值喂养对照。

## L-F 部署工件

- C++ ClimbTask 单次运行装一对权重，按实测墙角选带（80±5/88±2/90 覆盖 75°~90° 作业面）。
- 每带 LUT = 该带**中心角** mount 表 ‖ cycle 表拼接（`export_policy.py:144-180` 的命令向量本身角度无关，机制照用；仅 mount 段 schedule 因 clip 而异）。PD 表烙 meta 照旧。
- golden 离线 replay diff ×3；仓库外硬件四件套不变。

## 执行顺序（用户指定：**代码全写完，再开训；训练首发 L-C**）

**阶段一：代码全量落地（不开训练）**

写完"汇总工作项"表的全部代码：宽盆双尺度奖励项（首发实验用）、六阶课程表+带 pin、生成器/validator 角度化、88/80 clip 族生成与审计、MountMotionCommand clip 族支持、分带任务注册、engage 表拼接（滚入式）、KL-PPO 蒸馏训练器、合跑压缩训练器、考卷/eval 编排。完成门：pytest 合同测试群全绿 + **90° 逐位回归**（角度化后的生成器/validator 在 90° 复现现役 clip 与审计结果逐位一致）+ 20-iter 各任务冒烟。

**阶段二：训练队列**

```
T0: ★ M(90) 宽盆 σ 实验（v4/5850 续训,L-C 首发）
T1: M(90) 学成 → M(88)/M(80) finetune（并行）
    ∥ 带教师六阶课程续跑（per-unit 线,另卡,与 L-C 互不阻塞）
T2: 90 链式考卷（方案C基线）→ U(90) 探路（λ/混采/淡出带调通）
T3: U(80)/U(88) 继承常数并行 ∥ (π_90,φ_90) 压缩探路
T4: (π_80,φ_80)/(π_88,φ_88) 压缩 → 三带 estfreeze 重标 → 三对出货
```

## 附：clip 角度化几何/存储契约（任务 3/4/6 共享，已定）

- **运行时旋转约定**（现役，`curriculums.wall_pose_for_angle`）：绕基线 (x=1.5, z=0) 的 Rot_y(90−a)。沿墙高 h 的表面点 = (1.5 + h·sin(90−a), y, h·cos(90−a))；外法向 n̂ = (−sin a, 0, cos a)；沿墙上方向 û = (cos a, 0, sin a)。a=90 时全为恒等（运行时有 bit-exact 快路径）。
- **native clip 存储**：生成器在真实世界（真重力、水平地板、a 角斜墙）解算，落盘前把所有世界系数据（root pos/quat、EE/pad/foot 位姿、anchors）**绕基线做 R⁻¹ 去旋转到 canonical 帧**存储；运行时把墙摆到采样角后经现有 `_to_world_*` 机制原样还原。采样角≠native 角时残差旋转 |Δ|≤2° 自动流经同一机制（地板帧误差 ≈0.23·sin Δ ≤ 8mm）。
- **a=90 快路径**：生成器显式跳过变换（不做恒等矩阵乘法），保证与现役 clip 逐位一致。
- **元数据**：新增 `wall_angle: <native 角,度>` 键；loader 缺键按 90.0 兼容旧 clip；其余键名/形状全部不变。现役 90 clip 资产不重写（回归验证生成到临时路径 diff）。
- **站立 standoff 语义**：改为对斜面的**垂距** 0.23（基座高 0.726 处解 x = 1.5 − (0.23 − 0.726·cos a)/sin a；a=90 还原 1.27）。
- **世界系保持不变的量**：地板 z、ground_z、CoM-over-support 裕度、站立段基座 upright（重力语义）；**墙帧化的量**：趾抵墙目标、墙净空、任务轴权重（法向/侧向/沿墙）、state-1 快照与锚梯（预乘 R）。

## 汇总工作项（按包）

| 区域 | 工作项 |
|---|---|
| `climb_wall/mdp/curriculums.py` | 六阶表；带 pin 变体 |
| `scripts/gen_mount_clip.py` + `gen_climb_clip.py` 常数 | `--wall-angle-deg` 斜面化（触点清单见 L-B） |
| `scripts/validate_climb_reference.py` | 角度化（法向/基/接触判定/wall_angle 键） |
| `climb_transition/mdp/motion.py` | clip 族支持；分带角度项；engage 表拼接（滚入式） |
| `climb_transition/config/` | 三个带任务 id；init pose/角度参数化 |
| 训练脚本 | 按带 resume/finetune 启动（FROM0_ANGLE_STAGES 模式复用）；U/压缩阶段的 KL-PPO 与合跑实现（2026-08-10 文档 §6.2/§7） |
| `scripts/periodic_eval.sh`、raster | 80 探针角；分带考卷编排 |
| 测试 | 冻结课程表断言更新；clip 族/角度化合同测试（镜像 test_mount_clip 模式）；transition 分带配置测试 |
| 文档 | `climb_env_cfg_notes.md` doctrine 表；2026-08-10 计划书升级为三带矩阵版（开放问题 6 关闭、出货对改三对） |

## 验证

- 每层考卷即验证（各层"硬门"）；三对部署物各过：带内 cycle 考卷、mount raster、链式全程、恢复考、estfreeze 考、在环 ê 全套。
- 新代码走 pytest 合同测试群扩展；clip 逐条 validator 审计绿；90° 路径全程回归（现役资产不许被角度化改动扰动——生成器/validator 改造后先在 90° 复现现役 clip 逐位一致）。

## 开放问题

1. 80° 编排是否需大改（reach frontier 重探后定；最大不确定项）。
2. 带内网格密度（默认 4°+旋转补隙；raster 带缘弱则加密）。
3. U(80/88) 可否从 U(90) 热启动省时（90 探路后定；默认从 T_cyc(b)）。
4. 80±5 带教师铸造的 resume 源节点（per-unit 线跑到哪算哪，考卷说话）。
5. descent 不进本轮矩阵，单独立项。

## 阶段一实施状态（2026-08-11）

已落地并测试绿：

- 宽盆双尺度奖励（`motion_swing_approach_far`，仅手、σ=0.4）+ 非对称臂 tracking 旋钮（`asym_arm_tracking`，默认关）——`climb_transition/mdp/rewards.py`
- 六阶 `WALL_ANGLE_STAGES`（80±5@3300 插入，后续档 +350）+ `pin_curricula_to_band`（保 jitter 带 pin）+ doctrine 表/探针角/冻结断言全套更新
- 生成器/validator 角度化（`--wall-angle-deg` 斜面、`wall_angle` 元数据键、canonical 去旋转存储），90° 逐位回归双硬门通过
- `MountMotionCommand` clip 族（时间轴拼接 + 每 env 窗口 + 最近 native 选择 + 局部相位/时长）；分带任务注册 `…-Band80-…`/`…-Band88-…`
- 缝合器 `stitch_mount_into_cycle`（跨缝 lookahead 表 + 连续性断言 + 滚入式时钟）+ `UnifiedMountClimbCommand`（三开局混采/段局部相位/`in_cycle_region` 路由信号）+ `Mjlab-Climb-Unified-Phybot-C2` 注册
- schedule-aware climb_velocity（mount 段诚实读零，过缝按 cycle 采样）
- KL-PPO 蒸馏器 `scr_phybot_mjab/rl/kl_distill.py`（按段路由/λ 退火/接缝淡出带/kl|mse 双模）+ `scripts/train_unified_kl.py`；真 clip 机制冒烟通过（handoff 帧 750、λ 退火在走、蒸馏损失非退化）

- 合跑压缩训练器 `scr_phybot_mjab/rl/concurrent_est.py` + `scripts/train_concurrent_est.py`（φ 与现役工件逐键同构、scheduled sampling 真值→ê、冻结窗；raster `estimator=`/estfreeze/C++ packer 可直接消费产物）——冒烟：φ 损失 0.55→0.16 下降、p_truth 退火、冻结窗生效
- 链式考卷 `scripts/chained_mount_exam.py`（27 散布格矢量化、`mount_handoff_ready` 触发切换、`--estimator` 部署形态分支、JSON 报告）
- **阶段一完成门通过**：全仓 pytest 164 过 / 1 失败=预存 pad_teleop（干净树复现，与本工作无关）；90° 逐位回归双门；统一 KL/合跑/链考三冒烟；Band80 真资产 E2E 探针

**分带考卷调用约定**：cycle 带考=engage_raster 位置角参数逐角（带缘+中心）；mount 带考=`chained_mount_exam.py --task Mjlab-Climb-Transition-Band{80,88}-Phybot-C2`；周期评估探针角已含 80（periodic_eval.sh）。

**训练队列（阶段二）**：T0 首发=M(90) 宽盆 σ 实验（`transition-v4-latchwindow/model_5850` 续训，press 退坡按 v4 已耗 750 iter 平移）；T1=带教师六阶续跑（resume 前先新栈重考当前段）∥ M(88)/M(80) finetune；此后按计划表推进。

## 2026-08-15 附记：clip 族已切 slow+staggered（分步落锚），band 审计状态

主 clip（90°）与 a84 全帧 PASS；**a88 红 1/1260 帧**（right_foot_mount 趾轴窗 anchor util 1.0016，老 fast 版余量本就 0.9998，`--rake-right-s 0.6`/`--lateral-weight-shift 0.02` 两旋钮均无效）；**a80 红 5-7/1260 帧**（hand_place 站立平衡，80° 本就是双手够顶的平衡边界，08-11 的 `--place-bulge -0.015` 配方在新编排下不再奏效）。band 任务未开训，先记账不阻塞。**看好的修复路线：把"分步"从落锚扩展到举臂**——左臂先举先挂、右臂留在体侧（真实降低 CoM），对 80 带是结构性收益，甚至可能解锁 a76；属 band 阶段的编排工作。
