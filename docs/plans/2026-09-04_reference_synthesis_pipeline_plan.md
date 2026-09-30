# 参考合成流水线（服务 motion tracking RL）：契约 → 逐帧字典序 IK → 整段运动学优化 → 出生稳定性门 → 参考族 → MotionGraphCommand

> 状态：计划书待审（2026-09-04）；代码未动。
> 关系：接续 [`2026-08-11_band_matrix_pipeline_plan.md`](2026-08-11_band_matrix_pipeline_plan.md) 的 L-B 层（分角度 mount clip），把"参考从哪来"升级为一个独立、可发表的方法；论文方法节草稿见 `docs/reference_synthesis_method.tex`（Sec 4.1-4.5 为本计划步骤二、三的目标形式）。
> 已定决策（用户 09-04 拍板）：
> 1. **优化阶段不建模接触。** 接触只以运动学事实进优化器：焊接位姿等式、万向节有界变量、关节硬带、rake 帧路径等式、闭合/交接等式、墙面净空。力学（Wang 包络、招募斜坡、摩擦锥、力矩限）只出现在证书里。
> 2. **参考族 = motion graph。** 节点 = 共享状态（交接状态、各档 cycle 第 0 帧），边 = 过证书的参考。"符合数据集"定义为图上任意路径加有界减速；**否决逐帧最近邻打分**（拼接怪物 / 停滞雕像 / 时刻表失真）。
> 3. **选边由 RL 学。** 节点处策略输出选择动作，任务项（爬升率、撕脱、占用税）驱动选择；部署不需要外部选择器。
> 4. 论文英文；计划与讲解中文；每步动手前审批。

---

## 0 现状与事实（09-04 核实）

**现役生成链**：`scripts/gen_climb_clip.py`（cycle，单层加权 DLS，焊接权重 1000/500）、`scripts/gen_mount_clip.py`（mount，同一 IK 引擎，5.7k 行，含 `--weld-hard` / `--weld-gimbal-deg` 两个未启用的 monkeypatch）、`scripts/validate_climb_reference.py`（静力存在性审计 + 时序审计）。
**现役消费端**：`climb_wall/mdp/motion.py::ClimbMotionCommand`（相位钟、RSI、R4 reseat hold、墙角旋转）；`climb_transition80/mdp/motion.py::TransitionMotionCommand`（终止保持、acyclic 表、FK 身体表 `build_fk_tables`、自适应 RSI、birth mask、rake 教室窗）。
**实测缺陷**（本计划要修的）：

| 缺陷 | 数值 | 出处 |
|---|---|---|
| 焊着的垫在摆动期漂移 | left_foot_swing RF 3.1 mm，right_hand_swing 3.4 mm | paced clip 逐帧 `pad_positions − ee_anchor_positions` |
| 驻留段漂移≈0 是假象 | 每段开头 `anchors[ee] = ik.pad_pose(ee)` 重坐锚点 | gen_climb_clip 段首逻辑 |
| 万向节在 IK 里锁死 | `sel_all` 不含 `*_gimbal_*`，a80 v6g recipe 未开 `--weld-gimbal-deg` | gen_climb_clip / a80 recipe |
| 段边界加速度尖峰 | 基座 |a| 中位 0.017 g、p95 0.11 g、4/224 帧 >0.2 g（均在 left_foot_swing 内） | paced clip 差分 |
| 闭合靠尾部混合 | 末 40% 帧线性混合到 state1+h | gen_climb_clip `closure` |
| 时长手调、单 clip | paced 三版血统靠人试 | docs/climb_env_cfg_notes.md |

**环境**：`qpsolvers 4.11`（`proxqp 0.7.2` 稀疏/稠密后端、`daqp` 可用；`osqp` 无）、`mink` 可用、`scipy 1.17`、`xelatex` 可用；`cma` **未装**。
**实机限值**：`PHYBOT_C2_DEPLOY_TORQUE_LIMITS` 存在；**没有关节限速常量**（步骤三需补）。

---

## 1 总览

```
步骤一  契约 schema v2 + 交接不变量        产出：所有生成器统一输出格式；(mount, cycle) 对的拼接测试
步骤二  逐帧字典序 IK                      产出：hard welds + 有界万向节的 gen_climb_clip；legacy 路径逐位保留（消融基线）
步骤三  整段运动学轨迹优化（KTO）           产出：scripts/kto_climb_clip.py；速度/加速度硬界、闭合改约束
步骤四  证书扩展：RSI 出生稳定性门          产出：scripts/reference_certificate.py 一键四关 → certificate.json
步骤五  参考族：ψ 搜索 + 铺档 + 镜像 + 图   产出：assets/motions/graph/<band>/…npy + motion_graph.json
步骤六  MotionGraphCommand（RL 侧）        产出：climb_transition80/mdp/motion_graph.py + 选择动作 + 小图任务
```

依赖：二 ← 一；三 ← 二；四独立（可与二并行）；五 ← 三、四；六 ← 一、四（小图可先用现役 paced + 一条新证书 clip）。

每步的验收门写在各节末尾；**任何一步不过门不进入下一步**。

---

## 2 步骤一：契约 schema v2 + 交接不变量

### 2.1 schema v2（在现有 `.npy` dict 上只增不改）

| 新键 | 形状/类型 | 含义 |
|---|---|---|
| `ee_gimbal_angles` | (T,4,3) float32 | 焊着垫的肢体相对垫的万向节角（放开垫恒 0）；legacy 生成写 0 |
| `ik_mode` | str | `"weighted"` / `"lexqp"` / `"kto"` |
| `gimbal_travel_deg` | float32 | 生成时的 γmax（legacy 0） |
| `wall_angle` | float32 | cycle 也写（90.0）；mount 已有 |
| `is_cyclic` | uint8 | cycle 1 / mount 0 |
| `handoff_frame` | int32 | 交接帧索引（mount = handoff_hold 首帧；cycle = 0） |
| `handoff_state_id` | str | 规范交接状态标识，如 `"a80/hz1.52"` |
| `recipe` | str | 生成命令行全文 |
| `velocity_bounds` | (J,) float32 | 生成时用的关节限速（步骤三起） |

**证书 sidecar**：同名 `.certificate.json`：静力最坏利用率、最小时序裕度、最小限位裕度、最大万向节使用角、reference-hold 结果、出生稳定性门结果、生成 recipe、git commit。步骤四产出，步骤一先定格式并留空字段。

**身体表不入 schema**：transition80 的 `build_fk_tables` 已在加载时用 MuJoCo FK 算全身位姿，clip 只带 qpos 即可；改动仅是 `base_qpos` 的万向节槽位改读 `ee_gimbal_angles`（若有），保证审计与训练看到的垫姿态一致。

### 2.2 交接不变量

- 定义**规范交接状态** S(b) ：每个墙角带 b 一个（四垫全焊、手锚点高度、基座离墙距离、关节角），存于 `assets/motions/handoff_states.json`（由 cycle 站姿 FK 生成，mount 以 `dz` 平移）。
- mount 生成器：终点 = S(b) 平移 dz，作为**硬等式**（步骤三）/ 尾部混合（步骤二过渡期沿用现状）。
- cycle 生成器：起点 = S(b)，闭合 = S(b) ⊕ h·ẑ。
- 测试推广：`tests/microspine/test_mount_clip.py::test_handoff_frame_matches_climb_state1` 推广为参数化测试，对图里每条 (mount, cycle) 边对、每对同档 cycle 首尾都跑。

### 2.3 改动文件

- `scripts/gen_climb_clip.py`：输出段加新键；`main` 记录 recipe。
- `scripts/gen_mount_clip.py`：同上；`handoff_frame`/`handoff_state_id`。
- `src/.../climb_wall/mdp/motion.py`：`from_dict` 保持忽略未知键（已是）；可选读 `ee_gimbal_angles`。
- `src/.../climb_transition80/mdp/motion.py::build_fk_tables`：万向节槽位读 clip。
- `scripts/validate_climb_reference.py::_set_motion_frame`：写万向节角。
- 新 `tests/microspine/test_motion_schema_v2.py`：round-trip、legacy 兼容、交接参数化测试。

### 2.4 验收门

- 现役三条 clip（paced、a80、a88/a84）加载与训练冒烟（`Mjlab-Climb-Wall-Phybot-C2` 与 `Mjlab-Climb-Transition80-Phybot-C2` 各 2 iter）不变。
- legacy 模式重生成 paced：`joint_positions` 逐位相同（cal4 时已证明生成不依赖物理参数，本步只加键）。
- pytest `tests/microspine` 全绿。

---

## 3 步骤二：逐帧字典序 IK（hard welds + 有界万向节）

### 3.1 数学（每帧每次迭代）

变量 `δq ∈ R^39`（基座 6 + 关节 21 + 万向节 12）。

```
第 0 级 盒约束   θ_lo + m ≤ θ + δθ ≤ θ_hi − m，m = 1% 量程（对齐策略 0.98 钳位）
               焊着垫：|γ + δγ| ≤ γmax（默认 2°，挡块 3°）；放开垫：γ + δγ = 0；|δq|∞ ≤ Δ
第 1 级 硬等式   焊接 J₁ δq = e₁（每焊着垫 6 行，作用在 pad_center 站点，雅可比含万向节列）
               净空 G δq ≤ g（放开垫球体与趾：x ≤ x_w − r）【可选，默认开】
第 2 级 软目标   摆动垫、趾撑、基座意图、姿态参考、限位裕度弹簧（10% 带）、万向节回零 w_γ=3、阻尼 λ²=0.5

步 1  δq₁ = argmin ½‖J₁δq − e₁‖²   s.t. 盒、净空        （proxqp / daqp）
步 2  δq₂ = argmin f₂(δq)          s.t. Uᵣᵀ J₁ δq = Uᵣᵀ J₁ δq₁、盒、净空
收敛  焊接 pos < 0.15 mm、rot < 2 mrad、|δq₂|∞ < 1e-4；上限 100 次
判决  收敛后残差超阈 → 该帧 infeasible，打印段名/帧号/活跃界
```

### 3.2 实现

- `_ClimbIK.__init__(hard_welds=False, gimbal_travel_deg=0.0, limit_hard_margin_frac=0.01, wall_clearance_hard=True, qp_solver="proxqp")`；收集 `gimbal_dofs[ee]`、硬限位数组；`PHYBOT_C2_PAD_GIMBAL=0` 时 `gimbal_travel_deg` 必须为 0。
- `dof_selection`：硬模式追加 12 个万向节 DoF（放开垫靠 bounds 钉 0，`sel_all` 单一）。
- `solve_frame` 拆 `_assemble_rows` + `_step_legacy`（现有代码原样搬家）+ `_step_lexqp`；哪个垫焊着由任务 `hard` 标记经 `sid→ee` 反查，调用方签名不变。`_Residual` 增 `hard_pos/hard_rot/feasible/active_bounds`。
- `_weld_task`：`hard = ik.hard_welds`。步后 `np.clip` 保留为断言。
- `generate_clip`/`main`：新旋钮 `--hard-welds --gimbal-travel-deg 2.0 --limit-hard-margin-frac 0.01 --no-wall-clearance-hard --qp-solver --strict`；逐帧记录 `γ`；报告新增：硬残差、每垫万向节最大使用角、活跃界计数、不可行帧汇总；输出 schema v2。**默认全关 = 旧行为**。
- `scripts/gen_mount_clip.py`：保留旧 `--weld-hard/--weld-gimbal-deg` monkeypatch 作 legacy；新增 `--lexqp` 直接透传 `_ClimbIK(...)`，不动其它逻辑。
- `scripts/validate_climb_reference.py`：写万向节角（步骤一已做）；报告打印 `ik_mode`。

### 3.3 验收门

- 新测试 `tests/microspine/test_climb_clip_lexqp.py`：10 fps 一周期小 clip，硬残差 < 阈、|γ| ≤ γmax、关节在硬带内、摆动垫 γ ≡ 0、无穿墙；`_step_legacy` 与旧实现在 20 帧上逐位一致。
- 用 paced 时序重生成 `climb_cycle_moment_safe_lexqp.npy`：焊接漂移 3.4 mm → < 0.15 mm；静力审计 PASS（利用率报告并与 0.877 对比）；时序审计 PASS；reference-hold PASS；与 paced 的逐关节 max diff 统计入报告。
- 155+ 测试全绿。

---

## 4 步骤三：整段运动学轨迹优化（KTO）

### 4.1 数学

```
变量  Q = (q_0 … q_{N−1})，每帧 39 维
第 0 级 盒     关节硬带；万向节（按 σ_t）；速度 |q_{t+1} − q_t| ≤ v_max/f_s；加速度 |q_{t+1} − 2q_t + q_{t−1}| ≤ a_max/f_s²
第 1 级 等式   焊接 FK_h(q_t) = A_{h,t}（σ_{h,t}=1）；rake 帧 p_h(q_t) = p*_h(t)、R_h = R^a_h；
              闭合 q_N = q_0 ⊕ h·ẑ（cycle）/ q_N = S(b) ⊕ dz（mount）
       不等式 净空
第 2 级 软     Σ_t [ 意图项（摆动弧/趾/基座/姿态）+ w_a‖q_{t+1} − 2q_t + q_{t−1}‖² + w_γ‖γ_t‖² ]
求解   SQP：当前 Q 处线性化 FK → 稀疏 QP（proxqp sparse），两步字典序；J₁ 逐帧分块对角 + 一个闭合块；信赖域；逐帧 IK 热启动
判决   不可行 → 报告冲突段与约束类型（速度界 vs 焊接 = 该段太短）
```

### 4.2 实现

- 新 `scripts/kto_climb_clip.py`：复用 `gen_climb_clip` 的编排与任务装配（意图目标、σ、锚点、rake 帧集合），只替换求解层；输出 schema v2，`ik_mode="kto"`。
- 新常量 `PHYBOT_C2_DEPLOY_VELOCITY_LIMITS`（需用户提供电机规格，或先取 XML/经验值 × 0.7 安全系数，标注来源）。
- 闭合改约束后删除 cycle 的尾部混合（KTO 路径）；逐帧路径保持混合。

### 4.3 验收门

- 基座 |a| 无 >0.2 g 帧；关节速度 ≤ v_max、加速度 ≤ a_max（报告）；闭合误差 0（逐位）。
- 三关证书 PASS；静力最坏利用率不劣于逐帧版 +0.02。
- 与逐帧版逐关节差异统计；渲染视频人工核（`scripts/render_climb_video.py`）。

---

## 5 步骤四：证书扩展 = RSI 出生稳定性门

- 新 `scripts/reference_certificate.py <clip>`：串行跑四关并写 `.certificate.json`：
  1. 静力审计（`validate_climb_reference.py` 主体，import 复用）；
  2. 时序审计（同上）；
  3. reference-hold 回放（`validate_microspine_constraint.py` reference-hold 模式）；
  4. **出生稳定性门**：复用 `scripts/climb_cycle_passive_gate.py` / `mount_clip_passive_gate.py` 的规则（每 stride 帧出生 128 env、deploy PD 被动保持 40 步、guard 25 步后：种子丢失 ≤5%、万向节 p90 ≤2.5°、零墙碰/零摔），统一成一个函数。
- 阈值沿用 passive gate 默认；`--strict` 下万向节超限为 FAIL。
- 验收：现役 paced、a80 各出一份证书（作为基线数字进论文）；步骤二/三产物证书四关全绿。

---

## 6 步骤五：参考族（ψ 搜索 + 铺档 + 镜像 + 图）

### 6.1 编排参数与目标

```
ψ = (各段时长 N_k, 步高 h, 手/脚鼓包 b, 压墙 δ_press, 侧移 δ_lat, rake 长度与时长, γmax, 手锚点高度 z_h, 离墙距离 d_b, …)  ≈ 15 维
J(ψ) = max_t s*_t（静力最坏利用率）
       + 10·[不可行帧数] + 10·[时序审计 FAIL] + 5·max(0, 0.05 − 限位裕度) + 2·max(0, γ_use − γmax·0.8)
```

- 优化器：`pip install cma`（纯 Python，建议）；备选自带 (1+λ)-ES。种群 8-12，每次评估 = 生成（步骤三）+ 证书前两关（秒到分钟级），多进程并行；出生稳定性门只对入围解跑。
- 每墙角带、每速度档、每步高档各一支；mount 三种序列各一支。

### 6.2 镜像

关节符号映射表（roll/yaw 类取反、pitch 不变、左右互换）+ 时刻表左右互换 + y 翻转 + 趾撑表互换；镜像后同样过证书。

### 6.3 图

`assets/motions/graph/motion_graph.json`：节点（`handoff_state_id` + 状态向量）、边（clip 路径、起止节点、档位标签、证书摘要、`role: primary|augment`）。校验：每条边首尾帧与节点状态逐位一致（步骤一测试推广）。

### 6.4 验收门

- 族内每条边证书四关全绿；图连通；`test_motion_graph.py`（节点一致性、镜像对称性）。
- 首批规模：cycle 两档速度 × 一档步高 × 镜像；mount a80 cross 一条 + 镜像。

---

## 7 步骤六：MotionGraphCommand（RL 侧）

### 7.1 设计

- 新 `climb_transition80/mdp/motion_graph.py::MotionGraphCommand(TransitionMotionCommand)`：
  - 状态 `(edge_id, phase)` per env；所有边的表按最长补齐堆成 `(E, C_max, …)`；
  - **边内**：相位 +1；`reseat_hold` 为真时停（现有 R4 机制），连续停步上限 `max_hold_steps`；不允许跳帧；
  - **节点**：边末帧 = 节点；此时读取选择动作 `sel ∈ R^d`（d=2 或 3），映射到该节点出边中欧氏最近的 `ψ_edge` 摘要（各边预存一个 d 维摘要，如归一化的速度档、步高、镜像标志）；切边后相位归 0、`root_offset` 按边的位移累加；
  - RSI：均匀/自适应抽 `(node, edge, phase)`；自适应桶按 `(edge, bin)` 记失败率（复用 `adaptive_bin_probabilities`）。
- 新动作项 `GraphSelectActionCfg`（`mjlab ActionTerm`：`action_dim=d`，`process_actions` 存值，`apply_actions` 无操作）；PPO 高斯策略不改。
- 观测新增：`node_countdown`（到节点的归一化帧数）、`edge_summaries`（出边摘要，固定最大出度补零 + mask）；现有 `motion_command` 10 车道照旧。
- 奖励：现有跟踪项照旧；任务项复用 `engage_progress`（沿墙爬升率）、`anchor_rip_penalty`；新增 `pad_occupancy_tax`（小额）；选择靠这三项学。
- 任务注册：`Mjlab-Climb-Graph80-Phybot-C2`（与 transition80 平行，不改现有任务）。

### 7.2 验收门

- 机制冒烟：随机策略 2000 步，切边只在节点发生、相位单调、acyclic 表无幻影边沿、`root_offset` 连续（断言）。
- 训练冒烟 200 iter：奖励上行、无 NaN、切边分布可打印（策略在两档速度间的选择比例）。
- 全测试绿。

---

## 8 消融与论文实验映射

| 论文表 | 对照 | 指标 | 步骤 |
|---|---|---|---|
| IK 求解器 | weighted vs lexqp vs kto | 焊接漂移、限位裕度、万向节使用、不可行帧、加速度 p95 | 二、三 |
| 万向节 | γmax ∈ {0,1,2,2.5}° | mount 漂移与不可行帧 | 二 |
| 证书 | fast vs paced vs 搜索 ψ | 静力/时序裕度 → 下游 match/occ/爬升/摔 | 四、五 |
| 参考族 | 单 clip vs 图 | 泛化（速度/墙角）、选择比例 vs DR 容量 | 五、六 |
| 出生稳定性 | 门前后 | 出生撕脱率 | 四 |

---

## 9 风险与待用户拍板

1. **γmax 默认 2°**（可扫 1/2/2.5）。
2. **实机关节限速**：常量不存在；需你给电机规格，或先用经验值 × 0.7 并标注。
3. **`pip install cma`** 是否可接受；否则自带 ES。
4. **KTO 是否主路径**：时间紧可降级为"逐帧 IK + 加速度界 + 事后审计"，方法叙事不变。
5. **首个图规模**：两档速度 × 一 mount 序列 × 镜像。
6. **出生稳定性阈值**：沿用 passive gate 默认（loss ≤5%、gimbal p90 ≤2.5°）。
7. 步骤六与现有 transition80 训练线并行，不改现役任务。

---

## 10 顺序与工作量估计

| 步骤 | 估时 | 可并行 |
|---|---|---|
| 一 契约 | 0.5 天 | — |
| 二 逐帧字典序 IK | 1.5 天（含 paced 重生成与五关） | 四 |
| 三 KTO | 2-3 天 | — |
| 四 证书脚本 | 1 天 | 二 |
| 五 参考族 | 1 天搭 + 机器时间 | — |
| 六 MotionGraphCommand | 2 天（小图） | 五 |

每步完成即汇报（见 feedback：长任务分块汇报）；>5 min 的机器任务先预告。
