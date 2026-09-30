# 从 SONIC 数据集取攀爬动作 → IK 微调 → Tracking：可行性调研

> 文档状态：调研结论（2026-07-19，三路并行调研汇总：SONIC 数据集本体 / 重定向与 tracking 管线 / 本地资产盘点）
> 关系：服务于 [`climb_task_space_effect_wrench_tracking_plan.md`](climb_task_space_effect_wrench_tracking_plan.md)——mocap 是给其 **阶段 A 参考源的升级选项**，不推翻 A→B 分阶段结构与事件门控原则。

---

## 0. TL;DR

**可行，且比预想便宜一半**——因为重定向工作台 5 月份已经在本机建好了大头（`/home/spider/projects/retargeting/`：GMR 的 phybot fork 含 **C2 全套 IK 配置**、GVHMR 视频链路带权重、LAFAN1 全量 77 条 BVH、成品 `crawl_c2.pkl.npz`）。

关于 SONIC 本身，三个必须先说清的事实：

1. **SONIC 不是数据集**，是 NVIDIA GEAR 的 tracking 基础模型（arXiv 2511.07820）；它的公开训练数据叫 **BONES-SEED**（HF `bones-studio/seed`，142K 条 / 288h / 120fps Vicon，gated 许可，学术免费）。
2. SEED 里**确有 climb / ladder / stair 类片段**（反证：SONIC 官方过滤脚本的排除关键词列表），但被 SONIC 训练管线剔除为"对 G1 物理不可行"——所以**这些片段的 G1 CSV 重定向不可信，必须取人体侧 SOMA BVH 自己重定向到 C2**。SEED 恰好公开了人体侧源数据（SOMA BVH + 逐演员体型），这条路是通的。
3. "四肢贴平墙爬"的动作在所有公开 mocap 里都不存在；最近的两类是 **ladder climb**（对应 90° 阶段的四肢相序协调）与 **crawl**（对应 45° 阶段——我们课程是墙恒竖直、重力倾斜，45° 阶段的力学本质就是爬坡，且 crawl 的"手掌平压"语义比抓握更贴 microspine pad）。

**最重要的外部先例：LadderMan**（arXiv 2606.05873，2026-06，G1 感知爬梯）。其消融结论直接决定了 mocap 的正确用法：**纯 DeepMimic 时间索引跟踪换一个梯子几何就失败；泛化主力是"由环境几何自动生成的接触目标"，mocap 跟踪只供风格（上肢权重还要打折）**。这与我们 plan 文档"事件门控参考 + effect/α 分工"是同一结论的独立印证——所以 clip 应当被消费为**相位参数化的路径形状**灌进阶段 A-1 的移动路点，而不是整段时间硬跟。

真正需要新写的只有四件：SOMA→C2 keypoint 映射（配置级）、mink 接触重钉脚本（"IK 微调"本体，2–4 天）、GMR 输出→相位化路点/tracking npz 的转换器（各 0.5–1 天）、（若走 tracking 路线）phybot_c2 tracking config。**首版端到端 5–8 天**，其中不依赖 SEED 审批的部分今天就能开工。

---

## 1. SONIC / BONES-SEED 事实卡

| 项 | 事实 |
|---|---|
| 本体 | "SONIC: Supersizing Motion Tracking for Natural Humanoid Whole-Body Control"，NVIDIA GEAR（Zhengyi Luo、Tairan He、Jim Fan、Yuke Zhu 等），arXiv 2511.07820，42M 参数 tracking 基础模型，G1 上 50 条真实轨迹零样本 100% |
| 代码 | github.com/NVlabs/GR00T-WholeBodyControl（`gear_sonic` 训练栈 + `motionbricks`），Apache-2.0；权重 HF `nvidia/GEAR-SONIC` |
| 数据集 | **BONES-SEED**：HF `bones-studio/seed`，142,220 条动作（含镜像）/ ~288h @120fps，522 名演员（身高 145–199cm），Vicon 光学动捕，Bones Studio 自采（**非** AMASS/LAFAN1 转录） |
| 许可 | 自定义 bones-seed-license，HF auto-gated：学术机构或年收入 <100 万美元实体点击同意即可下载；商业授权另谈 |
| 表示 | 三份：SOMA Uniform BVH（统一骨架）、SOMA Proportional BVH（逐演员骨架 + 体型 npz）、**Unitree G1 CSV**（29DoF MJCF 关节序，官方用 GMR + PyRoki 重定向——与我们本地工具同源） |
| 元数据 | `seed_metadata_v004.csv/parquet`，51 列，每条动作最多 6 条自然语言描述 + 时序分割；clip 命名 `{动作名}__{演员ID}[_M]`（如 `crawl_ff_loop_270_001__A130_M`） |
| 攀爬内容 | Locomotion 包描述明写 "climbing, crawling"；SONIC 过滤脚本排除词含 `climb / ladder / stair / monkey_jump / handstand`——**反证这些片段存在**，同时意味着其 G1 CSV 版不可用。具体清单需拿到 gated 元数据后 grep 确认 |
| SONIC 实际用量 | 700h 源数据 → 过滤后 611h/100M+ 帧 @50Hz 训练；类别主力是 locomotion/gesture/acting/combat，**攀爬类被全部剔除** |

**行动含义**：注册 HF 账号 → 申请 `bones-studio/seed` → 下 `seed_metadata_v004.csv` grep `climb|ladder|crawl` → 只取对应片段的 **SOMA BVH**（人体侧）走我们自己的 GMR→C2 链路。不要碰它的 G1 CSV 爬类数据。

---

## 2. 动作来源分层（按"何时能用"排）

| 层 | 来源 | 内容 | 获取成本 | 贴合度 |
|---|---|---|---|---|
| **今天可用** | 本地 LAFAN1（77 条 BVH 全量在盘）+ 成品 `crawl_c2.pkl.npz` | `ground*`（爬行）、`obstacles*`（翻越）；无爬梯 | 0（`bvh_lafan1_to_c2.json` 现成） | crawl↔45° 阶段力学同构、pad 平压语义最贴 |
| **今天可下** | CMU mocap subject 14：**14_33 / 14_34 "climb ladder"**、14_21–23 "climb 3 steps"；subject 1 攀爬架 | 真人爬立梯 | 免费，走 AMASS(SMPL) 或 BVH 镜像 → `smplx_to_c2` / `bvh_mocap_to_c2` | 90° 四肢相序协调的最快来源 |
| **申请后（数天~数周）** | **BONES-SEED** ladder/climb/crawl 片段（SOMA BVH） | 量大质优（120fps Vicon、多演员体型） | HF gated 点击申请 | 同上，且可挑演员体型接近 C2 比例的 |
| **申请后（周期不可控）** | CIMI4D（18 万帧）/ **AscendMotion/ClimbingCap**（41 万帧，CVPR 2025）真攀岩 SMPL + 场景点云 | 攀岩墙（hold 抓握） | 邮件申请制（厦大） | 抓握语义偏差大、重钉幅度大；二期再看 |
| **自制** | 自拍视频 → GVHMR（本机就绪）→ `gvhmr_to_robot.py` | **可"导演"演员按机器人步态爬**：单肢依次移动、手掌平压、躯干贴面 | 半天拍摄 + 已有链路 | 语义可控性最高；但全局位移/接触不可信（见 §5 风险 4），只取风格 |

关键匹配论证（为什么 ladder/crawl 就够了）：我们的墙是**平面 + 解析碰撞平面，没有横档**（`wall_constants.py`，"step" 只是命令里的 `step_height=0.06`），坡度靠重力倾斜曲线模拟。因此 (a) 人类抓握 hold/rung 的手部细节本来就转移不了（C2 末端是刚性 spine 板，无手指无手腕），要转移的只是**四肢宏观轨迹 + 相序 + 基座推进**；(b) 同一条运动学参考（相对墙面表达）在 45°→90° 全课程几何上都成立，只有载荷分配 α\*(φ) 随角度变——由阶段 B 的离线 QP 按角度重解，与参考剪辑解耦。

---

## 3. 本地已有资产（全部实机验证）

**重定向工作台 `/home/spider/projects/retargeting/`（2026-05 建）：**

- `GMR_phybot-main/`——GMR（YanjieZe，ICRA 2026，mink IK 后端，Retargeting Matters 评测的开源最优）的 phybot fork：
  - C2 IK 配置齐套：`smplx_to_c2_auto.json`、`bvh_lafan1_to_c2.json`、`bvh_mocap_to_c2(_auto).json`、`phybot_c2_tpose.json`，`params.py` 已注册 `phybot_c2`；
  - 脚本齐套：`bvh_to_robot_c2.py`、`smplx_to_robot_dataset_c2.py`、`gvhmr_to_robot.py`（视频链路）、`ik_config_manager/`（keypoint 映射自动生成——SOMA 骨架接入用它）；
  - 后处理雏形：`output_after_process/add_contact.py`（接触区间标注）、`split_npz.py`、`edit_npz_deleteandinter.py`；
  - **成品**：`outputs/crawl_c2.pkl.npz`（629 帧，keys：`root_trans/root_ori/dof_pos(21)/dof_vel/link_pos_base(13,3)/link_rot_base(13,4)/base_lin_vel/base_ang_vel`——21 关节与 `PHYBOT_C2_POLICY_JOINT_NAMES` 对齐）。
  - ⚠️ pip editable 悬空：`general_motion_retargeting` 指向已不存在的 `retargeting/GMR`，目录现名 `GMR_phybot-main`，用前 `pip install -e` 重装。
- `GVHMR/`——视频→SMPL，checkpoints 已下载、demo 已跑过，专门的 `gvhmr` conda 环境在。其重力-视角坐标系对垂直大位移（攀爬）恰好是关键性质（TPAMI 2026）。
- `ubisoft-laforge-animation-dataset/`——LAFAN1 全量 77 条 BVH。
- `robot-motion-editor/`——浏览器端逐帧曲线编辑器（URDF 可视化 + npz 读写 + 平滑），重钉后手工修摆动段可用。

**训练侧（env_mjlab15）：**

- **mjlab 1.5 自带 BeyondMimic 式 tracking 任务**（`mjlab/tasks/tracking`，机器人无关核心 + G1 config 模板）。motion npz schema：`joint_pos/joint_vel/body_pos_w/body_quat_w/body_lin_vel_w/body_ang_vel_w`，**无 fps 键、必须预重采样到 step_dt=0.02s**，`body_*` 必须按本机器人 entity 全 body 序生成（G1 的 npz 不能挪用）。官方接入范式见 `g1_spinkick_example`：pkl→csv→`csv_to_npz`（FK 回放生成世界系量）→ WandB registry→train；`csv_to_npz.py` 硬编码 G1 scene/关节表，**C2 化约半天**。奖励=锚点全局位姿 + 锚相对 body 位姿 + 全局 body 速度（exp 核），RSI 自适应采样。
- AMP 基建（`amp_rsl_rl 2.2.1`）：`.npy` dict schema（`joints_list/joint_positions/root_position/root_quaternion(xyzw)/fps`），判别器 obs 48 维。**与 tracking npz 是两套 schema**，转换器要各写各的。
- 仓库内 IK 底子：`gen_squat_clip.py::_SquatIK`（`mj_jacSite` + damped least squares + 足部钉死 + CoM 平衡）——接触重钉脚本的现成参考实现；`mink 1.1.0`、`qpsolvers 4.11.0` 已装未用。
- climb 任务现状（与 plan 文档一致）：`ClimbGaitCommand` 发 11 维命令（EE one-hot 4 + 相位 one-hot 4 + 端点 delta 3），**尚无移动路点**；相位机接触驱动（RELEASE→SWING→SEAT→TRANSFER），gait 相序固定 `(2,1,3,0)`=左脚→右手→右脚→左手；EE=`{left,right}_{hand,foot}_spine`；spawn 距墙 0.255m。

---

## 4. 推荐管线（五步）

```
S1 取材            SEED SOMA BVH (ladder/crawl) │ CMU 14_33/34 │ 本地 LAFAN1 ground* │ GVHMR 自拍
                                        ↓
S2 GMR 重定向      GMR_phybot-main，C2 配置现成；手部 rotation 权重→0（无腕 4DoF 臂只跟位置），
   （风格层）      肘部走 PostureTask；SOMA 骨架先用 ik_config_manager 生成 keypoint 映射
                                        ↓
S3 IK 微调         mink 逐帧精修（"贴墙化"）：接触中的 spine 板 = 高权重 FrameTask 钉到
   （接触层）      墙平面 x=1.5、目标高度=step_height 网格、姿态仅跟法向；摆动肢低权重跟 S2；
                   基座 standoff→0.255m + 根轨迹 motion-warp；关节限幅+速度限幅
                   （参考实现：_SquatIK 模式或直接 mink API；add_contact.py 做接触区间标注底子）
                                        ↓
S4 相位重索引      按 EE 接触区间把 clip 切成单肢摆动段，每段归一化 s∈[0,1]，
   （时间→事件）   按我们 gait 相序 (左脚→右手→右脚→左手) 重排成 per-phase 路径表
                   ——人类 crawl 是对角步态、与我们相序不同，整段时间硬跟必然打架，必须切段重排
                                        ↓
S5 消费（三个口，可并存）
   口1 主线   相位化路点表 → ClimbGaitCommand 阶段 A-1 的 SWING 移动 setpoint
              （事件门控保留：φ 不推进路点就不动，风险#3 不重开；LadderMan 结论的直接落法）
   口2 增援   mjlab tracking 任务整段跟踪做"风格预训练"或并行头（上肢权重打折 + 接触丢失终止；
              需 C2 版 csv_to_npz + tracking config + 把 wall 加进 scene）
   口3 备胎   AMP 判别器（48 维 obs 基建现成，clip 转 .npy 即可）——维持 plan §8 的备胎位
```

S3/S4 之所以敢做薄，是因为我们的"墙"是连续平面而非离散 rung——接触重钉只是"投影到平面 + 高度取 step 网格"，不存在 rung 对齐组合问题（OmniRetarget 那类 research-grade 接触保持工具二期做课程增广时才需要考虑）。

---

## 5. 风险清单

1. **接触语义差**：人类 grasp（腕卷、指抠）↔ pad 平压+拖拽剪切。缓解：只取肢体宏观轨迹（S2 手部 rotation 权重 0），按压/啮合时序仍由现有 SEAT shaping 和（阶段 B）α 跟踪管——**clip 不供静相位信息，静相位本来就是 α/margin 的地盘**（plan §3 分工不变）。
2. **时间索引 vs 事件门控**（plan 风险#3）：raw DeepMimic 跟踪会拖着策略在未挂稳时移动，且人类相序≠我们 gait 相序。缓解：主线走口1（切段+相位参数化）；若开口2，必须按 LadderMan 配方——跟踪权重供风格、接触目标由几何生成、接触丢失终止，且上肢跟踪权重打折。
3. **4DoF 无腕可达域**：人类手部轨迹里含腕补偿，C2 到不了的位形 IK 会推肘部飞出。缓解：GMR per-body 双权重（位置高/旋转 0）+ mink 关节限幅硬约束 + 重定向后逐帧可达性检查（超限帧占比>阈值就换 clip 或降幅度）。
4. **视频法全局漂移**：ClimbingCap 论证了现有 HMR 对离地动作的全局位置恢复不可靠（贴墙遮挡、接触漂移）。GVHMR 自拍素材只取风格与相序，全局平移一律以墙几何+step 网格为真值重建（S3 本来就做）。
5. **SEED gated 日历等待**：审批周期不可控。缓解：本地 LAFAN1 crawl + CMU 14_33/34 先行，SEED 到手后只是换更好的素材重跑 S2–S4（管线不变）。
6. **reward 电池膨胀纪律**（plan §9.1 不变）：口1 只是替换现有 `commanded_ee_target_error_l2` 的目标来源，不新增奖励项；口2/口3 若开，逐项单独验指纹。

---

## 6. 与现行计划的关系（结论：升级，不替代）

- **阶段 A**：A-1 的"手写三段路径"升级为"clip 驱动的相位化路径"，跟踪机械（移动 setpoint 的双向 L2、事件门控、margin 正奖励）一字不改。手写版仍按原计划做保底（LadderMan 表明几何接触目标才是泛化主力，风格层缺了也能跑通，只是难看+探索慢）。
- **阶段 B**：α\*(φ) 离线 QP 原样；且因"墙恒竖直、重力倾斜"，单条参考全课程复用，只有 α 表按 (φ, angle) 二维索引重解。
- **AMP**：维持备胎位（plan §8），素材从同一条管线出，多花的只是一个 `.npy` 转换器。

---

## 7. 工时估计（熟手工程师日）

| 路径 | 内容 | 估时 | 备注 |
|---|---|---|---|
| 保底 E | 手写相位样条 + 现有奖励接线（= plan 阶段 A 原案） | ~1 周 | 风险最低，与下面共享全部消费侧机械 |
| **主线 A** | CMU/LAFAN1 clip → GMR(现成) → mink 重钉 2–3d → 相位化抽取器 1d → 口1 接线 1–2d | **首版 5–8 天** | 今天可开工，不等 SEED |
| A+SEED | 申请到手后换素材重跑 S2–S4 | +1–2 天 | SOMA 骨架 keypoint 映射首次 +1 天 |
| 口2 tracking | C2 版 csv_to_npz 0.5–1d + tracking config 1–2d + 首轮训练 2–4d | +4–7 天 | 仅当口1 的稠密路点仍撞不破探索时上 |
| 口3 AMP | clip→.npy 转换器 | +0.5 天 | 判别器基建现成 |
| 二期 | OmniRetarget 式接触保持增广 / CIMI4D 攀岩风格 | 2–4 周 | 现阶段收益/成本比差，挂起 |

---

## 8. 起手 checklist

- [ ] （人工，今天）注册/登录 HuggingFace，申请 `bones-studio/seed` 访问；到手后 `grep -iE 'climb|ladder|crawl' seed_metadata_v004.csv` 出候选清单
- [ ] 修复悬空 editable：`cd /home/spider/projects/retargeting/GMR_phybot-main && pip install -e .`（gmr 环境）
- [ ] 冒烟：`vis_robot_motion.py` 回放 `outputs/crawl_c2.pkl.npz`，确认 5 月链路仍能跑
- [ ] 下载 CMU 14_33/14_34（AMASS-CMU SMPL 版优先），过 `smplx_to_robot_dataset_c2.py`
- [ ] 写 S3 mink 接触重钉脚本（输入 GMR npz + 墙参数，输出贴墙 npz；`add_contact.py`/`_SquatIK` 做底）
- [ ] 写 S4 相位化抽取器（GMR npz → per-phase EE 路径表），对接 `ClimbGaitCommand` 阶段 A-1 移动路点
- [ ] （若开口2）C2 版 `csv_to_npz` + `tasks/tracking/config/phybot_c2/`，scene 加 wall entity
- [ ] 验收沿用 plan §5：完整循环 success>0 + 侧位视频目视（`play --video True`, `MUJOCO_GL=egl`）

---

## 9. 关键来源

SONIC 论文 arxiv.org/abs/2511.07820 · GR00T-WBC github.com/NVlabs/GR00T-WholeBodyControl · BONES-SEED huggingface.co/datasets/bones-studio/seed · SEED G1 CSV 规范 nvlabs.github.io/ProtoMotions/getting_started/seed_g1_csv_preparation.html · GMR github.com/YanjieZe/GMR · Retargeting Matters arxiv.org/abs/2510.02252 · mink github.com/kevinzakka/mink · mjlab github.com/mujocolab/mjlab（tracking=BeyondMimic 复现；spinkick 示例 github.com/mujocolab/g1_spinkick_example）· BeyondMimic arxiv.org/abs/2508.08241 · **LadderMan arxiv.org/abs/2606.05873** · ETH 四足爬梯 arxiv.org/abs/2409.17731 · Aalto 攀爬三部曲（SIGGRAPH 2017 / SCA 2018 / CoG 2019）· Brachiation SIGGRAPH 2022 · VideoMimic arxiv.org/abs/2505.03729 · CIMI4D lidarhumanmotion.net/cimi4d · ClimbingCap/AscendMotion arxiv.org/abs/2503.21268 · GVHMR github.com/zju3dv/GVHMR · OmniRetarget arxiv.org/abs/2509.26633 · CMU mocap subject 14（14_33/14_34 climb ladder）· BABEL babel.is.tue.mpg.de
