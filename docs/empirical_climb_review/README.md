# 新实测 microspine 模型与 climbing 验收包

**预载语义修正：** 10 N/unit 是 engage 的最低预压。此前承载模型中的
`Fn >= -10 N` 错把这个预压用成压缩上限，已从运行时、静力审计和离线容量计算中移除。
拉脱分布和剪切/侧向数据支持范围保持原值，增加压缩不增加拟合的拉脱容量。
下面的旧图和动作统计保留为历史记录；四角硬件的重算结果以
[60° 评估](corners60/README.md) 和 [80° 评估](corners80/README.md) 为准。

**2026-09-06 四角硬件更新：** 当前 empirical climbing 配置可只使用四角 4 个 unit，
新增固定 80° 任务 `Mjlab-Climb-Wall-Phybot-C2-Corners-A80`。
本次完整可行性检查未通过，详见 [四角 4-unit / 80° 评估](corners80/README.md)。
下文保留之前 12-unit 版本的结果，不能作为四角版本的验收依据。

用户授权范围：完成 1、2、3、4、6；第 5 步等用户验收后再进行。
这里已经完成软件实现、离线检查和实验材料准备。没有启动 PPO、短训、整段物理 rollout，
也没有操作真实机械臂。静态图片来自真实任务的 reset，视频来自逐帧运动学渲染。

**目前适合验收实现与假设，尚不能认定随机接触下的 climbing 已可稳定训练。**
2026-09-06 已重新生成 climbing 循环并检查全部 490 帧。几何和已挂附阶段的
中位 unit 静力分配通过；新增的 10 N/unit 建压检查中，四次换点前的
12 帧均未通过当前姿态下的准静态容量约束。**新 engage 条件的可执行性尚未通过。**
详见 [最新参考审计](reference_audit.json) 的 `engage_preload`。
新的 reference 依赖已有球铰的 -3° 限位支撑；当前独立 unit、首个 unit 失效即释放 pad
的模型，对固定参考载荷给出的同时保持概率很低。下面保留这些结果。

## 完成内容

| 步骤 | 本次交付 | 状态 |
| --- | --- | --- |
| 1 实测失效分布 | 原始试验审计、条件分布表、保留位置评估、数据来源与假设 | 已完成 |
| 2 数据分布驱动的 DR | 每次挂附采样，每个 unit 的随机分位在挂附期间保持；替换旧容量均匀 DR | 已完成 |
| 3 C 点 wrench 与 unit 分配 | 新 4×3 几何、完整力矩分配、物理 unit 索引与失效原因 | 已完成 |
| 4 初值与 motion reference | 新初值、完整循环、被动球铰状态、几何/碰撞/静力检查 | 已完成离线版本，限制见下文 |
| 5 训练与 rollout 验证 | 短训、正式训练、策略验证 | **未开始，等用户验收** |
| 6 图与验证材料 | 原版布局的随机 pad 图、对照图、reference 视频、bench 协议与记录模板 | 已完成离线材料；真实 pad 实验未执行 |

## 先看这些文件

- [代码、数据与公共工具布局](code_layout.md)
- [原始 CAD 与刚性导入结果对照](../figures/phybot_c2_cad_rigid_comparison.png)、[编译后网格审计](rigid_cad_asset_audit.json)
- [新 climbing 初值：斜视与侧视](../figures/phybot_c2_climb_initial.png)
- [完整 reference 视频](climb_reference.mp4)：运动学动画，无控制器和动力学执行。
- [随机 pad limit surface](../figures/fig_pad_limit_surface_stochastic_cad.png)
- [实测分布与保留测试](../figures/fig_empirical_distribution_validation.png)
- [力矩消融与旧 Wang 参数对照](../figures/fig_empirical_model_comparison_cad.png)
- [完整检查结果](reference_audit.json)、[实际 reset 与碰撞检查](../figures/phybot_c2_climb_initial.audit.json)
- [真实 pad 实验协议](hardware_protocol.md)、[计划试验矩阵](pad_bench_matrix.csv)、[记录模板](hardware_trial_log_template.csv)

## 1. 实测数据怎样进入模型

数据源是 `data/microspine/raw/20260905-110314/results.csv` 中列出的 296 次试验：
275 次 `detached_normal`、20 次拖动/侧向加载阶段失效、1 次 `held`。
有 69 个载荷条件，每个条件只有 1–11 次；不能把单次条件估计为没有随机性。
另有一个 T297 明细文件没有进入 results 汇总，已在审计中列出，没有混入拟合。

终末法向失效使用 `fz_cmd_n`，保留斜坡事件当时的实际指令，不能改用更高的台阶目标。
`fz_meas_n` 可能已经处于掉载过程；T1 的 5.447 N 不作为失效容量，T1 使用 9 N。
`fz_first_release_n` 单独保留，用于初次部分释放分析，未与最终脱落合并。
`held` 是右删失：只知道至少承受了最后载荷，未把它当作在该点失效。
20 次早期失效作为加载成功概率的独立部分，未伪造成 -10 N 的法向拉脱点。

模型估计的是“给定剪切 S、侧向 L 后，法向终末失效载荷的条件分布”。
邻近条件的真实观测按高斯核加权，用加权 Kaplan–Meier 方法保留删失信息，
输出 1024 个逆 CDF 分位点；不是先平均载荷再添加任意比例噪声。
核宽在 calibration 位置上按留位置交叉验证选择：剪切 2.5 N、侧向 5 N。

现有数据按位置作回顾性划分：x < 540 mm 的 149 次用于 calibration，
其余 147 次用于 validation，仍来自同一块表面。核表支持其中 143 次，
不支持的 4 次明确报告；以下比较使用双方都支持的同一批 134 次法向终末试验：

| 指标 | 稀疏格点直接分布 | 当前加权条件分布 |
| --- | ---: | ---: |
| 中位预测 MAE | 2.662 N | 2.171 N |
| 10%–90% 区间覆盖率 | 29.85% | 73.88% |

名义覆盖率是 80%，当前没有完全达到。这个划分参与了模型开发判断，属于回顾性证据，
不能写成预注册独立确认或新表面泛化。运行表在上述评估后用全部 296 次重新拟合；
运行表不再拿自己的训练行声称 held-out 精度。运行表存放于
`src/scr_phybot_mjab/assets/microspine/empirical_unit_v1.npz`；拟合报告与
calibration/raw-cell 对照表存放于本验收目录的 `limit_surface/`。

剪切/侧向有效范围限于每个 unit：S 在 [-15, 30] N，|L| <= 30 N。
法向失效条件是超过采样的拉脱上界；10 N seating 不构成压缩侧上限。
拟合表 schema 2 使用 `protocol_preload_n=10`，避免把预压字段误读为承压限制；
旧 schema 1 的 `normal_min_n` 仅按历史协议元数据兼容读取。
缺失条件仅在支持区域内插值。负侧向由正侧向镜像，是未实测的对称性假设。
删失尾部保守截在观测下界并保留标志；没有生成未测得的高强度尾部。
这仍是“10 N 单 unit seating、每级 0.5 N、保持 3 s”的协议条件模型，
没有辨识不同保持时间、疲劳和动态失效风险率。

## 2. 随机性与 reset

默认 climbing 的 `contact_model` 已切为 `empirical_unit`。每个环境、pad、unit
保存两个随机数：强度分位和加载阶段失败分位。它们在 reset / 新挂附时生成，
同一次挂附期间不逐步重抽；载荷变化时查询同一个分位的曲面。
不同方向保持相同分位排名是额外相关性假设，现有跨位置数据不能辨识它。

旧的 F0 均匀容量 DR、seat 容量乘数、capacity ramp 和随机 catch ticket
在 empirical 模式关闭。额外强制随机脱落事件默认关闭，可显式打开。
质量、摩擦、编码器偏置、观测噪声等非接触随机化仍保留。
配置中的 360 N 仅是兼容诊断字段的域端点 12×30，**不是 pad 实测承载力**。

新挂附要求完整虚拟接触面进入几何范围，且 12 个 unit 的估算法向预压
**全部 ≥10 N**（`engage_unit_preload_n=10.0`），连续 3 个策略步达标后
整块 pad 才锁定。面朝向、禁止逆 rake 运动和 release 后的 cooldown 仍生效。

四角 netforce 传感器记录每个物理子步的合力、关于接触力加权中心 P 的
合力矩、P 的位置和 pad 阵列中心 C 的位置。先对齐参考点
`M_C = torque_P + (P-C) × force`，再对一个策略步内的接触 wrench 取平均；
用与 hold 判定相同的 CAD unit 坐标和等刚度分配器得到逐 unit 载荷。
外法向载荷取正时，预压为 `max(-f_n_unit, 0)`。诊断字段
`unit_preload_n` 和 `pad_contact_wrench_tln` 分别保存预压和 C 点接触 wrench。

120 N 只是零弯矩、均匀受力时的总力下限，偏载可能需要更大的总预压。
该判定仍是等刚度模型估计，四个角碰撞代理不能证明内部各 unit 的真实预压。
10 N 只约束 OFF→ON；锁定后不要求持续压墙，保持和释放由 limit surface 判断。
初值和 RSI 使用已经挂附的成熟接触，绕过 seating 过程；随机容量没有按初值载荷筛选，
也没有通过反复重抽强 unit 来掩盖初始保持率问题。

## 3. 统一到 C 的力矩与几何

采用原始 CAD 的刚性装配，撤销之前的局部网格变形。去掉假手，CAD 套筒止挡
对齐裸前臂末端；手臂球心 B 纵向伸出 55.5 mm，偏离前臂轴线 12.25 mm。
腿部 B 位置保持不变。所有 pad 的局部 B→C 约为 (0, -47.50, +77.98417) mm，
接触面朝局部 -y；4 列×3 行，横向间距 28 mm、纵向间距 31.5 mm。
四角 unit 使用贴齐表面的块状碰撞代理；12 个 unit 都参与容量与力矩计算。
PLA 与装配惯量已按未变形 CAD 实体重新估算：移动 pad 约 186.90 g，
每个手臂固定连接件约 205.98 g，未替代称重/实测。几何读取工具位于
`tools/microspine/cad.py`；完整安装说明见 [asset 文档](../phybot_c2_cad_asset.md)。

所有加载量统一成 `[t, l, n]`：t 为 rake，l 为横向，n 为离墙方向。
载荷符号表示机器人向接触施加的载荷，Fn > 0 为拉脱，Fn < 0 为压墙。
pad 名义局部方向对应 t=-z、l=-x、n=+y。CAD row/column 与随机 unit 索引已逐一校对。

终端可读的计算式如下，所有量必须先转到同一坐标系、同一载荷符号：

```text
外部量若测在 B：M_C = M_B - (C-B) cross F

当前 weld：平移约束作用点已经是 C；旋转约束乘子经四元数 Jacobian
转换后得到的是 C 处的物理 couple。因此 runtime 不再重复扣一次 B→C 力臂。

N = 12
x_i = 沿 t 的 unit 坐标；y_i = 沿 l 的 unit 坐标，原点都是 C
Sxx = sum(x_i*x_i), Syy = sum(y_i*y_i), Srr = Sxx + Syy

ft_i = Ft/N - Mn*y_i/Srr
fl_i = Fl/N + Mn*x_i/Srr
fn_i = Fn/N + Mt*y_i/Syy - Ml*x_i/Sxx

检查：sum(f_i) = F；sum(r_i cross f_i) = M_C
```

这是一种刚性平面 pad、unit 等刚度的载荷分配。当前保持判定取最弱 unit，
首个 unit 越界即释放整个 pad；**单 unit 数据没有证明完整 pad 就会这样脱落**。
unit 间相关性、部分失败后的重新分配和渐进释放，需要整 pad 试验来判断。
失效原因额外区分 7=超出实测支持域、8=加载阶段失败；超域不能当实测失效机制。

## 4. 新初值与 reference 的检查结果

活动文件是 `climb_cad_empirical_initial.json` 和 `climb_cad_empirical.npy`。
50 Hz，490 帧：循环 450 帧 / 9.0 s，上升 20 mm，另有 0.8 s 尾部保持。
循环期间至少 3 个 pad 挂附，包含换脚、换手、卸载、接近和再挂附的参考阶段。
21 个策略关节外，另存 12 个被动球铰坐标；RSI 和渲染会读它们，
它们没有变成策略动作或主动关节目标。未投影的 RSI 关节扰动暂设为零。

- 全部 490 帧：挂附中心距墙最大误差约 0.00016 mm，anchor 漂移最大约 0.00051 mm。
- 本次检查中，unit 碰撞盒最大穿墙量约 0.00018 mm；全片段 FK 与当前 CAD 一致。
  此项检查没有重新审计全部机身碰撞对。
- 主动关节、被动球铰范围及循环衔接通过；初值 host/GPU site 位置误差约 0.00009 mm。
- 全部 490 帧：均找到满足实测域、关节扭矩和“各 unit 取 0.5 分位”的
  已挂附静力分配。这不包含新挂附前的 10 N/unit 建压负担。

每个换点前另检查连续 3 帧：接近中的 pad 只允许普通压触与摩擦，
要求全部 12 个 unit 预压至少 10 N，其余三个 pad 按实测中位容量承载。
当前参考的四次换点均未通过这个准静态检查；即便将承载 unit 的法向容量
乐观放宽到整个实测域的中位最大值 6 N，线性约束仍无可行解。
这是当前姿态与该容量分位下的结论，不是动态策略或所有可能姿态的不可行证明。

**受力条件：全部 pad 的被动 x 球铰位于现有 -3° 下限位。**
静力检查为限位反力加入了单边符号约束，没有把自由球铰当成无限力矩关节。
参考检查中的最大限位反力矩约 **9.10 N·m**；初值两手约 **7.83 N·m**。
现有 PLA 结构/限位能否承受它，没有被这些计算证明。

固定上述每帧优化得到的载荷，再抽样 2048 组独立 unit 容量，
完整 490 帧的“所有当前 pad 同时保持”Monte Carlo 比例为 **0–0.44%**，
中位约 **0.049%**，结果逐帧保存在 `reference_audit.json`。
这是当前 IID + 首 unit 释放模型对固定参考载荷的结果，**不是实机可靠性**，
也不能证明换一种载荷分配或策略一定失败；采样 0 次成功也不表示真实概率严格为零。
逐 unit 的中位分位可行，更不等于整个 pad 的中位容量可行。
需要验收的核心是：这个 unit→pad 假设是否符合机构，成熟接触如何形成，
以及是否应依靠该限位支撑。不能为得到漂亮的保持率而静默放大容量。

静力分析是必要条件的放松：仅平衡承载被动关节，不含摆动关节的自由动力学、
惯性/加速度、控制误差和闭环时延。它不能替代第 5 步的动力学与策略检查。

## 6. 图与真实 pad 验证准备

新 pad 图沿用原版面板、配色和坐标风格，显示模型预测的 10% / 50% / 90%
保持概率边界；这些不是参数置信区间。力矩面板在固定 Ft、Fn=-60 N、Mn 下，
从零弯矩状态沿射线增加弯矩，初始状态就失效的抽样不会被丢出分母或允许重新保持。
它们是这一加载路径的保持边界，未穷举可能不连通的静态可行 wrench 区域。
指定路径和概率下无可行边界的区域留空，没有补成未经测量的闭合曲面。

Wang 对照使用仓库原有模型参数，没有重新数字化文献。
力矩消融在同一组随机样本和合成载荷上比较完整分配与去掉力矩影响，
这是数值对照，不是新完成的整 pad 力矩硬件验证。
历史 `fig_pad_limit_surface_empirical*` 图保留归档，本次使用 `_stochastic_cad` 版本。

`reference_static_loads.csv` 已导出采样帧的 C 点 wrench，明确标为
`reference_static_optimization`。它不是 policy 输出、实测数据或可直接发给 Flexiv
的控制指令。第 5 步后才能从冻结的策略 checkpoint 导出真正的 policy-derived loads。
本次准备了 60 次独立挂附的候选 bench 矩阵，其中 5 次为另一预压下的探索试验；
没有填写任何假实验结果，也未发送机器人指令。

## 回归与复现

本次针对实测表、随机状态、C 点 wrench、box 接触、新 motion/RSI、任务配置、
奖励和配置继承运行了回归检查；实际命令和最终结果保存在 [verification.json](verification.json)
及 [test_results.txt](test_results.txt)。本次 120 项相关回归通过，包含 climbing
全片段 FK、循环闭合、建压时间窗口、preload 审计和上墙衔接检查。
此前单独重生成的 A80 上墙参考已通过其 FK 回归；它沿用 legacy Wang 静力模型，
不能用它的受力检查替代这里的 empirical 建压检查。未运行全仓库测试或新策略训练。

从仓库根目录运行，使用已有 `env_mjlab15` 环境：

```sh
export PYTHONPATH=src
export MUJOCO_GL=egl
export OPENBLAS_NUM_THREADS=1
/home/spider/miniconda3/envs/env_mjlab15/bin/python scripts/build_empirical_limit_surface.py
/home/spider/miniconda3/envs/env_mjlab15/bin/python scripts/gen_cad_climb_reference.py
/home/spider/miniconda3/envs/env_mjlab15/bin/python scripts/audit_cad_climb_reference.py --stride 1
/home/spider/miniconda3/envs/env_mjlab15/bin/python scripts/plot_stochastic_pad_limit_surface.py
/home/spider/miniconda3/envs/env_mjlab15/bin/python scripts/render_climb_initial.py
```

这些命令只生成/检查离线材料，最后一个会构建 GPU 环境并 reset，不执行物理步进。
旧 Wang 绘图脚本已显式选择 legacy Wang；旧参考验证器会拒绝误用在新 empirical clip 上。
数据、asset 和 motion 的文件哈希保存在 [artifact_manifest.json](artifact_manifest.json)。
