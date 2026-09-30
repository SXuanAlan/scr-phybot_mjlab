# BeyondMimic 攀爬配置审阅（2026-09-07，美国洛杉矶时间）

16 项奖励、自适应 RSI、出生挂点时序和终止条件已实现并验证。**没有开启训练。**
321 项 microspine 相关测试通过，GPU 环境检查通过；参考关节位置 PD 回放仍会快速失附，
因此本记录不把 60° climbing 或重新挂接标记为通过。完整配置见
[配置快照](configuration_snapshot.json)，测试见 [测试日志](../../../empirical_climb_review/beyondmimic_2026-09-07/test_results.log)。

两项完整任务共用奖励、RSI、观察、PD 和 PPO，硬件与课程如下：

| 任务 | 单个 pad 的 active unit | 训练课程（迭代数：角度） | 默认评估 |
| --- | ---: | --- | --- |
| `Mjlab-Climb-Wall-Phybot-C2` | 12，3×4 全阵列 | 0：45±5°；1500：75±12°；3300：80±5°；3900：85±3°；4550：88±2°；5150：90° | 90°，frame 0 |
| `Mjlab-Climb-Wall-Phybot-C2-Corners-A60` | 4，CAD 四角 | 0：45±5°；1500：55±5°；3300：60° | 60°，frame 0 |

四角任务的最终阶段没有角度扰动，整个课程不会采到 60° 以上。
既有 `Corners-A80` 仍为四角固定 80°，同步采用新版奖励和 RSI。
Wang 模型的 transition / transition80 保留其既有任务配方。

**奖励。** 采用 BeyondMimic 的四种 body tracking 核和三项常规正则，再加入已讨论的攀爬项。
公式、body 位置的 XY/yaw 对齐方式和失败直方图采样来自
[官方奖励配置](https://github.com/HybridRobotics/whole_body_tracking/blob/cd65172032893724b445448818c34165846d847d/source/whole_body_tracking/whole_body_tracking/tasks/tracking/tracking_env_cfg.py)
及 [官方 motion command](https://github.com/HybridRobotics/whole_body_tracking/blob/cd65172032893724b445448818c34165846d847d/source/whole_body_tracking/whole_body_tracking/tasks/tracking/mdp/commands.py)。
下表是 Phybot 的适配值，不是直接照搬 G1 的参数。

| 分类 | 项目 | 权重 | 误差尺度 / 条件 |
| --- | --- | ---: | --- |
| 身体追踪 | `motion_body_pos` | +1 | 0.10 m |
| 身体追踪 | `motion_body_ori` | +1 | 0.30 rad |
| 身体追踪 | `motion_body_lin_vel` | +1 | 1.0 m/s（新增项） |
| 身体追踪 | `motion_body_ang_vel` | +1 | 3.14 rad/s（新增项） |
| 身体追踪 | `motion_root_ori` | +0.5 | 0.30 rad，绝对 root 朝向 |
| 攀爬追踪 | `motion_ref_height` | +2.5 | 0.025 m，沿墙方向，初始 0.5 s 就稳后比较进度 |
| 攀爬追踪 | `motion_anchor_landing` | +40 | 0.03 m，每个计划挂接窗口一次；50 Hz 下最大 +0.8 / 次 |
| 攀爬追踪 | `motion_swing_approach` | +0.5 | 0.05 m，摆动时靠近下一挂点 |
| 接触 | `motion_engage_mismatch` | −1 | 计划 ON/OFF 不匹配；窗口边缘有渐变宽限 |
| 接触 | `microspine_support_loss` | −4 | 少于 2 个 pad 支撑 |
| 接触 | `unsafe_microspine_release` | −2.5 | 按真实 release 原因及剩余支撑计事件代价 |
| 接触 | `microspine_release_load` | +1 | 单 unit 的 shear / lateral 尺度均为 2 N；有载释放扣分 |
| 正则 | `action_rate_l2` | −0.1 | 动作变化平方 |
| 正则 | `dof_pos_limits` | −5 | 主动关节软限位越界 |
| 正则 | `undesired_contacts` | −0.1 | 身体非预期接触，峰值力 >1 N；保留 10 个物理子步历史 |
| 失败 | `termination_penalty` | −500 | 50 Hz 下非超时终止 −10 / 次 |

body 位置、姿态、线速度、角速度先在 17 个身体上取平方误差的均值，再计算
`exp(−error / σ²)`。位置误差为欧氏距离，姿态误差为旋转角。
等效误差达到 σ 时，该核降到约 0.368。body 位置沿用已有末端位置尺度 0.10 m，
姿态沿用 0.30 rad；高度、落点和 swing 尺度保留原值。旧关节 q/dq 和仅末端位置核被 body 核替换。
参考 body 状态通过当前 CAD 模型 FK 得到，包含被动 gimbal；速度使用下一实际帧差分，
周期结尾包含 20 mm 的爬升闭合，避免回环产生假速度尖峰。

body 位姿按实际 root 的世界 XY 和 yaw 对齐，保留参考的世界高度；速度直接比较旋转到墙面后的
世界参考速度。绝对 root 朝向和沿墙高度仍独立约束。mismatch 折扣只乘在
`motion_body_pos`、`motion_body_ori`、`motion_root_ori` 上：
`g = 0.25 + 0.75 × (1 − mean(graced_mismatch))`。
边缘宽限为该 pad 当前 ON/OFF 窗口长度的 30%，DR 强制失附待恢复的 pad 豁免。
远离边缘、一个 pad 不匹配时 g=0.8125；独立的 mismatch 负奖励保留。

release-load 奖励读取真实释放前的每 unit 载荷，最大受载 unit 决定质量和成本。
质量为 `exp(−max((S/2)²+(L/2)²))`，成本为 `min(4, max(0.25(S/2)²+0.1(L/2)²))`。
只在计划释放窗口或其前 0.25 s、另外至少 3 个 pad 仍支撑时可能获得正分，每窗口一次；
未知域或 early-failure 释放只计载荷成本，DR 强制释放不计此项。
原有初始 0.5 s 的 release 结算宽限保留。**没有独立 survival / alive bonus。**
常规奖励由 manager 乘策略步长；release 事件项内部除以步长，使其权重表示每次事件的金额。

**出生与相位。** 训练覆盖整个 450 帧、9 s 的循环，使用 9 个约 1 s 的 bin。
对非超时终止的失败所在相位计数，EMA α=0.001；在归一化前加入 `0.1 / bin_count`。
这不是固定 10% 的均匀混合比例，但每个 bin 和每一帧始终可被采到。
初次 reset 与纯 timeout 不累计失败；评估固定 frame 0。

RSI 写入完整 root / 主动关节 / 被动 gimbal 的位置和速度，按相位初始挂接 3 或 4 个 pad。
主动关节出生噪声为 0；沿墙高度偏移 −0.30..0 m，保留随墙角变化的垂直高度补偿。
出生挂接是已有成熟挂点，不需要重新通过 10 N 建压，也不会获得出生 engage、落点或 release 奖励。
框架 reset 原先只写逻辑 engaged，第一次物理步尚无 equality 约束；现在 FK 刷新后立即建立物理挂点。
零时间 reset 不推进相位、防抖或保持年龄，自动 reset 的新生行也不抢先推进一帧。
局部 reset 不移动其他环境的挂点或推进其计数。

真实新 engage 仍要求 **每个 active unit 的压缩 preload ≥10 N**，连续 3 个策略步（0.06 s），
并满足接触、朝向、虚拟 spine 覆盖与速度条件；释放后冷却 10 步（0.2 s）。
保持时不再要求维持这份 preload，压缩超过 10 N 也不是失附原因。
承载仍逐 unit 检查实际 shear / lateral 测量域和法向拉脱容量，使用原有随机强度及 early-failure 抽样。
保持读数保护为 2 步，没有给出生 seed 加长保护。

用户已确认的新挂接就位行为保留：名义约 0.28 s，执行器 age 1..13 内沿 down-rake
每步最多跟随 1 mm，不能向外、向上或横向跟随。已核对原实现还会在该窗口内重新捕获 pad 姿态，
窗口结束后固定；这一角度跟随没有额外角度步长上限。此轮修正了配置注释对该行为的误述，
没有修改真实新挂接的姿态跟随规则。出生的成熟挂点跳过平移和姿态就位，直接固定在 reset 姿态。

**终止。** 除 30 s timeout、沿墙高度 <0.3 m、非有限/异常状态外，
新增连续全失附 0.4 s 的终止，给 0.2 s 冷却和 0.06 s 建压防抖留出一次恢复机会。
原平均关节误差 >0.7 rad 的 `bad_tracking` 改为：任一跟踪 body 的对齐位置误差 >0.3 m，
或绝对 root 朝向误差 >1 rad，或沿墙 root 高度误差 >0.25 m。
这些是失败阈值，与表中的奖励误差尺度不同。

**验证结果与限制。** [321 项回归测试](../../../empirical_climb_review/beyondmimic_2026-09-07/test_results.log)覆盖奖励核、折扣、相位尾帧、
失败采样归因与覆盖、防抖/冷却、9.9/10/13.2/20 N preload 边界、偏载及力矩、
12/4 unit 分配、释放奖励、初始挂点、就位边界、课程、CAD、transition、部署变体等。
37 个警告是依赖中的 `torch.jit.script` 弃用提示。

[GPU 集成检查](runtime_verification.json)每种工况使用 8 个环境、100 个策略步（每环境 2 s），
保留训练 RSI 与 DR，以参考关节位置作 PD 目标。出生 body 位置最大误差 <2.2 μm，
3/4 个物理挂点与逻辑状态一致，出生奖励为零，局部与自动 reset 检查通过。
actor / critic 观察维度为 150 / 170；状态、观察、奖励有限；接触与约束缓冲区均无溢出。
失败直方图正常更新，所有相位 bin 保持正采样概率。

| 工况 | 2 s 内非超时 episode 终止次数（8 环境合计） | 新挂接次数 |
| --- | ---: | ---: |
| 12 unit，45° | 32 | 0 |
| 12 unit，90° | 33 | 0 |
| 4 unit，45° | 32 | 0 |
| 4 unit，60° | 36 | 0 |

终止主要来自姿态/高度偏离或持续全失附。自动 reset 后重新出现的挂点不算新 engage。
以上工况均未完成一个攀爬循环。

[当前出生载荷复测](birth_load_verification.json)另用每工况 4 环境、frame 0、无额外 DR、
25 步（0.5 s），仅在诊断中关闭自动终止以观察原始挂点的结局。
初始物理挂点均已就绪，但四组均丢失全部初始挂点，重新挂接仍为 0。
以四角 60° 的 env 0 为例：第 3 步右手某 unit 的法向拉力 3.46 N 超过当次容量 1.76 N，
先以 reason 2 脱离；第 4 步脚部最大 unit shear 达 38.49 N，超过测量域上限 30 N，
以 reason 7 脱离。这里检查的是拉脱和剪切承载，不是给 ≥10 N 的压缩 preload 加上限。

当前动作参考只包含运动学状态，没有承载力或力矩前馈。时序修复能消除首步无约束的下坠，
普通参考 PD 仍未维持满足经验承载模型的受力分配。成功 hold/release/regrip 的动态控制验证仍需完成；
本结果既不能证明 60° climbing 已可行，也不能单凭未训练 PD 的失败证明硬件不可行。
[早期出生对照](birth_reset_probe.json)记录了修复前 / 立即建立挂点 / 成熟挂点的历史比较，
其脚本和日志对应迁移中的源码状态；当前实现的复现以 `birth_load_verification.py` 为准。

**PPO 设置。** 默认单 GPU 0、2048 环境，500 Hz 物理 / 50 Hz 策略，rollout 96 步，
最多 12000 迭代，每 25 迭代保存。actor / critic 均为 256-128-64 ELU、观察归一化，
21 维关节位置动作，缩放 0.25、软限位 clamp 0.98，Gaussian 初始 std=0.6。
学习率 3e−4、自适应 KL=0.01、γ=0.99、λ=0.95、clip=0.2、entropy=0.003，
每轮 5 epochs / 4 minibatches，seed=42、resume=False。这些 PPO 参数未因奖励迁移而调整。
四角 60° 使用独立实验目录 `phybot_c2_climb_wall_corners60`；没有创建训练 run。

复现时使用 `env_mjlab15` 环境。以下命令只做测试、配置导出和仿真诊断：

```bash
python -m pytest -q tests/microspine
python docs/empirical_climb_review/beyondmimic_2026-09-07/configuration_snapshot.py
CUDA_VISIBLE_DEVICES=0 MUJOCO_GL=egl OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python docs/empirical_climb_review/beyondmimic_2026-09-07/runtime_verification.py
CUDA_VISIBLE_DEVICES=0 MUJOCO_GL=egl OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python docs/empirical_climb_review/beyondmimic_2026-09-07/birth_load_verification.py
```
