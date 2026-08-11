# LadderMan: Learning Humanoid Perceptive Ladder Climbing

Paper: [[2606.05873v1.pdf]]  
Related: [[Recent Humanoid Research Survey 2025-2026]], [[DeepMimic Note]], [[BeyondMimic Note]], [[OmniRetarget Note]], [[TWIST Note]], [[FALCON Note]]

Authors: Siheng Zhao, Yuanhang Zhang, Ziqi Lu, Pieter Abbeel, Rocky Duan, Koushil Sreenath, Yue Wang, C. Karen Liu, Guanya Shi  
Lab: Amazon FAR + USC + UC Berkeley + Stanford + CMU  
Venue: arXiv preprint  
Date: 2026-06  
Links: arXiv https://arxiv.org/abs/2606.05873 · project https://ladderman-robot.github.io  
Keywords: humanoid ladder climbing, perceptive loco-manipulation, hybrid motion tracking, DAgger, reinforcement learning, depth perception, sim-to-real, dual-agent learning, Unitree G1

## 1. 核心

LadderMan 研究的是比楼梯更苛刻的接触任务：29-DoF Unitree G1 要在细窄、离散的横档上持续协调双手和双脚，既要根据不同梯子重新选择接触位置，又要承受任何一次深度或控制误差沿攀爬过程累积。

它没有直接从 depth 到 action 端到端训练，而是把问题拆成四段：

1. 只采集一条爬梯参考动作，并标注双手、双脚的接触时序；
2. 对不同梯子分别训练 state-based expert，用 hybrid motion tracking 保留动作结构，同时允许接触点随梯子几何变化；
3. 用 DAgger-style imitation + PPO，把多名 expert 蒸馏成一个只看本体感知、深度图和上下行命令的统一视觉策略；
4. 真机上用 stereo vision foundation model 生成更干净的深度；到达梯顶后，再切换到上下半身解耦的双 agent 操作策略。

最关键的方法思想是把参考动作拆成两部分：**动作的稳定骨架需要保留，和环境有关的接触位置必须允许改变**。如果完整追踪原动作，手脚会伸向旧梯子的横档；如果只做 task reward，策略又容易退化成下肢主导、缺少可靠四肢接触的行为。Hybrid motion tracking 正是在两者之间取平衡。

![[Whole-Body Control/LadderMan/assets/figure-2-overview.png|900]]

> LadderMan 的“unified system”并不等于一个 policy 完成所有事情：攀爬阶段是统一 visuomotor policy，梯上操作阶段是另一套 dual-agent policy，系统在梯顶切换控制器。

## 2. 整体训练流程

```text
一条 MoCap 参考动作
  + 四个末端的人工接触时序
                │
                ▼
针对多种 (梯子倾角 φ, 横档间距 z)
分别训练 state-based expert
                │  Hybrid motion tracking
                ▼
多个具备特权状态的 expert
                │  DAgger-style KL imitation + PPO
                ▼
统一 depth-based climbing policy
                │  仿真：Warp depth + RFM + 少量噪声
                │  真机：stereo grayscale → Fast FoundationStereo
                ▼
上下攀爬
                │  到达梯顶后切换
                ▼
lower-body stabilization agent
  + upper-body teleoperation agent
```

这条 pipeline 把三种不同的 generalization 分开处理：

- **动作到新梯子**：靠 hybrid motion tracking 生成适应该几何的 expert；
- **多个梯子到一个策略**：靠多 expert 蒸馏与 RL；
- **仿真图像到真实图像**：靠 VFM、RFM、轻量 depth randomization 和高保真碰撞几何。

## 3. Stage I：从一条参考动作学习多个 expert

### 3.1 单条 reference 提供什么

参考动作通过 OptiTrack 在倾角 $65.5^\circ$、横档间距 $24.8\,\mathrm{cm}$ 的梯子上采集，包含一次完整攀爬序列。作者另外只做一次人工标注：给双手、双脚四个 end-effector 标出每个时刻是否应当接触。

这里刻意只从 reference 继承：

- 身体局部姿态、关节位置、速度和动作 phase；
- 四个末端**什么时候**接触。

末端具体应该接触**哪里**并不照抄 reference，而是根据目标梯子的倾角、横档间距和当前目标横档自动生成。换句话说，接触时序来自 demonstration，接触位置来自新环境几何。

### 3.2 非对称的 hybrid motion tracking

标准 motion tracking 对全身所有 keypoint 一视同仁：

$$
r_X^{\mathrm{track}}
=\omega\exp\left(-\sigma\|\hat X[K]-X[K]\|_2^2\right).
$$

这对跨梯子迁移不合适，因为上半身尤其是双手必须移动到新横档。LadderMan 将 keypoint 分成上半身 $K_u$ 和下半身 $K_l$：

$$
r_X^{\mathrm{track}}
=
\frac{\omega}{2}
\exp\left(
-\frac{\sigma}{2}
\|\hat X[K_u]-X[K_u]\|_2^2
\right)
+
\omega
\exp\left(
-\sigma
\|\hat X[K_l]-X[K_l]\|_2^2
\right).
$$

这里上半身项的最大权重和误差敏感度都被减半，下半身则更严格：

- 下半身尽量复现 reference 中稳定的支撑、抬脚和重心移动模式；
- 上半身只保留大致动作结构，给手臂留出寻找新横档的自由度。

这并不是“只追下半身”。上半身仍受 tracking reward 约束，只是允许它在 task/contact reward 的驱动下偏离 reference。

### 3.3 Ladder-centric contact 与 climbing reward

对每个末端，论文使用：

$$
r^{\mathrm{contact}}
=
\omega\hat c
\exp\left(
-\frac{\|P-\hat P\|_2^2}{\tau}
\right)
+
(1-\hat c),
$$

其中 $\hat c$ 是 reference 中的接触标记，$P$ 是当前末端位置，$\hat P$ 是从目标梯子几何自动生成的横档接触点。

- 当 $\hat c=1$ 时，reward 把对应手或脚吸引到目标横档；
- 当 $\hat c=0$ 时，该项退化成常数，不要求摆动末端留在某个位置。

此外还有 root/pelvis 相对梯子的目标姿态 reward，使机器人沿梯子向目标位置推进。于是完整目标可理解为：

$$
\text{保留动作结构}
\;+\;
\text{在正确时刻接触新横档}
\;+\;
\text{让 pelvis 沿梯子移动}.
$$

这三个部分缺一不可。论文的 expert ablation 中，标准 DeepMimic 只在 reference 梯子上成功；移除 contact tracking 后也只在 reference 梯子上成功；移除 climbing/root task reward 后能适应较小梯子，但在 $(70^\circ,30\,\mathrm{cm})$ 的大跨度梯子上失败。

### 3.4 Expert 的 observation、action 与 reward

Expert 是训练期的 state-based privileged policy，输入分三组：

| 输入组 | 内容 |
|---|---|
| Reference motion | body local pose、reference joint position、phase $t/T$ |
| Proprioception | root 角速度与朝向、$q_{t-8:t}$ 关节历史、$a_{t-3:t-1}$ 动作历史 |
| Privileged state | pelvis 相对梯子 pose、body pose/velocity、root velocity，以及 body、velocity、joint tracking error |

动作是 target joint positions，由底层 PD controller 执行。

主要 tracking/contact reward 参数如下；温度越小，对误差越敏感：

| Reward | Upper weight / temp. | Lower weight / temp. |
|---|---:|---:|
| Body local pose | 1.0 / 1.0 | 2.0 / 0.5 |
| Body velocity | 0.5 / 5.0 | 1.0 / 2.5 |
| Joint tracking | 0.5 / 0.5 | 1.0 / 0.25 |
| Root pose | - | 1.0 / 0.5 |
| Hand contact position | - | 0.3 / 0.3 |
| Foot contact position | - | 0.3 / 0.3 |

正则项还约束 action rate、关节位置/速度、torque limit、足部冲击、foot slip 和 feet air time。它们的作用不是产生攀爬行为，而是抑制 hybrid reward 为了够到横档而利用不安全的动作。

### 3.5 Termination、domain randomization 与训练

除常规 local/global tracking error termination 外，LadderMan 加入接触失败终止：需要的梯子接触连续丢失超过 30 个 simulation step 就结束 episode。姿态误差超过 $0.5\,\mathrm m$ 或 $1.2\,\mathrm{rad}$ 并持续 25 step 也会终止。

每个 expert 针对一个中心梯子配置训练，但在其周围做局部随机化：

- 梯子倾角 $\Delta\phi\in[-1^\circ,1^\circ]$；
- 横档间距 $\Delta z\in[-1,1]\,\mathrm{cm}$；
- link mass、torso CoM、joint offset、reference initialization；
- static/dynamic friction、restitution、横档几何。

Expert 使用 PPO，在 Isaac Sim 中以 4096 个并行环境训练 8000 epochs。Actor MLP 为 $[512,256,256]$，critic 为 $[512,256,128]$。这种设计降低了每个 expert 的任务跨度，但也意味着它的数据需求低、训练计算量却仍会随梯子配置数量增加。

## 4. Stage II：把多个 expert 蒸馏成统一视觉策略

### 4.1 为什么不能只做 DAgger

多 expert 已经知道不同梯子上应该怎样爬，但它们依赖 reference、梯子相对 pose 和仿真真值，不能直接部署。自然做法是让 visual student 在自己的 rollout state 上模仿对应 expert。

问题是 ladder climbing 的失败具有累积性：expert 没覆盖所有梯子和偏离状态，student 的一个小动作误差会改变下一个接触，再把视觉和状态分布推得更远。纯 imitation 只能逼近已有 expert，缺乏主动恢复与超出 expert 覆盖范围后继续完成任务的能力。

### 4.2 Hybrid DAgger + RL

LadderMan 将 DAgger-style imitation 和 PPO 放在同一目标中：

$$
\mathcal L(\pi^{\mathrm{visual}})
=
\mathcal L_{\mathrm{RL}}(\pi^{\mathrm{visual}})
+
\lambda
D_{\mathrm{KL}}
\left(
\pi^{\mathrm{visual}}
\;\|\;
\pi^{\mathrm{expert}}
\right).
$$

$\mathcal L_{\mathrm{RL}}$ 使用与 expert 相同的 task reward，KL 项约束 student 接近当前梯子对应的 expert。训练中逐渐减小 $\lambda$：

- 早期以 imitation 为主，先获得可行的四肢接触顺序；
- 后期增大 task-driven optimization 的相对作用，学习 error recovery，并填补 expert 之间的 configuration gap。

这和固定权重 behavior cloning 的区别在于：expert 是初始化与行为先验，不是 student 最终能力的硬上限。实机消融也支持这一点：去掉 RL、只保留 imitation 后，三把梯子的成功率从 $9/10,6/10,7/10$ 降为 $2/10,0/10,0/10$。

### 4.3 部署策略真正看到什么

最终 $\pi^{\mathrm{visual}}$ 的输入只有：

- proprioception history；
- $60\times80$ depth image；
- 一个二值 climbing direction command：向上或向下。

它不再接收 reference motion、已知梯子参数或 privileged ladder-relative state。输出仍是全身 target joint positions。Policy 部署频率为 50 Hz。

视觉 actor 使用 $[2048,1024,512,256,128]$ MLP，depth encoder 是三层 CNN，channel 均为 8、kernel size 为 5；critic 为 $[512,256,128]$。仿真深度由 NVIDIA Warp 渲染，以避免 Isaac Sim 原生 renderer 成为大规模并行训练瓶颈。

## 5. Sim-to-real：感知差距与接触几何

### 5.1 将视觉误差分成两类

论文区分两种 depth gap：

1. **Unstructured noise**：缺失像素、测距噪声、飞点；
2. **Structured gap**：真实梯子的侧轨、支架、横档形状等在仿真中没有完整覆盖。

LadderMan 分别用 VFM 和 RFM 处理，而不是把所有差异都塞进一套强 depth randomization。

![[Whole-Body Control/LadderMan/assets/figure-3-perception.png|900]]

### 5.2 VFM depth：先把真实输入变干净

真机不直接使用 RealSense D435i 的 raw depth，而是把 stereo grayscale 输入 Fast FoundationStereo，生成更连续、更接近仿真几何的 depth。VFM 是 policy 前的感知预处理模块，并没有和 climbing policy 联合训练。

训练期只加入相对轻量的 augmentation：

- Gaussian noise $\mathcal N(0,0.005)$；
- pixel dropout，Appendix A.2 写的是 $p=0.05$；
- depth clip 到 $[0.1,2]\,\mathrm m$；
- 相机平移扰动 $\pm1\,\mathrm{cm}$、旋转扰动 $\pm1.5^\circ$。

真机上关闭相机的 infrared emitter 和 automatic exposure。Climbing policy 运行在机载 Jetson Orin，但 Fast FoundationStereo 在外部 RTX 4090 上实时推理，因此当前演示还不是完全 onboard 的 perception-control stack。

### 5.3 Rung-Focused Masking

RFM 在训练中以 $p=0.1$ 遮掉非横档区域，迫使 policy 把注意力放在真正决定手脚接触的 rung 上，而不是记住某种梯子的侧轨或支撑结构。它利用了仿真中可获得的 rung semantic mask，但部署时不需要显式分割横档。

RFM 的作用很强：去掉它后三把真梯的成功率都是 $0/10$。这说明视觉 policy 原本很容易对梯子背景几何过拟合，也说明真实泛化并不只是“加一点像素噪声”就能解决。

### 5.4 高保真 collision geometry

梯子接触不只受 depth 影响，也受机器人 collision mesh 影响。论文比较了 HDMI、TWIST2、OmniRetarget 等 G1 模型，指出平地 locomotion 中可以接受的简化 primitive，在梯子上会让 shin、hand、foot 的小误差逐步积累。

LadderMan 以 [[OmniRetarget Note]] 的高保真碰撞模型为起点，并把 collision geometry 视为关键 sim-to-real 变量。论文没有完整列出 LadderMan 最终 mesh 的逐部件修改，因此这部分对独立复现仍不够透明。

## 6. 梯上操作：dual-agent learning

通用 whole-body teleoperation policy 在梯顶直接接管时，upper-body target 会扰动全身平衡。LadderMan 因此另外训练：

$$
\pi^{\mathrm{manip}}(s_t,g_t)
=
\left[
\pi^u(s_t,g_t);
\pi^l(s_t)
\right].
$$

两个 agent 拥有独立 actor 和 critic：

- **Lower-body agent $\pi^l$**：只接收全身 proprioception，优化梯子 contact 与 pelvis stabilization；
- **Upper-body agent $\pi^u$**：还接收 arm、wrist、waist 的 teleoperation target，优化 upper-body tracking；
- 最终 action 直接拼接 lower-body 与 upper-body 输出，减少两个目标在同一网络中的梯度和动作干扰。

训练时 upper-body target 从 AMASS joint pose 随机采样；部署时由 PICO 4 Ultra 的 VR target 经 IK 生成。为了扩大操作空间，仿真训练中关闭了上半身与梯子的 collision。这有利于探索，但也隐藏了手臂撞击梯子的真实风险。

Dual-agent policy 的意义不是让机器人边爬边任意操作，而是在到达预设的梯顶稳定姿态后维持下半身四肢接触，再执行调整画框、换灯泡和递箱子等上半身任务。论文对这些任务主要给出定性展示，没有报告系统性的 manipulation 成功率。

## 7. 结果

### 7.1 仿真泛化

作者在倾角 $\{55^\circ,60^\circ,65^\circ,70^\circ\}$、横档间距 $\{20,22,24,26,28,30\}\,\mathrm{cm}$ 的网格上测试，每个配置 100 个 episode，并随机横档尺寸与方形/圆柱形状。

- 在间距 $24$-$28\,\mathrm{cm}$、倾角 $55^\circ$-$65^\circ$ 的区域成功率超过 95%；
- 在测试边界附近平均成功率约 57%；
- blind motion-tracking baseline 即使在 reference 附近峰值也只有 49%，离开训练几何后接近 0。

这组结果说明 depth feedback 不只是修正小扰动，而是统一策略适应不同梯子几何的必要输入。

### 7.2 真机

| 方法 | Ladder A | Ladder B | Ladder C |
|---|---:|---:|---:|
| LadderMan | 9/10 | 6/10 | 7/10 |
| w/o RL | 2/10 | 0/10 | 0/10 |
| w/o VFM | 3/10 | 0/10 | 2/10 |
| w/o RFM | 0/10 | 0/10 | 0/10 |

三把梯子在倾角、横档间距、宽度、材料和侧轨结构上都不同，reference 只在 Ladder A 上采集。机器人平均约 $3.4\,\mathrm{s/rung}$，论文测得两名人类平均约 $3.2\,\mathrm{s/rung}$。

所有成功率和速度均为作者自报，未独立复现。尤其 Ladder B/C 只有 10 次 trial，$60\%$-$70\%$ 更适合解读为“已经能 zero-shot 工作，但还没有达到工业级可靠性”。

## 8. 贡献

1. 用非对称 tracking、几何生成的接触目标和 climbing reward，把单条爬梯 reference 适配为多个梯子 expert。
2. 用退火 KL imitation + PPO 将 privileged expert bank 蒸馏成不需要 reference 和已知梯子参数的统一 visuomotor policy。
3. 将 depth sim-to-real 拆成 unstructured 与 structured gap，分别用 Fast FoundationStereo 和 rung-focused masking 处理。
4. 强调高保真 collision geometry 对细窄、多接触 humanoid 任务的重要性。
5. 用上下半身双 agent 将梯顶稳定与 teleoperated manipulation 解耦。

## 9. 局限与阅读时要注意的细节

- 目前只验证倾斜梯，论文给出的范围到约 $75^\circ$，没有解决 vertical ladder。
- 真机每把梯子只有 10 次 trial，Ladder B/C 成功率分别为 60% 和 70%。
- VFM depth 依赖外部 RTX 4090，完整系统并非全部在机载算力上运行。
- 攀爬与操作是分开的 policy，控制器切换由系统外部完成；没有学习边爬边操作或失败后的自主 mode selection。
- 操作训练关闭 upper-body-ladder collision，真实碰撞安全性没有被充分建模。
- 没有 dexterous hand，操作演示主要依靠腕端或简单夹持结构。
- “单条 reference”降低的是 motion-capture 数据成本，但多个 per-configuration expert 仍带来随梯子覆盖范围增长的训练成本。
- 正文 Eq. (2) 将 $r^{task}$ 排版成正的 $\|R-G\|_2^2$，若作为最大化 reward 会鼓励误差增大；Appendix 的表格将它描述为 exponential root-pose tracking。阅读时应以后者为准，这里很可能缺了负号/指数。
- Depth dropout 参数存在文本与图示不一致：正文和 Appendix A.2 写 pixel dropout $p=0.05$，Fig. 3 标成 $p=0.9$。独立复现需要作者代码或进一步说明。

## 10. 和已有工作的关系

[[DeepMimic Note]]、[[BeyondMimic Note]] 的标准 motion tracking 假设 reference 已经适合当前环境。LadderMan 的增量是允许 reference 中的上半身和 end-effector contact 随梯子几何变化，因此它更接近 environment-conditioned motion adaptation，而不是纯 imitation。

[[OmniRetarget Note]] 在数据生成阶段保存人与场景的 interaction geometry；LadderMan 则在 policy learning 阶段用接触时序和程序化 rung target 保存 interaction。两者思路相通，但 LadderMan 只需要一条 reference，并把视觉反馈放进最终闭环；同时它直接借用了 OmniRetarget 的高保真 G1 collision geometry。

[[TWIST Note]] 追求自由空间中的通用全身 teleoperation，在梯子这种受约束支撑状态下会因为上半身动作干扰平衡而失败。LadderMan 的 dual-agent policy 是一个更窄但更稳的 specialized solution：显式保留下半身 stabilization，再让上半身响应 VR target。

[[FALCON Note]] 关注未知手部受力下的稳定 whole-body control。LadderMan 当前主要依赖位置接触和几何感知，没有显式估计横档接触力；如果未来扩展到 vertical ladder 或负载操作，force adaptation 会成为自然的互补模块。
