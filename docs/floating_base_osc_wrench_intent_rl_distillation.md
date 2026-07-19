# 将 Floating-Base OSC 的 Task/Contact Wrench Intent 蒸馏为 Position-Level Control

## 面向随机 Microspine 接触攀爬机器人的技术方案

> 文档状态：方案设计总结  
> 核心定位：**wrench-guided、effect-tracking、goal-seeking、limit-surface-safe RL**  
> 目标系统：浮动基、全身关节仅支持位置控制、具有随机 microspine 多接触的攀爬机器人

---

## 摘要

本文提出一套面向随机 microspine 接触攀爬机器人的跨控制接口蒸馏方案。训练期使用能够访问完整动力学和接触状态的 floating-base Operational Space Controller（OSC）作为 privileged teacher，生成 task wrench、contact wrench、浮动基动力学响应、接触模式和安全裕度等物理意图。Student policy 不输出关节力矩，而输出真实机器人可执行的关节位置目标或位置残差，并通过位置伺服器与环境交互，复现 teacher 所期望的合成接触 wrench、载荷分配和任务空间动力学效果。

该方案与只追踪运动学参考的 DeepMimic 类方法不同：参考对象由单纯的姿态/速度轨迹扩展为包含接触模式、合成 wrench、内部载荷分配、任务空间运动效果和 microspine 安全裕度的 **hybrid wrench program**。同时，该方案也不要求 student 精确模仿 OSC 输出的关节力矩或每个接触的唯一 wrench，而是重点跟踪：

1. 多接触对浮动基产生的 **net wrench**；
2. 少量具有物理意义的 internal-force/load-sharing 坐标；
3. wrench 所造成的 body/end-effector motion effect；
4. 每个 microspine patch 相对于随机 limit surface 的安全裕度；
5. 攀爬进度、接触保持和失效恢复等任务目标。

推荐的最终结构为：

\[
\boxed{
\text{Planner / Hybrid Reference}
\rightarrow
\text{Floating-Base OSC Teacher}
\rightarrow
\text{Wrench/Effect-Guided RL Student}
\rightarrow
\text{Safety Projection}
\rightarrow
\text{Position Servo}
}
\]

### 主要符号

| 符号 | 含义 |
|---|---|
| \(q_j,\dot q_j\) | 可驱动关节位置与速度 |
| \(V_B\) | 浮动基 6D spatial velocity |
| \(v=[V_B^\top,\dot q_j^\top]^\top\) | 全机器人广义速度 |
| \(\tau_j\) | 关节执行力矩；仅由 teacher 或仿真动力学直接使用 |
| \(q_d\) | Student/位置伺服器可执行的关节位置目标 |
| \(\lambda_i\) | 第 \(i\) 个 gripper patch 的接触 wrench |
| \(G\lambda=w_G\) | 多接触在选定 body/CoM frame 上产生的 net wrench |
| \(\eta\) | 不改变 \(w_G\) 的 internal-force 坐标 |
| \(\alpha_i\) | 第 \(i\) 个接触的归一化载荷比例 |
| \(m_i\) | free、preload、scrub、load-bearing 等接触模式 |
| \(\mathcal W_i\) | 第 \(i\) 个 microspine patch 的随机 wrench limit surface |
| \(u_i=\rho_{\mathcal W_i}(\lambda_i)\) | 接触承载能力的归一化利用率 |
| \(\gamma_i=1-u_i\) | 接触安全裕度 |
| \(b_i\) | 对 engagement、limit surface 和接触模式的 belief |
| \(\phi\) | 动作或接触流程的 phase |

---

## 1. 问题背景

### 1.1 Microspine 接触为什么难以通过纯 RL 探索

Microspine gripper 的有效接触不是普通的刚性点接触或固定摩擦锥接触。其承载能力通常受到以下因素共同影响：

- 实际挂接的 spine 数量；
- 局部 asperity 的位置、方向和强度；
- 接近角、preload 和 scrub 过程；
- pulling direction 与接触局部坐标系；
- patch stiffness、结构柔顺和载荷重分配；
- partial engagement、micro-slip 和级联失效；
- 表面形貌与接触历史。

因此，接触状态包含明显的离散模式：

\[
\text{free}
\rightarrow
\text{approach}
\rightarrow
\text{touch}
\rightarrow
\text{preload}
\rightarrow
\text{scrub}
\rightarrow
\text{proof-load}
\rightarrow
\text{load-bearing}
\rightarrow
\text{unload/release},
\]

并伴随 retry、partial-slip 和 detach 分支。若让端到端 RL 同时发现接触模式、全身动力学、内部载荷分配和恢复策略，探索空间会过大，失败信号也会非常稀疏。

### 1.2 为什么纯 motion tracking 不够

传统 DeepMimic 类 tracking 主要参考：

\[
\left(q_t^\star,\dot q_t^\star\right).
\]

但对于 microspine climbing，同一姿态可能对应完全不同的接触承载状态。纯运动学参考无法说明：

- 哪些 gripper 已经可靠挂接；
- 每个接触应承担多少载荷；
- 当前 contact wrench 是否接近失效边界；
- 新接触是否通过 proof-load；
- 是否应移动重心并向其他接触转移载荷；
- 部分挂接失效后应继续、减速、重试还是释放。

因此，参考数据需要从 motion clip 扩展为 hybrid physical reference：

\[
z_t^\star=
\left[
m_t^\star,
w_{G,t}^\star,
\eta_t^\star\text{ 或 }\alpha_t^\star,
x_{B,t}^\star,
\dot x_{B,t}^\star,
x_{ee,t}^\star,
\dot x_{ee,t}^\star,
\gamma_t^\star,
\phi_t
\right].
\]

其中 \(m_t^\star\) 是接触模式，\(w_G^\star\) 是多接触合成 wrench，\(\eta^\star\) 或 \(\alpha^\star\) 描述内部载荷分配，\(\gamma^\star\) 是安全裕度，\(\phi\) 是动作 phase。

---

## 2. 硬件约束与设计结论

### 2.1 浮动基不是 WBC/OSC 的障碍

机器人广义速度写为：

\[
v=
\begin{bmatrix}
V_B\\
\dot q_j
\end{bmatrix}
\in\mathbb R^{6+n},
\]

浮动基动力学为：

\[
M(q)\dot v+h(q,v)
=
S^\top\tau_j+J_c(q)^\top\lambda,
\]

其中：

- \(V_B\) 是基座 6D spatial velocity；
- \(q_j\) 是可驱动关节；
- \(S=[0\ I]\) 是驱动选择矩阵；
- \(\tau_j\) 是关节力矩；
- \(J_c\) 是所有承载接触的 Jacobian；
- \(\lambda\) 是堆叠后的接触 wrench。

动力学前六行没有执行器输入，浮动基只能由接触 wrench 间接控制：

\[
\left[M(q)\dot v+h(q,v)\right]_{B}
=J_{c,B}^{\top}\lambda.
\]

所以关键不是“浮动基能否做 OSC”，而是当前 microspine 接触集合能否产生期望的 base wrench。

### 2.2 只有位置接口时不能直接部署 torque-level OSC

经典 OSC 最终产生关节力矩：

\[
\tau_{\mathrm{OSC}}
=J^\top F_{\mathrm{task}}
+N^\top\tau_{\mathrm{posture}}.
\]

若真实机器人只接受：

\[
q_j^{cmd}\quad\text{或}\quad\dot q_j^{cmd},
\]

且内部高增益位置环不可修改，则不能直接执行 \(\tau_{\mathrm{OSC}}\)，也不能严格保证 torque-level dynamic decoupling 和 internal-force regulation。

但位置控制接口仍可支持：

- floating-base whole-body inverse kinematics；
- position/velocity-level WBC；
- task-space admittance；
- quasi-static equilibrium/load-distribution optimization；
- learned wrench-to-position adaptation。

因此本方案将 OSC 定位为训练期或在线计算期的 **wrench/dynamics intent generator**，student 则通过 position command 复现其物理效果。

### 2.3 主要设计原则

1. **Teacher 可以使用理想力矩接口，student 必须使用真实位置接口。**
2. **训练 rollout 必须包含真实位置伺服模型。**
3. **OSC 的虚拟 task wrench 用作条件或控制意图，实际 contact wrench 和状态变化才是 effect supervision。**
4. **不精确模仿唯一的逐接触 wrench，而是跟踪 net wrench、少量 internal coordinates 和安全集合。**
5. **motion reference 不完全删除，而是降为消除欠约束、引导自由肢和保证姿态合理的正则项。**

---

## 3. 总体架构

### 3.1 训练期架构

\[
\boxed{
\begin{aligned}
&\text{Contact Planner / Reference Program}\\
&\qquad\downarrow\\
&r_t,\;m_t^\star,\;\phi_t\\
&\qquad\downarrow\\
&\text{Floating-Base OSC Teacher}\\
&\qquad\downarrow\\
&F_{\mathrm{task},t}^\star,\;
\lambda_t^\star,\;
w_{G,t}^\star,\;
\dot v_t^\star,\;
\gamma_t^\star\\
&\qquad\downarrow\\
&\text{Position-Level RL Student}\\
&\qquad\downarrow\\
&q_{d,t}^{\pi}\text{ 或 }\Delta q_{d,t}^{\pi}\\
&\qquad\downarrow\\
&\text{Safety / Equilibrium Projection}\\
&\qquad\downarrow\\
&q_{d,t}^{safe}\\
&\qquad\downarrow\\
&\text{Randomized Position Servo + Microspine Physics}
\end{aligned}}
}
\]

### 3.2 部署期的两种模式

#### 模式 A：在线 OSC intent

部署时保留 OSC 计算，但不执行其 torque：

\[
\text{current state}
\xrightarrow{\mathrm{OSC}}
z_t^\star
\xrightarrow{\pi_\theta}
q_{d,t}^{safe}.
\]

优点是 teacher intent 与当前真实状态、Jacobian 和接触模式对齐；缺点是部署仍需运行 OSC/WBC 模型。

#### 模式 B：完全蒸馏

OSC 只在训练时提供 privileged supervision。部署时 policy 输入任务参考、phase、接触观测和状态历史：

\[
\left(o_{t-H:t},r_t,m_t,\phi_t\right)
\xrightarrow{\pi_\theta}
q_{d,t}^{safe}.
\]

优点是运行简单；缺点是需要更充分的训练覆盖和 DAgger 式数据聚合。

### 3.3 推荐的 student action

第一版推荐：

\[
q_{d,t}
=q_{\mathrm{WBIK},t}^{\star}
+s_q\tanh\!\left(\Delta q_{d,t}^{\pi}\right),
\]

其中 \(q_{\mathrm{WBIK}}^\star\) 来自运动学 reference 或 position-WBC，RL 只输出有界关节位置残差。

更结构化的后续版本可以输出 task-space residual：

\[
\Delta r_t=
\left[
\Delta x_B,
\Delta p_{\mathrm{CoM}},
\Delta x_{ee},
\Delta x_{c,1:N},
\Delta\alpha_{1:N},
\Delta\dot\phi
\right],
\]

再由安全 position-WBC 转换为 \(q_d\)。这种形式更容易解释，也更方便显式处理关节限位和闭链接触约束。

---

## 4. OSC Teacher

### 4.1 Teacher 的输入与输出

Teacher 根据 student 当前状态、任务参考和 contact mode 计算：

\[
\mathcal C_{\mathrm{OSC}}
\left(s_t,r_t,m_t\right)
\rightarrow
\left(
\tau_t^\star,
F_{\mathrm{task},t}^\star,
\lambda_t^\star,
\dot v_t^\star,
\ddot x_t^\star
\right).
\]

建议保存的 teacher signal 包括：

- \(F_{\mathrm{task}}^\star\)：OSC 内部 task-space wrench intent；
- \(\lambda_i^\star\)：各接触的 nominal wrench；
- \(w_G^\star=G\lambda^\star\)：多接触合成 wrench；
- \(\eta^\star\) 或 \(\alpha^\star\)：少量 internal-force/load-sharing 目标；
- \(\dot v^\star,\ddot x_B^\star,\ddot x_{ee}^\star\)：任务空间动力学效果；
- \(m_t^\star\)：接触模式；
- \(\gamma_i^\star\)：接触安全裕度；
- \(x_B^\star,x_{ee}^\star\)：低权重运动学参考。

### 4.2 虚拟 task wrench 与实际 contact wrench 必须分开

OSC 中的 task wrench 通常形如：

\[
F_{\mathrm{task}}^\star
=
\Lambda
\left(
\ddot x_d+K_d\dot e+K_pe
\right)
+\mu+p.
\]

它是控制器内部的操作空间量，经过任务优先级、零空间投影、浮动基欠驱动约束和接触映射后才形成关节力矩及实际环境接触力。一般不能假设：

\[
F_{\mathrm{task}}^\star=\lambda_i.
\]

因此：

\[
\boxed{
F_{\mathrm{task}}^\star
\text{ 主要作为 policy 的 intent/condition；}
\quad
\lambda,\dot v,\dot x
\text{ 才是可比较的 physical effect。}
}
\]

### 4.3 Teacher 应在 student 当前状态上查询

固定离线轨迹中的 teacher signal 与 nominal state 对齐。Student 一旦偏离，该 signal 可能不再适合当前 Jacobian、姿态和接触状态。

训练期优先采用 state-aligned teacher：

\[
z_t^\star
=
\mathcal C_{\mathrm{OSC}}
\left(
s_t^{student},r(\phi_t),m_t
\right).
\]

这相当于让 teacher 对 student 实际访问的状态重新标注，可显著缓解 distribution shift。

---

## 5. Position-Policy Student

### 5.1 观测

Actor 只使用真机可获得的信息：

\[
o_t^\pi=
\left[
q_t,\dot q_t,
\mathrm{IMU}_t,
q_{d,t-1},
y_{\mathrm{gripper},t},
y_{\mathrm{load},t},
m_t,\phi_t,
r_t,
z_t^\star
\right],
\]

其中 \(z_t^\star\) 只在在线-intent 模式下输入。\(y_{\mathrm{gripper}}\) 和 \(y_{\mathrm{load}}\) 可包括：

- tendon force 或 load cell；
- gripper displacement；
- 电机电流；
- F/T sensor；
- micro-slip 或振动特征；
- 局部表面法向估计。

由于接触状态和执行器状态具有历史依赖，policy 应使用 observation history：

\[
a_t
=\pi_\theta(o_{t-H:t}),
\]

可实现为 GRU、LSTM、temporal encoder，或显式 observer 加普通 MLP。

### 5.2 Action

推荐优先级如下：

1. **第一版：** bounded joint-position residual \(\Delta q_d\)；
2. **后续版：** task-space/CoM/load-sharing residual，由 position-WBC 转为 \(q_d\)；
3. **不推荐：** student 直接输出 joint torque；
4. **高风险：** 无约束地直接输出绝对 \(q_d\)。

### 5.3 Position servo 必须显式存在于训练环境

仿真中的实际执行力矩应来自 position servo，而不是将 student action 当作 torque：

\[
\tau_t^{eff}
=K_p(q_{d,t}-q_t)
+K_d(\dot q_{d,t}-\dot q_t)
+\tau_{friction}
+\tau_{backlash}
+\tau_{delay},
\]

并包含：

- 位置/速度/电流饱和；
- command filter 与通信延迟；
- joint friction 和 deadband；
- gear backlash 和 transmission compliance；
- 姿态与负载相关的有效增益；
- 内环不可见积分或滤波状态的近似。

---

## 6. 为什么不直接模仿 OSC Torque

### 6.1 控制接口不匹配

Teacher 输出 \(\tau^\star\)，真机只能执行 \(q_d\)。若直接最小化：

\[
\left\|\hat\tau(q_d)-\tau^\star\right\|^2,
\]

student 必须同时隐式学习执行器逆模型、传动摩擦、接触刚度、闭链内部力和浮动基动力学。

### 6.2 映射不是静态且通常不唯一

真实映射应写成：

\[
q_{d,t}
=pi_{adapter}
\left(
\tau_t^\star,
q_{t-H:t},
\dot q_{t-H:t},
q_{d,t-H:t-1},
o_{contact,t-H:t}
\right),
\]

而不是静态的 \(\tau^\star\mapsto q_d\)。同一个关节力矩目标在不同姿态、接触模式、位置内环状态和传动状态下需要不同的位置命令。

### 6.3 某些 teacher torque 对位置接口不可实现

受限于内环刚度、位置误差上限、速度、电流和闭链刚度，一部分 \(\tau^\star\) 不在 position servo 的可达集合内。强制模仿会诱导过大的位置偏移或高频抖动。

### 6.4 相近 torque 不保证相同的接触安全性

在多接触闭链中，小的几何误差可能引起很大的 internal force：

\[
\Delta\lambda
\approx K_cJ_c\Delta q_d.
\]

即使 joint torque MSE 较小，各 microspine patch 的局部载荷仍可能越过失效边界。

### 6.5 本方案的替代目标

不要求 student 复制某一组 torque，而是复现 teacher 的任务与接触效果：

\[
\boxed{
w_G\approx w_G^\star,
\quad
\eta\approx\eta^\star,
\quad
\dot x\approx\dot x^\star,
\quad
\lambda_i\in\mathcal W_i^{safe}.
}
\]

---

## 7. Wrench 与 Physical-Effect Tracking

### 7.1 接触 wrench 的估计

将所有接触 wrench 堆叠：

\[
\lambda=
\begin{bmatrix}
\lambda_1^\top&\cdots&\lambda_{N_c}^\top
\end{bmatrix}^\top.
\]

仿真 reward 不应直接使用单个 physics substep 的尖峰接触力。建议在一个 policy period \(\Delta t\) 内累积 contact impulse：

\[
\bar\lambda_{i,t}
=
\frac{1}{\Delta t}
\sum_{k=1}^{N_{substep}}
p_{i,t,k},
\]

并进行适度低通滤波。真机上则使用 tendon/load/F/T/current 等传感估计，并明确其带宽和延迟。

### 7.2 Net wrench tracking

由 grasp/contact map \(G(q)\) 得到多接触对选定 body frame 或 CoM frame 的合成 wrench：

\[
w_G=G(q)\bar\lambda.
\]

误差为：

\[
e_{net}=w_G-w_G^\star,
\]

对应 reward：

\[
r_{net}
=
\exp\left(-e_{net}^\top Q_Ge_{net}\right).
\]

该项应比逐接触 wrench MSE 更高权重，因为 \(w_G\) 直接决定 floating-base 的整体动力学响应。

### 7.3 Physical-effect tracking

只跟踪 wrench 仍不能唯一确定运动。需要同时跟踪 teacher 的 task-space effect：

\[
r_{effect}
=
\exp\left(
-\|\dot x_B-\dot x_B^\star\|_{Q_B}^2
-\|\dot x_{ee}-\dot x_{ee}^\star\|_{Q_{ee}}^2
\right).
\]

必要时可加入低通后的加速度：

\[
\ddot x_B\approx\ddot x_B^\star,
\qquad
\ddot x_{ee}\approx\ddot x_{ee}^\star.
\]

速度和有限时域状态变化通常比瞬时数值加速度更稳定。实际实现中可使用 \(K\)-step effect loss：

\[
L_{K\text{-step}}
=
\sum_{k=1}^{K}
\left\|x_{t+k}^{student}-x_{t+k}^{teacher}\right\|_{Q_k}^2.
\]

### 7.4 Motion/posture tracking 的角色

运动学参考保留为低权重 regularizer：

\[
r_{kin}
=
\exp\left(
-\|q-q^\star\|_{Q_q}^2
-\|x_B-x_B^\star\|_{Q_x}^2
-\|x_{free}-x_{free}^\star\|_{Q_f}^2
\right).
\]

其作用是：

- 在 free-space/approach 阶段提供稠密引导；
- 约束同一 wrench 下的姿态冗余；
- 防止策略通过扭曲姿态或碰墙产生目标 wrench；
- 引导自由肢到达下一候选抓点；
- 改善视觉可读性和部署可预测性。

---

## 8. Net Wrench 与 Internal/Load-Sharing 分解

### 8.1 分解

多接触中的接触力可写为：

\[
\lambda
=G^\dagger w_G+N_G\eta,
\]

其中：

- \(G^\dagger w_G\) 是产生给定整体 wrench 的一种解；
- \(N_G\) 张成 \(G\) 的零空间；
- \(\eta\) 描述不改变整体 wrench 的 internal force/load-sharing。

由于 \(\eta\) 的表示依赖基和接触配置，不建议无差别跟踪全部零空间坐标。应只提取少量具有物理意义的量，例如：

- 左右或上下接触之间的 tension difference；
- 新旧接触的 load-transfer ratio；
- 每个 gripper 的归一化承载比例；
- proof-load 对应方向的内部拉力；
- 会直接影响 spine engagement 的 preload 坐标。

### 8.2 Load-sharing ratio

可定义支撑方向的归一化载荷比例：

\[
\alpha_i
=
\frac{\|P_s\bar\lambda_i\|}
{\sum_j\|P_s\bar\lambda_j\|+\epsilon},
\qquad
\sum_i\alpha_i\approx1.
\]

对应 reward：

\[
r_{share}
=
\exp\left(
-\|\alpha-\alpha^\star\|_{Q_\alpha}^2
\right).
\]

若方向符号和力矩分量很重要，应使用设计好的 internal coordinate \(\eta=P_\eta N_G^\top\lambda\)，而不是仅使用 wrench norm 比例。

### 8.3 为什么不严格跟踪每个 \(\lambda_i^\star\)

逐接触精确 tracking：

\[
\bar\lambda_i=\lambda_i^\star
\]

通常不合理，因为：

- 多接触力分配不唯一；
- teacher 与 student 的位置伺服和接触柔顺不同；
- 微小 anchor 或 stiffness 变化会改变最优分配；
- 某个非唯一 teacher 解不一定是 student 当前状态下最安全的解。

更合适的目标是：

\[
\boxed{
G\bar\lambda\approx w_G^\star,
\quad
\eta\approx\eta^\star\text{ 或 }\alpha\approx\alpha^\star,
\quad
\bar\lambda_i\in\mathcal W_i^{safe}.
}
\]

---

## 9. Contact Limit Surface 与安全裕度

### 9.1 Patch-level 宏观接触模型

不建议在全身控制或在线规划中逐根模拟所有 spine。每个 gripper patch 维护宏观随机状态：

\[
z_i=
\left(m_i,a_i,N_{engaged,i},\Theta_i,K_i,D_i\right),
\]

其中 \(\Theta_i\) 参数化方向相关的 wrench limit surface：

\[
\lambda_i\in\mathcal W_i(\Theta_i).
\]

\(\mathcal W_i\) 可以是：

- 各向异性椭球；
- convex polytope；
- 二阶锥或多个锥的组合；
- 数据驱动的概率边界；
- 随 pulling direction、engagement 和 history 改变的条件模型。

### 9.2 归一化利用率与安全裕度

定义 limit surface 的 gauge function：

\[
u_i
=
\rho_{\mathcal W_i(b_i)}(\bar\lambda_i).
\]

其中：

- \(u_i<1\)：位于预测承载集合内；
- \(u_i=1\)：到达边界；
- \(u_i>1\)：预测不可承载。

定义安全裕度：

\[
\gamma_i=1-u_i.
\]

使用收缩后的安全集合：

\[
u_i\leq\beta,
\qquad 0<\beta<1.
\]

容量 penalty：

\[
c_{cap}
=
\sum_i[u_i-\beta]_+^2.
\]

也可以最大化最危险接触的裕度：

\[
r_{margin}
=
\operatorname{softmin}
\left(\gamma_1,\ldots,\gamma_{N_c}\right).
\]

### 9.3 随机 limit surface

若 \(\Theta_i\) 不确定，应采用 belief 下的保守边界或 chance constraint：

\[
P_{\Theta_i\sim b_i}
\left(
\lambda_i\in\mathcal W_i(\Theta_i)
\right)
\geq1-\epsilon.
\]

也可以以分位数 limit surface \(\mathcal W_i^{(1-\epsilon)}\) 替代标称集合。Teacher、reward 和 safety projection 应使用一致的风险定义，避免 teacher 鼓励 student 靠近 projection 必然拒绝的边界。

---

## 10. Phase-Dependent Reward

### 10.1 总 reward

\[
\begin{aligned}
r_t={}&
w_{net}(\phi,m)r_{net}
+w_{share}(\phi,m)r_{share}
+w_{effect}(\phi,m)r_{effect}\\
&+w_{kin}(\phi,m)r_{kin}
+w_{margin}(\phi,m)r_{margin}
+w_{goal}r_{goal}\\
&-w_{cap}c_{cap}
-w_{slip}c_{slip}
-w_{detach}c_{detach}\\
&-w_{cmd}c_{cmd}
-w_{rate}c_{rate}
-w_{acc}c_{acc}
-w_{joint}c_{joint}
-w_{collision}c_{collision}.
\end{aligned}
\]

### 10.2 各阶段的主要目标

| Phase | 主要 reward | 主要安全条件 |
|---|---|---|
| Approach | 自由肢位置/姿态、低速接近、碰撞避免 | 不允许意外承载；限制接近速度 |
| Touch | 接触事件、低冲击、法向速度衰减 | 冲击/impulse 上限 |
| Preload | 指定方向 preload wrench、姿态保持 | preload window、禁止侧向过载 |
| Scrub | 切向位移/速度与 preload 联合跟踪 | slip、scrub 路径和 wrench envelope |
| Proof-load | wrench 上升曲线、位移稳定、置信度增长 | 剩余支撑 N−1 可行；测试载荷上限 |
| Load-bearing | net wrench、load sharing、margin、body effect | 所有承载接触位于安全集合 |
| Load transfer | 新旧接触载荷比例、最差 margin、CoM 调整 | 卸载速率和旧接触最小残余载荷 |
| Release | wrench 平滑降零、末端脱离轨迹 | 仅在其余接触可行时允许释放 |
| Recovery | 降速、重分配、re-scrub/re-engage | 防止级联失效和不可逆 detach |

### 10.3 位置命令正则

\[
c_{cmd}=\|\Delta q_{d,t}\|^2,
\]

\[
c_{rate}
=\|\Delta q_{d,t}-\Delta q_{d,t-1}\|^2,
\]

\[
c_{acc}
=\|q_{d,t}-2q_{d,t-1}+q_{d,t-2}\|^2.
\]

这些项用于防止 policy 通过高频 position excitation 在平均意义上匹配 wrench，却产生真机不可执行或会激发结构振动的命令。

### 10.4 Goal reward

任务 reward 可以包括：

- 向目标抓点或目标高度的进度；
- 成功完成接触转换；
- 成功完成 proof-load；
- 在规定时间内保持稳定；
- 避免不必要 regrasp；
- 发生局部失效后恢复而非终止。

Goal reward 不应压过安全项。若一次越界即可造成不可恢复 detach，相应条件应设为硬约束或 early termination，而非仅靠有限 penalty。

---

## 11. Phase/Event-Based Reference

固定时间参考 \(z^\star(t)\) 在随机接触下容易失配。例如实际机器人仍在 scrub，而固定参考已进入 load transfer，policy 会被迫过早加载。

因此采用：

\[
z^\star=z^\star(\phi,m),
\]

并以接触事件推进状态：

\[
\text{touch detected}\Rightarrow\text{preload},
\]

\[
\text{engagement confidence}>p_{min}
\Rightarrow\text{proof-load},
\]

\[
\text{proof-load passed}\Rightarrow\text{load transfer}.
\]

策略可以输出有界 phase-rate residual：

\[
\dot\phi
=\dot\phi_{nom}
+s_\phi\tanh(\Delta\dot\phi_\pi).
\]

当接触较弱或估计不确定时，policy 可以主动减速，而不是为追赶固定时间表强行增加载荷。

---

## 12. Contact Belief 与 History

### 12.1 隐变量

Microspine 的实际挂接质量通常不可直接观测。对第 \(i\) 个 patch，可维护：

\[
b_{i,t}
=P\left(
m_i,
N_{engaged,i},
\Theta_i,
a_i,
K_i
\mid y_{0:t}
\right).
\]

其中 \(y_{0:t}\) 包括力、位移、电流、IMU、振动和滑移等历史观测。

### 12.2 两种实现方式

#### 显式 belief/observer

使用 Bayesian filter、HMM、particle filter 或独立 RNN observer，输出：

- contact mode probability；
- engagement confidence；
- limit-surface 参数或分位数；
- slip/failure hazard；
- 等效 stiffness。

优点是可解释、便于安全层使用。

#### 隐式 recurrent belief

Policy 使用 GRU/LSTM 从 history 中隐式形成 latent state。Critic 和 reward 可以访问 privileged contact truth，但 actor 只能访问真实传感信息。

推荐结构为 asymmetric actor-critic：

\[
\pi_\theta(a_t\mid o_{t-H:t}^{real}),
\qquad
V_\psi(s_t^{privileged},\Theta_t,N_{engaged,t}).
\]

### 12.3 Proof-load 作为主动辨识

Proof-load 不仅是安全检查，也是主动信息采集动作：

1. 新 gripper 接触并 preload；
2. 执行受控 scrub；
3. 缓慢施加测试载荷；
4. 根据 force-displacement、微滑移和电流更新 belief；
5. 置信度足够高后才转移主要载荷；
6. 否则 re-scrub、retry 或换点。

训练中应奖励“以较小风险获得足够接触信息”，而不是仅奖励更大的测试载荷。

---

## 13. Domain Randomization

### 13.1 Contact randomization

- engaged spine 数量；
- anchor 位置和方向；
- asperity density 与表面法向误差；
- anisotropic limit-surface 尺寸和方向；
- patch stiffness/damping；
- preload 与 scrub effectiveness；
- partial-slip 和 detach threshold；
- load redistribution 后的级联失效；
- gripper sensing noise 和 delay。

### 13.2 Actuator randomization

- position-servo \(K_p,K_d\)；
- command delay 和 filter；
- velocity/current/position saturation；
- friction、deadband 和 backlash；
- transmission compliance；
- motor heating 或电压下降造成的能力变化；
- 控制周期 jitter。

### 13.3 Robot and environment randomization

- link mass、inertia 和 CoM offset；
- payload、cable/gripper mass；
- encoder/IMU noise 与 bias；
- 外部扰动；
- 表面几何、粗糙度和倾角；
- 视觉或候选抓点定位误差。

### 13.4 Episode 中途干预

只在 episode reset 时随机化，policy 可能仍记住 phase 到 action 的 feedforward 映射。应在同一 phase、同一 intent 下动态改变物理条件：

\[
\mathcal W_i\leftarrow\kappa\mathcal W_i,
\qquad0<\kappa<1,
\]

\[
N_{engaged,i}
\leftarrow
N_{engaged,i}-\Delta N,
\]

或突然改变 servo gain、延迟、外部载荷。这样才能迫使 recurrent policy 使用真实接触反馈并学习 recovery。

### 13.5 DR 分布的原则

- 初始范围来自单 gripper 标定和执行器系统辨识；
- 先窄后宽，采用 curriculum；
- 对低概率高后果失效进行定向 oversampling；
- 区分 aleatoric contact uncertainty 与模型辨识误差；
- planner/teacher 与 student 的风险假设应一致；
- 避免无物理意义的超宽 uniform randomization。

---

## 14. 在线与离线 Teacher

### 14.1 离线 teacher

预先生成数据集：

\[
\mathcal D=
\left\{
o_t,
r_t,
m_t,
\phi_t,
q_t^\star,
\tau_t^\star,
w_{G,t}^\star,
\eta_t^\star,
\lambda_t^\star,
\dot v_t^\star,
\gamma_t^\star
\right\}.
\]

优点：

- teacher 成本可以离线摊销；
- 方便大规模并行 RL 和数据分析；
- teacher 可使用较慢的 sampling/SQP/MPC 方法。

缺点：

- teacher intent 只与 nominal trajectory 对齐；
- student 偏离后 supervision 可能失效；
- recovery 状态覆盖不足。

### 14.2 在线 teacher

每个训练步在 student 当前状态查询 OSC：

\[
z_t^\star=\mathcal C_{OSC}(s_t^{student},r_t,m_t).
\]

优点：

- 自动 state alignment；
- teacher signal 与当前 Jacobian 和接触状态一致；
- 适合 DAgger 和 recovery training。

缺点：

- 计算成本更高；
- teacher 在随机或失败状态下可能不可行；
- 需要 teacher failure handling 和 fallback。

### 14.3 推荐混合方案

1. 用离线 teacher 生成 nominal demonstrations 和初始化 policy；
2. 让 student rollout；
3. 对偏离状态、load-transfer、near-slip 和 partial-failure 状态在线查询 teacher；
4. 聚合数据并继续训练；
5. teacher 不可行时使用安全 recovery target，而不是输出异常 wrench；
6. 后期逐渐减少对 teacher action imitation 的依赖，保留 effect、goal 和 safety objective。

---

## 15. 安全投影

### 15.1 为什么 penalty 不够

Microspine 一旦超过失效边界，可能发生不可恢复 detach。有限 penalty 只能改变期望回报，不能保证单步 action 安全。因此对以下条件应采用硬限幅、QP projection、reference governor 或 early termination：

- joint position/velocity/rate limit；
- command increment 和 jerk limit；
- 已挂接方向的接触运动学约束；
- 预测 contact wrench limit；
- proof-load 和 unload rate；
- 当前剩余接触的平衡可行性。

### 15.2 Position-level safety projection

\[
\begin{aligned}
q_{d,t}^{safe}
=\arg\min_{q_d}\quad&
\|q_d-q_{d,t}^{\pi}\|_{W_q}^2\\
\text{s.t.}\quad&
q_{min}\leq q_d\leq q_{max},\\
&|q_d-q_{d,t-1}|\leq\Delta q_{max},\\
&\|A_i\Delta x_i(q_d;q)\|\leq\epsilon_{c,i},
\quad i\in\mathcal C_{hard},\\
&\hat\rho_i(q_d,b_i)\leq\beta_i,
\quad i\in\mathcal C_{load},\\
&\text{floating-base equilibrium feasible},\\
&\text{collision and actuator constraints}.
\end{aligned}
\]

其中 \(\Delta x_i(q_d;q)\) 是从当前构型到候选位置目标所预测的接触位姿变化，\(A_i\) 只选择该 contact 当前真正约束的方向；touch/preload/scrub 阶段不能直接当成完整 6D 刚性接触。实际 QP 中可用 Jacobian 对 \(\Delta x_i\) 线性化。

### 15.3 分层安全实现

- **局部 gripper 层：** preload、scrub、slip detection、紧急卸载；
- **全身 projection 层：** equilibrium、load sharing、joint/contact constraints；
- **高层 FSM：** 只在 proof-load 通过且剩余接触可行时释放旧接触；
- **机械层：** 被动柔顺、tendon load sharing、结构冗余。

---

## 16. 训练流程

### 阶段 0：单 gripper 与执行器标定

1. 测量不同表面、接近角、preload、scrub 和 pulling direction；
2. 拟合 engagement probability、patch stiffness 和随机 limit surface；
3. 辨识位置伺服 gain、delay、saturation、friction 和 backlash；
4. 明确真机可用的 contact/load observations；
5. 建立仿真参数 posterior 或合理 DR 分布。

### 阶段 1：固定 contact sequence 的 teacher

1. 人工指定移动肢、新抓点和 contact FSM；
2. 使用 quasi-static optimizer、OSC 或 WBC 求 body/EE trajectory；
3. 计算 \(w_G^\star\)、load-sharing target 和安全裕度；
4. 先在标称模型验证可行性，再加入小范围 contact randomization。

### 阶段 2：DeepMimic 风格的 position student 初始化

1. Policy 输出 \(\Delta q_d\)；
2. 以 motion/posture tracking 为主要稠密信号；
3. 逐步加入 net wrench 和 effect reward；
4. 对 contact transition 使用 phase-dependent weighting；
5. 使用 asymmetric critic 和 observation history。

### 阶段 3：Wrench/effect-guided RL

1. 提高 \(r_{net}\)、\(r_{effect}\) 和 \(r_{margin}\) 权重；
2. 将 motion tracking 降为 regularizer；
3. 加入 internal/load-sharing target；
4. 对不可恢复 limit violation 采用 hard termination；
5. 引入 command smoothness、energy 和 progress reward。

### 阶段 4：DAgger/state-aligned teacher

1. 让 student 访问自己的状态分布；
2. 在偏离状态上在线查询 OSC/WBC teacher；
3. 聚合 near-slip、failed proof-load 和 load-transfer 数据；
4. 对 teacher 不可行状态增加 recovery policy 或 fallback label。

### 阶段 5：DR 与 failure curriculum

1. 扩大 actuator 和 contact DR；
2. 加入 episode 中途的 contact degradation；
3. 从单个 patch 轻微减弱逐渐扩展到 partial failure；
4. 训练 CoM/load transfer、re-scrub 和 slow-down 行为；
5. 以 CVaR、worst-contact margin 或 failure-focused sampling 强化尾部风险。

### 阶段 6：Sim-to-real 与安全部署

1. 先做固定机架和单接触验证；
2. 再做多接触静态承载与小幅 CoM 调整；
3. 验证 proof-load、load transfer 和 release；
4. 使用保守 limit surface 与低速命令；
5. 逐步开放更大 residual 和更复杂接触转换；
6. 用真机数据更新 actuator/contact posterior。

---

## 17. Baseline 与消融实验

### 17.1 必须比较的 baseline

1. **Motion-only DeepMimic**  
   仅跟踪 \(q^\star,\dot q^\star\) 和任务进度。

2. **Motion + exact per-contact wrench tracking**  
   直接最小化 \(\|\lambda-\lambda^\star\|^2\)，用于验证逐接触点目标的局限。

3. **Motion + net wrench tracking**  
   跟踪 \(G\lambda\approx w_G^\star\)。

4. **Net wrench + load-sharing**  
   加入 \(\alpha\) 或少量 \(\eta\) 坐标。

5. **Analytic admittance controller**  
   固定 \(M_a,D_a,K_a\) 的经典 force-to-motion baseline。

6. **Position-WBC / SEIKO-like optimizer**  
   不使用 RL，通过解析柔顺与 equilibrium optimization 间接调节 wrench。

7. **Torque-to-position imitation**  
   直接以 \(\tau_{OSC}^\star\) 为主要监督，比较跨接口 torque imitation 与 effect distillation。

8. **Residual RL without OSC wrench intent**  
   只使用 goal、motion 和 safety reward，验证 teacher intent 的价值。

### 17.2 关键消融

- 去掉 net wrench reward；
- 去掉 effect reward；
- 用 exact \(\lambda_i\) tracking 替代 set-valued safety；
- 去掉 load-sharing/internal coordinate；
- 去掉 limit-surface margin；
- 去掉 motion/posture regularizer；
- 固定时间参考 vs. phase/event reference；
- 单帧 MLP vs. history/RNN；
- 隐式 belief vs. 显式 contact observer；
- 仅 reset DR vs. episode 中途 contact degradation；
- 离线 teacher vs. state-aligned online teacher；
- 无 safety projection vs. safety projection；
- 直接 \(q_d\) vs. \(q_{WBIK}+\Delta q_d\)；
- joint residual vs. task-space residual；
- 标称期望代价 vs. CVaR/worst-case risk objective。

### 17.3 评价指标

#### 任务表现

- climbing success rate；
- 达到目标高度或抓点的时间；
- contact-transition success rate；
- regrasp/retry 次数；
- goal progress。

#### Wrench 与动力学效果

- net wrench RMSE；
- task-space velocity/finite-horizon effect error；
- load-sharing error；
- proof-load tracking error；
- contact impulse 峰值。

#### 安全和鲁棒性

- \(\min_i\gamma_i\) 及其分布；
- limit-surface violation rate；
- slip、partial-slip 和 detach rate；
- single-contact degradation recovery rate；
- 级联失效率；
- N−1 contact feasibility success rate。

#### 控制质量

- position command rate/acceleration/jerk；
- position servo saturation rate；
- estimated energy；
- safety projection 修改幅值和触发频率；
- 真机与仿真的 effect gap。

---

## 18. 新颖性判断

### 18.1 单个组成部分的新颖性有限

以下单点通常不足以构成主要方法贡献：

- 用 RL 输出位置目标；
- 用 force tracking reward 训练 position policy；
- learned admittance 或 force-to-motion mapping；
- OSC/WBC 作为 teacher；
- residual RL；
- domain randomization；
- 位置控制机器人进行多接触 load regulation；
- microspine 的随机抓持规划。

因此不能把贡献简单表述为：

> “首次将 task-space force 映射为 position command。”

这一表述过于接近 learned admittance 和 force-to-motion policy。

### 18.2 有潜力的新颖组合

更强的研究问题是：

> **如何将 state-aligned floating-base OSC 的 task/contact wrench intent 蒸馏为 position-only whole-body policy，使其在随机、部分可观测的 microspine 多接触下复现 net wrench 与任务空间动力学效果，并保持概率 limit surface 安全裕度。**

潜在核心贡献包括：

1. **Effect-level cross-interface distillation**  
   不拟合不可直接执行的 torque，而是蒸馏 net wrench、内部载荷意图和 task-space dynamics effect。

2. **Set-valued multi-contact wrench imitation**  
   使用 \(G\lambda\)、少量 internal coordinates 和 limit-surface set membership，避免强制模仿非唯一的逐接触 wrench。

3. **Belief-conditioned microspine adaptation**  
   根据 engagement、limit surface 和 contact mode 的 belief 在线调整 position command、CoM、load sharing 和 phase。

4. **Partial-failure recovery**  
   在单个 patch 承载能力突然下降时，student 能减速、转移载荷、移动 CoM 或重新 engagement。

5. **State-aligned teacher under interface mismatch**  
   Teacher 在 student 当前状态上生成 intent，同时 student rollout 始终使用 randomized position actuator。

### 18.3 新颖性强度的务实判断

- 裸方案：`desired wrench → RL → joint position`，新颖性较弱，但工程上合理。
- 加入 net/internal decomposition、effect tracking 和 position-interface mismatch：中等。
- 再加入 stochastic limit surface、contact belief、state-aligned teacher、partial-failure recovery 和真机验证：有望形成较强的系统与方法贡献。

正式声称“首次”之前，仍需系统检索 learned admittance、force-conditioned whole-body policy、contact-wrench imitation、cross-interface distillation、position-controlled multi-contact WBC 和 microspine climbing 文献。

推荐的论文定位：

> **Belief-Conditioned Wrench-Effect Distillation for Position-Controlled Floating-Base Climbing under Stochastic Microspine Contacts**

---

## 19. 推荐的第一版实现（MVP）

### 19.1 范围控制

第一版不做完整动态攀爬和自动 contact-sequence 搜索，只验证核心假设：

- 固定接触顺序；
- 慢速、准静态 body motion；
- 3–4 个已承载 gripper + 1 个自由肢；
- 简化的 patch-level stochastic limit surface；
- 位置控制 actuator model；
- 可控的单个 patch 承载能力下降。

### 19.2 Teacher reference

\[
z_t^\star=
\left[
w_{G,t}^\star,
\alpha_t^\star,
x_{B,t}^\star,
x_{free,t}^\star,
m_t^\star,
\phi_t,
\gamma_t^\star
\right].
\]

第一版可暂不直接输入 OSC 的完整 \(F_{task}^\star\)，而使用更稳定、更容易验证的 \(w_G^\star\)、load-sharing target 和 motion envelope。

### 19.3 Student

\[
\Delta q_{d,t}=\pi_\theta(o_{t-H:t},z_t^\star),
\]

\[
q_{d,t}
=q_{WBIK,t}^\star
+s_q\tanh(\Delta q_{d,t}).
\]

建议使用 GRU actor 和 asymmetric critic。

### 19.4 第一版 reward

\[
\begin{aligned}
r_t={}&
w_G\exp\left(-\|G\bar\lambda-w_G^\star\|_{Q_G}^2\right)\\
&+w_\alpha\exp\left(-\|\alpha-\alpha^\star\|_{Q_\alpha}^2\right)\\
&+w_e\exp\left(-\|\dot x_B-\dot x_B^\star\|_{Q_B}^2
-\|\dot x_{free}-\dot x_{free}^\star\|_{Q_f}^2\right)\\
&+w_x\exp\left(-\|x-x^\star\|_{Q_x}^2\right)
+w_{goal}r_{progress}\\
&-w_\gamma\sum_i[u_i-\beta]_+^2
-w_s c_{slip}
-w_d c_{detach}\\
&-w_a\|\Delta q_{d,t}\|^2
-w_{\Delta a}\|\Delta q_{d,t}-\Delta q_{d,t-1}\|^2\\
&-w_{acc}\|q_{d,t}-2q_{d,t-1}+q_{d,t-2}\|^2.
\end{aligned}
\]

### 19.5 第一版 curriculum

1. 标称 actuator + 固定可靠接触，学习 motion 和 net wrench；
2. 随机 stiffness 和 servo gain；
3. 随机 limit surface 大小和方向；
4. 加入 observation noise、delay 和 history；
5. episode 中途减弱单个 patch；
6. 学习 load transfer 和 CoM residual；
7. 加入 proof-load 与 event-based phase transition。

### 19.6 第一版成功标准

相对于 motion-only 和 analytic admittance baseline，至少展示：

- 更低的 net wrench error；
- 更大的 worst-contact safety margin；
- 更低的 slip/detach rate；
- 在随机 actuator/contact 参数下更高的成功率；
- 单 patch 承载能力下降后的载荷转移；
- 位置命令保持平滑且不过度依赖 safety projection。

---

## 20. 后续扩展

在 MVP 验证后，可逐步加入：

1. RL 输出 task-space residual，而非 joint residual；
2. contact planner 或 MCTS 搜索抓点与接触序列；
3. sampling-based trajectory optimization 生成 risk-aware teacher；
4. belief-dependent MPPI/CEM 在线局部修正；
5. CVaR 或 chance-constrained teacher；
6. 显式 engagement observer 与 uncertainty calibration；
7. 多表面、多 gripper 和跨机器人泛化；
8. 真实机器人 partial-failure recovery；
9. teacher-free deployment；
10. safety projection 的可行性与稳定性分析。

---

## 21. 关键公式汇总

### 浮动基动力学

\[
M(q)\dot v+h(q,v)
=S^\top\tau+J_c^\top\lambda.
\]

### Student 的位置控制 action

\[
q_{d,t}
=q_{WBIK,t}^\star
+s_q\tanh\left(\pi_\theta(o_{t-H:t},z_t^\star)\right).
\]

### 多接触 net wrench

\[
w_G=G(q)\bar\lambda.
\]

### Contact wrench 分解

\[
\lambda=G^\dagger w_G+N_G\eta.
\]

### Limit-surface utilization 与 margin

\[
u_i=\rho_{\mathcal W_i(b_i)}(\bar\lambda_i),
\qquad
\gamma_i=1-u_i.
\]

### 安全集合

\[
u_i\leq\beta<1.
\]

### Net wrench reward

\[
r_{net}
=\exp\left(
-\|G\bar\lambda-w_G^\star\|_{Q_G}^2
\right).
\]

### Effect reward

\[
r_{effect}
=\exp\left(
-\|\dot x_B-\dot x_B^\star\|_{Q_B}^2
-\|\dot x_{ee}-\dot x_{ee}^\star\|_{Q_{ee}}^2
\right).
\]

### Contact belief

\[
b_{i,t}
=P\left(
m_i,N_{engaged,i},\Theta_i,a_i,K_i
\mid y_{0:t}
\right).
\]

### Chance constraint

\[
P_{\Theta_i\sim b_i}
\left(
\lambda_i\in\mathcal W_i(\Theta_i)
\right)
\geq1-\epsilon.
\]

### Phase-dependent reference

\[
z_t^\star=z^\star(\phi_t,m_t).
\]

### Position safety projection

\[
q_{d,t}^{safe}
=
\arg\min_{q_d}
\|q_d-q_{d,t}^{\pi}\|^2
\quad
\text{s.t. contact、equilibrium、joint 与 rate constraints}.
\]

---

## 22. 最终结论

本方案不是让 position policy 机械地复制 OSC torque，也不是用 wrench reward 完全替换 motion tracking。更准确的定义是：

\[
\boxed{
\text{以 floating-base OSC 的 task/contact wrench program 为物理参考，}
\text{让 position-level policy 复现其 net wrench 与任务空间动力学效果，}
\text{并允许在 stochastic microspine 安全集合内进行目标导向的适应。}
}
\]

最终推荐的控制逻辑为：

\[
\boxed{
\text{state-aligned OSC intent}
+
\text{net/internal wrench decomposition}
+
\text{position-only recurrent student}
+
\text{phase-dependent reward}
+
\text{contact belief}
+
\text{limit-surface safety projection}
}
\]

该方案保留了 OSC 的物理结构和多接触动力学知识，同时尊重真实机器人只能执行 position command 的硬件事实。其方法新颖性不来自简单的 wrench-to-position 映射，而来自 **跨控制接口的 physical-effect distillation、随机 microspine 的 set-valued wrench imitation、contact-belief-conditioned adaptation，以及部分失效下的安全载荷重分配**。
