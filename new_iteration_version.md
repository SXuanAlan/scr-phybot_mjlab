# MuJoCo 中模拟 Microspine 的两种方法


## 方法 1：Contact-Triggered Equality Constraint

### 核心思想

利用 MuJoCo 内置的 equality constraint（等式约束）中的 `connect` 类型来模拟锁合。`connect` 约束的作用是强制两个 body 上各自的一个点保持重合——本质上等价于在该点焊接了一个球铰链。

关键创新在于：**不是让约束始终开启，而是通过 Python 控制器在运行时动态激活和关闭它**。当检测到 spine 与墙面的接触力满足 engage 条件时，激活约束，将 spine 锚定在接触点；当检测到拉力超标时，关闭约束，恢复自由状态。

### 物理映射

| 真实物理过程 | MuJoCo 中的对应 |
|---|---|
| 弹簧机构将 spine 压向墙面 | actuator 或 spring joint 施加预压力 |
| 尖端嵌入墙面粗糙度 | 检测 `data.contact` 中法向力超过阈值 |
| 机械锁合，承受大切向力 | 激活 `connect` equality constraint |
| 法向拉力超过嵌入强度，脱离 | 检测条件不满足后禁用 constraint |

### 状态机设计

控制器维护一个四状态的有限状态机：

**IDLE（空闲）**：spine 未与墙面接触。每步遍历 `data.contact` 数组，检查是否存在 spine geom 与 wall geom 之间的接触。如果检测到接触且法向力 > 0，转入 CONTACT。

**CONTACT（接触）**：存在物理接触但尚未锁合。持续监控法向力大小。只有当法向力超过 `engage_threshold`（典型值 2–5N）并且持续一段时间（`hold_time` ≈ 10ms，用于过滤碰撞弹跳）后，才执行 engage 操作，转入 ENGAGED。如果接触丢失，回退到 IDLE。

**ENGAGED（锁合）**：constraint 已激活，spine 被锚定在墙面接触点。这是核心工作状态。MuJoCo 的约束求解器会自动计算维持锚定所需的力——无论是 10N 还是 500N，都由求解器隐式处理。持续监控两个 disengage 条件：

- 条件 1：法向拉力超标。通过跟踪 body 的位置偏移（而非接触力本身，因为 constraint 也会产生力）来判断 spine 是否正在被拉离墙面。如果法向方向的位移超过阈值（如 2mm），触发 disengage。
- 条件 2：切向力超过摩擦锥。如果切向力 > μ_eff × 法向力，说明载荷已经超出了 spine 的承载能力，转入 SLIDING。

**SLIDING（滑动）**：切向过载导致 spine 在墙面上"犁"过一段距离。实现方式是更新锚点位置到当前接触点，然后重新尝试锁合（回到 ENGAGED）。如果累积滑移量超过阈值（如 10mm），说明 spine 已无法找到新的 asperity 嵌入点，触发完全 disengage，回到 IDLE。

### Engage 操作的关键步骤

engage 时需要完成三件事：

1. **记录接触点的世界坐标**，并将其转换为 spine body 的局部坐标。转换公式：`local = R^T × (world - body_pos)`，其中 R 是 body 的旋转矩阵，从 `data.xmat[body_id]` 获取并 reshape 为 3×3。

2. **将局部坐标写入约束锚点**。MuJoCo 的 `eq_data` 是一个扁平数组，每个约束占 11 个 float（常量 `mjNEQDATA = 11`）。对于 `connect` 类型，前 3 个是 body1 上的锚点局部坐标，接下来 3 个是 body2 上的锚点局部坐标。所以写入位置是 `model.eq_data[eq_id * 11 : eq_id * 11 + 3]`。同样需要把世界坐标转换为 wall body 的局部坐标后写入 `eq_data[eq_id * 11 + 3 : eq_id * 11 + 6]`。

3. **激活约束**：设置 `model.eq_active[eq_id] = 1`。从这一刻起，MuJoCo 的 PGS 或 CG 求解器会在每步自动计算维持两个锚点重合所需的约束力。

### 接触力提取

每步需要遍历 `data.contact` 数组来提取力信息：

- 遍历 `data.ncon` 个接触，通过 `contact.geom1` 和 `contact.geom2` 匹配 spine-wall 接触对。
- 调用 `mj_contactForce(model, data, contact_index, force_buffer)` 获取接触坐标系下的力。`force[0]` 是法向力（正值 = 压入），`force[1:3]` 是切向力。
- 接触坐标系的基向量在 `contact.frame` 中：`frame[0:3]` 是法向量，`frame[3:6]` 和 `frame[6:9]` 是两个切向量。
- 如果 geom1 和 geom2 的顺序与预期相反，法向量方向也相反，需要翻转。

### XML 模型设计要点

模型中需要以下关键元素：

**接触对定义**：通过 `<contact><pair>` 显式指定 spine geom 和 wall geom 之间的接触参数。即使约束会接管主要的力传递，接触对仍然必须存在——它提供 engage/disengage 的判断信号（法向力和接触点位置）。`friction` 可以设为中等值（如 5–8），因为 constraint 激活后它不再是主要承载机制。`solref` 和 `solimp` 需要设得较硬（小 timeconst、大 dmin/dmax），以确保接触检测灵敏。

**Equality constraint 定义**：使用 `<equality><connect>` 类型，指定 `body1`（spine 的接触部件）和 `body2`（墙面 body）。`active="false"` 确保初始状态关闭。`anchor` 的初始值不重要，因为 Python 会在 engage 时动态设置。`solref` 和 `solimp` 控制约束的刚度——更硬的约束（更小的 timeconst）意味着更小的穿透/偏移，但也需要更小的 timestep 来保持数值稳定。

**Spine compliance**：在 spine body 和接触面之间可以加一个小行程的 slide joint（range ≈ ±5mm），配合高刚度弹簧（stiffness ≈ 5000 N/m），模拟 spine 弹簧机构的柔顺性。这不是必须的，但可以改善接触的数值稳定性，避免硬碰硬的冲击。

### 优势

- **与 MuJoCo 求解器完全集成**：约束力由凸优化求解器隐式计算，天然满足能量守恒和力的互补性。humanoid 的关节控制器、其他接触、重力等全部协调处理，无需额外考虑。
- **数值稳定**：可以使用较大的 timestep（0.001s），因为力计算由求解器内部处理。
- **多 spine 场景自然扩展**：每个 spine 只需一组对应的 geom + constraint + 控制器实例，彼此通过求解器自动耦合。
- **简洁**：核心控制逻辑只涉及三个 API 调用（读 contact、写 eq_data、写 eq_active），容易调试和维护。

### 局限

- **二值状态**：engage/disengage 是跳变的。真实 microspine 在嵌入过程中有渐进的力增长，这里无法表达。
- **各向同性**：`connect` constraint 在所有方向上等效锁定，无法模拟"向下抓紧、向上松开"的方向性。虽然可以通过状态机中的 disengage 条件部分补偿（例如在向上方向设更低的 disengage 阈值），但本质上 engaged 状态内的力特性是对称的。
- **锚点更新有延迟**：SLIDING 状态下更新锚点后，constraint 需要几步才能稳定到新位置，这期间可能有轻微的力跳变。

### 参数调优指南

如果 300N 测试仍不稳定，按以下顺序排查：

1. **constraint 太软**：减小 `solref` 的 timeconst（如从 0.001 降到 0.0005），增大 `solimp` 的 dmin 和 dmax（如 0.99, 0.999）。
2. **接触检测不灵敏**：增大 contact pair 的 `margin`（如 0.002），让 MuJoCo 提前检测接触。
3. **engage 太慢**：减小 `hold_time`（甚至设为 0），加快 engage 响应。
4. **disengage 太敏感**：增大法向拉力阈值和切向力的 μ_eff。
5. **timestep 太大**：如果 constraint 的 solref 很硬，可能需要减小 timestep 到 0.0005s。

---

## 方法 3：自定义力模型（Custom Force Model）

### 核心思想

完全绕过 MuJoCo 的内置接触引擎，由 Python 在每个仿真步中手动计算所有接触力，然后通过 `data.xfrc_applied` 直接施加到 body 上。

这与"简单地 apply 一个固定外力"有本质区别。简单方法是人为指定一个常数（如"施加 300N 向上"），不随状态变化。本方案建立了一个完整的弹簧-阻尼-摩擦物理模型：力的大小和方向完全取决于 spine 的穿透深度、相对速度、与锚点的偏移量、以及 engage/disengage 状态。

### 数学模型

#### 法向力（Normal Force）

法向力基于经典的 Kelvin-Voigt 弹簧-阻尼模型：

> F_n = k_n × δ - c_n × v_n

其中 δ 是穿透深度（正值 = 穿透），v_n 是法向速度（正值 = 靠近墙面）。k_n 是法向刚度（N/m），c_n 是法向阻尼（N·s/m）。

穿透深度的计算：对于平面墙面，`δ = -(p - p_wall) · n_wall`，其中 p 是 spine 接触点的世界坐标，p_wall 是墙面上的一点，n_wall 是墙面外法向量。δ > 0 表示穿透。

法向力限制为非负：`F_n = max(F_n, 0)`。这确保了法向方向只有推力、没有拉力（adhesion 阶段例外，见后文）。

法向力方向沿墙面外法向量，将 spine 推离墙面。

#### 切向力（Tangential Force）— Engage 前

在 engage 之前（即 δ > 0 但 δ < engage_depth），只有基础的动摩擦力：

> F_t = μ_kinetic × F_n × (-v_t / |v_t|)

其中 v_t 是切向速度，μ_kinetic 是动摩擦系数（典型值 0.5–1.0）。力方向与切向速度相反。

#### 切向力（Tangential Force）— Engage 后

engage 后（δ ≥ engage_depth），切向力模型切换为弹簧-摩擦混合模型。这是本方案最核心的部分。

首先记录 engage 时 spine 接触点在墙面切平面上的投影位置作为锚点（anchor）。然后在每步计算当前位置与锚点的切向偏移量 Δx_t。

切向弹簧力：

> F_t_spring = k_t × |Δx_t|

切向阻尼力：

> F_t_damping = c_t × |v_t|

摩擦锥限制：

> F_t_max = μ_eff × F_n

接下来判断 stick 还是 slip：

- 如果 `F_t_spring + F_t_damping ≤ F_t_max`：处于**静摩擦（stick）**状态。实际施加的力等于弹簧力加阻尼力，方向从当前位置指向锚点。spine 不会滑动。
- 如果 `F_t_spring + F_t_damping > F_t_max`：处于**动摩擦（slip）**状态。实际施加的力被截断到 F_t_max。同时更新锚点位置——沿滑动方向移动 `(F_t_spring - F_t_max) / k_t` 的距离，使弹簧力在下一步降到摩擦锥内。

这个"弹簧 + 截断 + 更新锚点"的模式叫做 **Regularized Coulomb Friction**（正则化库仑摩擦），在接触力学仿真中是标准方法。

#### 各向异性摩擦

这是方法 3 独有的能力。计算滑动方向与预定义的"抓取轴"（grasp_axis）的点积：

> alignment = (Δx_t / |Δx_t|) · grasp_axis

- `alignment > 0`（滑动方向与抓取方向一致，如向下滑）：使用高摩擦系数 μ_grasp，典型值 10–15。这模拟了钩爪嵌入 asperity 后的强锁合。
- `alignment < 0`（滑动方向与抓取方向相反，如向上推）：使用低摩擦系数 μ_release，典型值 1.0–2.0。这模拟了钩爪反向容易脱出的特性。

对于竖直墙面上的爬行机器人，grasp_axis 通常设为 (0, 0, -1)（向下），意味着抵抗重力方向有高 μ，而向上推动（如主动脱离）时 μ 很低。

#### 粘附力（Adhesion）

真实 microspine 在脱离瞬间不是瞬间失力，而是有一段短暂的法向粘附力。模型如下：

当 δ < 0（已离开墙面）但间隙 |δ| 小于 adhesion_depth（典型值 1mm）时：

> F_adhesion = min(k_adhesion × |δ|, F_adhesion_max)

方向朝墙面（负法向），试图将 spine 拉回。当间隙超过 adhesion_depth 时，力归零，完全脱离。

这个模型对于模拟 disengage 过程中的渐进力衰减非常重要——spine 不是一瞬间从"300N 承载"变成"0N"，而是在 1–2mm 的行程内平滑过渡。

### 状态机设计

与方法 1 类似但更精细，共五个状态：

**FREE**：远离墙面（δ < 0 且无粘附）。无力施加。

**CONTACT**：有穿透（δ > 0）但未达到 engage 深度。只施加法向弹簧力和基础动摩擦力。

**ENGAGED_STICK**：已 engage 且处于静摩擦区。切向力由弹簧模型提供，spine 不滑动。这是稳定承载的主要工作状态。

**ENGAGED_SLIP**：已 engage 但切向力超过摩擦锥。spine 在墙面上滑动，锚点持续更新。如果累积滑移量超过阈值（如 15mm），转为完全 disengage。

**ADHESION**：刚脱离墙面，间隙在 adhesion_depth 以内。施加拉回力。间隙超过阈值后转为 FREE。

### 力的施加方式

计算得到的总力向量（法向 + 切向 + 粘附）写入 `data.xfrc_applied[body_id, 0:3]`。

如果接触点不在 body 质心（通常不在），还需要计算力矩：`τ = r × F`，其中 r 是接触点相对于质心的世界坐标偏移。力矩写入 `data.xfrc_applied[body_id, 3:6]`。

需要注意的是：`xfrc_applied` 中的力会与其他外力（包括来自 actuator 的预压力和拉力）叠加。所以在每步开始时，需要先清零 `xfrc_applied`，再写入所有外力（包括自定义接触力和测试载荷）。

### XML 模型设计要点

由于完全绕过 MuJoCo 接触引擎，XML 模型非常简单：

- 所有 geom 的 `contype` 和 `conaffinity` 设为 0，禁用所有内置碰撞检测。
- 不需要 `<contact>` 或 `<equality>` 部分。
- 墙面只是一个可视化的 geom（用于渲染），物理上通过 `WallGeometry` 类的解析几何来处理。
- Spine 是一个 freejoint body，只需要 mass 和 inertia。

### 优势

- **完全可控**：每个力分量（法向弹簧、法向阻尼、切向弹簧、切向摩擦、粘附）都可以单独可视化、记录和调试。可以清楚看到 300N 载荷下每个分量承担了多少。
- **精确匹配实验数据**：如果你有测力计数据（force-displacement 曲线），可以直接调 k_normal、k_tangential、μ_eff 等参数去拟合。每个参数都有明确的物理含义。
- **各向异性**：可以在任意方向设不同的摩擦系数，完美模拟钩爪的方向性。
- **粘附力建模**：可以精确控制 disengage 过程的力衰减特性。
- **状态连续性**：STICK → SLIP 的过渡是通过弹簧截断实现的，没有力的跳变。

### 局限

- **需要小 timestep**：弹簧-阻尼系统的稳定性条件要求 `dt < 2 × √(m/k)`。对于 k_normal = 50000 N/m 和 m = 0.1 kg，临界 dt ≈ 0.003s，安全 dt 约 0.0005s。比方法 1 的 0.001s 小一倍以上，计算量加倍。
- **绕过接触图**：`xfrc_applied` 的力不参与 MuJoCo 的接触求解。如果 humanoid 的其他部位也需要与墙面交互（如手、膝盖触墙），那些接触需要用 MuJoCo 的正常接触机制单独处理。两套系统共存会增加复杂度。
- **能量可能不守恒**：弹簧-阻尼模型在大 timestep 下可能注入能量（velocity-Verlet 对阻尼项的近似误差）。需要通过小 timestep 和适当的阻尼来控制。
- **墙面几何受限**：当前实现假设墙面是无限平面。对于曲面或有凸起/凹槽的墙面，需要扩展 `WallGeometry` 类来处理更复杂的距离计算。

### 参数标定方法

对于每个物理参数，建议的实验标定流程：

| 参数 | 标定方法 |
|---|---|
| k_normal | 用力传感器测量 spine 法向力-位移曲线，取线性段斜率 |
| c_normal | 从碰撞恢复系数 e 计算：`c = -2×ln(e)×√(k×m) / (π² + ln²(e))` |
| engage_depth | 显微观察 spine 尖端嵌入表面的深度，或从力-位移曲线的拐点确定 |
| k_tangential | engage 后施加渐增切向力，测量力-位移曲线的初始斜率 |
| μ_eff | engage 后持续增加切向力直到滑动，F_t_max / F_n 即为 μ_eff |
| μ_grasp / μ_release | 分别在抓取方向和释放方向测量上述 μ_eff |
| adhesion_depth | 测量 spine 从表面拔出过程中力降为零的位移 |
| k_adhesion | 拔出过程的力-位移曲线斜率 |

---

## 两种方法的选择决策

### 对比总结

| 维度 | 方法 1（Constraint） | 方法 3（Force Model） |
|---|---|---|
| 力的来源 | MuJoCo 约束求解器隐式计算 | Python 手动计算弹簧-阻尼-摩擦 |
| 与 MuJoCo 集成度 | 完全集成，参与全局求解 | 独立于接触引擎，通过 xfrc_applied 施加 |
| 推荐 timestep | 0.001s | 0.0005s 或更小 |
| 各向异性支持 | 不支持 | 原生支持 |
| 粘附力支持 | 不支持 | 原生支持 |
| Engage/disengage 过渡 | 二值跳变 | 平滑过渡 |
| 能量守恒 | 由求解器保证 | 取决于 timestep 和阻尼参数 |
| 与实验数据标定 | 间接（通过 engage/disengage 阈值） | 直接（参数 ↔ 物理量一一对应） |
| 多 spine 耦合 | 通过求解器自动耦合 | 需要手动处理 spine 间的力交互 |
| 实现复杂度 | 中等 | 高 |
| 调试可见性 | 低（力在求解器内部） | 高（每个力分量可独立输出） |

### 决策树

**选方法 1，如果：**

- humanoid 有多条腿同时在墙上，需要全身动力学协调（whole-body controller / QP-based controller）
- 仿真中有大量 spine 同时工作（> 20 个），需要计算效率
- 不需要各向异性摩擦（或者可以通过状态机的 disengage 条件近似）
- 需要与 MuJoCo 的其他功能（如 MPC、逆动力学）无缝配合

**选方法 3，如果：**

- 需要精确匹配单个 spine 的实验测力数据
- 各向异性是关键需求（如钩爪方向性对控制策略有影响）
- 需要精确建模 disengage 过程（如粘附力对脱离时机的影响）
- spine 数量少（< 10 个），计算预算充裕
- 需要对力模型做大量参数扫描或敏感性分析

**混合方案：**

两种方法可以组合使用。用方法 3 的力模型在单 spine 级别标定出物理参数（k_normal、μ_eff、engage_depth 等），然后将这些参数转化为方法 1 的 engage/disengage 阈值和 constraint 刚度，在全身仿真中使用方法 1 的 constraint 方案来获得与 MuJoCo 求解器的完全集成。

---

## 附录：MuJoCo API 速查

### 方法 1 核心 API

| 操作 | API |
|---|---|
| 获取对象 ID | `mujoco.mj_name2id(model, mjtObj.mjOBJ_GEOM, "name")` |
| 接触数量 | `data.ncon` |
| 接触 geom 对 | `data.contact[i].geom1`, `data.contact[i].geom2` |
| 接触点世界坐标 | `data.contact[i].pos` |
| 接触力（接触坐标系） | `mujoco.mj_contactForce(model, data, i, force_buf)` |
| 接触坐标系法向量 | `data.contact[i].frame[0:3]` |
| Body 世界位置 | `data.xpos[body_id]` |
| Body 旋转矩阵 | `data.xmat[body_id].reshape(3, 3)` |
| 激活约束 | `model.eq_active[eq_id] = 1` |
| 关闭约束 | `model.eq_active[eq_id] = 0` |
| 设置约束锚点 | `model.eq_data[eq_id * 11 : eq_id * 11 + 3] = local_point` |
| 设置约束刚度 | `model.eq_solref[eq_id] = (timeconst, dampratio)` |

### 方法 3 核心 API

| 操作 | API |
|---|---|
| Body 世界位置 | `data.xpos[body_id]` |
| Body 6D 速度 | `data.cvel[body_id]`（前 3 = 角速度，后 3 = 线速度） |
| 施加外力 | `data.xfrc_applied[body_id, 0:3] = force_vector` |
| 施加外力矩 | `data.xfrc_applied[body_id, 3:6] = torque_vector` |
| 清零外力 | `data.xfrc_applied[:] = 0` |
| freejoint 位置 | `data.qpos[qposadr : qposadr + 3]` |
| freejoint 四元数 | `data.qpos[qposadr + 3 : qposadr + 7]` |