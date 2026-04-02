# MuJoCo 官方 XML 示例分析：Microspine 仿真可借鉴的技术

## 分析对象

本文档分析了四个 MuJoCo 官方示例 XML 文件，评估其中哪些技术可以直接应用于 microspine 在竖直墙面上的仿真建模。

| 文件 | 来源 | 核心功能 |
|---|---|---|
| gripper.xml | DeepMind MuJoCo 示例 | FEM 可变形抓取面 + 高级摩擦配置 |
| gripper_trilinear.xml | DeepMind MuJoCo 示例 | 轻量三线性变形 + 相同摩擦配置 |
| touch_grid.xml | MuJoCo 传感器示例 | 网格触觉传感器插件 |
| tactile.xml | MuJoCo 传感器示例 | Mesh 触觉传感器 + SDF 几何插件 |

---

## 发现 1（最关键）：椭圆摩擦锥 + 摩擦比放大

**来源**：gripper.xml 和 gripper_trilinear.xml 的 `<option>` 配置。

**原文**：

```xml
<option cone="elliptic" impratio="10" integrator="implicitfast"/>
```

**作用机理**：

MuJoCo 的接触求解器在计算摩擦力时，需要在一个几何形状（摩擦锥）内求解可行力。默认的 `cone="pyramidal"` 使用金字塔形近似，而 `cone="elliptic"` 切换为椭圆锥。椭圆锥在数学上更精确，更重要的是它与 `impratio` 参数协同工作的方式完全不同。

`impratio` 控制的是椭圆锥中不同摩擦方向的比率。当 `impratio="10"` 时，扭转摩擦（torsional friction，即 condim 的第 3 维及以上）的等效承载能力被放大到滑动摩擦的 10 倍。具体而言：

- 在 `condim=4` 的接触中，前两个切向方向（滑动摩擦）的摩擦系数保持 `friction` 参数指定的值。
- 第三个方向（扭转摩擦）的等效承载能力变为 `friction × impratio`。
- 在 `condim=6` 中，第四和第五个方向（滚动摩擦）也受 `impratio` 影响。

**对 microspine 的意义**：

这可能是解决 300N 测试的最简单方案。Microspine 的锁合本质上是一种"抓住就极难拔出"的行为。在 MuJoCo 的接触模型中，这可以近似为：滑动摩擦正常（μ ≈ 1），但扭转和滚动方向的抗力极高。`impratio="10"` 恰好实现了这个效果——不需要修改任何 Python 控制逻辑，不需要自定义力模型，只需要在 `<option>` 中加一行。

DeepMind 的 gripper 使用这个配置的动机与 microspine 完全相同：柔性抓取面一旦包裹住物体，就应该"抓住不放"，除非主动松开。力学上这等价于 microspine 嵌入墙面粗糙度后的高摩擦锁合。

**建议的实施方式**：

直接在现有模型的 `<option>` 中添加这三个属性。`integrator="implicitfast"` 提供更好的数值稳定性，代价是轻微的计算精度损失（通常可以忽略）。

---

## 发现 2：六维接触约束 + 优先级控制

**来源**：gripper.xml 中被抓取物体的 geom 配置。

**原文**：

```xml
<geom type="box" size=".05 .1 .1" pos="0 0 .1" rgba=".5 .5 0 1"
      priority="1" contype="2" condim="6"/>
```

**作用机理**：

`condim="6"` 启用完整的 6 维接触约束模型：1 个法向力 + 2 个切向滑动摩擦力 + 1 个扭转摩擦力 + 2 个滚动摩擦力。默认的 `condim=3`（法向 + 2 切向）无法阻止接触面间的相对旋转。对于 microspine 附着，spine 在墙面上不仅要抵抗切向滑动（重力方向），还要抵抗因力矩引起的滚动和扭转——`condim=6` 让求解器处理所有这些分量。

`priority="1"` 解决了接触参数冲突问题。当两个 geom 碰撞时，MuJoCo 需要决定使用哪一方的 friction、condim 等参数。默认行为是取两者的混合值。设置 `priority="1"` 后，该 geom 的参数将覆盖对方——这确保了 microspine 的高摩擦设定不会被墙面的低摩擦参数稀释。

**对 microspine 的映射**：

在 spine 的接触 geom 上设置 `condim="6"` 和 `priority="1"`，确保 spine 的摩擦参数在接触计算中占主导。配合发现 1 的 `impratio="10"`，六维约束中的扭转和滚动分量都会获得 10 倍的摩擦放大，形成一个在所有旋转方向上都"锁死"的接触。

`contype` 和 `conaffinity` 的位掩码机制可以用来隔离不同的接触组——例如让 spine 只与墙面交互（`contype="2" conaffinity="2"`），不与 humanoid 的其他部位发生碰撞干扰。

---

## 发现 3：可变形接触面（flexcomp）

**来源**：gripper.xml 的 FEM 变形抓取面和 gripper_trilinear.xml 的三线性变形版本。

**FEM 版本原文**：

```xml
<flexcomp type="mesh" file="cap.obj" pos=".16 0 -.25" dim="3"
          euler="0 -90 0" origin="0 0 0" radius=".001"
          rgba="0 .7 .7 1" mass=".5" name="left">
  <edge equality="true" solimp="0.95 0.99 0.001 0.5 2"/>
  <contact selfcollide="none" internal="false" contype="2" conaffinity="2"/>
  <pin id="0 2 3 4 5 6 7 9 11 ..."/>
</flexcomp>
```

**三线性版本原文**：

```xml
<flexcomp type="mesh" file="cap.obj" pos=".16 0 -.25" dim="3"
          euler="0 -90 0" origin="0 0 0" radius=".001"
          rgba="0 .7 .7 1" mass=".5" name="left"
          dof="trilinear">
  <elasticity young="1e4" poisson="0.1" damping=".05"/>
  <contact selfcollide="none" internal="false" contype="2" conaffinity="2"/>
  <pin id="4 5 6 7"/>
</flexcomp>
```

**两种版本的区别**：

FEM 版本（gripper.xml）使用完整的有限元方法模拟变形，通过 `<edge equality="true">` 用等式约束维持边的长度，精度高但计算量大，需要 pin 住大量顶点。

三线性版本（gripper_trilinear.xml）使用 `dof="trilinear"` 的简化变形模型，每个体素只有 8 个自由度（三线性插值），通过 `<elasticity>` 直接指定杨氏模量和泊松比，计算量小得多，只需 pin 4 个顶点。

**关键组件解释**：

`pin` 把 flexcomp mesh 的特定顶点固定到父 body 上，模拟 spine 阵列的弹簧机构与刚性支架的连接——垫子的背面被固定，正面（接触面）可以自由变形。

`selfcollide="none"` 避免垫子内部的自碰撞检测，节省计算量。`internal="false"` 不生成内部接触力。

`edge equality` 用等式约束维持 mesh 边的长度，提供结构刚度。`solimp` 的 5 个参数精细控制约束的阻抗特性。

**对 microspine 的映射**：

传统方法是用刚性 box geom 作为 spine 的接触面。flexcomp 提供了一种更物理的替代——可变形接触垫可以在接触时变形包裹墙面的微观几何，增大实际接触面积，自然地提高摩擦力承载能力。

对于 microspine 阵列，建议使用三线性版本（`dof="trilinear"`）以控制计算开销。`young` 参数对应 spine 弹簧的整体刚度——较低的值（如 1e3–1e4）允许更大的变形和更好的表面贴合，较高的值（如 1e5+）接近刚性行为。`poisson` 控制横向收缩，对 spine 行为影响不大，可以保持低值（0.1）。`damping` 控制变形的耗散，有助于数值稳定性。

如果 spine 数量多（如 humanoid 四肢上各有一组），需要权衡 flexcomp 的计算开销。每个 flexcomp 实例会增加系统的广义坐标维度，影响整体仿真速度。

---

## 发现 4：网格触觉传感器（touch_grid 插件）

**来源**：touch_grid.xml 的传感器插件配置。

**原文**：

```xml
<extension>
  <plugin plugin="mujoco.sensor.touch_grid"/>
</extension>

<sensor>
  <plugin name="touch" plugin="mujoco.sensor.touch_grid"
          objtype="site" objname="touch">
    <config key="size" value="7 7"/>
    <config key="fov" value="45 45"/>
    <config key="gamma" value="0"/>
    <config key="nchannel" value="3"/>
  </plugin>
</sensor>
```

**作用机理**：

`touch_grid` 是 MuJoCo 的内置传感器插件，将一个 site 周围的接触力投影到一个二维网格上。`size="7 7"` 定义了 7×7 = 49 个采样点，`nchannel="3"` 表示每个点输出 3 个分量（法向力 + 2 个切向力），总共产生 49×3 = 147 个传感器读数。`fov="45 45"` 定义了传感器的视场角（覆盖范围）。`gamma` 控制力衰减的锐度。

传感器绑定到一个 `site`（不是 geom），所以它不影响物理仿真，只提供观测数据。

**对 microspine 的映射**：

这是 microspine 控制策略开发的理想工具。将 `touch_grid` 绑定到每个 spine 阵列的中心 site 上，可以实时获取接触力的空间分布。

具体应用场景：

- **Engage 检测**：当 touch_grid 输出的力分布从零变为非零且均匀时，表明 spine 已成功 engage。如果力集中在某几个格子上，可能意味着只有部分 spine 嵌入成功，附着不可靠。
- **Disengage 预警**：力分布变得不均匀或总力开始下降时，提前预警即将脱离。
- **RL/MPC 的观测空间**：7×7×3 的力分布图可以直接作为强化学习或模型预测控制的观测输入，让控制器学习根据触觉反馈调整预压力和姿态。
- **多 spine 协调**：每条腿的 touch_grid 输出可以用于全身控制器的力分配——例如当某条腿的 touch_grid 显示力在减小时，其他腿需要补偿。

推荐配置：`size="5 5"` 或 `size="7 7"`（更大的网格增加计算量但提供更精细的空间分辨率），`nchannel="3"`（获取完整的 3D 力信息），`fov` 根据 spine 阵列的物理尺寸调整。

---

## 发现 5：Mesh 触觉传感器（tactile）

**来源**：tactile.xml 的传感器配置。

**原文**：

```xml
<sensor>
  <tactile geom="ball" mesh="sphere"/>
  <tactile geom="finger" mesh="box"/>
</sensor>
```

配合用于传感的 ghost geom（不参与碰撞）：

```xml
<geom name="ball" type="mesh" mesh="sphere" mass="0"
      contype="0" conaffinity="0" rgba=".5 .5 .5 0"/>
```

**作用机理**：

`<tactile>` 传感器将接触力映射到一个 mesh 的每个三角形面上，输出一个与 mesh 面数等长的向量，每个元素是该面承受的法向接触力。分辨率由 mesh 的三角形密度决定——mesh 面数越多，触觉分辨率越高。

关键实现细节是使用了"ghost geom"模式：传感用的 mesh geom 设置 `contype="0" conaffinity="0"`（不参与碰撞）、`mass="0"`（不贡献惯性），并与一个参与碰撞的实体 geom（如 sphere 或 box）重叠放置。传感器从实体 geom 收集接触力，但通过 ghost mesh 的几何来确定力在表面上的分布。

**与 touch_grid 的区别**：

touch_grid 是规则的矩形网格，分辨率固定，适合平面或近平面的接触区域。tactile 是基于任意 mesh 的，可以覆盖曲面，分辨率由 mesh 密度控制，适合复杂几何形状。

**对 microspine 的映射**：

对于 spine 阵列的调试和可视化，tactile 传感器可以提供比 touch_grid 更精细的力分布图。如果 spine 的接触面不是平面（如曲面 gripper pad），tactile 传感器的 mesh 适配性更好。

但对于控制策略开发，touch_grid 更实用——它输出的是固定大小的矩阵，便于直接输入到神经网络或控制器。tactile 的输出维度取决于 mesh 面数，且面的位置不规则，需要额外的空间编码。

推荐使用场景：开发和调试阶段用 tactile 传感器配合可视化工具检查力分布；部署阶段用 touch_grid 作为控制器输入。

---

## 发现 6：SDF 几何插件

**来源**：tactile.xml 的 SDF 齿轮插件。

**原文**：

```xml
<extension>
  <plugin plugin="mujoco.sdf.gear">
    <instance name="gear">
      <config key="alpha" value="0"/>
    </instance>
  </plugin>
</extension>

<geom type="sdf" name="gear" mesh="gear" rgba="0.4 0.4 0.4 1" pos="-1.3 0 0">
  <plugin instance="gear"/>
</geom>
```

**作用机理**：

MuJoCo 的 SDF（Signed Distance Field）插件允许用隐式函数定义复杂几何。`sdf.gear` 用解析函数生成齿轮形状，不需要外部 mesh 文件。SDF 几何参与碰撞检测时使用距离场采样而非 mesh 三角形求交，可以表达任意光滑曲面。

**对 microspine 的适用性分析**：

理论上可以用 SDF 生成带有微观粗糙度纹理的墙面——这样 spine 尖端可以物理地嵌入表面凹凸中，实现真正的几何锁合。然而实际应用中存在问题：

- 微观粗糙度的尺度（微米到亚毫米）远小于仿真的典型 timestep 下物体的运动尺度，SDF 采样密度（`sdf_initpoints`）需要极高才能捕捉这些细节。
- 计算开销与 SDF 采样点数成正比。tactile.xml 中 `sdf_initpoints="20"` 对于宏观齿轮足够，但对微观粗糙度可能需要数百到数千个采样点。
- SDF 碰撞检测的稳定性在高曲率（尖锐粗糙度尖端）处会下降。

**结论**：SDF 不推荐用于 microspine 的粗糙度建模。等效摩擦系数方法（发现 1 和 2）或约束方法在计算效率和稳定性上都远优于几何粗糙度建模。SDF 的价值更多在于：如果需要模拟 spine 在非平面表面（如岩石、管道外壁）上的附着，可以用 SDF 定义墙面的宏观曲率。

---

## 推荐整合方案

基于以上六个发现，建议按以下优先级整合到 microspine 模型中：

### 第一步：仅修改 XML

在现有模型的 `<option>` 中添加三个属性：

```xml
<option cone="elliptic" impratio="10" integrator="implicitfast"/>
```

将 spine 接触 geom 修改为：

```xml
<geom name="spine_contact" type="box" size="..."
      condim="6" priority="1"
      friction="1.0 1.0 0.01 0.01 0.01"/>
```

`friction` 的 5 个参数分别是：滑动摩擦 1、滑动摩擦 2、扭转摩擦、滚动摩擦 1、滚动摩擦 2。滑动方向设 1.0（正常值），扭转和滚动设 0.01（因为 `impratio=10` 会将其等效放大到 0.1——再乘以法向力后通常足够）。如果仍不够，可以将滑动摩擦提高到 3.0–5.0。

这一步的改动最小、风险最低，建议先单独测试 300N 是否通过。

### 第二步：添加柔顺接触面（如果第一步不够）

将 spine 的接触面从刚性 box 替换为 flexcomp 可变形垫：

```xml
<flexcomp type="grid" count="4 4 1" spacing="0.005 0.005 0.002"
          pos="..." dim="3" mass="0.05" name="spine_pad"
          dof="trilinear">
  <elasticity young="5000" poisson="0.1" damping="0.1"/>
  <contact selfcollide="none" internal="false"
           contype="2" conaffinity="2"/>
  <pin id="..."/>  <!-- 固定背面顶点到 spine body -->
</flexcomp>
```

三线性模式（`dof="trilinear"`）的计算开销可控，一个 4×4×1 的网格只增加 16×3 = 48 个自由度。变形面在压向墙面时会贴合表面，增大实际接触面积，从而提高摩擦力承载上限。

### 第三步：添加触觉传感器（用于控制策略）

在每个 spine 阵列的中心放置一个 site，绑定 touch_grid 传感器：

```xml
<extension>
  <plugin plugin="mujoco.sensor.touch_grid"/>
</extension>

<site name="spine_touch_site" pos="..." size="0.005"/>

<sensor>
  <plugin name="spine_touch" plugin="mujoco.sensor.touch_grid"
          objtype="site" objname="spine_touch_site">
    <config key="size" value="5 5"/>
    <config key="fov" value="30 30"/>
    <config key="gamma" value="0"/>
    <config key="nchannel" value="3"/>
  </plugin>
</sensor>
```

传感器的 `fov` 应根据 spine 阵列的物理尺寸调整——如果阵列直径约 5cm，`fov="30 30"` 在距离墙面 2cm 时覆盖约 5cm×5cm 的区域。

### 三步与之前方法的关系

首先测试adhesion+friction的方法，看看是否能满足的要求。在多数场景下，`cone="elliptic"` + `impratio="10"` + `condim="6"` 的组合应该足以承受数百牛顿的切向力，因为 DeepMind 的 gripper 示例正是用这个配置来实现稳定的力闭合抓取。

但如果测试还是不通过，考虑使用force model——但此时状态机的作用从"让 spine 能附着"变为"让控制器知道 spine 的附着状态"，物理承载由 `cone="elliptic"` + `impratio` 保证。