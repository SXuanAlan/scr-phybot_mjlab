# ToddlerBot：每个 limb 一个 unit 的 climbing task

已导入 **ToddlerBot 2XM**，按原 C2 的布局在左右前臂后侧、小腿前侧各安装一个现有 microspine unit，
并生成了 60°、80° 和 90° 的 climbing motion reference。沿用约定：
**每个 unit 的 preload 为 1.5 N，原 empirical failure distribution 不变**。

当前结果是：**60° 和 80° 的候选通过中位数承载条件下的逐帧准静态审核，60° 完成一次
14 s 的基础 PD 动态回放。** 80° 的 9 帧首次求解失败已通过相邻帧初值重试恢复；
90° 保留原单初值审核结果，尚未用新增重试流程复核。视频展示的是运动学 reference。
动态回放中右臂曾提前脱落并成功重挂，尚未证明随机条件下的可靠性或实机能力。

Reference 视频采用 PHYBot 原视频的棋盘格墙面、蓝灰背景和灯光，同步并排显示两个视角：
左侧为 PHYBot 的 55° 方位角、-12° 仰角，随爬升高度移动；右侧保留原来的 25° 方位角、
-10° 仰角和固定取景。视频为 1920 × 720、25 fps，不叠加标题、状态文字或 limb 标签。

| 项目 | 60° | 80° | 90° |
|---|---|---|---|
| Reference 视频 | [reference_a60.mp4](reference_a60.mp4) | [reference_a80.mp4](reference_a80.mp4) | [reference_a90.mp4](reference_a90.mp4) |
| Reference 数据 | [toddlerbot_climb_a60.npz](../../src/scr_phybot_mjab/assets/motions/toddlerbot_climb_a60.npz) | [toddlerbot_climb_a80.npz](../../src/scr_phybot_mjab/assets/motions/toddlerbot_climb_a80.npz) | [toddlerbot_climb.npz](../../src/scr_phybot_mjab/assets/motions/toddlerbot_climb.npz) |
| 逐帧审核 | [a60 audit](../../src/scr_phybot_mjab/assets/motions/toddlerbot_climb_a60.audit.json) | [a80 audit](../../src/scr_phybot_mjab/assets/motions/toddlerbot_climb_a80.audit.json) | [a90 audit](../../src/scr_phybot_mjab/assets/motions/toddlerbot_climb.audit.json) |
| 几何审核 | 通过 | 通过 | 通过 |
| 准静态承载审核，unit 强度取 50% 分位 | 351 / 351 帧通过 | 351 / 351 帧通过，含 9 帧重试恢复 | 原单初值审核 209 / 351，待重试复核 |
| 每次重新挂接所需 1.5 N 预压的静力可行性 | 32 / 32 个预压帧通过 | 32 / 32 帧通过 | 8 / 32 帧通过 |
| 动态回放 | 2 个环境完成 1 周期，4 limb 均重新挂接 | 未运行 | 未证明 |

角度相对于水平面，90° 为垂直墙。新 60° reference 的最大电机力矩利用率约 **52.2%**，
发生在 7.64 s 的腰部 `waist_act_2` 电机；线性插值各段的最大速度约为对应电机速度上限的 **27.3%**。
这些上限来自上游模型的低速力矩拟合，不能解释为可长期输出的热额定力矩。
该轨迹在中位数失效边界内的最小法向余量约为 0.449 N，尚未证明随机挂接可靠性。

80° 原先未通过的 9 帧都在 10.28–10.92 s 的右腿摆动阶段，原解的左前臂法向拉力
超出对应剪切/侧向载荷下的经验上限约 0.32–0.44 N。采用相邻可行帧的受力分配作为初值，
先投影到当前姿态的静力平衡和力矩边界，再求解原问题，9 帧全部通过。
这 9 帧新解的最大电机力矩利用率约 91.3%；全周期仍有个别帧达到力矩上限，
且部分接触力在中位数失效边界上，没有可靠性余量。未进行 80° 动态回放。
机器人轨迹、1.5 N preload、失效曲线和所有审核容差均未改变。
[原数据](archive/a80_cold_start/README.md) 和 [独立复核记录](a80_solver_diagnosis.json) 保留，
当前 audit 中同时记录首次失败解、重试初值来源及恢复结果。

## Task 与 asset

- 垂直墙诊断 task：`Mjlab-Climb-Wall-ToddlerBot-SingleUnit`
- 60° task：`Mjlab-Climb-Wall-ToddlerBot-SingleUnit-A60`
- [模型与安装参数](../../src/scr_phybot_mjab/assets/toddlerbot/README.md)
- [任务配置](../../src/scr_phybot_mjab/tasks/toddler_climb/env_cfg.py)
- [接触与 reference 状态机](../../src/scr_phybot_mjab/tasks/toddler_climb/command.py)
- [IK 与完整机器人静力审核](../../src/scr_phybot_mjab/tools/climbing_reference/toddlerbot.py)

机器人保留原模型的 30 个电机、惯性、被动关节、传动比与颈部闭链。腰部齿隙暂以
零齿隙耦合 tendon 模型近似，没有固定腰部来回避电机负载。
unit 固定在前臂和小腿上，安装朝向沿用 C2：前臂接触面朝局部 -x，抓取方向朝肘部；
小腿接触面朝局部 +x，抓取方向朝脚部。踝、腕关节可独立调整。
两臂接触点横向间距为 **36 cm**、两腿为 **24 cm**，比上一版的 30 cm / 20 cm 都增加 20%。
IK 对齐接触面法向，允许抓取方向随四肢在墙面内转动；经验表始终在实际 unit 局部坐标中查询。
每个 unit 加安装座暂计 15 g，总模型质量约 **3.829 kg**。
这些安装位置和惯性需要实物设计确认，安装座 visual 也不是加工图。
这一版已替换掌面及 45° 脚尖安装。Reference 保存安装参数指纹，旧布局的数据会被拒绝加载。

每个 limb 只有一个 contact point，挂接仅传递三维力，使用 `connect`，没有新增
局部接触力矩能力或旋转锁定。普通接触用于测量预压，沿面法向压紧力连续三个控制步
达到 1.5 N，且位置、方向满足条件后才能重新挂接。预压使用每个控制步内的接触力均值。
netforce 的符号按 MuJoCo 的 unit → wall 方向处理，普通接触载荷与 anchor 载荷相加。

Task 默认从已挂墙的初始姿态开始，**不包含从地面上墙的过程**。初始 unit 强度未经筛选；
随机强度与 loading-success 随机数在每次新挂接时采样，在整个挂接期间保持不变。
默认训练与 play 使用随机失效分布；审核的 50% 分位模式仅用于诊断。
原经验表 10 N seating 协议的元数据保留不变，1.5 N 的使用是本次约定的假设。

## Reference 的内容和审核范围

每周期 14 s、25 Hz、沿墙向上 2 cm，依次换左臂、左腿、右臂、右腿。
换点时 nominal 离墙距离为 6 mm，抬肢前渐进卸载，重新挂接后渐进恢复承载。
末尾 2 s 收回到平移后的起始姿态。NPZ 包含完整 `qpos/qvel`、30 个电机角度与
准静态力矩、四个 unit 的位置/挂接计划/接触力、预压 limb、命名顺序和周期位移。

审核逐帧求解浮动基座和所有被动自由度的平衡，并保留内部传动与闭链约束反力；
逐个限制实际电机力矩，在每个 unit 的实际局部 rake 坐标中查询经验表并检查法向拉脱上限。
几何审核采用原机器人碰撞盒及 unit 点接触代理：自碰撞和穿墙容差 0.5 mm，
承载点目标误差容差 0.5 mm，摆动点目标误差容差 2 mm，接触面法向误差容差 5°。
墙面内抓取方向是软姿态偏好，不强制保持竖直。当前最大承载点目标误差约 0.040 mm，
全程最大接触点目标误差约 0.041 mm；自碰撞深度为 0，最大穿墙深度约 0.196 mm。
检查也包含关节限位、周期关节闭合误差及保守对称电机速度—力矩包络。

这是准静态必要条件检查，不包括运动惯性、闭环跟踪误差、热降额、齿隙和随机可靠性证明。
非线性优化未通过只表示当前求解没有找到合格解，**不等于证明该姿态或其他姿态不可行**。
生成器对未通过帧最多尝试四个相邻可行帧的受力初值，在当前平衡约束和原载荷边界下投影后重算，
每次均重新检查原力矩上限及经验失效曲线。没有放宽约束或直接借用相邻帧的通过标记。
`audit.json` 保存逐帧失败原因和 reference、empirical table 的 SHA-256。
失败的候选仍会保存，但生成器返回 exit code 2；逐帧失败结果保留在 `audit.json` 中。

GPU 动态诊断使用 2 个环境、seed 42、中位数 unit 强度、零 residual action，以及
受力矩/速度上限约束的 PD 加准静态前馈。这个诊断模式关闭 loading-failure 抽签；
默认 task 仍保留完整随机失效。新布局在 60° 下完成 14 s，最大根部跟踪误差约 **5.77 mm**，
结束时四个 unit 均已挂接，全程至少三个 unit 挂接。两个环境各发生一次计划外脱落：
右臂约 6.44 s 提前释放（计划约 6.46 s 开始释放），约 8.38 s 重新挂接。
其余换点也均完成。这个单周期中位数诊断不等于随机条件下的可靠攀爬。
[逐步动态记录](replay_a60.json) 保存每个 unit 的载荷、预压和脱落状态。
仿真数值与独立环境 reset 检查通过；没有训练策略，动态结果来自物理回放日志。

13 项自动测试覆盖安装父连杆、抓取朝向、踝腕运动隔离和旧 reference 拒绝加载，
以及 80° 受力重试、恢复帧的独立全机器人力/矩平衡检查。
另有 4 环境随机接触模式和 90° task 的 0.2 s GPU 运行检查，
分别记录在 [stochastic_smoke.json](stochastic_smoke.json) 和
[vertical_smoke.json](vertical_smoke.json)；这些短步检查只验证任务运行和 reset。

## 复现

使用项目的 `env_mjlab15` Python 环境。资源已经在本地，通常不需要重新下载。

```bash
# 重新导入固定 commit 的官方资源（需要网络）
python scripts/import_toddlerbot_asset.py

# 生成并审核；未通过的候选仍保存，但返回 2；80° 重试后通过
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python scripts/gen_toddlerbot_climb_reference.py
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python scripts/gen_toddlerbot_climb_reference.py \
  --angle 60 --output src/scr_phybot_mjab/assets/motions/toddlerbot_climb_a60.npz
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python scripts/gen_toddlerbot_climb_reference.py \
  --angle 80 --kinematics src/scr_phybot_mjab/assets/motions/toddlerbot_climb_a60.npz \
  --output src/scr_phybot_mjab/assets/motions/toddlerbot_climb_a80.npz

# 渲染同步双视角 reference，无文字覆盖层（支持 .mp4 或 .png）
MUJOCO_GL=egl python scripts/render_toddlerbot_climb.py \
  --output docs/toddlerbot_climb/reference_a90.mp4
MUJOCO_GL=egl python scripts/render_toddlerbot_climb.py \
  --reference src/scr_phybot_mjab/assets/motions/toddlerbot_climb_a60.npz \
  --output docs/toddlerbot_climb/reference_a60.mp4
MUJOCO_GL=egl python scripts/render_toddlerbot_climb.py \
  --reference src/scr_phybot_mjab/assets/motions/toddlerbot_climb_a80.npz \
  --output docs/toddlerbot_climb/reference_a80.mp4

# 60° 中位数强度的动态回放诊断；完整 14 s，结束时四个 unit 均挂接
MUJOCO_GL=egl python scripts/validate_toddlerbot_climb.py --device cuda:0

# CPU 自动测试：导入来源、模型、独立力/矩平衡、速度包络、预压与失效状态机
python -m pytest -q tests/microspine/test_toddlerbot_climb.py

# 如需开始训练，60° task 的命令如下；本次未启动训练
python scripts/train.py Mjlab-Climb-Wall-ToddlerBot-SingleUnit-A60 \
  --gpu-ids '[0]' --env.scene.num-envs 512 --agent.run-name single_unit_a60
```

本次新增独立 task 和资产；既有 PHYBot/C2 task 的物理参数、预压和经验表未调整。
