# 四角 4-unit、60° climbing：修正预载上限误用后的复评

**10 N/unit 是 engage 的最低预载，超过 10 N 不应据此判为失效。此前模型错误地增加了
`Fn >= -10 N` 的压缩限制，现已修正。原参考建压 12/12 帧通过，承载找到可行解的帧数为
463/490；双脚尖支撑的 21 个代表姿态全部通过，包括此前失败的两次抬手。**

沿用四角 `(0,0)、(0,3)、(2,0)、(2,3)`，每个 active unit 的 engage 最低预压仍为 10 N，
连续 3 个策略步；普通 pad 和脚尖摩擦系数均为 0.2。挂附承载按实测 unit 分布的 0.5 分位
特征工况检查。墙角相对水平面为 60°，真实重力始终向下。

## 原参考的完整复评

保留与 80° 检查完全相同的标准墙坐标系关节轨迹、接触时序和脚尖支撑时序。
只更改墙角元数据；审计和渲染将墙及参考一起旋转。以下两列均已移除错误压缩上限并重新计算。

| 检查 | 80° | 60° |
| --- | ---: | ---: |
| 全 490 帧 CAD 几何 | 通过 | 通过 |
| 挂附承载检查通过帧数 | 180 / 490 | 463 / 490 |
| 换点前建压通过帧数 | 0 / 12 | 12 / 12 |

60° 的 490 个承载帧全部通过线性必要条件。余下 27 帧属于脚部换点阶段，当前非线性
求解未找到满足全部约束的分配；这不证明这些姿态不存在可行解。完整原参考暂未全部通过。
这些数字是**当前模型、当前参考与支撑时序**的计算结果，不是硬件成功率。

## 预载语义修正

机器人质量为 36.261 kg。60° 时沿墙重力约 308.06 N，压向墙面的重力约 177.86 N。
初始姿态的 16 个 active unit 平均需要约 11.12 N 压缩，这满足最低预载要求，
不能用此前虚构的“总压缩上限 160 N”拒绝它。修正后，该姿态通过承载检查。

运行时、静力审计及离线 pad 容量计算均已移除这个错误上限。engage 仍要求每个 active
unit 连续 3 步满足预压 ≥10 N；挂附后检查法向拉脱上界、剪切/侧向支持范围、力矩分配等。
拉脱分布、随机分位、剪切/侧向边界、摩擦和电机限矩均保持原值。更大的预载不会提高
拟合出的拉脱容量。结构承压极限未由这些拉脱试验辨识，不能从预载数值推导。

## 双脚尖支撑对照

独立调整踝关节，使两个脚尖实际到达墙面，保留其余 pad 位姿，以普通单边摩擦接触承担载荷。
没有给未 engage 的 pad 计入黏附力，建压帧的接近 pad 只允许单边压缩与摩擦。

共检查 21 个代表姿态，其中包括全部 12 个建压窗口帧：

- 承载检查：21 / 21 个代表姿态通过，包括初始姿态、脚部换点和手部摆动姿态。
- 建压检查：12 / 12 帧通过，即四次换点前各 3 帧均可满足每 unit ≥10 N。
- 手部离墙的第 300、400 帧均通过完整静力约束检查；计算出的单 unit 最大压缩分别约
  13.85 N、14.00 N，未因超过 10 N 被拒绝。

这是独立姿态的静力对照，尚未生成、审核或执行新的连续脚尖支撑动作。
代表姿态通过尚不能代替完整连续动作与动力学验证。

此前以 13 N 超出 10 N 为由判定手部摆动不通过的解释撤回。
旧审计及诊断归档于 [修正前记录](archive/preload_ceiling_bug/INVALIDATED.md)，不再作为不可行证据。
当前结果见 [手部摆动复评](hand_swing_constraint_diagnosis.json)。

## 文件与复现

- [原参考的全帧审计](reference_audit.json)、[结果与文件哈希](verification.json)
- [双脚尖支撑对照](toe_support_probe.json)、[对照脚本](toe_support_probe.py)
- [60° 原参考运动学视频](../../../src/scr_phybot_mjab/assets/motions/climb_cad_empirical_corners_a60.mp4)
  （不是双脚尖支撑的新动作，也不是物理爬升成功录像）
- [60° 动作数据](../../../src/scr_phybot_mjab/assets/motions/climb_cad_empirical_corners_a60.npy)

在仓库根目录执行：

```sh
CLIMB_PYTHON=/home/spider/miniconda3/envs/env_mjlab15/bin/python
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 "$CLIMB_PYTHON" scripts/gen_cad_climb_reference.py \
  --source-motion src/scr_phybot_mjab/assets/motions/climb_cad_empirical.npy \
  --output src/scr_phybot_mjab/assets/motions/climb_cad_empirical_corners_a60.npy \
  --wall-angle 60 --unit-layout corners
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 "$CLIMB_PYTHON" scripts/audit_cad_climb_reference.py \
  --motion src/scr_phybot_mjab/assets/motions/climb_cad_empirical_corners_a60.npy \
  --stride 1 --output-dir docs/empirical_climb_review/corners60
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 "$CLIMB_PYTHON" docs/empirical_climb_review/corners60/toe_support_probe.py
```

当前全帧审计预期退出 1，表示该参考未通过。
本次没有执行动力学 rollout、控制器训练或硬件动作；没有用模型外推把未知能力变成通过结果。
