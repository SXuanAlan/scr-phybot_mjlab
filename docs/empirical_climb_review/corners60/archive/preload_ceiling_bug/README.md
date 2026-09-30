# 四角 4-unit、60° climbing 初评

**60° 下当前完整 climbing 动作仍未通过准静态检查。双脚尖支撑后，换点建压可通过代表姿态检查，但手部离墙摆动时的支撑尚未通过。**

沿用四角 `(0,0)、(0,3)、(2,0)、(2,3)`，每个 active unit 的 engage 最低预压仍为 10 N，
连续 3 个策略步；普通 pad 和脚尖摩擦系数均为 0.2。挂附承载按实测 unit 分布的 0.5 分位
特征工况检查。墙角相对水平面为 60°，真实重力始终向下。

## 原参考直接放到 60°

保留与 80° 检查完全相同的标准墙坐标系关节轨迹、接触时序和脚尖支撑时序。
只更改墙角元数据；审计和渲染将墙及参考一起旋转。

| 检查 | 80° | 60° |
| --- | ---: | ---: |
| 全 490 帧 CAD 几何 | 通过 | 通过 |
| 挂附承载检查通过帧数 | 140 / 490 | 0 / 490 |
| 换点前建压通过帧数 | 0 / 12 | 0 / 12 |

60° 的 490 个承载帧和 12 个建压帧都未通过线性必要条件，不能归因于非线性求解器的局部初值。
这些数字是**当前模型、当前参考与支撑时序**的结果，不是硬件成功率。

## 数据范围与硬件能力必须区分

机器人质量为 36.261 kg。60° 时沿墙重力约 308.06 N，压向墙面的重力约 177.86 N。
当前模型支持 `Fn >= -10 N`（负值表示压缩）；初始四点挂附的 16 个 active unit，
在模型数据范围内合计最多承担 160 N 压缩。该帧没有脚尖支撑来分担剩余法向力，
即使均匀分配也需要平均 11.12 N/unit，已超出模型范围。

**10 N 是 engage 的最低预压，不是已测得的硬件最大承压。**
承载模型的 `-10 N` 下界来自现有试验协议和数据支持范围；超出这个范围会被模型拒绝，
不能据此认定更大的压缩会使真实 unit 失效。降低坡度并不必然改善这个受限模型下的检查结果。
本次没有放宽实测域、预压阈值、摩擦系数或电机限矩来获得通过。

## 双脚尖支撑对照

独立调整踝关节，使两个脚尖实际到达墙面，保留其余 pad 位姿，以普通单边摩擦接触承担载荷。
没有给未 engage 的 pad 计入黏附力，建压帧的接近 pad 只允许单边压缩与摩擦。

共检查 21 个代表姿态，其中包括全部 12 个建压窗口帧：

- 承载检查：11 / 21 个代表姿态通过。初始姿态和检查到的脚部换点姿态通过。
- 建压检查：12 / 12 帧通过，即四次换点前各 3 帧均可满足每 unit ≥10 N。
- 手部离墙摆动的代表姿态（例如第 300、400 帧）仍未通过承载线性必要条件。

这是独立姿态的静力对照，尚未生成、审核或执行新的连续脚尖支撑动作。
建压能通过不等于从原支点到新支点的整个摆动过程能通过。

## 手部摆动为什么未通过

对上述双脚尖支撑的第 300、400 帧分别拆分约束，结果如下：

| 诊断 | 第 300 帧：左手离墙 | 第 400 帧：右手离墙 |
| --- | ---: | ---: |
| 原线性必要条件 | 无解 | 无解 |
| 仅取消电机限矩 | 仍无解 | 仍无解 |
| 仅允许提高 unit 切向载荷上限 | 仍无解 | 仍无解 |
| 保留其余线性约束，最小化单 unit 峰值压缩 | 至少 13.015 N | 至少 13.155 N |

这定位到了当前压缩侧模型范围的限制：`Fn >= -10 N` 无法容纳这些姿态所需的受力分配。
末行保留切向 30 N 上限、电机限矩、底座与被动关节平衡、限位反力以及脚尖线性接触约束，
仅将所有挂附 unit 的压缩上限替换为待最小化的公共变量。
脚尖摩擦使用真实圆锥的外接方形，且此线性检查省略了非线性经验承载面，因此数值是
**所需峰值压缩的必要下界，不是实测载荷，也不是扩大数据范围后完整动作就能通过的证明**。

抬手时只有三个 pad 挂附；手到达下一落点并建压时，该 pad 又能提供普通压缩与摩擦接触，
所以建压窗口可行与中途悬空姿态不可行并不矛盾。10 N 的 engage 最低预压应继续与
挂附模型的压缩数据边界区分，不能将本次数据域失败直接解释为硬件承压失败。

数值和输入文件哈希见 [手部摆动约束诊断](hand_swing_constraint_diagnosis.json)。

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
