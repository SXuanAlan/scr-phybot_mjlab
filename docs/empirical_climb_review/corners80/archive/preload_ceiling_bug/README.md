# 四角 4-unit、80° climbing 初评

2026-09-06。已启用四角 4 个 unit，每个 active unit 仍须达到 10 N 预压。
**当前动作参考未通过准静态可行性检查。初始四点挂附姿态通过，但完整换点循环未通过。**
这不是对所有可能的 80° 动作的否定，也不是已经完成的控制器或硬件实验。

## 本次工况

- 任务：`Mjlab-Climb-Wall-Phybot-C2-Corners-A80`。训练配置和 play 都固定 80°，无角度抖动。
- 四角 CAD unit：`(0,0)、(0,3)、(2,0)、(2,3)`，保留 63 × 84 mm 的原始力臂。
- `limit_surface.unit_layout="corners"`：只有 4 个 unit 分担预压、完整 wrench 和实测随机容量。
  其余 8 个不承担载荷，也不贡献容量。保留现有硬件质量，未假设拆除零件后的减重。
- 每个 unit ≥10 N，连续 3 个策略步后才允许 OFF→ON；无弯矩时最低总建压力为 40 N。
- 墙角是相对水平面的 80°。墙和参考一起旋转，真实重力始终为 `(0,0,-9.81)`。
- 普通 pad 接触和脚尖接触的摩擦系数均为 0.2，与实际任务一致。
  LP 使用摩擦圆锥的外包络，非线性解再检查真实圆锥。
- 挂附承载仍用实测 unit 分布的 0.5 分位特征工况；它不是“整个 pad 有 50% 成功率”。

## 结果

| 检查 | 结果 |
| --- | --- |
| 参考长度 | 490 帧 / 50 Hz；可循环部分 9 s，上移 20 mm |
| CAD 几何和循环闭合 | 通过；初始关节状态保留与 A80 上墙动作的衔接 |
| 初始四点挂附姿态 | 通过准静态特征容量检查 |
| 全帧挂附承载 | 140 / 490 帧通过，350 帧未通过 |
| 未通过帧的分类 | 276 帧线性必要条件不可行；74 帧未找到满足实测曲面的非线性分配 |
| 换点前建压 | 四次换点、共 12 帧，全部未通过线性必要条件 |
| 软件回归 | 156 项通过 |
| 环境短运行 | 8 个环境、12 个策略步，角度为 80°、每 pad 4 个 unit、数值有限；记录 32 次强制脱附事件 |

机器人质量为 36.261 kg，80° 时沿墙重力约 350 N。三点支撑只有 12 个 active unit；
即使直接相加每个 unit 的 30 N **数据剪切域上界**也只有 360 N。
这个和数不是可用 pad 容量：还要同时满足剪切方向、法向容量、力矩、被动球铰和电机限制。
减少 unit 确实降低了总建压力，也显著减少了换点时的承载余量。

8 环境短运行采用初始固定关节 PD 目标和成熟接触种子，用于检查环境接线与数值状态。
它没有跟踪优化后的接触力，也没有完成爬升，不能把数值有限解释成动作成功。
未启动训练或执行硬件动作。

## 脚尖支撑和靠墙姿态对照

另外检查了 13 个代表姿态：只调整独立踝关节，使两个脚尖真实到达墙面，
再按同一个 0.2 摩擦系数检查。仅 3 个代表姿态通过承载检查；
四个换点前代表帧中，仅左脚第 104 帧通过建压检查。
这些是独立姿态诊断，不是新的完整连续动作。

对两个手部建压姿态，又检查了向墙移动 1、2、4 cm 的候选，保留 pad 位姿并让两个脚尖接触。
六个候选仍未通过建压线性必要条件。没有把这些候选替换进正式参考。

旧审计曾将脚尖摩擦写为 0.5。本目录所有正式结果使用修正后的 0.2；
较乐观的中间结果不作为可行性证据。

## 文件与复现

- [完整审计](reference_audit.json)（`overall_pass=false`）
- [验证及文件哈希](verification.json)
- [脚尖支撑对照](toe_support_probe.json)、[身体靠墙对照](body_press_probe.json)
- [环境短运行](runtime_smoke.json)、[回归测试输出](test_results.txt)
- [80° 运动学视频](../../../src/scr_phybot_mjab/assets/motions/climb_cad_empirical_corners_a80.mp4)
  （灰色 unit 未启用；视频是逐帧渲染，不是物理爬升成功录像）
- [动作数据](../../../src/scr_phybot_mjab/assets/motions/climb_cad_empirical_corners_a80.npy)

在仓库根目录，使用 `env_mjlab15` 的 Python：

```sh
CLIMB_PYTHON=/home/spider/miniconda3/envs/env_mjlab15/bin/python
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 "$CLIMB_PYTHON" scripts/gen_cad_climb_reference.py \
  --source-motion src/scr_phybot_mjab/assets/motions/climb_cad_empirical.npy \
  --output src/scr_phybot_mjab/assets/motions/climb_cad_empirical_corners_a80.npy \
  --wall-angle 80 --unit-layout corners
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 "$CLIMB_PYTHON" scripts/audit_cad_climb_reference.py \
  --motion src/scr_phybot_mjab/assets/motions/climb_cad_empirical_corners_a80.npy \
  --stride 1 --output-dir docs/empirical_climb_review/corners80
MUJOCO_GL=egl "$CLIMB_PYTHON" scripts/render_climb_video.py \
  src/scr_phybot_mjab/assets/motions/climb_cad_empirical_corners_a80.npy --distance 2.8
```

当前审计命令应退出 **1**，表示可行性未通过；不把这个退出码改成成功。
`--source-motion` 复用已按当前 CAD 生成的标准墙坐标系轨迹，变更工况元数据，
由审计、环境和渲染各自在真实 80° 墙坐标系应用，不会重复旋转关节参考。
审计旧 12-unit 工况时显式使用 `--unit-layout all`。
