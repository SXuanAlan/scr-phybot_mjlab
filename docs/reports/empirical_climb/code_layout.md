# 实测 microspine 与 reference 的代码布局

本次整理保持既有的 assets / microspine / tasks / tools / scripts 分工。
模型假设、容量分布、参考运动和第 5 步的等待状态不因目录整理而改变。

## 数据与资源

```text
data/microspine/raw/20260905-110314/       原始 CSV，保持原始字节

src/scr_phybot_mjab/assets/
  microspine/
    __init__.py                         运行资源路径
    empirical_unit_v1.npz                训练实际加载的单 unit 条件分布
  motions/
    __init__.py                         task 和工具共用的 motion 路径
    climb_cad_empirical.npy              活动 reference
    climb_cad_empirical_initial.json     活动初值

docs/empirical_climb_review/limit_surface/
  empirical_unit_v1.json                 数据来源、协议、拟合和验证报告
  empirical_unit_v1_calibration.npz      calibration-only 表
  empirical_unit_v1_raw_cells.npz        稀疏格点对照
```

图继续放在 `docs/figures/`，静力 CSV、检查结果和视频继续放在验收目录。
拟合报告与对照表不会随训练资源一起加载。打包配置显式包含运行 NPZ 和 motion
资源，不包含原始试验目录、calibration 表或 raw-cell 对照表。

## 共享计算

```text
src/scr_phybot_mjab/microspine/
  pad_geometry.py          读取 asset 的 C 和 unit sites，校验阵列与坐标
  wrench.py                C 点 wrench → unit 载荷；NumPy/Torch 共用公式
  empirical_limit_surface.py  分布表查询、支持域和 unit 失效判定
  affordance.py            挂附随机状态、pad 判定、释放状态机
  constraint.py            weld 载荷读取与 C 点力矩
  contact.py               四角碰撞代理和接触配置

src/scr_phybot_mjab/tools/
  paths.py                 仅供离线使用的原始数据/报告/图片默认路径
  plotting.py              共用配色、绘图样式和导出函数
  microspine/
    cad.py                 原始 CAD 实体、装配位姿、STEP 安装基准读取
    cad_audit.py            编译后 asset 与原始 CAD 的网格比对和渲染
    geometry.py            从 PHYBot asset 取得共享 pad 几何
    empirical_fitting.py   原始试验 → 条件分布与验证报告
    stochastic_pad.py      用共享几何/分配计算 pad 概率边界
    wang_comparison.py     旧 Wang 模型的 unit 边界对照
  climbing_reference/
    kinematics.py          ClimbIK、IKTask、任务构造与插值工具
    cycle.py               共用 climbing 循环生成
    model.py               静力检查的模型构建与坐标工具
```

`pad_geometry_from_sites()` 从实际 sites 取得 unit 数量、位置和顺序。
默认 climbing 的行列数及跨度也由它生成；审计和绘图不再手写 31.5/28 mm。
所有共享一个分配器的 pad 必须有一致的 C 点相对阵列；发现缺失 unit、
C 不在阵列中心、非共面或 site 挂在其他 body 下时会报错。

`wrench.py` 内只有一份等刚度分配算式。Torch 运行时直接使用它，
NumPy 离线计算使用同一算式；静力优化需要的矩阵通过对六个基向量应用
这个分配器得到。改变阵列尺寸后，各处会一起获得新的力臂。
当前平面、等刚度、首 unit 失效即释放的物理假设仍保留。
旧 Wang/spring 模式的历史 unit 索引顺序保留，避免改变无关实验的随机状态对应关系。

## 脚本入口

| 脚本 | 复用的公共模块 | 输出 |
| --- | --- | --- |
| `build_phybot_microspine_assets.py` | `tools.microspine.cad` | 刚性 CAD 导入、碰撞包络、PLA 惯性、MJCF/URDF |
| `audit_phybot_cad_asset.py` | `tools.microspine.cad_audit` | 逐项网格/安装检查与正反面对照图 |
| `build_empirical_limit_surface.py` | `tools.microspine.empirical_fitting` | 运行表、独立目录下的分析结果 |
| `gen_climb_clip.py` | `tools.climbing_reference.cycle` | 原有参考生成入口 |
| `gen_cad_climb_reference.py` | `kinematics`、`cycle`、共享 motion 路径 | CAD 初值与新参考 |
| `gen_mount_clip.py` | `tools.climbing_reference.kinematics` | 原有 mount 生成入口；本次未重生成旧 mount 文件 |
| `audit_cad_climb_reference.py` | `model`、`pad_geometry`、`wrench` | 几何、静力和载荷 CSV |
| `plot_stochastic_pad_limit_surface.py` | `stochastic_pad`、`wang_comparison`、`plotting` | 当前概率边界和对照图 |

上述生成、拟合和审计链路不再导入其他 CLI 脚本的内部函数。
训练依旧通过 `tasks/climb_wall/config/phybot_c2/env_cfgs.py` 选择资源，
`tasks/climb_wall/mdp/motion.py` 读取 reference 并执行 RSI；运行时不导入离线 tools。

## 复现与检查

从仓库根目录使用已有环境，`PYTHONPATH` 只需 `src`：

```sh
export PYTHONPATH=src
export MUJOCO_GL=egl
export OPENBLAS_NUM_THREADS=1
/home/spider/miniconda3/envs/env_mjlab15/bin/python scripts/microspine/build_empirical_limit_surface.py
/home/spider/miniconda3/envs/env_mjlab15/bin/python scripts/reference/gen_cad_climb_reference.py
/home/spider/miniconda3/envs/env_mjlab15/bin/python scripts/reference/audit_cad_climb_reference.py
/home/spider/miniconda3/envs/env_mjlab15/bin/python scripts/microspine/plot_stochastic_pad_limit_surface.py
```

分开指定生成目标的示例：

```sh
/home/spider/miniconda3/envs/env_mjlab15/bin/python scripts/microspine/build_empirical_limit_surface.py \
  --data data/microspine/raw/20260905-110314/results.csv \
  --output /tmp/empirical_candidate.npz \
  --analysis-dir /tmp/empirical_candidate_analysis
```

重构核对包含原始文件哈希、分布表逐数组比较、参考重新生成、图中数值比较、
静力检查及非原始尺寸阵列的力/力矩守恒测试。具体数值见
[architecture_refactor.json](architecture_refactor.json)，活动文件哈希见
[artifact_manifest.json](artifact_manifest.json)。没有开始第 5 步训练。

随后按用户要求完成了刚性 CAD 修订，当前 reference 已重新生成；上述
`architecture_refactor.json` 保留的是目录重构当时的数值一致性记录。
当前 asset 与 CAD 的对照见 `rigid_cad_asset_audit.json`，当前运行检查见
`verification.json`，修订前的数值审计保存在 `archive/pre_rigid_cad/`。
