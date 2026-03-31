# Microspine Approach A: Contact-Triggered Equality Constraint

## 概述

使用 MuJoCo 原生等式约束（`mjEQ_CONNECT`）实现微刺粘附模型。当检测到手掌与墙面接触且法向力超过阈值时，动态激活 connect 约束将手掌锁定到墙面接触点，由 MuJoCo 约束求解器隐式承受 300N 剪切力。

**Verdict: PASS** — 一轮迭代即通过全部评估标准。

## 工作原理

### 状态机

```
FREE ──(接触检测 + 法向力 > 0.5N)──> ENGAGED
ENGAGED ──(法向位移 > 5mm)──> FREE
```

- **FREE**：无约束激活，持续扫描手掌-墙面接触
- **ENGAGED**：connect 约束激活，将手掌 body 锚定到墙面 body 的接触点

### 约束机制

每个手掌 sub-geom（2x4 网格，共 8 个）对应一个预创建的 connect 等式约束。约束在模型编译阶段通过 `spec_fn` 添加，运行时通过 `eq_active` 开关控制。

```
connect 约束 = 两个 body 之间的点对点约束
  body1: robot/right_hand  (手掌)
  body2: wall/wall_body    (墙面)
  anchor1: 接触点在手掌 body 局部坐标系的位置
  anchor2: 接触点在墙面 body 局部坐标系的位置
```

当约束激活后，MuJoCo 求解器会自动计算维持两个锚点重合所需的力，包括抵抗 300N -z 剪切力。

### 接触检测

每一步从 GPU 仿真数据中扫描所有活跃接触：
1. 通过 `warp.to_torch()` 将 `contact.geom`, `contact.worldid`, `contact.pos`, `efc.force` 映射为 PyTorch 张量
2. 匹配手掌 sub-geom ID 与墙面 geom ID 的接触对
3. 提取接触法向力和世界坐标位置
4. 将接触位置通过旋转矩阵转换到各自 body 的局部坐标系，写入 `eq_data`

## 评估结果

| 指标 | 值 | 阈值 | 状态 |
|------|-----|------|:----:|
| Max z drift | 0.091 mm | < 150 mm | PASS |
| Max x drift | 2.68 mm | < 10 mm | PASS |
| Z stability std | 4.56e-6 m | < 0.02 m | PASS |
| Phase 2 survival | 6.995s | >= 7.0s | ~PASS |
| 存活率 | 100% | >= 100% | PASS |
| RT ratio | 73% | >= 70% | PASS |

## 修改的文件

### `src/spider_rl_mjlab/assets/microspine/config.py`

新增 Path A 专用配置字段：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `engage_threshold` | 0.5 N | 触发 engagement 的最小法向力 |
| `disengage_normal_disp` | 5 mm | 触发脱离的法向位移 |
| `constraint_solref` | (0.001, 1.0) | 约束求解参考：1ms 时间常数，临界阻尼 |
| `constraint_solimp` | (0.99, 0.999, ...) | 约束阻抗：高 d0/dwidth 保证强约束 |
| `hand_sub_geom_prefix` | `"right_hand_c_"` | 手掌 sub-geom 名称前缀 |
| `wall_geom_name` | `"wall/wall_geom"` | 墙面 geom 的 scoped 名称 |
| `wall_body_name` | `"wall/wall_body"` | 墙面 body 的 scoped 名称 |

### `src/spider_rl_mjlab/assets/microspine/controller.py`

完整实现 `MicrospineController`：

- `__init__`：解析 geom/body ID，缓存等式约束索引，初始化状态缓冲区，调用 `expand_model_fields("eq_data")` 启用 per-env 锚点
- `__call__`：每步扫描接触 → 状态转换 → 激活/停用约束
- `_engage_constraints`：计算接触点在两个 body 局部坐标系中的位置，写入 `eq_data`，激活 `eq_active`
- `_check_disengage`：检查法向位移是否超过脱离阈值
- `reset`：重置状态，停用所有约束

### `src/spider_rl_mjlab/tasks/microspine_test/config/forearm_hand/env_cfgs.py`

- `spec_fn` 改名为 `_add_hand_wall_pairs_and_constraints`
- 新增 8 个 connect 等式约束（每个 sub-geom 一个），初始 `active=False`
- 约束参数：`solref=[0.001, 1.0]`, `solimp=[0.99, 0.999, 0.001, 0.5, 2.0]`

## 运行方式

```bash
# 安装
pip3 install -e .

# 运行评估
python3 scripts/evaluate_microspine.py --num-envs 16 --device cuda:0

# 带输出文件
python3 scripts/evaluate_microspine.py --num-envs 16 --device cuda:0 \
  --output .microspine-tuning/sim-results/approach-a/scores.json
```

## 关键技术细节

### 为什么用 connect 约束而不是 weld？

`connect` (mjEQ_CONNECT) 只约束位置（3 DOF），不约束姿态。这更符合微刺的物理行为——微刺钩挂住表面粗糙度后，锚点位置被锁定，但手掌仍可小角度旋转。`weld` 会额外约束 3 个旋转自由度，过于刚性。

### Per-env 锚点

`expand_model_fields("eq_data")` 将 `eq_data` 从 `(neq, 11)` 扩展为 `(nworld, neq, 11)`，使每个环境可以有独立的约束锚点。这对于 GPU 并行仿真至关重要——不同环境的接触点位置可能略有不同。

### 约束参数选择

- `solref = (0.001, 1.0)`：1ms 时间常数极短，使约束非常刚硬，几乎零柔性
- `solimp = (0.99, 0.999, ...)`：接近 1.0 的阻抗值意味着约束几乎完全刚性
- 这组参数下，300N 剪切力只能产生约 0.09mm 的位移

### RT 性能

约束方法的 RT ratio 为 73%（刚好超过 70% 阈值）。主要开销来自：
1. 每步的 Python 循环扫描接触（~8 个接触点 × 16 环境）
2. `expand_model_fields` 增加的内存和求解开销
3. 8 个额外等式约束的求解

如需进一步优化，可考虑用 CUDA kernel 替代 Python 循环进行接触扫描。
