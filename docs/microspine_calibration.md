# Microspine Limit-Surface 标定协议

仿真中的 microspine 用 connect 约束"虚拟锚点" + limit surface 释放规则建模
（`microspine/constraint.py`、`microspine/affordance.py`）。limit surface 的
参数与台架实测一一对应——本文档定义需要测什么、数据格式，以及如何用
`scripts/fit_limit_surface.py` 把实测数据直接拟合成
`MicrospineLimitSurfaceCfg` 字段。

## 模型回顾

锚点载荷分解到啮合坐标系（t̂ = 耙合/啮合方向（顺墙向下）、l̂ = 横向、
n̂ = 墙外法向，拉脱为正）。垫子保持咬合当且仅当：

1. `f_t ≥ -F_rev_max` —— 反向剪切（向上推）容量
2. `f_n ≤ μ_adh · max(f_t, 0) + F_n0` —— 拉脱容量随啮合剪切增长（spine 锥）
3. `|f_l| ≤ μ_lat · max(f_t, 0) + F_l0` —— 横向容量
4. `f_t ≤ F_t_max` —— 极限剪切（asperity/刺强度，逐次啮合有散布）

## 台架测试协议

设备：墙面样件（与真实攀爬面同材质/粗糙度）、六维力传感器（或拉力计 +
角度夹具）、直线拖动机构。

每次试验按真实啮合流程：

1. **压紧**：以法向预载 `preload_N`（建议 5-15 N，记录实际值）把垫子压上样件。
2. **拖拽行程**：沿 +t̂ 方向拖 `stroke_mm`（建议 5-10 mm，记录实际值）让刺挂住。
3. **加载至脱附**：沿测试平面内固定角度 `angle_deg` 的方向缓慢加载
   （建议 ≤ 20 N/s，准静态），记录脱附瞬间的力 `release_force_N`。

角度约定（与仿真一致）：

- `plane = tn`：加载方向 = `cos(angle)·t̂ + sin(angle)·n̂`。
  `angle = 0°` 纯啮合剪切（顺墙向下拉），`90°` 纯法向拉脱，
  `180°/-180°` 纯反向剪切（向上推）。
- `plane = tl`：加载方向 = `cos(angle)·t̂ + sin(angle)·l̂`。

采样建议：

- `tn` 平面：`angle ∈ {-180, -150, ..., 0, 15, 30, ..., 90}`，
  重点加密 `0°~90°`（拉脱锥）与 `±180°` 附近（反向脱附）。
- `tl` 平面：`angle ∈ {0, 15, ..., 90}`（对称性可只测一侧，若怀疑不对称则两侧都测）。
- **锥面斜率 μ 的可辨识性**：沿固定角度加载时，锥面高 f_t 段只在
  `atan(μ)` 附近的窄角度带内可达（更小的角度会先打到极限剪切面）。
  粗角度网格会让所有锥面释放点都挤在低 f_t 端，μ 严重欠定。
  **务必在预期 `atan(μ)` 附近（拉脱锥 ~15-30°、横向锥 ~25-40°）以
  2-3° 步距加密采样**；拟合脚本会在 f_t 覆盖不足时给出警告。
- **每个角度 ≥ 10 次重复**：重复散布直接决定 `f_t_max_range` 等区间
  （拟合脚本取分位数）。
- 纯压方向（`tn` 的 `-90°` 附近）不会脱附，不必测。

## CSV 数据格式

一行一次试验：

```csv
plane,angle_deg,preload_N,stroke_mm,release_force_N
tn,0,10.0,8.0,412.5
tn,0,10.0,8.0,377.0
tn,30,10.0,8.0,46.1
tn,90,10.0,8.0,16.8
tn,180,10.0,8.0,21.3
tl,45,10.0,8.0,33.0
...
```

`release_force_N` 是脱附瞬间合力大小（沿加载方向）。若有六维力数据，
用脱附前最后一帧的 (f_t, f_n, f_l) 投影更准——格式可附加可选列
`f_t_N,f_n_N,f_l_N`，脚本优先使用。

## 拟合

```bash
python scripts/fit_limit_surface.py data/pull_tests.csv
```

输出可直接粘贴的 `MicrospineLimitSurfaceCfg` 字段（区间 = 实测分位数，
默认 [p10, p90]，可用 `--quantiles 0.05 0.95` 调整）。

拿到标定值后，把 `config.py` 中 `MicrospineLimitSurfaceCfg` 的默认区间
替换为输出值；DR 区间宽度建议在实测散布外再放宽 10-20% 覆盖材质差异。

## 后续（M3 接口预留）

- 啮合拖拽行程门控 `engage_stroke_min`：取实测中"挂住所需最小行程"的分布。
- 啮合成功率 `engage_success_prob`：固定 preload/stroke 下重复啮合的成功比例。
- `F_t_max` 逐次啮合重采样：直接用 `tn@0°` 的释放力经验分布。
