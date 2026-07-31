# Microspine 旋转柔性标定（"4°" 定案，2026-07-31）

## 0. 结论速览

| 参数 | 定案值 | 依据 |
|---|---|---|
| `tilt_peel_max_deg_range` | **(3.5, 4.5)°，每次抓取抽签** | le_max 5 mm ÷ rake 跨 74.7 mm = 3.84°（边缘枢轴） |
| `anchor_torquescale` | **0.01（维持不变）** | 探针证明 ts 对可观测刚度无贡献（见 §3），其唯一职责是不制造假力矩 |
| `engaged_tilt` 税 | **>3° 起征，scale 5°，w=-1.0** | 剥离悬崖前的梯度化预警；参考只用 ≤1.1°，合规执行零税 |
| `tilt_release_cost`（reason 5） | **0.5**（tension 0.25 与 ultimate 1.0 之间） | 转掉活锚 = 可完全避免的结构性失败 |

释放判据：**engaged pad 相对 engage 捕获姿态的转角 > 抽中的 θ_max → 立即强制释放**（几何事实，无 debounce），release reason **5 = tilt-peel**，raster 图例 teal ▼。

## 1. 物理模型：真实 pad 是"针尖锁死 + 底盘悬浮"

Wang et al. 2017 Sec 2.3：每根针装在法向刚度 ~5 mN/mm、行程 **le_max = 5 mm** 的柔性悬臂上（力自由贴合）。挂墙后针尖钉死；机器人小腿/前臂连接的**底盘**浮在悬臂上：

- 底盘绕受载边排旋转，伸长侧行程耗尽于 **5 / 74.7 mm ≈ 3.84°**（C2 rake 跨度）
- 行程内回复力矩 ~0.1 N·m 量级（悬臂）；超界后受拉边排结构受载 → **拔出剥离**（~2 N·m 极限）
- 枢轴不确定性：绕中心 7.7° / 绕边 3.84° → 取保守侧 **3.5–4.5° DR 区间**（同 F0 的"每次落在不同微凸体"抽签哲学）

参考轨迹的实测用量（quat 相对 engage 帧）：手 0.07–0.13°、脚 0.5–1.1° —— 对 3.5° 下限有 **3 倍余量**，任务可解性不受影响。改前的策略（motionref17e 系）实测 capture 附近漂移 p95 12°/max 22° —— 正是本次标定关闭的 sim2real 空洞。

## 2. 为什么"刚性锁"是假物理（背景，07-29 已定案）

刚性 weld 把"底盘-墙"焊成刚体 = 发明硬件不存在的传力路径。无腕臂（4 DoF）控不了 6 维手姿态，闭链里 1–2° 的 PD 位形冲突被刚性旋转行放大成 ~6 N·m 假力矩（×§3 实测结构刚度 100–260 N·m/rad 正好对上量级），喂给诚实的力矩计费 → reference-hold 65/79 假释放。ts=0.01 让寄生力偶以力重分配收场（滞留 ≤0.07 N·m）。

## 3. 刚度探针与"ts 无关"发现

工具：`scripts/calibrate_tilt_stiffness.py`（零重力、四锚 spawn、limit surface 关、单 EE 纯力矩斜坡 0→0.3 N·m / 150 步、过原点最小二乘拟合 k = dM/dθ）。

| ts | 手(ee0) y 轴 | 手 z 轴 | 脚(ee2) y 轴 | 脚 z 轴 |
|---|---|---|---|---|
| 0.002 | 246 | 49 | 223 | 89 |
| 0.005 | 259 | 50 | 213 | 88 |
| 0.01 | 227 | 49 | 219 | 89 |
| 0.05 | — | — | 233 | 103 |

单位 N·m/rad。**三档小 ts 完全同值** → 测得的刚度来自 **PD-结构路径**（转动 pad 必须带动臂/腿关节，k=100 的 PD 经杠杆 ≈ 50–260 N·m/rad），weld 旋转行的贡献在 ts≤0.01 时不可见（0.05 时 z 轴 89→103 = weld 开始叠加）。

**推论（本次标定的关键重构）**：
1. 硬件上同样存在 PD-结构路径 —— 这部分 sim/硬件天然一致；
2. 两者唯一差异在底盘-墙连接：硬件 ~0.1 N·m 悬臂 + ~4° 剥离。0.1 N·m 对 58 N·m 力矩上限的臂是零阻力 —— **硬件的真实行为不是"抗转"，是"转过 ~4° 就掉"**；
3. 因此"把 ts 调到悬臂等效刚度（1.4 N·m/rad）"是病态目标（悬臂不是唯一路径，且永远被结构刚度淹没）——**ts 保持 0.01**，保真度由 tilt-peel 释放 + 接近税承载。

## 4. 实现清单（本仓库落点）

- `microspine/config.py`：`MicrospineLimitSurfaceCfg.tilt_peel_max_deg_range`（默认 (0,0)=关闭，legacy 不受影响）
- `microspine/affordance.py`：`limit_tilt_peel_max`（reset 采样 + engage transition 重抽签）、`tilt_from_capture` buffer（`body_link_quat_w` vs executor `anchor_quat_w` 的 quat 角）、`_limit_surface_release` 内 instant 释放 + reason 5
- `tasks/climb_wall/mdp/rewards.py`：`engaged_tilt_penalty` 改读 capture-relative（3°/5°）；`unsafe_microspine_release_penalty` 加 `tilt_release_cost=0.5`
- `config/phybot_c2/env_cfgs.py`：range (3.5,4.5)、税参数、罚款参数
- `scripts/engage_raster.py`：reason 5 图例/直方图、`engaged-pad rotation from capture` 统计行
- `scripts/calibrate_tilt_stiffness.py`：刚度探针（`build_env` 新增 `anchor_torquescale` 参数）
- 测试：tests/microspine 59 项全绿（含新断言）

## 5. 硬件 bench 对齐表（到货后替换）

| sim 参数 | 现值（推导/探针） | bench 实测项 |
|---|---|---|
| θ_max（tilt_peel range） | 3.5–4.5° | 力矩-倾角曲线的剥离角（俯仰/滚转/偏航分测） |
| 悬臂力矩尺度 | ~0.1 N·m（论文推导，sim 中无需表达） | 曲线软区高度 |
| moment_arm_fraction | 0.5（离散两排折算） | 剥离力矩 ÷ 剥离时边排拉力 |
| le_max | 5 mm（论文） | 实际悬臂行程 |
| ts | 0.01（假力矩安全区） | 无需匹配 bench；只需 weld 贡献 ≪ 结构刚度 |

### 5.1 已知近似：yaw 轴（绕墙法线）各向异性未建模（07-31，用户指出）

物理：yaw 转动不抬针尖、令针尖切向滑动——现实中应**极硬**（针尖剪切刚度 × pad
半径），失效判据是"力矩超侧滑容量"（M_n ≈ 0.57·F0 × 37 mm ≈ 8.5 N·m @F0=400），
不是行程/角度。当前 sim：weld 各轴同软，yaw 折进统一 4° 释放（机制张冠李戴、
结果近似可用）。

量化边界（m1500 实测，raster PRY/YAW 分解）：策略 yaw 用量 p95 1.5°(45°) /
2.6°(75°)——换算硬件后果 ≈ 手臂绕挠 ~2° + pad yaw 力矩 1.7–3.5 N·m，低于
8.5 N·m 侧滑容量（利用率 ≤40%，不滑）。**当前用量下为二阶效应**；raster 每卷
打印 PRY/YAW 分解监控滥用（p95 超 ~3° 应重审）。

诚实修法（推迟至 bench 后决定）：executor 加 yaw 伺服力矩（k_yaw ≥ 500
N·m/rad，clamp 于实测滑移容量）+ M_n 计费 + 滑移释放（预留 reason 6）。
bench 需测：yaw 力矩-角曲线与滑移起始力矩。

## 6. 验证链（新物理下）

- 静态审计：不受影响（力平衡与旋转柔性无关）
- reference-hold（tilt-peel 激活）：见 worklog 当日条目
- 参考重生成：`climb_cycle_moment_safe_cal4.npy`（paced 同参数）+ 五关验收
- 训练：motionref18 从零，监控 tilt_from_capture 统计与 reason-5 释放率
