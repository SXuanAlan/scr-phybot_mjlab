# 四角 60° 训练监督

`scripts/supervise_climb60.py` 监视已在 tmux 中运行的
`Mjlab-Climb-Wall-Phybot-C2-Corners-A60`，训练使用 GPU 0、4096 个环境，
考试子进程单独设置 `CUDA_VISIBLE_DEVICES=1`（内部设备名 `cuda:0`）。
监督程序只更改当前 run 的课程许可和未来迭代门槛，不改奖励、硬件、物理容量，也不启动或重启训练。

当前 run：

```
logs/rsl_rl/phybot_c2_climb_wall_corners60/2026-09-07_21-17-18_bm16_4096_live
```

启动命令（项目根目录，独立 tmux session）：

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 /home/spider/miniconda3/envs/env_mjlab15/bin/python \
  scripts/supervise_climb60.py \
  logs/rsl_rl/phybot_c2_climb_wall_corners60/2026-09-07_21-17-18_bm16_4096_live --gpu 1
```

加 `--once` 只刷新一次并最多评估一个待检 checkpoint，可用于验证和重读人工审阅结果。
默认每 30 s 轮询；评估子进程运行时也每 30 s 刷新训练 metrics 与课程许可。
文件锁防止同一 run 重复启动监督器。
工具或 checkpoint 错误默认隔至少 10 分钟重试一次，最多尝试 2 次，不会每个轮询都重跑坏考试；
失败及重试不会放开课程。`--once` 的考试工具失败返回非零状态码。

## 检查频率和产物

- model 0：45° canonical，32 env × 30 s，首次管线检查，不作为升课依据。
- model 25、100，此后每 250 iter：当前课程 band 的低、中、高角度，各 32 env × 30 s。
- 下个课程门槛前 250、100、50 iter：优先检查；这些节点以及 canonical 已数值达标的 checkpoint，补做全部角度的 training-rsi。
- 最终 60° 阶段只有一个精确角度，避免重复考试。
- 如果监督停机导致积压，保留最初检查，随后合并过期的周期考试到最新 checkpoint，并将被合并迭代写入状态。不会据过期考试自动认定掌握。

`run_dir/monitor/status.json` 保存训练标量、考试结果、当前课程、错误和正在运行的评估；
`monitor/feedback.md` 是定期更新的中文报告，`monitor/supervisor.log` 保存操作日志。
完整结果在 `exams/iter_XXXXXX/exam.json`，其下每个 profile/angle 有评估日志、
`summary.json`，及评估器产生的数值、轨迹、engage schedule PNG、rollout MP4 和视频抽帧图。
训练指标来自只读的 `monitor_climb60_metrics.py`；训练平均 reward 或末帧 match 不直接触发升课。

## 升课与延长

课程初始许可 `max_stage=0`：45±5°。后续保持原始阶梯，iter 1500 → 55±5°，
iter 3300 → 精确 60°，只有得到许可才进入。实际 stage 读取训练进程的
`curriculum_runtime.json`，每个 checkpoint 的训练 stage 从
`infos.climb_monitor.active_stage` 读取，避免后来延课导致旧 checkpoint 被误标。

放行必须满足全部条件：

1. 同一当前课程的连续两次完整考试，涵盖全部角度及 canonical/training-rsi。
2. 每套至少 32 env、30 s；只统计每个环境首次连续 episode，不把 reset 后的支撑当新挂接。
3. 所有环境最差 engage match ≥ 0.95；首次 episode 零失败且全部存活完整 30 s。
4. 每个 pad 在最差环境的 `actual occupancy − reference occupancy` < 0.04。
5. 最小实际/参考爬升进度比 ≥ 0.8；至少完成 2 个参考循环，且每个环境、每个 pad 至少 2 次真实重挂。
6. 两次考试对应的 engage raster 和 rollout 视频都实际审阅通过；证据绑定 checkpoint SHA-256 与 stage。
7. 放行仅发生在下一门槛前 250 iter 内，两份考试也必须来自这个窗口，避免很早通过后退化仍自动升课。

NaN、缺失指标、进度比不能定义、未完整加载 actor、考试 profile/角度不一致、缺图或视频，均不能放行。
未获得完整证据且距下个节点不足 50 iter 时，将该节点及后续节点同步延后 250 iter。
若监督恢复时已落后多个节点，则以 250 的整数倍补足提前量。`max_stage` 持续锁住未批准阶段。
所有决定追加到 `curriculum_control.json` 的 `supervisor_audit`。
训练侧另有每个 stage 的最小原始训练时长保护。若实际入课晚于表中时间，监督先按
`stage_started_iter + 1500/1800` 补足未来门槛、同步平移后续阶段，再检查升课证据窗口；
已有阶段保持原样。未来门槛移动时，旧日期的预先许可也撤回，必须重新取得新窗口中的证据。

## 实际视觉审阅

每个考试根目录有 `visual_review.template.json`，默认 `approved=false`。
打开本套各个 profile/angle 的 engage raster、rollout 视频或实际视频帧后，
将观察另存为同目录 `visual_review.json`：

```json
{
  "schema_version": 1,
  "checkpoint_iteration": 1400,
  "checkpoint_sha256": "从同目录 exam.json 原样复制",
  "stage_index": 0,
  "approved": true,
  "raster_reviewed": true,
  "video_reviewed": true,
  "reviewer": "实际审阅者",
  "reviewed_at": "ISO 8601 时间",
  "notes": "具体观察：release/relatch 时序、连续爬升、是否静态焊住、是否靠 reset 等"
}
```

数值或视觉未通过时保留 `approved=false` 并记录原因。监督程序不自动生成通过审阅。

## 验证记录

- `tests/microspine/test_climb60_supervisor.py`：21 项 CPU 测试通过，覆盖身份/指标/课程保护、晚入课时的未来门槛和过期许可、错误重试上限，以及真实子进程 fixture 的完整 6-case 持久化。
- 联合评估器的 9 项测试：合计 30 项通过。
- 当前 run 的真实 `--once` 于 2026-09-08 04:26 UTC 完成：model 0，canonical 45°、32 env、30 s，产物完整。
- 该初始模型的首个 episode 32/32 失败，最差 match 0.08696，真实重挂为 0；进度比无有效值。保留 stage 0。

本记录验证了监督和考试流程，不代表初始策略已经具备攀爬能力。
