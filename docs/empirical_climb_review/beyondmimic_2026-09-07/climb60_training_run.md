# 4096 环境四角 60° 实际训练

本次训练已于 2026-09-07 21:17（America/Los_Angeles）启动。前面的训练前验证记录保留其当时状态；本文件记录用户随后授权的实际运行。

- Task：`Mjlab-Climb-Wall-Phybot-C2-Corners-A60`，每 pad 四角 4 unit。
- 训练：GPU 0，4096 envs，seed 42，最多 12000 iterations，每 25 iterations 保存 checkpoint。
- 奖励：已审阅的 16 项 BeyondMimic 模式，保持现有追踪尺度。
- 课程：45±5° → 55±5° → 精确 60°，原始门槛为 1500、3300。升课另受评估和实际视觉审阅控制，未通过则延长。
- 出生：保留全周期自适应 RSI。episode 计时从实际出生的 0 开始，避免随机伪造 episode 年龄绕过初始稳定及事件奖励保护。
- 检查：GPU 1；0、25、100、此后每 250 iterations，以及课程节点前 250/100/50 iterations。详细判据见 [监督说明](climb60_supervision.md)。

运行目录：

```
logs/rsl_rl/phybot_c2_climb_wall_corners60/2026-09-07_21-17-18_bm16_4096_live
```

查看训练和监控：

```bash
tmux attach -t climb60_4096
tmux attach -t climb60_monitor
```

`climb60_monitor` 包含 `supervisor` 和 `feedback` 两个窗口。监督器每 30 秒刷新状态，反馈窗口每 60 秒刷新显示；`Ctrl-b d` 可离开 tmux 而保留运行。

[W&B 训练曲线](https://wandb.ai/sxuan-alan-stanford-university/scr_phybot_mjab-climb-wall/runs/czivhqxd)。

运行目录内的重要文件：

| 文件 | 用途 |
|---|---|
| `launch.sh`、`launch.json`、`params/` | 实际训练命令、启动选择和完整环境/PPO 配置 |
| `train.log` | 训练日志 |
| `model_N.pt` | 完整原子发布的 checkpoint，包含优化器、RSI 失败统计和实际课程阶段 |
| `curriculum_control.json` | 当前阶段许可、未来门槛和延课/放行记录 |
| `curriculum_runtime.json` | 训练进程实际采用的阶段及该阶段开始时间 |
| `monitor/feedback.md` | 定期更新的中文反馈，以此文件查看最新结果 |
| `monitor/status.json`、`monitor/supervisor.log` | 机器可读状态和监督日志 |
| `exams/iter_XXXXXX/` | 对应模型各角度的 metrics、engage schedule、实际 rollout 视频与视觉审阅 |
| `source_manifest.json`、`source_snapshot.tar.gz` | 启动时的任务源代码和动作参考快照 |

model 0 是完成首轮 PPO 更新后的早期基线。它的 45° 检查中，32/32 个环境首段失败、真实新挂接为 0；实际图像和视频审阅未通过，课程保持 stage 0。后续 checkpoint 的观察持续写入各自考试目录和最新反馈。

model 25 的 canonical 检查已于 21:46（America/Los_Angeles）完成，各角度均运行 32 env × 30 s。只用每个环境的首次连续 episode 判定，结果如下：

| 角度 | 平均首段时长 | 首段失败 | 首段真实新挂接 |
|---|---:|---:|---:|
| 40° | 0.530 s | 32/32 | 0 |
| 45° | 0.478 s | 32/32 | 0 |
| 50° | 0.446 s | 32/32 | 0 |

45° 图像与真实视频帧显示左脚出生挂点比 model 0 多保持几步，随后仍失附、下滑和重置；没有连续爬升。训练窗口 0–5 与 20–25 的平均 reward 从 −12.311 提高到 −12.181，但平均 episode 没有变长，动作变化惩罚减少解释了部分提升。这一节点不满足升课条件。下一次计划检查 model 100。

监督程序自动评估数值、保存图像视频、延长课程并更新反馈文件。课程放行还需要对应图像和视频被实际审阅并留下记录；缺少审阅时维持当前阶段。
