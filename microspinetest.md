# Microspine Contact Model Tuning — Architecture Brief

## Goal

在这个 MuJoCo humanoid robot 仿真 repo 中，实现 microspine 材料的接触特性。Microspine 是一种在粗糙岩石表面通过机械互锁产生各向异性附着力的特殊材料。MuJoCo 原生接触模型无法直接表达这种行为，需要探索和迭代出合适的实现方式。

## 核心物理特性

Microspine 的接触行为与普通摩擦有本质区别：
- 各向异性：切向附着力远大于法向拉脱力（比值 ≥ 3:1）
- 表面依赖：只在粗糙表面有效，光滑表面退化为普通摩擦
- 方向性加载：钩刺沿表面拖动时力逐渐增大，脱离时突然失效
- 统计聚合：阵列中约 30-60% 的 spine 同时有效啮合
- 无需预载：不需要大法向力即可建立附着

## 迭代架构：worktree 隔离并行 + Generator-Evaluator Loop

参考 Anthropic 的 harness design 文章，采用 Planner-Generator-Evaluator 三层架构。三条实现路径修改的文件不重叠（XML / Python / C），通过 git worktree 隔离的 subagent 并行探索。

### 整体流程

```
主 Agent（Planner + Orchestrator）
  │
  │  准备 eval-criteria.md + approach-contract，然后 spawn 三个 subagent
  │
  ├── subagent: microspine-approach-a (worktree) → Ralph Loop (MJCF 参数)
  ├── subagent: microspine-approach-b (worktree) → Ralph Loop (callback)
  └── subagent: microspine-approach-c (worktree) → Ralph Loop (plugin)
                                    │
                                    ▼
                     各自返回 best scores.json
                                    │
                                    ▼
              主 Agent 比较分数，merge 胜者到 main
```

### 实现方式：Custom Agent + `isolation: worktree`

在 `.claude/agents/` 下定义三个 agent，每个都带 `isolation: worktree`：

```
.claude/agents/
├── microspine-approach-a.md   # MJCF 参数调优
├── microspine-approach-b.md   # Python callback
└── microspine-approach-c.md   # C engine plugin
```

每个 agent 的 frontmatter：
```yaml
---
name: microspine-approach-a
description: "Microspine via MJCF parameter tuning"
isolation: worktree
---
```

Claude Code 会自动为每个 subagent 创建独立的 worktree 和 branch，完成后返回结果给主 agent。无需手动 `git worktree add`。

### 启动方式

在项目根目录的 Claude Code 交互会话中，一句话启动：

```
读取 .microspine-tuning/eval-criteria.md 作为共享评估标准。
同时 spawn 三个 subagent，使用 worktree 隔离：
- microspine-approach-a：用 MJCF 参数调优实现 microspine 接触
- microspine-approach-b：用 Python callback 实现
- microspine-approach-c：用 C plugin 实现
每个 subagent 在自己的 worktree 中独立迭代，最多 8 轮。
所有完成后，比较三个 scores.json，merge 最佳方案到 main。
```

### 三条实现路径

**路径 A：MJCF 参数调优** — condim摩擦锥 + 高切向摩擦 + `<pair>` 区分粗糙/光滑表面。

**路径 B：mjcb_control callback** — Python callback 每 timestep 读接触信息，按接触几何注入附着力到 `qfrc_applied`。

**路径 C：MuJoCo engine plugin** — C plugin 实现完整 microspine 力计算。

### Generator-Evaluator 分离（在每个 subagent 内部）

每个 subagent 内部交替扮演 Generator 和 Evaluator：
- Generator 阶段：修改模型，跑仿真，每轮只改 1-2 个参数
- Evaluator 阶段：独立验证结果，检查物理合理性，反作弊
- 每轮开始先写假设："上次为什么失败，这次改什么，预期什么效果"

### Shared contract：eval-criteria.md

唯一跨 subagent 共享的是评估标准，主 Agent 写好后每个 worktree 都能读到（因为 worktree 从 main 分支出来时已经包含这个文件）。


### 每个 worktree 内的状态

```
.microspine-tuning/
├── approach-contract.md        # subagent 自己写的实现计划
├── eval-criteria.md            # 评分标准（从 main 继承）
├── progress.md                 # 迭代记录
├── sim-results/
│   └── iteration-NNN/
│       ├── generator-notes.md  # "改了什么，为什么"
│       ├── scores.json         # 评分卡
│       └── *.csv               # 原始仿真数据
└── best-config/                # 当前最佳配置
```

## 现在不需要做什么

这个文档是架构设计。理解后，如果对架构有疑问或改进建议，先提出讨论。具体实施后续再做。