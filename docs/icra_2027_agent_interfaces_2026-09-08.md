**Agentic robotics 的模型入口与执行接口核查**

2026-09-08，洛杉矶时间。对应“演示和论文究竟使用 API 还是 Coding Plan”的问题。本轮读取了论文相关章节、作者公开代码和 README，并从四段原视频中抽取关键帧检查；没有运行这些系统，也没有获得作者的账户账单或完整模型调用日志。

**结论：已核查的五个开源论文项目提供模型 API / 云端服务调用实现。四个社交演示中，MuJoCo 帖文明确称使用 ChatGPT 网页入口，另外三个的实际计费入口仍未确认。** 自定义机器人界面、Skills 按钮、聊天输入框或模型名称，都不能单独证明使用订阅或直接 API。

| 演示 | 核查到的使用方式 | 未确认部分 |
| --- | --- | --- |
| [dhvanil 刷尖校准](https://x.com/dhvanil/status/2097350152713363812) | 帖文描述通过试探运动估计相机运动关系与刷尖偏移；关键帧显示 PB27 画面、多相机采样和校准结果，右栏注明内容来自保存的运行记录 | 未见模型调用代码、认证入口或套餐。不能把整理后的 agent feed 画面认作原始实时客户端日志 |
| [ARX 洗衣机旋钮](https://x.com/ARXrobotics/status/2096328304794210604) | 帖文称将 Astra 接入机器人；已检查画面明确标注 16x speed | 未见模型接入代码与计费方式。加速视频不提供模型调用延迟或底层控制频率 |
| [Axel 柜门操作](https://x.com/ax_pey/status/2097037847396295037) | 约 1 秒关键帧显示 Innate 自定义界面运行 `open_cabinet_with_gpt`，并显示 `max_steps` 参数值 60；同时存在其他技能入口 | 此处可确认技能封装，不能确认该技能内部直接调用模型 API、调用编码客户端，或采用其他服务。参数 60 也不能直接解释为 60 次实际模型调用 |
| [Shengjie 转发的 MuJoCo 重建](https://x.com/ShengjieWa34067/status/2097515070519734662) | 帖文明确写 ChatGPT web UI，称系统准备 VM、MuJoCo 和 DROID/Franka 资产；关键帧是 real/sim 多相机对照 | 未公开账户套餐、是否测试额度、后台工具调用费用或完整执行日志；不能据此认定普通 Plus/Pro 的具体额度足够复现 |

本轮视频取证文件保存在临时目录 `/tmp/robotics_usage_audit_20260908/`。检查了每段视频均匀抽取的 4 帧，并补查刷尖视频 3 秒、65 秒以及柜门视频 1 秒、8 秒的 1080p 图像。这属于关键帧检查，不等于审计整段交互或验证所有作者声明。

**论文实现逐项核查如下。** 模型名称区分论文实验与当前开源默认配置；公开代码可能在论文发表后更新。

| 项目 | 模型入口的直接证据 | 模型与数值程序的分工 |
| --- | --- | --- |
| [Agentic Real2Sim](https://github.com/agentic-real2sim/agentic_real2sim) | README 的后端表要求 API credentials；`ar2s/agent_configs/models.py` 封装 OpenAI、Anthropic、OpenRouter 等后端。Gemma/Qwen 可通过 OpenRouter 访问。当前默认配置使用 Claude Opus 4.7，不能把它当作论文所有实验的统一模型 | 论文中的模型作对象发现、关键帧/掩码选择与回放修正等受约束决策；分割、深度、位姿追踪及 grasp sweep 由专门组件计算 |
| [SimFoundry](https://github.com/NVlabs/SimFoundry) | README 说明使用 Google Cloud Vertex AI 或 Gemini API key；`simfoundry/models/vlm.py` 明确区分两种认证路由 | [论文附录 E.2](https://arxiv.org/html/2606.28276#A5.SS2) 列出 Gemini-Pro-3 与图像模型等；模型参与场景理解、对象图像处理和关节生成，重建、仿真与策略学习由相应模块完成 |
| [EToT](https://github.com/HEA1OR/Embodied-Tree-of-Thoughts/blob/main/vlm/README_VLM.md) | README 明确使用 OpenAI API 调用 GPT-4o，notebook 配置 `api_key` 和 `base_url` | 模型提出/评价计划分支，仿真检验候选；[论文](https://arxiv.org/html/2512.08188) 说明执行原语由系统提供，OPEN/CLOSE 使用人工编写的控制原语 |
| [ReKep](https://github.com/huangwl18/ReKep/blob/main/constraint_generation.py) | `constraint_generation.py` 使用 OpenAI SDK，并调用 `client.chat.completions.create`；配置文件使用 `chatgpt-4o-latest` | 模型把图像关键点与任务转为 Python 约束，优化器求解末端动作；这一历史模型配置不代表当前仍可直接使用相同 API alias |
| [RoboGen](https://github.com/Genesis-Embodied-AI/RoboGen) | README 要求 OpenAI API key；`gpt_4/query.py` 使用 `openai.ChatCompletion.create`，默认模型 `gpt-4` | 模型生成任务、场景与学习监督；公开 PyBullet 版本通过规划/学习程序获得技能，不是每个仿真步调用 GPT |
| [VLM 行为树主动 Real2Sim](https://arxiv.org/html/2601.08454v2) | 论文明确使用 GPT-4o 或 Gemini 1.5 Pro；本轮未找到可验证认证与计费入口的作者公开实现 | VLM 生成行为树，由 `py_trees` 与 ROS Noetic 执行；Panda 使用笛卡尔阻抗控制器进行探测。不能只凭模型名称断定其账户或服务商 |

Agentic Real2Sim 的相关代码可直接检查：[后端说明](https://github.com/agentic-real2sim/agentic_real2sim/blob/main/ar2s/agent_configs/README.md)、[后端封装](https://github.com/agentic-real2sim/agentic_real2sim/blob/main/ar2s/agent_configs/models.py)、[Gemma/OpenRouter 配置](https://github.com/agentic-real2sim/agentic_real2sim/blob/main/ar2s/agent_configs/config_orouter_gemma4.yaml)。API 协议不等于一定向 OpenAI 付费；可以是其他提供商或机构的兼容服务。这里仅记录开源实现支持的入口，不推断作者个人付费方式。

**可直接引用的费用数据。** [Agentic Real2Sim v3，4.2–4.3](https://arxiv.org/html/2607.19190v3#S4.SS3) 报告在 DROID-100 的 100 个输入 episode 上，Gemma 4 31B 的模型总费用为 $2.62，GPT-5.4 为 $82.30。折算每个尝试输入约 $0.0262 和 $0.823；这是计算值，不是单次调用价格。

这些数值不是每个成功转换的费用，也不是全系统 GPU、仿真、机器人和训练总成本。文中的 replay success 由 VLM judges 对候选回放评分定义；Gemma 和 GPT-5.4 分别为 48/100 与 43/100，不能改称真机任务成功率。便宜后端的结果说明该受约束流水线可以使用多种模型，不证明通用机器人自主工程任务都具有相同成本或成功率。

**对我们项目的实际影响。** 直接复现这些 API 项目通常需要配置其文档指定的模型服务凭据；购买 Coding Plan 不会自动改变源码里的 API 请求。若选择订阅版 Codex，需把外层决策接到官方客户端、`codex exec` 或 app-server，复用本地的测量、拟合、仿真与训练工具。原始 VLM API、订阅客户端和完整编码 agent 的上下文、工具权限、输出格式与开销可能不同，需要重新验证。

目前继续建议：用订阅版 Codex 构建和调试系统；让实验工具与模型入口分离，为论文重复实验保留可替换的模型后端。选用订阅并不削弱 agentic claim，但只让 agent 写一次代码也不能证明它参与了后续实验决策。需记录每次观察、模型输出、工具调用、人工介入和实际消耗。
