#pragma once
#include "phybot/task_id.h"
namespace phybot {
// 最小黑板。A1+ 阶段在此扩展传感器/命令/动作字段。
struct DataPackage {
  TaskId requested_task{TaskId::ZERO};  // 输入侧写: 想切到哪个 task
  TaskId current_task{TaskId::ZERO};    // TaskManager 写: 当前激活 task
};
}  // namespace phybot
