#pragma once
namespace phybot {
// 唯一中心任务清单。新增 task 在 COUNT 之前加一行;COUNT 必须始终最后。
enum class TaskId { ZERO, WALK_AMP, MIMIC, WALK, COUNT };
}  // namespace phybot
