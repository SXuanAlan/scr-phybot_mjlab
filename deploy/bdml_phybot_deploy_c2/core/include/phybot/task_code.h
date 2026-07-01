#pragma once
namespace phybot {
using TaskCode = int;                 // a task's id == its controller.yaml select_code
constexpr TaskCode kNoTask = 0;       // "none / stay / use default"
constexpr int kMaxTaskCode = 64;      // capacity for code-indexed arrays
}  // namespace phybot
