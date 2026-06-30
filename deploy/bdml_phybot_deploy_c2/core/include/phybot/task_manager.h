#pragma once
#include <array>
#include <cstddef>
#include <memory>
#include "phybot/task_id.h"
#include "phybot/task.h"
#include "phybot/data_package.h"

namespace phybot {
class TaskManager {
 public:
  void add(std::unique_ptr<Task> t) { tasks_[(std::size_t)t->id()] = std::move(t); }

  void start(TaskId id, DataPackage& io) {
    current_ = tasks_[(std::size_t)id].get();
    io.current_task = id;
    current_->enter(io);
  }

  TaskId current() const { return current_ ? current_->id() : TaskId::ZERO; }

  void tick(DataPackage& io) {
    TaskId next = current_->checkChange(io);            // ① 自决(返回自身=不变)
    if (next == current_->id()) next = io.requested_task;  // ② 否则外部请求
    if (next != current_->id() && tasks_[(std::size_t)next]) {
      if (current_->canExit(io)) {                      // ③ 互锁
        current_->exit(io);
        current_ = tasks_[(std::size_t)next].get();
        current_->enter(io);
      }
    }
    io.current_task = current_->id();
    current_->run(io);                                  // ④ 跑当前
  }

 private:
  std::array<std::unique_ptr<Task>, (std::size_t)TaskId::COUNT> tasks_{};
  Task* current_ = nullptr;
};
}  // namespace phybot
