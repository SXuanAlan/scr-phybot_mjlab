#pragma once
#include <array>
#include <cstddef>
#include <memory>
#include "phybot/task_code.h"
#include "phybot/task.h"
#include "phybot/data_package.h"
namespace phybot {
class TaskManager {
 public:
  void add(std::unique_ptr<Task> t, TaskCode code) {
    if (code > 0 && code < kMaxTaskCode) tasks_[(std::size_t)code] = std::move(t);
  }
  void start(TaskCode code, DataPackage& io) {
    switchTo(code, io); io.current_task = current_;
  }
  void tick(DataPackage& io) {
    TaskCode want = io.requested_task;
    if (cur_) { TaskCode self = cur_->checkChange(io); if (self != kNoTask) want = self; }
    if (want != current_ && valid(want) && (!cur_ || cur_->canExit(io))) switchTo(want, io);
    io.current_task = current_;
    if (cur_) cur_->run(io);
  }
  TaskCode current() const { return current_; }
 private:
  bool valid(TaskCode c) const { return c > 0 && c < kMaxTaskCode && tasks_[(std::size_t)c]; }
  void switchTo(TaskCode c, DataPackage& io) {
    if (!valid(c)) return;
    if (cur_) cur_->exit(io);
    current_ = c; cur_ = tasks_[(std::size_t)c].get(); cur_->enter(io);
  }
  std::array<std::unique_ptr<Task>, kMaxTaskCode> tasks_{};
  TaskCode current_{kNoTask};
  Task* cur_{nullptr};
};
}  // namespace phybot
