#pragma once
#include <string>
#include <utility>
#include "phybot/task_id.h"

namespace phybot {
struct DataPackage;  // 前置声明: Task 接口只按引用使用黑板

class Task {
 public:
  Task(TaskId id, std::string name) : id_(id), name_(std::move(name)) {}
  virtual ~Task() = default;

  TaskId id() const { return id_; }
  const std::string& name() const { return name_; }

  virtual void enter(DataPackage&) {}                 // 进入(复位/加载)
  virtual void run(DataPackage&) = 0;                  // 每拍执行
  virtual void exit(DataPackage&) {}                   // 退出(清零)
  virtual TaskId checkChange(DataPackage&) { return id_; }  // 自跳转, 返回自身=不变
  virtual bool canExit(DataPackage&) { return true; }       // 互锁

 protected:
  TaskId id_;
  std::string name_;
};
}  // namespace phybot
