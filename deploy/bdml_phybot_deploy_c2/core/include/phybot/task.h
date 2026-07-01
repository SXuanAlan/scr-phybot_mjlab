#pragma once
#include <string>
#include <utility>
#include "phybot/task_code.h"
namespace phybot {
struct DataPackage;
class Task {
 public:
  explicit Task(std::string name) : name_(std::move(name)) {}
  virtual ~Task() = default;
  const std::string& name() const { return name_; }
  virtual void enter(DataPackage&) {}
  virtual void run(DataPackage&) = 0;
  virtual void exit(DataPackage&) {}
  virtual TaskCode checkChange(DataPackage&) { return kNoTask; }   // kNoTask = stay
  virtual bool canExit(DataPackage&) { return true; }
 protected:
  std::string name_;
};
}  // namespace phybot
