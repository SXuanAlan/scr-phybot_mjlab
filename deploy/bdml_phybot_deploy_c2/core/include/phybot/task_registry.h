#pragma once
#include <array>
#include <cstddef>
#include <cstring>
#include <functional>
#include <memory>
#include <stdexcept>
#include "phybot/task_code.h"
#include "phybot/task.h"
namespace phybot {
class TaskRegistry {
 public:
  using Factory = std::function<std::unique_ptr<Task>()>;
  static TaskRegistry& instance() { static TaskRegistry r; return r; }
  void add(const char* name, Factory f) {
    for (int i = 0; i < n_; ++i) if (std::strcmp(names_[i], name) == 0)
      throw std::runtime_error("duplicate task name");
    if (n_ >= kMaxTaskCode) throw std::runtime_error("too many tasks");
    names_[n_] = name; factories_[n_] = std::move(f); ++n_;
  }
  int size() const { return n_; }
  const char* name(int i) const { return names_[i]; }
  std::unique_ptr<Task> make(int i) const { return factories_[i](); }
 private:
  TaskRegistry() = default;
  int n_{0};
  std::array<const char*, kMaxTaskCode> names_{};
  std::array<Factory, kMaxTaskCode> factories_{};
};
}  // namespace phybot
#define REGISTER_TASK(NAME, CLASS)                                            \
  namespace { struct CLASS##_Reg { CLASS##_Reg() {                            \
    ::phybot::TaskRegistry::instance().add(                                   \
        (NAME), [] { return std::unique_ptr<::phybot::Task>(new CLASS()); }); \
  } } CLASS##_reg_instance; }
