#pragma once
#include <array>
#include <cstddef>
#include <functional>
#include <memory>
#include "phybot/task_id.h"
#include "phybot/task.h"
#include "phybot/task_manager.h"

namespace phybot {
class TaskRegistry {
 public:
  using Factory = std::function<std::unique_ptr<Task>()>;

  static TaskRegistry& instance() {
    static TaskRegistry r;
    return r;
  }

  void add(TaskId id, const char* name, Factory f) {
    factories_[(std::size_t)id] = std::move(f);
    names_[(std::size_t)id] = name;
  }

  void loadInto(TaskManager& m) const {
    for (std::size_t i = 0; i < (std::size_t)TaskId::COUNT; ++i)
      if (factories_[i]) m.add(factories_[i]());
  }

  bool has(TaskId id) const { return static_cast<bool>(factories_[(std::size_t)id]); }
  const char* name(TaskId id) const { return names_[(std::size_t)id]; }

 private:
  std::array<Factory, (std::size_t)TaskId::COUNT> factories_{};
  std::array<const char*, (std::size_t)TaskId::COUNT> names_{};
};
}  // namespace phybot

// 自注册宏: 在静态初始化期把 CLASS 工厂登记到注册表。
// 注意: 当 task 编进静态库时, 需用 CMake OBJECT 库/alwayslink 防止链接器丢弃
// (见 A1 plan 的构建任务);本 A0 测试中 task 与测试同 TU, 注册必然生效。
#define REGISTER_TASK(ID, NAME, CLASS)                                       \
  namespace {                                                                \
  struct CLASS##_Reg {                                                       \
    CLASS##_Reg() {                                                          \
      ::phybot::TaskRegistry::instance().add(                               \
          (ID), (NAME),                                                      \
          [] { return std::unique_ptr<::phybot::Task>(new CLASS()); });     \
    }                                                                        \
  } CLASS##_reg_instance;                                                    \
  }
