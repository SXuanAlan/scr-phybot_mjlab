#include "phybot/task_id.h"
#include "phybot/data_package.h"
#include "phybot/task.h"
#include "check.h"
#include "phybot/task_manager.h"
#include <memory>

namespace {
// 哑 task: 记录各生命周期被调用次数。
struct DummyTask : phybot::Task {
  int entered = 0, ran = 0, exited = 0;
  using phybot::Task::Task;
  void enter(phybot::DataPackage&) override { ++entered; }
  void run(phybot::DataPackage&) override { ++ran; }
  void exit(phybot::DataPackage&) override { ++exited; }
};

// 可配置哑 task: 控制 checkChange / canExit。
struct CfgTask : phybot::Task {
  using phybot::Task::Task;
  phybot::TaskId force_change;     // 设为非自身则强制自跳转
  bool block_exit = false;
  int ran = 0, entered = 0, exited = 0;
  CfgTask(phybot::TaskId id, std::string n)
      : phybot::Task(id, std::move(n)), force_change(id) {}
  void enter(phybot::DataPackage&) override { ++entered; }
  void run(phybot::DataPackage&) override { ++ran; }
  void exit(phybot::DataPackage&) override { ++exited; }
  phybot::TaskId checkChange(phybot::DataPackage&) override { return force_change; }
  bool canExit(phybot::DataPackage&) override { return !block_exit; }
};

std::unique_ptr<CfgTask> make(phybot::TaskId id, const char* n,
                              CfgTask** out) {
  auto t = std::make_unique<CfgTask>(id, n);
  *out = t.get();
  return t;
}
}  // namespace

static void test_datapackage_defaults() {
  phybot::DataPackage io;
  CHECK(io.requested_task == phybot::TaskId::ZERO);
  CHECK(io.current_task == phybot::TaskId::ZERO);
}

static void test_task_basics() {
  DummyTask t{phybot::TaskId::WALK_AMP, "walk_amp"};
  phybot::DataPackage io;
  CHECK(t.id() == phybot::TaskId::WALK_AMP);
  CHECK(t.name() == "walk_amp");
  CHECK(t.checkChange(io) == phybot::TaskId::WALK_AMP);  // 默认: 返回自身=不变
  CHECK(t.canExit(io) == true);                          // 默认: 允许退出
  t.enter(io); t.run(io); t.exit(io);
  CHECK(t.entered == 1 && t.ran == 1 && t.exited == 1);
}

static void test_manager_runs_active_each_tick() {
  using namespace phybot;
  TaskManager m;
  CfgTask *zero;
  m.add(make(TaskId::ZERO, "zero", &zero));
  DataPackage io;
  m.start(TaskId::ZERO, io);
  CHECK(zero->entered == 1);
  CHECK(io.current_task == TaskId::ZERO);
  m.tick(io);
  m.tick(io);
  CHECK(zero->ran == 2);          // 每拍都跑当前
  CHECK(m.current() == TaskId::ZERO);
}

static void test_manager_switches_on_request() {
  using namespace phybot;
  TaskManager m;
  CfgTask *zero, *walk;
  m.add(make(TaskId::ZERO, "zero", &zero));
  m.add(make(TaskId::WALK_AMP, "walk_amp", &walk));
  DataPackage io;
  m.start(TaskId::ZERO, io);
  io.requested_task = TaskId::WALK_AMP;
  m.tick(io);
  CHECK(zero->exited == 1);        // 退出旧
  CHECK(walk->entered == 1);       // 进入新
  CHECK(walk->ran == 1);           // 切换当拍跑新 task
  CHECK(zero->ran == 0);           // 切换当拍不跑旧 task
  CHECK(io.current_task == TaskId::WALK_AMP);
  CHECK(m.current() == TaskId::WALK_AMP);
}

static void test_manager_no_switch_when_same() {
  using namespace phybot;
  TaskManager m;
  CfgTask *zero;
  m.add(make(TaskId::ZERO, "zero", &zero));
  DataPackage io;
  m.start(TaskId::ZERO, io);
  io.requested_task = TaskId::ZERO;   // 与当前相同
  m.tick(io);
  CHECK(zero->exited == 0);           // 不切换
  CHECK(zero->ran == 1);
}

static void test_manager_canExit_blocks_switch() {
  using namespace phybot;
  TaskManager m;
  CfgTask *zero, *walk;
  m.add(make(TaskId::ZERO, "zero", &zero));
  m.add(make(TaskId::WALK_AMP, "walk_amp", &walk));
  DataPackage io;
  m.start(TaskId::ZERO, io);
  zero->block_exit = true;             // 互锁: 拒绝离开
  io.requested_task = TaskId::WALK_AMP;
  m.tick(io);
  CHECK(zero->exited == 0);
  CHECK(walk->entered == 0);
  CHECK(m.current() == TaskId::ZERO);  // 维持当前
}

static void test_manager_checkChange_auto_transition() {
  using namespace phybot;
  TaskManager m;
  CfgTask *zero, *walk;
  m.add(make(TaskId::ZERO, "zero", &zero));
  m.add(make(TaskId::WALK_AMP, "walk_amp", &walk));
  DataPackage io;
  m.start(TaskId::ZERO, io);
  zero->force_change = TaskId::WALK_AMP;   // 状态自决跳转, 优先于 requested
  io.requested_task = TaskId::ZERO;
  m.tick(io);
  CHECK(m.current() == TaskId::WALK_AMP);
  CHECK(walk->entered == 1);
}

static void test_manager_ignores_unregistered_target() {
  using namespace phybot;
  TaskManager m;
  CfgTask *zero;
  m.add(make(TaskId::ZERO, "zero", &zero));   // MIMIC 未注册
  DataPackage io;
  m.start(TaskId::ZERO, io);
  io.requested_task = TaskId::MIMIC;          // 目标不存在 → 安全忽略
  m.tick(io);
  CHECK(zero->exited == 0);
  CHECK(m.current() == TaskId::ZERO);
}

int main() {
  test_datapackage_defaults();
  test_task_basics();
  test_manager_runs_active_each_tick();
  test_manager_switches_on_request();
  test_manager_no_switch_when_same();
  test_manager_canExit_blocks_switch();
  test_manager_checkChange_auto_transition();
  test_manager_ignores_unregistered_target();
  if (g_check_failures == 0) std::printf("ALL TESTS PASSED\n");
  return g_check_failures == 0 ? 0 : 1;
}
