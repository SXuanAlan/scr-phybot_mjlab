#include "phybot/task_code.h"
#include "phybot/data_package.h"
#include "phybot/task.h"
#include "check.h"
#include "phybot/task_manager.h"
#include "phybot/task_registry.h"
#include <memory>
#include <string>

namespace {
// Codes used by this test (a task's code is assigned by the caller at add()-time;
// it is no longer baked into the task itself). kMimic is intentionally never
// added to any TaskManager/TaskRegistry instance in this file, so it exercises
// the "target not present" paths below.
constexpr phybot::TaskCode kZero    = 1;
constexpr phybot::TaskCode kWalkAmp = 2;
constexpr phybot::TaskCode kMimic   = 3;   // unregistered on purpose

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
  phybot::TaskCode force_change{phybot::kNoTask};   // 设为非 kNoTask 则强制自跳转
  bool block_exit = false;
  int ran = 0, entered = 0, exited = 0;
  explicit CfgTask(std::string n) : phybot::Task(std::move(n)) {}
  void enter(phybot::DataPackage&) override { ++entered; }
  void run(phybot::DataPackage&) override { ++ran; }
  void exit(phybot::DataPackage&) override { ++exited; }
  phybot::TaskCode checkChange(phybot::DataPackage&) override { return force_change; }
  bool canExit(phybot::DataPackage&) override { return !block_exit; }
};

std::unique_ptr<CfgTask> make(const char* n, CfgTask** out) {
  auto t = std::make_unique<CfgTask>(n);
  *out = t.get();
  return t;
}
}  // namespace

namespace {
struct RegZeroTask : phybot::Task {
  RegZeroTask() : phybot::Task("zero") {}
  void run(phybot::DataPackage&) override {}
};
struct RegWalkTask : phybot::Task {
  RegWalkTask() : phybot::Task("walk_amp") {}
  void run(phybot::DataPackage&) override {}
};
}  // namespace
REGISTER_TASK("zero", RegZeroTask)
REGISTER_TASK("walk_amp", RegWalkTask)

static void test_datapackage_defaults() {
  phybot::DataPackage io;
  CHECK(io.requested_task == phybot::kNoTask);
  CHECK(io.current_task == phybot::kNoTask);
}

static void test_task_basics() {
  DummyTask t{"walk_amp"};
  phybot::DataPackage io;
  CHECK(t.name() == "walk_amp");
  CHECK(t.checkChange(io) == phybot::kNoTask);   // 默认: kNoTask=不变
  CHECK(t.canExit(io) == true);                  // 默认: 允许退出
  t.enter(io); t.run(io); t.exit(io);
  CHECK(t.entered == 1 && t.ran == 1 && t.exited == 1);
}

static void test_manager_runs_active_each_tick() {
  using namespace phybot;
  TaskManager m;
  CfgTask *zero;
  m.add(make("zero", &zero), kZero);
  DataPackage io;
  m.start(kZero, io);
  CHECK(zero->entered == 1);
  CHECK(io.current_task == kZero);
  m.tick(io);
  m.tick(io);
  CHECK(zero->ran == 2);          // 每拍都跑当前
  CHECK(m.current() == kZero);
}

static void test_manager_switches_on_request() {
  using namespace phybot;
  TaskManager m;
  CfgTask *zero, *walk;
  m.add(make("zero", &zero), kZero);
  m.add(make("walk_amp", &walk), kWalkAmp);
  DataPackage io;
  m.start(kZero, io);
  io.requested_task = kWalkAmp;
  m.tick(io);
  CHECK(zero->exited == 1);        // 退出旧
  CHECK(walk->entered == 1);       // 进入新
  CHECK(walk->ran == 1);           // 切换当拍跑新 task
  CHECK(zero->ran == 0);           // 切换当拍不跑旧 task
  CHECK(io.current_task == kWalkAmp);
  CHECK(m.current() == kWalkAmp);
}

static void test_manager_no_switch_when_same() {
  using namespace phybot;
  TaskManager m;
  CfgTask *zero;
  m.add(make("zero", &zero), kZero);
  DataPackage io;
  m.start(kZero, io);
  io.requested_task = kZero;   // 与当前相同
  m.tick(io);
  CHECK(zero->exited == 0);           // 不切换
  CHECK(zero->ran == 1);
}

static void test_manager_canExit_blocks_switch() {
  using namespace phybot;
  TaskManager m;
  CfgTask *zero, *walk;
  m.add(make("zero", &zero), kZero);
  m.add(make("walk_amp", &walk), kWalkAmp);
  DataPackage io;
  m.start(kZero, io);
  zero->block_exit = true;             // 互锁: 拒绝离开
  io.requested_task = kWalkAmp;
  m.tick(io);
  CHECK(zero->exited == 0);
  CHECK(walk->entered == 0);
  CHECK(m.current() == kZero);   // 维持当前
  CHECK(zero->ran == 1);   // 被互锁拒绝切换的那拍, 当前 task 仍照常 run
}

static void test_manager_checkChange_auto_transition() {
  using namespace phybot;
  TaskManager m;
  CfgTask *zero, *walk;
  m.add(make("zero", &zero), kZero);
  m.add(make("walk_amp", &walk), kWalkAmp);
  DataPackage io;
  m.start(kZero, io);
  zero->force_change = kWalkAmp;   // 状态自决跳转, 优先于 requested
  io.requested_task = kZero;
  m.tick(io);
  CHECK(m.current() == kWalkAmp);
  CHECK(walk->entered == 1);
  CHECK(walk->ran == 1);   // 自动跳转当拍: 新 task 跑一次
  CHECK(zero->ran == 0);   // 自动跳转当拍: 旧 task 不跑
}

static void test_manager_ignores_unregistered_target() {
  using namespace phybot;
  TaskManager m;
  CfgTask *zero;
  m.add(make("zero", &zero), kZero);   // kMimic 未注册 (never add()-ed)
  DataPackage io;
  m.start(kZero, io);
  io.requested_task = kMimic;          // 目标不存在 → 安全忽略
  m.tick(io);
  CHECK(zero->exited == 0);
  CHECK(m.current() == kZero);
}

static void test_manager_ignores_out_of_range_request() {
  using namespace phybot;
  TaskManager m;
  CfgTask *zero;
  m.add(make("zero", &zero), kZero);
  DataPackage io;
  m.start(kZero, io);
  io.requested_task = kMaxTaskCode;   // 越界哨兵值 → 必须安全忽略, 不越界访问
  m.tick(io);
  CHECK(m.current() == kZero);
  CHECK(zero->ran == 1);               // 仍照常跑当前
}

static void test_registry_self_registration() {
  using namespace phybot;
  auto& reg = TaskRegistry::instance();
  CHECK(reg.size() == 2);
  CHECK(std::string(reg.name(0)) == "zero");
  CHECK(std::string(reg.name(1)) == "walk_amp");
}

// Name-keyed registry no longer stores a code (codes are assigned by the caller
// at TaskManager::add()-time), so "loading the registry into a manager" now means
// walking size()/make() and choosing explicit codes -- exercised here end-to-end.
static void test_registry_make_and_manage() {
  using namespace phybot;
  auto& reg = TaskRegistry::instance();
  TaskManager m;
  for (int i = 0; i < reg.size(); ++i) m.add(reg.make(i), i + 1);   // "zero"->1, "walk_amp"->2
  DataPackage io;
  m.start(kZero, io);
  io.requested_task = kWalkAmp;
  m.tick(io);
  CHECK(m.current() == kWalkAmp);   // 注册的 task 可被切换
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
  test_manager_ignores_out_of_range_request();
  test_registry_self_registration();
  test_registry_make_and_manage();
  if (g_check_failures == 0) std::printf("ALL TESTS PASSED\n");
  return g_check_failures == 0 ? 0 : 1;
}
