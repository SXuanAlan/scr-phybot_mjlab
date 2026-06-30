#include "phybot/task_id.h"
#include "phybot/data_package.h"
#include "phybot/task.h"
#include "check.h"

namespace {
// 哑 task: 记录各生命周期被调用次数。
struct DummyTask : phybot::Task {
  int entered = 0, ran = 0, exited = 0;
  using phybot::Task::Task;
  void enter(phybot::DataPackage&) override { ++entered; }
  void run(phybot::DataPackage&) override { ++ran; }
  void exit(phybot::DataPackage&) override { ++exited; }
};
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

int main() {
  test_task_basics();
  test_datapackage_defaults();
  if (g_check_failures == 0) std::printf("ALL TESTS PASSED\n");
  return g_check_failures == 0 ? 0 : 1;
}
