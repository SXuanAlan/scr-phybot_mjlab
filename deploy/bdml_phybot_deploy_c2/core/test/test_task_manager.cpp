#include "phybot/task_id.h"
#include "phybot/data_package.h"
#include "check.h"

static void test_datapackage_defaults() {
  phybot::DataPackage io;
  CHECK(io.requested_task == phybot::TaskId::ZERO);
  CHECK(io.current_task == phybot::TaskId::ZERO);
}

int main() {
  test_datapackage_defaults();
  if (g_check_failures == 0) std::printf("ALL TESTS PASSED\n");
  return g_check_failures == 0 ? 0 : 1;
}
