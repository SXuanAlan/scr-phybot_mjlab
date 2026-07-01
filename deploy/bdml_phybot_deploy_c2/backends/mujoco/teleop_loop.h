#pragma once
// ApplyCmdAndTick: one control-loop step, shared by the LCM-driven real-host main
// (apps/sim_teleop.cpp) and the LCM-free headless smoke test
// (core/test/test_teleop_smoke.cpp). Both callers are responsible for producing a
// DriveCmd (from LCM or set directly in-process) and pass it in here unchanged.
//
// Maps DriveCmd.model -> TaskCode (1=ZERO, 2=WALK; anything else defaults to ZERO,
// the safe power-on state), clamps the joystick axes into the DataPackage, then
// drives one full backend<->task-manager tick: read sensors, tick the TaskManager
// (which performs the ZERO<->WALK switch when requested), write actuator targets,
// step physics.
#include <algorithm>
#include "phybot/data_package.h"
#include "phybot/input_protocol.h"
#include "phybot/task_manager.h"
#include "mujoco_backend.h"

namespace phybot {

inline int ModelToTask(int8_t model) {
  return model == 2 ? 2 : 1;
}

inline void ApplyCmdAndTick(const DriveCmd& cmd, DataPackage& io,
                             TaskManager& manager, MujocoBackend& backend) {
  io.requested_task   = ModelToTask(cmd.model);
  io.js_vx_desire      = std::clamp(static_cast<double>(cmd.vx), -0.4, 1.0);
  io.js_vy_desire      = std::clamp(static_cast<double>(cmd.vy), -0.1, 0.1);
  io.js_OmegaZ_desire  = std::clamp(static_cast<double>(cmd.wz), -1.0, 1.0);

  backend.readToPackage(io);
  manager.tick(io);
  backend.writeFromPackage(io);
  backend.step();
}

}  // namespace phybot
