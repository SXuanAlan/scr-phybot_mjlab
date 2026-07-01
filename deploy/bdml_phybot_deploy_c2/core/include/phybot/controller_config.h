#pragma once
#include <algorithm>
#include "phybot/task_code.h"
namespace phybot {
struct AxisRange { double lo{0}, hi{0};
  double apply(double n) const { double v=(n>=0)?n*hi:n*(-lo); return std::clamp(v,lo,hi); } };
struct ControllerConfig { TaskCode select_code{0}; bool is_default{false}; AxisRange vx, vy, wz; };
ControllerConfig load_controller_config(const char* task_name);
}
