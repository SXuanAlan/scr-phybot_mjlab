#include "phybot/controller_config.h"
#include <yaml-cpp/yaml.h>
#include <string>
#ifndef PHYBOT_TASKS_DIR
#error "PHYBOT_TASKS_DIR must be defined"
#endif
namespace phybot {
static AxisRange axis(const YAML::Node& n){ AxisRange a; if(n&&n["range"]){a.lo=n["range"][0].as<double>();a.hi=n["range"][1].as<double>();} return a; }
ControllerConfig load_controller_config(const char* name){
  const YAML::Node y=YAML::LoadFile(std::string(PHYBOT_TASKS_DIR)+"/"+name+"/config/controller.yaml");
  ControllerConfig c; c.select_code=y["select_code"].as<int>();
  c.is_default=y["default"]?y["default"].as<bool>():false;
  c.vx=axis(y["vx"]); c.vy=axis(y["vy"]); c.wz=axis(y["wz"]); return c; }
}
