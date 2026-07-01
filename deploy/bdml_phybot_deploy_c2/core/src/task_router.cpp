#include "phybot/task_router.h"
#include "phybot/input_protocol.h"
#include "phybot/data_package.h"
#include <stdexcept>
namespace phybot {
void TaskRouter::addTask(TaskCode code, const ControllerConfig& cc){
  if(code<=0||code>=kMaxTaskCode) throw std::runtime_error("select_code out of range");
  if(has_[code]) throw std::runtime_error("duplicate select_code");
  has_[code]=true; cfg_[code]=cc;
  if(cc.is_default){ if(default_!=kNoTask) throw std::runtime_error("more than one default task"); default_=code; }
}
TaskCode TaskRouter::route(TaskCode code) const {
  return (code>0&&code<kMaxTaskCode&&has_[code])?code:default_;
}
void TaskRouter::shape(TaskCode active, const DriveCmd& cmd, DataPackage& io) const {
  const ControllerConfig& c=cfg_[active];
  io.requested_task=active;
  io.js_vx_desire=c.vx.apply(cmd.vx); io.js_vy_desire=c.vy.apply(cmd.vy); io.js_OmegaZ_desire=c.wz.apply(cmd.wz);
}
}
