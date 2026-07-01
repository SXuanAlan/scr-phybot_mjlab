#include "phybot/task_router.h"
#include "phybot/input_protocol.h"
#include "phybot/data_package.h"
#include <cmath>
#include <cstdio>
static bool nr(double a,double b){return std::fabs(a-b)<1e-9;}
int main(){ using namespace phybot; TaskRouter r;
  ControllerConfig z; z.select_code=1; z.is_default=true;
  ControllerConfig w; w.select_code=2; w.vx={-0.4,1.0}; w.wz={-1.0,1.0};
  r.addTask(1,z); r.addTask(2,w);
  if(r.route(2)!=2||r.route(1)!=1||r.route(9)!=1||r.defaultCode()!=1){std::printf("FAIL route\n");return 1;}
  DriveCmd c; c.model=2; c.vx=1.0f; c.wz=-1.0f; DataPackage io; r.shape(2,c,io);
  if(!nr(io.js_vx_desire,1.0)||!nr(io.js_OmegaZ_desire,-1.0)||io.requested_task!=2){std::printf("FAIL shape\n");return 1;}
  bool t=false; try{ControllerConfig d;d.select_code=2;r.addTask(2,d);}catch(...){t=true;} if(!t){std::printf("FAIL dup\n");return 1;}
  std::printf("OK\n");return 0; }
