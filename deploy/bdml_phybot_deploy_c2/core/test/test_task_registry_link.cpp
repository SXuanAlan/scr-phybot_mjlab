#include "phybot/task_registry.h"
#include <cstring>
#include <cstdio>
int main(){ auto& r=phybot::TaskRegistry::instance(); bool z=false,w=false;
  for(int i=0;i<r.size();++i){ if(!std::strcmp(r.name(i),"zero"))z=true; if(!std::strcmp(r.name(i),"walk"))w=true; }
  std::printf("size=%d zero=%d walk=%d\n",r.size(),z,w); return (z&&w)?0:1; }
