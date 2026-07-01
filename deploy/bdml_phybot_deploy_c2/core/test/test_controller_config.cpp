#include "phybot/controller_config.h"
#include <cmath>
#include <cstdio>
static bool nr(double a,double b){return std::fabs(a-b)<1e-9;}
int main(){ using namespace phybot;
  auto w=load_controller_config("walk");
  if(w.select_code!=2||w.is_default) {std::printf("FAIL walk\n");return 1;}
  if(!nr(w.vx.apply(1),1.0)||!nr(w.vx.apply(-1),-0.4)||!nr(w.vx.apply(0),0)||!nr(w.vx.apply(2),1.0)){std::printf("FAIL map\n");return 1;}
  auto z=load_controller_config("zero");
  if(z.select_code!=1||!z.is_default){std::printf("FAIL zero\n");return 1;}
  std::printf("OK\n");return 0; }
