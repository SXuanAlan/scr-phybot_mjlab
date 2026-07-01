#include "walk/walk_config.h"
#include <cstdio>
#include <string>
int main(){ auto c=phybot::walk_config();
  if(c.policy_path.empty()||c.policy_path[0]!='/'||c.policy_path.find("cpg_policy.pt")==std::string::npos){
    std::printf("FAIL %s\n",c.policy_path.c_str());return 1;} std::printf("OK %s\n",c.policy_path.c_str());return 0; }
