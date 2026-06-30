#pragma once
namespace phybot {
// 归一化手柄状态。设备层只填原始归一化值([-1,1]); 死区/缩放/反向在 mapping 层。
struct ControllerState {
  bool a=false, b=false, x=false, y=false;
  bool lb=false, rb=false;
  bool back=false, start=false, guide=false;
  bool ls=false, rs=false;                       // 摇杆按下
  float left_x=0, left_y=0, right_x=0, right_y=0; // [-1,1]
  float lt=0, rt=0;                              // 扳机, xpad 原始约定(松≈-1, 满≈+1)
  float dpad_x=0, dpad_y=0;                      // 方向键(轴形式)
  bool connected=false;
};
}  // namespace phybot
