#pragma once
#include <string>
#include "phybot/controller_state.h"
namespace phybot {
class JsController {
 public:
  ~JsController();
  bool discover();                  // scan /dev/input/js*, open first Xbox-like by name
  bool connected() const { return fd_ >= 0; }
  bool poll(ControllerState& out);  // drain events; false if disconnected
  void close();
  const std::string& name() const { return name_; }
 private:
  int fd_ = -1;
  std::string name_;
  ControllerState state_{};
};
}  // namespace phybot
