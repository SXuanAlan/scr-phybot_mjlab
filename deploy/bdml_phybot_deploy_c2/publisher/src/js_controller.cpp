#include "phybot/js_controller.h"
#include "phybot/js_mapping.h"
#include <linux/joystick.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/ioctl.h>
#include <cerrno>
#include <cstdio>

namespace phybot {

JsController::~JsController() { close(); }

void JsController::close() {
  if (fd_ >= 0) { ::close(fd_); fd_ = -1; }
  state_.connected = false;
}

bool JsController::discover() {
  close();
  char path[32], nbuf[256];
  for (int i = 0; i < 32; ++i) {
    std::snprintf(path, sizeof(path), "/dev/input/js%d", i);
    int fd = ::open(path, O_RDONLY | O_NONBLOCK);
    if (fd < 0) continue;
    nbuf[0] = '\0';
    if (ioctl(fd, JSIOCGNAME(sizeof(nbuf)), nbuf) < 0) { ::close(fd); continue; }
    std::string name(nbuf);
    if (NameMatchesController(name)) {
      fd_ = fd; name_ = name; state_ = ControllerState{}; state_.connected = true;
      return true;
    }
    ::close(fd);
  }
  return false;
}

bool JsController::poll(ControllerState& out) {
  if (fd_ < 0) return false;
  struct js_event ev;
  for (;;) {
    ssize_t n = ::read(fd_, &ev, sizeof(ev));
    if (n == sizeof(ev)) { ApplyJsEvent(ev.type, ev.number, ev.value, state_); continue; }
    if (n < 0 && (errno == EAGAIN || errno == EWOULDBLOCK)) break;  // no more events now
    if (n < 0 && (errno == ENODEV || errno == EBADF)) { close(); return false; }  // unplugged
    break;  // partial/other -> stop draining this cycle
  }
  out = state_;
  return true;
}

}  // namespace phybot
