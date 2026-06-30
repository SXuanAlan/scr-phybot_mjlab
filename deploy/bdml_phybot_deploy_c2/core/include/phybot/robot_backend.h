#pragma once
#include "phybot/data_package.h"

namespace phybot {

// Abstract interface for a robot backend (real HW or MuJoCo simulation).
class RobotBackend {
 public:
  virtual ~RobotBackend() = default;

  // Read sensor data from the backend into the DataPackage.
  virtual void readToPackage(DataPackage& pkg) = 0;

  // Write control outputs from the DataPackage to the backend.
  virtual void writeFromPackage(const DataPackage& pkg) = 0;
};

}  // namespace phybot
