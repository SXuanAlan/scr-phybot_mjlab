#pragma once
#include <array>
#include <string>
#include <vector>
#include <mujoco/mujoco.h>
#include "phybot/robot_backend.h"
#include "phybot/data_package.h"
#include "phybot/robot_consts.h"

namespace phybot {

// MujocoBackend: headless physics-only MuJoCo backend.
// Loads the robot XML and uses keyframe-0 as initial state.
// Sensor/joint mapping exactly replicates the official MujocoInterface.
class MujocoBackend : public RobotBackend {
 public:
  // xml_path: absolute or relative path to phybot_c2.xml
  explicit MujocoBackend(const std::string& xml_path);
  ~MujocoBackend() override;

  // RobotBackend interface
  void readToPackage(DataPackage& pkg) override;
  void writeFromPackage(const DataPackage& pkg) override;

  // Advance the simulation by one physics step.
  void step();

  // Diagnostics
  double baseHeight() const { return m_ ? d_->qpos[2] : 0.0; }
  double baseX()      const { return m_ ? d_->qpos[0] : 0.0; }
  double baseY()      const { return m_ ? d_->qpos[1] : 0.0; }
  double simTime()    const { return m_ ? d_->time     : 0.0; }

 private:
  mjModel* m_{nullptr};
  mjData*  d_{nullptr};

  // Joint indices (policy order, 21 joints)
  std::vector<int> jnt_qpos_adr_;   // qpos address for each joint
  std::vector<int> jnt_qvel_adr_;   // qvel address for each joint
  std::vector<int> act_id_;         // actuator id for each joint

  // Sensor addresses
  int sensor_quat_adr_{0};  // baselink-quat (dim 4, order w,x,y,z)
  int sensor_gyro_adr_{0};  // baselink-gyro (dim 3)

  // Cached motor state (needed for PD in writeFromPackage)
  std::array<double, kNumDofs> motor_pos_{};
  std::array<double, kNumDofs> motor_vel_{};

  // ZYX euler from unit quaternion (matches official quatToZyx exactly)
  static std::array<double, 3> quatToZyx(double w, double x, double y, double z);
};

}  // namespace phybot
