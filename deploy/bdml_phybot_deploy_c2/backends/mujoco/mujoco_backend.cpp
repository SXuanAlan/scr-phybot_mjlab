#include "mujoco_backend.h"
#include <cmath>
#include <cstring>
#include <stdexcept>
#include <iostream>

namespace phybot {

// Policy-order joint names (21 joints) — exactly as in mujoco_sim.yaml
static const char* kJointNames[kNumDofs] = {
    "left_hip_pitch",  "left_hip_roll",   "left_hip_yaw",
    "left_knee",       "left_ankle_pitch","left_ankle_roll",
    "right_hip_pitch", "right_hip_roll",  "right_hip_yaw",
    "right_knee",      "right_ankle_pitch","right_ankle_roll",
    "waist_yaw",
    "left_shoulder_pitch","left_shoulder_roll","left_shoulder_yaw","left_elbow_pitch",
    "right_shoulder_pitch","right_shoulder_roll","right_shoulder_yaw","right_elbow_pitch"
};

MujocoBackend::MujocoBackend(const std::string& xml_path) {
    char error[2000] = "";
    m_ = mj_loadXML(xml_path.c_str(), nullptr, error, sizeof(error));
    if (!m_) {
        throw std::runtime_error(std::string("MujocoBackend: mj_loadXML failed: ") + error);
    }
    d_ = mj_makeData(m_);
    if (!d_) {
        mj_deleteModel(m_); m_ = nullptr;
        throw std::runtime_error("MujocoBackend: mj_makeData failed");
    }

    // Reset to keyframe 0 (standing pose)
    mj_resetDataKeyframe(m_, d_, 0);
    mj_forward(m_, d_);  // compute sensor data for initial state

    // Resolve joint indices (policy order)
    jnt_qpos_adr_.resize(kNumDofs);
    jnt_qvel_adr_.resize(kNumDofs);
    act_id_.resize(kNumDofs);

    for (int i = 0; i < kNumDofs; ++i) {
        int jnt_id = mj_name2id(m_, mjOBJ_JOINT, kJointNames[i]);
        if (jnt_id < 0) {
            throw std::runtime_error(std::string("MujocoBackend: joint not found: ") + kJointNames[i]);
        }
        jnt_qpos_adr_[i] = m_->jnt_qposadr[jnt_id];
        jnt_qvel_adr_[i] = m_->jnt_dofadr[jnt_id];

        int act_id = mj_name2id(m_, mjOBJ_ACTUATOR, kJointNames[i]);
        if (act_id < 0) {
            throw std::runtime_error(std::string("MujocoBackend: actuator not found: ") + kJointNames[i]);
        }
        act_id_[i] = act_id;
    }

    // Resolve sensor addresses by name (matches official mujoco_interface)
    {
        int sid = mj_name2id(m_, mjOBJ_SENSOR, "baselink-quat");
        if (sid < 0) throw std::runtime_error("MujocoBackend: sensor 'baselink-quat' not found");
        sensor_quat_adr_ = m_->sensor_adr[sid];
    }
    {
        int sid = mj_name2id(m_, mjOBJ_SENSOR, "baselink-gyro");
        if (sid < 0) throw std::runtime_error("MujocoBackend: sensor 'baselink-gyro' not found");
        sensor_gyro_adr_ = m_->sensor_adr[sid];
    }

    std::cout << "[MujocoBackend] loaded: nq=" << m_->nq
              << " nv=" << m_->nv << " nu=" << m_->nu
              << " nsensor=" << m_->nsensor
              << " timestep=" << m_->opt.timestep
              << " base_height=" << d_->qpos[2] << "\n";
}

MujocoBackend::~MujocoBackend() {
    if (d_) { mj_deleteData(d_);  d_ = nullptr; }
    if (m_) { mj_deleteModel(m_); m_ = nullptr; }
}

// ZYX Euler angles from unit quaternion — exactly replicates official quatToZyx.
// Quaternion convention: Eigen stores as (x,y,z,w); we receive components directly.
std::array<double, 3> MujocoBackend::quatToZyx(double w, double x, double y, double z) {
    std::array<double, 3> zyx{};

    // Match official: as = min(-2*(q.x*q.z - q.w*q.y), 0.99999), then clamp to [-1,1]
    double as = -2.0 * (x * z - w * y);
    as = std::min(as, 0.99999);
    as = std::max(as, -0.99999);

    // zyx(0) = yaw,  zyx(1) = pitch,  zyx(2) = roll
    zyx[0] = std::atan2(2.0 * (x * y + w * z),
                        w*w + x*x - y*y - z*z);
    zyx[1] = std::asin(as);
    zyx[2] = std::atan2(2.0 * (y * z + w * x),
                        w*w - x*x - y*y + z*z);
    return zyx;
}

void MujocoBackend::readToPackage(DataPackage& pkg) {
    // --- Joint state ---
    for (int i = 0; i < kNumDofs; ++i) {
        motor_pos_[i] = d_->qpos[jnt_qpos_adr_[i]];
        motor_vel_[i] = d_->qvel[jnt_qvel_adr_[i]];
        pkg.motor_pos[i] = motor_pos_[i];
        pkg.motor_vel[i] = motor_vel_[i];
    }

    // --- IMU quaternion (MuJoCo sensor order: w, x, y, z) ---
    double qw = d_->sensordata[sensor_quat_adr_ + 0];
    double qx = d_->sensordata[sensor_quat_adr_ + 1];
    double qy = d_->sensordata[sensor_quat_adr_ + 2];
    double qz = d_->sensordata[sensor_quat_adr_ + 3];

    // ZYX euler angles
    pkg.imu_zyx = quatToZyx(qw, qx, qy, qz);

    // --- Gyro ---
    pkg.imu_angular_vel[0] = d_->sensordata[sensor_gyro_adr_ + 0];
    pkg.imu_angular_vel[1] = d_->sensordata[sensor_gyro_adr_ + 1];
    pkg.imu_angular_vel[2] = d_->sensordata[sensor_gyro_adr_ + 2];
}

void MujocoBackend::writeFromPackage(const DataPackage& pkg) {
    // PD formula — exactly as in official MujocoInterface::SetTorque
    // ctrl[act_i] = tau_ff + P*(pos_des - pos) + D*(vel_des - vel)
    for (int i = 0; i < kNumDofs; ++i) {
        d_->ctrl[act_id_[i]] =
            pkg.motor_Torque_desire[i]
            + pkg.sim_P[i] * (pkg.motor_Pos_desire[i] - motor_pos_[i])
            + pkg.sim_D[i] * (pkg.motor_Vel_desire[i] - motor_vel_[i]);
    }
}

void MujocoBackend::step() {
    mj_step(m_, d_);
}

}  // namespace phybot
