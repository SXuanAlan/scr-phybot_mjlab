// cpg_golden_gen.cpp
//
// Golden-fixture generator for the official CPG walk task (rl_deploy_cpg).
//
// It builds a deterministic sequence of N=60 input frames (no RNG), drives one
// rl_deploy_cpg instance through all 60 ticks, and writes two CSV files capturing
// the EXACT numeric behavior so a separate implementation can be bit-exact verified.
//
// MUST be run from a build/ dir directly under deploy/phybot_deploy_c2/ so that
// the task ctor's relative paths (../RL_deploy_cpg/config/rl_params.yaml and the
// policy .pt) resolve, and so the test/ output paths land beside the config.
//
// Outputs (relative to the working dir, i.e. ../RL_deploy_cpg/test/...):
//   golden_inputs.csv   : one row per input frame (60 rows)
//   golden_official.csv : one row per tick (60 rows)

#include "RL_deploy_cpg/include/rl_deploy.h"

#include <eigen3/Eigen/Dense>

#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>

namespace {

constexpr int kNumFrames = 60;
constexpr int kNumDofs = 21;
constexpr int kObsDim = 283;
constexpr int kDecimation = 10;

// Same default_dof_pos as RL_deploy_cpg/config/rl_params.yaml.
const Eigen::VectorXd &DefaultDofPos() {
    static const Eigen::VectorXd defaults = [] {
        Eigen::VectorXd v(kNumDofs);
        v << -0.242, 0.029, 0.007, 0.49, -0.295, 0.0,
             -0.242, -0.029, -0.007, 0.49, -0.295, 0.0,
             0.0,
             0.0, 0.0, 0.0, -0.17,
             0.0, 0.0, 0.0, -0.17;
        return v;
    }();
    return defaults;
}

struct Frame {
    Eigen::Vector3d imu_zyx;          // ZYX euler (yaw, pitch, roll)
    Eigen::Vector3d imu_angular_vel;  // body angular velocity
    Eigen::VectorXd motor_pos;        // 21
    Eigen::VectorXd motor_vel;        // 21
    Eigen::VectorXd motor_torque;     // 21 (read into tor_origin; not in golden_official)
    double js_vx;
    double js_vy;
    double js_wz;
};

// Deterministic frame from index t (and joint i). All formulas use std::sin/cos,
// magnitudes kept physically modest. js_vx is shaped so |js_vx| crosses the
// walk_flag threshold 0.3 in BOTH directions across the inference ticks.
Frame MakeFrame(int t) {
    Frame f;
    const double td = static_cast<double>(t);

    // nonzero yaw/pitch/roll within +/- 0.3 rad
    f.imu_zyx = Eigen::Vector3d(
        0.20 * std::sin(0.07 * td),
        0.15 * std::sin(0.05 * td + 0.5),
        0.10 * std::sin(0.09 * td + 1.0));

    // nonzero angular velocity within +/- 0.5
    f.imu_angular_vel = Eigen::Vector3d(
        0.30 * std::sin(0.11 * td),
        0.25 * std::cos(0.08 * td),
        0.20 * std::sin(0.13 * td + 0.3));

    f.motor_pos = Eigen::VectorXd(kNumDofs);
    f.motor_vel = Eigen::VectorXd(kNumDofs);
    f.motor_torque = Eigen::VectorXd(kNumDofs);
    const Eigen::VectorXd &def = DefaultDofPos();
    for (int i = 0; i < kNumDofs; ++i) {
        const double id = static_cast<double>(i);
        f.motor_pos(i) = def(i) + 0.05 * std::sin(0.10 * td + 0.30 * id);
        f.motor_vel(i) = 0.20 * std::sin(0.15 * td + 0.20 * id);
        f.motor_torque(i) = 0.10 * std::sin(0.12 * td + 0.25 * id);
    }

    // js commands: js_vx amplitude 0.55 so |js_vx| crosses 0.3 both ways.
    f.js_vx = 0.55 * std::sin(0.21 * td + 0.10);
    f.js_vy = 0.30 * std::sin(0.17 * td);
    f.js_wz = 0.40 * std::cos(0.12 * td);
    return f;
}

void WriteVec(std::ofstream &os, const Eigen::VectorXd &v) {
    for (int i = 0; i < v.size(); ++i) {
        os << "," << v(i);
    }
}

}  // namespace

int main() {
    std::cout << "[cpg_golden_gen] starting" << std::endl;

    // ---- Build the deterministic input sequence ----
    std::vector<Frame> frames;
    frames.reserve(kNumFrames);
    for (int t = 0; t < kNumFrames; ++t) {
        frames.push_back(MakeFrame(t));
    }

    // ---- Write golden_inputs.csv ----
    {
        std::ofstream in("../RL_deploy_cpg/test/golden_inputs.csv");
        if (!in.is_open()) {
            std::cerr << "[FAIL] cannot open ../RL_deploy_cpg/test/golden_inputs.csv" << std::endl;
            return 1;
        }
        in << std::setprecision(17);
        in << "# columns: frame, imu_zyx(3), imu_angular_vel(3), motor_pos(21), "
              "motor_vel(21), js_vx, js_vy, js_wz  [total 1+3+3+21+21+3 = 52]\n";
        for (int t = 0; t < kNumFrames; ++t) {
            const Frame &f = frames[t];
            in << t;
            WriteVec(in, f.imu_zyx);
            WriteVec(in, f.imu_angular_vel);
            WriteVec(in, f.motor_pos);
            WriteVec(in, f.motor_vel);
            in << "," << f.js_vx << "," << f.js_vy << "," << f.js_wz;
            in << "\n";
        }
        std::cout << "[cpg_golden_gen] wrote golden_inputs.csv (" << kNumFrames
                  << " rows)" << std::endl;
    }

    // ---- Construct the official task (loads policy + config from ../) ----
    rl_deploy_cpg task;

    // ---- Open golden_official.csv ----
    std::ofstream out("../RL_deploy_cpg/test/golden_official.csv");
    if (!out.is_open()) {
        std::cerr << "[FAIL] cannot open ../RL_deploy_cpg/test/golden_official.csv" << std::endl;
        return 1;
    }
    out << std::setprecision(17);
    out << "# columns: tick, did_infer, obs[0..282](283), motor_Pos_desire(21), "
           "sim_P(21), sim_D(21)  [total 1+1+283+21+21+21 = 348]\n";

    // ---- Drive all 60 ticks ----
    for (int t = 0; t < kNumFrames; ++t) {
        const Frame &f = frames[t];

        DataPackage data;
        // Size the dynamic vectors explicitly (mirrors the walk_amp dry-run mock;
        // avoids depending on DataPackage::init()/datapackage.yaml).
        data.actuatedDofNum = kNumDofs;
        data.motor_pos = f.motor_pos;
        data.motor_vel = f.motor_vel;
        data.motor_torque = f.motor_torque;
        data.imu_zyx = f.imu_zyx;
        data.imu_angular_vel = f.imu_angular_vel;

        data.torq_desire = Eigen::VectorXd::Zero(kNumDofs);
        data.motor_Pos_desire = Eigen::VectorXd::Zero(kNumDofs);
        data.motor_Vel_desire = Eigen::VectorXd::Zero(kNumDofs);
        data.motor_Torque_desire = Eigen::VectorXd::Zero(kNumDofs);
        data.sim_P = Eigen::VectorXd::Zero(kNumDofs);
        data.sim_D = Eigen::VectorXd::Zero(kNumDofs);

        data.js_vx_desire = f.js_vx;
        data.js_vy_desire = f.js_vy;
        data.js_OmegaZ_desire = f.js_wz;

        task.GetDataFromPackage(data);
        task.Step();
        task.SetDataToPackage(data);

        // step_num after the t-th Step() equals t+1 (step_num: 0 -> 1, then +1).
        const int step_num = t + 1;
        const int did_infer = (step_num % kDecimation == 0) ? 1 : 0;

        const Eigen::VectorXd &obs = task.observation();
        if (obs.size() != kObsDim) {
            std::cerr << "[FAIL] obs size " << obs.size() << " != " << kObsDim << std::endl;
            return 1;
        }

        out << t << "," << did_infer;
        WriteVec(out, obs);
        WriteVec(out, data.motor_Pos_desire);
        WriteVec(out, data.sim_P);
        WriteVec(out, data.sim_D);
        out << "\n";
    }

    out.flush();
    std::cout << "[cpg_golden_gen] wrote golden_official.csv (" << kNumFrames
              << " rows, " << kObsDim << " obs cols)" << std::endl;
    std::cout << "[cpg_golden_gen] DONE" << std::endl;
    return 0;
}
