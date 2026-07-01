// Bit-exact alignment test: replay golden_inputs.csv through new headers and
// compare obs[283] to golden_official.csv at each of the 6 inference ticks.
// Cadence reproduced from rl_deploy_cpg/src/rl_deploy.cpp:
//   step_num: 0→1 on first call, +1 thereafter; inference when step_num%10==0.
//   History pushed inference-only, AFTER obs is built.
//   last_action: zeros at first inference; subsequent = prev raw network output.
//   History stores q*dof_pos_scale (no default subtraction).

#ifndef GOLDEN_DIR
#error "GOLDEN_DIR must be defined via target_compile_definitions"
#endif

#include "phybot/robot_consts.h"
#include "walk/walk_config.h"
#include "phybot/gravity.h"
#include "walk/walk_history.h"
#include "walk/walk_obs.h"

#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

static std::vector<std::vector<double>> parse_csv(const std::string& path) {
    std::ifstream f(path);
    if (!f.is_open()) {
        std::fprintf(stderr, "Cannot open: %s\n", path.c_str());
        std::exit(1);
    }
    std::vector<std::vector<double>> rows;
    std::string line;
    while (std::getline(f, line)) {
        if (line.empty() || line[0] == '#') continue;
        std::vector<double> row;
        std::stringstream ss(line);
        std::string tok;
        while (std::getline(ss, tok, ',')) row.push_back(std::stod(tok));
        rows.push_back(std::move(row));
    }
    return rows;
}

int main() {
    const std::string gdir = GOLDEN_DIR;
    auto inp = parse_csv(gdir + "/golden_inputs.csv");
    auto out = parse_csv(gdir + "/golden_official.csv");

    if (inp.size() != 60 || out.size() != 60) {
        std::fprintf(stderr, "Expected 60 rows each; got %zu/%zu\n", inp.size(), out.size());
        return 1;
    }

    auto cfg = phybot::walk_config();
    phybot::WalkHistory hist;

    std::array<double, phybot::kNumDofs> last_action{};  // zeros for first inference
    int step_num = 0;
    int inf_idx = 0;
    int failures = 0;

    // per-tick reporting
    double max_diff[6] = {};
    int    worst_idx[6];
    for (int i = 0; i < 6; i++) worst_idx[i] = -1;

    static const double kTol = 1e-6;

    for (int frame = 0; frame < 60; frame++) {
        const auto& ir = inp[frame];
        const auto& or_ = out[frame];

        // --- parse inputs ---
        std::array<double,3> imu_zyx  = {ir[1], ir[2], ir[3]};
        std::array<double,3> imu_ang  = {ir[4], ir[5], ir[6]};
        std::array<double, phybot::kNumDofs> q{}, dq{};
        for (int i = 0; i < phybot::kNumDofs; i++) q[i]  = ir[7 + i];
        for (int i = 0; i < phybot::kNumDofs; i++) dq[i] = ir[28 + i];
        double js_vx = ir[49], js_vy = ir[50], js_wz = ir[51];

        // --- parse golden output row ---
        int tick      = static_cast<int>(or_[0]);
        int did_infer = static_cast<int>(or_[1]);
        // obs[0..282] → cols 2..284
        // motor_Pos_desire[0..20] → cols 285..305
        std::array<double, phybot::kObsDim> g_obs{};
        for (int i = 0; i < phybot::kObsDim; i++) g_obs[i] = or_[2 + i];
        std::array<double, phybot::kNumDofs> g_mpd{};
        for (int i = 0; i < phybot::kNumDofs; i++) g_mpd[i] = or_[285 + i];

        // --- replicate Step() cadence ---
        if (step_num == 0) step_num = 1; else step_num++;

        if (step_num % phybot::kDecimation == 0) {
            // sanity: golden should agree this is an inference tick
            if (!did_infer) {
                std::fprintf(stderr, "frame %d: our cadence says inference but golden did_infer=0\n", frame);
                return 1;
            }

            // build obs
            phybot::WalkObsInputs in{};
            in.gravity    = phybot::GravityFromEulerZYX(imu_zyx);
            in.ang_vel    = imu_ang;
            in.q          = q;
            in.dq         = dq;
            in.last_action = last_action;
            in.js_vx = js_vx; in.js_vy = js_vy; in.js_wz = js_wz;

            float pos105[phybot::kHistFramesInObs * phybot::kNumDofs];
            float vel105[phybot::kHistFramesInObs * phybot::kNumDofs];
            hist.NewestPos(pos105);
            hist.NewestVel(vel105);

            float obs[phybot::kObsDim];
            phybot::BuildWalkObs(in, cfg, pos105, vel105, obs);

            // compare element-by-element
            double md = 0.0; int wi = -1;
            for (int i = 0; i < phybot::kObsDim; i++) {
                double d = std::fabs(static_cast<double>(obs[i]) - g_obs[i]);
                if (d > md) { md = d; wi = i; }
            }
            max_diff[inf_idx]   = md;
            worst_idx[inf_idx]  = wi;

            std::printf("tick %2d  inf#%d  max_abs_diff=%.3e  worst=obs[%d]  %s\n",
                        tick, inf_idx, md, wi, md <= kTol ? "PASS" : "FAIL");

            if (md > kTol) {
                // print mismatching range for diagnosis
                for (int i = 0; i < phybot::kObsDim; i++) {
                    double d = std::fabs(static_cast<double>(obs[i]) - g_obs[i]);
                    if (d > kTol)
                        std::printf("    obs[%d]: got %.8e  want %.8e  diff=%.3e\n",
                                    i, (double)obs[i], g_obs[i], d);
                }
                failures++;
            }

            // recover last_action from golden motor_Pos_desire
            for (int i = 0; i < phybot::kNumDofs; i++)
                last_action[i] = (g_mpd[i] - cfg.default_dof_pos[i]) / cfg.action_scale[i];

            // push to history AFTER obs build (CPG cadence: history updated after inference)
            hist.Push(q, dq, cfg);

            inf_idx++;
        } else {
            if (did_infer) {
                std::fprintf(stderr, "frame %d: golden says did_infer=1 but our cadence says no inference\n", frame);
                return 1;
            }
        }
    }

    std::printf("\n--- Summary (tol=%.0e) ---\n", kTol);
    for (int i = 0; i < 6; i++)
        std::printf("  inf#%d: max_abs_diff=%.3e  worst=obs[%d]  %s\n",
                    i, max_diff[i], worst_idx[i], max_diff[i] <= kTol ? "PASS" : "FAIL");

    if (failures == 0) std::printf("ALL obs_alignment TESTS PASSED\n");
    else std::printf("FAILED: %d inference tick(s) mismatched\n", failures);

    return failures == 0 ? 0 : 1;
}
