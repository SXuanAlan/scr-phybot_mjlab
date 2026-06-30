#ifndef GOLDEN_DIR
#error "GOLDEN_DIR must be defined"
#endif

#include "phybot/data_package.h"
#include "phybot/robot_consts.h"
#include "walk/walk_task.h"

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
    if (!f.is_open()) { std::fprintf(stderr,"Cannot open: %s\n",path.c_str()); std::exit(1); }
    std::vector<std::vector<double>> rows;
    std::string line;
    while (std::getline(f, line)) {
        if (line.empty() || line[0]=='#') continue;
        std::vector<double> row;
        std::stringstream ss(line);
        std::string tok;
        while (std::getline(ss,tok,',')) row.push_back(std::stod(tok));
        rows.push_back(std::move(row));
    }
    return rows;
}

int main() {
    const std::string gdir = GOLDEN_DIR;
    auto inp = parse_csv(gdir + "/golden_inputs.csv");
    auto out = parse_csv(gdir + "/golden_official.csv");

    if (inp.size() != 60 || out.size() != 60) {
        std::fprintf(stderr,"Expected 60 rows; got %zu/%zu\n",inp.size(),out.size());
        return 1;
    }

    static const double kTol = 1e-4;
    const std::string model_path = gdir + "/cpg_policy.pt";

    phybot::WalkTask task(model_path);
    phybot::DataPackage io;
    task.enter(io);

    double worst_mpd=0, worst_simp=0, worst_simd=0;
    int failures=0;

    for (int frame=0; frame<60; frame++) {
        const auto& ir = inp[frame];
        const auto& or_ = out[frame];

        // load sensors into io
        for (int i=0;i<3;i++) io.imu_zyx[i]         = ir[1+i];
        for (int i=0;i<3;i++) io.imu_angular_vel[i]  = ir[4+i];
        for (int i=0;i<phybot::kNumDofs;i++) io.motor_pos[i] = ir[7+i];
        for (int i=0;i<phybot::kNumDofs;i++) io.motor_vel[i] = ir[28+i];
        io.js_vx_desire      = ir[49];
        io.js_vy_desire      = ir[50];
        io.js_OmegaZ_desire  = ir[51];

        task.run(io);

        // parse golden output
        // cols: tick(0), did_infer(1), obs[283](2..284), mpd[21](285..305), simP[21](306..326), simD[21](327..347)
        int tick     = static_cast<int>(or_[0]);
        int did_inf  = static_cast<int>(or_[1]);

        double md_mpd=0, md_simp=0, md_simd=0;
        for (int i=0;i<phybot::kNumDofs;i++) {
            double d = std::fabs(io.motor_Pos_desire[i] - or_[285+i]);
            if (d>md_mpd) md_mpd=d;
        }
        for (int i=0;i<phybot::kNumDofs;i++) {
            double d = std::fabs(io.sim_P[i] - or_[306+i]);
            if (d>md_simp) md_simp=d;
        }
        for (int i=0;i<phybot::kNumDofs;i++) {
            double d = std::fabs(io.sim_D[i] - or_[327+i]);
            if (d>md_simd) md_simd=d;
        }

        if (md_mpd > worst_mpd) worst_mpd = md_mpd;
        if (md_simp> worst_simp) worst_simp= md_simp;
        if (md_simd> worst_simd) worst_simd= md_simd;

        bool ok = (md_mpd<=kTol && md_simp<=kTol && md_simd<=kTol);
        std::printf("tick %2d did_infer=%d mpd_diff=%.3e simP_diff=%.3e simD_diff=%.3e %s\n",
                    tick, did_inf, md_mpd, md_simp, md_simd, ok?"PASS":"FAIL");
        if (!ok) failures++;
    }

    std::printf("\n--- Summary (tol=%.0e) ---\n", kTol);
    std::printf("  worst motor_Pos_desire diff: %.3e\n", worst_mpd);
    std::printf("  worst sim_P diff:            %.3e\n", worst_simp);
    std::printf("  worst sim_D diff:            %.3e\n", worst_simd);
    if (failures==0) std::printf("ALL walk_e2e TESTS PASSED\n");
    else             std::printf("FAILED: %d tick(s) mismatched\n", failures);
    return failures==0 ? 0 : 1;
}
