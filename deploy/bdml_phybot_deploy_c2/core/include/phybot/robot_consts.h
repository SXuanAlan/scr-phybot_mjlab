#pragma once
namespace phybot {
constexpr int kNumDofs        = 21;
constexpr int kObsDim         = 283;   // 73 proprio + 5*21 pos-hist + 5*21 vel-hist
constexpr int kProprio        = 73;
constexpr int kHistLen        = 20;    // full history depth
constexpr int kHistFramesInObs = 5;    // newest 5 frames enter the obs
constexpr int kDecimation     = 10;
}  // namespace phybot
