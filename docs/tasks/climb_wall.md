# Wall climbing (`Mjlab-Climb-Wall-Phybot-C2*`)

PHYBot C2 climbs with four microspine pads (`left/right_hand_spine`,
`left/right_foot_spine`) by imitating a CAD reference cycle. The pads use the
empirical unit model: ordinary contacts carry compression, equality anchors
carry grip, and per-unit load limits decide release. Every active unit must
reach at least 10 N of estimated compression before a pad engages.

The configuration lives in `src/scr_phybot_mjab/tasks/climb_wall/config/phybot_c2/`
(one module per section, composed in `env_cfgs.py`); the per-step terms are in
`climb_wall/mdp/`, and the pad plant in `src/scr_phybot_mjab/microspine/`.

## Variants

| Task | Active units per pad | Training wall angles | Evaluation |
| --- | ---: | --- | --- |
| `Mjlab-Climb-Wall-Phybot-C2` | 12 (3×4) | 45° → 75±12° → 80±5° → 85±3° → 88±2° → 90° | exact 90° |
| `Mjlab-Climb-Wall-Phybot-C2-Corners-A60` | 4 corners | 45±5° → 55±5° → 60° | exact 60° |
| `Mjlab-Climb-Wall-Phybot-C2-Corners-A80` | 4 corners | fixed 80° | exact 80° |

The main ladder changes stage at iterations 0, 1500, 3300, 3900, 4550 and 5150
(`mdp.WALL_ANGLE_STAGES`); the 60-degree ladder at 0, 1500 and 3300
(`mdp.CORNERS60_WALL_ANGLE_STAGES`), and its final stage has no jitter, so
training never samples above 60°. A resetting environment takes the current
stage; running episodes keep their wall.

Gravity always points down: the curriculum tilts the wall about its base
line. Shallow angles are a ramp the robot lies on; 90° is a vertical wall.
`Mjlab-Climb-Transition80-Phybot-C2` mounts onto the same wall at 80° and
hands off to this cycle.

## Reference and resets

The main task imitates `assets/motions/climb_cad_empirical.npy`, the corner
tasks their `_corners_a60` / `_corners_a80` variants; the recipe next to each
clip regenerates it. A cycle lasts 9 s and climbs 20 mm, and the velocity
command follows that average along-wall rate. Training resets sample the
whole cycle with a failure-adaptive histogram (one-second bins, EMA 0.001,
additive uniform floor 0.1) and restore the reference pose, velocity, passive
gimbals and the scheduled initial grips; play starts at frame 0.

## Rewards

Sixteen terms: BeyondMimic body position, orientation and linear/angular
velocity kernels and root orientation; reference height, anchor landing and
swing approach; the engage-schedule mismatch; support loss, unsafe release
and release load quality for the pads; action rate, joint limits and
undesired contacts; and the termination penalty. The schedule-mismatch gate
scales the body position/orientation and root orientation kernels.

## Terminations

- `time_out`
- `base_height`: the base fell below 0.3 m of along-wall altitude
- `bad_state`: NaN or excessive velocity
- `bad_tracking`: a tracked body's aligned position error above 0.3 m, a root
  orientation error above 1.0 rad, or an along-wall height error above 0.25 m
- `lost_support`: no pad latched for 0.4 s

## Evaluation

- `scripts/eval/evaluate_climb_checkpoint.py`: the Corners-A60 exam (metrics,
  trajectory, engage raster and video per angle).
- `scripts/train/train_climb60_monitored.py`, `supervise_climb60.py` and
  `monitor_climb60_metrics.py`: a Corners-A60 run with a live curriculum gate.
- `scripts/eval/plot_pad_wrench_timeseries.py`: wall-to-pad wrenches over a rollout.

Design reviews and evaluation reports from September are in
`docs/reports/empirical_climb/`.
