# Retired code

Code removed during the September–October 2026 cleanup (branch
`cleanup/2026-09`) and how to get it back. Nothing below is registered or
imported any more.

## Getting it back

The tag `archive/pre-cleanup-2026-09` (c0a07ee) is the working tree as it was
before the cleanup; everything retired in the first column of commits below
is there. Code removed later is in the parent of the commit that removed it.

```bash
git show archive/pre-cleanup-2026-09:<path>           # read a retired file
git checkout archive/pre-cleanup-2026-09 -- <path>    # restore it into the tree
git show <commit>^:<path>                             # a file removed by <commit>
```

Other archive tags (local, not pushed): `archive/climb-contact-command`,
`archive/climb-osc-wbc-base` and `archive/feat-microspine-virtual-anchor` are
the tips of those branches; `archive/stash-moment-envelope-v1` is a stash
from main; `archive/microspine-constraint-bench` is the last commit with the
constraint benches.

## What was retired

| Removed | Why | Commit |
| --- | --- | --- |
| Unused assets, ROS export leftovers, duplicate meshes | not loaded by any task | 6b1d8f0 |
| One-off probe and broken scripts | probes of finished experiments | 5ec8081 |
| `toddler_climb` (ToddlerBot single-unit experiment) | experiment concluded | d16c717 |
| `climb_transition`: 90° mount, band/from0/unified variants, offline OSC teacher | replaced by `climb_transition80` | a3de444 |
| v4 deploy/distillation stack: `rl/`, climb `-Distill`/`-Polish`, student observations, estimator and polish launchers | the deployed policy line moved on | 893447f |
| Rizon4 bench, limit-surface teleop, Wang-plant limit-surface tools | the bench lives in `~/projects/scr-rizon4-unit_mjlab` | 4b13c30 |
| climb_wall's Wang contact path, legacy clips, `PHYBOT_C2_PAD_GIMBAL` | the empirical model replaced Wang; the gimbal is always on | 85a0fab |
| Wedge, Wang and spring microspine backends (`wang_limit_surface` moved to `tools/microspine/`) | only the empirical backend is used | 90b2254 |
| Tilt-peel release | not part of the empirical plant | 1bf7436 |
| Microspine config fields of the old models (42 fields) and switches every task leaves at one value | no task sets them | 47cc562, 55f0f0c |
| Climb's unscheduled-release drill, reseat hold and frozen clock | unused by the registered tasks | 623da5b |
| Release codes 1 and 3–6 and their costs; climb joint-position noise | the empirical judge never writes them | c4fb27d |
| transition80 switches (`constraint_mode`, `relative_frame`, `asym_arm_tracking`, …) | never flipped | 5866803 |
| `gen_mount_clip` hands-first, foot-first and interleaved mounts and their options | the a80 clip uses the cross mount | 575975c, 0390021 |
| Walk's rough terrain, the unregistered AMP walk config, `angular_momentum_penalty` | no task uses them | e58b0a5 |
| Microspine constraint benches (`validate_microspine_constraint`) | their premise does not hold on the empirical plant | 11aaf45 |
