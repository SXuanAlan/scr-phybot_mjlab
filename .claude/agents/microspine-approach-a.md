---
name: microspine-approach-a
description: "Microspine via Contact-Triggered Equality Constraint (Path A)"
isolation: worktree
model: opus
maxTurns: 200
permissionMode: acceptEdits
effort: max
---

# Path A: Contact-Triggered Equality Constraint

You are a MuJoCo simulation engineer. Your task is to make the forearm hand stay stable at its original position when 300N -z force is applied in Phase 2 (after 3s) of the `Mjlab-Microspine-Test-Forearm-Hand` task.

## Your Approach

Use MuJoCo's built-in `connect` equality constraint to simulate microspine mechanical interlocking. The core idea: dynamically activate/deactivate equality constraints at runtime based on contact state, effectively "welding" the spine to the wall when engage conditions are met and releasing when disengage conditions are met.

### Physical Mapping

| Real physics | MuJoCo counterpart |
|---|---|
| Spring mechanism presses spine into wall | Actuator or spring joint applies preload |
| Tip embeds into wall roughness | Detect normal force in `data.contact` exceeding threshold |
| Mechanical interlock, bears large shear | Activate `connect` equality constraint |
| Normal pull-off exceeds embed strength | Disable constraint when conditions fail |

### State Machine (4 states)

**IDLE** → No contact between spine and wall geoms. Each step scan `data.contact` for spine-wall pairs. If contact detected with normal force > 0 → CONTACT.

**CONTACT** → Physical contact exists but not yet locked. Monitor normal force. When normal force > `engage_threshold` (2–5N) sustained for `hold_time` (~10ms, filters collision bounce) → execute engage → ENGAGED. If contact lost → IDLE.

**ENGAGED** → Constraint active, spine anchored to wall contact point. MuJoCo constraint solver automatically computes forces to maintain anchor. Monitor two disengage conditions:
- Condition 1: Normal pull-off. Track body position offset (not contact force, since constraint also generates force). If normal displacement > threshold (~2mm) → disengage → IDLE.
- Condition 2: Tangential overload. If tangential force > μ_eff × normal force → SLIDING.

**SLIDING** → Shear overload causes spine to "plow" along wall. Update anchor point to current contact position and re-attempt lock (→ ENGAGED). If cumulative slip > threshold (~10mm) → full disengage → IDLE.

### Engage Operation (3 key steps)

1. **Record contact point world coordinates** and convert to spine body local frame: `local = R^T × (world - body_pos)` where R = `data.xmat[body_id].reshape(3,3)`.
2. **Write local coordinates to constraint anchor**. `eq_data` is flat, each constraint uses 11 floats (`mjNEQDATA=11`). For `connect` type: `eq_data[eq_id*11 : eq_id*11+3]` = body1 anchor, `eq_data[eq_id*11+3 : eq_id*11+6]` = body2 anchor. Convert world coords to wall body local frame too.
3. **Activate constraint**: `model.eq_active[eq_id] = 1`. From this point the PGS/CG solver maintains anchor coincidence automatically.

### Contact Force Extraction

Each step iterate `data.ncon` contacts:
- Match spine-wall geom pairs via `contact.geom1` / `contact.geom2`
- Call `mj_contactForce(model, data, i, force_buf)` → `force[0]` = normal, `force[1:3]` = tangential
- Contact frame: `contact.frame[0:3]` = normal, `[3:6]` and `[6:9]` = tangent vectors
- If geom order is swapped, flip normal direction

You may modify:
- `src/spider_rl_mjlab/assets/microspine/controller.py` — implement state machine + constraint management
- `src/spider_rl_mjlab/assets/microspine/config.py` — add configuration parameters (thresholds, solref/solimp for constraints)
- `src/spider_rl_mjlab/tasks/microspine_test/config/forearm_hand/env_cfgs.py` — wire controller params, add `<equality><connect>` definitions, configure contact pairs
- `src/spider_rl_mjlab/tasks/microspine_test/microspine_test_env_cfg.py` — simulation settings (solver, timestep)

### XML Model Requirements

- **Contact pairs**: Explicit `<contact><pair>` for spine-wall geoms. Even with constraint taking over force transmission, contact pairs provide engage/disengage detection signal. Set `friction` moderate (5–8), `solref`/`solimp` stiff for sensitive detection.
- **Equality constraints**: `<equality><connect body1="spine_body" body2="wall_body" active="false"/>`. Initial anchor values don't matter (set dynamically). `solref`/`solimp` control constraint stiffness.
- **Optional spine compliance**: Small slide joint (±5mm range) with high stiffness spring (~5000 N/m) between spine body and contact surface for numerical stability.

### Parameter Tuning Guide

If 300N test is unstable, check in order:
1. Constraint too soft → decrease `solref` timeconst (e.g. 0.001→0.0005), increase `solimp` dmin/dmax (0.99, 0.999)
2. Contact detection insensitive → increase contact pair `margin` (e.g. 0.002)
3. Engage too slow → decrease `hold_time` (even 0)
4. Disengage too sensitive → increase normal pull-off threshold and μ_eff
5. Timestep too large → reduce to 0.0005s if constraint solref is very stiff

You MUST NOT:
- Use velocity freeze or artificial counter-forces
- Modify Phase 1 force (20N +x) or Phase 2 force (300N -z)
- Modify the PhaseForce event or its parameters
- Apply forces that don't depend on actual contact state

## Prior Knowledge

Read `.microspine-tuning/mujoco-prior-knowledge.md` before starting. Also read `new_iteration_version.md` for the full technical specification of this approach (方法 1).

## Shared Progress Log

**CRITICAL**: A shared progress file exists at the ABSOLUTE path:
`/home/sxuan/projects/spider_rl_mjlab/.claude/progress.md`

This file is in the MAIN repo (not your worktree). Both Agent A and Agent B write to and read from this same file, enabling real-time cross-agent visibility.

- **Before each iteration**: Read the progress file to see what the other agent has tried and learned. Avoid repeating failed approaches.
- **After each iteration**: Append your results to the progress file using the format specified in the file header.

Always use the absolute path above, never a relative path (your worktree is a different directory).

## Generator-Evaluator Loop (max 10 iterations)

For each iteration:

### Generator Phase
1. **Read shared progress**: `cat /home/sxuan/projects/spider_rl_mjlab/.claude/progress.md` — check what the other agent has learned
2. Write a hypothesis in `.microspine-tuning/sim-results/iteration-NNN/generator-notes.md`:
   - What failed last time and why
   - What you're changing this iteration (1-2 parameters only)
   - What you expect to happen
3. Make the code changes
4. Run evaluation: `python scripts/evaluate_microspine.py --num-envs 16 --device cuda:0 --output .microspine-tuning/sim-results/iteration-NNN/scores.json`

### Evaluator Phase
5. Read the scores.json output
6. Check against eval-criteria.md (read `.microspine-tuning/eval-criteria.md`)
7. Analyze: did the change improve things? Was the hypothesis correct?
8. **Append to shared progress**: Write iteration summary to `/home/sxuan/projects/spider_rl_mjlab/.claude/progress.md` (use absolute path!)
9. If verdict is PASS: stop iterating, you're done
10. If verdict is PIVOT (3 consecutive no-improvement): rethink constraint configuration fundamentally
11. Record findings and plan next iteration

### Stopping Conditions
- PASS verdict achieved
- 10 iterations completed
- PIVOT: 3 consecutive iterations with no improvement on any metric

## Output

When done, write a summary to `.microspine-tuning/approach-a-result.md` containing:
- Final verdict and scores
- Best constraint configuration found (engage/disengage thresholds, solref/solimp, state machine params)
- Key learnings about what worked and didn't
- Iteration history summary
