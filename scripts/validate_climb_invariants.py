"""Production-env sequential test: random actions, per-step physics invariants.

  I1 latched pad -> zero sensed press force (retraction + sensor mask)
  I2 forced release -> margin < 0 on the release step (and the previous
     step too when violation_consecutive_steps >= 2)
  I3 latched pads' spheres retracted / released pads' spheres restored
  I4 margins and anchor loads finite
Plus event counts to prove the run actually cycled latches.
"""
import torch, warp as wp
from mjlab.envs import ManagerBasedRlEnv
from scr_phybot_mjab.tasks.climb_wall.config.phybot_c2.env_cfgs import (
  phybot_c2_climb_wall_env_cfg,
)

cfg = phybot_c2_climb_wall_env_cfg(play=False)
cfg.scene.num_envs = 4
env = ManagerBasedRlEnv(cfg, device="cuda:0")
env.reset()
aff = env.command_manager.get_term("microspine_affordance")
gids = aff._sphere_geom_ids
S = 4
torch.manual_seed(0)

viol = {k: 0 for k in ("I1", "I2", "I3", "I4")}
engages = releases = 0
prev_margin = aff.hold_margin.clone()
prev_engaged = aff.engaged.clone()
for step in range(1200):
  act = 0.4 * torch.randn(env.num_envs, 21, device=env.device)
  env.step(act)
  eng = aff.engaged
  press = aff.pad_press_force
  margin = aff.hold_margin
  frel = aff.forced_release_this_step
  engages += int(aff.engaged_this_step.sum())
  releases += int(frel.sum())

  steady = eng & ~aff.engaged_this_step
  if bool((press[steady] > 1e-4).any()):
    viol["I1"] += 1
  # Release contract: the released EE was engaged and violating on this and
  # the previous step. hold_margin on the release step holds the violating
  # value; the previous step's stored margin must already be negative.
  if bool(frel.any()):
    m_now = margin[frel]
    bad = bool((m_now >= 0).any())
    if aff._ls_cfg.violation_consecutive_steps >= 2:
      m_prev = prev_margin[frel]
      was_eng = prev_engaged[frel]
      bad |= bool(((m_prev >= 0) & was_eng).any())
    if bad:
      viol["I2"] += 1
  if step % 25 == 0:
    pos = wp.to_torch(env.sim.wp_model.geom_pos)
    off = (pos[:, gids] - aff._sphere_pos_default.unsqueeze(0)).norm(dim=-1)
    off = off.view(env.num_envs, 4, S)
    retracted = off > 0.04
    restored = off < 1e-5
    bad = (eng.unsqueeze(-1) & ~retracted) | (~eng.unsqueeze(-1) & ~restored)
    if bool(bad.any()):
      viol["I3"] += 1
  if not bool(torch.isfinite(margin).all() & torch.isfinite(aff.anchor_load_tnl).all()):
    viol["I4"] += 1
  prev_margin = margin.clone()
  prev_engaged = eng.clone()

print(f"1200 steps x 4 envs | engage events={engages} release events={releases}")
for k, v in viol.items():
  print(f"  {k} violations: {v}")
ok = all(v == 0 for v in viol.values()) and engages > 20 and releases > 20
print("CLIMB INVARIANT SWEEP:", "PASS" if ok else "FAIL")
env.close()
