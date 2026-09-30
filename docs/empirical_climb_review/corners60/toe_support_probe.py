"""Bounded pose diagnostic: both ordinary toe contacts during pad changes."""
import importlib.util
import json
from pathlib import Path

import mjlab
import scr_phybot_mjab.tasks
import mujoco
import numpy as np
from scipy.optimize import least_squares

from scr_phybot_mjab.tools.climbing_reference.kinematics import ClimbIK

root = Path(__file__).resolve().parents[3]
module = importlib.util.spec_from_file_location("cad_audit", root / "scripts/audit_cad_climb_reference.py")
audit = importlib.util.module_from_spec(module)
module.loader.exec_module(audit)
clip = np.load(audit.MOTION, allow_pickle=True).item()
clip["unit_layout"] = "corners"
ik = ClimbIK()
model, sites, bodies, toes, _ = audit.build_reference_model(tuple(clip["ee_list"]))
data = mujoco.MjData(model)
surface = audit.EmpiricalLimitSurface()
results = []
preload_frames = (102,103,104,207,208,209,327,328,329,447,448,449)
frames = sorted(set([0,30,50,135,150,250,300,370,400]) | set(preload_frames))
for frame in frames:
  canonical = {**clip, "wall_angle":90.0}
  audit.set_frame(ik.model,ik.data,canonical,frame)
  q0 = ik.data.qpos.copy()
  joint_names = [n for n in clip["joints_list"] if "_ankle_" in n]
  adr = np.array([ik.qpos_addresses[n] for n in joint_names])
  x0 = q0[adr].copy()
  targets = {ee:ik.data.site_xpos[sid].copy() for ee,sid in ik.toe_sid.items()}
  for ee in targets:
    targets[ee][0] = 1.5-ik.toes[ee].radius
  def residual(x):
    ik.data.qpos[:] = q0
    ik.data.qpos[adr] = x
    mujoco.mj_forward(ik.model,ik.data)
    errors = [(ik.data.site_xpos[ik.toe_sid[ee]]-target)*[10000.,.1,.1]
              for ee,target in targets.items()]
    return np.r_[np.concatenate(errors), .01*(x-x0)]
  sol = least_squares(residual, x0, bounds=(
    [ik.joint_ranges[n][0]+.005 for n in joint_names],
    [ik.joint_ranges[n][1]-.005 for n in joint_names]), max_nfev=300,
    ftol=1e-11,xtol=1e-11,gtol=1e-9)
  residual(sol.x)
  gap = max(abs(ik.data.site_xpos[ik.toe_sid[ee],0]-target[0]) for ee,target in targets.items())
  candidate = {**clip, "wall_angle":60.0, "joint_positions":clip["joint_positions"].copy(),
               "toe_braced":clip["toe_braced"].copy()}
  for name,value in zip(joint_names,sol.x):
    candidate["joint_positions"][frame,list(clip["joints_list"]).index(name)] = value
  candidate["toe_braced"][frame] = True
  audit.set_frame(model,data,candidate,frame)
  row,_ = audit.static_frame(model,data,candidate,frame,sites,bodies,toes,surface)
  row.update(max_toe_gap_mm=gap*1000., variant="both_toes_braced", ankle_solution=sol.x.tolist())
  if frame in preload_frames:
    ee = clip["ee_list"][int(np.flatnonzero(~clip["ee_engaged"][frame].astype(bool))[0])]
    press,_ = audit.static_frame(model,data,candidate,frame,sites,bodies,toes,surface,preload_names=(ee,))
    row["preload"] = press
  results.append(row)
  print(frame, 'gap mm', gap*1000., 'hold',row.get('characteristic_feasible',False),
        'linear',row['linear_feasible'], 'preload',row.get('preload',{}).get('characteristic_feasible'), flush=True)
Path(__file__).with_suffix('.json').write_text(json.dumps(results,indent=2)+'\n')
