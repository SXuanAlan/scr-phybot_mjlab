"""Offline geometry and empirical-load audit. No environment steps or training.

Static loading is a necessary-condition relaxation: engaged passive gimbals
are balanced, free swinging gimbals/trajectory accelerations are not. Exported
loads are reference-derived optimization results, never policy or measured
loads. Failure here is useful evidence; it must not be relabelled a pass.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import mjlab
import mujoco
import numpy as np
from scipy.optimize import linprog, minimize

from scr_phybot_mjab.assets.motions import CAD_CLIMB_MOTION as MOTION
from scr_phybot_mjab.tools.climbing_reference.model import build_reference_model, engage_basis, joint_group, WallGeometry
from scr_phybot_mjab.assets.phybot_c2.phybot_c2_constants import PHYBOT_C2_DEPLOY_TORQUE_LIMITS
from scr_phybot_mjab.microspine.empirical_limit_surface import EmpiricalLimitSurface
from scr_phybot_mjab.microspine.pad_geometry import matching_pad_geometry
from scr_phybot_mjab.microspine.wrench import unit_wrench_matrix
from scr_phybot_mjab.tools.paths import CLIMB_REVIEW_DIR

OUT = CLIMB_REVIEW_DIR


def set_frame(model, data, clip, frame):
  data.qpos[:] = 0.; data.qvel[:] = 0.
  data.qpos[:3] = clip["root_position"][frame]
  data.qpos[3:7] = clip["root_quaternion"][frame, [3, 0, 1, 2]]
  for key_names, key_values in (("joints_list", "joint_positions"),
                                ("passive_joint_names", "passive_joint_positions")):
    for i, name in enumerate(clip.get(key_names, ())):
      data.qpos[model.joint(name).qposadr[0]] = clip[key_values][frame, i]
  wall = WallGeometry(float(clip.get("wall_angle", 90.0)), 1.5)
  if not wall.identity:
    data.qpos[:3] = wall.rotate_point(data.qpos[:3])
    q = data.qpos[3:7].copy()
    mujoco.mju_mulQuat(data.qpos[3:7], wall.quat, q)
  mujoco.mj_forward(model, data)


def static_frame(model, data, clip, frame, pad_sites, bodies, toes, surface, quantile=.5,
                 preload_names=(), unit_preload_n=10.0, preload_friction=.2, toe_friction=.2):
  wall = WallGeometry(float(clip.get("wall_angle", 90.0)), 1.5)
  names = [ee for ee, on in zip(clip["ee_list"], clip["ee_engaged"][frame])
           if on or ee in preload_names]
  comp = [(ee, sid) for i, (ee, (sid, radius)) in enumerate(toes.items())
          if clip["toe_braced"][frame, i] and abs(wall.distance(data.site_xpos[sid])-radius) <= .0005]
  n, dim = len(names), len(names)*6 + len(comp)*3
  jac = np.zeros((model.nv, dim))
  bases = []
  unactuated = list(range(6))
  active_stops = []
  for i, ee in enumerate(names):
    basis = np.stack(engage_basis(data, bodies[ee], wall.outward), axis=-1)
    bases.append(basis)
    jp, jr = np.zeros((3, model.nv)), np.zeros((3, model.nv))
    mujoco.mj_jacSite(model, data, jp, jr, pad_sites[ee])
    jac[:, 6*i:6*i+3] = jp.T@basis
    jac[:, 6*i+3:6*i+6] = jr.T@basis
    for axis in ("x", "y", "z"):
      joint = model.joint(f"{ee}_gimbal_{axis}")
      value = data.qpos[joint.qposadr[0]]
      dof = int(joint.dofadr[0])
      if abs(value-joint.range[0]) < 1e-6:
        active_stops.append((joint.name, dof, 1.))  # lower-stop reaction >=0
      elif abs(value-joint.range[1]) < 1e-6:
        active_stops.append((joint.name, dof, -1.))  # upper-stop reaction <=0
      else:
        unactuated.append(dof)
  for i, (_, sid) in enumerate(comp):
    jp, jr = np.zeros((3, model.nv)), np.zeros((3, model.nv))
    mujoco.mj_jacSite(model, data, jp, jr, sid)
    # Toe load coordinates rotate with the wall; gravity remains world-down.
    jac[:, 6*n+3*i:6*n+3*i+3] = jp.T @ wall.R
  bias = data.qfrc_bias - data.qfrc_passive
  eq, rhs = jac[unactuated], -bias[unactuated]
  dofs = [int(model.joint(name).dofadr[0]) for name in clip["joints_list"]]
  torque_max = np.array([PHYBOT_C2_DEPLOY_TORQUE_LIMITS[joint_group(name)] for name in clip["joints_list"]])
  torque_map, torque_bias = jac[dofs], bias[dofs]
  # Linear measured-domain restrictions on EVERY unit, plus torque bounds.
  geometry = matching_pad_geometry(model, names, unit_layout=clip.get("unit_layout", "all"))
  um = unit_wrench_matrix(geometry)
  n_units = geometry.count
  force_map = np.zeros((n*n_units, 3, dim))
  for i in range(n):
    force_map[n_units*i:n_units*(i+1), :, 6*i:6*i+6] = um
  preload_units = np.repeat([ee in preload_names for ee in names], n_units)
  A, b = [], []
  # Normal pull-off capacity is an upper bound, evaluated below. The seating
  # preload is not a lower normal-force bound on already engaged units.
  for axis, lower, upper in ((0, surface.s_min, surface.s_max), (1, -surface.l_max, surface.l_max)):
    a = force_map[~preload_units, axis]
    A.extend(-a); b.extend(np.full(len(a), -lower))
    if upper is not None:
      A.extend(a); b.extend(np.full(len(a), upper))
  if preload_units.any():
    # Before OFF->ON the approaching pad is an ordinary unilateral
    # contact. Every unit must press >=10 N; no adhesive capacity is
    # credited to it. An outer Coulomb square gives the LP a
    # necessary relaxation; SLSQP checks the true circular cone below.
    # The 10 N seating threshold is a minimum, with no preload-derived ceiling.
    A.extend(force_map[preload_units, 2])
    b.extend(np.full(int(preload_units.sum()), -unit_preload_n))
    for axis in (0, 1):
      for sign in (-1, 1):
        A.extend(sign*force_map[preload_units, axis] + preload_friction*force_map[preload_units, 2])
        b.extend(np.zeros(int(preload_units.sum())))
    # A necessary upper bound: no holding unit can exceed the largest
    # capacity anywhere in the measured domain at this quantile. If even
    # this optimistic linear relaxation fails, a different SLSQP seed
    # cannot make this reference pose pass the characteristic model.
    shear, lateral = np.meshgrid(
      np.linspace(surface.s_min, surface.s_max, surface.ns),
      np.linspace(0.0, surface.l_max, surface.nl), indexing="ij")
    ceiling, _, valid, _ = surface.query_numpy(shear, lateral, quantile)
    capacity_ceiling = float(np.max(ceiling[valid]))
    A.extend(force_map[~preload_units, 2])
    b.extend(np.full(int((~preload_units).sum()), capacity_ceiling))
  A.extend(torque_map); b.extend(torque_max-torque_bias)
  A.extend(-torque_map); b.extend(torque_max+torque_bias)
  for _, dof, sign in active_stops:
    A.append(-sign*jac[dof]); b.append(sign*bias[dof])
  for i in range(len(comp)):
    j = 6*n+3*i
    a = np.zeros(dim); a[j] = -1; A.append(a); b.append(0.)
    a = np.zeros(dim); a[j] = 1; A.append(a); b.append(240.)
    # Outer square is a necessary LP relaxation of the actual toe cone.
    # The nonlinear solve checks its circular cross-section below.
    for axis in (1, 2):
      for sign in (-1, 1):
        a = np.zeros(dim); a[j] = -toe_friction; a[j+axis] = sign
        A.append(a); b.append(0.)
  A, b = np.asarray(A), np.asarray(b)
  # Minimize a common per-unit tensile bound for a feasible deterministic seed.
  Al = np.c_[A, np.zeros(len(A))]
  Al = np.r_[Al, np.c_[force_map[:, 2], -np.ones(n*n_units)]]
  bl = np.r_[b, np.zeros(n*n_units)]
  objective = np.r_[np.zeros(dim), 1.]
  lp = linprog(objective, A_ub=Al, b_ub=bl, A_eq=np.c_[eq, np.zeros(len(eq))],
               b_eq=rhs, bounds=[(None, None)]*(dim+1), method="highs")
  if not lp.success:
    # Diagnose a failed necessary condition without silently increasing the
    # model's capacity. This relaxed allocation is audit evidence only.
    scale = np.maximum(np.abs(b), 1.)
    relaxation = linprog(np.r_[np.zeros(dim), 1.],
      A_ub=np.c_[A, -scale], b_ub=b,
      A_eq=np.c_[eq, np.zeros(len(eq))], b_eq=rhs,
      bounds=[(None,None)]*dim+[(0,None)], method="highs")
    diagnosis = {}
    if relaxation.success:
      x = relaxation.x[:dim]
      force = np.einsum("uad,d->ua", force_map, x)
      diagnosis = dict(min_relative_bound_relaxation=float(relaxation.x[-1]),
        relaxed_unit_shear_range_n=[float(force[:,0].min()),float(force[:,0].max())],
        relaxed_unit_lateral_range_n=[float(force[:,1].min()),float(force[:,1].max())],
        relaxed_unit_normal_range_n=[float(force[:,2].min()),float(force[:,2].max())],
        relaxed_max_torque_ratio=float(np.max(np.abs(torque_bias+torque_map@x)/torque_max)))
    return dict(frame=frame, linear_feasible=False, solver_message=lp.message,
                unit_quantile=quantile, evaluated_contacts=names,
                preload_contacts=list(preload_names),
                **({"holding_unit_normal_ceiling_n": capacity_ceiling,
                    "required_unit_preload_n": unit_preload_n}
                   if preload_names else {}), **diagnosis), []

  def capacities(x):
    force = np.einsum("uad,d->ua", force_map, x)
    cap, _, valid, _ = surface.query_numpy(force[:, 0], force[:, 1], quantile)
    cap[preload_units] = 0.0  # unilateral seating contact, no tensile strength
    return force, cap

  f, cap = capacities(lp.x[:dim])
  z0 = np.r_[lp.x[:dim], np.max(f[:, 2]-cap)+.001]

  def envelope(z):
    f, cap = capacities(z[:dim])
    return cap - f[:, 2] + z[-1]

  def preload_cone(z):
    f = np.einsum("uad,d->ua", force_map[preload_units], z[:dim])
    return -preload_friction*f[:, 2] - np.linalg.norm(f[:, :2], axis=1)

  def toe_cone(z):
    f = z[6*n:dim].reshape(len(comp),3)
    return toe_friction*f[:,0] - np.linalg.norm(f[:,1:],axis=1)

  constraints = [dict(type="eq", fun=lambda z:eq@z[:dim]-rhs,
                      jac=lambda z:np.c_[eq, np.zeros(len(eq))]),
                 dict(type="ineq", fun=lambda z:b-A@z[:dim],
                      jac=lambda z:np.c_[-A, np.zeros(len(A))]),
                 dict(type="ineq", fun=envelope)]
  if preload_units.any():
    constraints.append(dict(type="ineq", fun=preload_cone))
  if comp:
    constraints.append(dict(type="ineq", fun=toe_cone))
  sol = minimize(lambda z:z[-1]+1e-8*np.dot(z[:dim], z[:dim]), z0,
    method="SLSQP", constraints=constraints,
    options=dict(maxiter=160, ftol=1e-8))
  x = sol.x[:dim]
  f, cap = capacities(x)
  excess = float(np.max(f[:, 2]-cap))
  eqerr = float(np.max(np.abs(eq@x-rhs)))
  domainerr = float(np.maximum(A@x-b, 0).max())
  if preload_units.any():
    domainerr = max(domainerr, float(np.maximum(-preload_cone(sol.x), 0).max()))
  if comp:
    domainerr = max(domainerr, float(np.maximum(-toe_cone(sol.x), 0).max()))
  # Draw all units independently for THIS fixed static loading solution.
  # This is a model estimate, not calibrated full-pad survival probability.
  rng = np.random.default_rng(1409+frame)
  q, p, support, tail = surface.query_numpy(f[None, :, 0], f[None, :, 1], rng.random((2048, len(f))))
  valid = support & np.isfinite(f[None, :, 2])
  holds = valid & (f[None, :, 2] <= q) & (rng.random(q.shape) >= p)
  records = []
  for i, ee in enumerate(names):
    w = x[6*i:6*i+6]
    record = dict(frame=frame, time_s=frame/float(clip["fps"]), ee=ee,
      phase=str(clip["segment_names"][int(clip["phase_segment"][frame])]),
      Ft_C_n=w[0], Fl_C_n=w[1], Fn_C_n=w[2], Mt_C_nm=w[3], Ml_C_nm=w[4], Mn_C_nm=w[5],
      load_source="reference_static_optimization", unit_quantile=quantile)
    records.append(record)
  report = dict(frame=frame, unit_quantile=quantile, linear_feasible=True,
    solver_success=bool(sol.success), solver_message=sol.message,
    max_unit_normal_excess_n=excess, equality_error=eqerr, linear_inequality_error=domainerr,
    characteristic_feasible=bool(excess <= 1e-3 and eqerr < 1e-5 and domainerr < 1e-5),
    model_iid_all_contacts_hold_fraction=float(holds[:,~preload_units].all(axis=1).mean()),
    model_iid_pad_hold_fractions={ee:float(holds[:,n_units*i:n_units*(i+1)].all(axis=1).mean()) for i,ee in enumerate(names) if ee not in preload_names},
    max_actuator_torque_ratio=float(np.max(np.abs(torque_bias+torque_map@x)/torque_max)),
    maximum_unit_compression_n=float(np.maximum(-f[:, 2], 0).max()),
    evaluated_contacts=names, toe_contacts=[ee for ee,_ in comp],
    toe_load_wall_xyz_n={ee:x[6*n+3*i:6*n+3*i+3].tolist() for i,(ee,_) in enumerate(comp)},
    toe_friction=toe_friction,
    any_censored_tail=bool(tail.any()))
  if preload_names:
    report["preload_contacts"] = list(preload_names)
    report["minimum_unit_preload_n"] = {
      ee:float((-f[n_units*i:n_units*(i+1), 2]).min())
      for i,ee in enumerate(names) if ee in preload_names}
  report["stop_reaction_nm"] = {name:float(bias[dof]+jac[dof]@x) for name,dof,_ in active_stops}
  return report, records


def audit_geometry(model, data, clip, sites, toes):
  wall = WallGeometry(float(clip.get("wall_angle", 90.0)), 1.5)
  gap, drift, penetration, joint_margin, gimbal_margin, speeds = [], [], [], [], [], []
  toe_gaps = []
  active_pad_boxes = [g for g in range(model.ngeom)
    if "_c_" in (model.geom(g).name or "") and model.geom_type[g] == mujoco.mjtGeom.mjGEOM_BOX]
  records = []
  for frame in range(len(clip["joint_positions"])):
    set_frame(model, data, clip, frame)
    for i, ee in enumerate(clip["ee_list"]):
      c = data.site_xpos[sites[ee]]
      if clip["ee_engaged"][frame, i]:
        gap.append(abs(wall.distance(c)))
        drift.append(np.linalg.norm(c-wall.rotate_point(clip["ee_anchor_positions"][frame, i])))
    for g in active_pad_boxes:
      extent = np.abs(wall.outward @ data.geom_xmat[g].reshape(3,3))@model.geom_size[g]
      penetration.append(extent-wall.distance(data.geom_xpos[g]))
    for key in ("joints_list", "passive_joint_names"):
      for name in clip.get(key, ()):
        j = model.joint(name); v = data.qpos[j.qposadr[0]]
        clearance = min(v-j.range[0], j.range[1]-v)
        if "gimbal" in name:
          gimbal_margin.append(clearance)
        else:
          joint_margin.append(clearance/(j.range[1]-j.range[0]))
    for i, (ee,(sid,radius)) in enumerate(toes.items()):
      if clip["toe_braced"][frame,i]:
        toe_gaps.append(abs(wall.distance(data.site_xpos[sid])-radius))
  fps=float(clip["fps"])
  cycle=int(np.flatnonzero(clip["phase_segment"]==list(clip["segment_names"]).index("end_hold"))[0])
  speed=np.abs(np.diff(clip["joint_positions"],axis=0))*fps
  pspeed=np.abs(np.diff(clip["passive_joint_positions"],axis=0))*fps
  return dict(frames=len(clip["joint_positions"]), cycle_frames=cycle,
    cycle_duration_s=cycle/fps, file_duration_s=len(clip["joint_positions"])/fps,
    step_height_m=float(clip["step_height"]), min_engaged=int(np.min(np.sum(clip["ee_engaged"],axis=-1))),
    max_engaged_center_wall_gap_mm=float(max(gap)*1000),
    max_engaged_anchor_drift_mm=float(max(drift)*1000),
    max_unit_box_wall_penetration_mm=float(max(penetration)*1000),
    minimum_actuated_joint_range_fraction=float(min(joint_margin)),
    minimum_gimbal_stop_clearance_deg=float(np.degrees(min(gimbal_margin))),
    max_joint_speed_rad_s=float(speed.max()), max_passive_speed_rad_s=float(pspeed.max()),
    max_braced_toe_gap_mm=float(max(toe_gaps, default=0.)*1000),
    loop_joint_closure_rad=float(np.abs(clip["joint_positions"][cycle]-clip["joint_positions"][0]).max()),
    loop_passive_closure_rad=float(np.abs(clip["passive_joint_positions"][cycle]-clip["passive_joint_positions"][0]).max()),
    geometry_pass=bool(max(gap)<.00015 and max(drift)<.00015 and max(penetration)<.00015
                       and min(joint_margin)>0 and min(gimbal_margin)>-2e-7),
    gimbal_stop_support=str(clip.get("gimbal_stop_support","none")))


def audit_engage_preload(model, data, clip, sites, bodies, toes, surface,
                        unit_preload_n=10.0, debounce_steps=3,
                        quantile=.5, preload_friction=.2, toe_friction=.2):
  """Check the three pre-latch frames using unilateral pressing contacts.

  The other three pads use the empirical characteristic hold model.
  This proves neither actual force tracking nor stochastic reliability.
  """
  checks = []
  wall = WallGeometry(float(clip.get("wall_angle", 90.0)), 1.5)
  edges = np.argwhere(np.diff(clip["ee_engaged"].astype(int), axis=0) > 0)
  for previous, index in edges:
    edge = int(previous + 1)
    ee = clip["ee_list"][index]
    for frame in range(max(0, edge-debounce_steps), edge):
      set_frame(model, data, clip, frame)
      gap = abs(wall.distance(data.site_xpos[sites[ee]]))
      off = not bool(clip["ee_engaged"][frame, index])
      row, _ = static_frame(model, data, clip, frame, sites, bodies, toes, surface,
                            quantile=quantile, preload_names=(ee,),
                            unit_preload_n=unit_preload_n, preload_friction=preload_friction,
                            toe_friction=toe_friction)
      row.update(engage_frame=edge, approaching_pad=ee, center_wall_gap_mm=gap*1000,
                 preload_contact_geometry_ok=bool(off and gap <= .0005))
      row["preload_characteristic_feasible"] = bool(
        row["preload_contact_geometry_ok"] and row.get("characteristic_feasible", False))
      checks.append(row)
  return dict(required_unit_preload_n=unit_preload_n, debounce_steps=debounce_steps,
    characteristic_unit_quantile=quantile, preload_friction=preload_friction,
    toe_friction=toe_friction,
    engage_edges=len(edges), frames=checks,
    all_feasible=bool(len(checks) == len(edges)*debounce_steps and
                      all(row["preload_characteristic_feasible"] for row in checks)),
    scope="Quasistatic characteristic-model feasibility before latching; not dynamic or hardware validation")


def main():
  parser=argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--motion",type=Path,default=MOTION)
  parser.add_argument("--stride",type=int,default=20)
  parser.add_argument("--initial-only",action="store_true")
  parser.add_argument("--output-dir",type=Path,default=OUT)
  parser.add_argument("--wall-angle",type=float,default=None,
    help="Wall angle from horizontal; overrides clip metadata")
  parser.add_argument("--unit-layout",choices=("all", "corners"),default=None)
  parser.add_argument("--unit-quantile",type=float,default=.5)
  parser.add_argument("--preload-friction",type=float,default=.2,
    help="Pre-latch friction coefficient; .2 matches the climbing contact pairs")
  parser.add_argument("--toe-friction",type=float,default=.2,
    help="Toe friction coefficient; .2 matches the climbing contact pairs")
  args=parser.parse_args(); args.output_dir.mkdir(parents=True,exist_ok=True)
  clip=np.load(args.motion,allow_pickle=True).item()
  if args.wall_angle is not None:
    clip["wall_angle"] = args.wall_angle
  if args.unit_layout is not None:
    clip["unit_layout"] = args.unit_layout
  # Canonical clips without metadata belong to the twelve-unit main task.
  # Corner variants carry their layout, or select it with --unit-layout.
  clip.setdefault("unit_layout", "all")
  if not 0. < float(clip.get("wall_angle",90.)) <= 90.:
    parser.error("--wall-angle must be in (0, 90]")
  if not 0. <= args.unit_quantile <= 1. or min(args.preload_friction,args.toe_friction) < 0. or args.stride < 1:
    parser.error("Invalid unit quantile, preload friction, or stride")
  model,sites,bodies,toes,_=build_reference_model(tuple(clip["ee_list"]))
  data=mujoco.MjData(model); surface=EmpiricalLimitSurface()
  unit_geometry=matching_pad_geometry(model,clip["ee_list"],unit_layout=clip.get("unit_layout","all"))
  geometry=audit_geometry(model,data,clip,sites,toes)
  print(json.dumps(geometry,indent=2),flush=True)
  frames=np.unique(np.r_[np.arange(0,len(clip["joint_positions"]),args.stride),
    np.flatnonzero(np.any(np.diff(clip["ee_engaged"].astype(int),axis=0)!=0,axis=1)),
    np.flatnonzero(np.any(np.diff(clip["ee_engaged"].astype(int),axis=0)!=0,axis=1))+1])
  if args.initial_only: frames=np.array([0])
  statics,loads=[],[]
  for frame in frames:
    set_frame(model,data,clip,int(frame))
    row,forces=static_frame(model,data,clip,int(frame),sites,bodies,toes,surface,
      quantile=args.unit_quantile,toe_friction=args.toe_friction)
    statics.append(row);loads.extend(forces)
    print(f"static frame {frame}: linear={row['linear_feasible']}, excess={row.get('max_unit_normal_excess_n')}",flush=True)
  report=dict(motion=str(args.motion), motion_sha256=hashlib.sha256(args.motion.read_bytes()).hexdigest(),
    wall_angle_deg=float(clip.get("wall_angle",90.)), unit_layout=clip.get("unit_layout","all"),
    active_units_per_pad=unit_geometry.count, active_unit_sites=list(unit_geometry.unit_names),
    robot_mass_kg=float(model.body_mass.sum()), gravity_world_m_s2=model.opt.gravity.tolist(),
    unit_quantile=args.unit_quantile, preload_friction=args.preload_friction,
    toe_friction=args.toe_friction,
    compression_limit_per_unit_n=None,
    normal_capacity_semantics="Upper pull-off bound; 10 N engagement preload is a minimum, not a compression limit",
    geometry=geometry, statics=statics, static_all_characteristic_feasible=all(r.get("characteristic_feasible",False) for r in statics),
    limitations=["No physics steps, policy execution or training",
      "Static necessary-condition relaxation: only engaged passive gimbals balanced; free swinging gimbals and acceleration excluded",
      "SLSQP nonconvex local solution: a failure is not proof that all allocations are impossible",
      "Equal median unit quantiles are a characteristic audit, not median pad capacity",
      "IID survival fractions are model estimates on fixed reference-derived loads, not observed reliability",
      "10 N per-unit preload gate estimates compression from the C-referenced contact wrench using equal stiffness; "
      "corner proxies do not measure individual unit preload, and seeded mature contacts bypass seating"])
  if not args.initial_only:
    report["engage_preload"] = audit_engage_preload(model,data,clip,sites,bodies,toes,surface,
      quantile=args.unit_quantile, preload_friction=args.preload_friction,toe_friction=args.toe_friction)
    print(f"Engage preload: {report['engage_preload']['engage_edges']} edges, "
          f"all characteristic feasible={report['engage_preload']['all_feasible']}", flush=True)
  report["overall_pass"] = bool(geometry["geometry_pass"]
    and report["static_all_characteristic_feasible"]
    and report.get("engage_preload", {}).get("all_feasible", True))
  (args.output_dir/"reference_audit.json").write_text(json.dumps(report,indent=2)+"\n")
  if loads:
    with (args.output_dir/"reference_static_loads.csv").open("w") as stream:
      writer=csv.DictWriter(stream,fieldnames=list(loads[0]));writer.writeheader();writer.writerows(loads)
  raise SystemExit(0 if report["overall_pass"] else 1)


if __name__=="__main__":main()
