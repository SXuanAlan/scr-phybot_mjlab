"""The mount's IK context and frame accumulator, shared by every choreography segment."""

from __future__ import annotations

from pathlib import Path

import mujoco
import numpy as np

from scr_phybot_mjab.assets.phybot_c2.phybot_c2_constants import (
  PHYBOT_C2_POLICY_JOINT_NAMES,
  PHYBOT_C2_STANDING_JOINT_POS,
)
from scr_phybot_mjab.tools.climbing_reference import kinematics as kin

from .recipe import MountRecipe
from .wall_frame import WallFrame

SEGMENT_NAMES = (
  "ground_stand",
  "hand_raise",
  "hand_place",
  "hand_set_left",
  "hand_place_right",
  "hand_set_hold",
  "left_foot_mount",
  "foot_switch_hold",
  "right_foot_mount",
  "brace_hold",
  "hand_up_left",
  "hand_up_hold",
  "hand_up_right",
  "handoff_hold",
  "end_hold",
  "hand_raise_right",
  "hand_set_right",
  "hand_up_left_mid",
  "hand_up_left_hold",
)

ARM_JOINTS = ("shoulder_pitch", "shoulder_roll", "shoulder_yaw", "elbow_pitch")


def leg_joints(side: str) -> tuple[str, ...]:
  return tuple(
    f"{side}_{j}"
    for j in (
      "hip_pitch",
      "hip_roll",
      "hip_yaw",
      "knee",
      "ankle_pitch",
      "ankle_roll",
    )
  )


def arm_joints(side: str) -> tuple[str, ...]:
  return tuple(f"{side}_{j}" for j in ARM_JOINTS)


def key_blend(keys: tuple[tuple[float, float], ...], s: float) -> float:
  """Piecewise-smootherstep through (s_key, value) waypoints."""
  if s <= keys[0][0]:
    return keys[0][1]
  for (s0, v0), (s1, v1) in zip(keys[:-1], keys[1:]):
    if s <= s1:
      u = kin.smootherstep((s - s0) / max(s1 - s0, 1.0e-9))
      return float((1.0 - u) * v0 + u * v1)
  return keys[-1][1]


class MountBuilder:
  def __init__(self, recipe: MountRecipe, handoff_motion: Path) -> None:
    self.recipe = recipe

    if not 0.0 < self.recipe.wall_angle_deg <= 90.0:
      raise ValueError("--wall-angle-deg must be in (0, 90].")
    self.wall = WallFrame(self.recipe.wall_angle_deg)
    self.ik = kin.ClimbIK(limit_margin_frac=self.recipe.limit_margin_frac, limit_weight=self.recipe.limit_weight)
    self.model, self.data = self.ik.model, self.ik.data
    self.lh, self.rh, self.lf, self.rf = self.ik.ee_names
    self.hands, self.feet = (self.lh, self.rh), (self.lf, self.rf)
    self.seg_id = {name: i for i, name in enumerate(SEGMENT_NAMES)}

    self.seg_frames = {
      "ground_stand": max(int(round(self.recipe.stand_s * self.recipe.fps)), 1),
      "hand_raise": max(int(round(self.recipe.raise_s * self.recipe.fps)), 2),
      "hand_place": max(int(round(self.recipe.place_s * self.recipe.fps)), 2),
      "hand_set_left": max(int(round((self.recipe.set_left_s) * self.recipe.fps)), 1),
      "hand_place_right": max(int(round(self.recipe.place_right_s * self.recipe.fps)), 2),
      "hand_set_hold": max(int(round(self.recipe.set_hold_s * self.recipe.fps)), 1),
      "left_foot_mount": max(int(round(self.recipe.foot_swing_s * self.recipe.fps)), 2),
      "foot_switch_hold": max(int(round(self.recipe.switch_hold_s * self.recipe.fps)), 1),
      "right_foot_mount": max(int(round(self.recipe.foot_swing_right_s * self.recipe.fps)), 2),
      "brace_hold": max(int(round(self.recipe.brace_hold_s * self.recipe.fps)), 2),
      "hand_up_left": max(int(round(self.recipe.hand_up_s * self.recipe.fps)), 2),
      "hand_up_hold": max(int(round(self.recipe.up_hold_s * self.recipe.fps)), 1),
      "hand_up_right": max(int(round(self.recipe.hand_up_s * self.recipe.fps)), 2),
      "hand_up_left_mid": max(int(round((self.recipe.hand_up_s) * self.recipe.fps)), 2),
      "hand_up_left_hold": max(int(round(self.recipe.up_left_mid_hold_s * self.recipe.fps)), 1),
      "handoff_hold": max(int(round(self.recipe.handoff_hold_s * self.recipe.fps)), 2),
      "end_hold": max(int(round(self.recipe.end_hold_s * self.recipe.fps)), 1),
    }
    # The right hand's raise and set dwell take the left hand's durations.
    self.seg_frames["hand_raise_right"] = self.seg_frames["hand_raise"]
    self.seg_frames["hand_set_right"] = self.seg_frames["hand_set_left"]
    self.n_rake = max(int(round(self.recipe.rake_s * self.recipe.fps)), 1)
    # The right foot rakes longer: its floor support ends before the pad face
    # reaches the wall, and the face-on-wall rake covers that weak window.
    n_rake_right = max(int(round(self.recipe.rake_right_s * self.recipe.fps)), 1)
    self.seg_rake = {seg: self.n_rake for seg in SEGMENT_NAMES}
    self.seg_rake["right_foot_mount"] = n_rake_right
    # The left foot's rake is shorter than the hands': a 0.5 s, 4 mm foot rake
    # dips the lifting ankle 4.7 mm into the floor at the pivot's lift-off.
    n_rake_foot = max(int(round(self.recipe.foot_rake_s * self.recipe.fps)), 1)
    self.seg_rake["left_foot_mount"] = n_rake_foot
    self.n_lt = max(int(round(self.recipe.hand_load_transfer_s * self.recipe.fps)), 0)
    for seg in (
      "hand_place",
      "hand_place_right",
      "left_foot_mount",
      "right_foot_mount",
      "hand_up_left",
      "hand_up_right",
    ):
      if self.seg_frames[seg] - self.seg_rake[seg] < 2:
        raise ValueError(f"rake leaves fewer than two arc frames in {seg}")
    if self.n_lt >= min(self.seg_frames["hand_up_left"], self.seg_frames["hand_up_right"]):
      raise ValueError("--hand-load-transfer-s must be shorter than a hand-up swing")
    # The pre-engage seating stroke rakes ALONG the wall surface.
    self.rake_vec = float(self.recipe.rake_dist) * self.wall.up
    self.rake_vec_foot = float(self.recipe.foot_rake_dist) * self.wall.up

    # -- the climb cycle's first frame -> the final, shifted stance ---------------
    # The cycle's stance is stored for a vertical wall; at a tilted angle the
    # stance, its anchors and the rungs rotate about the wall's base line.
    self.ik.set_state1()
    self.handoff = np.load(handoff_motion, allow_pickle=True).item()
    self.data.qpos[:] = self.model.qpos0
    self.data.qpos[:3] = self.handoff["root_position"][0]
    self.data.qpos[3:7] = np.asarray(self.handoff["root_quaternion"])[0, [3, 0, 1, 2]]
    for name, value in zip(self.handoff["joints_list"], self.handoff["joint_positions"][0]):
      self.data.qpos[self.ik.qpos_addresses[name]] = value
    for i, name in enumerate(self.handoff.get("passive_joint_names", ())):
      self.data.qpos[self.ik.qpos_addresses[name]] = self.handoff["passive_joint_positions"][0, i]
    mujoco.mj_forward(self.model, self.data)
    self.state1_qpos = self.data.qpos.copy()
    self.state1_joint = {nm: float(self.data.qpos[self.ik.qpos_addresses[nm]]) for nm in PHYBOT_C2_POLICY_JOINT_NAMES}
    state1_anchor = {ee: self.ik.pad_pose(ee) for ee in self.ik.ee_names}
    self.state1_qpos = self.wall.rotate_root_qpos(self.state1_qpos)
    state1_anchor = {
      ee: (self.wall.rotate_point(pos), self.wall.rotate_mat(rot)) for ee, (pos, rot) in state1_anchor.items()
    }
    self.state1_pad_rot = {ee: state1_anchor[ee][1].copy() for ee in self.ik.ee_names}
    state1_hand_z = self.wall.along_wall(state1_anchor[self.lh][0])
    self.dz = self.recipe.hand_z - state1_hand_z
    self.final_qpos = self.state1_qpos.copy()
    self.final_qpos[0:3] = self.final_qpos[0:3] + self.dz * self.wall.up
    self.final_anchor_pos = {ee: state1_anchor[ee][0] + self.dz * self.wall.up for ee in self.ik.ee_names}
    self.low_anchor_pos = {
      ee: self.final_anchor_pos[ee] + (self.recipe.hand_z_low - self.recipe.hand_z) * self.wall.up for ee in self.hands
    }
    if self.recipe.hand_z_low >= self.recipe.hand_z:
      raise ValueError("--hand-z-low must be below --hand-z")

    # -- standing keyframe -------------------------------------------------------
    # ``standoff`` is the horizontal distance to the wall's base line, not the
    # perpendicular distance at torso height: toe clearance at the base, where
    # the surface sits at x = 1.5 at any tilt, binds the standing robot.
    self.base_x0 = kin.WALL_SURFACE_X - self.recipe.standoff

    self.set_standing()
    self.standing_qpos = self.data.qpos.copy()
    self.standing_joint = {nm: float(self.data.qpos[self.ik.qpos_addresses[nm]]) for nm in PHYBOT_C2_POLICY_JOINT_NAMES}
    self.foot_body_bid = {
      ee: mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, toe.body_name) for ee, toe in self.ik.toes.items()
    }
    self.foot_hold_pose = {}
    for i, ee in enumerate(self.feet):
      sid = self.ik.foot_sid[i]
      self.foot_hold_pose[ee] = (
        self.data.site_xpos[sid].copy(),
        self.data.site_xmat[sid].reshape(3, 3).copy(),
        sid,
      )
    # Ground level is where the standing soles rest: the lowest AABB corner of
    # the foot collision geoms (the keyframe has ~0.7 mm of sole penetration;
    # the toe spheres hover 10 mm higher and must not define the floor).
    self.ground_z = np.inf
    _corner_signs = np.array(
      [[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)],
      dtype=np.float64,
    )
    for ee in self.feet:
      bid = self.foot_body_bid[ee]
      for g in range(self.model.ngeom):
        if self.model.geom_bodyid[g] != bid:
          continue
        if self.model.geom_contype[g] == 0 and self.model.geom_conaffinity[g] == 0:
          continue
        world = (
          self.data.geom_xpos[g]
          + (self.model.geom_aabb[g, :3] + self.model.geom_aabb[g, 3:] * _corner_signs)
          @ self.data.geom_xmat[g].reshape(3, 3).T
        )
        self.ground_z = min(self.ground_z, float(world[:, 2].min()))

    # -- bookkeeping -------------------------------------------------------------
    self.qpos_frames: list[np.ndarray] = []
    self.labels: list[int] = []
    self.engaged_frames: list[np.ndarray] = []
    self.anchor_frames: list[np.ndarray] = []
    self.ground_frames: list[np.ndarray] = []
    self.seg_residual = {name: kin.IKResidual() for name in SEGMENT_NAMES}
    self.landing_report: dict[str, tuple[float, float]] = {}
    self.toe_gap_report: dict[str, float] = {}

    # Each end-effector's anchor, as in the cycle clip: the weld target while
    # engaged, the scheduled landing target while free.
    self.anchor_now = {ee: self.low_anchor_pos[ee].copy() for ee in self.hands}
    self.anchor_now.update({ee: self.final_anchor_pos[ee].copy() for ee in self.feet})
    self.on_ground = {self.lf: True, self.rf: True}

    self.sel_all = self.ik.dof_selection(PHYBOT_C2_POLICY_JOINT_NAMES)

    self.ankle_joints = {
      ee: tuple(nm for nm in PHYBOT_C2_POLICY_JOINT_NAMES if nm.startswith(toe.body_name.removesuffix("_roll")))
      for ee, toe in self.ik.toes.items()
    }

    self.base_track: dict[str, np.ndarray] = {}

    self.toe_wall_hold: dict[str, np.ndarray] = {}

    # The right hand's first rung is a middle one, reachable flat-footed; it
    # walks up to its final rung after the feet mount.
    if not (self.recipe.hand_z_low < self.recipe.hand_z_mid < self.recipe.hand_z):
      raise ValueError("hand_z_mid must lie between hand_z_low and hand_z")
    self.mid_anchor_rh = (
      self.final_anchor_pos[self.rh]
      + (self.recipe.hand_z_mid - self.recipe.hand_z) * self.wall.up
      + np.array([0.0, self.recipe.mid_rung_y, 0.0])
    )
    self.anchor_now[self.rh] = self.mid_anchor_rh.copy()
    # The left hand's first rung may sit outboard of its final lane: on the
    # lane, a low rung needs the shoulder adducted past its -0.35 rad stop
    # once the body rises (roll pinned at -0.34, weld torn 14 mm).
    self.low_anchor_pos[self.lh] = self.low_anchor_pos[self.lh] + np.array([0.0, self.recipe.low_rung_y, 0.0])
    self.anchor_now[self.lh] = self.low_anchor_pos[self.lh].copy()

  def weld_task(self, ee: str, anchor: tuple[np.ndarray, np.ndarray]) -> kin.IKTask:
    # Support welds weigh recipe.weld_weight_scale times the cycle's weld task.
    task = kin.weld_task(self.ik, ee, anchor)
    task.w_pos = task.w_pos * self.recipe.weld_weight_scale
    task.w_rot = task.w_rot * self.recipe.weld_weight_scale
    return task

  def swing_target(self, t, n_seg, p0, p1, p2, n_rake_seg=None, rake_vec_seg=None):
    nr = self.n_rake if n_rake_seg is None else n_rake_seg
    rake_vec = self.rake_vec if rake_vec_seg is None else rake_vec_seg
    n_arc = n_seg - nr
    if t < n_arc:
      s = float(kin.smootherstep(min(t / (n_arc - 1), 1.0)))
      return kin.bezier(p0, p1, p2 + rake_vec, s)
    s_rake = float(kin.smootherstep((t - n_arc + 1) / nr))
    return p2 + (1.0 - s_rake) * rake_vec

  # ``stand_y`` parks the standing footprint laterally off the anchor lane,
  # which stays at y = 0 with the handoff stance.
  def set_standing(self) -> None:
    self.data.qpos[:] = 0.0
    self.data.qpos[0:3] = (self.base_x0, self.recipe.stand_y, 0.726)
    self.data.qpos[3:7] = (1.0, 0.0, 0.0, 0.0)
    for name, angle in PHYBOT_C2_STANDING_JOINT_POS.items():
      if name in self.ik.qpos_addresses:
        self.data.qpos[self.ik.qpos_addresses[name]] = angle
    # Keep the CAD gimbal rest pose during IK as well as at the seam.
    # Locking it at zero would rotate the forearm into the wall while
    # matching a pad orientation generated at the -3 degree stop.
    for name in self.handoff.get("passive_joint_names", ()):
      adr = self.ik.qpos_addresses[name]
      self.data.qpos[adr] = self.state1_qpos[adr]
    mujoco.mj_forward(self.model, self.data)
    # Narrow the stance: the walk keyframe plants the feet at +-0.148 m; IK
    # re-plants them at +-stance_width/2 (same x, flat, base pose pinned), so
    # a lifted foot leaves the CoM closer to the stance foot and the hands
    # carry less tipping.
    sel_st = self.ik.dof_selection(PHYBOT_C2_POLICY_JOINT_NAMES)
    post_st = {nm: float(self.data.qpos[self.ik.qpos_addresses[nm]]) for nm in PHYBOT_C2_POLICY_JOINT_NAMES}
    base_st = self.data.qpos[0:3].copy()
    tasks_st = [kin.base_task(self.ik, base_st, (1000.0, 1000.0, 1000.0), rot=np.eye(3), w_rot=500.0)]
    for sid_st in self.ik.foot_sid:
      pos_st = self.data.site_xpos[sid_st].copy()
      rot_st = self.data.site_xmat[sid_st].reshape(3, 3).copy()
      pos_st[1] = float(np.sign(pos_st[1])) * self.recipe.stance_width / 2.0
      tasks_st.append(
        kin.IKTask(kind="site", sid=sid_st, pos=pos_st, rot=rot_st, w_pos=np.full(3, 1000.0), w_rot=500.0)
      )
    for _ in range(4):
      self.ik.solve_frame(tasks_st, sel_st, post_st, posture_weight=0.2, iters=200, max_step=0.1)
    mujoco.mj_forward(self.model, self.data)

  def push(self, segment: str, engaged: dict[str, bool]) -> None:
    self.qpos_frames.append(self.data.qpos.copy())
    self.labels.append(self.seg_id[segment])
    self.engaged_frames.append(np.array([engaged[ee] for ee in self.ik.ee_names], dtype=np.uint8))
    self.anchor_frames.append(np.stack([self.anchor_now[ee] for ee in self.ik.ee_names]))
    self.ground_frames.append(np.array([self.on_ground[self.lf], self.on_ground[self.rf]], dtype=np.uint8))

  def foot_hold_task(self, ee: str) -> kin.IKTask:
    pos, rot, sid = self.foot_hold_pose[ee]
    return kin.IKTask(
      kind="site",
      sid=sid,
      pos=pos,
      rot=rot,
      w_pos=np.full(3, 1000.0),
      w_rot=500.0,
    )

  def note_base(self, segment: str, target: np.ndarray) -> None:
    """Record |base - target| per axis: base tasks never enter the IK residual."""
    err = np.abs(self.data.qpos[0:3] - np.asarray(target, dtype=np.float64))
    self.base_track[segment] = np.maximum(self.base_track.get(segment, np.zeros(3)), err)

  def dwell(
    self,
    segment: str,
    engaged: dict[str, bool],
    weld_ees: tuple[str, ...],
    hold_feet: tuple[str, ...],
    base_shift: np.ndarray | None = None,
    plant_toes: tuple[str, ...] = (),
    toe_hold_targets: dict[str, np.ndarray] | None = None,
    base_w: tuple[float, float, float] = (60.0, 60.0, 60.0),
    extra_overrides: dict[str, float] | None = None,
  ) -> None:
    """A static dwell on the current pose: the welded pads hold their poses and
    ``hold_feet`` stay flat on the floor, while the base target may drift by
    ``base_shift``. ``plant_toes`` rest on the wall (the cycle's ankle-only
    move) and ``toe_hold_targets`` keep planted toes where they are."""
    for ee in weld_ees:
      cur = self.ik.pad_pose(ee)
      self.anchor_now[ee] = cur[0].copy()
    weld_anchor = {ee: self.ik.pad_pose(ee) for ee in weld_ees}
    base_hold = self.data.qpos[0:3].copy()
    base_rot = np.zeros(9)
    mujoco.mju_quat2Mat(base_rot, self.data.qpos[3:7])
    base_rot = base_rot.reshape(3, 3)
    posture_hold = {nm: float(self.data.qpos[self.ik.qpos_addresses[nm]]) for nm in PHYBOT_C2_POLICY_JOINT_NAMES}
    overrides: dict[str, float] = {}
    plant0, plant_goal = {}, {}
    for ee in toe_hold_targets or ():
      for nm in self.ankle_joints[ee]:
        if nm.endswith("_roll"):
          overrides[nm] = 3.0
    for ee in hold_feet:
      # A held foot stays flat: on a high reach the solver otherwise pitches
      # the stance foot off its 1000/500 hold and both sole points lift.
      for nm in self.ankle_joints[ee]:
        overrides[nm] = 6.0
    overrides.update(extra_overrides or {})
    for ee in plant_toes:
      toe0 = self.data.site_xpos[self.ik.toe_sid[ee]].copy()
      # Aim at exact contact: the static audit credits a released support
      # within 0.5 mm of the wall and the IK undershoots by a few tenths. The
      # reference commands no press: pressing a toe whose own pad is latched
      # loads that anchor toward its weak release directions.
      plant0[ee] = toe0
      # Same target on the tilted plane: project the toe onto the wall,
      # 5 cm up-wall, one sphere radius short of the surface.
      plant_goal[ee] = (
        self.wall.pivot
        + (self.wall.along_wall(toe0) + 0.05) * self.wall.up
        + np.array([0.0, toe0[1], 0.0])
        + self.ik.toes[ee].radius * self.wall.outward
      )
      for nm in self.ankle_joints[ee]:
        if nm.endswith("_roll"):
          overrides[nm] = 3.0
    for t in range(self.seg_frames[segment]):
      s = float(kin.smootherstep(t / max(self.seg_frames[segment] - 1, 1)))
      tasks = [self.weld_task(ee, weld_anchor[ee]) for ee in weld_ees]
      tasks.extend(self.foot_hold_task(ee) for ee in hold_feet)
      for ee in plant_toes:
        # IKTask weights are per world axis and cannot express a tilted
        # anisotropy: with a fixed up-wall goal the ankle's along-wall
        # shortfall (~5 cm at weight 10) leaks into the surface normal and the
        # toe lands ~2 mm through the wall. Re-project the goal every frame at
        # the toe's own along-wall height with a decaying up-pull, so the
        # strong normal weight converges to a clean touch.
        toe_now = self.data.site_xpos[self.ik.toe_sid[ee]]
        goal_now = (
          self.wall.pivot
          + (self.wall.along_wall(toe_now) + 0.05 * (1.0 - s)) * self.wall.up
          + np.array([0.0, plant0[ee][1], 0.0])
          + self.ik.toes[ee].radius * self.wall.outward
        )
        toe_target = (1.0 - s) * plant0[ee] + s * goal_now
        tasks.append(
          kin.toe_task(
            self.ik,
            ee,
            toe_target,
            w_pos=(600.0, 3.0, 10.0),
          )
        )
      if toe_hold_targets:
        for ee, pos in toe_hold_targets.items():
          tasks.append(kin.toe_task(self.ik, ee, pos, w_pos=(600.0, 3.0, 10.0)))
      for ee in plant_toes:
        self.toe_gap_report[ee] = (
          self.wall.wall_distance(self.data.site_xpos[self.ik.toe_sid[ee]]) - self.ik.toes[ee].radius
        )
      target = base_hold if base_shift is None else base_hold + s * base_shift
      tasks.append(kin.base_task(self.ik, target, base_w, rot=base_rot, w_rot=8.0))
      res = self.ik.solve_frame(
        tasks,
        self.sel_all,
        posture_hold,
        posture_weight=0.3,
        posture_weight_overrides=overrides,
      )
      self.note_base(segment, target)
      self.seg_residual[segment].merge(res)
      self.push(segment, engaged)

  def stiff_foot_hold(self, ee: str) -> kin.IKTask:
    # The only floor support of a braced reach or step: twice the flat hold,
    # aimed 1 mm low. The audit gates each sole point at 8 mm and the
    # standing toe points already sit at 6.6-7.5 mm, so the 1-2 mm the
    # 1000/500 hold concedes to a deep reach drops a support corner; at
    # 3000/1500 the toe still grazed 8.1 mm while a hand weld drifted 3.6 mm.
    pos, rot, sid = self.foot_hold_pose[ee]
    return kin.IKTask(
      kind="site",
      sid=sid,
      pos=pos - np.array([0.0, 0.0, 0.002]),
      rot=rot,
      w_pos=np.full(3, 2000.0),
      w_rot=1000.0,
    )

  def engaged(self, *ees: str) -> dict[str, bool]:
    """The latch chart for one frame: the given end-effectors engaged, the rest free."""
    return {ee: ee in ees for ee in self.ik.ee_names}
