"""Offline synthesis of the four-regrip climbing cycle by whole-body IK.

One cycle raises the robot by one step: the feet regrip one after the other,
then the hands. Every swing keeps three pads latched. ``generate_clip``
returns the clip arrays and a quality report; the caller saves them.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import mujoco
import numpy as np

from scr_phybot_mjab.assets.phybot_c2.phybot_c2_constants import PHYBOT_C2_POLICY_JOINT_NAMES

from .kinematics import (
  FOOT_SITE_NAMES,
  PAD_TOUCH_X,
  WALL_SURFACE_X,
  ClimbIK,
  IKResidual,
  IKTask,
  base_task,
  bezier,
  rotation_angle,
  slerp_matrix,
  smootherstep,
  toe_task,
  weld_task,
)

SEGMENT_NAMES = (
  "hold_start",  # state 1
  "left_foot_swing",  # 1 -> 2
  "foot_transfer_hold",  # state 2
  "right_foot_swing",  # 2 -> 3
  "seat_hold",  # state 3
  "left_hand_swing",  # 3 -> 4
  "regrip_hold",  # state 4
  "right_hand_swing",  # 4 -> 5
  "end_hold",  # state 5 == state 1 + step height
)
_SEGMENT_ID = {name: i for i, name in enumerate(SEGMENT_NAMES)}
_SWINGS = ("left_foot_swing", "right_foot_swing", "left_hand_swing", "right_hand_swing")
_JOINTS = PHYBOT_C2_POLICY_JOINT_NAMES

# A dwelling toe hovers this far off the wall: it carries no force while its
# own pad is still latched and closes the gap once the pad releases.
_TOE_HOVER = 0.0015
# Hand swings start with a few throttled IK solves: releasing a hand moves the
# soft-task equilibrium, and an unthrottled first solve pops the whole body.
_THROTTLED_FRAMES = 8


@dataclass(frozen=True)
class ClimbCycleRecipe:
  """Timing (s), geometry (m) and IK weighting of one climbing cycle."""

  step_height: float | None = 0.02  # None: the IK model's default step
  cycles: int = 1
  fps: float = 50.0
  hold_start_s: float = 0.6
  foot_swing_s: float = 1.5
  foot_transfer_hold_s: float = 0.6
  seat_hold_s: float = 0.6
  swing_s: float = 1.8
  regrip_hold_s: float = 0.6
  end_hold_s: float = 0.8
  hand_load_transfer_s: float = 0.2  # the outgoing hand stays latched this long
  rake_s: float = 0.2  # final pre-engage drag down the wall
  rake_dist: float = 0.002
  swing_bulge: float = 0.01  # hand arc bulge away from the wall
  swing_lateral: float = 0.0
  foot_bulge: float = 0.01
  body_press: float = 0.0  # base shift toward the wall during transfers
  lateral_weight_shift: float = 0.0
  limit_margin_frac: float = 0.08
  limit_weight: float = 3.0


# The committed ``climb_cad_empirical`` reference.
CAD_EMPIRICAL = ClimbCycleRecipe()


@dataclass
class CycleClip:
  """Clip arrays (the saved ``.npy`` dict) and a printable quality report."""

  data: dict
  report: list[str]


def generate_clip(recipe: ClimbCycleRecipe, initial_state: dict | None = None) -> CycleClip:
  ik = ClimbIK(
    limit_margin_frac=recipe.limit_margin_frac, limit_weight=recipe.limit_weight, initial_state=initial_state
  )
  ctx = _CycleContext.build(ik, recipe)
  ik.set_state1()
  state1_qpos = ik.data.qpos.copy()
  clip = _ClipRecorder(ik)
  ctx.capture_state1(clip)
  lh, rh, lf, rf = ik.ee_names  # microspine order: hands then feet

  _dwell(ctx, clip, "hold_start", plant_ee=lf)
  # Both foot targets are one step above the anchors at the end of state 1.
  step = np.array([0.0, 0.0, ctx.height])
  foot_targets = {ee: (clip.anchors[ee][0] + step, ctx.state1_pad_rot[ee]) for ee in (lf, rf)}
  _foot_swing(ctx, clip, "left_foot_swing", lf, foot_targets[lf], foot_index=0)
  _dwell(ctx, clip, "foot_transfer_hold", plant_ee=rf, retract_ee=lf)
  _foot_swing(ctx, clip, "right_foot_swing", rf, foot_targets[rf], foot_index=1)
  _dwell(ctx, clip, "seat_hold", retract_ee=rf)
  _hand_swing(ctx, clip, "left_hand_swing", lh, lateral_sign=1.0)
  clip.hold("regrip_hold", ctx.all_on, ctx.frames["regrip_hold"])
  _hand_swing(ctx, clip, "right_hand_swing", rh, lateral_sign=-1.0)
  closure = _close_cycle(ctx, clip, state1_qpos)
  clip.hold("end_hold", ctx.all_on, ctx.frames["end_hold"])

  per_cycle = len(clip.qpos) - ctx.frames["end_hold"]
  tiled = _tile_cycles(clip, recipe.cycles, ctx.height, per_cycle)
  data, fk_quality = _clip_arrays(ik, tiled, recipe.fps, ctx.height)
  report = _report(ctx, clip, data, fk_quality, closure, per_cycle)
  return CycleClip(data=data, report=report)


@dataclass
class _CycleContext:
  """Quantities derived once from the recipe and the state-1 stance."""

  ik: ClimbIK
  recipe: ClimbCycleRecipe
  height: float
  frames: dict[str, int]
  rake_frames: int
  load_transfer_frames: int
  rake_vec: np.ndarray
  all_on: dict[str, bool]
  dof_selection: np.ndarray
  ankle_joints: dict[str, tuple[str, ...]]
  arm_joints: dict[str, tuple[str, ...]]
  posture_ref: dict[str, float] = field(default_factory=dict)
  base_xy: np.ndarray = field(default_factory=lambda: np.zeros(2))
  base_z1: float = 0.0
  state1_pad_rot: dict[str, np.ndarray] = field(default_factory=dict)
  stance_toe_clear: dict[str, float] = field(default_factory=dict)

  @classmethod
  def build(cls, ik: ClimbIK, recipe: ClimbCycleRecipe) -> _CycleContext:
    if recipe.cycles < 1:
      raise ValueError("cycles must be >= 1")
    frames = _segment_frames(recipe)
    rake_frames = max(int(round(recipe.rake_s * recipe.fps)), 1)
    load_transfer_frames = max(int(round(recipe.hand_load_transfer_s * recipe.fps)), 0)
    _check_swing_lengths(frames, rake_frames, load_transfer_frames)
    lh, rh = ik.ee_names[:2]
    return cls(
      ik=ik,
      recipe=recipe,
      height=ik.default_step_height if recipe.step_height is None else float(recipe.step_height),
      frames=frames,
      rake_frames=rake_frames,
      load_transfer_frames=load_transfer_frames,
      rake_vec=np.array([0.0, 0.0, float(recipe.rake_dist)]),
      all_on={ee: True for ee in ik.ee_names},
      dof_selection=ik.dof_selection(_JOINTS),
      # A bracing toe moves only its own ankle: left_ankle_roll -> left_ankle_*.
      ankle_joints={
        ee: tuple(nm for nm in _JOINTS if nm.startswith(toe.body_name.removesuffix("_roll")))
        for ee, toe in ik.toes.items()
      },
      arm_joints={
        lh: tuple(nm for nm in _JOINTS if nm.startswith("left_sh") or nm == "left_elbow_pitch"),
        rh: tuple(nm for nm in _JOINTS if nm.startswith("right_sh") or nm == "right_elbow_pitch"),
      },
    )

  def capture_state1(self, clip: _ClipRecorder) -> None:
    """Record the stance references; call right after ``ik.set_state1()``."""
    ik = self.ik
    self.base_xy = ik.init_base_pos[:2]
    self.base_z1 = float(ik.init_base_pos[2])
    self.state1_pad_rot = {ee: clip.anchors[ee][1].copy() for ee in ik.ee_names}
    self.posture_ref = {nm: ik.init_joint_pos[nm] for nm in _JOINTS}
    self.stance_toe_clear = {
      ee: float(WALL_SURFACE_X - ik.data.site_xpos[ik.toe_sid[ee]][0] - toe.radius) for ee, toe in ik.toes.items()
    }

  def arc_progress(self, t: int, n_seg: int) -> float:
    """Approach-arc progress in [0, 1]; saturated during the rake frames."""
    n_arc = n_seg - self.rake_frames
    return float(smootherstep(min(t / (n_arc - 1), 1.0)))

  def swing_target(self, t: int, n_seg: int, p0: np.ndarray, p1: np.ndarray, p2: np.ndarray) -> np.ndarray:
    """Arc to ``rake_dist`` above the anchor, then drag straight down onto it."""
    n_arc = n_seg - self.rake_frames
    if t < n_arc:
      return bezier(p0, p1, p2 + self.rake_vec, self.arc_progress(t, n_seg))
    s_rake = float(smootherstep((t - n_arc + 1) / self.rake_frames))
    return p2 + (1.0 - s_rake) * self.rake_vec

  def posture_now(self) -> dict[str, float]:
    ik = self.ik
    return {nm: float(ik.data.qpos[ik.qpos_addresses[nm]]) for nm in _JOINTS}


class _ClipRecorder:
  """Per-frame records of one cycle plus the anchor bookkeeping they refer to."""

  def __init__(self, ik: ClimbIK) -> None:
    self.ik = ik
    self.qpos: list[np.ndarray] = []
    self.labels: list[int] = []
    self.engaged: list[np.ndarray] = []
    self.anchor_points: list[np.ndarray] = []
    self.toe_braced: list[np.ndarray] = []
    self.residual = {name: IKResidual() for name in SEGMENT_NAMES}
    self.anchors = {ee: ik.pad_pose(ee) for ee in ik.ee_names}  # weld target (pos, rot) per pad
    self.anchor_now = {ee: self.anchors[ee][0].copy() for ee in ik.ee_names}  # recorded anchor per pad
    self.toe_braced_now = {ee: False for ee in ik.toes}
    self.landing: dict[str, tuple[float, float]] = {}
    self.toe_report: dict[str, dict[str, float]] = {}

  def record(self, segment: str, engaged: dict[str, bool]) -> None:
    ik = self.ik
    self.qpos.append(ik.data.qpos.copy())
    self.labels.append(_SEGMENT_ID[segment])
    self.engaged.append(np.array([engaged[ee] for ee in ik.ee_names], dtype=np.uint8))
    self.anchor_points.append(np.stack([self.anchor_now[ee] for ee in ik.ee_names]))
    self.toe_braced.append(np.array([self.toe_braced_now[ee] for ee in ik.toes], dtype=np.uint8))

  def hold(self, segment: str, engaged: dict[str, bool], frames: int) -> None:
    for _ in range(frames):
      self.record(segment, engaged)

  def reseat(self, ees) -> None:
    """Re-anchor welds at the current pad poses, as the sim re-anchors them."""
    for ee in ees:
      self.anchors[ee] = self.ik.pad_pose(ee)
      self.anchor_now[ee] = self.anchors[ee][0].copy()


def _segment_frames(recipe: ClimbCycleRecipe) -> dict[str, int]:
  def frames(seconds: float, minimum: int) -> int:
    return max(int(round(seconds * recipe.fps)), minimum)

  return {
    "hold_start": frames(recipe.hold_start_s, 1),
    "left_foot_swing": frames(recipe.foot_swing_s, 2),
    "foot_transfer_hold": frames(recipe.foot_transfer_hold_s, 1),
    "right_foot_swing": frames(recipe.foot_swing_s, 2),
    "seat_hold": frames(recipe.seat_hold_s, 1),
    "left_hand_swing": frames(recipe.swing_s, 2),
    "regrip_hold": frames(recipe.regrip_hold_s, 1),
    "right_hand_swing": frames(recipe.swing_s, 2),
    "end_hold": frames(recipe.end_hold_s, 1),
  }


def _check_swing_lengths(frames: dict[str, int], rake_frames: int, load_transfer_frames: int) -> None:
  for name in _SWINGS:
    if frames[name] - rake_frames < 2:
      raise ValueError(f"rake_s leaves fewer than two arc frames in '{name}'; shorten the rake or lengthen the segment")
  if load_transfer_frames >= min(frames["left_hand_swing"], frames["right_hand_swing"]):
    raise ValueError("hand_load_transfer_s must be shorter than each hand swing")


def _dwell(
  ctx: _CycleContext, clip: _ClipRecorder, segment: str, plant_ee: str | None = None, retract_ee: str | None = None
) -> None:
  """Four-pad dwell that moves only ankles: plant one toe and/or fold one back.

  The pads sit on the shins and the toes on the articulated feet, so ankle
  motion is decoupled from every anchor; base and posture are held.
  """
  ik = ctx.ik
  clip.reseat(ik.ee_names)
  base_hold = ik.data.qpos[0:3].copy()
  base_rot_hold = np.zeros(9)
  mujoco.mju_quat2Mat(base_rot_hold, ik.data.qpos[3:7])
  base_rot_hold = base_rot_hold.reshape(3, 3)
  posture_hold = ctx.posture_now()
  overrides: dict[str, float] = {}
  retract_from: dict[str, float] = {}
  if retract_ee is not None:
    # Blend the finished toe back to stance angles; a step reference would
    # snap the free ankle in one frame.
    for nm in ctx.ankle_joints[retract_ee]:
      retract_from[nm] = posture_hold[nm]
      overrides[nm] = 6.0
  toe0 = goal = None
  if plant_ee is not None:
    toe0 = ik.data.site_xpos[ik.toe_sid[plant_ee]].copy()
    # Hover shy of the wall ~6 cm above the stance toe: pressing while the pad
    # is still latched would load the fresh anchor toward its weak release
    # directions. y rides free and the ankle roll is pinned, so the pitch
    # does the reaching (chasing y flips the pitch/roll IK branch).
    goal = np.array([WALL_SURFACE_X - ik.toes[plant_ee].radius - _TOE_HOVER, toe0[1], toe0[2] + 0.06])
    for nm in ctx.ankle_joints[plant_ee]:
      if nm.endswith("_roll"):
        overrides[nm] = 3.0
  n = ctx.frames[segment]
  for t in range(n):
    s = float(smootherstep(t / max(n - 1, 1)))
    tasks = [weld_task(ik, ee, clip.anchors[ee]) for ee in ik.ee_names]
    if plant_ee is not None:
      tasks.append(toe_task(ik, plant_ee, (1.0 - s) * toe0 + s * goal, w_pos=(300.0, 3.0, 10.0)))
    tasks.append(base_task(ik, base_hold, (30.0, 30.0, 30.0), rot=base_rot_hold, w_rot=8.0))
    for nm, start in retract_from.items():
      posture_hold[nm] = (1.0 - s) * start + s * ctx.posture_ref[nm]
    res = ik.solve_frame(tasks, ctx.dof_selection, posture_hold, posture_weight=0.3, posture_weight_overrides=overrides)
    clip.residual[segment].merge(res)
    clip.record(segment, ctx.all_on)


def _foot_swing(
  ctx: _CycleContext,
  clip: _ClipRecorder,
  segment: str,
  swing_ee: str,
  target: tuple[np.ndarray, np.ndarray],
  foot_index: int,
) -> None:
  """One foot regrips one step higher while its toe braces the wall.

  Feet never release together: two hands and one foot close the body moment
  through a large force couple instead of local pad bending.
  """
  ik, recipe = ctx.ik, ctx.recipe
  # Shift toward the still-anchored foot to cancel the yaw couple.
  support_side = -1.0 if swing_ee == ik.ee_names[2] else 1.0
  engaged = {ee: ee != swing_ee for ee in ik.ee_names}
  clip.reseat([ee for ee in ik.ee_names if ee != swing_ee])
  p0, rot0 = ik.pad_pose(swing_ee)
  p2, rot2 = target
  clip.anchor_now[swing_ee] = p2.copy()
  p1 = 0.5 * (p0 + p2) + np.array([-recipe.foot_bulge, 0.0, 0.0])
  n = ctx.frames[segment]
  toe = _ToeBrace.start(ctx, swing_ee, n)
  clip.toe_report[swing_ee] = toe.report
  base_start = ik.data.qpos[0:3].copy()
  # Pressing the body toward the wall keeps the released pad close enough to
  # re-seat and shortens gravity's wall-normal lever arm.
  base_end = np.array(
    [
      ctx.base_xy[0] + recipe.body_press,
      ctx.base_xy[1] + support_side * recipe.lateral_weight_shift,
      ctx.base_z1 + ctx.height * (foot_index + 1) / 2,
    ]
  )
  posture_start = ctx.posture_now()
  for t in range(n):
    s = float(smootherstep(t / (n - 1)))
    tasks = [weld_task(ik, ee, clip.anchors[ee]) for ee in ik.ee_names if ee != swing_ee]
    tasks.append(
      IKTask(
        kind="site",
        sid=ik.pad_sid[swing_ee],
        pos=ctx.swing_target(t, n, p0, p1, p2),
        rot=slerp_matrix(rot0, rot2, s),
        w_pos=np.full(3, 1000.0),
        w_rot=100.0,
      )
    )
    clip.toe_braced_now[swing_ee] = toe.braced(t)
    tasks.append(toe_task(ik, swing_ee, toe.target(t)))
    tasks.append(base_task(ik, (1.0 - s) * base_start + s * base_end, (100.0, 30.0, 50.0)))
    posture_t = {nm: (1.0 - s) * posture_start[nm] + s * ctx.posture_ref[nm] for nm in _JOINTS}
    res = ik.solve_frame(tasks, ctx.dof_selection, posture_t, posture_weight=0.3)
    clip.residual[segment].merge(res)
    toe.measure(t)
    clip.record(segment, engaged)
  clip.reseat([swing_ee])
  clip.landing[swing_ee] = (
    float(np.linalg.norm(clip.anchors[swing_ee][0] - p2)),
    rotation_angle(clip.anchors[swing_ee][1], rot2),
  )


@dataclass
class _ToeBrace:
  """Press schedule of a swinging foot's toe and how well it held the wall.

  The toe closes its hover gap over the first frames, braces through the
  travel and lifts back out during the landing rake; the wall-normal gap
  inside the press window must stay ~0.
  """

  ik: ClimbIK
  ee: str
  plant: np.ndarray  # toe position when the swing starts
  press_x: float  # toe surface on the wall
  press_in: int  # frames that close the hover gap
  press_end: int  # first rake frame, where the toe lifts out
  rake_frames: int
  report: dict[str, float]

  @classmethod
  def start(cls, ctx: _CycleContext, ee: str, n: int) -> _ToeBrace:
    ik = ctx.ik
    plant = ik.data.site_xpos[ik.toe_sid[ee]].copy()
    radius = ik.toes[ee].radius
    return cls(
      ik=ik,
      ee=ee,
      plant=plant,
      press_x=WALL_SURFACE_X - radius,
      press_in=max(int(round(0.06 * ctx.recipe.fps)), 2),
      press_end=n - ctx.rake_frames,
      rake_frames=ctx.rake_frames,
      report={"plant_gap": float(WALL_SURFACE_X - plant[0] - radius), "wall_gap": 0.0, "glide": 0.0},
    )

  def braced(self, t: int) -> bool:
    return self.press_in <= t < self.press_end

  def target(self, t: int) -> np.ndarray:
    target = self.plant.copy()
    press = self._press(t)
    target[0] = (1.0 - press) * self.plant[0] + press * self.press_x
    return target

  def measure(self, t: int) -> None:
    """Track the worst wall gap while bracing and the in-plane glide."""
    toe_now = self.ik.data.site_xpos[self.ik.toe_sid[self.ee]]
    if self.braced(t):
      gap = float(WALL_SURFACE_X - toe_now[0] - self.ik.toes[self.ee].radius)
      self.report["wall_gap"] = max(self.report["wall_gap"], gap)
    self.report["glide"] = max(self.report["glide"], float(np.linalg.norm(toe_now[1:] - self.plant[1:])))

  def _press(self, t: int) -> float:
    """Press fraction: ramp in, full brace, lift out during the rake."""
    if t < self.press_in:
      return (t + 1) / self.press_in
    if t >= self.press_end:
      return max(1.0 - (t - self.press_end + 1) / self.rake_frames, 0.0)
    return 1.0


def _hand_swing(ctx: _CycleContext, clip: _ClipRecorder, segment: str, swing_ee: str, lateral_sign: float) -> None:
  """One hand regrips one step higher; the other three pads stay latched.

  The outgoing hand stays latched for ``hand_load_transfer_s`` while the other
  limbs take up its load. Soft references (base, posture) start at the
  segment-start state and blend to the canonical stance, so releasing the hand
  does not re-equilibrate the whole body in one frame.
  """
  ik = ctx.ik
  engaged_swing = {ee: ee != swing_ee for ee in ik.ee_names}
  clip.reseat([ee for ee in ik.ee_names if ee != swing_ee])
  p0, rot0 = ik.pad_pose(swing_ee)
  p2 = clip.anchors[swing_ee][0] + np.array([0.0, 0.0, ctx.height])
  clip.anchor_now[swing_ee] = p0.copy()
  p1 = 0.5 * (p0 + p2) + np.array([-ctx.recipe.swing_bulge, lateral_sign * ctx.recipe.swing_lateral, 0.0])
  base_start = ik.data.qpos[0:3].copy()
  base_rot_start = np.zeros(9)
  mujoco.mju_quat2Mat(base_rot_start, ik.data.qpos[3:7])
  base_rot_start = base_rot_start.reshape(3, 3)
  posture_start = ctx.posture_now()
  first = segment == "left_hand_swing"
  n = ctx.frames[segment]
  for t in range(n):
    engaged = ctx.all_on if t < ctx.load_transfer_frames else engaged_swing
    if t == ctx.load_transfer_frames:
      clip.anchor_now[swing_ee] = p2.copy()
    s = float(smootherstep(t / (n - 1)))
    tasks = [weld_task(ik, ee, clip.anchors[ee]) for ee in ik.ee_names if ee != swing_ee]
    tasks.append(_hand_pad_task(ctx, swing_ee, t, n, (p0, p1, p2), rot0))
    tasks.append(_hand_base_task(ctx, swing_ee, first, s, base_start, base_rot_start))
    posture_t = {nm: (1.0 - s) * posture_start[nm] + s * ctx.posture_ref[nm] for nm in _JOINTS}
    if first:
      # The other hand is still anchored one step low, where its stance angles
      # are unreachable; its reference waits at the segment-start angles.
      for nm in ctx.arm_joints[ik.ee_names[1]]:
        posture_t[nm] = posture_start[nm]
    throttled = t < _THROTTLED_FRAMES
    res = ik.solve_frame(
      tasks,
      ctx.dof_selection,
      posture_t,
      # A stiff posture anchor keeps the 4-DoF arm from flipping between its
      # elbow-in/out null-space branches frame to frame.
      posture_weight=0.6 if first else 10.0,
      iters=2 if throttled else 100,
      max_step=0.003 if throttled else 0.12,
    )
    clip.residual[segment].merge(res)
    clip.record(segment, engaged)
  clip.reseat([swing_ee])
  clip.landing[swing_ee] = (
    float(np.linalg.norm(clip.anchors[swing_ee][0] - p2)),
    rotation_angle(clip.anchors[swing_ee][1], rot0),
  )


def _hand_pad_task(
  ctx: _CycleContext, swing_ee: str, t: int, n: int, arc: tuple[np.ndarray, np.ndarray, np.ndarray], rot0: np.ndarray
) -> IKTask:
  """Swing-hand pad target with the pad kept flat to the wall throughout.

  The released hand starts strongly constrained and fades to swing weights
  over the first 30% of the arc; an abrupt stiffness drop would release the
  limit-spring stress stored in the anchored arm as a one-frame pop.
  """
  s_arc = ctx.arc_progress(t, n)
  w_fade = min(s_arc / 0.3, 1.0)
  return IKTask(
    kind="site",
    sid=ctx.ik.pad_sid[swing_ee],
    pos=ctx.swing_target(t, n, *arc),
    rot=rot0,
    w_pos=np.full(3, 500.0 + 1500.0 * max(1.0 - w_fade, s_arc * s_arc)),
    w_rot=50.0 + 450.0 * max(1.0 - w_fade, s_arc * s_arc),
  )


def _hand_base_task(
  ctx: _CycleContext, swing_ee: str, first: bool, s: float, base_start: np.ndarray, base_rot_start: np.ndarray
) -> IKTask:
  """Base target: keep the wall press through the first hand, unwind it on the second."""
  recipe = ctx.recipe
  base_target_x = ctx.base_xy[0] + recipe.body_press if first else ctx.base_xy[0]
  support_side = -1.0 if swing_ee == ctx.ik.ee_names[0] else 1.0
  base_target = (1.0 - s) * base_start + s * np.array(
    [base_target_x, ctx.base_xy[1] + support_side * recipe.lateral_weight_shift, ctx.base_z1 + ctx.height]
  )
  return base_task(
    ctx.ik,
    base_target,
    (100.0, 30.0, 30.0) if first else (100.0, 30.0, 50.0),
    rot=slerp_matrix(base_rot_start, np.eye(3), s),
    w_rot=8.0 if first else 50.0,
  )


def _close_cycle(ctx: _CycleContext, clip: _ClipRecorder, state1_qpos: np.ndarray) -> tuple[float, float]:
  """Blend the last frames so state 5 is exactly state 1 one step higher.

  Returns the base (m) and joint (rad) correction the blend applied.
  """
  ik = ctx.ik
  state5_qpos = state1_qpos.copy()
  state5_qpos[2] += ctx.height
  delta = state5_qpos - clip.qpos[-1]
  closure_frames = max(int(round(0.4 * ctx.frames["right_hand_swing"])), 2)
  for i in range(closure_frames):
    idx = len(clip.qpos) - closure_frames + i
    blend = float(smootherstep((i + 1) / closure_frames))
    clip.qpos[idx] = clip.qpos[idx] + blend * delta
    clip.qpos[idx][3:7] /= np.linalg.norm(clip.qpos[idx][3:7])
  # The end hold sits exactly on the state-1 anchors shifted one step.
  ik.data.qpos[:] = state5_qpos
  mujoco.mj_forward(ik.model, ik.data)
  for ee in ik.ee_names:
    clip.anchor_now[ee] = ik.pad_pose(ee)[0]
  return float(np.linalg.norm(delta[0:3])), float(np.abs(delta[7:]).max())


@dataclass
class _TiledFrames:
  qpos: list[np.ndarray]
  labels: np.ndarray
  engaged: np.ndarray
  anchors: np.ndarray
  toe_braced: np.ndarray
  cycle_index: np.ndarray


def _tile_cycles(clip: _ClipRecorder, cycles: int, height: float, per_cycle: int) -> _TiledFrames:
  """Repeat the cycle ``cycles`` times, each one step higher; only the last keeps its end hold."""
  qpos, labels, engaged, anchors, toe, cycle = [], [], [], [], [], []
  z_shift = np.zeros(len(clip.qpos[0]))
  for k in range(cycles):
    z_shift[2] = k * height
    for i in range(per_cycle) if k < cycles - 1 else range(len(clip.qpos)):
      qpos.append(clip.qpos[i] + z_shift)
      labels.append(clip.labels[i])
      engaged.append(clip.engaged[i])
      anchors.append(clip.anchor_points[i] + np.array([0.0, 0.0, k * height]))
      toe.append(clip.toe_braced[i])
      cycle.append(k)
  return _TiledFrames(
    qpos=qpos,
    labels=np.asarray(labels, dtype=np.uint8),
    engaged=np.stack(engaged),
    anchors=np.stack(anchors).astype(np.float32),
    toe_braced=np.stack(toe),
    cycle_index=np.asarray(cycle, dtype=np.uint16),
  )


def _clip_arrays(ik: ClimbIK, tiled: _TiledFrames, fps: float, height: float) -> tuple[dict, dict[str, float]]:
  """Forward kinematics of every frame into the saved clip dict.

  Also returns the worst engaged-pad weld drift and swing-sphere wall
  penetration (m) as IK quality measures.
  """
  d = ik.data
  frames = len(tiled.qpos)
  joint_positions = np.zeros((frames, len(_JOINTS)), np.float32)
  root_position = np.zeros((frames, 3), np.float32)
  root_quaternion = np.zeros((frames, 4), np.float32)
  ee_positions = np.zeros((frames, 4, 3), np.float32)
  ee_quaternions = np.zeros((frames, 4, 4), np.float32)
  pad_positions = np.zeros((frames, 4, 3), np.float32)
  foot_positions = np.zeros((frames, 2, 3), np.float32)
  max_weld_drift, max_wall_pen = 0.0, -np.inf
  for t, q in enumerate(tiled.qpos):
    d.qpos[:] = q
    mujoco.mj_forward(ik.model, d)
    joint_positions[t] = [d.qpos[ik.qpos_addresses[nm]] for nm in _JOINTS]
    root_position[t] = q[0:3]
    root_quaternion[t] = (q[4], q[5], q[6], q[3])  # wxyz -> xyzw
    for i, ee in enumerate(ik.ee_names):
      bid = ik.ee_bid[ee]
      ee_positions[t, i] = d.xpos[bid]
      xq = d.xquat[bid]
      ee_quaternions[t, i] = (xq[1], xq[2], xq[3], xq[0])
      pad_positions[t, i] = d.site_xpos[ik.pad_sid[ee]]
      if tiled.engaged[t, i]:
        max_weld_drift = max(max_weld_drift, float(np.linalg.norm(pad_positions[t, i] - tiled.anchors[t, i])))
      else:
        max_wall_pen = max(max_wall_pen, _sphere_wall_penetration(ik, ee))
    for i, sid in enumerate(ik.foot_sid):
      foot_positions[t, i] = d.site_xpos[sid]
  data = {
    "joints_list": list(_JOINTS),
    "joint_positions": joint_positions,
    "root_position": root_position,
    "root_quaternion": root_quaternion,
    "foot_sites_list": list(FOOT_SITE_NAMES),
    "foot_positions": foot_positions,
    "fps": np.float32(fps),
    "ee_list": list(ik.ee_names),
    "ee_positions": ee_positions,
    "ee_quaternions": ee_quaternions,
    "pad_positions": pad_positions,
    "ee_engaged": tiled.engaged,
    "ee_anchor_positions": tiled.anchors,
    "toe_braced": tiled.toe_braced,
    "toe_list": list(ik.toes.keys()),
    "phase_segment": tiled.labels,
    "segment_names": list(SEGMENT_NAMES),
    "cycle_index": tiled.cycle_index,
    "step_height": np.float32(height),
    "wall_surface_x": np.float32(WALL_SURFACE_X),
    "pad_touch_x": np.float32(PAD_TOUCH_X),
  }
  return data, {"weld_drift": max_weld_drift, "wall_penetration": max_wall_pen}


def _sphere_wall_penetration(ik: ClimbIK, ee: str) -> float:
  """Deepest wall penetration (m) of a released pad's contact spheres or boxes."""
  m, d = ik.model, ik.data
  worst = -np.inf
  for bid in ik.sphere_bids[ee]:
    gid = int(m.body_geomadr[bid])
    if m.geom_type[gid] == mujoco.mjtGeom.mjGEOM_BOX:
      extent = np.abs(d.geom_xmat[gid].reshape(3, 3)[0]) @ m.geom_size[gid]
    else:
      extent = m.geom_size[gid, 0]
    worst = max(worst, float(d.geom_xpos[gid, 0] + extent - WALL_SURFACE_X))
  return worst


def _report(
  ctx: _CycleContext,
  clip: _ClipRecorder,
  data: dict,
  fk_quality: dict[str, float],
  closure: tuple[float, float],
  per_cycle: int,
) -> list[str]:
  """Human-readable quality summary: residuals, speeds, landings, toe, rake, limits."""
  fps, ik = ctx.recipe.fps, ctx.ik
  joint_positions, root_position = data["joint_positions"], data["root_position"]
  pad_positions, anchors, labels = data["pad_positions"], data["ee_anchor_positions"], data["phase_segment"]
  frames = len(joint_positions)
  lines = [
    f"{frames} frames, {frames / fps:.2f}s @ {fps:.1f} fps, {ctx.recipe.cycles} cycle(s), "
    f"step_height={ctx.height:.3f} m"
  ]
  for name in _SWINGS:
    r = clip.residual[name]
    lines.append(f"  {name:18s} IK residual: pos {r.pos * 1000.0:6.2f} mm, rot {np.degrees(r.rot):5.2f} deg")
  joint_speed = np.abs(np.diff(joint_positions, axis=0)) * fps
  peak_joint = int(np.unravel_index(np.argmax(joint_speed), joint_speed.shape)[1])
  base_speed = float(np.abs(np.diff(root_position, axis=0)).max() * fps)
  lines.append(
    f"  peak joint speed {joint_speed.max():.2f} rad/s ({_JOINTS[peak_joint]}), "
    f"peak base speed {base_speed:.3f} m/s, minimum engaged pads 3"
  )
  for ee, (pos_err, rot_err) in clip.landing.items():
    lines.append(f"  {ee:18s} landing vs scheduled anchor: {pos_err * 1000.0:6.2f} mm, {np.degrees(rot_err):5.2f} deg")
  for ee, rep in clip.toe_report.items():
    lines.append(
      f"  toe {ee:14s} hover gap {rep['plant_gap'] * 1000.0:5.2f} mm, "
      f"press wall gap max {rep['wall_gap'] * 1000.0:5.2f} mm, glide {rep['glide'] * 1000.0:5.2f} mm, "
      f"stance clear {ctx.stance_toe_clear[ee] * 1000.0:6.1f} mm"
    )
  lines += _rake_lines(ctx, pad_positions, anchors, labels, per_cycle)
  lines.append(
    f"  engaged-pad weld drift max      {fk_quality['weld_drift'] * 1000.0:6.2f} mm\n"
    f"  swing sphere wall penetration   {fk_quality['wall_penetration'] * 1000.0:+6.2f} mm "
    f"(<= 0 means clear of the wall)\n"
    f"  cycle closure correction        {closure[0] * 1000.0:6.2f} mm base, {closure[1]:.4f} rad joints"
  )
  lines += _joint_limit_lines(ik, joint_positions)
  if fk_quality["weld_drift"] > 5.0e-3 or fk_quality["wall_penetration"] > 2.0e-3:
    lines.append("  WARNING: IK quality below tolerance; inspect the render.")
  return lines


def _rake_lines(ctx: _CycleContext, pad_positions, anchors, labels, per_cycle: int) -> list[str]:
  """Downward on-wall stroke of each landing pad over its final rake frames."""
  lines = []
  n_rake = ctx.rake_frames
  for segment, ee_index in zip(_SWINGS, (2, 3, 0, 1)):
    frames_in_seg = np.where(labels == _SEGMENT_ID[segment])[0]
    last = frames_in_seg[frames_in_seg < per_cycle][-1]
    drop = float(pad_positions[last - n_rake, ee_index, 2] - pad_positions[last, ee_index, 2])
    off_wall = float(np.abs(pad_positions[last - n_rake : last + 1, ee_index, 0] - anchors[last, ee_index, 0]).max())
    lines.append(
      f"  rake {ctx.ik.ee_names[ee_index]:18s} ({segment}): down {drop * 1000.0:5.2f} mm, "
      f"off-anchor-plane max {off_wall * 1000.0:5.2f} mm"
    )
  return lines


def _joint_limit_lines(ik: ClimbIK, joint_positions: np.ndarray) -> list[str]:
  """Worst clearance to the hard stops; below 5% of range is flagged."""
  worst_name, worst_margin, flagged = "", np.inf, []
  for i, nm in enumerate(_JOINTS):
    lo, hi = ik.joint_ranges[nm]
    jmin, jmax = float(joint_positions[:, i].min()), float(joint_positions[:, i].max())
    margin = min(jmin - lo, hi - jmax) / (hi - lo)
    if margin < worst_margin:
      worst_name, worst_margin = nm, margin
    if margin < 0.05:
      flagged.append(
        f"  LOW LIMIT MARGIN: {nm} [{jmin:+.3f}, {jmax:+.3f}] range [{lo:+.3f}, {hi:+.3f}] margin {margin * 100.0:.1f}%"
      )
  return [f"  joint-limit clearance min       {worst_margin * 100.0:6.2f}% of range ({worst_name})", *flagged]
