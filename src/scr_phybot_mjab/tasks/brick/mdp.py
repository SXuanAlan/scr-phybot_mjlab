"""Rewards, privileged diagnostics and completion for the brick task."""

import torch

from .command import quaternion_error


def task(env):
  return env.command_manager.get_term("brick_task")


def grip(env):
  return env.command_manager.get_term("microspine_affordance")


def tracking(env):
  c = task(env)
  error = c.robot.data.joint_pos[:, c.joint_ids] - c.joint_pos_ref
  root = c.ref("root_pose")
  angle = quaternion_error(c.robot.data.root_link_quat_w, root[:, 3:] / root[:, 3:].norm(dim=-1, keepdim=True))
  return torch.exp(-error.square().mean(-1) / 0.12) + 0.5 * torch.exp(-angle.square() / 0.12)


def object_tracking(env):
  c = task(env)
  pos = c.brick.data.root_link_pos_w
  target = c.ref("brick_pos") + env.scene.env_origins
  distance = (pos - target).square().sum(-1)
  # Before lifting, reward approach with the two hand centers. Ground pushing
  # cannot earn carry progress because the measured lift gate must complete.
  hands = c.nominal_hands_w()
  hand_target = c.ref("hand_pos") + env.scene.env_origins[:, None]
  reach = torch.exp(-(hands - hand_target).square().sum(-1).mean(-1) / 0.01)
  return torch.where(c.phase < 2, reach, torch.exp(-distance / 0.008))


def milestone(env):
  return task(env).milestone / env.step_dt


def completion(env):
  c = task(env)
  completed = c.success if c.cfg.training_stage == "full" else c.training_complete
  return (completed & ~failed(env)).float() / env.step_dt


def grip_quality(env):
  c, g = task(env), grip(env)
  carrying = (c.phase >= 2) & (c.phase <= 4)
  preload = (g.unit_preload_n / max(g._mcfg.engage_unit_preload_n, 1.0)).clamp(0, 1).mean((1, 2))
  return carrying * g.engaged.float().mean(-1) + (c.phase == 1) * preload


def squeeze(env):
  c, g = task(env), grip(env)
  compression = torch.where(g.engaged, (-g.anchor_load_tnl[..., 2]).clamp_min(0), g.pad_press_force).sum(-1)
  # Less than a quarter of nominal weight can be supplied by Coulomb friction
  # at this compression budget. Initial seating is explicitly exempt.
  mass = env.sim.model.body_mass[:, env.sim.mj_model.body("brick/brick").id]
  budget = 0.25 * mass * 9.81 / 0.2
  carry = ((c.phase == 2) & (c.elapsed > 0.5)) | (c.phase == 3)
  return carry * ((compression - budget).clamp_min(0) / 20.0).square().clamp(max=25.0)


def overload(env):
  c, g = task(env), grip(env)
  return ((c.phase >= 2) & (c.phase <= 4)) * (-g.hold_margin).clamp(0, 2).sum(-1)


def slip(env):
  c, g = task(env), grip(env)
  executor = env.event_manager.get_term_cfg("microspine_grip").func
  point = executor.pad_center_world()
  pos = c.robot.data.body_link_pos_w[:, g._ee_body_ids]
  velocity = c.robot.data.body_link_lin_vel_w[:, g._ee_body_ids] + torch.cross(
    c.robot.data.body_link_ang_vel_w[:, g._ee_body_ids], point - pos, dim=-1
  )
  relative = velocity - executor._surfaces.velocity_at(point)
  normal = executor._surfaces.frame()[1]
  tangent = relative - (relative * normal).sum(-1, keepdim=True) * normal
  carrying = (c.phase >= 2) & (c.phase <= 4)
  return carrying * (tangent.square().sum(-1) * g.engaged).sum(-1).clamp(max=1.0)


def foot_drift(env):
  c = task(env)
  return (c.robot.data.site_pos_w[:, c.foot_ids] - c.foot_anchors).square().sum((1, 2))


def contact_force(env, sensor_name):
  data = env.scene[sensor_name].data
  force = data.force
  if data.force_history is not None:
    return torch.linalg.vector_norm(data.force_history, dim=-1).flatten(1).amax(-1)
  return torch.linalg.vector_norm(force, dim=-1).flatten(1).amax(-1)


def illegal_contact(env):
  return (contact_force(env, "illegal_brick_contact") > 5.0) | (contact_force(env, "illegal_ground_contact") > 5.0)


def failed(env):
  c = task(env)
  # No stand-height/upright cutoff: valid floor pickup involves deep leaning.
  root_z = c.robot.data.root_link_pos_w[:, 2] - env.scene.env_origins[:, 2]
  foot_slip = torch.linalg.vector_norm(c.robot.data.site_pos_w[:, c.foot_ids] - c.foot_anchors, dim=-1).amax(-1) > 0.06
  below_floor = c.brick.data.root_link_pos_w[:, 2] < env.scene.env_origins[:, 2]
  dropped = c.lifted & (c.phase < 4) & (c.brick.data.root_link_pos_w[:, 2] - c.initial_brick[:, 2] < 0.025)
  return (root_z < 0.20) | foot_slip | below_floor | dropped | illegal_contact(env)


def finished(env):
  c = task(env)
  completed = c.success if c.cfg.training_stage == "full" else c.training_complete
  return completed & ~failed(env)


def privileged_state(env):
  c, g = task(env), grip(env)
  mass = env.sim.model.body_mass[:, env.sim.mj_model.body("brick/brick").id]
  return torch.cat(
    (
      c.brick.data.root_link_pos_w - c.goal[:, :3],
      c.brick.data.root_link_quat_w,
      c.brick.data.root_link_lin_vel_w,
      c.brick.data.root_link_ang_vel_w,
      c.robot.data.root_link_lin_vel_w,
      c.robot.data.joint_pos[:, c.gimbal_ids],
      g.engaged.float(),
      g.anchor_load_tnl.flatten(1) * 0.01,
      g.anchor_moment_tnl.flatten(1),
      g.hold_margin,
      g.unit_strength_quantile.flatten(1),
      mass.expand(env.num_envs)[:, None],
    ),
    -1,
  )
