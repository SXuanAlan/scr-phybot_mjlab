"""Independent physics and single-unit state contracts for ToddlerBot."""

import hashlib
import json
from types import SimpleNamespace

import mjlab  # noqa: F401
import mujoco
import numpy as np
import pytest
import torch

from scr_phybot_mjab.assets.toddlerbot.actuator import ToddlerServo
from scr_phybot_mjab.assets.toddlerbot.toddlerbot_constants import (
    CORNER_SPEEDS,
    MOTOR_NAMES,
    MOUNT_SIGNATURE,
    ROOT,
    SPEED_LIMITS,
    TORQUE_AT_MAX_SPEED,
    TORQUE_LIMITS,
    UNIT_NAMES,
    speed_torque_limits,
)
from scr_phybot_mjab.tasks.toddler_climb.command import ToddlerClimbCommand, ToddlerClimbCommandCfg
from scr_phybot_mjab.tools.climbing_reference.toddlerbot import (
    REFERENCE_PATH,
    ToddlerReferenceModel,
    load_reference,
    wall_rotation,
)


@pytest.fixture(scope="module")
def rig():
    return ToddlerReferenceModel()


def test_import_is_pinned_and_one_rigid_unit_per_limb(rig):
    provenance = json.loads((ROOT / "provenance.json").read_text())
    for name, entry in provenance["files"].items():
        assert hashlib.sha256((ROOT / "upstream" / name).read_bytes()).hexdigest() == entry["sha256"]
    m = rig.model
    assert (m.nu, m.nq, m.nv) == (30, 51, 50)
    assert len([m.body(i).name for i in range(m.nbody) if m.body(i).name.endswith("_unit")]) == 4
    assert len([m.geom(i).name for i in range(m.ngeom) if m.geom(i).name.endswith("_c_0_0")]) == 4
    assert all(m.body_jntnum[m.body(name).id] == 0 for name in UNIT_NAMES)
    np.testing.assert_allclose(m.body_mass.sum(), 3.76887188 + 4 * 0.015)
    np.testing.assert_allclose(m.actuator_forcerange[:, 1], TORQUE_LIMITS)
    assert TORQUE_LIMITS[MOTOR_NAMES.index("left_hip_pitch")] == 2.95
    assert TORQUE_LIMITS[MOTOR_NAMES.index("left_wrist_pitch_drive")] == 1.09


def test_units_attach_to_limb_surfaces_and_bypass_wrist_and_ankle(rig):
    m = rig.model
    parents = ("left_knee_link", "right_knee_link", "left_elbow_yaw_link", "right_elbow_yaw_link")
    for i, (name, parent) in enumerate(zip(UNIT_NAMES, parents)):
        b = m.body(name)
        assert m.body_parentid[b.id] == m.body(parent).id
        R = np.zeros(9)
        mujoco.mju_quat2Mat(R, b.quat)
        R = R.reshape(3, 3)
        np.testing.assert_allclose(-R[:, 1], [1 if i < 2 else -1, 0, 0], atol=1e-8)
        np.testing.assert_allclose(-R[:, 2], [0, 0, -1 if i < 2 else 1], atol=1e-8)
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    before = d.site_xpos[rig.site_ids].copy()
    for side in ("left", "right"):
        for suffix in ("ankle_pitch", "ankle_roll", "wrist_pitch_driven", "wrist_roll"):
            d.qpos[m.joint(f"{side}_{suffix}").qposadr] += 0.2
    mujoco.mj_forward(m, d)
    np.testing.assert_allclose(d.site_xpos[rig.site_ids], before, atol=1e-12)


def test_reference_rejects_old_mount_layout(tmp_path):
    ref = load_reference(REFERENCE_PATH.with_name("toddlerbot_climb_a60.npz"))
    ref["mount_signature"] = np.asarray("old palm and toe mounts")
    path = tmp_path / "wrong_mounts.npz"
    np.savez_compressed(path, **ref)
    with pytest.raises(ValueError, match="mounts differ"):
        load_reference(path)


@pytest.mark.parametrize(
    "name,expected",
    [("toddlerbot_climb_a60.npz", True), ("toddlerbot_climb_a80.npz", True), ("toddlerbot_climb.npz", False)],
)
def test_reference_audit_is_bound_to_data_and_preserves_failure(name, expected):
    path = REFERENCE_PATH.with_name(name)
    ref = load_reference(path)
    audit = json.loads(path.with_suffix(".audit.json").read_text())
    assert hashlib.sha256(path.read_bytes()).hexdigest() == audit["reference_sha256"]
    assert audit["mount_signature"] == MOUNT_SIGNATURE == str(ref["mount_signature"])
    assert audit["passed"] is expected
    assert audit["geometry_pass"]
    assert not audit["dynamic_execution_validated"]
    assert ref["qpos"].shape == (351, 51)
    assert np.all(ref["engaged"].sum(axis=1) >= 3)
    assert np.all((~ref["engaged"]).any(axis=0))
    np.testing.assert_allclose(ref["qpos"][-1, 7:], ref["qpos"][0, 7:], atol=0.001)
    np.testing.assert_allclose(ref["qpos"][-1, :3] - ref["qpos"][0, :3], ref["cycle_delta"], atol=1e-4)
    if expected:
        assert ref["static_pass"].all()
        assert audit["preload_pass_frames"] == audit["preload_frames"] > 0


@pytest.mark.parametrize("angle", [60, 80])
def test_nominal_force_solution_balances_full_robot_and_respects_each_motor(rig, angle):
    ref = load_reference(REFERENCE_PATH.with_name(f"toddlerbot_climb_a{angle}.npz"))
    m, d = rig.model, rig.data
    R = wall_rotation(angle).as_matrix()
    frames = (0, 32, 52, 110, 130, 185, 206, 260, 280, 350)
    if angle == 80:
        frames += (257, 258, 264, 265, 267, 269, 271, 273)
    for frame in frames:
        d.qpos[:] = ref["qpos"][frame]
        d.qvel[:] = 0
        mujoco.mj_forward(m, d)
        force = ref["forces_tln"][frame]
        normal = R @ np.array([-1.0, 0.0, 0.0])
        world = []
        for sid, f in zip(rig.site_ids, force):
            rake = -d.site_xmat[sid].reshape(3, 3)[:, 2]
            rake -= normal * (rake @ normal)
            rake /= np.linalg.norm(rake)
            world.append(-f[0] * rake - f[1] * np.cross(normal, rake) - f[2] * normal)
        world = np.asarray(world)
        # Whole-body force and moment balance, independent of the optimizer's nullspace.
        np.testing.assert_allclose(world.sum(axis=0), -m.body_mass.sum() * m.opt.gravity, atol=1e-4)
        moment = np.cross(d.site_xpos[rig.site_ids] - d.subtree_com[0], world).sum(axis=0)
        np.testing.assert_allclose(moment, 0, atol=1e-4)
        residual = -d.qfrc_bias.copy()
        residual[rig.motor_dofs] += ref["motor_torque"][frame]
        for sid, f in zip(rig.site_ids, world):
            J = np.zeros((3, m.nv))
            mujoco.mj_jacSite(m, d, J, None, int(sid))
            residual += J.T @ f
        eq = d.efc_J.reshape(-1, m.nv)[: d.ne]
        reaction = np.linalg.lstsq(eq.T, -residual, rcond=1e-7)[0]
        np.testing.assert_allclose(residual + eq.T @ reaction, 0, atol=1e-4)
        assert np.all(np.abs(ref["motor_torque"][frame]) <= TORQUE_LIMITS + 1e-5)
        on = ref["engaged"][frame]
        cap, _, valid, _ = rig.table.query_numpy(force[on, 0], force[on, 1], 0.5)
        assert valid.all() and np.all(force[on, 2] <= cap + 1e-4)
        press = int(ref["press_unit"][frame])
        if press >= 0:
            np.testing.assert_allclose(force[press], [0, 0, -1.5], atol=1e-6)


def test_static_solver_reuses_a_neighboring_pose_force_solution(rig):
    """Regression: a cold LP seed stalled at the 80-degree right-leg swing."""
    ref = load_reference(REFERENCE_PATH.with_name("toddlerbot_climb_a80.npz"))
    frame, neighbor = 257, 259
    rig.state(ref["x"][frame], 80)
    prior = np.r_[ref["forces_tln"][neighbor].ravel(), ref["motor_torque"][neighbor]]
    result = rig.statics(ref["engaged"][frame], 80, initial=prior)
    assert result["pass"]
    assert result["balance_residual"] < 1e-7
    assert result["normal_margin_n"] >= -1e-7
    assert result["peak_motor_fraction"] < 1.0
    forces = np.asarray(result["forces_tln"])
    np.testing.assert_array_equal(forces[~ref["engaged"][frame]], 0)
    assert np.all(np.abs(result["motor_torque"]) <= TORQUE_LIMITS + 1e-7)


def test_torque_speed_bound_matches_audit_and_braking_is_bounded():
    speeds = np.stack([np.zeros(30), CORNER_SPEEDS, SPEED_LIMITS, SPEED_LIMITS * 1.01])
    expected = np.stack([TORQUE_LIMITS, TORQUE_LIMITS, TORQUE_AT_MAX_SPEED, np.zeros(30)])
    np.testing.assert_allclose(speed_torque_limits(speeds), expected)
    for j in range(30):
        servo = object.__new__(ToddlerServo)
        servo.cfg = SimpleNamespace(
            corner_speed=CORNER_SPEEDS[j], max_speed=SPEED_LIMITS[j], torque_at_max_speed=TORQUE_AT_MAX_SPEED[j]
        )
        servo.force_limit = torch.tensor(TORQUE_LIMITS[j])
        servo.speed = torch.tensor(speeds[:, j])
        for sign in (-1, 1):
            result = servo._clip_effort(torch.full((4,), sign * 100.0))
            np.testing.assert_allclose(result.numpy(), sign * expected[:, j], atol=1e-7)


@pytest.fixture
def command(rig):
    """Real command and empirical table, controlled sensor/constraint inputs."""
    archive = load_reference(REFERENCE_PATH.with_name("toddlerbot_climb_a60.npz"))
    m, d = rig.model, rig.data
    d.qpos[:] = archive["qpos"][0]
    mujoco.mj_forward(m, d)
    robot = SimpleNamespace(
        find_joints=lambda names, preserve_order: ([m.joint(n).id - 1 for n in names], names),
        find_sites=lambda names, preserve_order: ([m.site(n).id for n in names], names),
        find_bodies=lambda names, preserve_order: ([m.body(n).id for n in names], names),
        data=SimpleNamespace(
            body_link_quat_w=torch.tensor(d.xquat, dtype=torch.float32)[None].repeat(2, 1, 1),
            site_pos_w=torch.tensor(d.site_xpos, dtype=torch.float32)[None].repeat(2, 1, 1),
        ),
    )
    contact = SimpleNamespace(
        primary_names=[u + "_c_0_0" for u in UNIT_NAMES],
        data=SimpleNamespace(force=torch.zeros(2, 4, 3), force_history=torch.zeros(2, 4, 10, 3)),
    )

    class Scene(dict):
        env_origins = torch.zeros(2, 3)

    scene = Scene(robot=robot, unit_contact=contact)
    anchor = torch.zeros(2, 4, 3)
    grip = SimpleNamespace(compute_anchor_loads=lambda: anchor)
    env = SimpleNamespace(
        num_envs=2,
        device="cpu",
        scene=scene,
        event_manager=SimpleNamespace(get_term_cfg=lambda name: SimpleNamespace(func=grip)),
    )
    c = ToddlerClimbCommand(
        ToddlerClimbCommandCfg(
            reference_file=str(REFERENCE_PATH.with_name("toddlerbot_climb_a60.npz")), wall_angle_deg=60
        ),
        env,
    )
    c.fresh[:] = False
    c._dt = 0.02
    c.age[:] = 10
    c.strength[:] = 0.5
    c.loading[:] = 1.0
    # Hold all four for these sensor tests; keep the real interpolation for a
    # separate cycle test and the real CPU/GPU integration validator.
    c.sample = lambda key: torch.ones(2, 4, dtype=torch.bool) if key == "engaged" else torch.full((2,), -1)
    return c, contact.data, anchor


def test_preload_requires_each_unit_and_three_steps_then_draw_is_persistent(command):
    c, sensor, _ = command
    pressures = torch.tensor([[1.49, 1.5, 2.0, 0.0], [1.5, 1.5, 1.5, 1.5]])
    sensor.force[:] = -pressures[:, :, None] * c.normal
    sensor.force_history[:] = sensor.force[:, :, None]
    saved = c.strength.clone()
    for _ in range(2):
        c._update_command()
    assert not c.engaged.any()
    c._update_command()
    torch.testing.assert_close(c.engaged, pressures >= 1.5)
    assert torch.equal(c.strength[~c.engaged], saved[~c.engaged])
    c.loading[:] = 1.0
    saved = c.strength.clone()
    for _ in range(4):
        c._update_command()
    torch.testing.assert_close(c.strength, saved)


def test_tension_cannot_preload_and_hold_overload_releases_one_unit(command):
    c, sensor, anchor = command
    sensor.force[:] = 2 * c.normal
    sensor.force_history[:] = sensor.force[:, :, None]
    for _ in range(4):
        c._update_command()
    assert not c.engaged.any()
    sensor.force.zero_()
    sensor.force_history.zero_()
    c.engaged[:] = True
    # 100 N pull is beyond the empirical normal capacity, applied to one unit.
    anchor[0, 2] = 100 * c.normal
    c._update_command()
    assert not c.engaged[0, 2]
    assert c.engaged.sum() == 7
    assert c.failed_grips.tolist() == [1.0, 0.0]


def test_reference_cycle_and_environment_offsets_are_independent(command):
    c, _, _ = command
    c._env.scene.env_origins = torch.tensor([[0.0, 0.0, 0.0], [3.0, 2.0, 0.0]])
    c.elapsed[:] = torch.tensor([0.0, c.duration])
    root = ToddlerClimbCommand.sample(c, "qpos")
    units = ToddlerClimbCommand.sample(c, "unit_pos")
    delta = c.reference["cycle_delta"] + c._env.scene.env_origins[1]
    torch.testing.assert_close(root[1, :3] - root[0, :3], delta)
    torch.testing.assert_close(root[1, 3:], root[0, 3:])
    torch.testing.assert_close(units[1] - units[0], delta.expand(4, 3))
    c.elapsed[:] = 0.5 / c.fps
    motors = ToddlerClimbCommand.sample(c, "motor_pos")
    expected = (c.reference["motor_pos"][0] + c.reference["motor_pos"][1]) / 2
    torch.testing.assert_close(motors, expected.expand(2, -1))
