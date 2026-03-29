"""Quick diagnostic: check adhesion state during first few steps."""
import torch
import spider_rl_mjlab.tasks  # noqa: F401
from spider_rl_mjlab.tasks.microspine_test.config.forearm_hand.env_cfgs import (
    forearm_hand_microspine_test_env_cfg,
)
from mjlab.envs import ManagerBasedRlEnv

cfg = forearm_hand_microspine_test_env_cfg(play=False)
cfg.scene.num_envs = 1
cfg.observations["actor"].enable_corruption = False
cfg.episode_length_s = 10.0

env = ManagerBasedRlEnv(cfg=cfg, device="cuda:0")
obs, _ = env.reset()

robot = env.scene["robot"]
mj_model = env.sim.mj_model

# Print actuator info
print(f"\nACTUATOR INFO: nu={mj_model.nu}")
for i in range(mj_model.nu):
    act = mj_model.actuator(i)
    print(f"  [{i}] {act.name} gainprm={act.gainprm[:3]}")

# Find adhesion actuator index
adhesion_idx = mj_model.actuator("robot/adhesion_right_hand").id
print(f"Adhesion actuator at index: {adhesion_idx}")

# Find hand body
hand_body_id = None
for i in range(mj_model.nbody):
    b = mj_model.body(i)
    if "right_hand" in b.name and "elbow" not in b.name:
        hand_body_id = i
        print(f"Hand body: [{i}] {b.name}")

print(f"\nenv_origins[0]: {env.scene.env_origins[0].tolist()}")

wall_entity = env.scene["wall"]
wall_geom = mj_model.geom("wall/wall_geom")
wall_half_x = wall_geom.size[0]
wall_pos_x = wall_entity.cfg.init_state.pos[0]
print(f"Wall pos_x={wall_pos_x}, half_x={wall_half_x}, face_x={wall_pos_x - wall_half_x}")

print(f"\nSIMULATION STEPS:")
for step_i in range(30):
    actions = torch.zeros(1, 1, device="cuda:0")
    obs, rewards, terminated, truncated, info = env.step(actions)

    t = (step_i + 1) * env.step_dt
    root_pos = robot.data.root_link_pos_w[0]
    ee_pos = robot.data.body_link_pos_w[0]
    ctrl = env.sim.data.ctrl[0]
    adh_ctrl = ctrl[adhesion_idx].item()

    print(f"  t={t:.3f}s root={root_pos[:3].tolist()} "
          f"ctrl=[adh={adh_ctrl:.4f}, elbow={ctrl[1-adhesion_idx].item():.4f}]")

# Now run to phase 2
print(f"\nRUNNING TO PHASE 2...")
for step_i in range(30, 650):
    actions = torch.zeros(1, 1, device="cuda:0")
    obs, rewards, terminated, truncated, info = env.step(actions)
    t = (step_i + 1) * env.step_dt
    if step_i % 50 == 0 or (step_i >= 595 and step_i <= 615):
        root_pos = robot.data.root_link_pos_w[0]
        ctrl = env.sim.data.ctrl[0]
        adh_ctrl = ctrl[adhesion_idx].item()
        done = (terminated | truncated).any().item()
        print(f"  t={t:.3f}s root_z={root_pos[2].item():.4f} adh_ctrl={adh_ctrl:.4f} done={done}")
    if (terminated | truncated).any():
        print(f"  TERMINATED at step {step_i+1}")
        break

env.close()
