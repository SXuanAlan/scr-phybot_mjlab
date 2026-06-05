"""Debug: check MuJoCo contacts between hand and wall."""
import traceback
import torch
import scr_phybot_mjab.tasks  # noqa: F401
from scr_phybot_mjab.tasks.microspine_test.config.forearm_hand.env_cfgs import (
    forearm_hand_microspine_test_env_cfg,
)
from mjlab.envs import ManagerBasedRlEnv

try:
    cfg = forearm_hand_microspine_test_env_cfg(play=False)
    cfg.scene.num_envs = 1
    cfg.observations["actor"].enable_corruption = False
    cfg.episode_length_s = 10.0

    env = ManagerBasedRlEnv(cfg=cfg, device="cuda:0")
    obs, _ = env.reset()
    mj_model = env.sim.mj_model

    print(f"PAIRS: {mj_model.npair}", flush=True)
    for i in range(mj_model.npair):
        pair = mj_model.pair(i)
        g1n = mj_model.geom(int(pair.geom1[0])).name
        g2n = mj_model.geom(int(pair.geom2[0])).name
        print(f"  {g1n} <-> {g2n}", flush=True)

    # Run a few steps
    for step_i in range(5):
        actions = torch.zeros(1, 1, device="cuda:0")
        obs, rewards, terminated, truncated, info = env.step(actions)
        print(f"step {step_i}: done", flush=True)

    # Check what contact data is available
    data = env.sim.data
    print(f"ncon type: {type(data.ncon)}", flush=True)
    print(f"ncon: {data.ncon}", flush=True)

    env.close()
except Exception:
    traceback.print_exc()
