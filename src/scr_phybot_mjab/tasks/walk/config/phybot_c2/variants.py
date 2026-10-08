"""The three tasks built from the walking configuration.

- ``walk``: velocity tracking with plain PPO; a posture reward is the only
  gait-style prior.
- ``stand``: the zero command. The policy recovers from the reset
  perturbation, then keeps its footholds and the default pose. It is the
  deployed "prepare" policy, which hands a standing robot to walk or squat.
- ``amp``: preset gait commands (stand, slow, walk, turns) with an AMP style
  reward, resets into the AMP clips and the deployed PD damping.
"""

from typing import Literal

WalkVariant = Literal["walk", "stand", "amp"]
