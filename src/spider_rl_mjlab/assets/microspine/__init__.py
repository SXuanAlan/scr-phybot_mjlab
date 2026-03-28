"""Microspine adhesion model for wall climbing.

Stochastic spine-asperity interaction (Jiang et al. 2018) with
constraint-based anisotropic friction via MuJoCo Warp contactfilter.
"""

from spider_rl_mjlab.assets.microspine.actuators import add_adhesion_actuators
from spider_rl_mjlab.assets.microspine.config import MicrospineCfg
from spider_rl_mjlab.assets.microspine.controller import MicrospineController

__all__ = [
    "MicrospineCfg",
    "MicrospineController",
    "add_adhesion_actuators",
]
