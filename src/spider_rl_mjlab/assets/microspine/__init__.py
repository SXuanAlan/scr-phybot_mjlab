"""Microspine adhesion model for wall climbing.

Combines MuJoCo's native adhesion actuators with a stochastic spine
engagement model (Jiang et al. 2018).
"""

from spider_rl_mjlab.assets.microspine.actuators import add_adhesion_actuators as add_adhesion_actuators
from spider_rl_mjlab.assets.microspine.config import MicrospineCfg as MicrospineCfg
from spider_rl_mjlab.assets.microspine.controller import MicrospineController as MicrospineController
