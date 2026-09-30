import pytest

from scr_phybot_mjab.microspine.config import MicrospineCfg

EE_NAMES = ("left_hand", "right_hand", "left_foot", "right_foot")


def make_cfg(**kwargs) -> MicrospineCfg:
  return MicrospineCfg(
    ee_body_names=EE_NAMES,
    spheres_per_ee_rows=2,
    spheres_per_ee_cols=2,
    **kwargs,
  )


@pytest.mark.parametrize("initial_on", [False, True])
def test_initial_mask_preserves_legacy_all_on_off(initial_on: bool):
  cfg = make_cfg(initial_on=initial_on)
  assert cfg.resolve_initial_engaged_mask() == (initial_on,) * len(EE_NAMES)


def test_initial_mask_can_select_hands_only():
  cfg = make_cfg(
    initial_on=True,
    initial_engaged_ee_names=("left_hand", "right_hand"),
  )
  assert cfg.resolve_initial_engaged_mask() == (True, True, False, False)


def test_initial_mask_rejects_unknown_ee_name():
  cfg = make_cfg(initial_engaged_ee_names=("left_hand", "tail"))
  with pytest.raises(ValueError, match="tail"):
    cfg.resolve_initial_engaged_mask()


def test_microspine_grip_ablation_defaults_off_only_when_requested():
  enabled = make_cfg()
  disabled = make_cfg(grip_enabled=False)

  assert enabled.grip_enabled
  assert not disabled.grip_enabled
