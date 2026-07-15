import torch

from scr_phybot_mjab.tasks.climb_wall.mdp.commands import (
  ClimbGaitPhase,
  resolve_climb_gait_phase,
)


def test_climb_gait_phase_is_driven_by_release_contact_and_regrip_events():
  phase = torch.tensor(
    [
      ClimbGaitPhase.RELEASE,
      ClimbGaitPhase.SWING,
      ClimbGaitPhase.SEAT,
      ClimbGaitPhase.SEAT,
      ClimbGaitPhase.TRANSFER,
    ]
  )
  engaged = torch.tensor([False, False, True, False, False])
  contact = torch.tensor([False, True, True, False, False])

  result = resolve_climb_gait_phase(phase, engaged, contact)

  assert result.tolist() == [
    ClimbGaitPhase.SWING,
    ClimbGaitPhase.SEAT,
    ClimbGaitPhase.TRANSFER,
    ClimbGaitPhase.SWING,
    ClimbGaitPhase.SWING,
  ]


def test_climb_gait_does_not_release_an_engaged_target_by_command():
  phase = torch.tensor([ClimbGaitPhase.RELEASE])
  result = resolve_climb_gait_phase(
    phase,
    target_engaged=torch.tensor([True]),
    target_contact=torch.tensor([True]),
  )
  assert result.item() == ClimbGaitPhase.RELEASE
