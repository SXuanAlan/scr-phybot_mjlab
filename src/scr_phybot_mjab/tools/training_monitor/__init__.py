"""Monitoring tools for a live-curriculum climbing run.

- ``metrics``: a read-only digest of the run's training scalars.
- ``supervisor``: examines settled checkpoints and holds a curriculum stage
  until its exams pass and a person has reviewed them.
- ``runner``: a PPO runner that saves the curriculum state in its checkpoints.
"""
