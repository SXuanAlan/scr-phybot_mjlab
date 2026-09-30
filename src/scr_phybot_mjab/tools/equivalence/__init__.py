"""Behaviour snapshots used to prove a refactor changes nothing.

A snapshot directory holds four tiers, each comparable on its own:

  cfg      canonical text dump of every kept task's env/rl config
  model    fingerprint of the compiled MuJoCo model per task
  rollout  fixed-seed rollouts (obs, per-term rewards, terminations, command
           state, sim state) under zero, random and reference actions
  clips    motion clips regenerated from their recipes

``scripts/dev/equivalence.py`` records and compares snapshots.
"""
