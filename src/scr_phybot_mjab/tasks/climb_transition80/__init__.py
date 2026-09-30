"""80-degree walk->climb transition task: BeyondMimic-minimal reward, termination-first.

A PARALLEL task package, sibling of ``climb_wall`` and ``climb_transition``
and fully DECOUPLED from the latter: the only task-level dependencies are
``climb_wall`` (base builder, climb motion command/clip loader, climb mdp)
and ``assets``. Everything the one-shot prologue needs (acyclic engage
tables, terminal clamp, blind-delivery spawn scatter, RSI bias window,
schedulers, observation lanes, the contact law) lives here as its own
copy, so the two transition generations can evolve independently.

Design (mentor call, 2026-08-21): BeyondMimic's six exp-kernel tracking
terms + one staged engage-progress potential + two regularisers; every
illegal or wrong-engage state is a TERMINATION (Constraints-as-Terminations
for the engage cone); adaptive RSI sampling; the joint PD table is written
out explicitly and equals the real robot's motor firmware table.
"""

import scr_phybot_mjab.tasks.climb_transition80.config.phybot_c2  # noqa: F401
