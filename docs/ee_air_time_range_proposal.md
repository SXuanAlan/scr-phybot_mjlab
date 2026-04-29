# `ee_capped_disengage_time` Reward Design

> Design note only. The active C2 climb-wall config has been restored to the
> 2026-04-28_14-51-31 `ee_grip_time` gait reward; this capped disengage-time
> variant is kept here as review context.

## Goal

Replace the previous dense `ee_grip_time` gait shaping with a reward closer to
this repo's walking `feet_air_time` style, while using the capped
disengage-time version for the first training trial:

- reward an EE while it is in a reasonable swing / disengaged-time window;
- do not reward an EE for staying engaged forever;
- do not reward an EE for staying disengaged forever;
- optionally require the global adhesion pattern to stay in the stable
  two-anchor / two-swing regime.

This is intentionally denser than IsaacLab's official quadruped
`feet_air_time`, which pays only on first contact after an air phase.

## Symbols

Let:

- $e_{t,h} \in \{0, 1\}$ be whether end-effector $h$ is engaged at step $t$.
- $H = 4$ for PHYBot C2's four spine end-effectors.
- $\Delta t$ be `env.step_dt`.
- $a_{t,h}$ be the current continuous disengaged time for EE $h$.
- $t_{\min}$ be the lower air-time threshold.
- $t_{\max}$ be the upper air-time threshold.
- $\mathbf{v}^{cmd}_t = (v_x^{cmd}, v_y^{cmd}, v_z^{cmd})$ be the live
  `climb_velocity` command.
- $\varepsilon_v$ be the command activation threshold.
- $N_{\text{eng}}$ be the optional target number of engaged EEs.

Proposed default parameters:

| Parameter | Value | Meaning |
|---|---:|---|
| $t_{\min}$ | 0.05 s | ignore tiny detach jitter |
| $t_{\max}$ | 0.50 s | do not reward being disengaged too long |
| $\varepsilon_v$ | 0.01 m/s | reward active only when commanded to climb |
| $N_{\text{eng}}$ | 2 | keep two-anchor / two-swing structure |
| reward weight | 1.5 | initial conservative value |

## Detailed Reward Formula

This section writes the proposed reward exactly as it would be computed for
one environment at one control step.

### Step 1: update the disengage timer

For each EE $h$, define $a_{t,h}$ as **how long this EE has continuously been
disengaged**.

$$
a_{t,h} =
\begin{cases}
a_{t-1,h} + \Delta t, & e_{t,h} = 0 \quad \text{(disengaged)} \\
0, & e_{t,h} = 1 \quad \text{(engaged)}
\end{cases}
$$

At episode reset:

$$
a_{0,h} = 0
$$

Important: this is not grip time. It only increases while the EE is
disengaged. As soon as the EE engages again, its timer resets to 0.

### Step 2: compute per-EE swing-window reward

Each EE contributes 1 raw reward only if it is disengaged and its disengaged
timer is inside the valid swing window:

$$
r^{ee}_{t,h}
= (1 - e_{t,h})\,
  \mathbb{1}\left[t_{\min} < a_{t,h} < t_{\max}\right]
$$

Equivalently, for one EE:

$$
r^{ee}_{t,h} =
\begin{cases}
1, & e_{t,h}=0 \text{ and } t_{\min}<a_{t,h}<t_{\max} \\
0, & \text{otherwise}
\end{cases}
$$

With the proposed defaults:

$$
t_{\min}=0.05\text{ s}, \quad t_{\max}=0.50\text{ s}
$$

So:

- engaged EE: $r^{ee}_{t,h}=0$;
- disengaged for 0.02 s: $r^{ee}_{t,h}=0$;
- disengaged for 0.20 s: $r^{ee}_{t,h}=1$;
- disengaged for 0.70 s: $r^{ee}_{t,h}=0$.

### Step 3: command gate

Only activate the gait reward when the climb command is nonzero:

$$
g^{cmd}_{t}
= \mathbb{1}\left[\left|v^{cmd}_{z,t}\right| > \varepsilon_v\right]
$$

With the current fixed command:

$$
v^{cmd}_{z,t}=0.08,\quad \varepsilon_v=0.01
$$

therefore:

$$
g^{cmd}_{t}=1
$$

### Step 4: optional engaged-count gate

The C2 version can require exactly two EEs to be engaged:

$$
g^{count}_{t}
= \mathbb{1}\left[\sum_{h=1}^{H} e_{t,h}=N_{\text{eng}}\right]
$$

For C2:

$$
H=4,\quad N_{\text{eng}}=2
$$

So:

$$
g^{count}_{t}
= \mathbb{1}\left[e_{t,1}+e_{t,2}+e_{t,3}+e_{t,4}=2\right]
$$

If this gate is disabled, set:

$$
g^{count}_{t}=1
$$

### Step 5: sum over EEs

The raw reward is:

$$
R^{raw}_{t}
= g^{cmd}_{t}\,g^{count}_{t}
  \sum_{h=1}^{H} r^{ee}_{t,h}
$$

Expanded:

$$
R^{raw}_{t}
=
\mathbb{1}\left[\left|v^{cmd}_{z,t}\right| > \varepsilon_v\right]
\mathbb{1}\left[\sum_{h=1}^{H} e_{t,h}=N_{\text{eng}}\right]
\sum_{h=1}^{H}
\left(
  (1-e_{t,h})
  \mathbb{1}\left[t_{\min}<a_{t,h}<t_{\max}\right]
\right)
$$

For C2 with the count gate enabled, the raw reward range is:

$$
0 \le R^{raw}_{t} \le 2
$$

because if exactly two EEs are engaged, at most two EEs are disengaged and
eligible for the swing-window reward.

### Step 6: apply reward weight

The final reward term is:

$$
R_{t}
= w_{air} R^{raw}_{t}
$$

With the proposed initial weight:

$$
w_{air}=1.5
$$

therefore:

$$
0 \le R_{t} \le 3
$$

## Concrete Examples

Assume:

$$
t_{\min}=0.05,\quad t_{\max}=0.50,\quad N_{\text{eng}}=2,\quad g^{cmd}_{t}=1
$$

### Example A: good two-anchor / two-swing window

$$
e_t=[1,1,0,0],\quad a_t=[0,0,0.20,0.25]
$$

Two EEs are engaged, and both disengaged EEs are inside the valid window:

$$
g^{count}_t=1,\quad \sum_h r^{ee}_{t,h}=2
$$

So:

$$
R^{raw}_t=2,\quad R_t=1.5 \times 2 = 3
$$

### Example B: one swing has lasted too long

$$
e_t=[1,1,0,0],\quad a_t=[0,0,0.20,0.70]
$$

Only one disengaged EE is inside the valid window:

$$
\sum_h r^{ee}_{t,h}=1
$$

So:

$$
R^{raw}_t=1,\quad R_t=1.5
$$

### Example C: static two-engaged pattern after the window expires

$$
e_t=[1,1,0,0],\quad a_t=[0,0,0.80,0.90]
$$

The engaged-count gate passes, but both disengaged timers are too large:

$$
g^{count}_t=1,\quad \sum_h r^{ee}_{t,h}=0
$$

So:

$$
R^{raw}_t=0,\quad R_t=0
$$

This is the main difference from `ee_grip_time`: holding the same two EEs
engaged forever does not keep producing gait reward.

### Example D: wrong engaged count

$$
e_t=[1,1,1,0],\quad a_t=[0,0,0,0.20]
$$

The disengaged EE is inside the valid window, but there are three engaged EEs:

$$
g^{count}_t=0
$$

So with the count gate enabled:

$$
R^{raw}_t=0,\quad R_t=0
$$

If the count gate is disabled, this same state would get:

$$
R^{raw}_t=1,\quad R_t=1.5
$$

## Alternative B: capped disengage time with max-time gate

This is the variant under discussion:

- exactly 2 EEs must be engaged;
- each disengaged EE accumulates disengage time;
- each EE's contribution is capped at `cap_time`;
- if any EE's disengage time exceeds `max_disengage_time`, the whole gait
  reward turns off.

### Plain-text formula

Recommended initial constants:

```text
cap_time = 0.3
max_disengage_time = 0.5
target_engaged_count = 2
weight = 5.0
```

State update:

```text
if engaged[h] == 1:
  disengage_time[h] = 0
else:
  disengage_time[h] += dt
```

Gates:

```text
count_gate = 1 if sum(engaged) == target_engaged_count else 0
max_gate = 1 if max(disengage_time) < max_disengage_time else 0
cmd_gate = 1 if abs(vz_cmd) > command_threshold else 0
```

Raw reward:

```text
reward_raw =
  cmd_gate
  * count_gate
  * max_gate
  * sum_h min(disengage_time[h], cap_time)
```

Weighted reward:

```text
reward = weight * reward_raw
```

### Pseudocode

```python
def ee_capped_disengage_time_reward(
    engaged,
    disengage_time,
    dt,
    vz_cmd,
    *,
    cap_time=0.3,
    max_disengage_time=0.5,
    target_engaged_count=2,
    command_threshold=0.01,
    weight=5.0,
):
  # Timer update.
  for h in range(num_ees):
    if engaged[h]:
      disengage_time[h] = 0.0
    else:
      disengage_time[h] += dt

  # Gates.
  cmd_gate = 1.0 if abs(vz_cmd) > command_threshold else 0.0
  count_gate = 1.0 if sum(engaged) == target_engaged_count else 0.0
  max_gate = 1.0 if max(disengage_time) < max_disengage_time else 0.0

  # Capped accumulated disengage-time reward.
  raw = sum(min(t, cap_time) for t in disengage_time)

  return weight * cmd_gate * count_gate * max_gate * raw
```

### Math notation

Timer:

$$
a_{t,h} =
\begin{cases}
0, & e_{t,h}=1 \\
a_{t-1,h}+\Delta t, & e_{t,h}=0
\end{cases}
$$

Gates:

$$
g^{count}_{t}
= \mathbb{1}\left[\sum_h e_{t,h}=2\right]
$$

$$
g^{max}_{t}
= \mathbb{1}\left[\max_h a_{t,h} < t_{\max}\right]
$$

$$
g^{cmd}_{t}
= \mathbb{1}\left[\left|v^{cmd}_{z,t}\right| > \varepsilon_v\right]
$$

Reward:

$$
R_t
= w \cdot g^{cmd}_{t} \cdot g^{count}_{t} \cdot g^{max}_{t}
  \sum_h \min(a_{t,h}, t_{cap})
$$

With the proposed constants:

```text
t_cap = 0.3
t_max = 0.5
w = 5.0
```

### Examples

Good state:

```text
engaged = [1, 1, 0, 0]
disengage_time = [0.0, 0.0, 0.2, 0.3]

count_gate = 1
max_gate = 1
raw = min(0.0, 0.3) + min(0.0, 0.3) + min(0.2, 0.3) + min(0.3, 0.3)
raw = 0.5
reward = 5.0 * 0.5 = 2.5
```

Near-cap state:

```text
engaged = [1, 1, 0, 0]
disengage_time = [0.0, 0.0, 0.4, 0.4]

count_gate = 1
max_gate = 1       # because 0.4 < 0.5
raw = 0.3 + 0.3
reward = 5.0 * 0.6 = 3.0
```

Too-long disengage:

```text
engaged = [1, 1, 0, 0]
disengage_time = [0.0, 0.0, 0.8, 0.6]

count_gate = 1
max_gate = 0       # because max(disengage_time) = 0.8 >= 0.5
raw would be 0.3 + 0.3, but max_gate shuts it off
reward = 0
```

Wrong engaged count:

```text
engaged = [1, 1, 1, 0]
disengage_time = [0.0, 0.0, 0.0, 0.2]

count_gate = 0
reward = 0
```

### Main risk

This version is denser than the window reward, but it can encourage the policy
to keep swing EEs disengaged until close to `cap_time` or `max_disengage_time`.
The max gate prevents infinite reward from permanently disengaged EEs, but it
does not by itself reward successful re-engagement or upward progress.

Practical mitigations:

```text
1. keep weight modest;
2. keep max_disengage_time close to cap_time;
3. keep ee_slide_penalty active;
4. add progress / anti-stall reward or termination if needed;
5. consider using this only as early shaping, then reducing its weight.
```

### Recommendation for this variant

This version is a reasonable middle ground to try before going to a more sparse
landing-event reward. It is safer than direct `ee_grip_time`, because it does
not reward engaged EEs just for staying engaged. It is also denser than the
strict window reward, because reward grows as swing time accumulates.

I would not start with:

```text
cap_time = 0.5
max_disengage_time = 0.7
weight = 1.5
```

That is probably too permissive and may encourage slow, long swing phases.

Recommended first trial:

```text
cap_time = 0.3
max_disengage_time = 0.5
target_engaged_count = 2
weight = 5.0
```

Plain-text reward with the recommended constants:

```text
count_gate = 1 if sum(engaged) == 2 else 0
max_gate = 1 if max(disengage_time) < 0.5 else 0
cmd_gate = 1 if abs(vz_cmd) > 0.01 else 0

reward_raw =
  cmd_gate
  * count_gate
  * max_gate
  * sum_h min(disengage_time[h], 0.3)

reward = weight * reward_raw
```

With exactly two EEs engaged, at most two EEs are disengaged and can contribute:

```text
max raw reward = 2 * 0.3 = 0.6
max weighted reward = 3.0           # if weight is 5.0
```

This keeps the gait shaping much smaller than the main climb velocity /
progress terms. The intended role is:

```text
track_lin_vel_z / progress: main reason to climb upward
ee_capped_disengage_time: mild shaping to encourage real swing phases
ee_slide_penalty: prevents engaged EEs from sliding along the wall
```

Remaining risk:

```text
The policy may still learn "detach, wait near cap_time, reattach nearby" unless
progress or anti-stall terms make real upward replacement necessary.
```

Final judgment:

```text
Use this as early shaping, not as the final gait objective.
If it helps bootstrap swing, later reduce its weight or replace it with a
stricter window / landing-event reward plus progress.
```

## Comparison To Previous `ee_grip_time`

Previous `ee_grip_time`:

$$
R_{\text{grip},t}
= \mathbb{1}_{cmd}
  \min\left(
    \tau,\,
    \mathbb{1}\left[\sum_h e_{t,h}=2\right]
    \min_h t_{\text{mode},t,h}
  \right)
$$

It can reward a static pattern:

$$
e_t = [1,1,0,0]
$$

as long as it stays stable.

The proposed `ee_air_time_range` does not reward the two engaged EEs directly.
It rewards only the disengaged EEs while their disengaged timers are in the
valid swing window:

$$
R^{raw}_{air\_range,t}
= \mathbb{1}_{cmd}
  \mathbb{1}\left[\sum_h e_{t,h}=2\right]
  \sum_h \mathbb{1}\left[t_{\min}<a_{t,h}<t_{\max}\right]
$$

So "two points stay engaged forever" is no longer enough. The policy must keep
some EEs in active swing windows to receive gait reward.

## Comparison To IsaacLab Official Quadruped `feet_air_time`

IsaacLab official quadruped velocity tracking uses a landing-event form:

$$
R_{\text{IsaacLab},t}
= \sum_h
  \left(a^{last}_{h} - \theta\right)
  \mathbb{1}_{\text{first-contact},t,h}
$$

That reward is paid only on the step a foot contacts the ground after an air
phase.

The proposed range version is denser:

$$
R^{raw}_{air\_range,t}
= \sum_h \mathbb{1}\left[t_{\min}<a_{t,h}<t_{\max}\right]
$$

It pays on every step inside the swing window, like this repo's walking
`feet_air_time` implementation.

## Observation Recommendation

Because the policy is feed-forward, it should observe the same timer used by
the reward:

$$
o^{air}_{t,h}
= \frac{\min(a_{t,h}, t_{\max})}{t_{\max}}
$$

So:

$$
o^{air}_{t,h} \in [0,1]
$$

This mirrors the walking task's `foot_air_time` observation and lets the
policy know whether a detached EE is early, inside, or near the end of the
valid swing window.

## Expected Failure Modes

1. If $w_{air}$ is too high, the policy may prefer keeping EEs detached inside
   the reward window instead of prioritizing progress.
2. If $t_{\max}$ is too large, the policy may learn long hangs.
3. If $t_{\min}$ is too small, single-frame detach jitter may be rewarded.
4. If the engaged-count gate is too strict, learning may become brittle during
   early training.

## Initial Config Candidate

```python
"ee_capped_disengage_time": RewardTermCfg(
  func=mdp.ee_capped_disengage_time,
  weight=5.0,
  params={
    "command_name": "climb_velocity",
    "command_threshold": 0.01,
    "event_term_name": "microspine_adhesion",
    "cap_time": 0.3,
    "max_disengage_time": 0.5,
    "target_engaged_count": 2,
  },
)
```

Recommended accompanying observation:

```python
"ee_disengage_time": ObservationTermCfg(
  func=mdp.ee_disengage_time_obs,
  params={
    "event_term_name": "microspine_adhesion",
    "clip": 0.5,
  },
)
```
