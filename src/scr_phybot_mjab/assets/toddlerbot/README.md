# ToddlerBot 2XM with four single units

This asset adapts [the official ToddlerBot model](https://github.com/hshi74/toddlerbot)
at commit `e337f3b177b4b53abff70b31d1695a7b66cc6d2e`. It is the 2XM variant,
not the lighter 2XC variant. The imported files are unmodified in `upstream/`;
`provenance.json` records every source path and SHA-256. Importing the Python
package never accesses the network.

`get_spec()` builds the climbing model with the original inertias, 30 motors,
passive joints, gears and neck linkage. The two waist backlash tendons become
ideal zero-backlash equality constraints, preserving both waist motor loads.
Upstream motor limits come from `default.yml` plus the 2XM overrides in
`robot.yml`, rather than the generic XML control ranges.

Exactly one unit attaches rigidly to each forearm and shin, following the C2
layout: posterior forearms and anterior shins. The contact names still use
`hand` and `foot` for task compatibility. Ankle and wrist motion does not move
the units relative to their parent limbs. Each unit has one
point contact and a force-only connect anchor. It has no independent rotation
lock or measured contact moment allowance. The native unit frame has contact
center at zero, outward face `-y`, and down-rake direction `-z`.

| Parent body | Contact center in parent frame, m | Adapter |
|---|---|---|
| `left_knee_link` | `(0.048, 0.019, -0.055)` | Anterior shin; face +x, rake -z |
| `right_knee_link` | `(0.048, -0.019, -0.055)` | Anterior shin; face +x, rake -z |
| `left_elbow_yaw_link` | `(-0.040, 0.00335, -0.029)` | Posterior forearm; face -x, rake +z |
| `right_elbow_yaw_link` | `(-0.040, -0.00335, -0.029)` | Posterior forearm; face -x, rake +z |

Directions in the table are in each parent link's local frame. The forearm
rake points toward the elbow, so raised arms and descending legs can both
engage the wall. Schematic adapters extend 18 mm behind the 15 mm unit on the
shins and 12 mm on the forearms; their backs meet the parent collision surface.
These replace the earlier palm and 45° toe mounts.
The adapter visual is schematic, not fabrication CAD. Each unit plus adapter
is assigned **15 g**, with approximate box inertia; total modeled mass is
**3.82887188 kg**. Hardware adapter mass, stiffness and clearances still need
measurement.

The updated reference spreads the arm contacts to 36 cm and the leg contacts
to 24 cm, versus 30 cm and 20 cm in the earlier layout. IK aligns unit faces
to the wall while allowing in-plane rake rotation with the limbs. Static and
runtime failure checks use each unit's actual rake frame. References store a
mount signature and are rejected if loaded with a different mounting layout.

`meshes/microspine_unit.stl` is copied from the project's existing normalized
single-unit CAD (`phybot_c2/new_meshes/microspine_unit.stl`), about
27.9 × 15 × 25.1 mm. The physical proxy is one 3 mm sphere tangent to the
contact plane. Robot self-collision uses the upstream box approximations.

The task uses PD servos with the pinned low-speed torque limits and a
conservative symmetric speed envelope based on `q_dot_tau_max`, `q_dot_max`,
and `tau_q_dot_max`. This does not establish continuous thermal torque,
backlash behavior, friction compensation or hardware controller performance.

The user's hypothesis is **1.5 N preload per unit with the existing empirical
failure distribution unchanged**. The empirical table's original 10 N seating
metadata is retained. No table values are rescaled or new measurements implied.

The upstream software is MIT licensed; the upstream README identifies the
robot design as **CC BY-NC-SA 4.0**. Attribution and the original README/LICENSE
are retained in `upstream/`. The procedural mounts and contact/actuator model
are modifications made in this repository. See
[task documentation](../../../../docs/toddlerbot_climb/README.md).
