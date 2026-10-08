# Microspine experiment data

`raw/20260905-110314/` contains the original single-unit robot-arm trials:
`results.csv` and per-trial logs. These 298 files were moved from
`src/scr_phybot_mjab/microspine/limit_surface/20260905-110314/` without changing
their contents. The accompanying image is also retained under `raw/`.

The three storage roles are separate:

| Role | Repository path |
| --- | --- |
| Original observations | `data/microspine/raw/` |
| Packaged runtime distribution | `src/scr_phybot_mjab/assets/microspine/empirical_unit_v1.npz` |
| Fitting report and calibration/baseline tables | `data/microspine/empirical_unit_v1/` |

Run `scripts/microspine/build_empirical_limit_surface.py` to rebuild. Its `--data`,
`--output`, and `--analysis-dir` arguments select these three destinations
independently. Runtime tasks only read the packaged distribution table.
Calibration and raw-cell baseline tables are analysis artifacts, not inputs
to the active climbing task. The report records the raw summary's SHA256.

See [the code layout](../../docs/reports/empirical_climb/code_layout.md) for
shared geometry, mechanics, reference-generation tools and the import flow.

`raw/20260922-163807/` contains the newer 231-trial, 3 N preload experiment,
including 86 trials carried over from `20260922-144929`. Analyze first slip
across drag, lateral and normal loading with:

```bash
python scripts/microspine/analyze_microspine_first_failure.py --data data/microspine/raw/20260922-163807
```

The first-failure report (generated locally into the ignored `artifacts/limit_surface/20260922-163807/README.md`)
contains classification and surface figures, per-trial event tables, threshold
sensitivity flags and all-trial review plots. It uses force and TCP pose to
estimate the first event before any re-engagement, while retaining the robot's
terminal label for comparison. The 3 N experiment and the older 10 N experiment
are distinct protocols. These analysis outputs do not update the runtime model.

The four-panel figure is drawn by
[`first_failure_plot.py`](../../src/scr_phybot_mjab/tools/microspine/first_failure_plot.py).
Its layout is reconstructed from the retained reference PNG; the original
reference plotting source was not found. The new plot preserves the first-slip
analysis and does not reuse the older experiment's curve or outlier labels.
