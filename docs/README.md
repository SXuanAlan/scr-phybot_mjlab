# Documentation

- `tasks/`: the [climb_wall](tasks/climb_wall.md) guide. The walk, squat
  and transition80 configs document themselves in their
  `config/phybot_c2/` module docstrings.
- `microspine/`: the pad model. [state_machine.md](microspine/state_machine.md)
  covers engage, hold and release; [phybot_c2_cad_asset.md](microspine/phybot_c2_cad_asset.md)
  covers how the pad CAD becomes the robot asset.
- `figures/`: figures for the paper, with a caption each where the script
  writes one. The scripts in `scripts/microspine/`, `scripts/reference/` and
  `scripts/validate/` regenerate them.
- `paper/`: method write-ups in LaTeX.
- `reference/`: papers this work builds on and reading notes, kept on disk
  and out of git.

Analyses of a checkpoint stay beside it in `logs/`, and other generated
outputs in the ignored `artifacts/`; see `src/scr_phybot_mjab/tools/paths.py`.
