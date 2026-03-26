"""Play/evaluate RL agent — delegates to mjlab's play script after registering tasks.

Supports --agent zero|random|trained modes:
  python scripts/play.py Mjlab-ClimbWall-Phybot-C2 --agent zero
  python scripts/play.py Mjlab-ClimbWall-Phybot-C2 --agent random
  python scripts/play.py Mjlab-ClimbWall-Phybot-C2 --checkpoint-file path/to/model.pt
"""

import os
import sys


def _display_available() -> bool:
  """Check if a working display is available for the native viewer."""
  display = os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")
  if not display:
    return False
  try:
    import glfw

    glfw.init()
    glfw.window_hint(glfw.VISIBLE, glfw.FALSE)
    window = glfw.create_window(1, 1, "", None, None)
    if window:
      glfw.destroy_window(window)
      glfw.terminate()
      return True
    glfw.terminate()
    return False
  except Exception:
    return False


if __name__ == "__main__":
  # If no --viewer flag was explicitly passed and display is unavailable, default to viser.
  if "--viewer" not in sys.argv and not _display_available():
    sys.argv.extend(["--viewer", "viser"])

  import spider_rl_mjlab.tasks  # noqa: F401

  from mjlab.scripts.play import main

  main()
