"""Project-local helpers for embedding files into MuJoCo specs."""

from pathlib import Path
from typing import Any


def update_assets(
  assets: dict[str, Any],
  path: str | Path,
  meshdir: str | None = None,
  glob: str = "*",
  recursive: bool = False,
) -> None:
  """Add files from ``path`` to a MuJoCo asset dictionary.

  MJLab exposed this helper through ``mjlab.utils.os`` until v1.4. Keeping the
  small file-loading utility local decouples project assets from that removed
  convenience API while preserving the existing asset keys.
  """
  for file_path in Path(path).glob(glob):
    if file_path.is_file():
      asset_key = f"{meshdir}/{file_path.name}" if meshdir else file_path.name
      assets[asset_key] = file_path.read_bytes()
    elif file_path.is_dir() and recursive:
      update_assets(assets, file_path, meshdir, glob, recursive)
