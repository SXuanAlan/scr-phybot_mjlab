"""Command-line pieces shared by the scripts."""

from __future__ import annotations

import argparse

__all__ = ("add_bool_option",)


def _bool(value: str) -> bool:
  if value not in ("True", "False"):
    raise argparse.ArgumentTypeError(f"expected True or False, got {value!r}")
  return value == "True"


def add_bool_option(parser: argparse.ArgumentParser, flag: str, *, default: bool, help: str) -> None:
  """A boolean option written ``--flag``, ``--flag True`` or ``--flag False``.

  The explicit form is the one mjlab's tyro CLIs use, so commands written
  for them keep working.
  """
  parser.add_argument(flag, type=_bool, nargs="?", const=True, default=default, help=f"{help} (default: {default})")
