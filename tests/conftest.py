"""Tests marked ``gpu`` run only with ``--gpu`` and a CUDA device.

They use ``cuda:0``; pick the card with ``CUDA_VISIBLE_DEVICES``, so they
stay off a card that another job is using.
"""

import pytest


def pytest_addoption(parser: pytest.Parser) -> None:
  parser.addoption("--gpu", action="store_true", help="also run the tests marked gpu (on cuda:0)")


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
  if config.getoption("--gpu"):
    import torch

    if torch.cuda.is_available():
      return
    reason = "no CUDA device"
  else:
    reason = "GPU test: run with --gpu"
  skip = pytest.mark.skip(reason=reason)
  for item in items:
    if "gpu" in item.keywords:
      item.add_marker(skip)
