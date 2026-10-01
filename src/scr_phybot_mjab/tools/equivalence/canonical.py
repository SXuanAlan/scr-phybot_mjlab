"""Canonical ``path = value`` text form of a config object tree.

Two configs behave the same when their line lists are equal. Field and dict
order is kept on purpose: observation concatenation and reward summation
follow it. A shared mutable object is written once and referenced as
``alias:<path>`` afterwards, because sharing is behaviour too (a mutation
through one path is seen through the other).
"""

from __future__ import annotations

import dataclasses
import enum
import functools
import hashlib
import re
import sys
import types
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[4]
_SITE_ROOT = Path(np.__file__).resolve().parents[1]
_HOME = str(Path.home())
_ADDRESS = re.compile(r" at 0x[0-9a-fA-F]+")
_NAME_TOKEN = re.compile(r"\b(fn|cls|method|dataclass|object|enum|depth-limit):([\w.<>]+)")
_MAX_DEPTH = 80


def canonical_lines(obj: Any, root: str) -> list[str]:
  walker = _Walker()
  walker.visit(obj, root, depth=0)
  return walker.lines


def apply_renames(lines: list[str], renames: dict[str, str]) -> list[str]:
  """Rewrite function/class names that moved, so a baseline matches new code.

  A rename of ``mod.Class`` also covers the names nested in it, such as an
  enum member ``mod.Class.MEMBER``.
  """
  if not renames:
    return lines

  def renamed(name: str) -> str:
    head, tail = name, ""
    while head:
      if head in renames:
        return renames[head] + tail
      head, _, last = head.rpartition(".")
      tail = f".{last}{tail}"
    return name

  def swap(match: re.Match) -> str:
    return f"{match.group(1)}:{renamed(match.group(2))}"

  return [_NAME_TOKEN.sub(swap, line) for line in lines]


class _Walker:
  def __init__(self) -> None:
    self.lines: list[str] = []
    self._first_path: dict[int, str] = {}
    self._pinned: list[Any] = []  # keeps visited objects alive so ids stay unique
    self._active_functions: set[int] = set()

  def emit(self, path: str, value: str) -> None:
    self.lines.append(f"{path} = {value}")

  def visit(self, obj: Any, path: str, depth: int) -> None:
    if depth > _MAX_DEPTH:
      self.emit(path, f"depth-limit:{_qualname(type(obj))}")
      return
    if _is_shared_state(obj):
      first = self._first_path.get(id(obj))
      if first is not None:
        self.emit(path, f"alias:{first}")
        return
      self._first_path[id(obj)] = path
      self._pinned.append(obj)
    if not self._visit_leaf(obj, path):
      self._visit_container(obj, path, depth + 1)

  def _visit_leaf(self, obj: Any, path: str) -> bool:
    text = _leaf_text(obj)
    if text is None:
      return False
    self.emit(path, text)
    return True

  def _visit_container(self, obj: Any, path: str, depth: int) -> None:
    if isinstance(obj, functools.partial):
      self.emit(path, "partial")
      self.visit(obj.func, f"{path}.func", depth)
      self.visit(obj.args, f"{path}.args", depth)
      self.visit(obj.keywords, f"{path}.keywords", depth)
    elif isinstance(obj, types.FunctionType):
      self._visit_function(obj, path, depth)
    elif dataclasses.is_dataclass(obj):
      self.emit(path, f"dataclass:{_qualname(type(obj))}")
      for field in dataclasses.fields(obj):
        self.visit(getattr(obj, field.name), f"{path}.{field.name}", depth)
    elif isinstance(obj, dict):
      self.emit(path, f"dict[{len(obj)}]")
      for key, value in obj.items():
        self.visit(value, f"{path}[{_key_text(key)}]", depth)
    elif isinstance(obj, (list, tuple)):
      self.emit(path, f"{type(obj).__name__}[{len(obj)}]")
      for index, item in enumerate(obj):
        self.visit(item, f"{path}[{index}]", depth)
    elif hasattr(obj, "__dict__"):
      self.emit(path, f"object:{_qualname(type(obj))}")
      for name, value in vars(obj).items():
        self.visit(value, f"{path}.{name}", depth)
    else:
      self.emit(path, f"repr:{_normalize(repr(obj))}")

  def _visit_function(self, fn: types.FunctionType, path: str, depth: int) -> None:
    self.emit(path, f"fn:{_qualname(fn)}")
    if id(fn) in self._active_functions:
      return
    self._active_functions.add(id(fn))
    # mjlab calls terms as func(env, **params): defaults are part of behaviour.
    if fn.__defaults__:
      self.visit(fn.__defaults__, f"{path}.<defaults>", depth)
    if fn.__kwdefaults__:
      self.visit(fn.__kwdefaults__, f"{path}.<kwdefaults>", depth)
    for name, cell in zip(fn.__code__.co_freevars, fn.__closure__ or ()):
      self.visit(_cell_value(cell), f"{path}.<closure:{name}>", depth)
    self._active_functions.discard(id(fn))


def _leaf_text(obj: Any) -> str | None:
  """Text for values that have no children, or None for containers."""
  if obj is None or isinstance(obj, (bool, int)):
    return repr(obj)
  if isinstance(obj, float):
    return f"f:{obj!r}"
  if isinstance(obj, str):
    return f"s:{_normalize(repr(obj))}"
  if isinstance(obj, bytes):
    return f"bytes:{_digest(obj)}"
  if isinstance(obj, Path):
    return f"path:{_normalize(str(obj))}"
  if isinstance(obj, enum.Enum):
    return f"enum:{_qualname(type(obj))}.{obj.name}"
  if isinstance(obj, np.ndarray) or _is_tensor(obj):
    return _array_text(obj)
  if isinstance(obj, np.generic):
    return f"np.{obj.dtype}:{obj.item()!r}"
  if isinstance(obj, type):
    return f"cls:{_qualname(obj)}"
  if isinstance(obj, types.MethodType):
    return f"method:{_qualname(obj.__func__)}@{_qualname(type(obj.__self__))}"
  if isinstance(obj, (types.BuiltinFunctionType, types.MethodWrapperType, types.WrapperDescriptorType)):
    return f"fn:{_qualname(obj)}"
  if isinstance(obj, types.ModuleType):
    return f"module:{obj.__name__}"
  if isinstance(obj, (set, frozenset)):
    items = sorted(_leaf_text(item) or _normalize(repr(item)) for item in obj)
    return f"{type(obj).__name__}{{{', '.join(items)}}}"
  return None


def _is_shared_state(obj: Any) -> bool:
  if isinstance(obj, (dict, list, set)):
    return True
  if isinstance(obj, (type, enum.Enum, np.ndarray, types.FunctionType, types.ModuleType)) or _is_tensor(obj):
    return False
  if dataclasses.is_dataclass(obj):
    return not type(obj).__dataclass_params__.frozen
  return hasattr(obj, "__dict__") and not isinstance(obj, functools.partial)


def _is_tensor(obj: Any) -> bool:
  torch = sys.modules.get("torch")
  return torch is not None and isinstance(obj, torch.Tensor)


def _array_text(obj: Any) -> str:
  kind = "tensor" if _is_tensor(obj) else "ndarray"
  data = obj.detach().cpu().numpy() if _is_tensor(obj) else obj
  if data.dtype == object:
    payload = _normalize(repr(data.tolist())).encode()
  else:
    payload = np.ascontiguousarray(data).tobytes()
  return f"{kind}:{data.dtype}{list(data.shape)}:{_digest(payload)}"


def _key_text(key: Any) -> str:
  return key if isinstance(key, str) else _normalize(repr(key))


def _cell_value(cell: types.CellType) -> Any:
  try:
    return cell.cell_contents
  except ValueError:
    return "<empty cell>"


def _qualname(obj: Any) -> str:
  module = getattr(obj, "__module__", None) or "?"
  name = getattr(obj, "__qualname__", None) or getattr(obj, "__name__", None) or type(obj).__name__
  return f"{module}.{name}"


def _normalize(text: str) -> str:
  text = text.replace(str(REPO_ROOT), "<REPO>").replace(str(_SITE_ROOT), "<SITE>")
  return _ADDRESS.sub(" at 0x?", text.replace(_HOME, "~"))


def _digest(payload: bytes) -> str:
  return hashlib.sha256(payload).hexdigest()[:16]
