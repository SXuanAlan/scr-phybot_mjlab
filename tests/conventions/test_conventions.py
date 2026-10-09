"""The repository rules in AGENTS.md that a test can check."""

import ast
import importlib
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
SCRIPTS = ROOT / "scripts"
RUNTIME_PACKAGES = ("assets", "microspine", "tasks")
# The train and list_envs wrappers hand the whole command line to mjlab.
MJLAB_WRAPPERS = {SCRIPTS / "train.py", SCRIPTS / "list_envs.py"}


def _python_files(directory: Path) -> list[Path]:
  return sorted(p for p in directory.rglob("*.py") if "__pycache__" not in p.parts)


def _module_name(path: Path) -> str:
  parts = path.relative_to(SRC).with_suffix("").parts
  return ".".join(parts[:-1] if parts[-1] == "__init__" else parts)


def test_readme_lists_exactly_the_registered_tasks():
  from mjlab.tasks.registry import list_tasks

  registered = {task for task in list_tasks() if "Phybot" in task}
  documented = set(re.findall(r"`(Mjlab-[A-Za-z0-9-]*Phybot[A-Za-z0-9-]*)`", (ROOT / "README.md").read_text()))
  assert documented == registered


def test_private_names_stay_inside_their_package():
  offenders = []
  for path in _python_files(SRC):
    module = _module_name(path)
    package = module if path.name == "__init__.py" else module.rpartition(".")[0]
    for node in ast.walk(ast.parse(path.read_text())):
      if isinstance(node, ast.ImportFrom) and node.level == 0 and (node.module or "").startswith("scr_phybot_mjab"):
        source_package = node.module.rpartition(".")[0]
        for alias in node.names:
          if alias.name.startswith("_") and not alias.name.startswith("__") and source_package != package:
            offenders.append(f"{module} imports {node.module}.{alias.name}")
  assert not offenders, offenders


def test_runtime_packages_read_no_environment_variables():
  offenders = []
  for package in RUNTIME_PACKAGES:
    for path in _python_files(SRC / "scr_phybot_mjab" / package):
      for node in ast.walk(ast.parse(path.read_text())):
        if isinstance(node, ast.Attribute) and node.attr in ("environ", "getenv"):
          offenders.append(f"{path.relative_to(ROOT)}:{node.lineno}")
  assert not offenders, offenders


def _star_imports(init: Path):
  package = _module_name(init)
  for node in ast.parse(init.read_text()).body:
    if isinstance(node, ast.ImportFrom) and any(alias.name == "*" for alias in node.names):
      name = "." * node.level + (node.module or "")
      yield importlib.import_module(importlib.util.resolve_name(name, package) if node.level else node.module)


@pytest.mark.parametrize(
  "init",
  [p for p in _python_files(SRC) if p.name == "__init__.py" and "import *" in p.read_text()],
  ids=lambda p: _module_name(p),
)
def test_star_imports_do_not_shadow_names(init):
  owner: dict[str, str] = {}
  for module in _star_imports(init):
    if module.__name__.startswith("scr_phybot_mjab"):
      assert hasattr(module, "__all__"), f"{module.__name__} is star-imported without __all__"
      names = module.__all__
    else:
      names = [name for name in dir(module) if not name.startswith("_")]
    for name in names:
      assert name not in owner, f"{name} from {module.__name__} shadows the one from {owner[name]}"
      owner[name] = module.__name__


@pytest.mark.parametrize("path", _python_files(SCRIPTS), ids=lambda p: str(p.relative_to(SCRIPTS)))
def test_scripts_are_command_line_modules(path):
  tree = ast.parse(path.read_text())
  for index, node in enumerate(tree.body):
    if isinstance(node, (ast.Import, ast.ImportFrom, ast.FunctionDef, ast.ClassDef, ast.Assign, ast.AnnAssign)):
      continue
    if index == 0 and isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant):
      continue  # the docstring
    if isinstance(node, ast.If) and ast.unparse(node.test) == "__name__ == '__main__'":
      continue
    # Choosing the GL or plot backend has to happen before the imports.
    call = ast.unparse(node)
    assert re.fullmatch(r"(os\.environ\.setdefault|matplotlib\.use)\(.*\)", call), f"runs on import: {call}"
  guards = [n for n in tree.body if isinstance(n, ast.If) and ast.unparse(n.test) == "__name__ == '__main__'"]
  assert guards, "no __main__ guard"
  if path in MJLAB_WRAPPERS:
    return
  main = next((n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "main"), None)
  assert main is not None, "no main()"
  assert [a.arg for a in main.args.args] == ["argv"], "main() should take argv"


# Directories that hold generated files; a script's default path into one comes from tools/paths.py.
GENERATED_DIRS = ("docs", "logs", "results", "artifacts", "scratch", "/tmp")


def _spelled_out_output(default: ast.expr) -> bool:
  names = {n.id for n in ast.walk(default) if isinstance(n, ast.Name)} - {"Path", "REPOSITORY_ROOT", "ROOT", "REPO"}
  parts = [n.value for n in ast.walk(default) if isinstance(n, ast.Constant) and isinstance(n.value, str)]
  return not names and any(part.startswith(GENERATED_DIRS) for part in parts)


@pytest.mark.parametrize("path", _python_files(SCRIPTS), ids=lambda p: str(p.relative_to(SCRIPTS)))
def test_script_defaults_take_output_locations_from_tools_paths(path):
  offenders = []
  for node in ast.walk(ast.parse(path.read_text())):
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "add_argument":
      default = next((k.value for k in node.keywords if k.arg == "default"), None)
      if default is not None and _spelled_out_output(default):
        offenders.append(f"line {node.lineno}: default={ast.unparse(default)}")
  assert not offenders, offenders


def test_scripts_and_tools_never_import_a_cli_script():
  scripts = _python_files(SCRIPTS)
  script_modules = {p.stem for p in scripts}
  for path in scripts + _python_files(SRC / "scr_phybot_mjab" / "tools"):
    for node in ast.walk(ast.parse(path.read_text())):
      if isinstance(node, ast.ImportFrom):
        modules = [node.module or ""]
      elif isinstance(node, ast.Import):
        modules = [alias.name for alias in node.names]
      else:
        continue
      assert not any(m.split(".")[0] in script_modules for m in modules), path
