"""Repository pytest plugin: import paths and explicit skip rules.

Loaded through ``-p alz_pytest_plugin`` in ``pyproject.toml``. It lives here,
not in a root or ``runtime/`` conftest.py, because ``runtime/binding.py``
requires every ``*.py`` file in the repository root and in ``runtime/`` to be
pinned in a calibration authorization. A new file there would change what
those historical gates accept.

The plugin does three things, all reported in the test output:

1. When ``particlegan`` is not importable, it appends the native ParticleGAN
   0.10.0 release target (``artifacts/native-releases/particlegan-0.10.0/
   site-packages``, or ``$ALZ_PARTICLEGAN_0_10_SITE_PACKAGES``) to ``sys.path`` if that
   directory exists. It never shadows an installed ParticleGAN.
2. In the test files that are known to need optional third-party packages
   (``OPTIONAL_DEPENDENCY_SCOPE``: ``paper_watch/tests``, the seven
   ``legacy_pg080`` modules and the ParticleGAN probe), a module whose import
   fails only because such a package is not installed at all is reported as
   skipped with that reason, instead of a collection error; so is a test whose
   body raises that ModuleNotFoundError. Every other test file (the core
   microglia and repository tests, and any new file) stays strict: a missing
   package there is an error, so a stray import cannot become a silent skip.
   Any other import failure always errors or fails.
3. ``runtime/`` binding tests need the local GPU0 workspace (absolute
   ``/ml2/codex-storage`` paths and the deployed calibration authorization)
   and, for seven of them, the ignored ``data/calibration_fold0`` inventory at
   the absolute path the deployed authorization pins (so only the original
   checkout qualifies). Otherwise the tests are skipped with the reason named.
"""
from __future__ import annotations

import ast
import importlib.util
import json
import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
NATIVE_PG_SITE = ROOT / "artifacts/native-releases/particlegan-0.10.0/site-packages"

# Top-level names of third-party packages that some test modules need and that
# the default CI job deliberately does not install. A ModuleNotFoundError for
# any other name (for example a repository module) is still a collection error.
OPTIONAL_PACKAGES = {
    "anndata": "anndata (legacy and paper-watch data loaders)",
    "h5py": "h5py",
    "pandas": "pandas",
    "particlegan": "ParticleGAN (native release target or envs/*.txt)",
    "scipy": "scipy",
    "sklearn": "scikit-learn",
    "torch": "torch",
}

# Only these test files may turn a missing optional package into a skip. The
# core tests (tests/test_microglia_*.py, tests/test_repo_*.py) need numpy and
# pytest only and are deliberately left out, so a new torch/scipy/pandas import
# reached from them fails CI instead of being reported as skipped.
OPTIONAL_DEPENDENCY_SCOPE = (
    "paper_watch/tests/",
    # legacy_pg080 modules (see tests/support/alz_legacy.py)
    "tests/test_baselines.py",
    "tests/test_cell_eval_adapter.py",
    "tests/test_data.py",
    "tests/test_gan.py",
    "tests/test_hypergan_bridge.py",
    "tests/test_metrics.py",
    "tests/test_smoke.py",
    # needs_particlegan probe (runs torch and ParticleGAN in a subprocess)
    "tests/test_empf_conditional_probe.py",
)

RUNTIME_DIR = ROOT / "runtime"
# These runtime tests build a full calibration source closure, which reads the
# ignored data/calibration_fold0 inventory. Without it they fail with
# "calibration input inventory absent or changed" rather than testing the gate.
NEEDS_CALIBRATION_INVENTORY = {
    "test_bcap_binding.py::BCAPBinding::test_complete_source_closure_with_virtual_enabled_plan",
    "test_bcap_binding.py::BCAPBinding::test_disabled_plan_refuses_and_no_model_imports",
    "test_ensemble_scope.py::EnsembleBinding::test_complete_control_source_closure_for_both_parents",
    "test_learned_binding.py::LearnedBinding::test_complete_source_closure_with_virtual_enabled_plan",
    "test_learned_binding.py::LearnedBinding::test_disabled_plan_refuses_and_no_model_imports",
    "test_network_rate_binding.py::NetworkRateBinding::test_complete_source_closure_with_virtual_enabled_plan",
    "test_network_rate_binding.py::NetworkRateBinding::test_disabled_plan_refuses_and_no_model_imports",
}


def native_particlegan_site() -> Path:
    override = os.environ.get("ALZ_PARTICLEGAN_0_10_SITE_PACKAGES")
    return Path(override) if override else NATIVE_PG_SITE


def add_native_particlegan(site: Path | None = None) -> str:
    """Make the native release importable without shadowing an installed one."""
    if importlib.util.find_spec("particlegan") is not None:
        return "particlegan: installed in the interpreter"
    site = native_particlegan_site() if site is None else site
    if site.is_dir() and (site / "particlegan").is_dir():
        if str(site) not in sys.path:
            sys.path.append(str(site))
        return f"particlegan: native release target {site}"
    return f"particlegan: not available (no install, no {site})"


def missing_optional_package(error: BaseException | None) -> str | None:
    """Return the optional package behind an import failure, if that is all it is."""
    seen = set()
    while error is not None and id(error) not in seen:
        seen.add(id(error))
        if isinstance(error, ModuleNotFoundError) and error.name:
            top = error.name.split(".")[0]
            return top if top in OPTIONAL_PACKAGES else None
        error = error.__cause__ or error.__context__
    return None


def _path_literals(source: Path, names: set[str]) -> dict[str, Path]:
    """Read ``NAME=Path('/literal')`` constants without importing the module."""
    found = {}
    for node in ast.parse(source.read_text()).body:
        if (isinstance(node, ast.Assign) and len(node.targets) == 1
                and isinstance(node.targets[0], ast.Name) and node.targets[0].id in names
                and isinstance(node.value, ast.Call) and getattr(node.value.func, "id", None) == "Path"
                and len(node.value.args) == 1 and isinstance(node.value.args[0], ast.Constant)):
            found[node.targets[0].id] = Path(node.value.args[0].value)
    return found


def calibration_inventory_problem(root: Path = ROOT) -> str | None:
    """Why the seven closure tests cannot exercise their gate here, if they cannot.

    The deployed authorization pins the inventory by absolute path, so the tests
    pass only in the checkout that authorization names (and only when the
    ignored inventory is present there).
    """
    inventory = root / "data/calibration_fold0/manifest.json"
    try:
        deployment = json.loads((root / "calibration_deployment.json").read_text())
        pins = json.loads(Path(deployment["authorization_path"]).read_text())["files_SHA256"]
    except (OSError, ValueError, KeyError, TypeError):
        return "needs the deployed calibration authorization named in calibration_deployment.json"
    if str(inventory) not in pins:
        pinned = sorted(k for k in pins if k.endswith("data/calibration_fold0/manifest.json"))
        return ("needs the checkout whose calibration input inventory the deployed authorization pins "
                f"({', '.join(pinned) or 'none'}); this checkout is {root}")
    if not inventory.is_file():
        return f"needs the ignored calibration input inventory {inventory} (absent in this checkout)"
    return None


def runtime_workspace_missing(root: Path = ROOT) -> list[str]:
    """Absolute workspace paths the runtime binding tests read, if absent."""
    binding = root / "runtime/binding.py"
    if not binding.is_file():
        return [str(binding)]
    paths = _path_literals(binding, {"PROJECT", "SCIENCE", "GUARD"})
    required = [paths[name] for name in ("PROJECT", "SCIENCE") if name in paths]
    if "GUARD" in paths:
        required.append(paths["GUARD"] / "REVIEW_MANIFEST.json")
    deployment = root / "calibration_deployment.json"
    try:
        required.append(Path(json.loads(deployment.read_text())["authorization_path"]))
    except (OSError, ValueError, KeyError):
        required.append(deployment)
    return [str(path) for path in required if not path.exists()]


class _SkippingModule(pytest.Module):
    """Report a module as skipped, before importing it."""

    skip_reason = ""

    def _getobj(self):
        pytest.skip(self.skip_reason, allow_module_level=True)


class _OptionalDependencyModule(pytest.Module):
    """Turn 'optional package not installed' into an explicit module skip."""

    def _getobj(self):
        try:
            return super()._getobj()
        except self.CollectError as error:
            name = missing_optional_package(error)
            if name is None:
                raise
            pytest.skip(
                f"{_relative(self.path)}: optional dependency {OPTIONAL_PACKAGES[name]} "
                "is not installed; see tests/README.md and envs/README.md",
                allow_module_level=True,
            )


def optional_dependency_allowed(path, root: Path = ROOT) -> bool:
    """Whether a missing optional package may skip tests in this file.

    ``root`` is the pytest rootdir (the repository root in normal use), and the
    file must match ``OPTIONAL_DEPENDENCY_SCOPE`` relative to it.
    """
    try:
        relative = Path(path).resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return False
    return any(relative == scope or (scope.endswith("/") and relative.startswith(scope))
               for scope in OPTIONAL_DEPENDENCY_SCOPE)


def _relative(path) -> str:
    try:
        return str(Path(path).resolve().relative_to(ROOT))
    except ValueError:
        return str(path)


def _is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.resolve().relative_to(parent.resolve())
        return True
    except ValueError:
        return False


def pytest_configure(config):
    config._alz_particlegan = add_native_particlegan()


def pytest_report_header(config):
    return [getattr(config, "_alz_particlegan", "")]


@pytest.hookimpl(tryfirst=True)
def pytest_pycollect_makemodule(module_path, parent):
    path = Path(module_path)
    if _is_relative_to(path, RUNTIME_DIR):
        missing = runtime_workspace_missing()
        if missing:
            module = _SkippingModule.from_parent(parent, path=path)
            module.skip_reason = (
                "runtime binding tests need the local GPU0 workspace; missing: " + ", ".join(missing)
            )
            return module
    if optional_dependency_allowed(path, parent.config.rootpath):
        return _OptionalDependencyModule.from_parent(parent, path=path)
    return None


@pytest.hookimpl(wrapper=True)
def pytest_runtest_call(item):
    """Same rule for an optional package imported inside a test body."""
    try:
        return (yield)
    except ModuleNotFoundError as error:
        name = missing_optional_package(error)
        if name is None or not optional_dependency_allowed(item.path, item.config.rootpath):
            raise
        pytest.skip(f"{item.nodeid}: optional dependency {OPTIONAL_PACKAGES[name]} is not installed "
                    "(imported inside the test); see tests/README.md and envs/README.md")


def pytest_collection_modifyitems(config, items):
    inventory_problem = None
    if any(_is_relative_to(Path(str(item.path)), RUNTIME_DIR) for item in items):
        inventory_problem = calibration_inventory_problem()
    for item in items:
        path = Path(str(item.path))
        if not _is_relative_to(path, RUNTIME_DIR):
            continue
        item.add_marker(pytest.mark.needs_local_workspace)
        local_id = path.name + "::" + item.nodeid.split("::", 1)[1] if "::" in item.nodeid else path.name
        if inventory_problem and local_id in NEEDS_CALIBRATION_INVENTORY:
            item.add_marker(pytest.mark.skip(reason=inventory_problem))
