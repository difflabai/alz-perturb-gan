"""Data-free checks for the test plugin, the Markdown link checker and check_env."""
import importlib.util
import json
import os
import platform
import subprocess
import sys
from pathlib import Path

import pytest

import alz_pytest_plugin as plugin

ROOT = Path(__file__).resolve().parents[1]


def load(relative, name):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


links = load("scripts/ci/check_md_links.py", "check_md_links")
check_env = load("scripts/check_env.py", "check_env")


# --- plugin -----------------------------------------------------------------

def test_only_missing_optional_packages_become_skips():
    def wrapped(error):
        try:
            try:
                raise error
            except ImportError as inner:
                raise RuntimeError("collect error") from inner
        except RuntimeError as outer:
            return outer

    assert plugin.missing_optional_package(wrapped(ModuleNotFoundError("x", name="torch.nn"))) == "torch"
    assert plugin.missing_optional_package(wrapped(ModuleNotFoundError("x", name="particlegan"))) == "particlegan"
    # Repository modules and broken installs stay collection errors.
    assert plugin.missing_optional_package(wrapped(ModuleNotFoundError("x", name="alz_perturb_gan"))) is None
    assert plugin.missing_optional_package(wrapped(ImportError("cannot import name 'x' from 'torch'"))) is None


def test_native_particlegan_is_appended_only_when_absent(tmp_path, monkeypatch):
    site = tmp_path / "site"
    (site / "particlegan").mkdir(parents=True)
    monkeypatch.setattr(plugin.importlib.util, "find_spec", lambda name: None)
    monkeypatch.setattr(sys, "path", list(sys.path))
    assert "native release target" in plugin.add_native_particlegan(site)
    assert sys.path[-1] == str(site)
    assert "not available" in plugin.add_native_particlegan(tmp_path / "missing")
    monkeypatch.setattr(plugin.importlib.util, "find_spec", lambda name: object())
    assert "installed" in plugin.add_native_particlegan(site)


def _fake_runtime_root(tmp_path, auth_pins=None, write_auth=True):
    (tmp_path / "runtime").mkdir()
    (tmp_path / "runtime/binding.py").write_text(
        f"from pathlib import Path\nPROJECT=Path('{tmp_path}/project')\n"
        f"SCIENCE=Path('{tmp_path}/science')\nGUARD=Path('{tmp_path}/guard')\n")
    auth = tmp_path / "auth.json"
    if write_auth:
        auth.write_text(json.dumps({"files_SHA256": auth_pins or {}}))
    (tmp_path / "calibration_deployment.json").write_text(json.dumps({"authorization_path": str(auth)}))
    return auth


def test_runtime_workspace_paths_are_read_without_importing_binding(tmp_path):
    auth = _fake_runtime_root(tmp_path, write_auth=False)
    missing = plugin.runtime_workspace_missing(tmp_path)
    assert missing == [str(tmp_path / "project"), str(tmp_path / "science"),
                       str(tmp_path / "guard/REVIEW_MANIFEST.json"), str(auth)]
    for path in ("project", "science", "guard"):
        (tmp_path / path).mkdir()
    (tmp_path / "guard/REVIEW_MANIFEST.json").write_text("{}")
    auth.write_text("{}")
    assert plugin.runtime_workspace_missing(tmp_path) == []


def test_calibration_inventory_must_be_the_pinned_path(tmp_path):
    inventory = tmp_path / "data/calibration_fold0/manifest.json"
    _fake_runtime_root(tmp_path, {"/elsewhere/data/calibration_fold0/manifest.json": "0" * 64})
    assert "/elsewhere/" in plugin.calibration_inventory_problem(tmp_path)
    (tmp_path / "auth.json").write_text(json.dumps({"files_SHA256": {str(inventory): "0" * 64}}))
    assert "absent" in plugin.calibration_inventory_problem(tmp_path)
    inventory.parent.mkdir(parents=True)
    inventory.write_text("{}")
    assert plugin.calibration_inventory_problem(tmp_path) is None


def test_runtime_skip_list_names_existing_tests():
    for node in plugin.NEEDS_CALIBRATION_INVENTORY:
        filename, cls, test = node.split("::")
        source = (ROOT / "runtime" / filename).read_text()
        assert f"class {cls}" in source and f"def {test}" in source, node


# --- Markdown links ---------------------------------------------------------

def test_link_checker_finds_broken_relative_links_only(tmp_path):
    (tmp_path / "docs").mkdir()
    (tmp_path / "docs/ok.md").write_text("x")
    page = tmp_path / "README.md"
    page.write_text("\n".join([
        "[ok](docs/ok.md#part) [web](https://example.org/x) [anchor](#here)",
        "[broken](docs/missing.md) ![img](docs/none.png)",
        "`[code](not/a/link.md)`",
        "```",
        "[fenced](also/not/a/link.md)",
        "```",
        "[local](runs/output.npz)",
        "[ref]: docs/gone.md",
        "[spaced](<docs/has space.md>) [spaced ok](<docs/ok.md>)",
    ]))
    broken, local = links.check([page], tmp_path)
    assert broken == ["README.md:2: docs/missing.md", "README.md:2: docs/none.png", "README.md:8: docs/gone.md",
                      "README.md:9: docs/has space.md"]
    assert local == ["README.md:7: runs/output.npz"]


def test_tracked_markdown_has_no_broken_relative_links():
    broken, _ = links.check(links.tracked_markdown(ROOT), ROOT)
    assert broken == []


# --- check_env --------------------------------------------------------------

def test_check_env_compares_exact_pins_includes_and_python(tmp_path):
    numpy = pytest.importorskip("numpy")
    pytest_version = pytest.__version__
    major_minor = ".".join(platform.python_version_tuple()[:2])
    (tmp_path / "base.txt").write_text(f"pytest=={pytest_version}\n")
    recipe = tmp_path / "recipe.txt"
    recipe.write_text("\n".join([
        f"# check_env: python=={major_minor}",
        "--extra-index-url https://example.invalid/simple",
        "-r base.txt",
        f"numpy=={numpy.__version__}+local",
        "definitely-not-installed-alz==1.0",
    ]))
    rows = check_env.run(recipe, tmp_path)
    status = {note.split(":")[0]: ok for ok, note in rows}
    assert status["python"] is True
    assert status["pytest"] is True
    assert status["numpy"] is False  # +local label differs
    assert status["definitely-not-installed-alz"] is False
    assert status["pip option (not compared)"] is None
    relaxed = {note.split(":")[0]: ok for ok, note in check_env.run(recipe, tmp_path, allow_local=True)}
    assert relaxed["numpy"] is True


def test_check_env_reads_the_isolated_native_target(tmp_path):
    target = tmp_path / "site"
    (target / "particlegan").mkdir(parents=True)
    info = target / "particlegan-0.10.0.dist-info"
    info.mkdir()
    (info / "METADATA").write_text("Metadata-Version: 2.1\nName: particlegan\nVersion: 0.10.0\n")
    (info / "direct_url.json").write_text(json.dumps(
        {"url": "https://github.com/HyperGAN/ParticleGAN.git", "vcs_info": {"vcs": "git", "commit_id": "952c3b62" + "0" * 32}}))
    ok, note = check_env.check_native("native particlegan==0.10.0 target=site commit=952c3b62", tmp_path)
    assert ok, note
    ok, note = check_env.check_native("native particlegan==0.10.0 target=site commit=deadbeef", tmp_path)
    assert not ok
    ok, note = check_env.check_native("native particlegan==0.9.7 target=site", tmp_path)
    assert not ok
    ok, note = check_env.check_native("native particlegan==0.10.0 target=absent", tmp_path)
    assert not ok and "no distribution" in note


def test_check_env_cli_exit_status(tmp_path):
    recipe = tmp_path / "r.txt"
    recipe.write_text(f"pytest=={pytest.__version__}\n")
    run = lambda: subprocess.run([sys.executable, "-B", str(ROOT / "scripts/check_env.py"), str(recipe)],
                                 capture_output=True, text=True)
    assert run().returncode == 0
    recipe.write_text("pytest==0.0.1\n")
    result = run()
    assert result.returncode == 1 and "FAIL pytest" in result.stdout



def run_plugin_on(tmp_path, files, *extra):
    """Run pytest with only the repository plugin on throwaway test files."""
    for name, body in files.items():
        (tmp_path / name).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / name).write_text(body)
    (tmp_path / "pytest.ini").write_text("[pytest]\n")
    env = dict(os.environ, PYTHONPATH=str(ROOT / "tests/support"), PYTHONDONTWRITEBYTECODE="1")
    return subprocess.run([sys.executable, "-B", "-m", "pytest", "-p", "alz_pytest_plugin", "-p", "no:cacheprovider",
                           "-rs", *extra, str(tmp_path)], cwd=tmp_path, env=env, capture_output=True, text=True).stdout


def test_optional_import_inside_a_test_body_is_a_skip_not_a_failure(tmp_path):
    out = run_plugin_on(tmp_path, {"paper_watch/tests/test_body.py": (
        "def test_needs_pandas():\n"
        "    raise ModuleNotFoundError(\"No module named 'pandas'\", name='pandas')\n\n"
        "def test_needs_repo_module():\n"
        "    raise ModuleNotFoundError(\"No module named 'nope'\", name='nope')\n")})
    assert "1 failed, 1 skipped" in out, out
    assert "optional dependency pandas is not installed (imported inside the test)" in out


def test_optional_import_at_module_level_is_a_skip_not_an_error(tmp_path):
    out = run_plugin_on(tmp_path, {
        "paper_watch/tests/test_mod.py": "import particlegan_absent_probe\n",
        "tests/test_gan.py": "raise ModuleNotFoundError(\"No module named 'torch'\", name='torch')\n"})
    assert "1 skipped, 1 error" in out, out
    assert "optional dependency torch is not installed" in out


def test_core_tests_never_skip_for_a_missing_optional_package(tmp_path):
    # A stray torch/scipy import reached from a core (numpy-only) test file must
    # fail, not turn into a skip; only OPTIONAL_DEPENDENCY_SCOPE files may skip.
    out = run_plugin_on(tmp_path, {
        "tests/test_microglia_default.py": "raise ModuleNotFoundError(\"No module named 'scipy'\", name='scipy')\n",
        "tests/test_repo_body.py": (
            "def test_body():\n"
            "    raise ModuleNotFoundError(\"No module named 'torch'\", name='torch')\n"),
        "tests/test_new_file.py": "raise ModuleNotFoundError(\"No module named 'pandas'\", name='pandas')\n"},
        "--continue-on-collection-errors")
    assert "1 failed" in out and "2 errors" in out and "skipped" not in out, out


def test_optional_dependency_scope_lists_existing_files():
    for scope in plugin.OPTIONAL_DEPENDENCY_SCOPE:
        assert (ROOT / scope).exists(), scope
    assert plugin.optional_dependency_allowed(ROOT / "paper_watch/tests/test_harness.py")
    assert plugin.optional_dependency_allowed(ROOT / "tests/test_gan.py")
    assert not plugin.optional_dependency_allowed(ROOT / "tests/test_microglia_default.py")
    assert not plugin.optional_dependency_allowed(ROOT / "tests/test_repo_docs.py")
