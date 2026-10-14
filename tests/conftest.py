import pytest


@pytest.fixture(autouse=True)
def _scope_root_is_tmp(tmp_path, monkeypatch):
    """Harness outputs must pass scoped(); tests run with the scope root set to their own tmp dir."""
    monkeypatch.setenv('ALZ_SCOPE_ROOT', str(tmp_path))
