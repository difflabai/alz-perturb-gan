"""Shared test isolation.

scripts/microglia_runtime.py holds a shared source-epoch lease under runs/.source-epoch of the checkout
for its lifetime. Tests must never create it in the checkout (or contend with a refreeze there), so every
test gets its own epoch directory through MICROGLIA_EPOCH_DIR; subprocesses inherit it.
"""
import pytest


@pytest.fixture(autouse=True)
def _isolated_source_epoch(tmp_path_factory, monkeypatch):
    monkeypatch.setenv('MICROGLIA_EPOCH_DIR', str(tmp_path_factory.mktemp('source-epoch')))
