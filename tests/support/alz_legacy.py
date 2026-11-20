"""Guards for the legacy synthetic smoke tests (src/alz_perturb_gan).

Those tests are marked ``legacy_pg080`` and deselected by default. When they
are selected, they need torch and ParticleGAN 0.8.x, the version pinned in
pyproject.toml. The native 0.10.0 release used by the microglia model is not a
substitute, so a different version is a skip with a reason, not a failure.
"""
import pytest

LEGACY_PG = "0.8."
TORCH_REASON = "legacy_pg080 tests need torch (envs/legacy-pg080.txt)"


def require_legacy_particlegan():
    particlegan = pytest.importorskip(
        "particlegan", reason="legacy_pg080 tests need particlegan==0.8.0 (envs/legacy-pg080.txt)")
    version = getattr(particlegan, "__version__", None)
    if version is None:
        try:
            from importlib.metadata import version as dist_version
            version = dist_version("particlegan")
        except Exception:
            version = "unknown"
    if not str(version).startswith(LEGACY_PG):
        pytest.skip(
            f"legacy_pg080 tests need particlegan 0.8.x; found {version} "
            f"from {getattr(particlegan, '__file__', '?')} (envs/legacy-pg080.txt)",
            allow_module_level=True,
        )
    return particlegan
