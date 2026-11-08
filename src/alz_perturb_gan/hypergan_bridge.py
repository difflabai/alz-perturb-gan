"""Isolated HyperGAN integration.

Released HyperGAN 1.0.0-rc2 is the git tag whose `setup.py` version is 1.0.6,
published on PyPI as `hypergan==1.0.6` on 2020-08-09. That release does not
import on this Python 3.12 stack. The failure, checked in this environment, is:

1. `import hypergan` raises `ModuleNotFoundError: No module named 'hyperchamber'`
   from `hypergan/layers/operation.py`. The package declares no `install_requires`.
   Its classifiers list Python 3.4–3.6 only. The README still describes a
   TensorFlow framework.
2. After installing the contemporary dependency `hyperchamber==0.3.1`, import
   raises `ModuleNotFoundError: No module named 'PIL'` from
   `hypergan/samplers/base_sampler.py`.
3. With Pillow present, the next import raises
   `ModuleNotFoundError: No module named 'torchvision'` from
   `hypergan/configurable_component.py`. `StandardGAN.build` also calls
   `torch.onnx.export(..., device='cuda')`.

Training therefore uses the develop branch, which is a rebuild rather than a
release. The commit pinned here is `01a3aaec45f49923922787b0412a30fd71a2c897`
(package version 2.0.0a1). It trains a conditional generator with ParticleGAN
0.8.0 inside HyperGAN's own loop. Install it only if you want this bridge:

    pip install 'hndl>=0.8.0' \\
      'hypergan @ git+https://github.com/HyperGAN/HyperGAN.git@01a3aaec45f49923922787b0412a30fd71a2c897'

The decision-model loop in `gan.py` does not import this module.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch

from .data import PerturbationSet

DEVELOP_COMMIT = "01a3aaec45f49923922787b0412a30fd71a2c897"
DEVELOP_VERSION = "2.0.0a1"

RC2_FINDINGS = {
    "git_tag": "1.0.0-rc2",
    "pypi_version": "1.0.6",
    "published": "2020-08-09",
    "importable_here": False,
    "errors": [
        "ModuleNotFoundError: No module named 'hyperchamber' (hypergan/layers/operation.py); no install_requires; classifiers list Python 3.4-3.6",
        "ModuleNotFoundError: No module named 'PIL' (hypergan/samplers/base_sampler.py) after hyperchamber==0.3.1",
        "ModuleNotFoundError: No module named 'torchvision' (hypergan/configurable_component.py) once Pillow is present",
    ],
    "develop_commit": DEVELOP_COMMIT,
    "develop_version": DEVELOP_VERSION,
    "develop_reason": (
        "1.0.0-rc2 cannot be imported on Python 3.12. The develop branch is the "
        "rebuild whose training loop uses ParticleGAN 0.8.0. It is not a PyPI release."
    ),
}


class TensorConditionSource:
    """HyperGAN data factory: batches of (control, embedding) -> real outcome.

    HyperGAN calls `source(batch_size, generator=...)`. `resume_stateless`
    tells its checkpoint code that the sampler has no cursor to store.
    """

    resume_stateless = True

    def __init__(self, path: str):
        blob = torch.load(path, map_location="cpu", weights_only=True)
        self.control = blob["control"].float()
        self.outcome = blob["outcome"].float()
        self.embedding = blob["embedding"].float()
        if not (len(self.control) == len(self.outcome) == len(self.embedding)):
            raise ValueError("condition table tensors must share a batch dimension")
        self.path = str(path)

    def __call__(self, batch_size: int, *, generator):
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        index = torch.randint(self.control.shape[0], (batch_size,), generator=generator)
        control = self.control[index]
        embedding = self.embedding[index]
        return {
            "condition": torch.cat([control, embedding], dim=-1),
            "real": self.outcome[index],
        }

    def resume_identity(self) -> dict:
        return {
            "kind": "tensor-condition",
            "rows": int(self.control.shape[0]),
            "genes": int(self.control.shape[1]),
            "embedding": int(self.embedding.shape[1]),
        }


def save_condition_table(data: PerturbationSet, path: str | Path) -> Path:
    """Write the training cells HyperGAN's factory reads."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "control": data.cell_control,
            "outcome": data.cell_outcome,
            "embedding": data.cell_embedding,
        },
        path,
    )
    return path


def describe_hypergan() -> dict:
    """Say which HyperGAN, if any, this interpreter can import."""
    try:
        import hypergan
    except Exception as exc:
        return {
            "status": "unavailable",
            "error": f"{type(exc).__name__}: {exc}",
            "rc2": RC2_FINDINGS,
        }
    version = getattr(hypergan, "__version__", None)
    if version == DEVELOP_VERSION or (isinstance(version, str) and version.startswith("2.")):
        return {
            "status": "develop",
            "version": version,
            "commit": DEVELOP_COMMIT,
            "rc2": RC2_FINDINGS,
        }
    return {
        "status": "unexpected",
        "version": version,
        "rc2": RC2_FINDINGS,
        "error": "This bridge expects HyperGAN develop (2.0.0a1), not the 1.0.0-rc2 package.",
    }


def write_recipe(directory: str | Path, data: PerturbationSet, *, steps: int = 5, batch_size: int = 8) -> Path:
    """Write a conditional HyperGAN recipe aimed at this perturbation set."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    table = save_condition_table(data, directory / "cells.pt")
    genes = data.n_genes
    condition = genes + data.embedding_dim
    hidden = 32
    # The path is absolute so HyperGAN can load the factory from any cwd.
    recipe = f'''# Learning-experiment conditional perturbation GAN. Not a qualified image recipe.
schema_version = 1
name = "alz-perturb-gan/condition"

[data]
factory = "alz_perturb_gan.hypergan_bridge:TensorConditionSource"
[data.args]
path = {json.dumps(str(table.resolve()))}

[components.generator]
factory = "hndl"
inputs = {{ x = "latent", condition = "batch.condition" }}
[components.generator.args]
input_shape = {{ x = ["B", 4], condition = ["B", {condition}] }}
output_shape = ["B", {genes}]
source = """
concat(x, condition)
linear({hidden})
leaky_relu(0.2)
linear({hidden})
leaky_relu(0.2)
linear()
"""

[components.discriminator]
factory = "hndl"
inputs = {{ x = "candidate", condition = "batch.condition" }}
[components.discriminator.args]
input_shape = {{ x = ["B", {genes}], condition = ["B", {condition}] }}
output_shape = ["B", 1]
source = """
concat(x, condition)
linear({hidden})
leaky_relu(0.2)
linear({hidden})
leaky_relu(0.2)
linear()
"""

[[objectives]]
factory = "mse"
inputs = {{ input = "generated", target = "batch.real" }}
weight = 1.0
detach = ["target"]

[prior]
kind = "particles"
[prior.args]
num_particles = 64
z_dim = 4

[prior_regularizer]
weight = 0.0

[optimizer]
lr = 0.001
d_lr_mult = 1.0
prior_lr_mult = 1.0

[training]
steps = {int(steps)}
batch_size = {int(batch_size)}
seed = 42
device = "cpu"
# Short smokes sample the EMA generator. The default decay of 0.995 barely
# moves in a handful of updates, so this fixture copies the live weights.
ema = 0.0

[sampling]
count = 8
seed = 123
'''
    path = directory / "config.toml"
    path.write_text(recipe)
    return path


def train_condition_gan(data: PerturbationSet, run_dir: str | Path, *, steps: int = 5, batch_size: int = 8) -> dict:
    """Run HyperGAN develop on this dataset. Raises if develop is not installed."""
    info = describe_hypergan()
    if info["status"] != "develop":
        raise RuntimeError(info.get("error") or f"HyperGAN develop is not active ({info['status']})")
    from hypergan.training import train

    run_dir = Path(run_dir)
    config = write_recipe(run_dir.parent / (run_dir.name + "-config"), data, steps=steps, batch_size=batch_size)
    summary = train(str(config), str(run_dir), steps=steps, checkpoint_every=max(steps, 1), preview_every=0)
    manifest_path = run_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.is_file() else {}
    return {
        "status": manifest.get("status"),
        "steps": manifest.get("steps"),
        "run_dir": str(run_dir),
        "config": str(config),
        "hypergan": info["version"],
        "summary_keys": sorted(summary)[:12] if isinstance(summary, dict) else [],
    }


def predict_conditions(run_dir: str | Path, condition: torch.Tensor, *, draws: int = 4) -> torch.Tensor:
    """Average several HyperGAN particle draws for a batch of conditions."""
    from hypergan.artifacts import sample

    run_dir = Path(run_dir)
    count = int(condition.shape[0])
    total = None
    for seed in range(draws):
        output = run_dir / f"predict-seed{seed}.json"
        if output.exists():
            output.unlink()
        written = sample(str(run_dir), count=count, seed=seed, output=str(output), inputs={"condition": condition})
        payload = json.loads(Path(written).read_text())
        draw = torch.tensor(payload["samples"], dtype=torch.float32)
        total = draw if total is None else total + draw
    return total / draws
