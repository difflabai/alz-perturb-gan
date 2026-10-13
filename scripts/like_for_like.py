"""Like-for-like decision-head check from a saved real_data checkpoint (CPU only).

Scores TRAINING and HELD-OUT target pseudobulks with identical inputs and
protocol: every library candidate represented by the generator's averaged
predicted profile (`--draws` draws, seeded), label = candidate == target,
via `harness.decision_pair_report` (the same function behind the training-gene
check `decision_label_report` and the held-out `decision_heldout_report`).
"""

import argparse
import json
from pathlib import Path

import torch

from alz_perturb_gan.gan import train_gan_decision
from alz_perturb_gan.harness import decision_pair_report
from alz_perturb_gan.real_data import clean_json, load_real_data


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run-dir", type=Path, required=True)
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    cmd = json.loads((a.run_dir / "command.json").read_text())
    ckpt = torch.load(a.run_dir / "checkpoint.pt", map_location="cpu", weights_only=False)
    data, _ = load_real_data(a.data, split_seed=cmd["seed"], cell_seed=cmd["seed"] + 1,
                             train_cells=cmd["train_cells"], eval_cells=cmd["eval_cells"],
                             eval_fraction=cmd["eval_fraction"], input_scale=cmd["input_scale"])
    # Build modules with the run's shapes (one throwaway step), then load the trained weights.
    gan = train_gan_decision(data, device="cpu", seed=cmd["seed"], steps=1, batch_size=cmd["batch_size"],
                             hidden=cmd["hidden"], lr=cmd["lr"], n_particles=256, z_dim=16)
    for key in ["generator", "critic", "decision", "prior"]:
        getattr(gan, key).load_state_dict(ckpt[key])
    library = [item.name for item in data.named()]
    out = {"checkpoint": str(a.run_dir / "checkpoint.pt"), "draws": cmd["draws"], "seed": cmd["seed"] + 3}
    torch.manual_seed(cmd["seed"] + 3)  # same seed decision_heldout_report used
    predicted = dict(zip(library, gan.predict_profiles(data, library, n_draws=cmd["draws"])))
    for split in ["train", "eval"]:
        out[split] = decision_pair_report(data, gan, data.named(split), predicted,
                                          seed=cmd["seed"] + 3, split=split + "_targets")
    a.out.write_text(json.dumps(clean_json(out), indent=2) + "\n")
    print(json.dumps(clean_json(out), indent=2))


if __name__ == "__main__":
    main()
