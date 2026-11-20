"""Check whether a cached or generated gene panel covers the LL modules."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

LIPID = ("APOE", "ABCA1", "SOAT1", "ACSL1", "ACSL3", "ACSL4", "PLIN2")
LYSO = ("NPC1", "NPC2", "LIPA", "GBA", "CTSD", "CTSB", "CTSS", "LAMP1", "LAMP2")


def panel_coverage(path: Path) -> dict:
    with np.load(path, allow_pickle=False) as archive:
        genes = archive["genes"]
    if genes.ndim != 1 or genes.dtype.kind != "U":
        raise ValueError("expected a one-dimensional gene-symbol array")
    panel = set(genes)
    modules = {
        name: {
            "present": [gene for gene in markers if gene in panel],
            "missing": [gene for gene in markers if gene not in panel],
        }
        for name, markers in (("lipid", LIPID), ("lysosome", LYSO))
    }
    return {
        "panel_size": len(genes),
        "full_module_coverage": all(not module["missing"] for module in modules.values()),
        **modules,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("panels", type=Path, nargs="+", help="space.npz or generated NPZ files")
    args = parser.parse_args()
    try:
        report = {str(path): panel_coverage(path) for path in args.panels}
    except (OSError, KeyError, ValueError) as exc:
        parser.error(str(exc))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
