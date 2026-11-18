"""Run one experiment iteration for a named test and print JSON margins.

Usage: python3 scripts/run_once.py <test_module> <seed> <steps>
Example: python3 scripts/run_once.py test_seaad_vulnerability_readout 0 200
"""
import importlib
import json
import sys

sys.path.insert(0, ".")
sys.path.insert(0, "tests")

module = importlib.import_module(sys.argv[1])
seed = int(sys.argv[2]) if len(sys.argv) > 2 else 0
steps = int(sys.argv[3]) if len(sys.argv) > 3 else 300
print(json.dumps(module.run_once(seed=seed, steps=steps)))
