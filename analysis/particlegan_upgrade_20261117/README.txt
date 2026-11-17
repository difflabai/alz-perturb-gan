ParticleGAN0.10.0 isolated ALZ release test

Official v0.10.0 (Nov16), commit952c3b62ba434f7da194067923351ff5748ef8dd.
release_receipt.json pins the unmodified tagged sources and locally built wheel.
No dependencies or global environment changed. Existing default remains
microglia-pbulk-countshape-v2 on0.9.7; old training/evaluation sources remain
frozen. The covariance study finished FAIL and is independently recorded.

Two fixed fold0/seed0 fresh20k pilots:
  compat: v0.10.0, same explicit BISPF/T005 recipe as current model.
  rhbi: explicitly adapted rhbi_optimized_alz_conditional_no_simplex.
RHBI uses native optimized rates/penalty/prior/conditional transport. Its
unconditional simplex chart has no defined condition/library assignment for
ALZ; public override rhbi_simplex_weight=0 is explicit and separately named.
This does not test or qualify the exact upstream default. See PROTOCOL.txt,
resolved_recipes.json, rhbi_adapter.py and host.py for exact boundaries.

The host loader derives isolated trainer/generator copies under ignored
artifacts/native-releases/particlegan-0.10.0/alz-host-{compat,rhbi}; it records
original and derived SHA256 values. Native library code is never patched.
Each phase runs in a fresh process. Never import two release hosts together.

Verification includes actual TRAIN-only GPU calls, full old/new init+5update
parity,300fresh+100same-process continuation per new lane, and100more calls
from a fresh subprocess per lane. RHBI checks actual differentiable conditional
transport with2150features+separate7conditions, shared records, forbidden
simplex calls, incompatible-checkpoint rejection, exact full snapshot/RNG
replay, three-thread affinity and.04GPU allocator limit. Pin source/proof
hashes in source_manifest.json/prelaunch.json before any full pilot.

Start/reconnect the fixed queue (one ALZ GPU1 worker, shared with others):
  /mnt/nvme/ai/conda/anaconda3/bin/python3 -B analysis/particlegan_upgrade_20261117/run_pilot.py
It performs train/CPU generation+scoring serially for both lanes. Check
runs/particlegan_upgrade_20261117/status.json and logs/; valid terminal
receipts/hash links are required for resume. Partial scores fail closed.
Outputs have a1.5GB cap; scored generated arrays are hashed and pruned.

Scoring uses identical late six-state readout, TRAIN joint calibration and
covariance-control CRN namespace, allowing comparison with verified retained
0.9.7control and empirical scores. pilot_comparison.json records every result
and full20k compatibility identity. A single development fold/seed provides
no population/default qualification. No sealed or external donors are used.
