ParticleGAN0.10.0 EMPF comparison (owner request, Nov17)

Fixed native EMPF and a separately named EMPF variant matching RHBI's shared
hyperparameters. Same unmodified v0.10.0 installation, ALZ architecture/data,
initial G/D/prior parameters,11streams, fold0/seed0,20k updates, six late model
states, calibration and matched M32K256 CPU readout as the completed release
pilot. This is a development pilot, with no automatic default promotion.

Both use reg_every=lazy_k=1: the regularizer runs every update. Verification
checks actual native calls=critic observed steps=completed updates, and each
full fit must also supply a final-checkpoint no_skip_audit.json.

Native preset: G.012,D.018,prior.03,prior-reg.01,EBI6,smoothing.001,
batched SVD,newton_schulz_hybrid_v1. Matched: G.014,D.021,prior.0406,
prior-reg.02,EBI12,smoothing.0061,native SVD,polar_cpu_v1,float32.
All163 fields were compared; exactly8 shared fields differ between the native
and matched EMPF settings. Shared fields in the matched lane equal RHBI;
EMPF retains its own formulation/prior and all RHBI-specific fields disabled.
See resolved_recipes.json and hyperparameter_match_audit.json.

The native high-dimensional conditioned route has no prototype/chord bank;
its bank branch is restricted to low-dimensional inputs. Continuous2150
features and separate7conditions are passed intact. No artificial prototypes
or mechanism changes. Native convolution-only bf16 autocast is inactive for
the fixed ALZ MLP. Native non-saturating GANLoss is retained.

Code/recipe/protocol/ancestry source manifest345files:
6e09a80b60b49d53be13f9e50b0bcd34c90cfcdce0313d93c25e92e9981c2958.
Source review found no blocker. Actual TRAIN-only verification requires per
lane300fresh+100same-process replay+100fresh-process replay updates, exact full
snapshots/metrics and incompatible-header rejection before full fitting.

After prelaunch.json records all four proof hashes, run/reconnect:
 /mnt/nvme/ai/conda/anaconda3/bin/python3 -B analysis/empf_comparison_20261117/run.py
It uses one shared GPU1 worker with.04allocator and3CPU threads/affinity,
then CPU generation/scoring, serially for each lane. Own run root:
runs/empf_comparison_20261117; status.json, logs/{train,generate}_{native,matched}.log.
Runtime cap1.5GB. Generated arrays hashed/scored/pruned, all weights retained.
No scheduler admission or waiting for a GPU to become free. Other jobs untouched.

All previous326/330 sources, completed receipts and default remain unchanged.
Preceding release pilot completed and was committed9fb2372: exact old-recipe
compatibility; RHBI profile-0.7814%,SDscore+2.3357%,combined-0.0638% vs current.
Empirical still wins both primary scores. Both final checkpoints contain
exactly20,000 penalty calls for20,000 updates; no skipping occurred.
