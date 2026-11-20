# ParticleGAN v0.10.0 review of conditional variability

**Finding.** EMPF's transport residual can report a perfect match after two conditions exchange their entire spreads. In the released continuous-vector path, widths up to 64 can use the conditional discrepancy. At width 65 the transport observer switches to a pooled bidirectional nearest-neighbor distance. A permutation of the same rows across conditions preserves that distance. The critic still receives the condition, and the residual feeds the regularizer's settling calculation. The effect on trained samples requires a training comparison. [Release source, branch selection](https://github.com/255BITS/ParticleGAN/blob/952c3b62ba434f7da194067923351ff5748ef8dd/particlegan/_empf/penalty.py#L318-L368) · [Critic and penalty calculation](https://github.com/255BITS/ParticleGAN/blob/952c3b62ba434f7da194067923351ff5748ef8dd/particlegan/_empf/penalty.py#L424-L469)

## Reproducing the observation

The CPU probe calls the public EMPF penalty with continuous input, 32 rows, seven condition columns, and regularization on every step. Two balanced groups have zero means and standard deviations of 0.5 and 2.0 on one coordinate. The damaged batch exchanges their spreads while preserving the entire pooled sample. Each case uses a fresh, identically initialized critic and penalty. All extra input columns are zero. A second control halves both groups' spreads, changing the pooled distribution. [Standalone probe](probe_conditional_spread.py)

| Input width | Correct groups | Swapped spreads | Both spreads halved |
| ---: | ---: | ---: | ---: |
| 64 | 0 | 0.931209 | 0.617582 |
| 65 | 0 | 0 | 0.586302 |
| 2,150 | 0 | 0 | 0.586302 |

Values are the reported `ema_residual` after one penalty call. Each call produced finite, nonzero critic weight gradients. The frozen v0.10.0 installation and the source checkout at `952c3b62` gave the same values. From the ALZ repository in its research Python environment, run

```bash
PYTHONPATH=artifacts/native-releases/particlegan-0.10.0/site-packages \
  python -B analysis/particlegan_review_20261117/probe_conditional_spread.py
```

## What to test upstream

1. **Condition-aware residual.** For a candidate that claims to track conditional fit at wide widths, require the spread swap to score worse than the correct pairing at 64, 65 and 2,150 features. Include a control that reorders paired rows and conditions together. Preserve the existing no-bank behavior for wide conditioned inputs. The release also infers categorical inputs for rank-two width 170 and above in automatic mode. Document the explicit continuous setting for EMPF and exercise it in the public API tests. The existing width and domain tests explicitly exercise BISPF. [Domain routing](https://github.com/255BITS/ParticleGAN/blob/952c3b62ba434f7da194067923351ff5748ef8dd/particlegan/_empf/penalty.py#L170-L186) · [Existing tests](https://github.com/255BITS/ParticleGAN/blob/952c3b62ba434f7da194067923351ff5748ef8dd/tests/test_bispf_input_domain.py)
2. **Learned quality.** Compare native EMPF with one change confined to how its wide continuous path measures conditional discrepancy. Train both on a synthetic grouped distribution with known conditional means and spreads. Fix the architecture, initialization, minibatches, hyperparameters, sampling and readout, including `lazy_k=1`. Score raw generated draws against independent groups using group mean error, group SD error, 80% interval coverage, width and interval score. Keep the correct-sampler and swapped-spread controls. Verify restored training reproduces the uninterrupted trajectory. The existing unequal-width and conditional Gaussian chain tasks supply additional regression checks. [Forge tasks](https://github.com/255BITS/ParticleGAN/blob/952c3b62ba434f7da194067923351ff5748ef8dd/reports/forge/problems.md)

### Local regression test

`tests/test_empf_conditional_probe.py` runs this probe against the frozen 0.10.0 site-packages in a CPU subprocess. Passing cases pin the zero residual for the correct pairing at all three widths, the 0.931209 swapped residual at width 64, and the halved-spread control values 0.617582 and 0.586302. Two strict expected failures assert that the swap scores above the correct pairing at widths 65 and 2,150. They record the upstream requirement against the pinned 0.10.0 artifact only: the test refuses any other version or location, so a newer release produces setup errors rather than unexpected passes. To check a candidate release, run the probe against its site-packages or copy the test with a new version pin. The test skips with a reason when the site-packages directory is absent (`ALZ_PARTICLEGAN_0_10_SITE_PACKAGES` overrides the location). It checks the reported residual for one deterministic batch, not trained samples.

### Draft upstream issue (not filed)

This text is a draft. File it only after the repository owner or a coordinator agrees.

> **EMPF: wide continuous residual is invariant to swapping conditional spreads (v0.10.0)**
>
> With `get_recipe("empf")`, `input_domain="continuous"` and a conditioned critic, the reported `ema_residual` distinguishes a batch whose two conditions have exchanged their spreads at 64 features (0.931209 against 0 for the correct pairing). At 65 and 2,150 features it reports 0 for both; in the 0.10.0 source any continuous width above 64 takes the same unconditioned chamfer path. The wide path uses a pooled bidirectional nearest-neighbor distance, and a cross-condition permutation of the same rows leaves that distance unchanged. A control that halves both spreads is detected at every width (0.617582 at 64; 0.586302 at 65 and 2,150). The repro uses CPU only, 32 rows, seven condition columns and one penalty call per case: [probe](probe_conditional_spread.py).
>
> The critic still receives the condition, and the residual feeds the regularizer's settling calculation. We have not measured an effect on trained samples.
>
> Suggested tests: (1) the spread swap scores worse than the correct pairing at 64, 65 and 2,150 features, plus a control that reorders rows and conditions together; (2) existing behavior without a bank is preserved for wide conditioned inputs; (3) a public API test exercises EMPF with explicit `input_domain="continuous"` at rank-two width 170 and above, where automatic mode infers categorical input. A learned-quality comparison (item 2 above) would show whether a change matters for training.

## What the ALZ results establish

The two readouts used the same trained weights. Reducing within-bag contraction from 0.5 to 0.25 and refitting the existing training-donor count calibration increased raw SD coverage from 58.3% to 62.2% and combined coverage from 69.2% to 71.5%. The subsequent interval calibration reached 86.7% and 83.7% by widening reported SD bounds. The target was 80%. This one-fold, one-seed pilot on 12 held-out development donors provides evidence about the readout and intervals. The contribution of the trainer to undercoverage remains unmeasured. Empirical resampling draws previously observed cells and serves as a separate baseline. [ALZ measurements](../coverage_empf_20261117/README.md)

The earlier training comparison favored native EMPF over the version with RHBI's shared settings on all three headline errors. Eight settings changed together. Both completed fits recorded 20,000 regularizer calls in 20,000 updates. Those results support retaining the original preset as the control for an upstream comparison. [Training comparison](../empf_comparison_20261117/README.txt)
