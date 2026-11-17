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

## What the ALZ results establish

The two readouts used the same trained weights. Reducing within-bag contraction from 0.5 to 0.25 and refitting the existing training-donor count calibration increased raw SD coverage from 58.3% to 62.2% and combined coverage from 69.2% to 71.5%. The subsequent interval calibration reached 86.7% and 83.7% by widening reported SD bounds. The target was 80%. This one-fold, one-seed pilot on 12 held-out development donors provides evidence about the readout and intervals. The contribution of the trainer to undercoverage remains unmeasured. Empirical resampling draws previously observed cells and serves as a separate baseline. [ALZ measurements](../coverage_empf_20261117/README.md)

The earlier training comparison favored native EMPF over the version with RHBI's shared settings on all three headline errors. Eight settings changed together. Both completed fits recorded 20,000 regularizer calls in 20,000 updates. Those results support retaining the original preset as the control for an upstream comparison. [Training comparison](../empf_comparison_20261117/README.txt)
