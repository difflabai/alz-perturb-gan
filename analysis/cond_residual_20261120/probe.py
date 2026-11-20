"""Spread-swap probe: native EMPF residual vs the condition-aware adapter.

Reuses the review's fixture (analysis/particlegan_review_20261117) unchanged:
two balanced groups, SD 0.5 vs 2.0 on one coordinate, widths 64/65/2150, a
fresh identically initialised critic and penalty per case. Adds a paired-row
reorder control: real, fake and condition rows permuted together.
Run:  PYTHONPATH=<release site-packages> python -B probe.py [--json out.json]
"""
import argparse
import importlib.util
import json
from pathlib import Path
import sys

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import cond_residual as CR  # noqa: E402

_spec = importlib.util.spec_from_file_location(
    'review_probe', HERE.parent / 'particlegan_review_20261117/probe_conditional_spread.py')
RP = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(RP)
WIDTHS = (64, 65, 2150)


def residual(real, fake, condition, adapter):
    from particlegan import get_recipe
    recipe = get_recipe('empf', reg_every=1, batch_size=real.shape[0])
    critic = RP.ConditionalCritic(real.shape[1])
    optimizer = recipe.make_critic_optimizer(critic)
    penalty = recipe.make_critic_penalty(optimizer, input_domain='continuous', collect_stats=True)
    if adapter:
        CR.install(penalty)
    value = penalty(critic, real, fake, condition)
    value.backward()
    g = critic.linear.weight.grad
    assert penalty.last_stats['applied'] and optimizer.record.calls == 1
    assert g is not None and torch.isfinite(g).all() and g.norm() > 0
    return dict(ema_residual=penalty.last_stats['ema_residual'], penalty=float(value.detach()),
                grad=g.detach().clone(), cond_calls=getattr(penalty.regularizer, 'cond_calls', None))


def table():
    torch.set_num_threads(1)
    perm = torch.randperm(32, generator=torch.Generator().manual_seed(7))
    rows = []
    for width in WIDTHS:
        real, swapped, condition = RP.fixture(width)
        cases = {'correct': (real, real, condition), 'swapped': (real, swapped, condition),
                 'halved': (real, real * 0.5, condition),
                 'correct_reordered': (real[perm], real[perm], condition[perm]),
                 'swapped_reordered': (real[perm], swapped[perm], condition[perm])}
        for adapter in (False, True):
            out = {k: residual(*v, adapter) for k, v in cases.items()}
            rows.append(dict(width=width, arm='adapter' if adapter else 'native',
                             **{k: v['ema_residual'] for k, v in out.items()},
                             cond_calls={k: v['cond_calls'] for k, v in out.items()},
                             penalty_correct=out['correct']['penalty'], penalty_swapped=out['swapped']['penalty']))
    return rows


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--json')
    a = p.parse_args()
    rows = table()
    print('width arm      correct  swapped   halved   correct_reord swapped_reord')
    for r in rows:
        print(f"{r['width']:5} {r['arm']:8} {r['correct']:.6f} {r['swapped']:.6f} {r['halved']:.6f} "
              f"{r['correct_reordered']:.6f}      {r['swapped_reordered']:.6f}")
    if a.json:
        Path(a.json).write_text(json.dumps(rows, indent=2) + '\n')
