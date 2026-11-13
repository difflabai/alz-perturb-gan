"""review_failures_regen_check.py - do the 12:39 re-generated generated.npz files (offset, donorpc,
covars+genemean+offset, generated after the trainer.py restore fix) agree with a fresh CPU generation from final.pt?

For fold 0 seed 0 of donorpc, offset, covars,genemean,offset (and covars, generated before the fix, as control):
fresh CPU generation (own random streams, same M=32, K=256, target-library sampling) for all 37 targets, then
mean held-out / train pb_rmse vs the stored generated.npz scored the same way.  Different RNG streams, so
agreement is expected within Monte-Carlo error (~0.01-0.03), not exactly.
"""
import json
import os
import sys

sys.dont_write_bytecode = True
os.environ['CUDA_VISIBLE_DEVICES'] = ''
sys.path.insert(0, '/ml2/alz/analysis/improve_20261109')
sys.path.insert(0, '/ml2/alz/analysis/improve_20261109/screen_analysis')
import numpy as np
import torch

import trainer
from review_failures_common import CACHES, FITS, OUTDIR, load_gen, load_real, logcp
from review_failures_covars_cf import gen

torch.set_num_threads(6)


def main():
    R = load_real(0)
    obs = logcp(R['pb'])
    out = {}
    for arm in ['donorpc', 'offset', 'covars,genemean,offset', 'covars']:
        flags = tuple(arm.split(','))
        ctx = trainer.Context(CACHES / 'hvg2000' / 'fold0', 'cpu', flags)
        q = torch.load(FITS / arm / 'hvg2000' / 'fold0' / 'seed0' / 'final.pt', map_location='cpu', weights_only=False)
        b, *_ = trainer.build(ctx, 0, stream_root=q['stream_root'])
        G, P = b[1], b[3]
        G.load_state_dict(q['policy']['models']['generator']); P.load_state_dict(q['policy']['models']['prior']); G.eval(); P.eval()
        cond = trainer.build_conditions(R['raw'], flags, ctx.age_mean, ctx.age_sd)
        g = load_gen(arm, 0, 0, donors=list(R['donors']))
        fresh, stored = [], []
        for i in range(len(R['donors'])):
            pb = gen(G, P, np.repeat(cond[i][None], 32, 0), R['cells'][i].sum(1), 555000 + i)
            fresh.append(np.sqrt(np.mean((logcp(pb).mean(0) - obs[i]) ** 2)))
            stored.append(np.sqrt(np.mean((logcp(g['pb_counts'][i]).mean(0) - obs[i]) ** 2)))
        fresh, stored = np.array(fresh), np.array(stored)
        out[arm] = {'fresh_heldout': float(fresh[R['held']].mean()), 'stored_heldout': float(stored[R['held']].mean()),
                    'fresh_train': float(fresh[~R['held']].mean()), 'stored_train': float(stored[~R['held']].mean()),
                    'per_donor_corr': float(np.corrcoef(fresh, stored)[0, 1]), 'max_abs_diff': float(np.abs(fresh - stored).max())}
        print(arm, {a: round(v, 4) for a, v in out[arm].items()}, flush=True)
    (OUTDIR / 'review_failures_regen_check.json').write_text(json.dumps(out, indent=1))


if __name__ == '__main__':
    main()
