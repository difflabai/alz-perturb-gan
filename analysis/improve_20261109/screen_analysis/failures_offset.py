"""failures_offset.py - what happened to the learnable stage x gene offset table?

For offset and covars,genemean,offset (hvg2000, 3 folds x 2 seeds), read final.pt on CPU:
  * init table (arm_state.offset_init, from offset_init) vs final generator stage_offset:
    per-stage RMS, RMS of the change, the part of the change common to all 4 rows (a gene-level
    shift the output bias could carry) vs the stage-contrast part (rows minus their mean);
  * correlation of the final High-minus-NotAD row difference with the raw (unshrunk) train
    donor-averaged stage log-ratio contrast (recomputed here from the train cache) and with the
    shrunk init;
  * the output-bias drift (final out[-1].bias minus the log pooled mean it was initialised to);
  * gradients at the final weights: 24 fresh generator-loss batches (training-shaped: 32 donors x
    128 cells from the train donor sampler) -> per-batch gradient of stage_offset; consistency
    ratio ||mean grad|| / mean ||grad|| (about 1/sqrt(24)=0.20 for pure noise, 1 for a consistent
    direction) and the same for out[-1].bias; size of one DualNorm polar step for the table.
Writes runs/.../screen_analysis/failures_offset.json.
"""
import json
import sys
from pathlib import Path

sys.dont_write_bytecode = True
import numpy as np
import torch

import failures_common as fc
import arms
import trainer
from pg_model import critic_features

OUT = Path('/ml2/alz/runs/improve_20261109/screen_analysis/failures_offset.json')
NB = 24


def raw_stage_logratio(ctx):
    C = torch.stack([ctx.X[torch.as_tensor(r)].sum(0) for r in ctx.by]).double().numpy()
    st = arms.stage_of(ctx.raw_c)
    a = 1.0; N = C.sum(1); G = C.shape[1]
    logq = np.log(C.sum(0) + a) - np.log(N.sum() + a * G)
    r = np.log(C + a) - np.log(N[:, None] + a * G) - logq
    m = r.mean(0)
    return np.stack([r[st == s].mean(0) - m if (st == s).any() else np.zeros(G) for s in range(4)])


def corr(a, b):
    a = a - a.mean(); b = b - b.mean()
    d = np.sqrt((a @ a) * (b @ b))
    return float(a @ b / d) if d > 0 else float('nan')


def grads(ctx, G, D, P, q, s):
    rec = trainer.recipe()
    loss = rec.make_loss()
    rng = np.random.default_rng(12345 + s)
    lat = torch.Generator().manual_seed(1); cell = torch.Generator().manual_seed(2); cnt = torch.Generator().manual_seed(3)
    torch.manual_seed(4)
    G.train(); D.eval(); D.requires_grad_(False)
    go, gb = [], []
    for _ in range(NB):
        _, c, lib = ctx.batch(rng)
        z, _ = P.sample(trainer.BATCH, generator=lat)
        rate = G.rates(z.detach(), c, trainer.BAG, lib, cell)
        counts = torch.poisson(rate.detach(), generator=cnt)
        f = ctx.features.fake(critic_features(counts, rate, ctx.mean, ctx.sd))
        lg = loss.g_loss(D(f, c))
        G.zero_grad()
        lg.backward()
        go.append(G.stage_offset.grad.detach().double().numpy().copy())
        gb.append(G.out[-1].bias.grad.detach().double().numpy().copy())
    go = np.stack(go); gb = np.stack(gb)
    mg = go.mean(0)
    # one polar (orthogonalised) step of the mean-direction, as DualNorm applies to a 2-D generator weight
    U, S, Vt = np.linalg.svd(go[0], full_matrices=False)
    polar = U @ Vt
    return {
        'offset_consistency': float(np.linalg.norm(mg) / np.mean([np.linalg.norm(g) for g in go])),
        'bias_consistency': float(np.linalg.norm(gb.mean(0)) / np.mean([np.linalg.norm(g) for g in gb])),
        'offset_grad_norm_mean': float(np.mean([np.linalg.norm(g) for g in go])),
        'bias_grad_norm_mean': float(np.mean([np.linalg.norm(g) for g in gb])),
        'offset_grad_row_norm_mean': np.mean([np.linalg.norm(g, axis=1) for g in go], 0).tolist(),
        'offset_grad_singular_values_batch0': S.tolist(),
        'polar_step_entry_rms_at_lr': float(rec.lr * np.sqrt((polar ** 2).mean())),
        'bias_step_entry_rms_at_lr': float(rec.lr / np.sqrt(gb.shape[1])),
        'mean_grad_contrast_corr_with_bias_grad': corr(mg[3] - mg[0], gb.mean(0)),
    }


def main():
    out = {}
    for arm in ['offset', 'covars,genemean,offset']:
        ctx = None
        for k in range(3):
            ctx = None
            for s in range(2):
                ctx, G, D, P, q, cfg = fc.load_fit(arm, k, s, ctx=ctx)
                init = q['arm_state']['offset_init'].double().numpy()
                fin = G.stage_offset.detach().double().numpy()
                raw = raw_stage_logratio(ctx)
                delta = fin - init
                common = delta.mean(0)
                contrast = delta - common
                bias0 = torch.log((ctx.mu.clamp_min(1e-5) / ctx.mu.clamp_min(1e-5).sum())).double().numpy()
                bdrift = G.out[-1].bias.detach().double().numpy() - bias0
                st = arms.stage_of(ctx.raw_c)
                n_s = [int((st == x).sum()) for x in range(4)]
                d = {'n_per_stage': n_s, 'tau2': cfg['arm_info']['offset']['tau2'],
                     'init_rms_per_stage': np.sqrt((init ** 2).mean(1)).tolist(),
                     'final_rms_per_stage': np.sqrt((fin ** 2).mean(1)).tolist(),
                     'delta_rms': float(np.sqrt((delta ** 2).mean())),
                     'delta_common_rms': float(np.sqrt((common ** 2).mean())),
                     'delta_contrast_rms': float(np.sqrt((contrast ** 2).mean())),
                     'final_hn_corr_raw_train_contrast': corr(fin[3] - fin[0], raw[3] - raw[0]),
                     'final_hn_corr_init_contrast': corr(fin[3] - fin[0], init[3] - init[0]),
                     'raw_train_hn_rms': float(np.sqrt(((raw[3] - raw[0]) ** 2).mean())),
                     'final_hn_rms': float(np.sqrt(((fin[3] - fin[0]) ** 2).mean())),
                     'final_hn_regression_slope_on_raw': float(np.polyfit(raw[3] - raw[0], fin[3] - fin[0], 1)[0]),
                     'bias_drift_rms': float(np.sqrt((bdrift ** 2).mean())),
                     'corr_common_delta_vs_bias_drift': corr(common, bdrift)}
                d.update(grads(ctx, G, D, P, q, s))
                out[f'{arm}|{k}|{s}'] = d
                print(arm, k, s, json.dumps({kk: (np.round(v, 4).tolist() if isinstance(v, (list, float)) else v) for kk, v in d.items()}), flush=True)
    OUT.write_text(json.dumps(out, indent=1))
    print('wrote', OUT)


if __name__ == '__main__':
    main()
