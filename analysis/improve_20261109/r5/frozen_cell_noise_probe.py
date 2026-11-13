"""Separate frozen network cell heterogeneity from Gamma-Poisson count noise.

Local interventions only, NOT trained or calibrated candidate recipes. Use
unchanged calibrated pbulk weights, matched donor latents, conditions, library
draws and cell-noise vectors. Vary only the within-bag mean heterogeneity or
NB dispersion before drawing the first 64 cells. No saved raw cell tensors.
Reads only the 37-donor development caches. Never updates a default or fit.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.dont_write_bytecode = True
import numpy as np
import torch
import trainer as T
import generate as G
import evaluate as E
import mode_biascal as MB
from mode_cellshape import shrink_mu
from interval_architecture_audit import summarize_bounds

RUNS = T.RUNS
OUT = RUNS / 'results_r5/interval_architecture'
VARIANTS = ('baseline', 'mean_contract_half', 'mean_contract_full', 'theta_times4', 'poisson_limit')


@torch.no_grad()
def run(fold, seed, device):
    fit = RUNS / f'fits_r3/pbulk/hvg2000/fold{fold}/seed{seed}'
    cache = RUNS / f'caches/hvg2000/fold{fold}'
    cfg = json.loads((fit / 'config.json').read_text())
    ctx = T.Context(cache, device, tuple(cfg['flags']))
    targets = G.targets(ctx)
    states = G.available_states(fit, cfg)
    plans, _, _ = G.mode_plans(ctx, 'stagehalf', targets, states, updates=cfg['updates'])
    names = [n for n in states if any(n in group for _, group in plans)]
    models = G.load_models(ctx, fit, cfg, names)
    receipt = json.loads((fit / 'generated_biascal@stagehalf.receipt.json').read_text())
    bias = torch.as_tensor(receipt['mode_extra']['biascal']['delta'], dtype=torch.float32, device=device)
    for model, prior in models.values():
        MB.head(model).add_(bias)
        model.eval(); prior.eval()
    summaries = {variant: np.zeros((len(targets), 32, 100)) for variant in VARIANTS}
    order = T.development_order()
    for target_index, (donor, _, _, pool) in enumerate(targets):
        streams = G.generation_streams(fold, order[donor])
        lat = torch.Generator(device=device).manual_seed(streams['latent'])
        cell = torch.Generator(device=device).manual_seed(streams['cell'])
        libraries = np.random.default_rng(streams['library']).choice(pool, size=(32, 256), replace=True).astype(np.float32)
        conditions, selected = plans[target_index]
        start = 0
        while start < 32:
            stop = start + 1
            while stop < 32 and selected[stop] == selected[start] and stop - start < G.CHUNK:
                stop += 1
            model, prior = models[selected[start]]
            z, _ = prior.sample(stop - start, generator=lat)
            lib = torch.as_tensor(libraries[start:stop], device=device)
            c = torch.as_tensor(conditions[start:stop], device=device)
            mu, theta = model.parameters_nb(z, c, 256, lib, cell)
            for variant in VARIANTS:
                mean = shrink_mu(mu, .5 if variant == 'mean_contract_half' else 1) if variant.startswith('mean_contract') else mu
                mean = mean[:, :64]
                shape = theta * (4 if variant == 'theta_times4' else 1)
                local_seed = E.MX.stable_seed('frozen-cell-noise-probe', fold, seed, donor, start)
                # Same random stream reset for each intervention; no optimizer RNG exists here.
                torch.manual_seed(local_seed)
                rate = mean if variant == 'poisson_limit' else torch.distributions.Gamma(shape, shape / mean).sample()
                count_gen = torch.Generator(device=device).manual_seed(local_seed + 1)
                counts = torch.poisson(rate, generator=count_gen)
                pcs = ctx.project(counts).cpu().numpy().astype(np.float16).astype(np.float32)
                summaries[variant][target_index, start:stop] = E.MX.pc_bag_summary(pcs)
            start = stop
    real = E.load_real(cache)
    indices = [list(real.donors).index(t[0]) for t in targets]
    held = real.is_heldout[indices]
    observed = np.stack([np.stack([E.MX.pc_bag_summary(real.pcs[i][E.MX.det_subsample(
        len(real.pcs[i]), 64, E.MX.stable_seed('pc-summary-rep', donor, r))])
        for i, donor in zip(indices, [t[0] for t in targets])]) for r in range(24)])
    result = {'fold': fold, 'seed': seed, 'scope': __doc__, 'variants': {}}
    for variant, Y in summaries.items():
        lo, hi = np.percentile(Y, [10, 90], axis=1)
        result['variants'][variant] = {'heldout': summarize_bounds(lo, hi, observed, held),
                                      'generated_heldout_PC_SD_mean': float(Y[held, :, 50:].mean()),
                                      'observed_heldout_PC_SD_mean': float(observed[:, held, 50:].mean())}
    (OUT / f'noise_probe_fold{fold}_seed{seed}.json').write_text(json.dumps(result, indent=2) + '\n')
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--device', choices=['cpu', 'cuda'], default='cpu')
    p.add_argument('--folds', default='0,1,2')
    a = p.parse_args()
    torch.set_num_threads(3)
    if a.device == 'cuda':
        torch.cuda.set_per_process_memory_fraction(.04)
    OUT.mkdir(parents=True, exist_ok=True)
    results = []
    for fold in map(int, a.folds.split(',')):
        for seed in (0, 1):
            results.append(run(fold, seed, a.device))
            print('probed', fold, seed, flush=True)
    summary = {variant: {'mean_coverage': float(np.mean([r['variants'][variant]['heldout']['PC means']['coverage'] for r in results])),
                         'SD_coverage': float(np.mean([r['variants'][variant]['heldout']['PC standard deviations']['coverage'] for r in results])),
                         'SD_below': float(np.mean([r['variants'][variant]['heldout']['PC standard deviations']['below'] for r in results])),
                         'generated_PC_SD_mean': float(np.mean([r['variants'][variant]['generated_heldout_PC_SD_mean'] for r in results])),
                         'observed_PC_SD_mean': float(np.mean([r['variants'][variant]['observed_heldout_PC_SD_mean'] for r in results]))}
               for variant in VARIANTS}
    (OUT / 'noise_probe.json').write_text(json.dumps({'scope': __doc__, 'summary': summary, 'per_fit': results}, indent=2) + '\n')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
