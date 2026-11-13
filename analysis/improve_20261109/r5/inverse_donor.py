"""Input-cell-only inversion of a frozen donor generator, with split-cell tests.

Do not decode a conditional mean latent without checking its reconstruction.
Initialize at a prior particle and optimize a donor's latent coordinates only.
No GAN weights or calibration are fitted. This is a reconstruction task, not
unconditional donor prediction or a validated biological coordinate system.
"""
from __future__ import annotations

import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(key, '3')
os.environ.setdefault('CUDA_VISIBLE_DEVICES', '')

import argparse
import json
from pathlib import Path
import time
import numpy as np
import torch
import trainer as T
import generate as G
import mode_biascal as MB
import evaluate as E
from donor_encoder import decode_profile, repeatability


def log_profile(counts):
    pb = np.asarray(counts.sum(0)).ravel().astype(np.float64)
    return np.log1p(1e4 * pb / max(pb.sum(), 1.))


def expected_profile(model, z, condition, cells=128, seed=20261113):
    """Deterministic differentiable integration; no Gamma/Poisson noise."""
    generator = torch.Generator(device=z.device).manual_seed(seed)
    library = torch.ones((len(z), cells), device=z.device)
    mu, _ = model.parameters_nb(z, condition, cells, library, generator)
    return torch.log1p(1e4 * mu.mean(1))


@torch.no_grad()
def particle_dictionary(model, particles, conditions):
    profiles = {}
    for stage in sorted(set(np.rint(conditions[:, 3].cpu().numpy() * 3).astype(int))):
        c = torch.zeros(1, 7, device=particles.device)
        c[:, 3] = stage / 3
        chunks = []
        for start in range(0, len(particles), 8):
            z = particles[start:start+8]
            chunks.append(expected_profile(model, z, c.expand(len(z), -1)))
        profiles[int(stage)] = torch.cat(chunks)
    return profiles


def inverse_fit(model, particles, dictionary, condition, observed, steps=200):
    """Use observed input profiles only; never receives target-cell profiles."""
    stage = torch.round(condition[:, 3] * 3).long()
    selected = []
    for row, st in zip(observed, stage.tolist()):
        distances = (dictionary[st] - row).square().mean(1)
        selected.append(int(distances.argmin()))
    initial = particles[selected].detach().clone()
    center = particles.mean(0).detach()
    scale = particles.std(0, unbiased=False).clamp_min(.05).detach()
    standardized = torch.nn.Parameter((initial - center) / scale)
    initial_standardized = standardized.detach().clone()
    optimizer = torch.optim.Adam([standardized], lr=.05)
    start_loss = None
    for step in range(steps):
        prediction = expected_profile(model, center + standardized * scale, condition)
        reconstruction = (prediction - observed).square().mean()
        penalty = .01 * (standardized - initial_standardized).square().mean()
        loss = reconstruction + penalty
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError('nonfinite inverse-fit loss')
        if step == 0:
            start_loss = float(reconstruction.detach())
        optimizer.zero_grad()
        loss.backward()
        if standardized.grad is None or not bool(torch.isfinite(standardized.grad).all()):
            raise FloatingPointError('inverse latent gradient')
        optimizer.step()
        with torch.no_grad():
            standardized.clamp_(-4., 4.)
    z = (center + standardized.detach() * scale).detach()
    with torch.no_grad():
        finish = expected_profile(model, z, condition)
    return z, {'initial_input_mse': start_loss,
               'final_input_mse': float((finish - observed).square().mean()),
               'standardized_displacement': float((standardized.detach() - initial_standardized).square().mean()),
               'steps': steps}


def donor_inputs(real, cells=64, depth=False):
    profiles, conditions, libraries = [], [], []
    for i, donor in enumerate(real.donors):
        rows = np.random.default_rng(E.MX.stable_seed('encoder-cell-halves', donor)).permutation(len(real.libs[i]))
        a, b = np.array_split(rows, 2)
        for half in (a, b):
            key = ('encoder-depth', donor, cells) if depth else ('encoder-input', donor)
            n = cells if depth else min(cells, len(half))
            choice = np.random.default_rng(E.MX.stable_seed(*key)).choice(half, n, replace=n > len(half))
            profiles.append(log_profile(real.cells[i][choice]))
            conditions.append(real.raw_c[i])
            libraries.append(float(real.libs[i][choice].mean()))
    return np.array(profiles), np.array(conditions), np.array(libraries)


def run(fold, seed, out_root, steps=200):
    T.check_out_root(out_root)
    out = Path(out_root) / f'fold{fold}/seed{seed}'
    if (out / 'evaluation.json').exists():
        raise FileExistsError(out)
    cache = T.RUNS / f'caches/hvg2000/fold{fold}'
    fit = T.RUNS / f'fits_r3/pbulk/hvg2000/fold{fold}/seed{seed}'
    cfg = json.loads((fit / 'config.json').read_text())
    ctx = T.Context(cache, 'cpu', tuple(cfg['flags']))
    model, prior = G.load_models(ctx, fit, cfg, ['final'])['final']
    receipt_path = fit / 'generated_biascal@stagehalf.receipt.json'
    receipt = json.loads(receipt_path.read_text())
    if Path(receipt['fit']).resolve() != fit.resolve():
        raise ValueError('calibration belongs to a different fit')
    with torch.no_grad():
        MB.head(model).add_(torch.tensor(receipt['mode_extra']['biascal']['delta']))
    model.eval().requires_grad_(False)
    prior.eval().requires_grad_(False)
    particles = prior.z.detach().clone()
    original_weights = {k: v.detach().clone() for k, v in model.state_dict().items()}
    real = E.load_real(cache)
    profiles, raw, libraries = donor_inputs(real)
    condition = torch.tensor(ctx.condition_matrix(raw), dtype=torch.float32)
    observed = torch.tensor(profiles, dtype=torch.float32)
    dictionary = particle_dictionary(model, particles, condition)
    started = time.monotonic()
    z, info = inverse_fit(model, particles, dictionary, condition, observed, steps)
    # A new nuisance stream and more integration cells test numerical overfit.
    prediction = decode_profile(model, z, condition, torch.tensor(libraries, dtype=torch.float32), seed=77, k=1024)
    depth = {}
    scale = particles.std(0, unbiased=False).clamp_min(.05).cpu().numpy()
    for k in (32, 128):
        pb, raw, _ = donor_inputs(real, k, depth=True)
        c = torch.tensor(ctx.condition_matrix(raw), dtype=torch.float32)
        zk, depth_info = inverse_fit(model, particles, dictionary, c, torch.tensor(pb, dtype=torch.float32), steps)
        depth[str(k)] = {'latent_standardized_mse': ((zk.numpy()-z.numpy())/scale)**2,
                         'fit': depth_info}
    train_mean = np.mean([log_profile(real.cells[i]) for i in np.flatnonzero(~real.is_heldout)], axis=0)
    records = []
    for i, donor in enumerate(real.donors):
        rows = np.random.default_rng(E.MX.stable_seed('encoder-cell-halves', donor)).permutation(len(real.libs[i]))
        a, b = np.array_split(rows, 2)
        target = log_profile(real.cells[i][b])
        rmse = lambda p: float(np.sqrt(np.mean((p-target)**2)))
        records.append({'donor': str(donor), 'heldout': bool(real.is_heldout[i]),
                        'latent_a': z[2*i].tolist(), 'latent_b': z[2*i+1].tolist(),
                        'inverse_reconstruction_rmse': rmse(prediction[2*i]),
                        'input_pseudobulk_rmse': rmse(profiles[2*i]),
                        'train_mean_rmse': rmse(train_mean),
                        'predicted_profile': prediction[2*i].tolist(),
                        'depth_standardized_latent_mse': {str(k): float(depth[str(k)]['latent_standardized_mse'][2*i].mean()) for k in (32,128)}})
    aggregate = {}
    for split, held in [('train', False), ('heldout', True)]:
        group = [r for r in records if r['heldout'] == held]
        aggregate[split] = {key: float(np.mean([r[key] for r in group])) for key in (
            'inverse_reconstruction_rmse', 'input_pseudobulk_rmse', 'train_mean_rmse')}
        aggregate[split]['split_cell_repeatability'] = repeatability(
            np.array([r['latent_a'] for r in group])/scale, np.array([r['latent_b'] for r in group])/scale)
        aggregate[split]['depth_standardized_latent_mse'] = {str(k): float(np.mean([
            r['depth_standardized_latent_mse'][str(k)] for r in group])) for k in (32,128)}
    # No true latents exist for real patients. Independent synthetic count bags
    # test recovery rather than reporting real reconstruction as latent truth.
    with torch.no_grad():
        chosen = np.random.default_rng(20261113).choice(len(particles),32,replace=False)
        truth = particles[chosen]
        syn_c = condition[torch.arange(32) % len(condition)]
        library = torch.full((32,64),float(ctx.fitlibs.median()))
        torch.manual_seed(20261116)
        rate = model.rates(truth, syn_c, 64, library, torch.Generator().manual_seed(20261117))
        counts = torch.poisson(rate,generator=torch.Generator().manual_seed(20261118))
        pb = counts.sum(1)
        synthetic = torch.log1p(1e4*pb/pb.sum(1,keepdim=True).clamp_min(1))
    recovered, synthetic_info = inverse_fit(model,particles,dictionary,syn_c,synthetic,steps)
    synthetic_mse = float(((recovered-truth)/torch.tensor(scale)).square().mean())
    synthetic_null = float(((truth-particles.mean(0))/torch.tensor(scale)).square().mean())
    assert all(torch.equal(v, original_weights[k]) for k,v in model.state_dict().items()), 'GAN weights changed'
    assert all(p.grad is None for p in model.parameters()), 'GAN parameter gradients accumulated'
    result = {'fold':fold,'seed':seed,'steps':steps,'aggregate':aggregate,'fit_diagnostics':info,
              'synthetic_latent_standardized_mse':synthetic_mse,'synthetic_null_mse':synthetic_null,
              'synthetic_fit_diagnostics':synthetic_info,'records':records,
              'wall_seconds':time.monotonic()-started,'fit_final_sha256':T.sha(fit/'final.pt'),
              'calibration_receipt_sha256':T.sha(receipt_path),'GAN_weights_unchanged':True,
              'scope':'input-cell-only latent inversion; disjoint target cells; no unconditional/biological-axis claim'}
    result['passes_initial_checks'] = bool(aggregate['heldout']['inverse_reconstruction_rmse'] < aggregate['heldout']['train_mean_rmse']
        and aggregate['heldout']['split_cell_repeatability'] >= .8)
    out.mkdir(parents=True,exist_ok=True)
    T.dump(out/'evaluation.json',result)
    print(json.dumps({k:result[k] for k in ('aggregate','fit_diagnostics','synthetic_latent_standardized_mse',
         'synthetic_null_mse','passes_initial_checks','wall_seconds')},indent=2),flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--fold',type=int,choices=range(9),required=True)
    p.add_argument('--seed',type=int,required=True)
    p.add_argument('--steps',type=int,default=200)
    p.add_argument('--out-root',type=Path,default=T.RUNS/'smoke_r5/architecture_pass/inverse')
    a=p.parse_args()
    if a.seed<0 or a.steps<1:p.error('invalid seed/steps')
    torch.set_num_threads(3)
    run(a.fold,a.seed,a.out_root,a.steps)
