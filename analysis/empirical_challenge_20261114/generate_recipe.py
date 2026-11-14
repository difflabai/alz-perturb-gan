"""Isolated fixed readouts for the empirical challenge; no old source edits."""
import os
for _key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[_key] = '3'
os.environ['CUDA_VISIBLE_DEVICES'] = ''

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
import provenance as PV

HERE = Path(__file__).resolve().parent
R5 = HERE.parent/'improve_20261109/r5'
sys.path.insert(0, str(R5))
import numpy as np
import torch
import trainer as T
import generate as G
import arms
import mode_cellshape as CS
import mode_zcal as ZC
import mode_biascal as BC
import baselines as BL
import architecture_score as AS

RUNS = T.RUNS
ROOT = RUNS/'results_r5/empirical_challenge_20261114'
MODES = RUNS/'smoke_r5/empirical_challenge_20261114'
RECIPES = ('original32', 'current32', 'blind32', 'radial32',
           'replay32', 'current128', 'empirical32', 'empirical128')
FIT_FILES = PV.FIT_FILES
MODES_BY_NAME = {'original32': 'biascal@stagehalf',
                 'current32': 'cellshape@0.5:biascal:stagehalf',
                 'blind32': 'cellshape@0.5:biascal:stageblind',
                 'radial32': 'radialshape@biascal:stagehalf'}


def sha(path):
    return T.sha(Path(path))


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix('.partial.json')
    temp.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')
    temp.replace(path)


def source_manifest():
    paths = [HERE/'PROTOCOL.txt', HERE/'generate_recipe.py', HERE/'compare.py',
             HERE/'run.py', HERE/'provenance.py', PV.ANCHORS,
             HERE.parent/'improve_20261109/default_config.json']
    paths += list(R5.glob('*.py'))
    paths += [Path(sys.modules[name].__file__).resolve() for name in (
        'pg_model','critic','usage','released_update','released_adapter','release_env')]
    package = Path(sys.modules['particlegan'].__file__).resolve().parent
    paths += list(package.rglob('*.py'))
    return {str(p): sha(p) for p in sorted(set(paths))}


def generation_streams(fold, dev_index, replicate):
    children = np.random.SeedSequence([20261114, 1, int(fold), int(dev_index), int(replicate)]).spawn(5)
    return {key: int(child.generate_state(1, dtype=np.uint64)[0]) % (2**63-16)
            for key, child in zip(G.GEN_STREAMS, children)}


def empirical_rng(real, target, replicate):
    return np.random.default_rng(AS.E.MX.stable_seed(
        'empirical-challenge', 20261114, real.panel, real.fold, real.donors[target], replicate))


def radial_alpha(z, center, sorted_radii):
    radii = (z-center).square().sum(-1).contiguous()
    lo = torch.searchsorted(sorted_radii, radii, right=False)
    hi = torch.searchsorted(sorted_radii, radii, right=True)
    return (lo+hi).to(z.dtype)/(2*len(sorted_radii))


def contract(mu, alpha):
    library = mu.sum(-1, keepdim=True)
    proportions = mu.sum(1, keepdim=True)/library.sum(1, keepdim=True).clamp_min(1e-30)
    a = alpha[:, None, None]
    return (1-a)*mu+a*library*proportions


def patch_radial(model, prior):
    if getattr(prior, 'code_law', None) != 'table':
        raise ValueError('radial hypothesis requires the original uniform particle-table prior')
    if getattr(model, '_radialshape', False):
        raise ValueError('radial contraction applied twice')
    center = prior.z.detach().mean(0)
    radii = (prior.z.detach()-center).square().sum(-1)
    ordered = torch.sort(radii).values.contiguous()
    original = model.parameters_nb

    def parameters_nb(z, c, k, library, gen=None):
        mu, theta = original(z, c, k, library, gen)
        return contract(mu, radial_alpha(z, center, ordered)), theta

    model.parameters_nb = parameters_nb
    model._radialshape = True
    alpha = radial_alpha(prior.z.detach(), center, ordered)
    return {'particles': len(radii), 'unique_radii': int(torch.unique(radii).numel()),
            'alpha_min': float(alpha.min()), 'alpha_max': float(alpha.max()),
            'alpha_mean': float(alpha.mean()),
            'law': 'training-prior centered radial midrank; dependent on donor latent',
            'sorted_radii_sha256': hashlib.sha256(ordered.cpu().numpy().tobytes()).hexdigest()}


def radial_plan(ctx, value, target, rng, states):
    return ZC.plan(ctx, value, target, rng, states)


def radial_prepare(ctx, value, name, model, prior):
    info = patch_radial(model, prior)
    ZC.prepare(ctx, value, name, model, prior)
    extra = dict(getattr(ctx, 'mode_receipt_extra', None) or {})
    extra.setdefault('radialshape', {})[name] = info
    ctx.mode_receipt_extra = extra


arms.register_mode('radialshape', radial_plan, value_policy='required',
                   parse_value=ZC.parse_value, prepare=radial_prepare,
                   doc='Fixed radial-dependent within-bag contraction, train-only joint calibration')


def check_finished(score_path, recipe, fold, replicate, sources):
    if not score_path.exists():
        return False
    value = json.loads(score_path.read_text())
    expected = {'recipe': recipe, 'fold': fold, 'replicate': replicate, 'sources': sources}
    meta = value.get('challenge', {})
    if any(meta.get(k) != v for k, v in expected.items()):
        raise ValueError('existing challenge result provenance differs')
    receipt = MODES/recipe/f'fold{fold}/seed{replicate}.receipt.json'
    archive = MODES/recipe/f'fold{fold}/seed{replicate}.npz'
    payload = json.loads(receipt.read_text())
    source_hash = value['fingerprint']['generated_sha256']
    draws = 128 if recipe.endswith('128') else 32
    if (value.get('pruned_source_sha256') != source_hash or sha(receipt) != meta['receipt_sha256']
            or payload['generated_SHA256'] != source_hash or archive.exists()
            or payload['sources'] != sources or payload['challenge_recipe'] != recipe
            or payload['computational_replicate'] != replicate
            or value['fold'] != fold or value['M'] != draws or payload['M'] != draws
            or payload['K'] != 256 or meta.get('K') != 256 or meta.get('PC_CELLS') != 64
            or meta.get('generation_device') != 'cpu' or payload['generation_device'] != 'cpu'
            or payload['cache_manifest_sha256'] != value['fingerprint']['cache_sha256']
            or value['fingerprint']['cache_sha256'] != sha(RUNS/f'caches/hvg2000/fold{fold}/manifest.json')):
        raise ValueError('existing challenge receipt/pruning mismatch')
    artifacts = payload.get('fit_artifacts_sha256', {})
    if not recipe.startswith('empirical') and (set(artifacts) != set(FIT_FILES)
            or Path(payload['fit']).resolve() != (RUNS/f'fits_r3/pbulk/hvg2000/fold{fold}/seed{replicate}').resolve()):
        raise ValueError('missing or wrong frozen fit provenance')
    for name, expected_sha in artifacts.items():
        if sha(Path(payload['fit'])/name) != expected_sha:
            raise ValueError('frozen fit artifact changed after challenge generation')
    return True


@torch.no_grad()
def replay(fit, cache, receipt_path, output, draws):
    receipt = json.loads(receipt_path.read_text())
    if (Path(receipt['fit']).resolve() != fit.resolve() or receipt['generation_device'] != 'cpu'
            or receipt['M'] != 32 or receipt['mode'] != MODES_BY_NAME['current32']):
        raise ValueError('frozen calibration must come from this CPU M32 current fit')
    cfg = json.loads((fit/'config.json').read_text())
    ctx = T.Context(cache, 'cpu', tuple(cfg['flags']))
    G.M = draws
    targets = G.targets(ctx)
    states = G.available_states(fit, cfg)
    plans, _, _ = G.mode_plans(ctx, 'stagehalf', targets, states, updates=cfg['updates'])
    used = [name for name in states if any(name in names for _, names in plans)]
    models = G.load_models(ctx, fit, cfg, used)
    extra = receipt['mode_extra']
    for model, _ in models.values():
        BC.head(model).add_(torch.tensor(extra['biascal']['delta'], dtype=torch.float32))
        CS.patch_generator(model, .5)
        state = ZC.patch_generator(model, ctx.device)
        state['s'].copy_(torch.tensor(extra['zcal']['s'], dtype=torch.float32))
    arrays, stats = G.generate_arrays(ctx, models, plans, T=targets)
    arrays.update(mode=np.array(MODES_BY_NAME['current32']), state_names=np.array(used),
                  state_index=np.array([[used.index(s) for s in names] for _, names in plans], dtype=np.int8),
                  conditions=np.stack([c for c, _ in plans]).astype(np.float32))
    AS.E.save_npz(output, **arrays)
    stats.update(fit=str(fit.resolve()), generation_device='cpu', mode=MODES_BY_NAME['current32'],
                 generated_SHA256=sha(output), calibration_M=32,
                 states_used=used, mode_extra=extra,
                 calibration_receipt_sha256=sha(receipt_path),
                 calibration='frozen rounded M32 bias/dispersion replay; no refit',
                 note='MC control; M128 changes state allocation and draws, not just quantile estimation')
    return stats


def allocation(output):
    with np.load(output, allow_pickle=False) as arrays:
        names=arrays['state_names'].astype(str).tolist()
        indices=arrays['state_index']
        stage=arms.stage_of(arrays['conditions'])
        return {'states':names,'targets':arrays['targets'].astype(str).tolist(),
                'is_heldout':arrays['is_heldout'].tolist(),
                'target_state_stage_counts':[[[int(((indices[t]==i)&(stage[t]==s)).sum())
                    for s in range(4)] for i in range(len(names))] for t in range(len(indices))]}


def run(recipe, fold, replicate):
    if recipe not in RECIPES or fold not in range(9) or replicate not in (0, 1):
        raise ValueError('invalid fixed recipe/fold/replicate')
    torch.set_num_threads(3)
    sources = source_manifest()
    manifest = ROOT/'source_manifest.json'
    if not manifest.exists() or json.loads(manifest.read_text()) != sources:
        raise ValueError('launch the pinned queue first; source manifest missing or changed')
    PV.verify_cache(fold, payloads=False)
    if not recipe.startswith('empirical'):
        PV.verify_fit(fold, replicate)
    score_path = ROOT/recipe/f'fold{fold}/seed{replicate}.json'
    if check_finished(score_path, recipe, fold, replicate, sources):
        return
    cache = RUNS/f'caches/hvg2000/fold{fold}'
    fit = RUNS/f'fits_r3/pbulk/hvg2000/fold{fold}/seed{replicate}'
    output = MODES/recipe/f'fold{fold}/seed{replicate}.npz'
    if output.exists() or score_path.exists():
        raise ValueError('partial output exists; inspect before resuming this job')
    output.parent.mkdir(parents=True, exist_ok=True)
    G.generation_streams = lambda f, j: generation_streams(f, j, replicate)
    started = time.monotonic()
    if recipe.startswith('empirical'):
        real = AS.E.load_real(cache)
        BL._target_rng = lambda r, t: empirical_rng(r, t, replicate)
        draws = 128 if recipe.endswith('128') else 32
        arrays, info = BL.generate('resample_blind', real, M=draws, K=256, kpc=64)
        AS.E.save_npz(output, **arrays)
        receipt = dict(info, baseline='resample_blind', generation_device='cpu',
                       generated_SHA256=sha(output), targets=len(real.donors),
                       source_selection='training donors only; target excluded when in training',
                       empirical_stream=replicate)
    elif recipe in ('replay32', 'current128'):
        if not check_finished(ROOT/'current32'/f'fold{fold}/seed{replicate}.json',
                              'current32',fold,replicate,sources):
            raise ValueError('control requires a complete validated current32 calibration/source score')
        receipt = replay(fit, cache, MODES/'current32'/f'fold{fold}/seed{replicate}.receipt.json',
                         output, 128 if recipe.endswith('128') else 32)
    else:
        G.M = 32
        _, receipt = G.generate_fit(fit, cache, mode=MODES_BY_NAME[recipe], out=output, device='cpu')
    receipt.update(challenge_recipe=recipe, computational_replicate=replicate,
                   generation_streams='challenge20261114; see committed PROTOCOL.txt',
                   seconds_total=time.monotonic()-started, sources=sources,
                   runtime={'python':sys.version,'numpy':np.__version__,'torch':str(torch.__version__),
                            'particlegan':str(T.VERSION)},
                   cache_manifest_sha256=sha(cache/'manifest.json'))
    if not recipe.startswith('empirical'):
        receipt['fit_artifacts_sha256'] = {name: sha(fit/name) for name in FIT_FILES}
        receipt['state_stage_allocation'] = allocation(output)
    write(output.with_suffix('.receipt.json'), receipt)
    AS.score(output, cache, score_path)
    value = json.loads(score_path.read_text())
    if source_manifest() != sources:
        raise ValueError('challenge source changed during generation/scoring')
    value['pruned_source_sha256'] = value['fingerprint']['generated_sha256']
    value['challenge'] = {'recipe': recipe, 'fold': fold, 'replicate': replicate, 'sources': sources,
                          'receipt_sha256': sha(output.with_suffix('.receipt.json')),
                          'generation_device': 'cpu', 'K': 256, 'PC_CELLS': 64}
    write(score_path, value)
    output.unlink()
    print(json.dumps({'recipe': recipe, 'fold': fold, 'replicate': replicate,
                     'M': value['M'], 'heldout': value['intervals']['heldout']['percentile']}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--recipe', choices=RECIPES, required=True)
    parser.add_argument('--fold', type=int, choices=range(9), required=True)
    parser.add_argument('--replicate', type=int, choices=(0, 1), required=True)
    args = parser.parse_args()
    run(args.recipe, args.fold, args.replicate)
