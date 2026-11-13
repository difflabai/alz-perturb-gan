"""Tests for trainer.py / arms.py / generate.py (CPU, <= 6 threads).

    python3 test_trainer.py devcache   # expose legacy fold-0 train cache in Interface-1 layout
    python3 test_trainer.py parity     # base arm == released build()/advance() (bitwise)
    python3 test_trainer.py grads      # every arm: gradients reach new parameters / inputs
    python3 test_trainer.py arms       # every arm + offset,genemean,covars: 100 updates + generated.npz
    python3 test_trainer.py timing     # seconds/update base vs genemean (+ others)
    python3 test_trainer.py resume     # --stop-after and SIGINT interruption, resume == uninterrupted
    python3 test_trainer.py resume_offset  # offset arm: in-process determinism, --stop-after resume bitwise,
                                           # generation-only path refuses a different --updates
    python3 test_trainer.py cleanup    # delete large test outputs
    python3 test_trainer.py all

Dev cache adapter: reads ONLY the legacy fold-0 ``train/`` files (33 benchmark
optimization donors, all of them improve_20261109 development donors),
``space.npz`` and ``manifest.json`` (donor ids only) from
core-comparison-20261103/caches/fold0; never the evaluation/ directory (sealed
donors).  It writes, under RUNS/tmp_trainer_devcache/:

* ``cache_fold0``: Interface-1 layout, panel 'devlegacy', inner_fold 0, the 33
  donors as train, no heldout/ (generation targets = train donors only).
* ``cache_fold0_ho``: the same cells split into 29 train + 4 'heldout' donors
  (one per stage) so the heldout generation path and covariates of a donor
  not seen in training are exercised.  Its space/PCs were fitted on all 33
  donors, so it is a plumbing fixture only, never a scientific split
  (requires --allow-nonprotocol-split, as does cache_fold0).
"""
import os
import sys

sys.dont_write_bytecode = True  # read-only source dirs must not get __pycache__

for _k in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[_k] = '6'
os.environ['CUDA_VISIBLE_DEVICES'] = ''

import json
import shutil
import signal
import subprocess
import time
from pathlib import Path

import numpy as np
import scipy.sparse as sp

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import torch  # noqa: E402

torch.set_num_threads(6)
import trainer  # noqa: E402
import arms  # noqa: E402
import generate  # noqa: E402

PY = '/mnt/nvme/ai/conda/anaconda3/bin/python3'
LEGACY = trainer.SCIENCE / 'caches/fold0'
DEV = trainer.RUNS / 'tmp_trainer_devcache'
CACHE = DEV / 'cache_fold0'
CACHE_HO = DEV / 'cache_fold0_ho'
FITS = DEV / 'fits'
RESULTS = DEV / 'results'
REAL = trainer.RUNS / 'caches/hvg2000/fold0'  # read-only; written by build_cache.py (other engineer)
ALL_ARMS = ['base', 'offset', 'genemean', 'donorpc', 'covars', 'covars,genemean,offset']


def log(*a):
    print('[test]', *a, flush=True)


def save_result(name, obj):
    trainer.dump(RESULTS / f'{name}.json', obj)


# ----------------------------------------------------------------------------- dev cache adapter
def _write_cache(root, panel, train_rows_mask, train_ids, heldout_ids, counts, meta, space, note):
    root = Path(root)
    if root.exists():
        shutil.rmtree(root)
    (root / 'train').mkdir(parents=True)
    genes = meta['genes'].astype(str)
    sp_kw = {k: space[k] for k in space}
    sp_kw['genes'] = genes
    np.savez(root / 'space.npz', **sp_kw)

    def part(sub, ids):
        rows = np.flatnonzero(np.isin(meta['donor'], ids))
        donor_idx = [list(meta['donors']).index(d) for d in ids]
        (root / sub).mkdir(exist_ok=True)
        sp.save_npz(root / sub / 'counts.npz', counts[rows].astype(np.float32).tocsr())
        np.savez(root / sub / 'metadata.npz', cell=meta['cell'][rows], donor=meta['donor'][rows], type=meta['type'][rows],
                 donors=np.asarray(ids), raw_c=meta['raw_c'][donor_idx], genes=genes)
    part('train', train_ids)
    if heldout_ids:
        part('heldout', heldout_ids)
    files = sorted(str(p.relative_to(root)) for p in root.rglob('*.npz'))
    manifest = {'panel': panel, 'inner_fold': 0, 'train_donors': list(train_ids), 'heldout_donors': list(heldout_ids),
                'n_genes': int(counts.shape[1]), 'forced_genes': {'present': [], 'missing': []},
                'files_SHA256': {f: trainer.sha(root / f) for f in files},
                'recipe_notes': note, 'dev_fixture_not_protocol_split': True}
    trainer.dump(root / 'manifest.json', manifest)


def devcache():
    meta = dict(np.load(LEGACY / 'train/metadata.npz', allow_pickle=False))
    space = dict(np.load(LEGACY / 'space.npz', allow_pickle=False))
    counts = sp.load_npz(LEGACY / 'train/counts_hvg.npz').tocsr()
    folds = trainer.load_folds()
    ids = [str(d) for d in meta['donors']]
    assert len(ids) == 33 and set(ids) <= set(folds['development_donors'])
    assert not set(ids) & set(folds['sealed_evaluation_donors_fold0'])
    note = ('DEV FIXTURE: legacy core-comparison fold0 train cache (33 benchmark optimization donors) in '
            'Interface-1 layout; space fitted on these 33 donors by the legacy recipe.')
    _write_cache(CACHE, 'devlegacy', None, ids, [], counts, meta, space, note)
    stage = arms.stage_of(meta['raw_c'])
    held = []
    for s in range(4):  # last donor of each stage
        held.append(ids[np.flatnonzero(stage == s)[-1]])
    train = [d for d in ids if d not in held]
    _write_cache(CACHE_HO, 'devlegacyho', None, train, held, counts, meta, space,
                 note + ' Split 29 train + 4 heldout (one per stage); NOT a protocol split (space saw all 33).')
    log('dev caches written', CACHE, CACHE_HO, 'heldout', held)
    return {'cache': str(CACHE), 'cache_ho': str(CACHE_HO), 'heldout': held}


def ensure_cache():
    if not (CACHE / 'manifest.json').exists() or not (CACHE_HO / 'manifest.json').exists():
        devcache()


# ----------------------------------------------------------------------------- parity
def _released_context():
    import released_adapter as A
    import released_update as U
    manifest = json.loads((LEGACY / 'manifest.json').read_text())
    # contracts.checked_cache SHA-hashes every cache file INCLUDING evaluation/
    # (sealed donors) -> replace it by a manifest-only version for this test.
    A.checked_cache = lambda cache, role: (manifest, {'optimization_donors': list(manifest['train_donors'])})
    U.configure_hyperparameters(dict(trainer.T005))
    return A, U, A.Context(LEGACY, 'cpu')


def _state_equal(a, b):
    sa, sb = a.state_dict(), b.state_dict()
    return set(sa) == set(sb) and all(torch.equal(sa[k], sb[k]) for k in sa)


def parity(n_updates=5):
    ensure_cache()
    A, U, rctx = _released_context()
    rb, rg, rr, ru, rstreams = A.build(rctx.mu, rctx.theta, rctx.device, 0, 0)
    init_r = [{k: v.clone() for k, v in m.state_dict().items()} for m in (rb[1], rb[2], rb[3])]
    rmetrics = [U.advance(rctx, rb, rg, rr, ru, execution_limit=20000) for _ in range(n_updates)]
    ctx = trainer.Context(CACHE, 'cpu', (), allow_nonprotocol_split=True)
    data_equal = {'X': torch.equal(ctx.X, rctx.X), 'trainpcs': torch.equal(ctx.trainpcs, rctx.trainpcs),
                  'mu': torch.equal(ctx.mu, rctx.mu), 'theta': torch.equal(ctx.theta, rctx.theta),
                  'conditions': torch.equal(ctx.conditions, rctx.conditions), 'W': torch.equal(ctx.W, rctx.W),
                  'by': all(np.array_equal(a, b) for a, b in zip(ctx.by, rctx.by))}
    b, g, r, u, streams = trainer.build(ctx, 0, stream_root=trainer.RELEASED_STREAM_ROOT)
    init_equal = {name: set(s) == set(m.state_dict()) and all(torch.equal(s[k], m.state_dict()[k]) for k in s)
                  for name, s, m in zip(['G', 'D', 'P'], init_r, (b[1], b[2], b[3]))}
    metrics = [trainer.advance(ctx, b, g, r, u, execution_limit=20000) for _ in range(n_updates)]
    keys = ['D_adversarial', 'D_penalty', 'D_total', 'G_adversarial', 'prior_regularization_unweighted', 'G_total', 'critic_gap',
            'D_gradient_norm', 'G_gradient_norm', 'prior_gradient_norm', 'G_prior_LRs', 'D_LRs']
    loss_equal = [all(m1[k] == m2[k] for k in keys) and m1['penalty_stats'] == m2['penalty_stats'] for m1, m2 in zip(rmetrics, metrics)]
    final_equal = {name: _state_equal(x, y) for name, x, y in zip(['G', 'D', 'P'], (rb[1], rb[2], rb[3]), (b[1], b[2], b[3]))}
    rng_equal = rr.bit_generator.state == r.bit_generator.state and all(torch.equal(rg[k].get_state(), g[k].get_state()) for k in g)
    res = {'streams_equal': rstreams == streams, 'data_equal': data_equal, 'initial_tensors_bitwise_equal': init_equal,
           'per_update_losses_bitwise_equal': loss_equal, 'parameters_after_updates_bitwise_equal': final_equal,
           'rng_states_equal_after': rng_equal, 'updates': n_updates,
           'released_losses': [{k: m[k] for k in ['D_total', 'G_total', 'critic_gap']} for m in rmetrics],
           'trainer_losses': [{k: m[k] for k in ['D_total', 'G_total', 'critic_gap']} for m in metrics],
           'released_LRs': [rmetrics[0]['G_prior_LRs'], rmetrics[0]['D_LRs']]}
    ok = (res['streams_equal'] and all(data_equal.values()) and all(init_equal.values()) and all(loss_equal)
          and all(final_equal.values()) and rng_equal)
    res['PASS'] = bool(ok)
    save_result('parity', res)
    log('parity', json.dumps(res, indent=1))
    assert ok, 'parity failed'
    return res


# ----------------------------------------------------------------------------- gradients per arm
def grads():
    ensure_cache()
    from pg_model import critic_features
    out = {}
    for arm in ALL_ARMS + ['donorpc,offset']:
        flags = arms.parse_arm(arm)
        ctx = trainer.Context(CACHE, 'cpu', flags, allow_nonprotocol_split=True)
        b, g, r, u, _ = trainer.build(ctx, 0)
        rec, G, D, P = b[:4]
        row = {'critic_features': ctx.features.n_features, 'critic_class': type(D).__name__}
        # One real update (asserts inside advance: G.out/log_theta/P.z/stage_offset grads nonzero at
        # step 0 and critic gradient on added input columns).
        m = trainer.advance(ctx, b, g, r, u, execution_limit=20000)
        row['first_update_finite'] = all(np.isfinite(m[k]) for k in ['D_total', 'G_total', 'critic_gap'])
        # Generator gradient through the ADDED critic block only.
        if ctx.features.n_extra:
            gen = torch.Generator().manual_seed(1)
            z, _ = P.sample(4, generator=gen)
            ds = np.arange(4) * 7
            idx = torch.as_tensor(np.stack([np.resize(ctx.by[d], 128) for d in ds]))
            lib = ctx.X[idx].sum(-1)
            rate = G.rates(z, ctx.conditions[ds], 128, lib, gen)
            cf = critic_features(torch.poisson(rate.detach(), generator=gen), rate, ctx.mean, ctx.sd)
            f = ctx.features.fake(cf)
            f2 = torch.cat([f[:, :150].detach(), f[:, 150:]], 1)
            gr = torch.autograd.grad(D(f2, ctx.conditions[ds]).sum(), [G.out[-1].weight])[0]
            row['G_grad_norm_through_added_block_only'] = float(gr.norm())
            assert row['G_grad_norm_through_added_block_only'] > 0
            real = ctx.features.real(torch.as_tensor(np.stack([ctx.by[d][:50] for d in range(len(ctx.by))])))
            row['real_block_sq_norm_base_vs_added'] = [float(real[:, :150].square().sum(1).mean()), float(real[:, 150:].square().sum(1).mean())]
            row['features_info'] = {k: v for k, v in ctx.features.info.items() if k != 'train_donor_score_sd'}
        if 'offset' in flags:
            row['offset_grad_norm_step0'] = None  # asserted nonzero inside advance
            row['offset_info'] = ctx.offset_info
            row['offset_moved_after_1_update'] = float((G.stage_offset.detach() - torch.as_tensor(ctx.offset_table)).abs().max())
            assert row['offset_moved_after_1_update'] > 0
            # offset param is in the generator optimizer group
            row['offset_in_generator_group'] = any(p is G.stage_offset for p in b[4].param_groups[0]['params'])
            assert row['offset_in_generator_group']
        cond = ctx.conditions.cpu().numpy()
        row['condition_slot_sd'] = cond.std(0).round(4).tolist()
        if 'covars' in flags:
            # legacy dev cache has Male/APOE4 zeroed in raw_c (masked upstream); real caches carry them
            assert (cond[:, :3] == 0).all() and np.array_equal(cond[:, 4:6], ctx.raw_c[:, 4:6].astype(np.float32))
            assert abs(cond[:, 6].mean()) < 1e-5 and abs(cond[:, 6].std() - 1) < 1e-5
        else:
            assert (np.delete(cond, 3, axis=1) == 0).all()
        out[arm] = row
        log('grads', arm, json.dumps(row)[:400])
    save_result('grads', out)
    return out


# ----------------------------------------------------------------------------- CLI runs
def cli(arm, cache, out_root, *extra, updates=100, seed=0, check=True, **kw):
    cmd = [PY, str(HERE / 'trainer.py'), '--cache', str(cache), '--arm', arm, '--seed', str(seed), '--out-root', str(out_root),
           '--updates', str(updates), '--device', 'cpu', '--threads', '6']
    cmd += [x for x in extra if x != '--no-dev'] if '--no-dev' in extra else ['--allow-nonprotocol-split', *extra]
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='6', OPENBLAS_NUM_THREADS='6', MKL_NUM_THREADS='6')
    t = time.monotonic()
    p = subprocess.run(cmd, env=env, capture_output=True, text=True, **kw)
    if check and p.returncode != 0:
        raise RuntimeError(f'{cmd} failed:\n{p.stdout[-3000:]}\n{p.stderr[-5000:]}')
    return p, time.monotonic() - t


def check_generated(fit, ctx_cache):
    q = np.load(fit / 'generated.npz', allow_pickle=False)
    man = json.loads((Path(ctx_cache) / 'manifest.json').read_text())
    T = len(man['train_donors']) + (len(man['heldout_donors']) if (Path(ctx_cache) / 'heldout').exists() else 0)
    G = man['n_genes']
    want = {'targets': ((T,), 'U'), 'is_heldout': ((T,), 'b'), 'stage': ((T,), 'i'), 'genes': ((G,), 'U'),
            'pb_counts': ((T, 32, G), np.float32), 'mean_log': ((T, 32, G), np.float32), 'nonzero': ((T, 32, G), np.float32),
            'pcs': ((T, 32, 64, 50), np.float16), 'lib': ((T, 32), np.float32)}
    rep = {}
    for k, (shape, dt) in want.items():
        a = q[k]
        ok_shape = a.shape == shape
        ok_dt = (a.dtype.kind == dt) if isinstance(dt, str) else a.dtype == dt
        fin = bool(np.isfinite(a.astype(np.float32)).all()) if a.dtype.kind == 'f' else True
        rep[k] = {'shape': list(a.shape), 'dtype': str(a.dtype), 'ok': bool(ok_shape and ok_dt and fin)}
    pb, ml, nz, lib = q['pb_counts'], q['mean_log'], q['nonzero'], q['lib']
    rep['consistency'] = {
        'pb_integer_nonneg': bool((pb >= 0).all() and np.array_equal(pb, np.round(pb))),
        'lib_equals_pb_sum_over_K': float(np.abs(pb.sum(-1) / 256 - lib).max()),
        'nonzero_in_01': bool(((nz >= 0) & (nz <= 1)).all()), 'mean_log_nonneg': bool((ml >= 0).all()),
        'targets_order': list(q['targets'][:3]), 'heldout_targets': int(q['is_heldout'].sum()),
        'mean_lib': float(lib.mean())}
    rep['ALL_OK'] = all(v['ok'] for k, v in rep.items() if k in want) and rep['consistency']['pb_integer_nonneg'] \
        and rep['consistency']['lib_equals_pb_sum_over_K'] < 1e-3 * rep['consistency']['mean_lib'] and rep['consistency']['nonzero_in_01']
    return rep


def run_arms():
    ensure_cache()
    out = {}
    jobs = [(a, CACHE, ()) for a in ALL_ARMS] + [('covars,genemean,offset', CACHE_HO, ())]
    if (REAL / 'manifest.json').exists():  # real Interface-1 cache, protocol split enforced
        jobs += [('covars,genemean,offset', REAL, ('--no-dev',)), ('donorpc', REAL, ('--no-dev',))]
    for arm, cache, extra in jobs:
        man = json.loads((cache / 'manifest.json').read_text())
        fit = FITS / arm / man['panel'] / f"fold{man['inner_fold']}" / 'seed0'
        secs = None
        if not (fit / 'generated.npz').exists():
            _, secs = cli(arm, cache, FITS, *extra)
        rec = json.loads((fit / 'fit_receipt.json').read_text())
        curve = [json.loads(l) for l in (fit / 'learning_curve.jsonl').read_text().splitlines()]
        rep = check_generated(fit, cache)
        row = {'cache': cache.name, 'wall_seconds_process': secs, 'receipt': rec, 'curve_steps': [c['step'] for c in curve],
               'last_curve': {k: curve[-1][k] for k in ['D_total', 'G_total', 'critic_gap']}, 'generated': rep,
               'files_MB': {p.name: round(p.stat().st_size / 2 ** 20, 2) for p in fit.iterdir()}}
        assert rep['ALL_OK'], (arm, rep)
        assert rec['status'] == 'completed' and rec['completed_updates'] == 100
        out[f'{arm}@{man["panel"]}'] = row
        log('arm', arm, cache.name, secs, 'spu', round(rec['seconds_per_update_this_session'], 3),
            'zero-rate', rec['generated_zero_count_cell_rate'], 'gen s', round(rec['generation']['seconds'], 1))
    # refuse to overwrite a completed fit
    p, _ = cli('base', CACHE, FITS, check=False)
    out['refuse_overwrite'] = {'returncode': p.returncode, 'FileExistsError': 'FileExistsError' in p.stderr}
    assert p.returncode != 0 and 'FileExistsError' in p.stderr
    # generate.py standalone on a finished fit reproduces generated.npz bitwise (CRN)
    fit = FITS / 'base' / 'devlegacy' / 'fold0' / 'seed0'
    before = dict(np.load(fit / 'generated.npz'))
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='')
    p = subprocess.run([PY, str(HERE / 'generate.py'), '--fit', str(fit), '--cache', str(CACHE), '--threads', '6', '--force'],
                       env=env, capture_output=True, text=True)
    assert p.returncode == 0, p.stderr[-3000:]
    after = dict(np.load(fit / 'generated.npz'))
    out['standalone_generate_bitwise_equal'] = all(np.array_equal(before[k], after[k]) for k in before)
    assert out['standalone_generate_bitwise_equal']
    # CRN: same target streams across arms -> library draws identical across arms
    libs = {a: np.load(FITS / a / 'devlegacy/fold0/seed0/generated.npz')['lib'] for a in ['base', 'offset']}
    out['crn_note'] = 'libraries are drawn from the same per-target stream; lib means differ only through model counts'
    out['lib_corr_base_offset'] = float(np.corrcoef(libs['base'].ravel(), libs['offset'].ravel())[0, 1])
    save_result('arms', out)
    return out


# ----------------------------------------------------------------------------- timing
def timing(n=30, warm=3):
    ensure_cache()
    res = {'threads': torch.get_num_threads(), 'n_timed_updates': n, 'warmup_updates': warm}
    for arm in ['base', 'genemean', 'donorpc', 'offset', 'covars,genemean,offset']:
        ctx = trainer.Context(CACHE, 'cpu', arms.parse_arm(arm), allow_nonprotocol_split=True)
        b, g, r, u, _ = trainer.build(ctx, 0)
        for _ in range(warm):
            trainer.advance(ctx, b, g, r, u, execution_limit=20000)
        t = time.perf_counter()
        for _ in range(n):
            trainer.advance(ctx, b, g, r, u, execution_limit=20000)
        spu = (time.perf_counter() - t) / n
        res[arm] = {'seconds_per_update': spu, 'extrapolated_20000_updates_hours': spu * 20000 / 3600}
        log('timing', arm, res[arm])
        del ctx, b
    save_result('timing', res)
    return res


# ----------------------------------------------------------------------------- resume
def _final_models(fit):
    q = torch.load(fit / 'final.pt', map_location='cpu', weights_only=False)
    return q['policy']['models'], q['policy']['optimizers'], q['numpy_rng']


def _models_equal(a, b):
    return all(set(a[k]) == set(b[k]) and all(torch.equal(a[k][x], b[k][x]) for x in a[k]) for k in a) and set(a) == set(b)


def _curve(fit):
    rows = [json.loads(l) for l in (fit / 'learning_curve.jsonl').read_text().splitlines()]
    for r in rows:
        r.pop('wall_seconds')
    return rows


def resume():
    ensure_cache()
    ref = FITS / 'base' / 'devlegacy' / 'fold0' / 'seed0'
    if not (ref / 'final.pt').exists():
        cli('base', CACHE, FITS)
    res = {}
    # (a) deterministic interruption via --stop-after 60 with checkpoints every 40
    root = DEV / 'fits_resume_stopafter'
    shutil.rmtree(root, ignore_errors=True)
    p, _ = cli('base', CACHE, root, '--checkpoint-every', '40', '--stop-after', '60', check=False)
    fit = root / 'base' / 'devlegacy' / 'fold0' / 'seed0'
    q = torch.load(fit / 'recovery.pt', map_location='cpu', weights_only=False)
    res['stop_after'] = {'first_returncode': p.returncode, 'interrupted_error': 'InterruptedError' in p.stderr,
                         'recovery_completed_updates': q['completed_updates'], 'recovery_status': q['status'],
                         'failure_json': json.loads((fit / 'failure.json').read_text())}
    del q
    cli('base', CACHE, root, '--checkpoint-every', '40')
    a, b = _final_models(fit), _final_models(ref)
    res['stop_after'].update(final_models_bitwise_equal_uninterrupted=_models_equal(a[0], b[0]),
                             numpy_rng_equal=a[2] == b[2], curve_equal=_curve(fit) == _curve(ref),
                             receipt=json.loads((fit / 'fit_receipt.json').read_text()),
                             generated_equal=all(np.array_equal(x, y) for x, y in zip(np.load(fit / 'generated.npz').values(), np.load(ref / 'generated.npz').values())),
                             recovery_removed=not (fit / 'recovery.pt').exists(), failure_json_removed=not (fit / 'failure.json').exists())
    # (b) real SIGINT during training
    root2 = DEV / 'fits_resume_sigint'
    shutil.rmtree(root2, ignore_errors=True)
    fit2 = root2 / 'base' / 'devlegacy' / 'fold0' / 'seed0'
    cmd = [PY, str(HERE / 'trainer.py'), '--cache', str(CACHE), '--arm', 'base', '--seed', '0', '--out-root', str(root2),
           '--updates', '100', '--device', 'cpu', '--threads', '6', '--allow-nonprotocol-split']
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='')
    proc = subprocess.Popen(cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    t0 = time.monotonic()
    while not (fit2 / 'progress.json').exists() and time.monotonic() - t0 < 600:
        time.sleep(0.5)
    time.sleep(8)
    proc.send_signal(signal.SIGINT)
    so, se = proc.communicate(timeout=600)
    q = torch.load(fit2 / 'recovery.pt', map_location='cpu', weights_only=False)
    res['sigint'] = {'returncode': proc.returncode, 'interrupted_error': 'InterruptedError' in se,
                     'recovery_completed_updates': q['completed_updates']}
    del q
    cli('base', CACHE, root2)
    a = _final_models(fit2)
    res['sigint'].update(final_models_bitwise_equal_uninterrupted=_models_equal(a[0], b[0]), numpy_rng_equal=a[2] == b[2],
                         curve_equal=_curve(fit2) == _curve(ref),
                         receipt_resumed_from=json.loads((fit2 / 'fit_receipt.json').read_text())['resumed_from'])
    ok = all(res[k]['final_models_bitwise_equal_uninterrupted'] and res[k]['numpy_rng_equal'] and res[k]['curve_equal'] for k in res)
    res['PASS'] = bool(ok and res['stop_after']['generated_equal'] and 0 < res['sigint']['recovery_completed_updates'] < 100)
    save_result('resume', res)
    log('resume', json.dumps(res, default=str)[:3000])
    assert res['PASS']
    return res


def resume_offset(n_updates=20):
    """Offset arm (with covars,genemean): bitwise determinism and bitwise resume (fixer).

    (a) in-process: the same arm built 3 times, 6 updates each -> all parameters bitwise equal;
    (b) CLI: uninterrupted n_updates vs --stop-after 12 (checkpoint every 10) + resume -> final
        models, numpy RNG, curve and generated.npz bitwise equal;
    (c) generation-only path: config.json + final.pt copied into a fresh root, then trainer.py with a
        different --updates must fail; with the same --updates it generates and equals (b).
    """
    ensure_cache()
    arm = 'covars,genemean,offset'
    res = {'arm': arm}
    runs = []
    for _ in range(3):
        ctx = trainer.Context(CACHE, 'cpu', arms.parse_arm(arm), allow_nonprotocol_split=True)
        b, g, r, u, _ = trainer.build(ctx, 0)
        for _ in range(6):
            trainer.advance(ctx, b, g, r, u, execution_limit=20000)
        runs.append([{k: v.detach().clone() for k, v in m.state_dict().items()} for m in (b[1], b[2], b[3])])
        del ctx, b
    diffs = [max(float((runs[0][i][k].double() - runs[j][i][k].double()).abs().max()) for i in range(3) for k in runs[0][i])
             for j in (1, 2)]
    res['in_process_max_param_diff'] = diffs
    root_a = DEV / 'fits_offset_uninterrupted'
    root_b = DEV / 'fits_offset_stopafter'
    root_c = DEV / 'fits_offset_genonly'
    for root in (root_a, root_b, root_c):
        shutil.rmtree(root, ignore_errors=True)
    cli(arm, CACHE, root_a, '--checkpoint-every', '10', updates=n_updates)
    p, _ = cli(arm, CACHE, root_b, '--checkpoint-every', '10', '--stop-after', '12', check=False, updates=n_updates)
    fa = root_a / arm / 'devlegacy' / 'fold0' / 'seed0'
    fb = root_b / arm / 'devlegacy' / 'fold0' / 'seed0'
    q = torch.load(fb / 'recovery.pt', map_location='cpu', weights_only=False)
    res['stop_after'] = {'first_returncode': p.returncode, 'recovery_completed_updates': q['completed_updates']}
    del q
    cli(arm, CACHE, root_b, '--checkpoint-every', '10', updates=n_updates)
    a, b2 = _final_models(fa), _final_models(fb)
    ga, gb = np.load(fa / 'generated.npz'), np.load(fb / 'generated.npz')
    res['stop_after'].update(final_models_bitwise_equal_uninterrupted=_models_equal(a[0], b2[0]),
                             numpy_rng_equal=a[2] == b2[2], curve_equal=_curve(fa) == _curve(fb),
                             generated_equal=all(np.array_equal(ga[k], gb[k]) for k in ga.files),
                             resumed_from=json.loads((fb / 'fit_receipt.json').read_text())['resumed_from'])
    fc = root_c / arm / 'devlegacy' / 'fold0' / 'seed0'
    fc.mkdir(parents=True)
    for name in ('config.json', 'final.pt', 'fit_receipt.json'):
        shutil.copy2(fa / name, fc / name)
    p, _ = cli(arm, CACHE, root_c, '--checkpoint-every', '10', check=False, updates=n_updates + 10)
    res['genonly_other_updates'] = {'returncode': p.returncode, 'refused': 'refusing' in p.stderr,
                                    'generated_written': (fc / 'generated.npz').exists()}
    cli(arm, CACHE, root_c, '--checkpoint-every', '10', updates=n_updates)
    gc_ = np.load(fc / 'generated.npz')
    res['genonly_same_updates_generated_equal'] = all(np.array_equal(ga[k], gc_[k]) for k in ga.files)
    s = res['stop_after']
    res['PASS'] = bool(all(d == 0.0 for d in diffs) and s['first_returncode'] != 0 and s['recovery_completed_updates'] == 12
                       and s['resumed_from'] == 12 and s['final_models_bitwise_equal_uninterrupted'] and s['numpy_rng_equal']
                       and s['curve_equal'] and s['generated_equal'] and res['genonly_other_updates']['returncode'] != 0
                       and res['genonly_other_updates']['refused'] and not res['genonly_other_updates']['generated_written']
                       and res['genonly_same_updates_generated_equal'])
    save_result('resume_offset', res)
    log('resume_offset', json.dumps(res, default=str)[:3000])
    for root in (root_a, root_b, root_c):
        shutil.rmtree(root, ignore_errors=True)
    assert res['PASS']
    return res


def cleanup():
    """Delete test outputs larger than ~2 MB (keep caches, small json/curves)."""
    removed = []
    for root in [FITS, DEV / 'fits_resume_stopafter', DEV / 'fits_resume_sigint']:
        if not root.exists():
            continue
        for p in root.rglob('*'):
            if p.is_file() and p.stat().st_size > 2 * 2 ** 20:
                removed.append((str(p), p.stat().st_size))
                p.unlink()
    log('cleanup removed', len(removed), 'files', sum(s for _, s in removed) / 2 ** 20, 'MB')
    return removed


if __name__ == '__main__':
    what = sys.argv[1] if len(sys.argv) > 1 else 'all'
    steps = {'devcache': devcache, 'parity': parity, 'grads': grads, 'arms': run_arms, 'timing': timing,
             'resume': resume, 'resume_offset': resume_offset, 'cleanup': cleanup}
    if what == 'all':
        for k in ['devcache', 'parity', 'grads', 'arms', 'timing', 'resume', 'resume_offset']:
            steps[k]()
    else:
        steps[what]()
