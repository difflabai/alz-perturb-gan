"""Tests and real-data check for the F1 covariate arms (arm_covdrop.py, arm_covbin.py, mode_margcov.py).

CPU only, 3 threads per process, development donors only (inner-fold caches;
the sealed donors are never read).  Run from anywhere::

    python3 -B r2/test_arm_cov.py unit      # in-process: flags, encodings, tuple stats folds 0-8, streams, dropout, plans
    python3 -B r2/test_arm_cov.py cli       # tiny CLI fits (20 updates, fold 0): covdrop@1 == base bitwise, streams,
                                            # owncov/margcov generation, covdrop resume
    python3 -B r2/test_arm_cov.py posthoc   # owncov + margcov CPU generation for the 12 round-1 covariate fits
    python3 -B r2/test_arm_cov.py metrics   # held-out / train pb_rmse of those outputs (+ base reference)
    python3 -B r2/test_arm_cov.py cleanup   # delete the tiny-fit scratch tree (keeps results/*.json)

Scratch: runs/improve_20261109/smoke_r3/r2copy_tests/cov_test/ (results/*.json kept).
Post-hoc outputs: runs/improve_20261109/posthoc_r2/cov/{arm}/hvg2000_fold{k}_seed{s}_{mode}.npz
(+ .receipt.json) and posthoc_r2/cov/pb_rmse.json.

Metric (own code, same definition as round-1 metrics.pseudobulk_metrics):
pred_t = mean over the M generated donors of log1p(CP10k(pb_counts[t,m])),
obs_t = log1p(CP10k(sum of all of donor t's cache cells)); per donor RMSE over
the panel genes; per fit the mean over held-out (or train) donors; summary =
mean over the six fits (2 seeds x 3 folds, so fold-balanced).  The 2-seed
ensemble averages pred_t over the two seeds of a fold.  ``self_match`` (train
targets): fraction whose prediction is closer (RMSE) to its own observed
profile than to every other train donor's, a simple donor-memorisation
indicator.
"""
import os
import sys

sys.dont_write_bytecode = True
for _k in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[_k] = '3'
os.environ['CUDA_VISIBLE_DEVICES'] = ''

import json
import shutil
import subprocess
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) in sys.path:
    sys.path.remove(str(HERE))
sys.path.insert(0, str(HERE))

import numpy as np
import scipy.sparse as sp
import torch

torch.set_num_threads(3)
import trainer  # noqa: E402
import arms  # noqa: E402
import generate  # noqa: E402
import arm_covdrop  # noqa: E402,F401
import arm_covbin  # noqa: E402
import mode_margcov  # noqa: E402

PY = '/mnt/nvme/ai/conda/anaconda3/bin/python3'
RUNS = trainer.RUNS
CACHES = RUNS / 'caches/hvg2000'
CACHE0 = CACHES / 'fold0'
SCR = RUNS / 'smoke_r3/r2copy_tests/cov_test'
RES = SCR / 'results'
POST = RUNS / 'smoke_r3/r2copy_tests/posthoc_cov'
TINY = 20
ENV = dict(os.environ, CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='3', OPENBLAS_NUM_THREADS='3', MKL_NUM_THREADS='3',
           PYTHONDONTWRITEBYTECODE='1')
R1_ARMS = ['covars', 'covars,genemean,offset']
REF_BASE = 0.689


def log(*a):
    print('[test_arm_cov]', *a, flush=True)


def save(name, obj):
    trainer.dump(RES / f'{name}.json', obj)


def states(*mods):
    return [{k: v.detach().cpu().clone() for k, v in m.state_dict().items()} for m in mods]


def sd_equal(a, b):
    return set(a) == set(b) and all(torch.equal(a[k], b[k]) for k in a)


def run_parallel(cmds, n=3, may_fail=()):
    """Run commands, at most n at a time; raise with the log tail on failure (except commands listed in may_fail)."""
    pending, live, done = list(cmds), [], []
    while pending or live:
        while pending and len(live) < n:
            c = pending.pop(0)
            lp = SCR / 'logs' / (str(abs(hash(tuple(map(str, c))))) + '.log')
            lp.parent.mkdir(parents=True, exist_ok=True)
            f = lp.open('w')
            live.append((c, subprocess.Popen([str(x) for x in c], env=ENV, stdout=f, stderr=subprocess.STDOUT, cwd='/tmp'), f, lp, time.monotonic()))
        time.sleep(2)
        for item in list(live):
            c, p, f, lp, t0 = item
            if p.poll() is not None:
                f.close()
                live.remove(item)
                if p.returncode != 0 and c not in may_fail:
                    raise RuntimeError(f'{c} failed rc={p.returncode}:\n{lp.read_text()[-4000:]}')
                done.append((c, time.monotonic() - t0))
                lp.unlink()
    return done


# ============================================================================ unit
def all_raw_c():
    """raw_c of the 37 development donors from the fold-0 cache (train + held-out metadata; no counts)."""
    out = {}
    for part in ('train', 'heldout'):
        m = np.load(CACHE0 / part / 'metadata.npz', allow_pickle=False)
        for d, r in zip(m['donors'].astype(str), m['raw_c'].astype(np.float64)):
            out[d] = r
    return out


def tuple_stats_all_folds():
    raw = all_raw_c()
    q = trainer.load_folds()
    assert set(raw) == set(q['development_donors']) and len(raw) == 37
    res = {}
    for f in q['inner_folds']:
        tr = np.stack([raw[d] for d in f['train_donors']])
        he = np.stack([raw[d] for d in f['heldout_donors']])
        res[f['inner_fold']] = arm_covbin.tuple_stats(tr, he, arms.stage_of(tr))
    return res


def unit():
    res = {}
    refused = {}
    for bad in ['covdrop', 'covdrop@0', 'covdrop@0.50', 'covdrop@1.5', 'covdrop@-0.5', 'covdrop@0.5,covars', 'covbin@1',
                'covbin,covars']:
        try:
            arms.parse_arm(bad)
            refused[bad] = False
        except ValueError:
            refused[bad] = True
    res['refused'] = refused
    assert all(refused.values()), refused
    assert arms.parse_arm('covdrop@0.5,covbin') == ('covbin', 'covdrop@0.5')
    assert arms.parse_arm('genemean,covdrop@0.25') == ('covdrop@0.25', 'genemean')
    for m in ('margcov', 'owncov'):
        spec, v, _ = arms.parse_mode(m)
        assert spec.module == 'mode_margcov' and v is None
    for bad in ('margcov@1', 'owncov@0.5'):
        try:
            arms.parse_mode(bad)
            raise AssertionError(bad)
        except ValueError:
            pass

    # tuple sharing, folds 0-8 (metadata only)
    ts = tuple_stats_all_folds()
    res['tuple_stats'] = ts
    for k, s in ts.items():
        log(f"fold {k}: covbin unique {s['covbin']['unique_tuples']}/{s['covbin']['n_donors']} singletons "
            f"{s['covbin']['singleton_donors']} sizes {s['covbin']['donors_per_tuple']} | covars unique "
            f"{s['covars_round1']['unique_tuples']} | +stage covbin unique {s['covbin_with_stage']['unique_tuples']} "
            f"singletons {s['covbin_with_stage']['singleton_donors']} | heldout absent {s['heldout_tuple_absent_from_train']}/{s['heldout_n']}")

    # conditions / encodings on fold 0
    ctxb = trainer.Context(CACHE0, 'cpu', ())
    ctxd = trainer.Context(CACHE0, 'cpu', arms.parse_arm('covdrop@0.5'))
    ctxn = trainer.Context(CACHE0, 'cpu', arms.parse_arm('covbin'))
    ctxnd = trainer.Context(CACHE0, 'cpu', arms.parse_arm('covbin,covdrop@0.5'))
    ctxc = trainer.Context(CACHE0, 'cpu', ('covars',))
    raw = ctxb.raw_c
    cd, cn, cnd, cb, cc = (x.conditions.numpy() for x in (ctxd, ctxn, ctxnd, ctxb, ctxc))
    assert np.array_equal(cd[:, :4], cb[:, :4]) and np.array_equal(cn[:, :4], cb[:, :4])
    assert np.array_equal(cd[:, 4], 2 * raw[:, 4] - 1) and np.isin(cd[:, 4], (-1, 1)).all()
    assert np.allclose(cd[:, 5:7].mean(0), 0, atol=1e-6) and np.allclose(cd[:, 5:7].std(0), 1, atol=1e-5)
    assert np.allclose(cd[:, 6], cc[:, 6], atol=1e-6), 'covdrop age z must equal covars age z'
    assert np.array_equal(cnd, cn) and set(np.unique(cn[:, 6])) <= {-1, 0, 1} and np.isin(cn[:, 5], (-1, 1)).all()
    held = np.array([False, True])
    tc = ctxd.target_conditions(raw[:2], held)
    assert np.array_equal(tc[:, 4:7], np.zeros((2, 3), np.float32)) and np.array_equal(tc[:, :4], cb[:2, :4])
    tc = ctxnd.target_conditions(raw[:2], held)
    assert np.array_equal(tc, ctxb.target_conditions(raw[:2], held))
    assert np.array_equal(ctxn.target_conditions(raw[:2], held), cn[:2])
    res['covbin_info'] = ctxn.plugin_info()['covbin']['info']

    # covdrop@1 == base (3 updates, bitwise); covdrop@0.5 / covbin keep every released stream
    def stream_state(r, g, pol):
        return dict(numpy=r.bit_generator.state, torch_global=torch.get_rng_state(),
                    **{k: v.get_state() for k, v in g.items()},
                    **{k: getattr(pol, k).get_state() for k in ('latent_generator', 'penalty_generator', 'eval_generator', 'noise_generator')})

    def run(ctx, n=3):
        b, g, r, u, s = trainer.build(ctx, 0)
        init = states(b[1], b[2], b[3])
        mets, st = [], []
        for _ in range(n):
            mets.append(trainer.advance(ctx, b, g, r, u, execution_limit=20000))
            st.append(stream_state(r, g, b[-1]))
        return dict(init=init, fin=states(b[1], b[2], b[3]), mets=mets, streams=st)

    def diff_streams(A, B, i):
        a, b = A['streams'][i], B['streams'][i]
        return sorted(k for k in a if not (a[k] == b[k] if k == 'numpy' else torch.equal(a[k], b[k])))

    keys = ['D_adversarial', 'D_penalty', 'G_adversarial', 'critic_gap', 'penalty_stats']
    B = run(ctxb)
    ctx1 = trainer.Context(CACHE0, 'cpu', arms.parse_arm('covdrop@1'))
    D1 = run(ctx1)
    res['covdrop1_equals_base_3_updates'] = bool(all(sd_equal(x, y) for x, y in zip(B['init'] + B['fin'], D1['init'] + D1['fin']))
                                                and all(m1[k] == m2[k] for m1, m2 in zip(B['mets'], D1['mets']) for k in keys)
                                                and all(not diff_streams(B, D1, i) for i in range(3)))
    D5 = run(ctxd)
    N = run(ctxn)
    ND = run(ctxnd)
    # Every released stream except the Poisson 'count' generator stays bitwise base's after every update.  'count'
    # may differ: torch.poisson on CPU consumes a rate-dependent number of draws, so it diverges as soon as the
    # generated rates differ (conditions or parameters; true for any two arms, e.g. genemean vs base).  covdrop@1
    # (identical conditions) keeps 'count' equal too.
    for nm, X in [('covdrop05', D5), ('covbin', N), ('covbin_covdrop05', ND)]:
        res[f'{nm}_streams_differing_from_base_after_update'] = [diff_streams(B, X, i) for i in range(3)]
        res[f'{nm}_released_streams_ok'] = all(set(diff_streams(B, X, i)) <= {'count'} for i in range(3))
    res['covdrop05_initial_tensors_equal_base'] = all(sd_equal(x, y) for x, y in zip(B['init'], D5['init']))
    res['covdrop05_models_differ_from_base'] = not all(sd_equal(x, y) for x, y in zip(B['fin'], D5['fin']))
    log('streams', {k: v for k, v in res.items() if k.startswith('cov') and not k.endswith('info')})
    assert res['covdrop1_equals_base_3_updates'] and res['covdrop05_released_streams_ok']
    assert res['covbin_released_streams_ok'] and res['covbin_covdrop05_released_streams_ok']
    assert res['covdrop05_initial_tensors_equal_base'] and res['covdrop05_models_differ_from_base']
    # arm stream: 2 x 32 draws per update
    ref = np.random.default_rng(ctxd.arm_seed('covdrop', 0))
    ref.random(32 * 2 * 3)
    assert ctxd.arm_rngs['covdrop'].bit_generator.state == ref.bit_generator.state

    # dropout behaviour of the batch hook
    ctxd.arm_rngs['covdrop'] = np.random.default_rng(123)
    n_drop = n_rows = 0
    for i in range(100):
        _, c, _ = ctxd.batch(np.random.default_rng(1000 + i), 'critic' if i % 2 == 0 else 'generator')
        ds = np.random.default_rng(1000 + i).integers(len(ctxd.ids), size=32)   # the batch's first draw
        full = ctxd.conditions[torch.as_tensor(ds)]
        drop = (c[:, 4] == 0)
        assert torch.equal(c[:, :4], full[:, :4]) and torch.equal(c[~drop], full[~drop])
        assert bool((c[drop, 4:7] == 0).all())
        n_drop += int(drop.sum()); n_rows += 32
    res['dropout_fraction_p05'] = n_drop / n_rows
    assert abs(res['dropout_fraction_p05'] - 0.5) < 4 * np.sqrt(0.25 / n_rows)

    # checkpoint round trip restores the covdrop stream
    c1 = trainer.Context(CACHE0, 'cpu', arms.parse_arm('covdrop@0.5'))
    b1, g1, r1, u1, s1 = trainer.build(c1, 0)
    trainer.advance(c1, b1, g1, r1, u1, execution_limit=20000)
    q = trainer.snapshot(c1, b1, g1, r1, u1, 0, s1, 'incomplete', endpoint=20000, stream_root=trainer.STREAM_ROOT, wall_seconds=0.)
    for _ in range(2):
        trainer.advance(c1, b1, g1, r1, u1, execution_limit=20000)
    c2 = trainer.Context(CACHE0, 'cpu', arms.parse_arm('covdrop@0.5'))
    b2, g2, r2, u2, _ = trainer.restore(c2, q, 0)
    for _ in range(2):
        trainer.advance(c2, b2, g2, r2, u2, execution_limit=20000)
    res['covdrop_restore_bitwise'] = (all(sd_equal(x, y) for x, y in zip(states(b1[1], b1[2], b1[3]), states(b2[1], b2[2], b2[3])))
                                      and c1.arm_rngs['covdrop'].bit_generator.state == c2.arm_rngs['covdrop'].bit_generator.state)
    assert res['covdrop_restore_bitwise'] and set(q['arm_streams']) == {'covdrop'}
    del c1, c2, b1, b2

    # mode plans: CRN picks shared across arms, encodings, owncov
    T = generate.targets(ctxb)
    dev = trainer.development_order()
    tr_stage = arms.stage_of(ctxb.raw_c)
    plans = {}
    for name, ctx in [('covars', ctxc), ('covdrop@0.5', ctxd), ('covbin', ctxn), ('covbin,covdrop@0.5', ctxnd)]:
        plans[name] = {m: generate.mode_plans(ctx, m, T, ('final',))[0] for m in ('margcov', 'owncov')}
        default, _ = generate.default_plans(ctx, T)
        for ti, (donor, is_h, rawt, _) in enumerate(T):
            picks = mode_margcov.margcov_picks(ctxb.ids, ctxb.raw_c, donor, int(arms.stage_of(rawt[None])[0]), is_h, ctxb.fold, dev[donor])
            assert (tr_stage[picks] == arms.stage_of(rawt[None])[0]).all() and (is_h or donor not in ctxb.ids[picks])
            rr = np.repeat(rawt[None], 32, 0)
            rr[:, 4:7] = ctxb.raw_c[picks, 4:7]
            want = default[ti][0].copy()
            want[:, 4:7] = ctx.condition_matrix(rr)[:, 4:7]
            cond, nm = plans[name]['margcov'][ti]
            assert np.array_equal(cond, want) and nm == ['final'] * 32
            own, _ = plans[name]['owncov'][ti]
            assert np.array_equal(own[:, :4], default[ti][0][:, :4])
            assert np.array_equal(own[:, 4:7], np.repeat(ctx.condition_matrix(rawt[None])[:, 4:7], 32, 0))
            if name in ('covars', 'covbin'):
                assert np.array_equal(own, default[ti][0])
    res['margcov_unique_pick_fraction'] = float(np.mean([len(np.unique(p[0][:, 4:7], axis=0)) / 32 for p in plans['covars']['margcov']]))
    try:
        generate.mode_plans(ctxb, 'margcov', T[:1], ('final',))
        raise AssertionError('margcov accepted a base fit')
    except ValueError:
        pass
    res['PASS'] = True
    save('unit', res)
    log('unit PASS')
    return res


# ============================================================================ CLI tiny fits
def fitpath(arm):
    return SCR / 'fits' / arm / 'hvg2000' / 'fold0' / 'seed0'


def train_cmd(arm, *extra, root=None):
    return [PY, '-B', HERE / 'trainer.py', '--cache', CACHE0, '--arm', arm, '--seed', '0', '--out-root', root or SCR / 'fits',
            '--updates', str(TINY), '--device', 'cpu', '--threads', '3', *extra]


def gen_cmd(fit, mode, out=None):
    c = [PY, '-B', HERE / 'generate.py', '--fit', fit, '--cache', CACHE0, '--mode', mode, '--threads', '3']
    return c + (['--out', out] if out else [])


def npz(p):
    with np.load(p, allow_pickle=False) as q:
        return {k: q[k] for k in q.files}


def curve(fit):
    rows = [json.loads(l) for l in (fit / 'learning_curve.jsonl').read_text().splitlines()]
    for r in rows:
        r.pop('wall_seconds')
    return rows


def cli():
    res = {}
    fits = ['base', 'covdrop@1', 'covdrop@0.5', 'covbin', 'covbin,covdrop@0.5']
    rfit = SCR / 'fits_resume' / 'covdrop@0.5' / 'hvg2000' / 'fold0' / 'seed0'
    modes_done = all((fitpath(a) / f'generated_{m}.npz').exists() for a in ('covdrop@0.5', 'covbin') for m in ('owncov', 'margcov'))
    if not (modes_done and (rfit / 'generated.npz').exists() and all((fitpath(a) / 'generated.npz').exists() for a in fits)):
        shutil.rmtree(SCR / 'fits', ignore_errors=True)       # fresh run (a complete scratch tree is reused as is)
        shutil.rmtree(SCR / 'fits_resume', ignore_errors=True)
        stop = train_cmd('covdrop@0.5', '--stop-after', '13', '--checkpoint-every', '10', root=SCR / 'fits_resume')
        t = run_parallel([train_cmd(a) for a in fits] + [stop], may_fail=[stop])
        res['train_seconds'] = [x[1] for x in t]
        assert (rfit / 'recovery.pt').exists() and not (rfit / 'final.pt').exists()
        q = torch.load(rfit / 'recovery.pt', map_location='cpu', weights_only=False)
        assert q['completed_updates'] == 13 and set(q['arm_streams']) == {'covdrop'}
        del q
        run_parallel([train_cmd('covdrop@0.5', root=SCR / 'fits_resume')]
                     + [gen_cmd(fitpath(a), m) for a in ('covdrop@0.5', 'covbin') for m in ('owncov', 'margcov')])
    rc = json.loads((rfit / 'fit_receipt.json').read_text())
    res['resume_receipt_resumed_from'] = rc['resumed_from']
    assert rc['resumed_from'] == 13

    def fq(a):
        return torch.load(fitpath(a) / 'final.pt', map_location='cpu', weights_only=False)
    qb, q1, qd, qn, qnd = (fq(a) for a in fits)
    # covdrop@1 == base bitwise: models, curve, generated arrays
    gb, g1 = npz(fitpath('base') / 'generated.npz'), npz(fitpath('covdrop@1') / 'generated.npz')
    res['covdrop1_vs_base'] = {
        'models': all(sd_equal(qb['policy']['models'][k], q1['policy']['models'][k]) for k in qb['policy']['models']),
        'curve': curve(fitpath('base')) == curve(fitpath('covdrop@1')),
        'generated': set(gb) == set(g1) and all(np.array_equal(gb[k], g1[k]) for k in gb)}
    assert all(res['covdrop1_vs_base'].values()), res['covdrop1_vs_base']

    def same_streams(qa, qx, skip=()):
        # 'count' (Poisson) draws are rate dependent and diverge whenever the models differ (see unit()).
        return (qa['numpy_rng'] == qx['numpy_rng'] and torch.equal(qa['torch_cpu_rng'], qx['torch_cpu_rng'])
                and all(torch.equal(qa['generators'][k], qx['generators'][k]) for k in qa['generators'] if k not in skip))
    res['released_streams_equal_base_after_20'] = {a: same_streams(qb, qx, () if a == 'covdrop@1' else ('count',))
                                                   for a, qx in zip(fits[1:], (q1, qd, qn, qnd))}
    assert all(res['released_streams_equal_base_after_20'].values()), res['released_streams_equal_base_after_20']
    ss = np.random.SeedSequence([trainer.STREAM_ROOT, arms.ARM_STREAM_TAG, 0, 0, arms.crc('covdrop'), 0])
    ref = np.random.default_rng(int(ss.generate_state(1, dtype=np.uint64)[0]) % (2 ** 63 - 16))   # = ctx.arm_seed
    ref.random(32 * 2 * TINY)
    res['covdrop_stream_draws_64_per_update'] = qd['arm_streams']['covdrop'] == ref.bit_generator.state
    assert res['covdrop_stream_draws_64_per_update']
    cfg = json.loads((fitpath('covdrop@0.5') / 'config.json').read_text())
    assert cfg['round'] in (2, 3) and cfg['critic_features'] == 150 and cfg['r2_arms']['covdrop@0.5']['value'] == 0.5
    res['covbin_tuple_stats_fold0'] = json.loads((fitpath('covbin') / 'config.json').read_text())['r2_arms']['covbin']['info']['tuple_stats']

    # resume == uninterrupted
    gr, gd = npz(rfit / 'generated.npz'), npz(fitpath('covdrop@0.5') / 'generated.npz')
    qr = torch.load(rfit / 'final.pt', map_location='cpu', weights_only=False)
    res['covdrop_resume_bitwise'] = (all(sd_equal(qr['policy']['models'][k], qd['policy']['models'][k]) for k in qd['policy']['models'])
                                     and curve(rfit) == curve(fitpath('covdrop@0.5')) and all(np.array_equal(gr[k], gd[k]) for k in gd)
                                     and qr['arm_streams'] == qd['arm_streams'])
    assert res['covdrop_resume_bitwise']

    # modes
    base_keys = [k for k in gd if k not in ('mode', 'state_names', 'state_index', 'conditions')]
    on, mn = npz(fitpath('covbin') / 'generated_owncov.npz'), npz(fitpath('covbin') / 'generated_margcov.npz')
    gn = npz(fitpath('covbin') / 'generated.npz')
    res['covbin_owncov_equals_default'] = all(np.array_equal(on[k], gn[k]) for k in base_keys)
    od, md = npz(fitpath('covdrop@0.5') / 'generated_owncov.npz'), npz(fitpath('covdrop@0.5') / 'generated_margcov.npz')
    res['covdrop_owncov_differs_from_default'] = not np.array_equal(od['pb_counts'], gd['pb_counts'])
    res['covdrop_mode_slot4_present'] = bool(np.isin(od['conditions'][..., 4], (-1, 1)).all() and np.isin(md['conditions'][..., 4], (-1, 1)).all())
    # same donor picks in both arms: sex and APOE4-carrier codes of margcov agree
    res['margcov_crn_sex_equal_across_arms'] = bool(np.array_equal(mn['conditions'][..., 4], md['conditions'][..., 4]))
    am, asd = cfg['r2_arms']['covdrop@0.5']['info']['apoe_mean_sd']
    carrier_d = np.where(md['conditions'][..., 5] > (0.5 - am) / asd, 1.0, -1.0)
    res['margcov_crn_carrier_equal_across_arms'] = bool(np.array_equal(carrier_d, mn['conditions'][..., 5]))
    res['metadata_equal'] = all(np.array_equal(md[k], gd[k]) for k in ('targets', 'is_heldout', 'stage', 'genes'))
    assert all(res[k] for k in ('covbin_owncov_equals_default', 'covdrop_owncov_differs_from_default', 'covdrop_mode_slot4_present',
                                'margcov_crn_sex_equal_across_arms', 'margcov_crn_carrier_equal_across_arms', 'metadata_equal')), res
    res['PASS'] = True
    save('cli', res)
    log('cli PASS', json.dumps({k: v for k, v in res.items() if k != 'covbin_tuple_stats_fold0'}, default=str)[:2000])
    return res


# ============================================================================ post-hoc on round-1 covariate fits
def r1_fit(arm, k, s):
    return RUNS / 'fits' / arm / 'hvg2000' / f'fold{k}' / f'seed{s}'


def post_out(arm, k, s, mode):
    return POST / arm / f'hvg2000_fold{k}_seed{s}_{mode}.npz'


# Core workaround (reported as an open issue, core files not edited): round-1 genemean fits were trained on CUDA, and
# their final.pt stores features.genemean_scale computed on the GPU (fold 0: 0.8241964702478473); the CPU recomputation
# gives 0.8241964723596995, so generate.check_final's exact arm-state comparison refuses them.  genemean_scale only
# scales a critic input block and is not used at generation, so this wrapper accepts a relative difference < 1e-6 in
# exactly that scalar (both copies in arm_state) and keeps every other check exact.
GM_WRAPPER = r'''
import sys
sys.dont_write_bytecode = True
sys.path.insert(0, {here!r})
import torch
import generate
_orig = generate.check_final
def check_final(ctx, config, q):
    live = ctx.arm_state()
    st = q['arm_state']
    for part in ('features', 'features_info'):
        a, b = st.get(part, {{}}).get('genemean_scale'), live.get(part, {{}}).get('genemean_scale')
        if a is not None and b is not None and a != b:
            if abs(a - b) > 1e-6 * abs(b):
                raise ValueError('genemean_scale differs by more than 1e-6 relative')
            print(f'[gm-wrapper] {{part}}.genemean_scale checkpoint {{a!r}} vs CPU {{b!r}} accepted (not used at generation)', flush=True)
            st[part]['genemean_scale'] = b
    return _orig(ctx, config, q)
generate.check_final = check_final
torch.set_num_threads(3)
generate.main({argv!r})
'''


def posthoc():
    cmds = []
    for arm in R1_ARMS:
        for k in range(3):
            for s in range(2):
                fit = r1_fit(arm, k, s)
                if not (fit / 'generated.npz').exists():
                    log('skip incomplete fit', fit)
                    continue
                for mode in ('owncov', 'margcov'):
                    o = post_out(arm, k, s, mode)
                    if not o.exists():
                        argv = ['--fit', str(fit), '--cache', str(CACHES / f'fold{k}'), '--mode', mode, '--out', str(o), '--threads', '3']
                        if 'genemean' in arm:
                            cmds.append([PY, '-B', '-c', GM_WRAPPER.format(here=str(HERE), argv=argv)])
                        else:
                            cmds.append([PY, '-B', HERE / 'generate.py', *argv])
    log(f'{len(cmds)} generations')
    done = run_parallel(cmds, n=3)
    log('generation seconds', [round(x[1]) for x in done])


def real_pb(k):
    """Observed per-donor pseudobulk counts [T,G] in cache order (train then held-out) for fold k."""
    out, ids, held = [], [], []
    for part in ('train', 'heldout'):
        m = np.load(CACHES / f'fold{k}' / part / 'metadata.npz', allow_pickle=False)
        X = sp.load_npz(CACHES / f'fold{k}' / part / 'counts.npz').tocsr()
        lab = m['donor'].astype(str)
        for d in m['donors'].astype(str):
            out.append(np.asarray(X[np.flatnonzero(lab == d)].sum(0), dtype=np.float64).ravel())
            ids.append(d); held.append(part == 'heldout')
    return np.stack(out), np.array(ids), np.array(held)


def logcp(x):
    x = np.asarray(x, dtype=np.float64)
    return np.log1p(x * (1e4 / np.maximum(x.sum(-1, keepdims=True), 1)))


def pred_profile(path):
    with np.load(path, allow_pickle=False) as q:
        pb = q['pb_counts'].astype(np.float64)
        tg = q['targets'].astype(str)
        hd = q['is_heldout']
    return np.stack([logcp(pb[t]).mean(0) for t in range(len(pb))]), tg, hd


def score(pred, tg, hd, obs, ids, held):
    assert list(tg) == list(ids) and np.array_equal(hd, held)
    rm = np.sqrt(((pred - obs) ** 2).mean(1))
    tr = np.flatnonzero(~held)
    d = np.sqrt(((pred[tr][:, None] - obs[tr][None]) ** 2).mean(-1))
    return {'heldout': float(rm[held].mean()), 'train': float(rm[~held].mean()),
            'self_match_train': float(np.mean(d.argmin(1) == np.arange(len(tr))))}


def metrics():
    out = {'definition': __doc__.split('Metric (own code')[1].split('\n\n')[0], 'per_fit': {}, 'summary': {}, 'ensemble': {}}
    rows = {}
    preds = {}
    for k in range(3):
        obs_c, ids, held = real_pb(k)
        obs = logcp(obs_c)
        for s in range(2):
            units = {('base', 'gpu_default'): r1_fit('base', k, s) / 'generated.npz'}
            for arm in R1_ARMS:
                units[(arm, 'gpu_default')] = r1_fit(arm, k, s) / 'generated.npz'
                for mode in ('owncov', 'margcov'):
                    units[(arm, 'cpu_' + mode)] = post_out(arm, k, s, mode)
            for (arm, src), p in units.items():
                if not p.exists():
                    continue
                pr, tg, hd = pred_profile(p)
                sc = score(pr, tg, hd, obs, ids, held)
                out['per_fit'][f'{arm}|{src}|fold{k}|seed{s}'] = sc
                rows.setdefault((arm, src), []).append(sc)
                preds[(arm, src, k, s)] = pr
        for (arm, src) in rows:
            if (arm, src, k, 0) in preds and (arm, src, k, 1) in preds:
                pe = (preds[(arm, src, k, 0)] + preds[(arm, src, k, 1)]) / 2
                out['ensemble'].setdefault(f'{arm}|{src}', []).append(score(pe, ids, held, obs, ids, held))
    for (arm, src), v in rows.items():
        out['summary'][f'{arm}|{src}'] = {'n_fits': len(v), **{m: float(np.mean([x[m] for x in v])) for m in v[0]}}
    out['ensemble'] = {key: {'n_folds': len(v), **{m: float(np.mean([x[m] for x in v])) for m in v[0]}} for key, v in out['ensemble'].items()}
    POST.mkdir(parents=True, exist_ok=True)
    trainer.dump(POST / 'pb_rmse.json', out)
    for key, v in sorted(out['summary'].items()):
        e = out['ensemble'].get(key, {})
        log(f"{key:45s} n={v['n_fits']} heldout {v['heldout']:.4f} train {v['train']:.4f} self_match {v['self_match_train']:.2f} "
            f"| 2-seed ens heldout {e.get('heldout', float('nan')):.4f}")
    return out


def cleanup():
    for d in ('fits', 'fits_resume', 'logs'):
        shutil.rmtree(SCR / d, ignore_errors=True)
    for p in SCR.glob('*.log'):
        p.unlink()


if __name__ == '__main__':
    what = sys.argv[1:] or ['unit']
    for w in what:
        t0 = time.monotonic()
        {'unit': unit, 'cli': cli, 'posthoc': posthoc, 'metrics': metrics, 'cleanup': cleanup}[w]()
        log(w, f'done in {time.monotonic() - t0:.0f}s')
