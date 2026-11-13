"""Tests for the round-2 core (r2/trainer.py, r2/arms.py, r2/generate.py, r2/run_screen.sh).

CPU only, 3 threads per process, real fold-0 hvg2000 cache (read-only).  Run::

    python3 -B r2/test_core.py parity     # r2 vs round-1 trainer: initial tensors + first updates bitwise (all round-1 arms)
    python3 -B r2/test_core.py example    # example plug-in: every hook, composition, checkpoint round trip (in-process)
    python3 -B r2/test_core.py cli        # tiny fits: r1 vs r2 trainer+generation bitwise, snapshots, resume, snapens, CRN
    python3 -B r2/test_core.py screen     # run_screen.sh with an '@' arm and a modes column; example-arm CLI resume
    python3 -B r2/test_core.py round1     # round-1 GPU final.pt through r2 generation (vs r1 code on CPU and vs stored GPU arrays)
    python3 -B r2/test_core.py cleanup    # delete the scratch tree
    python3 -B r2/test_core.py all

Scratch: /ml2/alz/runs/improve_20261109/smoke_r2/core_test/ (results/*.json are
small and kept until ``cleanup``).  Round-1 code is executed only in
subprocesses (sys.path = round-1 folder) so the round-1 and round-2 modules
named ``trainer``/``arms``/``generate`` never meet in one interpreter; nothing
is written into round-1 folders (round-1 trainer.py runs with --out-root in
the scratch tree).

CPU vs GPU (``round1``): the round-1 fits were trained AND generated on CUDA.
The CUDA and CPU torch generators produce different streams (Philox vs
mt19937), so a CPU regeneration cannot equal the stored GPU arrays except for
the metadata arrays (targets, is_heldout, stage, genes; asserted equal).  What
is asserted: (1) r2 CPU generation == round-1 generate_arrays on CPU with the
same final.pt state, bitwise, on a subset of targets (code equivalence);
(2) against the stored GPU arrays, per-gene means over all T x M generated
donors of mean_log, nonzero and log pseudobulk agree within Monte Carlo error:
z = diff / sqrt(var_cpu/n + var_gpu/n) with n = T*M = 1184 generated donors
treated as independent; mean z^2 in [0.5, 2] and at most 1% of genes with
|z| > 4; and the per-target mean generated library over the M donors (the
numpy library draws are identical, the latent draws are not) has |z| < 4
with z from the between-donor SD of the M = 32 generated donors, and differs
by < 5% relative (measured: max 1.0%, mean z^2 0.15 because the library draws
are shared).
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
import textwrap
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) in sys.path:
    sys.path.remove(str(HERE))
sys.path.insert(0, str(HERE))

import numpy as np
import torch

torch.set_num_threads(3)
import trainer  # noqa: E402  (r2)
import arms  # noqa: E402
import generate  # noqa: E402

PY = '/mnt/nvme/ai/conda/anaconda3/bin/python3'
R1 = HERE.parent
RUNS = trainer.RUNS
CACHE0 = RUNS / 'caches/hvg2000/fold0'
SCR = RUNS / 'smoke_r2/core_test'
RES = SCR / 'results'
R1_GPU_FIT = RUNS / 'fits/base/hvg2000/fold0/seed0'
TINY = 20
ENV = dict(os.environ, CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='3', OPENBLAS_NUM_THREADS='3', MKL_NUM_THREADS='3', PYTHONDONTWRITEBYTECODE='1')


def log(*a):
    print('[test_core]', *a, flush=True)


def save(name, obj):
    trainer.dump(RES / f'{name}.json', obj)


def sh(cmd, check=True, **kw):
    t = time.monotonic()
    p = subprocess.run([str(c) for c in cmd], env=ENV, capture_output=True, text=True, **kw)
    if check and p.returncode != 0:
        raise RuntimeError(f'{cmd} failed rc={p.returncode}:\n{p.stdout[-3000:]}\n{p.stderr[-5000:]}')
    return p, time.monotonic() - t


def r1_python(code):
    """Run code with the ROUND-1 modules (cwd and sys.path[0] = round-1 folder)."""
    pre = ("import sys,os; sys.dont_write_bytecode=True; sys.path.insert(0, %r)\n"
           "import torch; torch.set_num_threads(3)\n") % str(R1)
    return sh([PY, '-B', '-c', pre + textwrap.dedent(code)], cwd=str(R1))


def states(*mods):
    return [{k: v.detach().cpu().clone() for k, v in m.state_dict().items()} for m in mods]


def sd_equal(a, b):
    return set(a) == set(b) and all(torch.equal(a[k], b[k]) for k in a)


def models_equal(a, b):
    return set(a) == set(b) and all(sd_equal(a[k], b[k]) for k in a)


# ============================================================================ parity
PARITY_ARMS = {'base': 5, 'genemean': 5, 'offset': 2, 'donorpc': 2, 'covars': 2}
METRIC_KEYS = ['D_adversarial', 'D_penalty', 'D_total', 'G_adversarial', 'prior_regularization_unweighted', 'G_total', 'critic_gap',
               'D_gradient_norm', 'G_gradient_norm', 'prior_gradient_norm', 'G_prior_LRs', 'D_LRs', 'penalty_stats']


def parity():
    """r2 trainer == round-1 trainer for initial tensors and the first updates (bitwise, CPU)."""
    SCR.mkdir(parents=True, exist_ok=True)
    out = SCR / 'parity_r1.pt'
    r1_python(f"""
        import trainer, arms
        res = {{}}
        for arm, n in {PARITY_ARMS!r}.items():
            ctx = trainer.Context({str(CACHE0)!r}, 'cpu', arms.parse_arm(arm))
            b, g, r, u, s = trainer.build(ctx, 0)
            init = [{{k: v.detach().clone() for k, v in m.state_dict().items()}} for m in (b[1], b[2], b[3])]
            mets = [trainer.advance(ctx, b, g, r, u, execution_limit=20000) for _ in range(n)]
            fin = [{{k: v.detach().clone() for k, v in m.state_dict().items()}} for m in (b[1], b[2], b[3])]
            res[arm] = dict(init=init, metrics=mets, final=fin, numpy=r.bit_generator.state,
                            gens={{k: v.get_state() for k, v in g.items()}}, torch_rng=torch.get_rng_state(),
                            conditions=ctx.conditions.clone(), arm_state=ctx.arm_state(), n_features=ctx.features.n_features,
                            critic=type(b[2]).__name__, generator=type(b[1]).__name__, streams=s)
        torch.save(res, {str(out)!r})
    """)
    ref = torch.load(out, weights_only=False)
    res = {}
    for arm, n in PARITY_ARMS.items():
        R = ref[arm]
        ctx = trainer.Context(CACHE0, 'cpu', arms.parse_arm(arm))
        b, g, r, u, s = trainer.build(ctx, 0)
        init = states(b[1], b[2], b[3])
        mets = [trainer.advance(ctx, b, g, r, u, execution_limit=20000) for _ in range(n)]
        fin = states(b[1], b[2], b[3])
        row = {'updates': n,
               'initial_tensors_bitwise': [sd_equal(x, y) for x, y in zip(init, R['init'])],
               'metrics_bitwise': [all(m1[k] == m2[k] for k in METRIC_KEYS) for m1, m2 in zip(mets, R['metrics'])],
               'final_tensors_bitwise': [sd_equal(x, y) for x, y in zip(fin, R['final'])],
               'numpy_rng_equal': r.bit_generator.state == R['numpy'],
               'torch_generators_equal': all(torch.equal(g[k].get_state(), R['gens'][k]) for k in g),
               'torch_global_rng_equal': torch.equal(torch.get_rng_state(), R['torch_rng']),
               'conditions_equal': torch.equal(ctx.conditions, R['conditions']),
               'arm_state_equal': trainer._same(ctx.arm_state(), R['arm_state']),
               'streams_equal': s == R['streams'],
               'classes': [type(b[1]).__name__, type(b[2]).__name__, R['generator'], R['critic']],
               'n_features': [ctx.features.n_features, R['n_features']]}
        row['PASS'] = bool(all(row['initial_tensors_bitwise']) and all(row['metrics_bitwise']) and all(row['final_tensors_bitwise'])
                           and row['numpy_rng_equal'] and row['torch_generators_equal'] and row['torch_global_rng_equal']
                           and row['conditions_equal'] and row['arm_state_equal'] and row['streams_equal']
                           and row['classes'][:2] == row['classes'][2:] and row['n_features'][0] == row['n_features'][1])
        res[arm] = row
        log('parity', arm, row['PASS'])
        del ctx, b
    out.unlink()
    res['PASS'] = all(v['PASS'] for v in res.values())
    save('parity', res)
    assert res['PASS'], json.dumps(res, indent=1, default=str)
    return res


# ============================================================================ example plug-in
def example():
    import arm_example  # noqa: F401  (registers 'example' and mode 'exshrink')
    from pg_model import CountGenerator
    res = {}
    # flag parsing
    refused = {}
    for bad in ['example@0.50', 'example@1.5', 'genemean@1', 'genemean@0.50', 'example,offset', 'covars@1', 'example@', 'Example']:
        try:
            arms.parse_arm(bad)
            refused[bad] = False
        except ValueError:
            refused[bad] = True
    res['refused_bad_flags'] = refused
    res['parse'] = [arms.parse_arm('genemean@0.5,example@0.5'), arms.parse_arm('example')]
    assert all(refused.values()) and res['parse'] == [('example@0.5', 'genemean@0.5'), ('example',)]
    for bad in ['exshrink', 'snap@x', 'nomode']:
        try:
            arms.parse_mode(bad)
            raise AssertionError(f'mode {bad} accepted')
        except ValueError:
            pass
    assert arms.parse_mode('exshrink@0.5')[1] == 0.5 and arms.parse_mode('snap@16000')[1] == 16000

    # HookedCountGenerator == CountGenerator (same init draws, bitwise forward)
    ctxb = trainer.Context(CACHE0, 'cpu', ())
    torch.manual_seed(5)
    g1 = CountGenerator(ctxb.mu, ctxb.theta * 1.0, zd=8, zc=8)
    torch.manual_seed(5)
    g2 = arms.hooked_count_generator()(ctxb.mu, ctxb.theta * 1.0, zd=8, zc=8)
    z = torch.randn(4, 8)
    c = ctxb.conditions[:4]
    lib = ctxb.fitlibs[:4 * 64].reshape(4, 64)
    torch.manual_seed(11)  # Gamma rsample draws from the global RNG
    o1 = g1.rates(z, c, 64, lib, torch.Generator().manual_seed(3))
    torch.manual_seed(11)
    o2 = g2.rates(z, c, 64, lib, torch.Generator().manual_seed(3))
    res['hooked_generator_init_equal'] = sd_equal(states(g1)[0], states(g2)[0])
    res['hooked_generator_forward_bitwise'] = bool(torch.equal(o1, o2))
    assert res['hooked_generator_init_equal'] and res['hooked_generator_forward_bitwise']

    # (a) conditions, setup, (d) widths
    ctxe = trainer.Context(CACHE0, 'cpu', arms.parse_arm('example@0.5'))
    ce, cb = ctxe.conditions, ctxb.conditions
    res['cond_slot0_is_half_stage'] = bool(torch.equal(ce[:, 0], cb[:, 3] * 0.5)) and bool(torch.equal(ce[:, 1:], cb[:, 1:]))
    raws = np.stack([ctxe.raw_c[0], ctxe.raw_c[1]])
    tc = ctxe.target_conditions(raws, np.array([False, True]))
    res['target_conditions_hook'] = bool(np.allclose(tc[:, 0], 0.5 * tc[:, 3]))
    res['example_width'] = ctxe.features.n_features
    res['example_blocks'] = [(b.name, lo, hi) for a, b, lo, hi in ctxe.features.blocks]
    assert res['cond_slot0_is_half_stage'] and res['target_conditions_hook']
    assert res['example_width'] == 159 and res['example_blocks'] == [('pb8', 150, 158), ('depth1', 158, 159)]
    ctxc = trainer.Context(CACHE0, 'cpu', arms.parse_arm('example@0.5,genemean'))
    res['composed_width'] = ctxc.features.n_features
    res['composed_blocks'] = [(b.name, lo, hi) for a, b, lo, hi in ctxc.features.blocks]
    assert res['composed_width'] == 150 + 2000 + 9 and res['composed_blocks'][0][1] == 2150
    ctxg = trainer.Context(CACHE0, 'cpu', arms.parse_arm('genemean@0.5'))
    ctxg1 = trainer.Context(CACHE0, 'cpu', arms.parse_arm('genemean'))
    res['genemean_at_half_scale_ratio'] = ctxg.features.scale / ctxg1.features.scale
    assert res['genemean_at_half_scale_ratio'] == 0.5 and ctxg.features.info['genemean_scale'] == ctxg1.features.scale
    del ctxg, ctxg1, ctxc

    # (b)(c) build + updates; released streams untouched, own stream advanced
    bb, gb, rb, ub, _ = trainer.build(ctxb, 0)
    be, ge, re_, ue, _ = trainer.build(ctxe, 0)
    Gb, Ge = states(bb[1])[0], states(be[1])[0]
    res['generator_class_chain'] = [k.__name__ for k in type(be[1]).__mro__[:3]]
    res['released_generator_init_equal_base'] = all(torch.equal(Gb[k], Ge[k]) for k in Gb)
    res['extra_generator_params'] = sorted(set(Ge) - set(Gb))
    res['prior_init_equal_base'] = sd_equal(states(bb[3])[0], states(be[3])[0])
    res['critic'] = [type(be[2]).__name__, be[2].n_features]
    rng0 = ctxe.arm_rngs['example'].bit_generator.state
    in_group = any(p is be[1].example_bias for p in be[4].param_groups[0]['params'])
    for _ in range(3):
        trainer.advance(ctxb, bb, gb, rb, ub, execution_limit=20000)
        trainer.advance(ctxe, be, ge, re_, ue, execution_limit=20000)   # asserts G.example_bias and block grads at step 0
    res['donor_sampler_stream_equal_base_after_3'] = rb.bit_generator.state == re_.bit_generator.state
    res['arm_stream_advanced'] = ctxe.arm_rngs['example'].bit_generator.state != rng0
    res['example_bias_in_generator_group'] = bool(in_group)
    res['example_bias_moved'] = float(be[1].example_bias.detach().abs().max())
    assert res['released_generator_init_equal_base'] and res['extra_generator_params'] == ['example_bias'] and res['prior_init_equal_base']
    assert res['critic'] == ['WideVectorCritic', 159] and res['donor_sampler_stream_equal_base_after_3'] and res['arm_stream_advanced']
    assert res['example_bias_in_generator_group'] and res['example_bias_moved'] > 0
    # batch hook: slot 0 dropped for some donors, never others changed
    r_tmp = np.random.default_rng(0)
    ctxe.arm_rngs['example'] = np.random.default_rng(1)
    _, cbatch, _ = ctxe.batch(r_tmp, 'critic')
    full = ctxe.conditions[torch.as_tensor(np.random.default_rng(0).integers(len(ctxe.ids), size=32))]
    res['batch_hook_dropped'] = int((cbatch[:, 0] == 0).sum() - (full[:, 0] == 0).sum())
    assert torch.equal(cbatch[:, 1:], full[:, 1:]) and res['batch_hook_dropped'] > 0

    # (f) checkpoint round trip incl. the arm stream, exact arm-state check
    ctx1 = trainer.Context(CACHE0, 'cpu', arms.parse_arm('example@0.5'))
    b1, g1_, r1_, u1, s1 = trainer.build(ctx1, 0)
    for _ in range(2):
        trainer.advance(ctx1, b1, g1_, r1_, u1, execution_limit=20000)
    q = trainer.snapshot(ctx1, b1, g1_, r1_, u1, 0, s1, 'incomplete', endpoint=20000, stream_root=trainer.STREAM_ROOT, wall_seconds=0.)
    for _ in range(2):
        trainer.advance(ctx1, b1, g1_, r1_, u1, execution_limit=20000)
    ctx2 = trainer.Context(CACHE0, 'cpu', arms.parse_arm('example@0.5'))
    b2, g2_, r2_, u2, _ = trainer.restore(ctx2, q, 0)
    for _ in range(2):
        trainer.advance(ctx2, b2, g2_, r2_, u2, execution_limit=20000)
    res['restore_then_2_updates_bitwise'] = all(sd_equal(x, y) for x, y in zip(states(b1[1], b1[2], b1[3]), states(b2[1], b2[2], b2[3])))
    res['arm_stream_equal_after_restore'] = ctx1.arm_rngs['example'].bit_generator.state == ctx2.arm_rngs['example'].bit_generator.state
    assert 'r2' in q['arm_state'] and set(q['arm_state']['r2']['example@0.5']['blocks']) == {'pb8', 'depth1'}
    q['arm_state']['r2']['example@0.5']['arm']['proj'][0, 0] += 1
    try:
        trainer.restore(trainer.Context(CACHE0, 'cpu', arms.parse_arm('example@0.5')), q, 0)
        res['tampered_arm_state_refused'] = False
    except ValueError:
        res['tampered_arm_state_refused'] = True
    assert res['restore_then_2_updates_bitwise'] and res['arm_stream_equal_after_restore'] and res['tampered_arm_state_refused']
    res['PASS'] = True
    save('example', res)
    log('example', json.dumps(res, default=str)[:1500])
    return res


# ============================================================================ CLI tiny fits
def r2_train(arm, root, *extra, check=True, updates=TINY):
    return sh([PY, '-B', HERE / 'trainer.py', '--cache', CACHE0, '--arm', arm, '--seed', '0', '--out-root', root,
               '--updates', str(updates), '--device', 'cpu', '--threads', '3', *extra], check=check)


def fitpath(root, arm):
    return Path(root) / arm / 'hvg2000' / 'fold0' / 'seed0'


def final_models(fit):
    q = torch.load(fit / 'final.pt', map_location='cpu', weights_only=False)
    return q['policy']['models'], q['numpy_rng']


def curve(fit):
    rows = [json.loads(l) for l in (fit / 'learning_curve.jsonl').read_text().splitlines()]
    for r in rows:
        r.pop('wall_seconds')
    return rows


def npz(p):
    with np.load(p, allow_pickle=False) as q:
        return {k: q[k] for k in q.files}


def cli():
    res = {}
    r1root, r2root, r2res = SCR / 'r1fits', SCR / 'r2fits', SCR / 'r2fits_resume'
    for d in (r1root, r2res):
        shutil.rmtree(d, ignore_errors=True)
    shutil.rmtree(fitpath(r2root, 'base'), ignore_errors=True)
    # A: round-1 trainer (read-only code) into scratch; B: r2 trainer uninterrupted
    _, ta = sh([PY, '-B', R1 / 'trainer.py', '--cache', CACHE0, '--arm', 'base', '--seed', '0', '--out-root', r1root,
                '--updates', str(TINY), '--device', 'cpu', '--threads', '3'], cwd=str(R1))
    _, tb = r2_train('base', r2root)
    fa, fb = fitpath(r1root, 'base'), fitpath(r2root, 'base')
    ma, mb = final_models(fa), final_models(fb)
    ga, gb = npz(fa / 'generated.npz'), npz(fb / 'generated.npz')
    res['seconds'] = {'r1_fit_and_generate': ta, 'r2_fit_and_generate': tb}
    res['r1_vs_r2_final_models_bitwise'] = models_equal(ma[0], mb[0])
    res['r1_vs_r2_numpy_rng_equal'] = ma[1] == mb[1]
    res['r1_vs_r2_curve_equal'] = curve(fa) == curve(fb)
    res['r1_vs_r2_generated_bitwise'] = {k: bool(np.array_equal(ga[k], gb[k])) for k in ga}
    res['r2_extra_arrays'] = sorted(set(gb) - set(ga))
    sq = gb['sq_log']
    res['sq_log'] = {'dtype': str(sq.dtype), 'shape': list(sq.shape),
                     'ge_mean_log_sq': bool((sq.astype(np.float32) >= gb['mean_log'] ** 2 * (1 - 2e-3) - 1e-3).all())}
    assert res['r1_vs_r2_final_models_bitwise'] and res['r1_vs_r2_numpy_rng_equal'] and res['r1_vs_r2_curve_equal']
    assert all(res['r1_vs_r2_generated_bitwise'].values()) and res['r2_extra_arrays'] == ['sq_log']
    assert res['sq_log']['dtype'] == 'float16' and res['sq_log']['shape'] == list(gb['mean_log'].shape) and res['sq_log']['ge_mean_log_sq']
    # snapshots
    cfg = json.loads((fb / 'config.json').read_text())
    want = trainer.snapshot_updates(TINY)
    res['snapshot_updates'] = cfg['snapshot_updates']
    res['snapshot_updates_20000'] = trainer.snapshot_updates(20000)
    res['snapshot_updates_300'] = trainer.snapshot_updates(300)
    snaps = {u: torch.load(fb / f'snap_{u}.pt', map_location='cpu', weights_only=False) for u in want}
    res['snapshot_files_MB'] = {u: round((fb / f'snap_{u}.pt').stat().st_size / 2 ** 20, 2) for u in want}
    res['snapshot_keys'] = sorted(snaps[want[0]])
    res['snapshots_differ_from_final'] = [not sd_equal(snaps[u]['generator'], mb[0]['generator']) for u in want]
    assert want == [16, 17, 18, 19] and cfg['snapshot_updates'] == want and res['snapshot_updates_20000'] == [16000, 17000, 18000, 19000]
    assert res['snapshot_updates_300'] == [240, 255, 270, 285] and all(res['snapshots_differ_from_final'])
    rec = json.loads((fb / 'fit_receipt.json').read_text())
    assert rec['snapshots_missing'] == [] and set(rec['snapshots_SHA256']) == {str(u) for u in want}
    # C: interrupted (--stop-after 17, checkpoint every 10) + resume == uninterrupted
    p, _ = r2_train('base', r2res, '--checkpoint-every', '10', '--stop-after', '17', check=False)
    fc = fitpath(r2res, 'base')
    rq = torch.load(fc / 'recovery.pt', map_location='cpu', weights_only=False)
    res['stop_after'] = {'returncode': p.returncode, 'recovery_updates': rq['completed_updates'],
                         'snaps_before_resume': sorted(x.name for x in fc.glob('snap_*.pt'))}
    del rq
    r2_train('base', r2res, '--checkpoint-every', '10')
    mc = final_models(fc)
    gc_ = npz(fc / 'generated.npz')
    res['resume'] = {'final_models_bitwise': models_equal(mb[0], mc[0]), 'numpy_rng_equal': mb[1] == mc[1],
                     'curve_equal': curve(fb) == curve(fc), 'generated_bitwise': all(np.array_equal(gb[k], gc_[k]) for k in gb),
                     'snapshots_bitwise': all(models_equal({'g': snaps[u]['generator'], 'p': snaps[u]['prior']},
                                                           {'g': (q := torch.load(fc / f'snap_{u}.pt', map_location='cpu', weights_only=False))['generator'],
                                                            'p': q['prior']}) for u in want),
                     'resumed_from': json.loads((fc / 'fit_receipt.json').read_text())['resumed_from']}
    assert p.returncode != 0 and res['stop_after']['recovery_updates'] == 17 and res['stop_after']['snaps_before_resume'] == ['snap_16.pt', 'snap_17.pt']
    assert all(v for k, v in res['resume'].items() if k != 'resumed_from') and res['resume']['resumed_from'] == 17
    # refuse to overwrite a completed fit
    p, _ = r2_train('base', r2root, check=False)
    res['refuse_overwrite'] = p.returncode != 0 and 'FileExistsError' in p.stderr
    assert res['refuse_overwrite']
    # D: snapens mode (CLI) on the r2 fit
    p, td = sh([PY, '-B', HERE / 'generate.py', '--fit', fb, '--cache', CACHE0, '--mode', 'snapens', '--threads', '3'])
    se = npz(fb / 'generated_snapens.npz')
    counts = [int((se['state_index'][0] == i).sum()) for i in range(len(se['state_names']))]
    res['snapens'] = {'state_names': se['state_names'].tolist(), 'counts_per_state_target0': counts, 'seconds': td,
                      'same_index_all_targets': bool((se['state_index'] == se['state_index'][0]).all()),
                      'keys': sorted(se), 'conditions_equal_default': bool(np.allclose(se['conditions'][:, :, 3], gb['stage'][:, None] / 3, atol=1e-6)),
                      'metadata_equal': all(np.array_equal(se[k], gb[k]) for k in ('targets', 'is_heldout', 'stage', 'genes')),
                      'differs_from_default': not np.array_equal(se['pb_counts'], gb['pb_counts']),
                      'receipt': (fb / 'generated_snapens.receipt.json').exists()}
    assert res['snapens']['state_names'] == ['snap_16', 'snap_17', 'snap_18', 'snap_19', 'final'] and counts == [7, 7, 6, 6, 6]
    assert res['snapens']['same_index_all_targets'] and res['snapens']['metadata_equal'] and res['snapens']['differs_from_default']
    assert res['snapens']['conditions_equal_default'] and res['snapens']['receipt']
    p, _ = sh([PY, '-B', HERE / 'generate.py', '--fit', fb, '--cache', CACHE0, '--mode', 'snapens', '--threads', '3'], check=False)
    assert p.returncode != 0 and 'exists' in p.stderr
    # CRN: the mode machinery with an all-'final' default plan reproduces generated.npz (3 targets, in-process),
    # and snap@19 via load_models equals a model built from the snapshot file directly
    ctx = trainer.Context(CACHE0, 'cpu', ())
    T = generate.targets(ctx)
    sub = [T[0], T[1], T[-1]]
    plans, _ = generate.default_plans(ctx, sub)
    models = generate.load_models(ctx, fb, cfg, ['final', 'snap_19'])
    arr, _ = generate.generate_arrays(ctx, {'final': models['final']}, plans, T=sub)
    rows = [0, 1, len(T) - 1]
    res['crn_mode_machinery_bitwise'] = all(np.array_equal(arr[k], gb[k][rows]) for k in ('pb_counts', 'mean_log', 'nonzero', 'pcs', 'lib', 'sq_log'))
    mixed = [(c, ['snap_19'] * 4 + ['final'] * 28) for c, _ in plans]
    arr2, _ = generate.generate_arrays(ctx, models, mixed, T=sub)
    res['mixed_plan_final_part_differs_but_valid'] = bool(np.isfinite(arr2['mean_log']).all())
    # load_models(final) == round-1 restore() path for a CPU fit (policy-restored G/P)
    q = torch.load(fb / 'final.pt', map_location='cpu', weights_only=False)
    bres, *_ = trainer.restore(ctx, q, 0)
    res['load_models_equals_restore'] = sd_equal(states(bres[1])[0], states(models['final'][0])[0]) and \
        sd_equal(states(bres[3])[0], states(models['final'][1])[0])
    assert res['crn_mode_machinery_bitwise'] and res['load_models_equals_restore']
    res['PASS'] = True
    save('cli', res)
    log('cli', json.dumps(res, default=str)[:3000])
    return res


# ============================================================================ run_screen + example CLI
def screen():
    import arm_example  # noqa: F401
    res = {}
    root, logs = SCR / 'r2fits', SCR / 'logs'
    arm = 'example@0.5'
    fit = fitpath(root, arm)
    shutil.rmtree(fit, ignore_errors=True)
    jobs = SCR / 'jobs.txt'
    jobs.write_text(f'# test\n{arm} hvg2000 0 0 snapens,exshrink@0.5\n')
    p, _ = sh(['bash', HERE / 'run_screen.sh', '-d', 'cpu', '-p', '1', '-j', jobs, '-u', str(TINY), '-o', root, '-l', logs, '-n'])
    res['dry_run'] = p.stdout.strip().splitlines()[-3:]
    assert 'trainer.py' in p.stdout and '--mode snapens' in p.stdout and '--mode exshrink@0.5' in p.stdout
    bad = SCR / 'jobs_bad.txt'
    for line in ['example@0.5 hvg2000 9 0', 'example@ hvg2000 0 0', 'base hvg2000 0 0 snap@ x']:
        bad.write_text(line + '\n')
        p, _ = sh(['bash', HERE / 'run_screen.sh', '-d', 'cpu', '-p', '1', '-j', bad, '-n'], check=False)
        assert p.returncode == 2, line
    p, _ = sh(['bash', HERE / 'run_screen.sh', '-d', 'cpu', '-p', '1', '-j', jobs, '-t', '6', '-n'], check=False)
    assert p.returncode == 2
    p, ts = sh(['bash', HERE / 'run_screen.sh', '-d', 'cpu', '-p', '1', '-j', jobs, '-u', str(TINY), '-o', root, '-l', logs])
    res['screen_seconds'] = ts
    res['screen_tail'] = p.stdout.strip().splitlines()[-2:]
    files = sorted(x.name for x in fit.iterdir())
    res['fit_files'] = files
    for f in ['generated.npz', 'generated_snapens.npz', 'generated_exshrink@0.5.npz', 'final.pt', 'snap_16.pt', 'snap_19.pt']:
        assert f in files, f
    cfg = json.loads((fit / 'config.json').read_text())
    res['config_r2_arms'] = cfg['r2_arms']
    res['config_critic_features'] = cfg['critic_features']
    assert cfg['round'] == 2 and cfg['critic_features'] == 159 and 'example@0.5' in cfg['r2_arms']
    gd, ge = npz(fit / 'generated.npz'), npz(fit / 'generated_exshrink@0.5.npz')
    st = gd['stage'].astype(np.float32) / 3
    c = ge['conditions']
    res['exshrink_slot0_values'] = sorted(set(np.round(c[..., 0] / np.maximum(st[:, None], 1e-9), 4).ravel().tolist()))[:4]
    shrunk = np.isclose(c[..., 0], 0.25 * st[:, None]) & (st[:, None] > 0)
    kept = np.isclose(c[..., 0], 0.5 * st[:, None]) & (st[:, None] > 0)
    res['exshrink_fraction_shrunk'] = float(shrunk.sum() / max((st[:, None] > 0).sum() * 32, 1))
    assert ((shrunk | kept) | (st[:, None] == 0)).all() and 0.3 < res['exshrink_fraction_shrunk'] < 0.7
    # second screen run: everything present -> skipped, exit 0
    p, _ = sh(['bash', HERE / 'run_screen.sh', '-d', 'cpu', '-p', '1', '-j', jobs, '-u', str(TINY), '-o', root, '-l', logs])
    assert '1 skipped' in p.stdout
    # example arm CLI resume (arm stream checkpointed across processes) == uninterrupted screen fit
    rr = SCR / 'r2fits_ex_resume'
    shutil.rmtree(rr, ignore_errors=True)
    p, _ = r2_train(arm, rr, '--checkpoint-every', '10', '--stop-after', '13', check=False)
    fr = fitpath(rr, arm)
    q = torch.load(fr / 'recovery.pt', map_location='cpu', weights_only=False)
    res['example_recovery_has_arm_stream'] = 'example' in q['arm_streams']
    del q
    r2_train(arm, rr, '--checkpoint-every', '10')
    a, b = final_models(fit), final_models(fr)
    gr = npz(fr / 'generated.npz')
    res['example_resume_bitwise'] = models_equal(a[0], b[0]) and a[1] == b[1] and curve(fit) == curve(fr) and \
        all(np.array_equal(gd[k], gr[k]) for k in gd)
    assert res['example_recovery_has_arm_stream'] and res['example_resume_bitwise']
    shutil.rmtree(rr, ignore_errors=True)
    res['PASS'] = True
    save('screen', res)
    log('screen', json.dumps(res, default=str)[:3000])
    return res


# ============================================================================ round-1 fit through r2 generation
def round1():
    import arm_example  # noqa: F401
    res = {'fit': str(R1_GPU_FIT)}
    cfg = json.loads((R1_GPU_FIT / 'config.json').read_text())
    assert cfg['device'] == 'cuda' and (R1_GPU_FIT / 'generated.npz').exists()
    out = SCR / 'round1' / 'base_hvg2000_fold0_seed0.npz'
    shutil.rmtree(out.parent, ignore_errors=True)
    before = {p.name: p.stat().st_mtime_ns for p in R1_GPU_FIT.iterdir()}
    # refusals: no --out, --out inside the round-1 fit tree
    for extra in ([], ['--out', R1_GPU_FIT / 'x.npz'], ['--out', RUNS / 'fits' / 'x.npz']):
        p, _ = sh([PY, '-B', HERE / 'generate.py', '--fit', R1_GPU_FIT, '--cache', CACHE0, '--threads', '3', *extra], check=False)
        assert p.returncode != 0 and ('round-1' in p.stderr or 'exists' in p.stderr), p.stderr[-500:]
    _, t = sh([PY, '-B', HERE / 'generate.py', '--fit', R1_GPU_FIT, '--cache', CACHE0, '--threads', '3', '--out', out])
    res['r2_cpu_generation_seconds'] = t
    cpu = npz(out)
    gpu = npz(R1_GPU_FIT / 'generated.npz')
    # (1) code equivalence: round-1 generate_arrays on CPU (G/P from final.pt state dicts) on 4 targets
    ref = SCR / 'round1' / 'r1_cpu_subset.npz'
    r1_python(f"""
        import numpy as np, trainer, generate, json
        cfg = json.load(open({str(R1_GPU_FIT / 'config.json')!r}))
        ctx = trainer.Context({str(CACHE0)!r}, 'cpu', tuple(cfg['flags']))
        b, *_ = trainer.build(ctx, cfg['seed'], stream_root=cfg['stream_root'])
        q = torch.load({str(R1_GPU_FIT / 'final.pt')!r}, map_location='cpu', weights_only=False)
        b[1].load_state_dict(q['policy']['models']['generator']); b[3].load_state_dict(q['policy']['models']['prior'])
        orig = generate.targets
        generate.targets = lambda c: [orig(c)[i] for i in (0, 1, -2, -1)]
        arr, _ = generate.generate_arrays(ctx, b[1], b[3])
        np.savez({str(ref)!r}, **arr)
    """)
    r1c = npz(ref)
    n = len(cpu['targets'])
    rows = [0, 1, n - 2, n - 1]
    res['r2_vs_r1_cpu_subset_bitwise'] = {k: bool(np.array_equal(r1c[k], cpu[k] if k == 'genes' else cpu[k][rows])) for k in r1c}
    assert all(res['r2_vs_r1_cpu_subset_bitwise'].values())
    # (2) vs stored GPU arrays
    res['metadata_equal'] = {k: bool(np.array_equal(cpu[k], gpu[k])) for k in ('targets', 'is_heldout', 'stage', 'genes')}
    res['bitwise_equal_numeric'] = {k: bool(np.array_equal(cpu[k], gpu[k])) for k in ('pb_counts', 'mean_log', 'nonzero', 'lib')}

    def zstats(a, b):
        a = a.reshape(-1, a.shape[-1]).astype(np.float64)
        b = b.reshape(-1, b.shape[-1]).astype(np.float64)
        se = np.sqrt(a.var(0, ddof=1) / len(a) + b.var(0, ddof=1) / len(b))
        ok = se > 0
        z = (a.mean(0) - b.mean(0))[ok] / se[ok]
        return {'mean_z2': float((z ** 2).mean()), 'frac_abs_z_gt4': float((np.abs(z) > 4).mean()),
                'max_abs_diff': float(np.abs(a.mean(0) - b.mean(0)).max()), 'genes': int(ok.sum())}
    M_ = cpu['lib'].shape[1]
    logpb = lambda x: np.log1p(x / np.maximum(x.sum(-1, keepdims=True), 1) * 1e4)  # noqa: E731
    res['z'] = {'mean_log': zstats(cpu['mean_log'], gpu['mean_log']), 'nonzero': zstats(cpu['nonzero'], gpu['nonzero']),
                'log_pseudobulk': zstats(logpb(cpu['pb_counts']), logpb(gpu['pb_counts']))}
    rel = np.abs(cpu['lib'].mean(1) - gpu['lib'].mean(1)) / gpu['lib'].mean(1)
    res['lib_per_target_max_rel_diff'] = float(rel.max())
    zl = (cpu['lib'].mean(1) - gpu['lib'].mean(1)) / np.sqrt(cpu['lib'].var(1, ddof=1) / M_ + gpu['lib'].var(1, ddof=1) / M_)
    res['lib_per_target_max_abs_z'] = float(np.abs(zl).max())
    assert all(res['metadata_equal'].values())
    for k, v in res['z'].items():
        assert 0.5 <= v['mean_z2'] <= 2.0 and v['frac_abs_z_gt4'] <= 0.01, (k, v)
    assert res['lib_per_target_max_abs_z'] < 4 and res['lib_per_target_max_rel_diff'] < 0.05
    # a plug-in mode on the round-1 fit, written outside the round-1 tree
    mout = SCR / 'round1' / 'base_exshrink.npz'
    sh([PY, '-B', HERE / 'generate.py', '--fit', R1_GPU_FIT, '--cache', CACHE0, '--threads', '3', '--mode', 'exshrink@0.5', '--out', mout])
    mo = npz(mout)
    res['round1_mode'] = {'keys': sorted(mo), 'state_names': mo['state_names'].tolist(),
                          'receipt': (mout.parent / 'base_exshrink.receipt.json').exists()}
    assert mo['state_names'].tolist() == ['final'] and res['round1_mode']['receipt']
    p, _ = sh([PY, '-B', HERE / 'generate.py', '--fit', R1_GPU_FIT, '--cache', CACHE0, '--threads', '3', '--mode', 'snapens', '--out', SCR / 'round1' / 'x.npz'], check=False)
    res['round1_snapens_refused'] = p.returncode != 0 and 'snapens needs four snapshots' in p.stderr
    assert res['round1_snapens_refused']
    after = {p.name: p.stat().st_mtime_ns for p in R1_GPU_FIT.iterdir()}
    res['round1_fit_folder_untouched'] = before == after
    assert res['round1_fit_folder_untouched']
    for f in (out, ref, mout):
        f.unlink()
    res['PASS'] = True
    save('round1', res)
    log('round1', json.dumps(res, default=str)[:3000])
    return res


def cleanup():
    for d in ('r1fits', 'r2fits', 'r2fits_resume', 'r2fits_ex_resume', 'round1', 'logs'):
        shutil.rmtree(SCR / d, ignore_errors=True)
    for f in ('jobs.txt', 'jobs_bad.txt', 'parity_r1.pt'):
        (SCR / f).unlink(missing_ok=True)
    log('cleanup done; kept', sorted(p.name for p in RES.glob('*.json')) if RES.exists() else [])


if __name__ == '__main__':
    what = sys.argv[1] if len(sys.argv) > 1 else 'all'
    steps = {'parity': parity, 'example': example, 'cli': cli, 'screen': screen, 'round1': round1, 'cleanup': cleanup}
    if what == 'all':
        for k in ['parity', 'example', 'cli', 'screen', 'round1']:
            steps[k]()
    else:
        steps[what]()
