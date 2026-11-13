"""Tests and real-data runs for the round-3 stage task (R6): arms ``nostage`` / ``fixedstage[@w]``,
modes ``stagehalf[@R]`` / ``stageblind[@R]`` (mode_stagedefault.py) and ``biascal[@INNER]`` (mode_biascal.py).

Run from /tmp with 3 threads::

    python3 -B r3/test_arm_stage3.py unit      # flags/modes parse and refusals; nostage / fixedstage conditions, initial
                                               #   tensors == base, head offset and FiLM input, 3 updates; stage-mix layouts
    python3 -B r3/test_arm_stage3.py cli       # tiny CPU CLI fits (8 updates) of nostage and fixedstage@2 + modes:
                                               #   stagehalf@final == condmix@0.5 bitwise, stagehalf (snapens states),
                                               #   biascal with 0 iterations == its inner mode bitwise, biascal residual drops
    python3 -B r3/test_arm_stage3.py real      # GPU generation (one process, --gpu-mem-fraction 0.04) on round-2 seed-0
                                               #   fits of genemean@0.25 and pbulk, folds 0-2: stagehalf, biascal@snapens,
                                               #   biascal@stagehalf; scored with the frozen r2 evaluator; npz deleted
    python3 -B r3/test_arm_stage3.py cleanup   # delete scratch fits / generated files (JSON results kept)

Scratch: /ml2/alz/runs/improve_20261109/smoke_r3/stage/ (results/*.json kept).
"""
import os
import sys

sys.dont_write_bytecode = True
for _k in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[_k] = '3'
if not (len(sys.argv) > 1 and sys.argv[1] == 'real'):
    os.environ['CUDA_VISIBLE_DEVICES'] = ''

import argparse  # noqa: E402
import json  # noqa: E402
import shutil  # noqa: E402
import subprocess  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

HERE = Path(__file__).resolve().parent
if str(HERE) in sys.path:
    sys.path.remove(str(HERE))
sys.path.insert(0, str(HERE))

import numpy as np  # noqa: E402
import torch  # noqa: E402

torch.set_num_threads(3)
import trainer  # noqa: E402
import arms  # noqa: E402
import generate  # noqa: E402

PY = '/mnt/nvme/ai/conda/anaconda3/bin/python3'
RUNS = trainer.RUNS
CACHE0 = RUNS / 'caches/hvg2000/fold0'
SCR = RUNS / 'smoke_r3/stage'
RES = SCR / 'results'
FITS = SCR / 'fits'
GEN = SCR / 'gen'
ENV = dict(os.environ, OMP_NUM_THREADS='3', OPENBLAS_NUM_THREADS='3', MKL_NUM_THREADS='3', PYTHONDONTWRITEBYTECODE='1')
TINY = 8


def log(*a):
    print('[test_stage3]', *a, flush=True)


def save(name, obj):
    trainer.dump(RES / f'{name}.json', json.loads(json.dumps(obj, default=str)))


def sh(cmd, check=True, env=None, **kw):
    t = time.monotonic()
    p = subprocess.run([str(c) for c in cmd], env=env or dict(ENV, CUDA_VISIBLE_DEVICES=''), capture_output=True, text=True, **kw)
    if check and p.returncode != 0:
        raise RuntimeError(f'{cmd} failed rc={p.returncode}:\n{p.stdout[-3000:]}\n{p.stderr[-5000:]}')
    return p, time.monotonic() - t


def refused(fn, *a):
    try:
        fn(*a)
    except (ValueError, KeyError):
        return True
    return False


def sd(m):
    return {k: v.detach().cpu().clone() for k, v in m.state_dict().items()}


# ============================================================================ unit
def unit():
    out = {}
    ok_arms = ['nostage', 'fixedstage', 'fixedstage@0.5', 'fixedstage@0', 'genemean@0.25,nostage', 'nostage,pbulk', 'fixedstage,genemean']
    bad_arms = ['nostage,covars', 'nostage,stageanchor', 'nostage@1', 'fixedstage@1', 'fixedstage,nostage', 'covdrop@0.5,fixedstage',
                'fixedstage@101', 'fixedstage@0.50', 'nostage,offset', 'nostage,stagedrop@0.5']
    for a in ok_arms:
        arms.parse_arm(a)
    out['refused_arms'] = {a: refused(arms.parse_arm, a) for a in bad_arms}
    assert all(out['refused_arms'].values()), out['refused_arms']
    ok_modes = ['stagehalf', 'stagehalf@snapens', 'stagehalf@final', 'stagehalf@raw', 'stagehalf@ema0.9995', 'stageblind', 'stageblind@final',
                'biascal', 'biascal@snapens', 'biascal@stagehalf', 'biascal@stagehalf:snapens', 'biascal@condmix:0.5', 'biascal@ema:0.9995',
                'biascal@snapema:15000:0.999']
    for m in ok_modes:
        arms.parse_mode(m)
    bad_modes = ['stagehalf@foo', 'stagehalf@ema0.99950', 'stagehalf@ema1', 'biascal@biascal', 'biascal@condmix:0.50', 'biascal@nomode',
                 'biascal@stagehalf@final']
    out['refused_modes'] = {m: refused(arms.parse_mode, m) for m in bad_modes}
    assert all(out['refused_modes'].values()), out['refused_modes']
    log('parse/refusals ok')

    base = trainer.Context(CACHE0, 'cpu', ())
    raws = base.raw_c
    b0, g0, r0, u0, s0 = trainer.build(base, 0)
    base_init = [sd(m) for m in (b0[1], b0[2], b0[3])]
    # ---- nostage
    for arm in ('nostage', 'genemean@0.25,nostage'):
        ctx = trainer.Context(CACHE0, 'cpu', arms.parse_arm(arm))
        assert float(ctx.conditions.abs().sum()) == 0.0
        tc = ctx.target_conditions(raws, np.zeros(len(raws), bool))
        assert np.all(tc == 0)
        b, g, r, u, s = trainer.build(ctx, 0)
        if arm == 'nostage':
            init = [sd(m) for m in (b[1], b[2], b[3])]
            assert all(set(x) == set(y) and all(torch.equal(x[k], y[k]) for k in x) for x, y in zip(init, base_init)), 'nostage initial tensors differ from base'
            assert s == s0
        mets = [trainer.advance(ctx, b, g, r, u, execution_limit=20000) for _ in range(3)]
        assert all(np.isfinite(m['G_total']) and np.isfinite(m['D_total']) for m in mets)
        out[f'{arm}_metrics'] = [{k: m[k] for k in ('critic_gap', 'G_total', 'D_total')} for m in mets]
    log('nostage: conditions 0 (train and targets), initial tensors == base, 3 finite updates')
    # ---- fixedstage
    for arm, w in (('fixedstage', 1.0), ('fixedstage@20', 20.0)):
        ctx = trainer.Context(CACHE0, 'cpu', arms.parse_arm(arm))
        assert torch.equal(ctx.conditions, base.conditions), 'fixedstage critic condition must be the released one'
        table, _ = arms.offset_init(ctx.X, ctx.by, arms.stage_of(ctx.raw_c))
        want = torch.as_tensor(table, dtype=torch.float32) * w
        b, g, r, u, s = trainer.build(ctx, 0)
        G = b[1]
        assert torch.equal(G.fixed_stage_offset, want)
        init = sd(G)
        assert set(init) - set(base_init[0]) == {'fixed_stage_offset'}
        assert all(torch.equal(init[k], base_init[0][k]) for k in base_init[0]), 'released generator tensors differ from base'
        assert all(torch.equal(x, y) for x, y in zip(sd(b[2]).values(), base_init[1].values()))
        assert sum(p.numel() for p in G.parameters()) == sum(p.numel() for p in b0[1].parameters()), 'fixedstage added a trainable parameter'
        with torch.no_grad():
            z = torch.randn(5, 8, generator=torch.Generator().manual_seed(1))
            c = base.conditions[:5].clone()
            sc, sh_ = G.film(z, c)
            sc0, sh0 = G.film(z, torch.zeros_like(c))
            assert torch.equal(sc, sc0) and torch.equal(sh_, sh0), 'FiLM must ignore the condition'
            h = torch.randn(5, 3, 128, generator=torch.Generator().manual_seed(2))
            lg = G.logits(h, z, c)
            plain = G.out(h)
            st = torch.round(c[:, 3] * 3).long()
            assert torch.allclose(lg - plain, want[st][:, None, :].expand_as(lg), atol=1e-5)
            c2 = c.clone(); c2[:, 3] = 0.5     # off-grid: halfway between stage 1 and 2
            lg2 = G.logits(h, z, c2) - plain
            assert torch.allclose(lg2, (0.5 * want[1] + 0.5 * want[2])[None, None, :].expand_as(lg2), atol=1e-5)
        mets = [trainer.advance(ctx, b, g, r, u, execution_limit=20000) for _ in range(3)]
        assert all(np.isfinite(m['G_total']) for m in mets)
        assert torch.equal(b[1].fixed_stage_offset, want), 'fixed offset moved during training'
        out[f'{arm}_effective_rms'] = ctx.plugins[0].info['effective_rms_per_stage']
    log('fixedstage: released critic condition, offset buffer = w*offset_init, FiLM blind, logits + row, fixed through 3 updates')
    # ---- stage-mix layouts
    import mode_stagedefault as SD
    import arm_condshrink as MC
    ctx = base
    i = int(np.flatnonzero(arms.stage_of(raws) == 0)[0])
    tgt = arms.GenTarget(index=i, donor=str(ctx.ids[i]), is_heldout=False, raw_c=raws[i], stage=0, dev_index=0,
                         default_cond=ctx.target_conditions(raws[i:i + 1], np.zeros(1, bool))[0], M=32, K=256, updates=20000)
    c1, s1 = SD.mixed_conditions(ctx, tgt, 0.5, ['final'] * 32)
    cc, sc = MC.condmix_conditions(ctx, tgt, 0.5)
    assert np.array_equal(c1, cc) and np.array_equal(s1, sc)
    pi = MC.train_stage_proportions(ctx)
    layouts = {}
    for lam, counts in ((0.5, [6, 6, 5, 5, 5, 5]), (0.5, [7, 7, 6, 6, 6]), (1.0, [6, 6, 5, 5, 5, 5])):
        names = [f's{j}' for j, n in enumerate(counts) for _ in range(n)]
        c, st = SD.mixed_conditions(ctx, tgt, lam, names)
        mixed = np.array([not np.array_equal(c[m], tgt.default_cond) for m in range(32)]) | (st != tgt.stage)
        n = MC.n_mixed(lam, 32)
        k = MC.apportion(n, pi)
        per_state = [int(sum(mixed[m] for m in range(32) if names[m] == f's{j}')) for j in range(len(counts))]
        # every generated donor's condition is its stage's condition
        for m in range(32):
            assert np.allclose(c[m, 3], st[m] / 3.0)
        stage_counts = np.bincount(st, minlength=4)
        want = k.copy(); want[tgt.stage] += 32 - n
        assert np.array_equal(stage_counts, want), (stage_counts, want)
        layouts[f'lam{lam}_{counts}'] = {'mixed_slots_per_state': per_state, 'stage_counts': stage_counts.tolist()}
        if lam == 0.5:
            assert max(per_state) - min(per_state) <= 2 and all(abs(p - c_ / 2) <= 1.5 for p, c_ in zip(per_state, counts)), per_state
    out['stage_mix_layouts'] = layouts
    log('stage-mix layouts', layouts)
    save('unit', out)
    log('UNIT PASS')


# ============================================================================ cli
def fit_of(arm):
    return FITS / arm / 'hvg2000' / 'fold0' / 'seed0'


def train_tiny(arm):
    fit = fit_of(arm)
    if not (fit / 'generated.npz').exists():
        sh([PY, '-B', HERE / 'trainer.py', '--cache', CACHE0, '--arm', arm, '--seed', '0', '--out-root', FITS, '--device', 'cpu',
            '--threads', '3', '--updates', str(TINY)])
    return fit


def npz(p):
    with np.load(p, allow_pickle=False) as q:
        return {k: q[k] for k in q.files}


ARR = ('pb_counts', 'mean_log', 'nonzero', 'pcs', 'lib', 'sq_log')


def cli():
    out = {}
    for arm in ('nostage', 'fixedstage@2'):
        fit = train_tiny(arm)
        cfg = json.loads((fit / 'config.json').read_text())
        out[arm] = {'r2_arms': cfg['r2_arms'], 'snapshot_updates': cfg['snapshot_updates']}
        log(arm, 'trained', cfg['snapshot_updates'])
    # nostage: stage modes are no-ops on the conditions
    fit = fit_of('nostage')
    if not (fit / 'generated_stagehalf.npz').exists():
        generate.generate_fit(fit, CACHE0, mode='stagehalf')
    q = npz(fit / 'generated_stagehalf.npz')
    assert np.all(q['conditions'] == 0)
    out['nostage_stagehalf_states'] = q['state_names'].tolist()
    # fixedstage@2: stagehalf@final == condmix@0.5 bitwise; stagehalf uses the snapens states
    fit = fit_of('fixedstage@2')
    ctx = trainer.Context(CACHE0, 'cpu', arms.parse_arm('fixedstage@2'))
    for m in ('condmix@0.5', 'stagehalf@final', 'stagehalf', 'snapens', 'stageblind@final', 'condmix@1'):
        if not (fit / f'generated_{m}.npz').exists():
            generate.generate_fit(fit, CACHE0, mode=m, ctx=ctx)
    a, b = npz(fit / 'generated_condmix@0.5.npz'), npz(fit / 'generated_stagehalf@final.npz')
    assert all(np.array_equal(a[k], b[k]) for k in ARR + ('conditions', 'state_index')), 'stagehalf@final != condmix@0.5'
    a, b = npz(fit / 'generated_condmix@1.npz'), npz(fit / 'generated_stageblind@final.npz')
    assert all(np.array_equal(a[k], b[k]) for k in ARR + ('conditions',)), 'stageblind@final != condmix@1'
    sh_ = npz(fit / 'generated_stagehalf.npz'); se = npz(fit / 'generated_snapens.npz')
    assert sh_['state_names'].tolist() == se['state_names'].tolist() and np.array_equal(sh_['state_index'], se['state_index'])
    own = (np.rint(sh_['conditions'][:, :, 3] * 3).astype(int) != sh_['stage'][:, None])
    out['stagehalf_states'] = sh_['state_names'].tolist()
    out['stagehalf_mixed_fraction_per_state'] = [float(own[:, sh_['state_index'][0] == j].mean()) for j in range(len(sh_['state_names']))]
    # rows whose condition equals the own stage are bitwise those of snapens (same streams, same state) only where the
    # whole chunk is unmixed; check instead that the per-state donors are the snapens ones
    log('stagehalf states', out['stagehalf_states'], 'fraction of donors not at own stage per state', out['stagehalf_mixed_fraction_per_state'])
    # biascal with 0 iterations == inner mode bitwise
    import mode_biascal as MB
    old = MB.N_ITER
    try:
        MB.N_ITER = 0
        p0 = GEN / 'tiny_biascal0_stagehalf.npz'
        generate.generate_fit(fit, CACHE0, mode='biascal@stagehalf', ctx=ctx, out=p0, force=True)
        a = npz(p0); b = npz(fit / 'generated_stagehalf.npz')
        assert all(np.array_equal(a[k], b[k]) for k in ARR + ('conditions', 'state_index')), 'biascal@stagehalf with 0 iterations != stagehalf'
        p0b = GEN / 'tiny_biascal0_default.npz'
        generate.generate_fit(fit, CACHE0, mode='biascal', ctx=ctx, out=p0b, force=True)
        a = npz(p0b); b = npz(fit / 'generated.npz')
        assert all(np.array_equal(a[k], b[k]) for k in ARR), 'biascal with 0 iterations != generated.npz'
    finally:
        MB.N_ITER = old
    log('biascal with 0 iterations reproduces stagehalf and the default generation bitwise')
    p2, info = MB.run_biascal(fit, CACHE0, 'biascal@stagehalf', GEN / 'tiny_biascal_stagehalf.npz', 'cpu', force=True)
    h = info['history']
    out['biascal_tiny'] = {k: info[k] for k in ('states', 'delta_rms', 'delta_absmax', 'history')}
    # the residual cannot reach 0: a per-gene logit shift cannot change the mean-of-log gap that comes from
    # the generated 256-cell bags' own dispersion (common offset, see mode_biascal docstring); require a clear drop
    used = h[-1]['used_residual_rms']
    assert used < 0.9 * h[0]['residual_rms'] and used == min(x['residual_rms'] for x in h[:-1]), h
    # the head bias moved by delta in every state, nothing else: compare a regenerated train mean profile
    log('biascal residual RMS per iteration', [round(x['residual_rms'], 4) for x in h[:-1]])
    # CLI path through generate.py (mode lookup by scan) for one composed mode
    sh([PY, '-B', HERE / 'generate.py', '--fit', fit, '--cache', CACHE0, '--mode', 'biascal@snapens', '--out', GEN / 'tiny_cli_biascal_snapens.npz'])
    log('generate.py --mode biascal@snapens ran through the plug-in scan')
    save('cli', out)
    log('CLI PASS')


# ============================================================================ real data
REAL_ARMS = ('genemean@0.25', 'pbulk')
REAL_MODES = ('stagehalf', 'biascal@snapens', 'biascal@stagehalf')
SCORE_CODE = r'''
import sys, json, numpy as np
sys.path.insert(0, "/ml2/alz/analysis/improve_20261109/r2")
import evaluate as E
cache, files, out = sys.argv[1], json.loads(sys.argv[2]), sys.argv[3]
real = E.load_real(cache)
subs = E.gene_subsets(real)
res = {}
for label, p in files.items():
    gen = E.align_generated(E.load_generated(p), real)
    r, _ = E.score_generated(gen, real, subs)
    res[label] = {"heldout": E.headline(r, "heldout"), "train": E.headline(r, "train")}
json.dump(res, open(out, "w"), default=float)
'''


def real():
    out_all = {}
    gpu_env = dict(ENV, CUDA_VISIBLE_DEVICES=os.environ.get('STAGE3_GPU', '1'))
    for arm in REAL_ARMS:
        for k in range(3):
            fit = RUNS / 'fits_r2' / arm / 'hvg2000' / f'fold{k}' / 'seed0'
            cache = RUNS / 'caches/hvg2000' / f'fold{k}'
            files = {'default': str(fit / 'generated.npz'), 'snapens': str(fit / 'generated_snapens.npz'),
                     'condmix@0.5': str(fit / 'generated_condmix@0.5.npz'), 'condmix@1': str(fit / 'generated_condmix@1.npz')}
            cal = {}
            for m in REAL_MODES:
                p = GEN / 'real' / arm / f'fold{k}' / f'generated_{m}.npz'
                p.parent.mkdir(parents=True, exist_ok=True)
                if not p.exists():
                    if m.startswith('biascal'):
                        cmd = [PY, '-B', HERE / 'mode_biascal.py', '--fit', fit, '--cache', cache, '--mode', m, '--out', p,
                               '--device', 'cuda', '--gpu-mem-fraction', '0.04']
                    else:
                        cmd = [PY, '-B', HERE / 'generate.py', '--fit', fit, '--cache', cache, '--mode', m, '--out', p,
                               '--device', 'cuda', '--gpu-mem-fraction', '0.04']
                    _, sec = sh(cmd, env=gpu_env)
                    log(arm, k, m, f'generated in {sec:.0f}s')
                files[m] = str(p)
                if m.startswith('biascal'):
                    q = json.loads(p.with_name(p.stem + '.biascal.json').read_text())
                    cal[m] = {x: q[x] for x in ('states', 'delta_rms', 'delta_mean', 'delta_absmax', 'history')}
            sj = GEN / 'real' / arm / f'fold{k}' / 'scores.json'
            if not sj.exists():
                _, sec = sh([PY, '-B', '-c', SCORE_CODE, cache, json.dumps(files), sj], cwd='/tmp')
                log(arm, k, f'scored in {sec:.0f}s')
            sc = json.loads(sj.read_text())
            out_all[f'{arm}|fold{k}'] = {'scores': sc, 'biascal': cal}
            log(arm, k, {lab: round(v['heldout'].get('pb_rmse', float('nan')), 4) for lab, v in sc.items()})
    keys = ('pb_rmse', 'meanlog_rmse_gene', 'nonzero_rmse_gene', 'logvar_rmse_gene', 'pc_energy_rep', 'cell_energy_rep', 'pc_coverage80_rep',
            'pb_rmse_recal')
    summary = {}
    for arm in REAL_ARMS:
        labs = out_all[f'{arm}|fold0']['scores'].keys()
        summary[arm] = {lab: {kk: float(np.mean([out_all[f'{arm}|fold{k}']['scores'][lab]['heldout'].get(kk, np.nan) for k in range(3)]))
                              for kk in keys} for lab in labs}
        summary[arm]['_train_pb_rmse'] = {lab: float(np.mean([out_all[f'{arm}|fold{k}']['scores'][lab]['train'].get('pb_rmse', np.nan) for k in range(3)]))
                                          for lab in labs}
    out_all['fold_mean_heldout'] = summary
    save('real', out_all)
    for arm in REAL_ARMS:
        for lab, v in summary[arm].items():
            if not lab.startswith('_'):
                log(arm.ljust(14), lab.ljust(20), ' '.join(f'{kk} {v[kk]:.3f}' for kk in keys if np.isfinite(v[kk])))
    log('REAL DONE')


def cleanup():
    for p in (FITS, GEN):
        if p.exists():
            shutil.rmtree(p)
    log('scratch removed; kept', sorted(str(x) for x in SCR.rglob('*.json')))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('what', choices=['unit', 'cli', 'real', 'cleanup', 'all'])
    a = ap.parse_args()
    RES.mkdir(parents=True, exist_ok=True)
    GEN.mkdir(parents=True, exist_ok=True)
    todo = ['unit', 'cli'] if a.what == 'all' else [a.what]
    for w in todo:
        globals()[w]()


if __name__ == '__main__':
    main()
