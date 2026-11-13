"""Tests for the stage-path fixes: arm_stageanchor.py, arm_stagedrop.py, arm_condshrink.py.

CPU only, 3 threads, real fold-0 hvg2000 cache (read-only).  Run::

    python3 -B r2/test_arm_stage.py unit     # parsing, hooks, init parity with base, 3 in-process updates, mode plans
    python3 -B r2/test_arm_stage.py cli      # tiny CLI fits (20 updates) of stageanchor and stagedrop@0.5 + their modes,
                                             # stagedrop resume bitwise, condmix@0 == generated.npz bitwise
    python3 -B r2/test_arm_stage.py round1   # condmix/condshrink through r2/generate.py on a round-1 GPU fit (--out),
                                             # equal to mode_condshrink.py gen output; condshrink refused on an offset fit
    python3 -B r2/test_arm_stage.py cleanup  # delete the scratch tree except results/*.json

Scratch: /ml2/alz/runs/improve_20261109/smoke_r2/stage_test/.
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
import torch

torch.set_num_threads(3)
import trainer  # noqa: E402
import arms  # noqa: E402
import generate  # noqa: E402
import arm_condshrink as AC  # noqa: E402
import arm_stageanchor as SA  # noqa: E402
import arm_stagedrop as SD  # noqa: E402

PY = '/mnt/nvme/ai/conda/anaconda3/bin/python3'
RUNS = trainer.RUNS
CACHE0 = RUNS / 'caches/hvg2000/fold0'
SCR = RUNS / 'smoke_r2/stage_test'
RES = SCR / 'results'
TINY = 20
R1_BASE = RUNS / 'fits/base/hvg2000/fold0/seed0'
R1_OFFSET = RUNS / 'fits/offset/hvg2000/fold0/seed0'
GEN_STAGE = RUNS / 'posthoc_r2/stage/gen/base_hvg2000_fold0_seed0.npz'
ENV = dict(os.environ, CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='3', OPENBLAS_NUM_THREADS='3', MKL_NUM_THREADS='3', PYTHONDONTWRITEBYTECODE='1')


def log(*a):
    print('[test_arm_stage]', *a, flush=True)


def save(name, obj):
    trainer.dump(RES / f'{name}.json', obj)


def sh(cmd, check=True):
    t = time.monotonic()
    p = subprocess.run([str(c) for c in cmd], env=ENV, capture_output=True, text=True, cwd='/tmp')
    if check and p.returncode != 0:
        raise RuntimeError(f'{cmd} failed rc={p.returncode}:\n{p.stdout[-3000:]}\n{p.stderr[-5000:]}')
    return p, time.monotonic() - t


def states(*mods):
    return [{k: v.detach().cpu().clone() for k, v in m.state_dict().items()} for m in mods]


def sd_equal(a, b):
    return set(a) == set(b) and all(torch.equal(a[k], b[k]) for k in a)


def refused(fn, *a):
    try:
        fn(*a)
        return False
    except ValueError:
        return True


def target(ctx, i, cond):
    st = int(arms.stage_of(ctx.raw_c[i:i + 1])[0])
    return arms.GenTarget(index=i, donor=str(ctx.ids[i]), is_heldout=False, raw_c=ctx.raw_c[i].copy(), stage=st, dev_index=0,
                          default_cond=cond[i].copy(), M=32, K=256)


# ============================================================================ unit
def unit():
    res = {}
    bad = ['stagedrop', 'stagedrop@0.50', 'stagedrop@1', 'stagedrop@0', 'stageanchor@1', 'offset,stageanchor', 'offset,stagedrop@0.5',
           'stageanchor,stagedrop@0.5']
    res['refused_flags'] = {b: refused(arms.parse_arm, b) for b in bad}
    assert all(res['refused_flags'].values()), res['refused_flags']
    res['parse'] = [arms.parse_arm('stageanchor,genemean'), arms.parse_arm('stagedrop@0.5,covars')]
    assert res['parse'] == [('genemean', 'stageanchor'), ('covars', 'stagedrop@0.5')]
    for m in ['condshrink@1.5', 'condshrink', 'condmix@0.50', 'condunk', 'anchorgain@2']:
        assert refused(arms.parse_mode, m), m
    assert [arms.parse_mode(m)[1] for m in ['condshrink@0.5', 'condmix@1', 'condunk@0.25', 'anchorgain@0']] == [0.5, 1.0, 0.25, 0.0]

    # interpolation weights: one-hot on the float32 grid, linear between knots
    grid = torch.tensor([0, 1 / 3, 2 / 3, 1], dtype=torch.float32)
    assert torch.equal(SA.interp_weights(grid), torch.eye(4))
    w = SA.interp_weights(torch.tensor([0.5, 0.1], dtype=torch.float32))
    assert torch.allclose(w, torch.tensor([[0, .5, .5, 0], [.7, .3, 0, 0]]), atol=1e-6)

    # ---------------------------------------------------------------- stageanchor
    ctxb = trainer.Context(CACHE0, 'cpu', ())
    ctxa = trainer.Context(CACHE0, 'cpu', arms.parse_arm('stageanchor'))
    tab, _ = arms.offset_init(ctxb.X, ctxb.by, arms.stage_of(ctxb.raw_c))
    assert torch.equal(ctxa.plugins[0].table, torch.as_tensor(tab))
    assert torch.equal(ctxa.conditions, ctxb.conditions)
    bb, gb, rb, ub, _ = trainer.build(ctxb, 0)
    ba, ga, ra, ua, _ = trainer.build(ctxa, 0)
    Gb, Ga = states(bb[1])[0], states(ba[1])[0]
    res['anchor_released_generator_init_equal_base'] = all(torch.equal(Gb[k], Ga[k]) for k in Gb)
    res['anchor_extra_state'] = sorted(set(Ga) - set(Gb))
    res['anchor_critic_prior_init_equal_base'] = sd_equal(states(bb[2])[0], states(ba[2])[0]) and sd_equal(states(bb[3])[0], states(ba[3])[0])
    assert res['anchor_released_generator_init_equal_base'] and res['anchor_critic_prior_init_equal_base']
    assert res['anchor_extra_state'] == ['stage_anchor', 'stage_gain']
    G = ba[1]
    assert any(p is G.stage_gain for p in ba[4].param_groups[0]['params'])
    # FiLM is stage blind; logits differ between stages by exactly gain_s * A_s
    z = torch.randn(2, 8, generator=torch.Generator().manual_seed(1))
    c = torch.zeros(2, 7); c[0, 3] = float(np.float32(1 / 3)); c[1, 3] = 1.0
    c2 = c.clone(); c2[:, 3] = 0
    with torch.no_grad():
        f1, f2 = G.film(z, c), G.film(z, c2)
        h = torch.randn(2, 5, 128, generator=torch.Generator().manual_seed(2))
        d = G.logits(h, z, c) - G.logits(h, z, c2)
    res['film_stage_blind'] = bool(torch.equal(f1[0], f2[0]) and torch.equal(f1[1], f2[1]))
    res['logit_shift_is_anchor_row'] = float((d[0] - (G.stage_anchor[1] - G.stage_anchor[0])).abs().max() +
                                             (d[1] - (G.stage_anchor[3] - G.stage_anchor[0])).abs().max())
    assert res['film_stage_blind'] and res['logit_shift_is_anchor_row'] < 1e-5
    g0 = G.stage_gain.detach().clone()
    for _ in range(3):
        trainer.advance(ctxb, bb, gb, rb, ub, execution_limit=20000)
        trainer.advance(ctxa, ba, ga, ra, ua, execution_limit=20000)    # step-0 assert includes stage_gain
    res['anchor_donor_sampler_equal_base_after_3'] = rb.bit_generator.state == ra.bit_generator.state
    res['anchor_gain_after_3'] = G.stage_gain.detach().tolist()
    res['anchor_gain_step_norms'] = float((G.stage_gain.detach() - g0).norm())
    res['anchor_buffer_unchanged'] = bool(torch.equal(G.stage_anchor, torch.as_tensor(tab)))
    assert res['anchor_donor_sampler_equal_base_after_3'] and res['anchor_buffer_unchanged'] and res['anchor_gain_step_norms'] > 0
    del bb, ba, gb, ga

    # ---------------------------------------------------------------- stagedrop
    ctxd = trainer.Context(CACHE0, 'cpu', arms.parse_arm('stagedrop@0.5'))
    arm = ctxd.plugins[0]
    res['stagedrop_cbar'] = float(arm.cbar)
    assert arm.cbar == np.float32(ctxd.raw_c[:, 3].mean()) and torch.equal(ctxd.conditions, ctxb.conditions)
    bd, gd, rd, ud, _ = trainer.build(ctxd, 0)
    bb, gb, rb, ub, _ = trainer.build(ctxb, 0)
    res['drop_init_equal_base'] = all(sd_equal(x, y) for x, y in zip(states(bb[1], bb[2], bb[3]), states(bd[1], bd[2], bd[3])))
    s0 = ctxd.arm_rngs['stagedrop'].bit_generator.state
    for _ in range(3):
        trainer.advance(ctxb, bb, gb, rb, ub, execution_limit=20000)
        trainer.advance(ctxd, bd, gd, rd, ud, execution_limit=20000)
    res['drop_donor_sampler_equal_base_after_3'] = rb.bit_generator.state == rd.bit_generator.state
    res['drop_arm_stream_advanced'] = ctxd.arm_rngs['stagedrop'].bit_generator.state != s0
    assert res['drop_init_equal_base'] and res['drop_donor_sampler_equal_base_after_3'] and res['drop_arm_stream_advanced']
    # batch hook statistics over 200 batches
    ctxd.arm_rngs['stagedrop'] = np.random.default_rng(3)
    r_tmp = np.random.default_rng(4)
    n_unk = n_tot = 0
    for _ in range(50):
        ds = r_tmp.integers(len(ctxd.ids), size=32)
        cfull = ctxd.conditions[torch.as_tensor(ds)]
        cb_ = arm.batch_condition(ctxd, cfull, ds, ctxd.arm_rngs['stagedrop'], 'critic')
        unk = cb_[:, 2] == 1
        assert torch.equal(cb_[~unk], cfull[~unk])
        assert bool((cb_[unk, 3] == torch.tensor(float(arm.cbar))).all()) and torch.equal(cb_[unk][:, [0, 1, 4, 5, 6]], cfull[unk][:, [0, 1, 4, 5, 6]])
        n_unk += int(unk.sum()); n_tot += 32
    res['drop_fraction_observed'] = n_unk / n_tot
    assert 0.44 < res['drop_fraction_observed'] < 0.56

    # ---------------------------------------------------------------- mode plans
    cond = ctxd.target_conditions(ctxd.raw_c, np.zeros(len(ctxd.ids), bool))
    tg = target(ctxd, 0, cond)
    cu, names = SD.condunk_plan(ctxd, 0.5, tg, None, ('final',))
    assert names == ['final'] * 32 and (cu[:16] == cond[0]).all() and (cu[16:, 2] == 1).all() and (cu[16:, 3] == arm.cbar).all()
    assert refused(SD.condunk_plan, ctxb, 0.5, target(ctxb, 0, cond), None, ('final',))
    cs, _ = AC.condshrink_plan(ctxb, 1.0, tg, None, ('final',))
    assert (cs[:, 3] == AC.train_mean_stage_value(ctxb)).all() and (cs[:, [0, 1, 2, 4, 5, 6]] == cond[0][[0, 1, 2, 4, 5, 6]]).all()
    cs0, _ = AC.condshrink_plan(ctxb, 0.0, tg, None, ('final',))
    assert (cs0 == cond[0]).all()
    pi = AC.train_stage_proportions(ctxb)
    res['train_stage_proportions_fold0'] = pi.tolist()
    cm, st = AC.condmix_conditions(ctxb, tg, 0.5)
    k = AC.apportion(16, pi)
    res['condmix_0.5_mixed_stage_counts'] = k.tolist()
    assert (cm[:16] == cond[0]).all() and (st[:16] == tg.stage).all()
    assert [int((st[16:] == s).sum()) for s in range(4)] == k.tolist()
    assert np.allclose(cm[16:, 3], st[16:] / 3.0) and (cm[16:, [0, 1, 2, 4, 5, 6]] == 0).all()
    cm0, _ = AC.condmix_plan(ctxb, 0.0, tg, None, ('final',))
    assert (cm0 == cond[0]).all()
    # covars: mixed donors keep the target's own covariates
    ctxc = trainer.Context(CACHE0, 'cpu', arms.parse_arm('covars'))
    condc = ctxc.target_conditions(ctxc.raw_c, np.zeros(len(ctxc.ids), bool))
    cmc, _ = AC.condmix_conditions(ctxc, target(ctxc, 0, condc), 1.0)
    assert (cmc[:, 4:] == condc[0][4:]).all()
    res['PASS'] = True
    save('unit', res)
    log('unit', json.dumps(res, default=str)[:2000])
    return res


# ============================================================================ CLI tiny fits
def r2_train(arm, root, *extra, updates=TINY, check=True):
    return sh([PY, '-B', HERE / 'trainer.py', '--cache', CACHE0, '--arm', arm, '--seed', '0', '--out-root', root,
               '--updates', str(updates), '--device', 'cpu', '--threads', '3', *extra], check=check)


def gen_mode(fit, mode, *extra):
    return sh([PY, '-B', HERE / 'generate.py', '--fit', fit, '--cache', CACHE0, '--mode', mode, '--threads', '3', *extra])


def npz(p):
    with np.load(p, allow_pickle=False) as q:
        return {k: q[k] for k in q.files}


def fitpath(root, arm):
    return Path(root) / arm / 'hvg2000' / 'fold0' / 'seed0'


def cli():
    res = {}
    root, rroot = SCR / 'fits', SCR / 'fits_resume'
    for d in (root, rroot):
        shutil.rmtree(d, ignore_errors=True)
    # stageanchor
    _, t = r2_train('stageanchor', root)
    fa = fitpath(root, 'stageanchor')
    cfg = json.loads((fa / 'config.json').read_text())
    q = torch.load(fa / 'final.pt', map_location='cpu', weights_only=False)
    gain = q['policy']['models']['generator']['stage_gain']
    res['stageanchor_fit_seconds'] = t
    res['stageanchor_gain_after_20'] = gain.tolist()
    res['stageanchor_config_r2_arms'] = list(cfg['r2_arms'])
    assert cfg['round'] == 2 and cfg['critic_features'] == 150 and (fa / 'generated.npz').exists()
    assert len(list(fa.glob('snap_*.pt'))) == 4 and not torch.equal(gain, torch.ones(4))
    gen_mode(fa, 'anchorgain@0')
    gen_mode(fa, 'condshrink@1')
    d0 = npz(fa / 'generated.npz'); a0 = npz(fa / 'generated_anchorgain@0.npz'); c1 = npz(fa / 'generated_condshrink@1.npz')
    res['anchorgain0_mode'] = str(a0['mode'])
    # anchorgain@0 removes the stage effect: generated profiles no longer depend on stage beyond library pools
    assert a0['pb_counts'].shape == d0['pb_counts'].shape and not np.array_equal(a0['pb_counts'], d0['pb_counts'])
    assert np.allclose(c1['conditions'][..., 3], AC.train_mean_stage_value(trainer.Context(CACHE0, 'cpu', ('stageanchor',))))
    # stagedrop
    _, t = r2_train('stagedrop@0.5', root)
    fd = fitpath(root, 'stagedrop@0.5')
    res['stagedrop_fit_seconds'] = t
    q = torch.load(fd / 'final.pt', map_location='cpu', weights_only=False)
    assert set(q['arm_streams']) == {'stagedrop'}
    gen_mode(fd, 'condunk@1')
    gen_mode(fd, 'condmix@0')
    gen_mode(fd, 'condmix@0.5')
    dd = npz(fd / 'generated.npz'); u1 = npz(fd / 'generated_condunk@1.npz')
    m0 = npz(fd / 'generated_condmix@0.npz'); m5 = npz(fd / 'generated_condmix@0.5.npz')
    res['condmix0_equals_generated_bitwise'] = all(np.array_equal(dd[k], m0[k]) for k in dd)
    assert res['condmix0_equals_generated_bitwise']
    assert (u1['conditions'][..., 2] == 1).all()
    # condmix@0.5: chunks 0,1 (donors 0-15) have only default conditions -> bitwise equal to generated.npz
    res['condmix05_first16_bitwise'] = bool(np.array_equal(m5['pb_counts'][:, :16], dd['pb_counts'][:, :16]))
    res['condmix05_last16_differ'] = bool(not np.array_equal(m5['pb_counts'][:, 16:], dd['pb_counts'][:, 16:]))
    assert res['condmix05_first16_bitwise'] and res['condmix05_last16_differ']
    # resume of stagedrop (arm stream in recovery.pt) is bitwise
    p, _ = r2_train('stagedrop@0.5', rroot, '--stop-after', '13', '--checkpoint-every', '10', check=False)
    assert p.returncode != 0 and 'stop-after reached' in p.stderr
    fr = fitpath(rroot, 'stagedrop@0.5')
    assert (fr / 'recovery.pt').exists() and not (fr / 'final.pt').exists()
    rq = torch.load(fr / 'recovery.pt', map_location='cpu', weights_only=False)
    res['recovery_at'] = rq['completed_updates']
    assert set(rq['arm_streams']) == {'stagedrop'}
    del rq
    r2_train('stagedrop@0.5', rroot)
    q2 = torch.load(fr / 'final.pt', map_location='cpu', weights_only=False)
    res['stagedrop_resume_models_bitwise'] = all(sd_equal(q['policy']['models'][k], q2['policy']['models'][k]) for k in q['policy']['models'])
    res['stagedrop_resume_arm_stream_equal'] = q['arm_streams'] == q2['arm_streams']
    g2 = npz(fr / 'generated.npz')
    res['stagedrop_resume_generated_bitwise'] = all(np.array_equal(dd[k], g2[k]) for k in dd)
    assert res['stagedrop_resume_models_bitwise'] and res['stagedrop_resume_arm_stream_equal'] and res['stagedrop_resume_generated_bitwise']
    res['PASS'] = True
    save('cli', res)
    log('cli', json.dumps(res, default=str)[:2000])
    return res


# ============================================================================ round-1 fits
def round1():
    res = {}
    out = SCR / 'round1'
    shutil.rmtree(out, ignore_errors=True)
    out.mkdir(parents=True)
    mt = {p.name: p.stat().st_mtime_ns for p in R1_BASE.iterdir()}
    p = sh([PY, '-B', HERE / 'generate.py', '--fit', R1_OFFSET, '--cache', CACHE0, '--mode', 'condshrink@0.5', '--out', out / 'x.npz'], check=False)[0]
    res['condshrink_refused_on_offset_fit'] = p.returncode != 0 and 'offset' in p.stderr
    assert res['condshrink_refused_on_offset_fit']
    _, t = sh([PY, '-B', HERE / 'generate.py', '--fit', R1_BASE, '--cache', CACHE0, '--mode', 'condmix@0.5', '--out', out / 'condmix.npz'])
    res['condmix_round1_seconds'] = t
    g = npz(out / 'condmix.npz')
    rc = json.loads((out / 'condmix.receipt.json').read_text())
    assert rc['round1_fit'] and rc['mode'] == 'condmix@0.5'
    if GEN_STAGE.exists():
        import mode_condshrink as MC
        q = npz(GEN_STAGE)
        i = list(q['variants']).index('condmix@0.5')
        order = [list(g['targets']).index(d) for d in q['targets']]
        pb = g['pb_counts'][order].astype(np.float64)
        pred = np.stack([MC.log_cp10k(pb[t]).mean(0) for t in range(len(pb))]).astype(np.float32)
        res['cli_equals_mode_condshrink_gen_bitwise'] = bool(np.array_equal(pred, q['pred'][i]))
        assert res['cli_equals_mode_condshrink_gen_bitwise']
    else:
        res['cli_equals_mode_condshrink_gen_bitwise'] = 'skipped (posthoc gen file missing)'
    res['round1_fit_untouched'] = mt == {p.name: p.stat().st_mtime_ns for p in R1_BASE.iterdir()}
    assert res['round1_fit_untouched']
    shutil.rmtree(out)
    res['PASS'] = True
    save('round1', res)
    log('round1', json.dumps(res, default=str))
    return res


def cleanup():
    for p in SCR.iterdir() if SCR.exists() else []:
        if p.name != 'results':
            shutil.rmtree(p) if p.is_dir() else p.unlink()
    log('cleanup done; kept', sorted(x.name for x in RES.glob('*.json')) if RES.exists() else [])


if __name__ == '__main__':
    what = sys.argv[1] if len(sys.argv) > 1 else 'unit'
    fns = {'unit': unit, 'cli': cli, 'round1': round1, 'cleanup': cleanup}
    for name in (['unit', 'cli', 'round1'] if what == 'all' else [what]):
        t = time.monotonic()
        fns[name]()
        log(f'{name} PASS in {time.monotonic() - t:.0f}s')
