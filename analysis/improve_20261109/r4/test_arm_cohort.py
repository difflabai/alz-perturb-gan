"""Tests for r4 trainer/generate edits and the cohort arms (r4/arm_cohort.py).  Run from /tmp::

    python3 -B r4/test_arm_cohort.py parity   # r4 == round 1 (base, 5 updates) and == r3 (pbulk, 3 updates) on the
                                              #   round-1 hvg2000 fold-0 cache, CPU: initial tensors, metrics, final
                                              #   tensors, all RNG states, conditions, arm state, optimizer state
    python3 -B r4/test_arm_cohort.py unit     # hvg2000x fold 0, CPU: cohort/cohortfix batches equal the naive pooled
                                              #   batches except slot 1 = cohort; released initial tensors unchanged;
                                              #   cohortfix: SEA-AD logits bitwise unchanged, MADRC logits + offset,
                                              #   FiLM ignores slot 1, offset is a buffer equal to the recomputed one,
                                              #   receives no update; generation targets = 37 SEA-AD donors with slot 1
                                              #   = 0; asmadrc sets slot 1 = 1; refusals; generation CRN identity with
                                              #   an hvg2000 Context for the SEA-AD targets; R4-6: mu, theta,
                                              #   age mean/sd, pbulk normalisation and the generator init equal
                                              #   hvg2000 bitwise; extperm control: batches = cohortfix except
                                              #   slot 3 of external donors (permuted stage), same offset
    python3 -B r4/test_arm_cohort.py cli      # tiny CPU fits (20 updates) through r4/run_screen.sh for pbulk /
                                              #   cohort,pbulk / cohortfix,pbulk on hvg2000x fold 0 with modes; guards
    python3 -B r4/test_arm_cohort.py gpu DEV  # one 300-update GPU fit per arm (+ pbulk on hvg2000 as the speed
                                              #   reference) through run_screen.sh -p 1 -g 0.04, screen modes; s/update,
                                              #   peak CUDA memory
    python3 -B r4/test_arm_cohort.py cleanup  # delete the CPU scratch fits

Scratch: /ml2/alz/runs/improve_20261109/smoke_r4/ (fits_cpu/, fits_gpu/, logs/); results in
smoke_r4/test_arm_cohort_*.json.
"""
import os
import sys

sys.dont_write_bytecode = True
for _k in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[_k] = '3'

import json
import shutil
import subprocess
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) in sys.path:
    sys.path.remove(str(HERE))
sys.path.insert(0, str(HERE))

GPU_MODE = len(sys.argv) > 1 and sys.argv[1] == 'gpu'
if not GPU_MODE:
    os.environ['CUDA_VISIBLE_DEVICES'] = ''

import numpy as np
import torch

torch.set_num_threads(3)
import trainer  # noqa: E402  (r4)
import arms  # noqa: E402
import generate  # noqa: E402

PY = '/mnt/nvme/ai/conda/anaconda3/bin/python3'
RUNS = trainer.RUNS
SM = RUNS / 'smoke_r4'
CACHE1 = RUNS / 'caches/hvg2000/fold0'
CACHEX = RUNS / 'caches/hvg2000x/fold0'
R1 = HERE.parent
R3 = HERE.parent / 'r3'
ENV = dict(os.environ, OMP_NUM_THREADS='3', OPENBLAS_NUM_THREADS='3', MKL_NUM_THREADS='3', PYTHONDONTWRITEBYTECODE='1')
SCREEN_MODES = 'snapens,stagehalf,stageblind,ema@0.9995,condmix@1,condmix@0.5,biascal@stagehalf'


def log(*a):
    print('[test_arm_cohort]', *a, flush=True)


def save(name, obj):
    SM.mkdir(parents=True, exist_ok=True)
    trainer.dump(SM / f'test_arm_cohort_{name}.json', json.loads(json.dumps(obj, default=str)))


def sh(cmd, check=True, env=None, **kw):
    t = time.monotonic()
    p = subprocess.run([str(c) for c in cmd], env=env or dict(ENV, CUDA_VISIBLE_DEVICES=''), capture_output=True, text=True, **kw)
    if check and p.returncode != 0:
        raise RuntimeError(f'{cmd} failed rc={p.returncode}:\n{p.stdout[-3000:]}\n{p.stderr[-5000:]}')
    return p, time.monotonic() - t


# ============================================================================ parity
def parity():
    import importlib.util   # r4 copy of r3/test_core.py, loaded by path (the benchmark dir on sys.path has its own test_core)
    spec = importlib.util.spec_from_file_location('r4_test_core', HERE / 'test_core.py')
    TC = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(TC)   # helpers only; its scratch paths are not used here
    scr = SM / 'scratch'
    scr.mkdir(parents=True, exist_ok=True)
    res = {}
    for label, folder, arm_set in (('round1', R1, {'base': 5}), ('round3', R3, {'pbulk': 3})):
        out = scr / f'parity_{label}.pt'
        TC.other_round_python(folder, TC.REF_CODE.format(arms_=arm_set, cache=str(CACHE1), out=str(out)))
        ref = torch.load(out, weights_only=False)
        out.unlink()
        for arm, n in arm_set.items():
            row, _ = TC._compare(arm, n, ref[arm])
            res[f'{label}:{arm}'] = row
            log('parity', label, arm, row['PASS'])
    res['PASS'] = all(v['PASS'] for v in res.values())
    save('parity', res)
    assert res['PASS'], json.dumps(res, indent=1, default=str)
    return res


# ============================================================================ unit
def _batches(ctx, seed=0, n=2):
    b, g, r, u, s = trainer.build(ctx, seed)
    out = []
    for i in range(n):
        for phase in ('critic', 'generator'):
            out.append(ctx.batch(r, phase))
    return b, out


def unit():
    res = {}
    ctxs = {a: trainer.Context(CACHEX, 'cpu', arms.parse_arm(a)) for a in ('pbulk', 'cohort,pbulk', 'cohortfix,pbulk')}
    coh = ctxs['pbulk'].cohort
    bs = {a: _batches(c) for a, c in ctxs.items()}
    b0, ref = bs['pbulk']
    for a in ('cohort,pbulk', 'cohortfix,pbulk'):
        b1, got = bs[a]
        rows = []
        for (r0, c0, l0), (r1, c1, l1) in zip(ref, got):
            cc = c0.clone()
            rows.append(bool(torch.equal(r0, r1) and torch.equal(l0, l1) and torch.equal(c1[:, [0, 2, 3, 4, 5, 6]], c0[:, [0, 2, 3, 4, 5, 6]])
                             and (c0[:, 1] == 0).all() and set(c1[:, 1].tolist()) <= {0.0, 1.0}))
        # slot 1 equals the cohort of the batch donors: replay the donor draws of the first batch
        ctx = ctxs[a]
        _, _, r, _, _ = trainer.build(ctx, 0)
        state = r.bit_generator.state
        ds = r.integers(len(ctx.ids), size=ctx.batch_size)
        r.bit_generator.state = state
        _, c, _ = ctx.batch(r, 'critic')
        slot_ok = bool(np.array_equal(c[:, 1].numpy(), coh[ds].astype(np.float32)))
        init_same = {k: bool(torch.equal(v, b0[1].state_dict()[k])) for k, v in b1[1].state_dict().items() if k in b0[1].state_dict()}
        res[a] = {'batches_equal_except_slot1': rows, 'slot1_equals_batch_cohort': slot_ok,
                  'released_generator_init_bitwise': all(init_same.values()),
                  'critic_init_bitwise': trainer._same(b1[2].state_dict(), b0[2].state_dict()),
                  'prior_init_bitwise': trainer._same(b1[3].state_dict(), b0[3].state_dict()),
                  'extra_generator_keys': sorted(set(b1[1].state_dict()) - set(b0[1].state_dict())),
                  'batch_share_external': float(np.mean([float(x[1][:, 1].mean()) for x in got]))}
        res[a]['PASS'] = bool(all(rows) and slot_ok and res[a]['released_generator_init_bitwise'] and res[a]['critic_init_bitwise']
                              and res[a]['prior_init_bitwise'])
        log('unit', a, res[a]['PASS'])
    # cohortfix generator: logits, FiLM, buffer, no update
    ctx = ctxs['cohortfix,pbulk']
    arm = ctx.plugins[0]
    (bfix, _), G = bs['cohortfix,pbulk'], bs['cohortfix,pbulk'][0][1]
    off_re, _ = __import__('arm_cohort').cohort_offset(ctx.X, ctx.by, arms.stage_of(ctx.raw_c), ctx.cohort)
    z = torch.randn(4, 8)
    c0 = ctx.conditions[:4].clone()
    c1 = c0.clone(); c1[:, 1] = 1
    h = torch.randn(4, 3, 128)
    with torch.no_grad():
        Lbase = G.out(h)
        L0, L1 = G.logits(h, z, c0), G.logits(h, z, c1)
        f0, f1 = G.film(z, c0), G.film(z, c1)
    names = {n for n, _ in G.named_parameters()}
    fx = {'sea_logits_bitwise_unchanged': bool(torch.equal(L0, Lbase)),
          'madrc_logits_plus_offset': bool(torch.allclose(L1 - Lbase, G.cohort_offset.expand_as(L1), atol=1e-5)),
          'film_ignores_slot1': bool(torch.equal(f0[0], f1[0]) and torch.equal(f0[1], f1[1])),
          'offset_is_buffer_not_parameter': 'cohort_offset' in dict(G.named_buffers()) and 'cohort_offset' not in names,
          'offset_equals_recomputed': bool(np.array_equal(off_re, G.cohort_offset.numpy())),
          'offset_rms': float(G.cohort_offset.pow(2).mean().sqrt())}
    b, g, r, u, s = trainer.build(ctx, 0)
    before = b[1].cohort_offset.clone()
    for _ in range(2):
        trainer.advance(ctx, b, g, r, u, execution_limit=20000)
    fx['offset_unchanged_after_updates'] = bool(torch.equal(before, b[1].cohort_offset))
    fx['shadow_ema_offset_equal'] = all(torch.equal(m.cohort_offset, before) for m, _ in trainer.shadow_of(b).models.values())
    fx['PASS'] = all(v for k, v in fx.items() if isinstance(v, bool))
    res['cohortfix_generator'] = fx
    log('unit cohortfix generator', fx['PASS'])
    # generation targets / conditions / CRN identity with hvg2000
    ctx1 = trainer.Context(CACHE1, 'cpu', arms.parse_arm('pbulk'))
    # trainer R4-6 (review F1/D6): train-set statistics over the SEA-AD targets equal the hvg2000 ones bitwise
    cx = ctxs['pbulk']
    pa1, pax = ctx1.plugins[0], cx.plugins[0]
    b1g = trainer.build(ctx1, 0)[0]
    r46 = {'mu_bitwise_hvg2000': bool(torch.equal(cx.mu, ctx1.mu)),
           'theta_bitwise_hvg2000': bool(torch.equal(cx.theta, ctx1.theta)),
           'age_mean_sd_equal_hvg2000': bool(cx.age_mean == ctx1.age_mean and cx.age_sd == ctx1.age_sd),
           'pbulk_norm_bitwise_hvg2000': bool(torch.equal(pa1.m, pax.m) and torch.equal(pa1.sd, pax.sd) and pa1.s_pb == pax.s_pb),
           'generator_init_bitwise_hvg2000': trainer._same(b0[1].state_dict(), b1g[1].state_dict())}
    r46['PASS'] = all(r46.values())
    res['r4_6_seaad_statistics'] = r46
    log('unit R4-6', r46)
    # extperm control (review F1): batches equal cohortfix except slot 3 of external donors
    import arm_cohort as AC
    cp = trainer.Context(CACHEX, 'cpu', arms.parse_arm('cohortfix,extperm,pbulk'))
    pa = [a for a in cp.plugins if a.flag == 'extperm'][0]
    _, gotp = _batches(cp)
    _, gotf = bs['cohortfix,pbulk']
    _, _, r, _, _ = trainer.build(cp, 0)
    state = r.bit_generator.state
    ds = r.integers(len(cp.ids), size=cp.batch_size)
    r.bit_generator.state = state
    _, cfirst, _ = cp.batch(r, 'critic')
    ext = cp.cohort[ds] == 1
    ep = {'batches_equal_except_slot3': all(bool(torch.equal(a[0], b_[0]) and torch.equal(a[2], b_[2])
                                                 and np.array_equal(np.delete(a[1].numpy(), 3, 1), np.delete(b_[1].numpy(), 3, 1)))
                                            for a, b_ in zip(gotp, gotf)),
          'slot3_is_permuted_external_stage': bool(np.array_equal(cfirst[:, 3].numpy(), cp.raw_c[pa.src[ds], 3].astype(np.float32))),
          'seaad_slot3_unchanged': bool(np.array_equal(cfirst[~ext, 3].numpy(), cp.raw_c[ds[~ext], 3].astype(np.float32))),
          'src_identity_on_seaad': bool(np.array_equal(pa.src[cp.cohort == 0], np.flatnonzero(cp.cohort == 0))),
          'src_is_permutation_of_external': bool(sorted(pa.src[cp.cohort == 1]) == list(np.flatnonzero(cp.cohort == 1))),
          'offset_equal_cohortfix': bool(np.array_equal([a for a in cp.plugins if a.flag == 'cohortfix'][0].offset, arm.offset)),
          'generation_conditions_equal': bool(np.array_equal(generate.default_plans(cp, generate.targets(cp))[1],
                                                             generate.default_plans(ctxs['cohortfix,pbulk'], generate.targets(ctxs['cohortfix,pbulk']))[1])),
          'perm_info': pa.info['perm']}
    try:
        arms.parse_arm('extperm,nostage,pbulk')
        ep['nostage_conflict'] = False
    except ValueError:
        ep['nostage_conflict'] = True
    ep['PASS'] = all(v for k, v in ep.items() if isinstance(v, bool))
    res['extperm'] = ep
    log('unit extperm', ep['PASS'], ep['perm_info'])
    del cp
    T1, Tx = generate.targets(ctx1), generate.targets(ctxs['cohort,pbulk'])
    same = len(T1) == len(Tx) == 37 and all(a[0] == b_[0] and a[1] == b_[1] and np.array_equal(a[2], b_[2]) and np.array_equal(a[3], b_[3])
                                              for a, b_ in zip(T1, Tx))
    pl, conds = generate.default_plans(ctxs['cohort,pbulk'], Tx)
    pl1, conds1 = generate.default_plans(ctx1, T1)
    st = tuple(generate.available_states(Path('/nonexistent'), {'round': 4, 'snapshot_updates': [], 'shadow_ema_decays': [0.999, 0.9995, 0.9998]}))
    mp, _, _ = generate.mode_plans(ctxs['cohort,pbulk'], 'asmadrc@final', Tx, st, updates=20)
    sp_, _, _ = generate.mode_plans(ctxs['cohort,pbulk'], 'stageblind@final', Tx, st, updates=20)
    sp1, _, _ = generate.mode_plans(ctx1, 'stageblind@final', T1, st, updates=20)
    gen = {'targets_equal_hvg2000_including_depth_pools': bool(same),
           'default_conditions_equal_hvg2000': bool(np.array_equal(conds, conds1)),
           'default_slot1_zero': bool((conds[:, 1] == 0).all()),
           'asmadrc_slot1_one_rest_default': bool(all((c[:, 1] == 1).all() and np.array_equal(np.delete(c, 1, 1), np.delete(np.repeat(conds[i][None], 32, 0), 1, 1))
                                                      for i, (c, _) in enumerate(mp))),
           'stageblind_conditions_equal_hvg2000': bool(all(np.array_equal(a[0], b_[0]) for a, b_ in zip(sp_, sp1))),
           'stage_proportions_seaad': __import__('arm_condshrink').train_stage_proportions(ctxs['cohort,pbulk']).tolist()}
    try:
        generate.mode_plans(ctxs['pbulk'], 'asmadrc@final', Tx, st, updates=20)
        gen['asmadrc_refused_on_naive'] = False
    except ValueError:
        gen['asmadrc_refused_on_naive'] = True
    try:
        trainer.Context(CACHE1, 'cpu', arms.parse_arm('cohort,pbulk'))
        gen['cohort_refused_on_hvg2000'] = False
    except ValueError:
        gen['cohort_refused_on_hvg2000'] = True
    try:
        arms.parse_arm('cohort,cohortfix')
        gen['cohort_cohortfix_conflict'] = False
    except ValueError:
        gen['cohort_cohortfix_conflict'] = True
    gen['nostage_composes'] = bool(trainer.Context(CACHEX, 'cpu', arms.parse_arm('cohort,nostage,pbulk')).conditions.abs().sum() == 0)
    gen['PASS'] = all(v for k, v in gen.items() if isinstance(v, bool))
    res['generation'] = gen
    log('unit generation', gen['PASS'])
    # static check: which r4 modules index condition slot 1 / raw_c slots 0-2
    import re
    pat = re.compile(r'(?:\bc|cond|out|conds?|raw_c|raws?)\[\s*(?:\.\.\.|:)\s*,\s*(?:0|1|2)\s*(?:\]|:)')
    hits = {}
    for p in sorted(HERE.glob('*.py')):
        if p.name.startswith(('test_', 'ext_')):
            continue
        for i, line in enumerate(p.read_text().splitlines(), 1):
            if pat.search(line):
                hits.setdefault(p.name, []).append(i)
    res['static_slot_readers'] = hits
    res['PASS'] = all(v['PASS'] for k, v in res.items() if isinstance(v, dict) and 'PASS' in v)
    save('unit', res)
    assert res['PASS'], json.dumps(res, indent=1, default=str)
    return res


# ============================================================================ cli (tiny CPU fits)
ARMS = ('pbulk', 'cohort,pbulk', 'cohortfix,pbulk', 'cohortfix,extperm,pbulk')


def cli():
    root, logs = SM / 'fits_cpu', SM / 'logs_cpu'
    jobs = SM / 'scratch' / 'jobs_cpu.txt'
    jobs.parent.mkdir(parents=True, exist_ok=True)
    jobs.write_text(''.join(f'{a} hvg2000x 0 0 snapens,stagehalf,stageblind,condmix@1{",asmadrc" if "cohort" in a else ""}\n' for a in ARMS))
    p, t = sh(['bash', HERE / 'run_screen.sh', '-d', 'cpu', '-p', '3', '-t', '3', '-u', '20', '-j', jobs, '-o', root, '-l', logs])
    res = {'screen_seconds': t, 'stdout_tail': p.stdout[-1500:]}
    for a in ARMS:
        fit = root / a / 'hvg2000x' / 'fold0' / 'seed0'
        cfg = json.loads((fit / 'config.json').read_text())
        rec = json.loads((fit / 'fit_receipt.json').read_text())
        g = np.load(fit / 'generated.npz')
        row = {'status': rec['status'], 'round': cfg['round'], 'n_train_donors': len(cfg['train_donors']),
               'external': len(cfg.get('external_train_donors', [])), 'targets': int(len(g['targets'])),
               'heldout': int(g['is_heldout'].sum()), 'targets_seaad_only': not any(str(x).startswith('MADRC') for x in g['targets']),
               's_per_update_cpu': rec['seconds_per_update_this_session'],
               'modes': sorted(x.name for x in fit.glob('generated_*.npz'))}
        if 'cohort' in a:
            m = np.load(fit / 'generated_asmadrc.npz')
            row['asmadrc_slot1'] = sorted(set(m['conditions'][..., 1].ravel().tolist()))
            d = np.load(fit / 'generated_snapens.npz')
            row['asmadrc_vs_snapens_mean_log_abs_diff'] = float(np.abs(m['mean_log'] - d['mean_log']).mean())
        row['PASS'] = bool(row['status'] == 'completed' and row['round'] == 4 and row['targets'] == 37 and row['targets_seaad_only']
                           and row['external'] == 31 and len(row['modes']) == (5 if 'cohort' in a else 4))
        res[a] = row
        log('cli', a, row['PASS'])
    # guards: frozen roots refused by run_screen.sh and trainer.py; round-3 fit generation needs --out
    guards = {}
    bad = SM / 'scratch' / 'jobs_one.txt'
    bad.write_text('pbulk hvg2000x 0 0\n')
    for root_ in (RUNS / 'fits_r3', RUNS / 'smoke_r3' / 'x', RUNS / 'ext' / 'x', RUNS / 'caches' / 'x', RUNS / 'somewhere_new'):
        p, _ = sh(['bash', HERE / 'run_screen.sh', '-d', 'cpu', '-p', '1', '-j', bad, '-o', root_, '-n'], check=False)
        p2, _ = sh([PY, '-B', HERE / 'trainer.py', '--cache', CACHEX, '--arm', 'pbulk', '--seed', '0', '--out-root', root_, '--updates', '1'], check=False)
        guards[str(root_)] = [p.returncode == 2, p2.returncode != 0 and 'refusing' in p2.stderr]
    r3fit = next((RUNS / 'fits_r3').glob('*/hvg2000/fold*/seed*/final.pt'), None)
    if r3fit is not None:
        p, _ = sh([PY, '-B', HERE / 'generate.py', '--fit', r3fit.parent, '--cache', CACHE1, '--mode', 'snapens'], check=False)
        guards['round3_fit_needs_out'] = p.returncode != 0 and 'frozen' in p.stderr
    res['guards'] = guards
    res['guards_PASS'] = all(all(v) if isinstance(v, list) else v for v in guards.values())
    res['PASS'] = all(res[a]['PASS'] for a in ARMS) and res['guards_PASS']
    save('cli', res)
    assert res['PASS'], json.dumps(res, indent=1, default=str)
    return res


# ============================================================================ gpu (300 updates)
def gpu(dev):
    root, logs = SM / 'fits_gpu', SM / 'logs_gpu'
    jobs = SM / 'scratch' / 'jobs_gpu.txt'
    jobs.parent.mkdir(parents=True, exist_ok=True)
    lines = [f'pbulk hvg2000 0 0 snapens', f'pbulk hvg2000x 0 0 {SCREEN_MODES}',
             f'cohort,pbulk hvg2000x 0 0 {SCREEN_MODES},asmadrc', f'cohortfix,pbulk hvg2000x 0 0 {SCREEN_MODES},asmadrc',
             f'cohortfix,extperm,pbulk hvg2000x 0 0 {SCREEN_MODES},asmadrc']
    jobs.write_text('\n'.join(lines) + '\n')
    env = dict(ENV)
    env.pop('CUDA_VISIBLE_DEVICES', None)
    p, t = sh(['bash', HERE / 'run_screen.sh', '-d', dev, '-p', '1', '-t', '3', '-u', '300', '-g', '0.04', '-j', jobs, '-o', root, '-l', logs],
              env=env, check=False)
    res = {'screen_seconds': t, 'returncode': p.returncode, 'stdout_tail': p.stdout[-3000:], 'stderr_tail': p.stderr[-2000:]}
    for line in lines:
        a, panel = line.split()[:2]
        fit = root / a / panel / 'fold0' / 'seed0'
        rec = json.loads((fit / 'fit_receipt.json').read_text()) if (fit / 'fit_receipt.json').exists() else {}
        row = {'status': rec.get('status'), 's_per_update': rec.get('seconds_per_update_this_session'),
               'peak_memory': rec.get('peak_memory'), 'peak_memory_after_generation': rec.get('peak_memory_after_generation'),
               'generation_seconds': (rec.get('generation') or {}).get('seconds'),
               'modes': sorted(x.name for x in fit.glob('generated_*.npz'))}
        mode_secs = {}
        for rp in fit.glob('generated_*.receipt.json'):
            q = json.loads(rp.read_text())
            mode_secs[q.get('mode')] = round(q.get('seconds', 0), 1)
        row['mode_seconds'] = mode_secs
        curve = [json.loads(l) for l in (fit / 'learning_curve.jsonl').read_text().splitlines()] if (fit / 'learning_curve.jsonl').exists() else []
        row['critic_gap'] = [(c['step'], round(c['critic_gap'], 3)) for c in curve]
        res[f'{a}|{panel}'] = row
    res['PASS'] = p.returncode == 0 and all(v.get('status') == 'completed' for k, v in res.items() if isinstance(v, dict))
    save('gpu', res)
    log('gpu', res['PASS'])
    return res


def cleanup():
    for d in ('fits_cpu', 'logs_cpu', 'scratch'):
        shutil.rmtree(SM / d, ignore_errors=True)


if __name__ == '__main__':
    what = sys.argv[1] if len(sys.argv) > 1 else 'unit'
    {'parity': parity, 'unit': unit, 'cli': cli, 'cleanup': cleanup}.get(what, lambda: gpu(sys.argv[2]))()
