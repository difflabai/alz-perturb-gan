"""Tests and real-data measurements for the round-5 critic block ``pzero`` (r5/arm_pzero.py, work item 2 Fix B).

CPU only except ``gpu`` (3 threads per process; development donors only; held-out
counts are never read: every measurement here uses TRAIN donors of the fold).
Run from /tmp::

    python3 -B r5/test_arm_pzero.py unit          # parsing, forward equality, gradients (autograd / FD / closed-form NB), scale, composability, state
    python3 -B r5/test_arm_pzero.py neutral       # importing / building pzero leaves a pbulk trajectory bitwise unchanged (subprocesses)
    python3 -B r5/test_arm_pzero.py separability  # round-3 fits, snapshot states: real vs fake bags along pbulk and pzero blocks
    python3 -B r5/test_arm_pzero.py fits          # tiny CPU CLI fits (20 updates) + snapens in a new 1-thread process
    python3 -B r5/test_arm_pzero.py gpu           # one 300-update GPU fit per recommended arm through r5/run_screen.sh + modes
    python3 -B r5/test_arm_pzero.py cleanup       # delete weights / generated files (keep results/*.json, configs, receipts, logs)
    python3 -B r5/test_arm_pzero.py all           # unit neutral separability fits (not gpu)

Scratch and results: /ml2/alz/runs/improve_20261109/smoke_r5/pzero/ (results/*.json kept).

unit
    * parsing: pzero, pzero@0.5 accepted; pzero@1, pzero@0, pzero@-1, pzero@0.50 refused;
      pbulk,pzero / genemean,pzero / genemean,pbulk,pzero accepted.
    * forward: block.fake(cf, rate, counts) == block.real(idx) BITWISE when counts are the real counts of
      the same cells (any rate).
    * gradient (a): autograd d<u,out>/d rate == -u_g w_g exp(-rate) / cells (closed form of the
      straight-through path), float32 rtol 1e-5.
    * gradient (b): float64 central finite difference of the EXPECTED block given the rates,
      E[out | rate] = s s_pz (mean_cells exp(-rate) - m)/sd, along a random direction equals <grad, v>
      (relative error < 1e-8 at eps 1e-6).
    * gradient (c): closed-form NB zero probability.  For 64 genes with mu in [0.02, 8], theta in [0.1, 20]:
      the Monte-Carlo mean (20 x 20,000 Gamma rsample draws) of the pathwise gradient of exp(-rate) w.r.t.
      (mu, theta) equals autograd of (theta/(theta+mu))^theta within 4 Monte-Carlo SE (+1e-4 abs); the
      Poisson zero frequency matches the closed form within 4 SE.  This is why the block's exp(-rate)
      path is an unbiased estimate of the closed-form gradient.
    * scale: on the 512 reference bags, recomputed through the float32 torch Features path,
      E||pzero block||^2 / E||f150||^2 = 1 within 1e-3 (pzero alone).
    * composability: widths 2150 (pzero), 4150 (pbulk,pzero), 4150 (genemean,pzero), 6150 (genemean,pbulk,pzero);
      block order; the pzero columns are identical in every combination and the pbulk columns of
      pbulk,pzero equal those of pbulk alone (bitwise).
    * generator gradient: a loss on the pzero columns of fake features through a freshly built
      pzero generator gives nonzero gradient on G.out[-1].weight and G.log_theta (and P.z via z).
    * state: the arm state recomputed in a 1-thread subprocess equals this process's exactly.

neutral
    Two subprocesses run 3 CPU updates of ``pbulk`` (seed 0, fold 0): one plain, one that first imports
    arm_pzero and sets up a ``pzero`` Context.  Initial tensors, final tensors, metrics, numpy and torch
    RNG states must be bitwise equal.  (Trainer parity r5 == round 1 / round 2 / r3 with no round-5 flag:
    test_core.py parity.)

separability (no training; round-3 fits fits_r3/{pbulk, g.dispz,pbulk,t.kappa@0.25}/hvg2000/fold{0,1,2}/seed0)
    For each fit and state (snap_10000, snap_15000, snap_19000, final): 512 fake training-style bags
    (32 train donors uniform x 16 batches, 128 cells; the condition of each batch donor; the libraries
    of 128 real cells of that donor; z from the fitted particle prior) and the 512 PAIRED real bags (the
    same donors' same real cells), plus an independent second real draw (null).  Per gene and block
    (pbulk, pzero; standardised with the arms' fixed reference statistics):
    d_g = (mean_fake_g - mean_real_g) / sd_real_g.  Reported per expression quartile (quartiles of the
    pbulk reference mean m_g): RMS d (fake and null), mean d (sign; pzero: >0 = too many zeros),
    corr(d_pz, d_pb), RMS of d_pz after regressing it on d_pb within the quartile (pzero offset not
    explained by the pbulk offset), the block's ||mean offset||^2 / V_real (as test_arm_critic shares)
    and the absolute RMS zero-fraction error (the unit of nonzero_rmse_gene, bag level).

fits
    trainer.py CLI (CPU, 3 threads, 20 updates) for pzero,t.kappa@0.25 / pbulk,pzero,t.kappa@0.19 /
    genemean,pbulk,pzero (the trainer's update-0 assertions: nonzero critic gradient on every block,
    generator gradient); then generate.py --mode snapens in a new process with 1 thread for the first two
    (state recomputed and compared with final.pt and every snapshot).

gpu
    r5/run_screen.sh -d cuda:N -p 1 -t 3 -u 300 -g 0.04 -o smoke_r5/pzero/fits -l smoke_r5/pzero/logs
    with jobs ``pzero,t.kappa@0.25``, ``pbulk,pzero,t.kappa@0.19`` and the variant
    ``g.dispz,pbulk,pzero,t.kappa@0.19`` (hvg2000 fold 0 seed 0, one process at a time), modes
    snapens,stagehalf,stageblind,biascal@stagehalf.  Records s/update, peak CUDA memory, critic gap
    curve, block info, mode files; then deletes the weights (*.pt) and the generated npz files.
"""
import os
import sys

sys.dont_write_bytecode = True
for _k in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(_k, '3')
if (sys.argv[1] if len(sys.argv) > 1 else '') != 'gpu':
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

torch.set_num_threads(int(os.environ.get('OMP_NUM_THREADS', '3')))
import trainer  # noqa: E402  (r5)
import arms  # noqa: E402
import generate  # noqa: E402
import arm_pbulk  # noqa: E402
import arm_pzero  # noqa: E402
from pg_model import critic_features  # noqa: E402

PY = '/mnt/nvme/ai/conda/anaconda3/bin/python3'
RUNS = trainer.RUNS
CACHES = RUNS / 'caches/hvg2000'
CACHE0 = CACHES / 'fold0'
FITS3 = RUNS / 'fits_r3'
SCR = RUNS / 'smoke_r5/pzero'
RES = SCR / 'results'
TINY = 20
ARMS_RECOMMENDED = ['pzero,t.kappa@0.25', 'pbulk,pzero,t.kappa@0.19']
VARIANT = 'g.dispz,pbulk,pzero,t.kappa@0.19'   # optional variant (see arm_pzero docstring, 'Recommended arms')
GPU_ARMS = ARMS_RECOMMENDED + [VARIANT]
GPU_MODES = 'snapens,stagehalf,stageblind,biascal@stagehalf'
ENV = dict(os.environ, OMP_NUM_THREADS='3', OPENBLAS_NUM_THREADS='3', MKL_NUM_THREADS='3', PYTHONDONTWRITEBYTECODE='1')
ENV_CPU = dict(ENV, CUDA_VISIBLE_DEVICES='')


def log(*a):
    print('[test_arm_pzero]', *a, flush=True)


def save(name, obj):
    trainer.dump(RES / f'{name}.json', json.loads(json.dumps(obj, default=str)))


def r(x, n=4):
    return float(round(float(x), n))


def sh(cmd, env=ENV_CPU, check=True, cwd='/tmp'):
    p = subprocess.run([str(c) for c in cmd], env=env, capture_output=True, text=True, cwd=cwd)
    if check and p.returncode != 0:
        raise RuntimeError(f'command failed ({p.returncode}): {cmd}\n{p.stdout[-3000:]}\n{p.stderr[-5000:]}')
    return p


def refused(text):
    try:
        arms.parse_arm(text)
    except ValueError:
        return True
    return False


def block_of(ctx, flag):
    return [(a, b, lo, hi) for a, b, lo, hi in ctx.features.blocks if a.flag == flag][0]


# =========================================================================== unit
def unit():
    t0 = time.monotonic()
    res = {}
    ok = {'pzero': not refused('pzero'), 'pzero@0.5': not refused('pzero@0.5'),
          'pzero@1 refused': refused('pzero@1'), 'pzero@0 refused': refused('pzero@0'), 'pzero@-1 refused': refused('pzero@-1'),
          'pzero@0.50 refused': refused('pzero@0.50'), 'pbulk,pzero': not refused('pbulk,pzero'),
          'genemean,pzero': not refused('genemean,pzero'), 'genemean,pbulk,pzero': not refused('genemean,pbulk,pzero'),
          'pbulk,pzero,t.kappa@0.19': not refused('pbulk,pzero,t.kappa@0.19')}
    res['parsing'] = ok
    assert all(ok.values()), ok
    log('parsing ok')

    ctx = trainer.Context(CACHE0, 'cpu', ('pzero',))
    a, blk, lo, hi = block_of(ctx, 'pzero')
    res['info'] = a.block_info
    log('block info', json.dumps(a.block_info))
    rng = np.random.default_rng(np.random.SeedSequence([20261109, 97, 1]))
    ds = rng.integers(len(ctx.by), size=8)
    idx = torch.as_tensor(np.stack([rng.choice(ctx.by[i], 128, replace=True) for i in ds]))
    x = ctx.X[idx]
    gen = torch.Generator().manual_seed(5)
    rate = (torch.rand(x.shape, generator=gen) * 3).requires_grad_(True)
    cf = critic_features(x, rate, ctx.mean, ctx.sd)
    fr = blk.real(ctx, idx, ds)
    ff = blk.fake(ctx, cf, rate, x)
    res['forward_bitwise'] = bool(torch.equal(fr, ff.detach()))
    assert res['forward_bitwise']
    # (a) autograd vs closed form of the straight-through path
    u = torch.randn(ff.shape, generator=gen)
    g = torch.autograd.grad((u * ff).sum(), rate)[0]
    want = -(u * a.w)[:, None, :] * torch.exp(-rate.detach()) / x.shape[1]
    res['grad_closed_form_max_rel'] = r(float((g - want).abs().max() / want.abs().max()), 10)
    assert torch.allclose(g, want, rtol=1e-5, atol=1e-9), res['grad_closed_form_max_rel']
    # (b) float64 FD of the expected block given the rates
    m64, w64 = a.m.double(), a.w.double()
    rate64 = rate.detach().double().requires_grad_(True)
    out64 = (arm_pzero.zero_frac(x.double(), rate64) - m64) * w64
    u64 = u.double()
    g64 = torch.autograd.grad((u64 * out64).sum(), rate64)[0]
    v = torch.randn(rate64.shape, generator=gen, dtype=torch.float64)

    def expected(rr):
        return float((u64 * ((torch.exp(-rr).mean(1) - m64) * w64)).sum())
    fd = {}
    for eps in (1e-4, 1e-5, 1e-6):
        num = (expected(rate64.detach() + eps * v) - expected(rate64.detach() - eps * v)) / (2 * eps)
        ana = float((g64 * v).sum())
        fd[str(eps)] = r(abs(num - ana) / abs(ana), 12)
    res['fd_expected_block_rel_err'] = fd
    assert fd['1e-06'] < 1e-8, fd
    log('forward / autograd / FD ok', res['grad_closed_form_max_rel'], fd)
    # (c) closed-form NB zero probability: pathwise MC gradient of exp(-Gamma) == autograd of (theta/(theta+mu))^theta
    G_ = 64
    gg = torch.Generator().manual_seed(11)
    mu = torch.exp(torch.linspace(np.log(0.02), np.log(8.0), G_, dtype=torch.float64)).requires_grad_(True)
    th = torch.exp(torch.linspace(np.log(0.1), np.log(20.0), G_, dtype=torch.float64)).flip(0)
    th = th[torch.randperm(G_, generator=gg)].clone().requires_grad_(True)
    p0 = (th / (th + mu)) ** th
    gmu_cf, gth_cf = torch.autograd.grad(p0.sum(), (mu, th))
    torch.manual_seed(1234)   # Gamma.rsample has no generator argument; this test owns its process RNG
    gm, gt, f0, fz = [], [], [], []
    for _ in range(20):
        rr = torch.distributions.Gamma(th.expand(20000, G_), (th / mu).expand(20000, G_)).rsample()
        e = torch.exp(-rr).mean(0)
        a1, a2 = torch.autograd.grad(e.sum(), (mu, th))
        gm.append(a1); gt.append(a2); f0.append(e.detach())
        fz.append((torch.poisson(rr.detach()) == 0).double().mean(0))
    gm, gt, f0, fz = (torch.stack(t) for t in (gm, gt, f0, fz))

    def zmax(est, ref):
        se = est.std(0) / np.sqrt(len(est))
        return float(((est.mean(0) - ref).abs() / (4 * se + 1e-4)).max())
    res['nb_closed_form'] = {
        'grad_mu_max_ratio_to_4se': r(zmax(gm, gmu_cf)), 'grad_theta_max_ratio_to_4se': r(zmax(gt, gth_cf)),
        'expected_exp_rate_max_ratio_to_4se': r(zmax(f0, p0.detach())), 'poisson_zero_freq_max_ratio_to_4se': r(zmax(fz, p0.detach())),
        'max_abs_grad_mu_diff': r(float((gm.mean(0) - gmu_cf).abs().max()), 6), 'max_abs_grad_mu': r(float(gmu_cf.abs().max()), 6),
        'mc_se_ratio_exp_rate_vs_indicator_median': r(float((f0.std(0) / fz.std(0).clamp_min(1e-12)).median()))}
    log('NB closed form', json.dumps(res['nb_closed_form']))
    assert all(v <= 1.0 for k, v in res['nb_closed_form'].items() if k.endswith('4se')), res['nb_closed_form']
    # scale through the float32 Features path on the reference bags
    num = den = 0.0
    with torch.no_grad():
        for _, rows in arm_pbulk.ref_bag_rows(ctx.by, 'pzero'):
            f = ctx.features.real(torch.as_tensor(rows), None).double()
            num += float((f[:, lo:hi] ** 2).sum()); den += float((f[:, :150] ** 2).sum())
    res['scale_ratio_E_norm2_block_over_150'] = r(num / den, 6)
    assert abs(num / den - 1) < 1e-3, num / den
    log('scale ok', res['scale_ratio_E_norm2_block_over_150'])
    # composability
    comp = {}
    ref_pz = fr
    ctx_pb = trainer.Context(CACHE0, 'cpu', ('pbulk',))
    ref_pb = block_of(ctx_pb, 'pbulk')[1].real(ctx_pb, idx, ds)
    for arm, width in [('pzero', 2150), ('pbulk,pzero', 4150), ('genemean,pzero', 4150), ('genemean,pbulk,pzero', 6150)]:
        c2 = ctx if arm == 'pzero' else trainer.Context(CACHE0, 'cpu', arms.parse_arm(arm))
        full = c2.features.real(idx, ds)
        _, _, lo2, hi2 = block_of(c2, 'pzero')
        row = {'n_features': c2.features.n_features, 'blocks': [(a2.flag, b2.name, l2, h2) for a2, b2, l2, h2 in c2.features.blocks],
               'pzero_columns_equal': bool(torch.equal(full[:, lo2:hi2], ref_pz))}
        if 'pbulk' in arm:
            _, _, plo, phi = block_of(c2, 'pbulk')
            row['pbulk_columns_equal'] = bool(torch.equal(full[:, plo:phi], ref_pb))
        row['PASS'] = row['n_features'] == width and row['pzero_columns_equal'] and row.get('pbulk_columns_equal', True) \
            and hi2 == width
        comp[arm] = row
        log('compose', arm, row['PASS'], row['blocks'])
    res['composability'] = comp
    assert all(v['PASS'] for v in comp.values())
    # generator gradient through a freshly built generator
    b, gens, rng2, us, s = trainer.build(ctx, 0)
    G, P = b[1], b[3]
    real, c, lib = ctx.batch(rng2, 'critic')
    z, _ = P.sample(ctx.batch_size, generator=torch.Generator().manual_seed(3))
    rate = G.rates(z, c, ctx.bag, lib, torch.Generator().manual_seed(4))
    cnt = torch.poisson(rate.detach(), generator=torch.Generator().manual_seed(6))
    feats = ctx.features.fake(critic_features(cnt, rate, ctx.mean, ctx.sd), rate, cnt)
    G.zero_grad(); P.zero_grad()
    (feats[:, lo:hi] * torch.randn(feats[:, lo:hi].shape, generator=gen)).sum().backward()
    gp = {'out_last_weight': float(G.out[-1].weight.grad.abs().sum()), 'log_theta': float(G.log_theta.grad.abs().sum()),
          'P.z': float(P.z.grad.abs().sum()) if P.z.grad is not None else 0.0}
    res['generator_gradient_abs_sums'] = gp
    assert gp['out_last_weight'] > 0 and gp['log_theta'] > 0, gp
    log('generator gradient ok', gp)
    # state recompute in a 1-thread subprocess
    out = SCR / 'state_1thread.pt'
    SCR.mkdir(parents=True, exist_ok=True)
    code = textwrap.dedent(f"""
        import sys, torch; sys.path.insert(0, {str(HERE)!r}); torch.set_num_threads(1)
        import trainer
        ctx = trainer.Context({str(CACHE0)!r}, 'cpu', ('pbulk', 'pzero'))
        torch.save(ctx.arm_state()['r2'], {str(out)!r})
    """)
    sh([PY, '-B', '-c', code], env=dict(ENV_CPU, OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1'))
    other = torch.load(out, weights_only=False)
    out.unlink()
    mine = trainer.Context(CACHE0, 'cpu', ('pbulk', 'pzero')).arm_state()['r2']
    res['state_equal_1thread_subprocess'] = bool(trainer._same(mine, other))
    assert res['state_equal_1thread_subprocess']
    res['seconds'] = time.monotonic() - t0
    res['PASS'] = True
    save('unit', res)
    log('unit PASS')
    return res


# =========================================================================== neutral
NEUTRAL_CODE = """
import sys, torch, numpy as np
sys.path.insert(0, {here!r}); torch.set_num_threads(3)
import trainer, arms
if {with_pzero}:
    import arm_pzero
    trainer.Context({cache!r}, 'cpu', ('pzero',))
    np.random.random(3); torch.rand(3)   # touch the global streams too: build() must reseed everything it uses
ctx = trainer.Context({cache!r}, 'cpu', ('pbulk',))
b, g, r, u, s = trainer.build(ctx, 0)
init = [{{k: v.detach().clone() for k, v in m.state_dict().items()}} for m in (b[1], b[2], b[3])]
mets = [trainer.advance(ctx, b, g, r, u, execution_limit=20000) for _ in range(3)]
fin = [{{k: v.detach().clone() for k, v in m.state_dict().items()}} for m in (b[1], b[2], b[3])]
torch.save(dict(init=init, fin=fin, mets=mets, numpy=r.bit_generator.state, gens={{k: v.get_state() for k, v in g.items()}},
                n_features=ctx.features.n_features), {out!r})
"""


def neutral():
    t0 = time.monotonic()
    SCR.mkdir(parents=True, exist_ok=True)
    outs = {}
    for w in (False, True):
        o = SCR / f'neutral_{int(w)}.pt'
        sh([PY, '-B', '-c', NEUTRAL_CODE.format(here=str(HERE), with_pzero=w, cache=str(CACHE0), out=str(o))])
        outs[w] = torch.load(o, weights_only=False)
        o.unlink()
    A, B = outs[False], outs[True]

    def sd_eq(x, y):
        return set(x) == set(y) and all(torch.equal(x[k], y[k]) for k in x)
    res = {'init_bitwise': all(sd_eq(x, y) for x, y in zip(A['init'], B['init'])),
           'final_bitwise': all(sd_eq(x, y) for x, y in zip(A['fin'], B['fin'])),
           'metrics_equal': json.dumps(A['mets'], default=str, sort_keys=True) == json.dumps(B['mets'], default=str, sort_keys=True),
           'numpy_equal': A['numpy'] == B['numpy'],
           'torch_generators_equal': all(torch.equal(A['gens'][k], B['gens'][k]) for k in A['gens']),
           'n_features': [A['n_features'], B['n_features']]}
    res['PASS'] = bool(res['init_bitwise'] and res['final_bitwise'] and res['metrics_equal'] and res['numpy_equal']
                       and res['torch_generators_equal'] and A['n_features'] == B['n_features'] == 2150)
    res['seconds'] = time.monotonic() - t0
    save('neutral', res)
    log('neutral', res)
    assert res['PASS']
    return res


# =========================================================================== separability
SEP_FITS = ['pbulk', 'g.dispz,pbulk,t.kappa@0.25']
SEP_STATES = ['snap_10000', 'snap_15000', 'snap_19000', 'final']
SEP_BATCHES = 16


def _bags(mctx, rng, n_batches):
    out = []
    for _ in range(n_batches):
        ds = rng.integers(len(mctx.by), size=32)
        idx = torch.as_tensor(np.stack([rng.choice(mctx.by[i], 128, replace=True) for i in ds]))
        out.append((ds, idx))
    return out


def _blocks_of_counts(mctx, x):
    """[B,128,G] counts -> {'pbulk': [B,G], 'pzero': [B,G]} standardised with the arms' fixed reference statistics (float64)."""
    pb, pz = mctx.plugins
    assert pb.name == 'pbulk' and pz.name == 'pzero'
    x = x.double()
    ypb = arm_pbulk.log_cp10k_pb(x)
    ypz = arm_pzero.zero_frac(x)
    return {'pbulk': ((ypb - pb.m.double()) / pb.sd.double()).numpy(), 'pzero': ((ypz - pz.m.double()) / pz.sd.double()).numpy(),
            'p0_raw': ypz.numpy()}


def _compare(real, fake, quart, null=None):
    sd = {k: real[k].std(0) + 1e-12 for k in ('pbulk', 'pzero')}
    d = {k: (fake[k].mean(0) - real[k].mean(0)) / sd[k] for k in ('pbulk', 'pzero')}
    out = {}
    for q in list(range(4)) + ['all']:
        gq = np.ones_like(quart, bool) if q == 'all' else quart == q
        dpb, dpz = d['pbulk'][gq], d['pzero'][gq]
        beta = float(dpz @ dpb / max(dpb @ dpb, 1e-12))
        row = {'rms_d_pbulk': r(np.sqrt((dpb ** 2).mean())), 'rms_d_pzero': r(np.sqrt((dpz ** 2).mean())),
               'mean_d_pbulk': r(dpb.mean()), 'mean_d_pzero': r(dpz.mean()),
               'corr_dpz_dpb': r(np.corrcoef(dpz, dpb)[0, 1], 3),
               'rms_d_pzero_resid_on_pbulk': r(np.sqrt(((dpz - beta * dpb) ** 2).mean())),
               'frac_genes_abs_d_pzero_gt_0.5': r((np.abs(dpz) > 0.5).mean(), 3),
               'rms_abs_p0_error': r(np.sqrt(((fake['p0_raw'].mean(0) - real['p0_raw'].mean(0))[gq] ** 2).mean()), 5),
               'mean_abs_p0_error_signed': r((fake['p0_raw'].mean(0) - real['p0_raw'].mean(0))[gq].mean(), 5)}
        out[f'q{q + 1}' if q != 'all' else 'all'] = row
    for k in ('pbulk', 'pzero'):
        V = float(((real[k] - real[k].mean(0)) ** 2).sum(1).mean())
        out[f'offset2_over_V_{k}'] = r(float(((fake[k].mean(0) - real[k].mean(0)) ** 2).sum()) / V)
    return out


def separability(folds=(0, 1, 2)):
    t0 = time.monotonic()
    out = {'fits': SEP_FITS, 'states': SEP_STATES, 'bags': SEP_BATCHES * 32, 'results': {}}
    for k in folds:
        cache = CACHES / f'fold{k}'
        mctx = trainer.Context(cache, 'cpu', ('pbulk', 'pzero'))
        pb = mctx.plugins[0]
        level = pb.m.double().numpy()
        quart = np.searchsorted(np.quantile(level, [0.25, 0.5, 0.75]), level)
        rng = np.random.default_rng(np.random.SeedSequence([20261109, 96, k]))
        bags = _bags(mctx, rng, SEP_BATCHES)
        bags2 = _bags(mctx, rng, SEP_BATCHES)
        with torch.no_grad():
            real = [_blocks_of_counts(mctx, mctx.X[idx]) for _, idx in bags]
            real2 = [_blocks_of_counts(mctx, mctx.X[idx]) for _, idx in bags2]
        cat = lambda L: {kk: np.concatenate([x[kk] for x in L]) for kk in L[0]}  # noqa: E731
        real, real2 = cat(real), cat(real2)
        fo = {'genes_per_quartile': np.bincount(quart).tolist(), 'pzero_info': mctx.plugins[1].info,
              'null_real_vs_real': _compare(real, real2, quart), 'fits': {}}
        log('fold', k, 'null', json.dumps(fo['null_real_vs_real']['all']))
        for arm in SEP_FITS:
            for seed in (0,):
                fit = FITS3 / arm / 'hvg2000' / f'fold{k}' / f'seed{seed}'
                if not (fit / 'final.pt').exists():
                    log('missing', fit); continue
                config = json.loads((fit / 'config.json').read_text())
                fctx = trainer.Context(cache, 'cpu', tuple(config['flags']))
                models = generate.load_models(fctx, fit, config, SEP_STATES)
                for st in SEP_STATES:
                    G, P = models[st]
                    G.eval(); P.eval()
                    gl, gc, gk = (torch.Generator().manual_seed(1000 * k + s_) for s_ in (11, 12, 13))
                    fake = []
                    with torch.no_grad():
                        for ds, idx in bags:
                            lib = mctx.X[idx].sum(-1)
                            cond = fctx.conditions[torch.as_tensor(ds)]
                            z, _ = P.sample(len(ds), generator=gl)
                            rate = G.rates(z, cond, 128, lib, gc)
                            fake.append(_blocks_of_counts(mctx, torch.poisson(rate, generator=gk)))
                    row = _compare(real, cat(fake), quart)
                    fo['fits'][f'{arm}/seed{seed}/{st}'] = row
                    log('fold', k, arm, st, json.dumps(row['all']), 'offset2/V pb', row['offset2_over_V_pbulk'], 'pz', row['offset2_over_V_pzero'])
                del models
        out['results'][f'fold{k}'] = fo
    # summary: mean over folds per fit/state and quartile
    summ = {}
    keys = sorted({kk for fo in out['results'].values() for kk in fo['fits']})
    for kk in keys + ['null']:
        rows = [fo['null_real_vs_real'] if kk == 'null' else fo['fits'].get(kk) for fo in out['results'].values()]
        rows = [x for x in rows if x]
        s = {}
        for q in ['q1', 'q2', 'q3', 'q4', 'all']:
            s[q] = {m: r(np.mean([x[q][m] for x in rows])) for m in rows[0][q]}
        s['offset2_over_V_pbulk'] = r(np.mean([x['offset2_over_V_pbulk'] for x in rows]))
        s['offset2_over_V_pzero'] = r(np.mean([x['offset2_over_V_pzero'] for x in rows]))
        s['n_folds'] = len(rows)
        summ[kk] = s
    out['summary_mean_over_folds'] = summ
    out['seconds'] = time.monotonic() - t0
    save('separability', out)
    log('separability done', round(out['seconds']))
    return out


# =========================================================================== fits (tiny CPU)
FIT_ARMS = ['pzero,t.kappa@0.25', 'pbulk,pzero,t.kappa@0.19', 'genemean,pbulk,pzero']


def fits():
    root = SCR / 'tiny'
    res = {}
    for arm in FIT_ARMS:
        t = time.monotonic()
        fit = root / arm / 'hvg2000/fold0/seed0'
        if not (fit / 'generated.npz').exists():
            sh([PY, '-B', HERE / 'trainer.py', '--cache', CACHE0, '--arm', arm, '--seed', 0, '--out-root', root,
                '--updates', TINY, '--device', 'cpu', '--threads', 3])
        cfg = json.loads((fit / 'config.json').read_text())
        rec = json.loads((fit / 'fit_receipt.json').read_text())
        curve = [json.loads(l) for l in (fit / 'learning_curve.jsonl').read_text().splitlines() if l.strip()]
        with np.load(fit / 'generated.npz') as z:
            fin = all(np.isfinite(z[k].astype(np.float32)).all() for k in ['pb_counts', 'mean_log', 'nonzero', 'lib', 'sq_log'])
        assert fin and rec['status'] == 'completed' and not rec['snapshots_missing']
        res[arm] = {'critic_features': cfg['critic_features'], 'departures': cfg.get('departures'),
                    'blocks': [(b['flag'], b['block'], b['columns']) for b in cfg['arm_info'].get('r2_blocks', [])],
                    'r2_arms': cfg['r2_arms'], 'reg_kappa': cfg.get('recipe', {}).get('reg_kappa') if isinstance(cfg.get('recipe'), dict) else None,
                    'snapshots': sorted(rec['snapshots_SHA256']), 'generated_finite': fin,
                    'critic_gap_curve': [(c['step'], r(c['critic_gap'])) for c in curve], 'seconds': time.monotonic() - t}
        log(arm, 'fit ok', res[arm]['critic_features'], res[arm]['blocks'], round(res[arm]['seconds']))
    env1 = dict(ENV_CPU, OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')
    for arm in FIT_ARMS[:2]:
        fit = root / arm / 'hvg2000/fold0/seed0'
        o = fit / 'generated_snapens.npz'
        if not o.exists():
            sh([PY, '-B', HERE / 'generate.py', '--fit', fit, '--cache', CACHE0, '--mode', 'snapens', '--threads', 1], env=env1)
        with np.load(o) as z:
            names = list(z['state_names'].astype(str))
        res[arm]['snapens_new_process_1thread'] = names
        log(arm, 'snapens (1 thread, new process) ok', names)
    save('fits', res)
    log('fits PASS')
    return res


# =========================================================================== gpu (300 updates through run_screen.sh)
def gpu(device=None):
    t0 = time.monotonic()
    if device is None:
        device = os.environ.get('PZERO_GPU', 'cuda:1')
    root, logs = SCR / 'fits', SCR / 'logs'
    jobs = SCR / 'jobs_gpu.txt'
    SCR.mkdir(parents=True, exist_ok=True)
    jobs.write_text(''.join(f'{a} hvg2000 0 0 {GPU_MODES}\n' for a in GPU_ARMS))
    env = dict(ENV)
    env.pop('CUDA_VISIBLE_DEVICES', None)
    p = sh(['bash', HERE / 'run_screen.sh', '-d', device, '-p', 1, '-t', 3, '-u', 300, '-g', 0.04, '-j', jobs,
            '-o', root, '-l', logs], env=env, check=False)
    (SCR / 'gpu_screen.out').write_text(p.stdout + '\n--- stderr ---\n' + p.stderr)
    res = {'device': device, 'screen_rc': p.returncode, 'arms': {}}
    for arm in GPU_ARMS:
        fit = root / arm / 'hvg2000/fold0/seed0'
        row = {'complete': (fit / 'generated.npz').exists()}
        if (fit / 'fit_receipt.json').exists():
            rec = json.loads((fit / 'fit_receipt.json').read_text())
            cfg = json.loads((fit / 'config.json').read_text())
            curve = [json.loads(l) for l in (fit / 'learning_curve.jsonl').read_text().splitlines() if l.strip()]
            row.update({'status': rec.get('status'), 'seconds_per_update': rec.get('seconds_per_update'),
                        'wall_seconds': rec.get('wall_seconds') or rec.get('seconds'), 'peak_memory': rec.get('peak_memory'),
                        'critic_features': cfg.get('critic_features'), 'departures': cfg.get('departures'),
                        'pzero_info': cfg.get('r2_arms', {}).get('pzero'),
                        'critic_gap_curve': [(c['step'], r(c['critic_gap'])) for c in curve],
                        'D_penalty_curve': [(c['step'], r(c.get('D_penalty', float('nan')), 6)) for c in curve],
                        'receipt_keys': sorted(rec)})
        modes = {}
        for m in GPU_MODES.split(','):
            f = fit / f'generated_{m}.npz'
            if f.exists():
                with np.load(f) as z:
                    modes[m] = {'finite': bool(all(np.isfinite(z[k].astype(np.float32)).all() for k in ('pb_counts', 'nonzero'))),
                                'states': list(z['state_names'].astype(str)) if 'state_names' in z else None,
                                'MB': r(f.stat().st_size / 2 ** 20, 1)}
            else:
                modes[m] = None
        row['modes'] = modes
        row['PASS'] = bool(row['complete'] and row.get('status') == 'completed' and all(v and v['finite'] for v in modes.values()))
        res['arms'][arm] = row
        log('gpu', arm, row['PASS'], row.get('seconds_per_update'), row.get('peak_memory'))
    res['seconds'] = time.monotonic() - t0
    res['PASS'] = all(v['PASS'] for v in res['arms'].values()) and p.returncode == 0
    save('gpu', res)
    removed = _delete_weights(root)
    log('gpu', 'PASS' if res['PASS'] else 'FAIL', 'removed', removed)
    return res


def _delete_weights(root):
    n = 0
    if root.exists():
        for f in list(root.rglob('*.pt')) + list(root.rglob('*.npz')):
            f.unlink(); n += 1
    return n


def cleanup():
    n = _delete_weights(SCR / 'fits') + _delete_weights(SCR / 'tiny')
    for f in SCR.glob('*.pt'):
        f.unlink(); n += 1
    log('removed', n, 'weight / generated files; kept results/, configs, receipts, logs')


def main(argv):
    what = argv[1] if len(argv) > 1 else 'all'
    steps = {'unit': unit, 'neutral': neutral, 'separability': separability, 'fits': fits, 'gpu': gpu, 'cleanup': cleanup}
    todo = ['unit', 'neutral', 'separability', 'fits'] if what == 'all' else [what]
    for s in todo:
        log('==', s)
        steps[s]()


if __name__ == '__main__':
    main(sys.argv)
