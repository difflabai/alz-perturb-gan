"""Tests and real-data checks for the critic-block arms ``donorpc2`` (F4) and ``pbulk`` (F5).

CPU only, 3 threads per process, development donors only (fold caches;
held-out development-donor counts are read ONLY by ``separability``, for
evaluation, never to fit anything).  Run from any directory::

    python3 -B r2/test_arm_critic.py unit          # parsing, forward/gradient checks, state recompute (incl. 1-thread subprocess)
    python3 -B r2/test_arm_critic.py shares        # F4: variance share of every block, real train bags vs fake bags of round-1 fits
    python3 -B r2/test_arm_critic.py separability  # F5: real vs stored generated.npz bags along pbulk vs mean-log features
    python3 -B r2/test_arm_critic.py fits          # tiny CPU fits (20 updates): donorpc2, pbulk, genemean+pbulk, donorpc2+pbulk
    python3 -B r2/test_arm_critic.py cleanup       # delete scratch fits (keep results/*.json)
    python3 -B r2/test_arm_critic.py all

Scratch / results: /ml2/alz/runs/improve_20261109/smoke_r3/r2copy_tests/critic_test/
(results/{unit,shares,separability,fits}.json are small and kept).

unit
    * flag parsing: donorpc2, donorpc2@0.1 accepted; donorpc2@0.25 (default),
      donorpc2@1, donorpc2@0, donorpc2@0.10 refused; donorpc2 with donorpc or
      genemean refused; pbulk@1, pbulk@0, pbulk@-1 refused; genemean,pbulk and
      donorpc2,pbulk accepted.
    * donorpc2 share: on the 256 reference bags, recomputed through the
      float32 torch Features path, alpha^2 V_pc/(V_150 + alpha^2 V_pc) equals
      the target within 1e-3.
    * pbulk forward: fake(counts, rate) == real(idx) bitwise when counts are
      the real counts of the same cells (any rate).
    * pbulk gradient: autograd d<u, out>/d rate equals the central finite
      difference of <u, out(counts + eps v)> in the counts direction v
      (float64; relative error < 1e-7 at eps 1e-5 and falling quadratically with eps), the gradient is the same for every
      cell of a bag, and its value matches the closed form in the arm_pbulk
      docstring.
    * generator gradient: through a freshly built genemean,pbulk (and
      donorpc2) generator, a loss on the plug-in block columns only gives a
      nonzero gradient on G.out[-1].weight and G.log_theta.
    * state recompute: arm_state of a second Context equals the first (exact)
      and equals the arm_state computed in a subprocess with 1 BLAS/torch
      thread (device / thread independence of the float64 CPU setup).

shares (F4; fold-0 hvg2000 cache; round-1 fits base, genemean, donorpc seeds 0/1)
    Real train bags: 16 bags x 128 cells per train donor (400 bags, numpy
    SeedSequence([20261109, 99, 1])); between-donor share = 1 - mean
    within-donor variance / total variance.  Fake bags: 256 bags per fit from
    the frozen final.pt generator and prior on CPU (loaded by
    _load_r1_final, which works around the GPU/CPU arm-state defect; uniform train donor,
    its condition and its real cells' libraries; 128 cells), streams fixed.
    For every bag set and block (released 150, round-1 genemean block, round-1
    whitened donorpc block, donorpc2 default, pbulk) it reports the total
    variance V_b, the pairwise share V_b / (V_150 + V_b), and for fake bags
    the standardised mean offset ||mean_fake - mean_real||^2 / V_b,real.

separability (F5; folds 0-2 hvg2000; round-1 base and genemean seeds 0/1, stored generated.npz)
    Per target donor (train and held-out): 32 real bags of 256 cells drawn
    with replacement from the donor's cells (numpy SeedSequence([20261109, 98,
    fold])) vs the 32 stored generated bags.  Features per bag: log1p CP10k of
    the bag pseudobulk (pb, what pbulk shows) and the mean per-cell log1p
    CP10k (ml, what genemean shows, up to the per-gene standardisation).
    Genes in quartiles of train-donor real mean log pseudobulk.  Per quartile:
    RMS over genes of the target-paired standardised mean difference
    d_g = mean_t(mean gen - mean real) / sd_real_g, and a donor-grouped
    5-fold cross-validated L2 logistic-regression AUC (real vs generated) on
    ml alone, pb alone, ml+pb and the per-gene residual of pb after a
    linear pb ~ ml map fitted on an independent second real draw
    (auc_pb_resid_on_ml), plus a real-vs-real null (two independent real
    draws); d_rms_pb_resid_on_ml is the RMS of d_pb after regressing it on
    d_ml over the quartile's genes.

fits
    trainer.py CLI subprocesses (CPU, 3 threads, 20 updates, snapshots
    16-19) for donorpc2, pbulk, genemean,pbulk and donorpc2,pbulk; the
    trainer's update-0 assertions check a nonzero critic gradient on each
    plug-in block and the generator gradient.  Then a separate generate.py
    process runs mode snapens on donorpc2,pbulk and donorpc2 (--threads 1) and genemean,pbulk (--threads 3)
    (recomputes the arm state from the cache in a new process with a
    different thread count and compares it exactly with final.pt and every
    snapshot).  genemean,pbulk uses 3 threads there: the core genemean_scale is
    thread-count dependent (core defect), the pbulk state is not.
"""
import os
import sys

sys.dont_write_bytecode = True
for _k in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(_k, '3')
os.environ['CUDA_VISIBLE_DEVICES'] = ''

import json
import pickle
import shutil
import subprocess
import tempfile
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) in sys.path:
    sys.path.remove(str(HERE))
sys.path.insert(0, str(HERE))

import numpy as np
import torch

torch.set_num_threads(int(os.environ.get('OMP_NUM_THREADS', '3')))
import trainer  # noqa: E402  (r2)
import arms  # noqa: E402
import generate  # noqa: E402
import arm_donorpc2  # noqa: E402
import arm_pbulk  # noqa: E402
from pg_model import critic_features  # noqa: E402

PY = '/mnt/nvme/ai/conda/anaconda3/bin/python3'
RUNS = trainer.RUNS
CACHES = RUNS / 'caches/hvg2000'
CACHE0 = CACHES / 'fold0'
FITS1 = RUNS / 'fits'
SCR = RUNS / 'smoke_r3/r2copy_tests/critic_test'
RES = SCR / 'results'
TINY = 20
ENV = dict(os.environ, CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='3', OPENBLAS_NUM_THREADS='3', MKL_NUM_THREADS='3', PYTHONDONTWRITEBYTECODE='1')


def log(*a):
    print('[test_arm_critic]', *a, flush=True)


def save(name, obj):
    trainer.dump(RES / f'{name}.json', obj)


def r(x, n=4):
    return float(round(float(x), n))


def sh(cmd, env=ENV, check=True):
    p = subprocess.run([str(c) for c in cmd], env=env, capture_output=True, text=True, cwd='/tmp')
    if check and p.returncode != 0:
        raise RuntimeError(f'command failed ({p.returncode}): {cmd}\n{p.stdout[-3000:]}\n{p.stderr[-5000:]}')
    return p


def refused(text):
    try:
        arms.parse_arm(text)
    except ValueError:
        return True
    return False


# =========================================================================== unit
def unit():
    out = {}
    ok = ['donorpc2', 'donorpc2@0.1', 'pbulk', 'pbulk@0.5', 'pbulk@2', 'genemean,pbulk', 'donorpc2,pbulk', 'genemean@0.5,pbulk@0.5']
    bad = ['donorpc2@0.25', 'donorpc2@1', 'donorpc2@0', 'donorpc2@0.10', 'donorpc,donorpc2', 'donorpc2,genemean',
           'donorpc2,genemean@0.5', 'pbulk@1', 'pbulk@0', 'pbulk@-1', 'pbulk@2.0']
    for t in ok:
        assert not refused(t), t
    for t in bad:
        assert refused(t), t
    out['parse'] = {'accepted': ok, 'refused': bad}
    log('parse ok')

    ctx = trainer.Context(CACHE0, 'cpu', ('donorpc2', 'pbulk'))
    dp, pb = ctx.plugins
    assert dp.name == 'donorpc2' and pb.name == 'pbulk'
    # ---- donorpc2 share through the torch path
    blk_pc = [b for a, b, lo, hi in ctx.features.blocks if a is dp][0]
    f150, xpc = [], []
    with torch.no_grad():
        for ds, rows in arm_donorpc2.ref_bag_rows(ctx.by, 'donorpc2'):
            idx = torch.as_tensor(rows)
            f150.append(ctx.features.bag_features(ctx.trainpcs[idx]).double().numpy())
            xpc.append(blk_pc.real(ctx, idx, ds).double().numpy())
    v150, vpc = arm_donorpc2.total_var(np.concatenate(f150)), arm_donorpc2.total_var(np.concatenate(xpc))
    share = vpc / (v150 + vpc)
    assert abs(share - dp.share) < 1e-3, (share, dp.share)
    out['donorpc2_share_torch_path'] = {'target': dp.share, 'measured': share, 'alpha': dp.alpha}
    log('donorpc2 share', share)

    # ---- pbulk forward equality and gradient
    blk_pb = [b for a, b, lo, hi in ctx.features.blocks if a is pb][0]
    rng = np.random.default_rng(5)
    rows = np.stack([rng.choice(ctx.by[i], 128, replace=True) for i in (0, 3)])
    idx = torch.as_tensor(rows)
    counts = ctx.X[idx]
    rate = (counts + torch.rand(counts.shape, generator=torch.Generator().manual_seed(1)) * 2).requires_grad_(True)
    real = blk_pb.real(ctx, idx, None)
    fake = blk_pb.fake(ctx, None, rate, counts)
    assert torch.equal(real, fake.detach()), 'pbulk fake forward != real forward on the same counts'
    # gradient vs finite difference (float64)
    c64 = counts.double()
    rate64 = (c64 + 0.7).requires_grad_(True)
    u = torch.randn(real.shape, generator=torch.Generator().manual_seed(2), dtype=torch.float64)
    L = (blk_pb.fake(ctx, None, rate64, c64) * u).sum()
    (g,) = torch.autograd.grad(L, rate64)
    assert torch.allclose(g, g[:, :1].expand_as(g)), 'gradient differs between cells of a bag'
    v = torch.randn(c64.shape, generator=torch.Generator().manual_seed(3), dtype=torch.float64)
    ad = (g * v).sum()
    errs = {}
    for eps in (1e-3, 1e-4, 1e-5):  # central difference: error must fall ~100x per 10x smaller step
        with torch.no_grad():
            fd = ((blk_pb._f(c64 + eps * v) * u).sum() - (blk_pb._f(c64 - eps * v) * u).sum()) / (2 * eps)
        errs[eps] = float((ad - fd).abs() / fd.abs())
    relerr = errs[1e-5]
    assert relerr < 1e-7 and errs[1e-4] < errs[1e-3] / 30, (float(ad), float(fd), errs)
    # closed form of the docstring
    pbs = c64.sum(1)
    N = pbs.sum(-1, keepdim=True)
    a = 1e4 / N
    w = blk_pb.arm.w.double()
    coef = u * w * a / (1 + a * pbs)            # dL/dpb_g through out_g, own-gene term
    closed = coef - (coef * pbs).sum(-1, keepdim=True) / N
    assert torch.allclose(g[:, 0], closed, rtol=1e-9, atol=1e-12), 'closed-form gradient mismatch'
    out['pbulk_gradient'] = {'autograd_dot_v': float(ad), 'finite_difference_eps1e-5': float(fd), 'relative_error_by_eps': {str(k): v for k, v in errs.items()},
                             'same_for_every_cell': True, 'closed_form_match': True}
    log('pbulk gradient ok', relerr)

    # ---- generator gradient through the plug-in columns (genemean,pbulk and donorpc2)
    gg = {}
    for flags in [('genemean', 'pbulk'), ('donorpc2',)]:
        c2 = trainer.Context(CACHE0, 'cpu', flags)
        bundle, gens, nrng, _, _ = trainer.build(c2, 0)
        G, P = bundle[1], bundle[3]
        ds = nrng.integers(len(c2.ids), size=8)
        idx2 = torch.as_tensor(np.stack([nrng.choice(c2.by[i], 128, replace=True) for i in ds]))
        lib = c2.X[idx2].sum(-1)
        cnd = c2.conditions[torch.as_tensor(ds)]
        z, _ = P.sample(8, generator=torch.Generator().manual_seed(4))
        rate = G.rates(z, cnd, 128, lib, gens['cell'])
        cnt = torch.poisson(rate.detach(), generator=gens['count'])
        f = c2.features.fake(critic_features(cnt, rate, c2.mean, c2.sd), rate, cnt)
        a2, b2, lo, hi = c2.features.blocks[-1]
        G.zero_grad()
        (f[:, lo:hi] * torch.randn(8, hi - lo, generator=torch.Generator().manual_seed(6))).sum().backward()
        n_out = float(G.out[-1].weight.grad.abs().sum())
        n_th = float(G.log_theta.grad.abs().sum())
        assert n_out > 0 and n_th > 0, (flags, n_out, n_th)
        gg[','.join(flags)] = {'block': f'{a2.flag}/{b2.name}', 'columns': [lo, hi], 'grad_abs_out_weight': n_out, 'grad_abs_log_theta': n_th}
    out['generator_gradient'] = gg
    log('generator gradient ok')

    # ---- state recompute: same process and a 1-thread subprocess
    ctx2 = trainer.Context(CACHE0, 'cpu', ctx.flags)
    assert trainer._same(ctx.arm_state(), ctx2.arm_state())
    with tempfile.TemporaryDirectory(dir='/tmp') as td:
        pkl = Path(td) / 'state.pt'
        code = ('import sys,torch; sys.path.insert(0,%r); import trainer; torch.set_num_threads(1); '
                'c=trainer.Context(%r,"cpu",%r); torch.save(c.arm_state(), %r)') % (str(HERE), str(CACHE0), ctx.flags, str(pkl))
        env1 = dict(ENV, OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')
        sh([PY, '-B', '-c', code], env=env1)
        st1 = torch.load(pkl, weights_only=False)
    assert trainer._same(ctx.arm_state(), st1), 'arm state differs with 1 thread'
    out['state_recompute'] = {'same_process': True, 'one_thread_subprocess': True}
    log('state recompute ok')
    save('unit', out)
    log('unit PASS')


# =========================================================================== shares (F4)
def _block_values(mctx, feats_gm, feats_dpc, cf, counts, lib_rate=None):
    """Feature blocks for a set of bags given cf [B,cells,G] and counts [B,cells,G] (no gradient)."""
    dp, pb = mctx.plugins
    blk = {b.name: b for a, b, lo, hi in mctx.features.blocks}
    gm = cf.mean(1)
    return {'f150': feats_gm.bag_features(cf @ mctx.W),
            'genemean': gm * feats_gm.scale,
            'donorpc_r1': feats_dpc._extra(gm),
            'donorpc2': blk['pc']._f(gm),
            'pbulk': blk['pb']._f(counts)}


def _load_r1_final(fctx, fit, config):
    """generate.load_models(final) for a round-1 fit; tolerate GPU-vs-CPU float differences of the round-1 block state.

    Core defect (reported): round-1 genemean final.pt files hold genemean_scale computed on CUDA
    (0.8241964702478473 on fold 0 seed 0) while the CPU recompute gives 0.8241964723596995, so
    generate.check_final's exact comparison refuses every GPU-trained genemean fit on CPU.  For this
    measurement only, the round-1 feature state (genemean_scale; donorpc pc_mu/pc_V/pc_sd, same issue) is
    accepted within 1e-5 relative max-abs difference; status, flags and seed are checked exactly.
    """
    try:
        return generate.load_models(fctx, fit, config, ['final'])['final']
    except ValueError as exc:
        if 'arm state' not in str(exc):
            raise
    q = torch.load(fit / 'final.pt', map_location='cpu', weights_only=False)
    a, b = q['arm_state']['features'], fctx.arm_state()['features']
    assert set(a) == set(b), (sorted(a), sorted(b))
    tol = {}
    for k in a:
        x, y = torch.as_tensor(a[k]).double(), torch.as_tensor(b[k]).double()
        tol[k] = float((x - y).abs().max() / y.abs().max())
        assert tol[k] < 1e-5, (k, tol[k])
    assert set(q['arm_state']) == set(fctx.arm_state())
    assert q['status'] == 'completed' and tuple(q['flags']) == fctx.flags and q['seed'] == config['seed']
    bundle, *_ = trainer.build(fctx, config['seed'], stream_root=config['stream_root'])
    G, P = bundle[1], bundle[3]
    G.load_state_dict(q['policy']['models']['generator'], strict=True)
    P.load_state_dict(q['policy']['models']['prior'], strict=True)
    return G, P


def shares():
    t0 = time.monotonic()
    mctx = trainer.Context(CACHE0, 'cpu', ('donorpc2', 'pbulk'))
    feats_gm = arms.Features(('genemean',), mctx)
    feats_dpc = arms.Features(('donorpc',), mctx)
    rng = np.random.default_rng(np.random.SeedSequence([20261109, 99, 1]))
    real = {k: [] for k in ['f150', 'genemean', 'donorpc_r1', 'donorpc2', 'pbulk']}
    donor = []
    with torch.no_grad():
        for i, rows in enumerate(mctx.by):
            idx = torch.as_tensor(np.stack([rng.choice(rows, 128, replace=True) for _ in range(16)]))
            x = mctx.X[idx]
            cf = mctx.Z[idx]
            for k, v in _block_values(mctx, feats_gm, feats_dpc, cf, x).items():
                real[k].append(v.double().numpy())
            donor += [i] * 16
    real = {k: np.concatenate(v) for k, v in real.items()}
    donor = np.array(donor)
    out = {'cache': str(CACHE0), 'real_bags': len(donor), 'genemean_scale': feats_gm.scale,
           'donorpc2_alpha': mctx.plugins[0].alpha, 'pbulk_scale': mctx.plugins[1].scale, 'real': {}, 'fake': {}}
    V150 = arm_donorpc2.total_var(real['f150'])
    for k, a in real.items():
        V = arm_donorpc2.total_var(a)
        within = float(np.mean([a[donor == i].var(0, ddof=1).sum() for i in range(len(mctx.by))]))
        out['real'][k] = {'width': a.shape[1], 'V': r(V), 'E_norm2': r((a ** 2).sum(1).mean()),
                          'share_vs_150': r(V / (V150 + V)) if k != 'f150' else None,
                          'between_donor_fraction': r(1 - within / V),
                          'between_donor_share_vs_150': r((V - within) / (V150 + V)) if k != 'f150' else None}
        if k != 'f150':  # variance in the block's leading directions, as a share of V_150 + V_b
            ev = np.linalg.svd(a - a.mean(0), compute_uv=False) ** 2 / len(a)
            out['real'][k]['top_k_direction_share_vs_150'] = {str(t): r(ev[:t].sum() / (V150 + V)) for t in (1, 2, 4, 16)}
    # all round-1 blocks together (round-1 donorpc arm: 150 + 16; genemean arm: 150 + 2000)
    log('real done', json.dumps(out['real']))
    for arm in ['base', 'genemean', 'donorpc']:
        flags = () if arm == 'base' else (arm,)
        fctx = trainer.Context(CACHE0, 'cpu', flags)
        for seed in (0, 1):
            fit = FITS1 / arm / 'hvg2000/fold0' / f'seed{seed}'
            if not (fit / 'generated.npz').exists():
                continue
            config = json.loads((fit / 'config.json').read_text())
            G, P = _load_r1_final(fctx, fit, config)
            G.eval(); P.eval()
            frng = np.random.default_rng(np.random.SeedSequence([20261109, 99, 2, seed]))
            gl, gc, gk = (torch.Generator().manual_seed(s) for s in (11 + seed, 12 + seed, 13 + seed))
            fake = {k: [] for k in real}
            with torch.no_grad():
                for _ in range(8):
                    ds = frng.integers(len(mctx.ids), size=32)
                    idx = torch.as_tensor(np.stack([frng.choice(mctx.by[i], 128, replace=True) for i in ds]))
                    lib = mctx.X[idx].sum(-1)
                    cond = fctx.conditions[torch.as_tensor(ds)]
                    z, _ = P.sample(32, generator=gl)
                    rate = G.rates(z, cond, 128, lib, gc)
                    cnt = torch.poisson(rate, generator=gk)
                    cf = critic_features(cnt, rate, mctx.mean, mctx.sd)
                    for k, v in _block_values(mctx, feats_gm, feats_dpc, cf, cnt).items():
                        fake[k].append(v.double().numpy())
            fake = {k: np.concatenate(v) for k, v in fake.items()}
            V150f = arm_donorpc2.total_var(fake['f150'])
            row = {}
            for k, a in fake.items():
                V = arm_donorpc2.total_var(a)
                off = float(((a.mean(0) - real[k].mean(0)) ** 2).sum())
                row[k] = {'V': r(V), 'share_vs_150': r(V / (V150f + V)) if k != 'f150' else None,
                          'mean_offset_over_real_V': r(off / arm_donorpc2.total_var(real[k]))}
            out['fake'][f'{arm}/seed{seed}'] = row
            log(arm, seed, json.dumps(row))
            del G, P
    out['seconds'] = time.monotonic() - t0
    save('shares', out)
    log('shares done')
    return out


# =========================================================================== separability (F5)
def _real_bag_summaries(counts, labels, donors, M, K, rng):
    """Per donor: M bags of K cells (with replacement) -> log pb [M,G], mean_log [M,G]."""
    pbs, mls = [], []
    for d in donors:
        rows = np.flatnonzero(labels == d)
        X = counts[rows].toarray().astype(np.float64)
        lib = np.maximum(X.sum(1, keepdims=True), 1)
        L = np.log1p(X * (1e4 / lib))
        pick = rng.choice(len(rows), size=(M, K), replace=True)
        pb = X[pick].sum(1)
        pbs.append(np.log1p(pb * (1e4 / np.maximum(pb.sum(-1, keepdims=True), 1))))
        mls.append(L[pick].mean(1))
    return np.stack(pbs), np.stack(mls)


def _auc_cv(Xr, Xg, groups, seed=0):
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import GroupKFold
    X = np.concatenate([Xr, Xg])
    y = np.r_[np.zeros(len(Xr)), np.ones(len(Xg))]
    g = np.r_[groups, groups]
    pred = np.zeros(len(y))
    for tr, te in GroupKFold(5).split(X, y, g):
        mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-6
        clf = LogisticRegression(C=0.1, max_iter=300)
        clf.fit((X[tr] - mu) / sd, y[tr])
        pred[te] = clf.decision_function((X[te] - mu) / sd)
    return float(roc_auc_score(y, pred))


def separability(folds=(0, 1, 2)):
    import scipy.sparse as sp
    t0 = time.monotonic()
    out = {}
    for k in folds:
        cache = CACHES / f'fold{k}'
        M, K = 32, 256
        rng = np.random.default_rng(np.random.SeedSequence([20261109, 98, k]))
        mt = np.load(cache / 'train/metadata.npz')
        mh = np.load(cache / 'heldout/metadata.npz')
        ct = sp.load_npz(cache / 'train/counts.npz').tocsr()
        ch = sp.load_npz(cache / 'heldout/counts.npz').tocsr()
        tr_ids, ho_ids = list(mt['donors'].astype(str)), list(mh['donors'].astype(str))
        pb_t, ml_t = _real_bag_summaries(ct, mt['donor'].astype(str), tr_ids, M, K, rng)
        pb_h, ml_h = _real_bag_summaries(ch, mh['donor'].astype(str), ho_ids, M, K, rng)
        pb_t2, ml_t2 = _real_bag_summaries(ct, mt['donor'].astype(str), tr_ids, M, K, rng)  # null draw (train)
        pb_h2, ml_h2 = _real_bag_summaries(ch, mh['donor'].astype(str), ho_ids, M, K, rng)
        real = {'train': (tr_ids, pb_t, ml_t, pb_t2, ml_t2), 'heldout': (ho_ids, pb_h, ml_h, pb_h2, ml_h2)}
        level = pb_t.mean((0, 1))
        qs = np.quantile(level, [0.25, 0.5, 0.75])
        quart = np.searchsorted(qs, level)  # 0..3
        sd_pb = {s: real[s][1].reshape(-1, level.size).std(0) + 1e-6 for s in real}
        sd_ml = {s: real[s][2].reshape(-1, level.size).std(0) + 1e-6 for s in real}
        fo = {'quartile_mean_log_pb_bounds': qs.tolist(), 'genes_per_quartile': np.bincount(quart).tolist(), 'fits': {}, 'null': {}}
        # null: real vs real
        for s in real:
            ids, pb1, ml1, pb2, ml2 = real[s]
            groups = np.repeat(np.arange(len(ids)), M)
            row = {}
            for q in range(4):
                gq = quart == q
                row[f'q{q + 1}'] = {
                    'd_rms_pb': r(np.sqrt(((((pb2 - pb1).mean(1)).mean(0)[gq] / sd_pb[s][gq]) ** 2).mean())),
                    'd_rms_ml': r(np.sqrt(((((ml2 - ml1).mean(1)).mean(0)[gq] / sd_ml[s][gq]) ** 2).mean())),
                    'auc_pb': r(_auc_cv(pb1[:, :, gq].reshape(-1, gq.sum()), pb2[:, :, gq].reshape(-1, gq.sum()), groups), 3)}
            fo['null'][s] = row
        log('fold', k, 'null', json.dumps(fo['null']))
        for arm in ['base', 'genemean']:
            for seed in (0, 1):
                p = FITS1 / arm / 'hvg2000' / f'fold{k}' / f'seed{seed}' / 'generated.npz'
                if not p.exists():
                    continue
                with np.load(p) as z:
                    tg = list(z['targets'].astype(str))
                    gpb = z['pb_counts'].astype(np.float64)
                    gml = z['mean_log'].astype(np.float64)
                    assert list(z['genes'].astype(str)) == list(mt['genes'].astype(str))
                gpb = np.log1p(gpb * (1e4 / np.maximum(gpb.sum(-1, keepdims=True), 1)))
                res = {}
                for s in real:
                    ids, pb1, ml1, _, _ = real[s]
                    ti = [tg.index(d) for d in ids]
                    Gpb, Gml = gpb[ti], gml[ti]
                    groups = np.repeat(np.arange(len(ids)), M)
                    dpb = (Gpb.mean(1) - pb1.mean(1)).mean(0) / sd_pb[s]
                    dml = (Gml.mean(1) - ml1.mean(1)).mean(0) / sd_ml[s]
                    row = {}
                    for q in range(4):
                        gq = quart == q
                        n = int(gq.sum())
                        Rpb, Rml = pb1[:, :, gq].reshape(-1, n), ml1[:, :, gq].reshape(-1, n)
                        Fpb, Fml = Gpb[:, :, gq].reshape(-1, n), Gml[:, :, gq].reshape(-1, n)
                        # per-gene linear map pb ~ ml fitted on the independent second real draw of this split
                        Npb, Nml = real[s][3][:, :, gq].reshape(-1, n), real[s][4][:, :, gq].reshape(-1, n)
                        mx, my = Nml.mean(0), Npb.mean(0)
                        beta = ((Nml - mx) * (Npb - my)).sum(0) / np.maximum(((Nml - mx) ** 2).sum(0), 1e-12)
                        Rres, Fres = Rpb - my - beta * (Rml - mx), Fpb - my - beta * (Fml - mx)
                        row[f'q{q + 1}'] = {
                            'auc_pb_resid_on_ml': r(_auc_cv(Rres, Fres, groups), 3),
                            'd_rms_pb': r(np.sqrt((dpb[gq] ** 2).mean())), 'd_rms_ml': r(np.sqrt((dml[gq] ** 2).mean())),
                            'corr_dpb_dml': r(np.corrcoef(dpb[gq], dml[gq])[0, 1], 3),
                            'd_rms_pb_resid_on_ml': r(np.sqrt(((dpb[gq] - dml[gq] * (dpb[gq] @ dml[gq]) / (dml[gq] @ dml[gq])) ** 2).mean())),
                            'auc_ml': r(_auc_cv(Rml, Fml, groups), 3), 'auc_pb': r(_auc_cv(Rpb, Fpb, groups), 3),
                            'auc_ml_pb': r(_auc_cv(np.c_[Rml, Rpb], np.c_[Fml, Fpb], groups), 3)}
                    row['all'] = {'d_rms_pb': r(np.sqrt((dpb ** 2).mean())), 'd_rms_ml': r(np.sqrt((dml ** 2).mean())),
                                  'corr_dpb_dml': r(np.corrcoef(dpb, dml)[0, 1], 3)}
                    res[s] = row
                fo['fits'][f'{arm}/seed{seed}'] = res
                log('fold', k, arm, seed, json.dumps(res))
        out[f'fold{k}'] = fo
    out['seconds'] = time.monotonic() - t0
    save('separability', out)
    log('separability done')
    return out


# =========================================================================== fits
FIT_ARMS = ['donorpc2', 'pbulk', 'genemean,pbulk', 'donorpc2,pbulk']


def fits():
    root = SCR / 'fits'
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
            libm = float(z['lib'].mean())
        assert fin and rec['status'] == 'completed' and not rec['snapshots_missing']
        res[arm] = {'critic_features': cfg['critic_features'], 'blocks': [(b['flag'], b['block'], b['columns']) for b in cfg['arm_info'].get('r2_blocks', [])],
                    'r2_arms': cfg['r2_arms'], 'snapshots': sorted(rec['snapshots_SHA256']), 'generated_finite': fin,
                    'mean_generated_lib': libm, 'critic_gap_curve': [(c['step'], r(c['critic_gap'])) for c in curve],
                    'seconds': time.monotonic() - t}
        log(arm, 'fit ok', res[arm]['critic_features'], res[arm]['blocks'], round(res[arm]['seconds']))
    # separate process: arm state recomputed from the cache vs final.pt and each snapshot.  Plug-in-only
    # arms with 1 thread (their state is thread independent); genemean,pbulk with the training's 3 threads,
    # because the core's round-1 genemean_scale is a float32 torch reduction that changes with the thread
    # count (0.8241964723596995 with 3 threads vs 0.8241964723556644 with 1; core defect, reported).
    env1 = dict(ENV, OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')
    for arm, th in [('donorpc2,pbulk', 1), ('donorpc2', 1), ('genemean,pbulk', 3)]:
        fit = root / arm / 'hvg2000/fold0/seed0'
        o = fit / 'generated_snapens.npz'
        if not o.exists():
            sh([PY, '-B', HERE / 'generate.py', '--fit', fit, '--cache', CACHE0, '--mode', 'snapens', '--threads', th],
               env=env1 if th == 1 else ENV)
        with np.load(o) as z:
            names = list(z['state_names'].astype(str))
            cnt = np.bincount(z['state_index'][0], minlength=len(names)).tolist()
        res[arm]['snapens_new_process'] = {'threads': th, 'states': names, 'counts_target0': cnt}
        log(arm, f'snapens ({th} thread(s), new process) ok', names, cnt)
    save('fits', res)
    log('fits PASS')
    return res


def cleanup():
    if SCR.exists():
        for p in SCR.iterdir():
            if p.name != 'results':
                shutil.rmtree(p) if p.is_dir() else p.unlink()
    log('scratch removed except results/')


def main(argv):
    what = argv[1] if len(argv) > 1 else 'all'
    steps = {'unit': unit, 'shares': shares, 'separability': separability, 'fits': fits, 'cleanup': cleanup}
    todo = ['unit', 'shares', 'separability', 'fits'] if what == 'all' else [what]
    for s in todo:
        log('==', s)
        steps[s]()


if __name__ == '__main__':
    main(sys.argv)
