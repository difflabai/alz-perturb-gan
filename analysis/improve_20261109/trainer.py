"""improve_20261109 trainer: released ParticleGAN 0.9.7 stage-conditioned donor-bag model.

CLI::

    python3 trainer.py --cache DIR --arm base|flag[,flag...] --seed N \
        --out-root /ml2/alz/runs/improve_20261109/fits --updates 20000 \
        --device cpu|cuda [--threads 6] [--gpu-mem-fraction F]

Output: {out-root}/{arm}/{panel}/fold{k}/seed{s}/ (PROTOCOL.md Interface 2):
config.json, learning_curve.jsonl, progress.json, recovery.pt (every 2000
updates, removed after completion), final.pt, fit_receipt.json, generated.npz
(written by generate.py after training).

Released sources reused BY IMPORT (unchanged): release_env.activate (pinned
0.9.7 wheel), released_update.resolved_recipe (recipe 'bispf_optimized', 512
particles, z_dim 8, batch 32 x 128), released_update.bag_features,
released_update.finite / gradnorm, released_adapter.passive_particle_diagnostics,
particlegan.UpdatePolicy / bind_thermal_source / serial_autograd /
constraint_geometry_backward, pg_model.CountGenerator / standardize_counts /
critic_features, critic.VectorCritic, usage.Usage.

Differences from analysis/faithful097/model/released_adapter.py + released_update.py
(every one is listed; the per-update sequence of policy / optimizer / penalty /
loss / prior-regulariser calls in ``advance`` is the released one, line for line):

1. Data layout: reads an Interface-1 cache (train/counts.npz float32 CSR,
   train/metadata.npz, space.npz incl. ``genes``; no type_classifier.pt)
   instead of the legacy layout via contracts.checked_cache/bench_core.load_space.
   The benchmark fold contract and legacy SHA pins are replaced by validation
   against improve_20261109/folds.json (all donors must be development donors,
   none sealed; train/heldout must equal inner_folds[k] unless the explicit
   developer flag --allow-nonprotocol-split is given, which is recorded).
2. Device: CPU fitting is allowed (released fit() refuses non-CUDA).
3. Hyperparameters: candidate t005 (generator_lr .012, critic_lr .018,
   particle_lr .06, reg_coeff 1, reg_kappa 1, prior_reg .01) are passed
   explicitly to released_update.resolved_recipe; the released module globals
   HYPER/CONFIG are not mutated (CONFIG must be the released default).
4. Training streams: the same 11 named streams derived by the same
   SeedSequence([root, fold, seed]).spawn rule, but root = 20261109 (a fresh
   namespace for this study; released root 20261103) and seed is any int >= 0
   (released: 0..4).  ``build(..., stream_root=20261103)`` reproduces the
   released streams exactly (used by the parity test).
5. advance(): copied because the critic observation is arm-dependent.  For the
   base arm ``fake`` computes bag_features(critic_features(counts,rate,mean,sd)@W)
   exactly as released and ``ctx.batch`` is the released uniform-donor batch.
   Changed lines: (a) features come from arms.Features (real/fake), (b) the
   real-feature shape assertion uses the arm width instead of N_FEATURES=150,
   (c) the step-0 gradient assertion additionally covers G.stage_offset
   (offset arm) and, after the critic backward, the critic input columns of
   the added block (genemean/donorpc).  No RNG is consumed by these checks.
6. Arms (see arms.py): offset -> OffsetCountGenerator (CountGenerator subclass,
   same constructor RNG draws); genemean/donorpc -> arms.WideVectorCritic
   (VectorCritic layers with a wider input); covars -> 3 extra condition slots.
7. Checkpointing: recovery.pt every 2000 updates (released 1000) and on
   SIGINT/SIGTERM/exception at an update boundary (as released); no diagnostic
   checkpoint05000/10000 files (disk); recovery.pt is deleted after final.pt
   is written.  The snapshot is the released snapshot plus study fields
   (arm, flags, panel, arm state, accumulated wall seconds).  The endpoint is
   --updates (released fixed 20000); the recipe keeps total_steps=20000 and
   execution_limit=--updates is passed to begin_step (an external cap only).
8. learning_curve.jsonl: same rows as released (step % 250 in (0,1), released
   fields + particle_usage) plus the final step when it is not on that grid;
   one added field ``wall_seconds``.  On resume, rows past the checkpoint are
   dropped so the curve has no duplicate steps.
9. No StageLaw / allocation clock / guarded GPU admission.
"""
from __future__ import annotations

import os
import sys

sys.dont_write_bytecode = True  # never write __pycache__ into the read-only released/benchmark source dirs


def _early_threads(argv):
    for i, a in enumerate(argv):
        if a == '--threads' and i + 1 < len(argv):
            return argv[i + 1]
        if a.startswith('--threads='):
            return a.split('=', 1)[1]
    return '6'


if __name__ == '__main__':
    _t = _early_threads(sys.argv)
    for _k in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
        os.environ[_k] = _t

import argparse
import copy
import hashlib
import json
import resource
import signal
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
MODEL = Path('/ml2/alz/analysis/faithful097/model')
SCIENCE = Path('/ml2/codex-storage/future-workspaces/2026-11-03/task-2/alz-audit/core-comparison-20261103')
BENCH = Path('/ml2/codex-storage/future-workspaces/2026-10-27/task-2/benchmark')
for _p in (BENCH, SCIENCE, MODEL):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
if str(HERE) not in sys.path:
    sys.path.append(str(HERE))

import numpy as np
import scipy.sparse as sp
import torch

from release_env import activate, VERSION, PENALTY_OPTIONS
activate()
from particlegan import UpdatePolicy, serial_autograd
from particlegan.optim.constraint_geometry import constraint_geometry_backward
from particlegan.optim.dualnorm import NormalizedOptimizer
from pg_model import CountGenerator, standardize_counts, critic_features
from critic import VectorCritic, N_FEATURES
from usage import Usage
import released_update
from released_update import resolved_recipe, finite, gradnorm, CONFIG as RELEASED_CONFIG
from released_adapter import passive_particle_diagnostics, STREAM_NAMES

import arms
from arms import parse_arm, arm_name, build_conditions, stage_of, Features, WideVectorCritic

STUDY = 'improve_20261109'
FOLDS_JSON = HERE / 'folds.json'
RUNS = Path('/ml2/alz/runs/improve_20261109')
T005 = {'generator_lr': .012, 'critic_lr': .018, 'particle_lr': .06, 'reg_coeff': 1., 'reg_kappa': 1., 'prior_reg': .01}
STREAM_ROOT = 20261109
RELEASED_STREAM_ROOT = 20261103
RECIPE_STEPS = 20000
CHECKPOINT_EVERY = 2000
BAG = 128
BATCH = 32
RELEASED_DEFAULT_CONFIG = {'z_dim': 8, 'stage_balance': False, 'cell_z_dim': 8, 'theta_init_scale': 1., 'fourier_feature_scale': 1.}


def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def dump(p, obj):
    p = Path(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name('.' + p.name + '.tmp')
    tmp.write_text(json.dumps(obj, indent=2, allow_nan=False) + '\n')
    tmp.replace(p)


def training_streams(fold, seed, root=STREAM_ROOT):
    """released_adapter.training_streams with a study root (see docstring item 4)."""
    if type(fold) is not int or fold < 0 or type(seed) is not int or seed < 0:
        raise ValueError('nonnegative int fold/seed required')
    children = np.random.SeedSequence([root, fold, seed]).spawn(len(STREAM_NAMES))
    return {k: int(c.generate_state(1, dtype=np.uint64)[0]) % (2 ** 63 - 16) for k, c in zip(STREAM_NAMES, children)}


def load_folds():
    return json.loads(FOLDS_JSON.read_text())


def validate_split(manifest, train_ids, heldout_ids, allow_nonprotocol):
    q = load_folds()
    dev = set(q['development_donors'])
    sealed = set(q['sealed_evaluation_donors_fold0'])
    every = list(train_ids) + list(heldout_ids)
    if not set(every) <= dev or set(every) & sealed:
        raise ValueError('cache contains donors outside the 37 development donors')
    if set(train_ids) & set(heldout_ids):
        raise ValueError('train/heldout overlap')
    if allow_nonprotocol:
        return False
    k = int(manifest['inner_fold'])
    f = q['inner_folds'][k]
    if f['inner_fold'] != k or set(train_ids) != set(f['train_donors']) or (heldout_ids and set(heldout_ids) != set(f['heldout_donors'])):
        raise ValueError('cache donors differ from folds.json inner fold %d' % k)
    return True


def load_counts(path):
    counts = sp.load_npz(path).tocsr().astype(np.float32)
    d = counts.data
    if not np.isfinite(d).all() or (d < 0).any() or not np.array_equal(d, np.floor(d)):
        raise ValueError('counts must be finite nonnegative integers: ' + str(path))
    return counts


class Context:
    """Interface-1 cache context (released Context semantics, own loader)."""

    def __init__(self, cache, device, flags, *, allow_nonprotocol_split=False):
        cache = Path(cache)
        self.cache = cache
        self.flags = tuple(flags)
        self.device = torch.device(device)
        self.manifest = json.loads((cache / 'manifest.json').read_text())
        self.panel = str(self.manifest['panel'])
        self.fold = int(self.manifest['inner_fold'])
        m = np.load(cache / 'train/metadata.npz', allow_pickle=False)
        self.fit_labels = m['donor'].astype(str)
        self.ids = m['donors'].astype(str)
        self.raw_c = m['raw_c'].astype(np.float64)
        if self.raw_c.shape != (len(self.ids), 7):
            raise ValueError('raw_c shape')
        if list(self.ids) != list(self.manifest['train_donors']):
            raise ValueError('ordered train donors differ from manifest')
        if set(self.fit_labels) != set(self.ids):
            raise ValueError('per-cell donors differ from donor list')
        self.heldout_ids = [str(x) for x in self.manifest.get('heldout_donors', [])]
        self.protocol_split = validate_split(self.manifest, self.ids, self.heldout_ids, allow_nonprotocol_split)
        self.fitc = self.raw_c.astype(np.float32)
        with np.load(cache / 'space.npz', allow_pickle=False) as s:
            space = {k: s[k] for k in s.files}
        self.space = space
        self.genes = space['genes'].astype(str) if 'genes' in space else m['genes'].astype(str)
        if 'genes' in m.files and not np.array_equal(m['genes'].astype(str), self.genes):
            raise ValueError('metadata genes differ from space genes')
        counts = load_counts(cache / 'train/counts.npz')
        self.n_genes = counts.shape[1]
        if counts.shape != (len(self.fit_labels), len(self.genes)) or space['W'].shape != (self.n_genes, 50):
            raise ValueError('training count / space shape invalid')
        dev = self.device
        self.X = torch.as_tensor(counts.toarray(), device=dev)
        del counts
        self.by = [np.where(self.fit_labels == d)[0] for d in self.ids]
        # Train-donor age standardisation (ddof=0, bench_core.fit_space convention).
        age = self.raw_c[:, 6]
        self.age_mean, self.age_sd = float(age.mean()), float(age.std())
        self.space_age = (float(space['age_mean']), float(space['age_sd'])) if 'age_mean' in space else None
        self.conditions = torch.as_tensor(build_conditions(self.raw_c, self.flags, self.age_mean, self.age_sd), device=dev)
        self.mean = torch.as_tensor(space['mean'], device=dev)
        self.sd = torch.as_tensor(space['sd'], device=dev)
        self.W = torch.as_tensor(space['W'] / space['pc_sd'], device=dev)
        self.trainpcs = self.project(self.X)
        self.fitlibs = self.X.sum(-1)
        if not bool((self.fitlibs > 0).all()):
            raise ValueError('zero-panel-count training cells present')
        self.mu = self.X.mean(0)
        v = self.X.var(0, unbiased=False)
        self.theta = (self.mu.square() / (v - self.mu).clamp_min(1e-4)).clamp(.1, 10000)
        self.Z = standardize_counts(self.X, self.mean, self.sd) if ('genemean' in self.flags or 'donorpc' in self.flags) else None
        self.features = Features(self.flags, self)
        self.offset_table = self.offset_info = None
        if 'offset' in self.flags:
            self.offset_table, self.offset_info = arms.offset_init(self.X, self.by, stage_of(self.raw_c))

    def project(self, x):
        return standardize_counts(x, self.mean, self.sd) @ self.W

    def batch(self, rng):
        """Released uniform-donor batch; critic observation from the arm."""
        ds = rng.integers(len(self.ids), size=BATCH)
        idx = torch.as_tensor(np.stack([rng.choice(self.by[i], BAG, replace=True) for i in ds]), device=self.device)
        return self.features.real(idx), self.conditions[torch.as_tensor(ds, device=self.device)], self.X[idx].sum(-1)

    def arm_state(self):
        q = {'features': self.features.state(), 'features_info': self.features.info}
        if self.offset_table is not None:
            q['offset_init'] = torch.as_tensor(self.offset_table).clone()
            q['offset_info'] = self.offset_info
        return q


def recipe():
    if RELEASED_CONFIG != RELEASED_DEFAULT_CONFIG:
        raise ValueError('released CONFIG must be the default PG8 configuration')
    return resolved_recipe(T005)


def build(ctx, seed, *, stream_root=STREAM_ROOT):
    """released_adapter.build with arm-dependent G/D classes (same call order)."""
    device = ctx.device
    streams = training_streams(ctx.fold, seed, stream_root)
    rec = recipe()
    torch.manual_seed(streams['generator_init'])
    if 'offset' in ctx.flags:
        G = arms.make_offset_generator_class()(ctx.mu.cpu(), (ctx.theta * 1.0).cpu(), ctx.offset_table, zd=rec.z_dim, zc=8).to(device)
    else:
        G = CountGenerator(ctx.mu.cpu(), (ctx.theta * 1.0).cpu(), zd=rec.z_dim, zc=8).to(device)
    torch.manual_seed(streams['critic_init'])
    nf = ctx.features.n_features
    D = (VectorCritic(N_FEATURES, h=128) if nf == N_FEATURES else WideVectorCritic(nf, h=128)).to(device)
    P = rec.make_prior(generator=torch.Generator().manual_seed(streams['prior_init'])).to(device)
    og, od = rec.make_optimizers(G, D, P)
    pen = rec.make_critic_penalty(od, **PENALTY_OPTIONS)
    loss = rec.make_loss()
    reg = rec.make_prior_regularizer(prior=P, reference=P, weight=1.)
    from particlegan.bispf import bind_thermal_source
    bind_thermal_source(od.record, loss, reg)
    policy = UpdatePolicy(rec, G, D, prior=P, generator_optimizer=og, critic_optimizer=od, penalty=pen, seed=seed,
                          row_semantics='conditional',
                          streams={name: torch.Generator(device=device).manual_seed(streams['policy_' + key]) for name, key in
                                   [('latent_generator', 'latent'), ('penalty_generator', 'penalty'), ('eval_generator', 'eval'), ('noise_generator', 'noise')]})
    if not (rec.name == 'bispf_optimized' and rec.loss == 'non_saturating' and rec.num_particles == 512 and rec.z_dim == 8
            and rec.ema_decay == rec.input_noise_std == rec.output_noise_std == 0 and rec.network_lr_horizon_cap is None
            and rec.lr_floor == rec.network_lr_floor == 1 and isinstance(og, NormalizedOptimizer) and isinstance(od, NormalizedOptimizer)):
        raise ValueError('canonical PG recipe changed')
    generators = {k: torch.Generator(device=device).manual_seed(streams[k]) for k in ['cell', 'count']}
    rng = np.random.default_rng(streams['donor_sampler'])
    torch.manual_seed(streams['gamma_global'])
    return (rec, G, D, P, og, od, pen, loss, reg, policy), generators, rng, Usage(P.z), streams


def advance(ctx, bundle, generators, rng, usage, *, execution_limit):
    """released_update.advance with the arm critic observation (docstring item 5)."""
    rec, G, D, P, og, od, pen, loss, reg, policy = bundle
    step = policy.completed_steps
    feats = ctx.features

    def fake(z, c, lib):
        rate = G.rates(z, c, BAG, lib, generators['cell'])
        counts = torch.poisson(rate.detach(), generator=generators['count'])
        return feats.fake(critic_features(counts, rate, ctx.mean, ctx.sd))
    with serial_autograd():
        real, c, lib = ctx.batch(rng)
        assert real.shape == (BATCH, feats.n_features)
        noise = policy.begin_step(real, execution_limit=execution_limit)
        assert noise.input_sigma == 0 and float(noise.output_sigma) == 0
        assert [x['lr'] for x in og.param_groups] == [rec.lr, rec.lr * rec.prior_lr_mult]
        assert [x['lr'] for x in od.param_groups] == [rec.lr * rec.d_lr_mult]
        D.train(); G.eval()
        with torch.no_grad():
            z, _ = P.sample(BATCH, generator=policy.latent_generator)
            policy.observe_support(z)
            f = fake(z, c, lib)
        assert f.shape == real.shape
        policy.observe_critic_pair(real, f)
        dr, df = D(real, c), D(f, c); adv = loss.d_loss(dr, df); pen.collect_stats = step % 250 == 0
        penalty = pen(D, real, f, c); ld = adv + penalty
        od.zero_grad(); ld.backward(); finite(D, 'D gradients', True)
        if step == 0 and feats.n_extra:
            assert bool(D.phi[0].weight.grad[:, 150:feats.n_features].abs().sum() > 0), 'no critic gradient on added block'
        dn = gradnorm(D) if step % 250 == 0 else None
        od.step(); finite(D, 'D parameters'); policy.after_critic_step()
        D.eval(); G.train(); D.requires_grad_(False)
        real2, c2, lib2 = ctx.batch(rng); z, idx = P.sample(BATCH, generator=policy.latent_generator)
        f = fake(z, c2, lib2); lg = loss.g_loss(D(f, c2))
        raw = reg(P.z if rec.num_particles <= 1024 else P.z[torch.unique(idx)]); total = lg + rec.prior_reg * raw
        og.zero_grad(); policy.observe_sampled_rows(idx)
        constraint_geometry_backward(total, og, (lg,))
        finite(G, 'G gradients', True); finite(P, 'prior gradients', True)
        if step == 0:
            must = [G.out[-1].weight, G.log_theta, P.z] + ([G.stage_offset] if hasattr(G, 'stage_offset') else [])
            assert all(p.grad is not None and bool(p.grad.abs().sum() > 0) for p in must)
        gn = gradnorm(G) if step % 250 == 0 else None; pn = gradnorm(P) if step % 250 == 0 else None
        policy.after_generator_backward(loss_gan=lg.detach(), loss_critic=adv.detach())
        before_prior = P.z.detach().clone()
        og.step(); finite(G, 'G parameters'); finite(P, 'prior parameters')
        policy.after_generator_step(); D.requires_grad_(True); policy.finish_step()
        usage.observe(P.z, before_prior, idx)
    assert policy.completed_steps == step + 1
    return {'step': policy.completed_steps, 'D_adversarial': float(adv.detach()), 'D_penalty': float(penalty.detach()), 'D_total': float(ld.detach()),
            'G_adversarial': float(lg.detach()), 'prior_regularization_unweighted': float(raw.detach()), 'G_total': float(total.detach()),
            'G_prior_LRs': [g['lr'] for g in og.param_groups], 'D_LRs': [g['lr'] for g in od.param_groups],
            'critic_gap': float((dr - df).mean().detach()), 'D_gradient_norm': dn, 'G_gradient_norm': gn, 'prior_gradient_norm': pn,
            'penalty_stats': pen.last_stats, 'penalty_diagnostics': pen.diagnostics()}


def snapshot(ctx, bundle, generators, rng, usage, seed, streams, status, *, endpoint, stream_root, wall_seconds):
    rec, G, D, P, og, od, pen, loss, reg, policy = bundle
    if policy._phase != 'ready':
        raise ValueError('no checkpoint at a partial optimizer-pair boundary')
    q = {'schema': 1, 'method': 'pg', 'study': STUDY, 'arm': arm_name(ctx.flags), 'flags': list(ctx.flags), 'panel': ctx.panel,
         'fold': ctx.fold, 'seed': seed, 'cache_manifest_SHA256': sha(ctx.cache / 'manifest.json'),
         'recipe': rec.to_dict(), 'policy': policy.state_dict(), 'prior_regularizer': rec.prior_regularizer_checkpoint(reg),
         'generators': {k: g.get_state() for k, g in generators.items()}, 'numpy_rng': rng.bit_generator.state,
         'usage': usage.state_dict(), 'training_streams': streams, 'stream_root': stream_root,
         'torch_cpu_rng': torch.get_rng_state(),
         'torch_device_rng': torch.cuda.get_rng_state(ctx.device) if ctx.device.type == 'cuda' else None,
         'completed_updates': policy.completed_steps, 'fixed_endpoint': endpoint, 'status': status,
         'particlegan_version': VERSION, 'penalty_options': dict(PENALTY_OPTIONS), 'variant_config': dict(RELEASED_CONFIG),
         'native_recipe': 'bispf_optimized', 'released_hyperparameters': dict(T005), 'arm_state': ctx.arm_state(),
         'wall_seconds': wall_seconds}
    return copy.deepcopy(q)


def _same(a, b):
    if isinstance(a, torch.Tensor) or isinstance(b, torch.Tensor):
        # snapshots are loaded on CPU while the live arm state sits on the fit device
        return torch.equal(torch.as_tensor(a).cpu(), torch.as_tensor(b).cpu())
    if isinstance(a, dict) and isinstance(b, dict):
        return set(a) == set(b) and all(_same(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        return len(a) == len(b) and all(_same(x, y) for x, y in zip(a, b))
    return a == b


def restore(ctx, q, seed=None):
    """released_adapter.restore for this study's snapshot."""
    if q.get('particlegan_version') != VERSION or q.get('penalty_options') != PENALTY_OPTIONS:
        raise ValueError('native release or input domain checkpoint differs')
    if q.get('native_recipe') != 'bispf_optimized' or q.get('released_hyperparameters') != T005:
        raise ValueError('native recipe or six hyperparameters differ')
    if q.get('status') not in ('incomplete', 'completed'):
        raise ValueError('unrecognized checkpoint status')
    if q['policy']['completed_steps'] != q['completed_updates']:
        raise ValueError('native update clock differs')
    if q['variant_config'] != RELEASED_CONFIG:
        raise ValueError('variant configuration differs')
    if q['cache_manifest_SHA256'] != sha(ctx.cache / 'manifest.json') or q['fold'] != ctx.fold or q['panel'] != ctx.panel:
        raise ValueError('checkpoint fold/panel/cache mismatch')
    if tuple(q['flags']) != ctx.flags or (seed is not None and q['seed'] != seed):
        raise ValueError('checkpoint arm/seed mismatch')
    if not _same(q['arm_state'], ctx.arm_state()):
        raise ValueError('arm state recomputed from the cache differs from the checkpoint')
    b, g, r, u, seeds = build(ctx, q['seed'], stream_root=q['stream_root'])
    rec = b[0]; policy = b[-1]
    if q['recipe'] != rec.to_dict() or q['training_streams'] != seeds:
        raise ValueError('checkpoint recipe/seeds differ')
    policy.load_state_dict(q['policy']); rec.restore_prior_regularizer(b[-2], q['prior_regularizer'].get('prior_regularizer'))
    for k, v in q['generators'].items():
        g[k].set_state(v.cpu())
    r.bit_generator.state = q['numpy_rng']
    for k in ['draws', 'selected_steps', 'moved_steps', 'unsampled_moved_steps']:
        setattr(u, k, q['usage'][k].to(ctx.device).clone())
    u.steps = q['usage']['steps']; u.unique_batch_histogram = list(q['usage']['unique_batch_histogram'])
    torch.set_rng_state(q['torch_cpu_rng'].cpu())
    if ctx.device.type == 'cuda':
        torch.cuda.set_rng_state(q['torch_device_rng'].cpu(), ctx.device)
    return b, g, r, u, seeds


def fit_dir(out_root, flags, ctx, seed):
    return Path(out_root) / arm_name(flags) / ctx.panel / f'fold{ctx.fold}' / f'seed{seed}'


def make_config(ctx, args, stream_root):
    return {'study': STUDY, 'arm': arm_name(ctx.flags), 'flags': list(ctx.flags), 'panel': ctx.panel, 'fold': ctx.fold,
            'seed': args.seed, 'cache': str(Path(args.cache).resolve()), 'cache_manifest_SHA256': sha(ctx.cache / 'manifest.json'),
            'protocol_split': ctx.protocol_split, 'train_donors': list(ctx.ids), 'heldout_donors': ctx.heldout_ids,
            'n_genes': ctx.n_genes, 'hyperparameters': dict(T005), 'recipe': recipe().to_dict(), 'native_recipe': 'bispf_optimized',
            'particlegan_version': VERSION, 'updates': args.updates, 'recipe_total_steps': RECIPE_STEPS, 'device': args.device,
            'stream_root': stream_root, 'batch': [BATCH, BAG], 'checkpoint_every': args.checkpoint_every,
            'critic_features': ctx.features.n_features, 'arm_info': {**ctx.features.info, **({'offset': ctx.offset_info} if ctx.offset_info else {}),
                                                                     'age_mean_train': ctx.age_mean, 'age_sd_train': ctx.age_sd,
                                                                     'age_space': ctx.space_age}}


_CONFIG_MUTABLE = {'cache', 'checkpoint_every'}


def peak_memory():
    q = {'max_rss_MB': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024}
    if torch.cuda.is_available() and torch.cuda.is_initialized():
        q['cuda_max_allocated_MB'] = torch.cuda.max_memory_allocated() / 2 ** 20
    return q


def run(args):
    flags = parse_arm(args.arm)
    torch.set_num_threads(args.threads)
    if args.device == 'cuda' and args.gpu_mem_fraction:
        torch.cuda.set_per_process_memory_fraction(args.gpu_mem_fraction)
    stream_root = args.stream_root
    ctx = Context(args.cache, args.device, flags, allow_nonprotocol_split=args.allow_nonprotocol_split)
    out = fit_dir(args.out_root, flags, ctx, args.seed)
    config = json.loads(json.dumps(make_config(ctx, args, stream_root), allow_nan=False))  # JSON-normalised for resume comparison
    import generate
    if (out / 'final.pt').exists() and (out / 'generated.npz').exists():
        raise FileExistsError(f'completed fit exists, refusing to overwrite: {out}')
    # The config check runs before every path that reuses existing outputs, including generation only.
    if (out / 'config.json').exists():
        old = json.loads((out / 'config.json').read_text())
        diff = [k for k in set(old) | set(config) if k not in _CONFIG_MUTABLE and old.get(k) != config.get(k)]
        if diff:
            raise ValueError(f'existing config.json differs in {sorted(diff)}; refusing to mix fits in {out}')
    elif (out / 'final.pt').exists():
        raise ValueError(f'final.pt without config.json in {out}; refusing to generate from an unverified fit')
    if (out / 'final.pt').exists():
        q = torch.load(out / 'final.pt', map_location='cpu', weights_only=False)
        bad = {k: (q.get(k), want) for k, want in [('fixed_endpoint', args.updates), ('completed_updates', args.updates),
                                                    ('status', 'completed'), ('seed', args.seed), ('flags', list(flags))]
               if q.get(k) != want}
        del q
        if bad:
            raise ValueError(f'final.pt in {out} does not match the request {bad}; refusing generation only')
        print(f'[trainer] final.pt present without generated.npz; generation only: {out}', flush=True)
        generate.generate_fit(out, args.cache, device=args.device, ctx=ctx)
        return out
    out.mkdir(parents=True, exist_ok=True)
    if not (out / 'config.json').exists():
        dump(out / 'config.json', config)
    curve = out / 'learning_curve.jsonl'
    resumed_from = None
    wall_prev = 0.
    if (out / 'recovery.pt').exists():
        q = torch.load(out / 'recovery.pt', map_location=ctx.device, weights_only=False)
        if q['fixed_endpoint'] != args.updates:
            raise ValueError('recovery checkpoint endpoint differs from --updates')
        b, g, r, u, streams = restore(ctx, q, args.seed)
        resumed_from = q['completed_updates']; wall_prev = float(q['wall_seconds'])
        del q
        rows = [l for l in (curve.read_text().splitlines() if curve.exists() else []) if l.strip() and json.loads(l)['step'] <= resumed_from]
        curve.write_text(''.join(l + '\n' for l in rows))
        print(f'[trainer] resumed {out} at update {resumed_from}', flush=True)
    else:
        if curve.exists():
            curve.unlink()
        b, g, r, u, streams = build(ctx, args.seed, stream_root=stream_root)
    policy = b[-1]
    stops = []
    handlers = {s: signal.signal(s, lambda sig, frame: stops.append(sig)) for s in [signal.SIGINT, signal.SIGTERM]}
    started = time.monotonic()

    def wall():
        return wall_prev + time.monotonic() - started

    def save(status):
        p = out / 'final.pt' if status == 'completed' else out / 'recovery.pt'
        tmp = p.with_suffix('.partial.pt')
        torch.save(snapshot(ctx, b, g, r, u, args.seed, streams, status, endpoint=args.updates, stream_root=stream_root, wall_seconds=wall()), tmp)
        tmp.replace(p)
        dump(out / 'checkpoint_receipt.json', {'path': p.name, 'SHA256': sha(p), 'status': status, 'completed_updates': policy.completed_steps})
        return p
    first = policy.completed_steps
    try:
        while policy.completed_steps < args.updates:
            if stops:
                raise InterruptedError('stop signal received; checkpoint at completed update')
            if args.stop_after is not None and policy.completed_steps >= args.stop_after:
                raise InterruptedError('--stop-after reached (developer interruption test)')
            metrics = advance(ctx, b, g, r, u, execution_limit=args.updates)
            if metrics['step'] % 250 in (0, 1) or metrics['step'] == args.updates:
                metrics['particle_usage'] = passive_particle_diagnostics(b, u)
                metrics['wall_seconds'] = wall()
                dump(out / 'progress.json', metrics)
                with curve.open('a') as stream:
                    stream.write(json.dumps(metrics, allow_nan=False) + '\n')
            if metrics['step'] % args.checkpoint_every == 0 and metrics['step'] < args.updates:
                save('incomplete')
        train_seconds = wall()
        save('completed')
        if (out / 'recovery.pt').exists():
            (out / 'recovery.pt').unlink()
    except BaseException as exc:
        if policy._phase == 'ready' and policy.completed_steps > 0:
            save('incomplete')
        dump(out / 'failure.json', {'error': repr(exc), 'completed_updates': policy.completed_steps, 'phase': policy._phase})
        raise
    finally:
        for s, h in handlers.items():
            signal.signal(s, h)
    if (out / 'failure.json').exists():
        (out / 'failure.json').unlink()
    new = policy.completed_steps - first
    receipt = {'status': 'training_completed', 'completed_updates': policy.completed_steps, 'endpoint': args.updates,
               'full_20k_endpoint': args.updates == RECIPE_STEPS, 'wall_seconds': train_seconds,
               'seconds_per_update_this_session': (time.monotonic() - started) / max(new, 1), 'updates_this_session': new,
               'resumed_from': resumed_from, 'device': args.device, 'threads': args.threads, 'peak_memory': peak_memory(),
               'final_SHA256': sha(out / 'final.pt')}
    dump(out / 'fit_receipt.json', receipt)
    del b
    generate.generate_fit(out, args.cache, device=args.device, ctx=ctx)
    return out


def parser():
    p = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    p.add_argument('--cache', required=True)
    p.add_argument('--arm', default='base')
    p.add_argument('--seed', type=int, required=True)
    p.add_argument('--out-root', default=str(RUNS / 'fits'))
    p.add_argument('--updates', type=int, default=RECIPE_STEPS)
    p.add_argument('--device', choices=['cpu', 'cuda'], default='cpu')
    p.add_argument('--threads', type=int, default=6)
    p.add_argument('--gpu-mem-fraction', type=float, default=None)
    p.add_argument('--checkpoint-every', type=int, default=CHECKPOINT_EVERY, help=argparse.SUPPRESS)
    p.add_argument('--stop-after', type=int, default=None, help=argparse.SUPPRESS)
    p.add_argument('--stream-root', type=int, default=STREAM_ROOT, help=argparse.SUPPRESS)
    p.add_argument('--allow-nonprotocol-split', action='store_true', help='developer caches only (recorded in config)')
    return p


def main(argv=None):
    args = parser().parse_args(argv)
    if not 1 <= args.updates <= RECIPE_STEPS:
        raise ValueError('updates must be in 1..20000')
    if args.device == 'cpu':
        os.environ.setdefault('CUDA_VISIBLE_DEVICES', '')
    out = run(args)
    print(f'[trainer] done: {out}', flush=True)


if __name__ == '__main__':
    sys.modules.setdefault('trainer', sys.modules['__main__'])
    main()
