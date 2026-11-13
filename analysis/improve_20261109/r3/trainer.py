"""improve_20261109 round-3 trainer: released ParticleGAN 0.9.7 stage-conditioned donor-bag model.

CLI::

    python3 -B r3/trainer.py --cache DIR --arm base|flag[@v][,flag...] --seed N \
        [--out-root /ml2/alz/runs/improve_20261109/fits_r3] [--updates U] \
        [--device cpu|cuda] [--threads 3] [--gpu-mem-fraction F]

Output: {out-root}/{arm}/{panel}/fold{k}/seed{s}/ (round-1 Interface 2 plus
round-2/3 additions): config.json, learning_curve.jsonl, progress.json,
recovery.pt (every 2000 updates, removed after completion), final.pt,
snap_{u}.pt, fit_receipt.json, generated.npz (r3/generate.py, after training).

Round-3 changes (R3-1 .. R3-8 below; with no t./g./d. flag the training
trajectory is bitwise the round-1 / round-2 one, tested in r3/test_core.py)
---------------------------------------------------------------------------

How the released adapter resolves the recipe (studied for R1/R2, 0.9.7 source):
``released_update.resolved_recipe(T005)`` calls ``get_recipe('gan',
trainer='bispf_optimized', total_steps=20000, batch_size=32,
num_particles=512, z_dim=8, lr=.012, d_lr_mult=1.5, prior_lr_mult=5,
reg_coeff=1, reg_kappa=1, prior_reg=.01)``.  The bispf_optimized preset
(recipes.get_recipe_fields: bcap_adam -> bcap -> bispf_optimized) sets
zero-momentum DualNorm (optimizer_family 'dualnorm', smoothing .001, batched
SVD, quintic convolution rule), direction_blend constraint geometry, the
non-saturating loss, the BISPF critic penalty, SBNP prior regularisation,
``lr_schedule='cosine'`` BUT ``lr_floor = network_lr_floor = 1`` and
``network_lr_horizon_cap = None`` (so the cosine is flat: CONSTANT rates),
``ema_decay = 0``, zero input/output noise, ``beta2_end = reg_coeff_end =
None``; ``lr_anneal_start`` keeps the dataclass default 0.6.

* Declared horizon of every round-1/2 fit: ``total_steps = 20000`` (the
  recipe), run length 20000 (``execution_limit`` = --updates; smoke fits
  shorter, horizon still 20000).
* Learning-rate schedule (recipes.learning_rate_scales, applied by
  ``UpdatePolicy.begin_step`` before every update from the number of
  COMPLETED updates k): G and D groups get base x network(k), the particle
  group base x prior(k), with learning_rate_scale(k, T, start, floor) =
  1 if floor == 1; else floor + (1-floor)/2 (1 + cos(pi f)),
  f = clip((k - start T)/((1 - start) T), 0, 1).  Network: T = min(total_steps,
  network_lr_horizon_cap), floor = network_lr_floor (None -> lr_floor); prior:
  T = total_steps, floor = lr_floor.  The base rates are the optimizer group
  rates at policy construction (``policy.initial_lrs``).  Other scheduled
  quantities: beta2 (``beta2_end``; plain Adam only), penalty coefficient
  (``reg_coeff_end``; fixed R1/R2/BCap reg_arm only), critic input noise
  (linear decay over input_noise_anneal_end T) and generator output noise
  (linear warm-up over output_noise_warmup T) - all inactive here (see
  arms.py "Native options that cannot be combined").
* EMA: ``UpdatePolicy`` always keeps ``ema_G`` / ``ema_prior`` (deep copies
  at construction) and in ``finish_step`` sets avg = d avg + (1-d) live for
  every parameter (copy for frozen ones, buffers copied, also
  ``averaged_table``), d = ema_decay.  With d = 0 the averages are exact
  copies of the live weights.  The averages are saved in
  ``policy.state_dict()['averages']`` (final.pt ``policy.averages``).  They
  are served (swapped into training) only when serve_average > 0, which this
  recipe cannot use; so ema_decay never changes training.  An EMA generator
  for sampling = a generator module loaded with ``averages['generator']``
  (and the prior with ``averages['prior']``).
* Beyond the horizon: well defined.  ``begin_step(execution_limit=...)``
  only refuses k >= execution_limit; learning_rate_scale clamps f at 1 so the
  rates hold at the floor; the policy checkpoint says total_steps "is not a
  checkpoint budget".  r3 nevertheless ties run length to the horizon
  (t.steps sets both) so the cosine ends where the run ends.
* Validations that constrain combinations: listed in the arms.py docstring
  (Recipe.__post_init__); plus the released-adapter facts used here: the
  prior regulariser sees the whole table only for num_particles <= 1024, and
  usage.Usage keeps a 33-bin unique-row histogram (resized here for batch >
  32, diagnostic only).

R3-1. Training options (``t.`` flags, arms.TRAIN_OPTIONS): ``recipe(topt)``
      = get_recipe with the adapter's arguments + arms.recipe_overrides; with
      no t. flag it is exactly released_update.resolved_recipe(T005).  The
      run length is topt.steps unless --updates shortens it.  config.json
      ``departures`` lists every t./g./d. flag with the recipe fields it
      changed; ``recipe`` holds the full resolved recipe.  The canonical-
      recipe assertion of round 1 is kept verbatim for default options and
      generalised (same family/loss/noise/optimizer classes, fields equal to
      the requested overrides) otherwise.  The per-update LR assertion in
      ``advance`` now checks the native schedule: G = lr*net(k), prior =
      lr*prior_lr_mult*prior(k), D = lr*d_lr_mult*net(k) (identical to round 1
      when the scale is 1).  Batch shape: ctx.batch_size donors x ctx.bag cells.
R3-2. Shadow EMA (read-out, ``ShadowEMA``): copies of the generator and the
      particle prior at decays SHADOW_EMA_DECAYS = (0.999, 0.9995, 0.9998),
      created in ``build`` (deep copies of the initial modules, requires_grad
      False, in no optimizer), updated at the end of every ``advance`` from
      the live parameters with exactly UpdatePolicy.finish_step's EMA
      arithmetic (so shadow at d equals the native t.ema@d average bitwise),
      no random draw.  Saved in recovery.pt / final.pt (``r3_shadow``) and
      restored on resume; snap files carry CPU copies (``shadow``).
R3-3. Snapshots: ``snapshot_updates(U)``: every ``dense`` updates at u >= 0.75 U
      plus every ``coarse`` updates, u < U; dense = 1000, coarse = 5000 for
      U >= 20000, else dense = max(1, round(1000 U / 20000)), coarse =
      max(1, round(5000 U / 20000)).  U = 20000 -> 5000, 10000, 15000-19000.
      snap_{u}.pt holds generator, prior, shadow EMAs and (t.ema) the native
      EMA, all CPU.
R3-4. Architecture hooks: generator ``generator_kwargs`` (c2) and the class
      chain (c); critic ``critic_module`` (d2) built under the critic_init
      seed with ``Features.layout()``; per-cell blocks.  The update-0
      gradient check covers plug-in critics through dD/dx per block.
R3-5. config.json gains ``round`` = 3, ``departures``, ``training_options``,
      ``shadow_ema_decays``, ``native_ema_decay``,
      ``default_generation_state``, ``critic_layout``, ``generator_kwargs``;
      checkpoints gain ``r3_shadow`` and ``train_options``.
R3-6. Default out root ``runs/improve_20261109/fits_r3``.  ``check_out_root``
      refuses every existing round-1/round-2 output tree (fits, fits_r2,
      results*, posthoc_r2, smoke_r2, logs*, caches, screen_analysis ...) and
      any other folder under runs/improve_20261109 except fits_r3, smoke_r3,
      results_r3, logs_r3.
R3-7. Refuse-to-overwrite, config diff check, bitwise resume (policy incl.
      schedule clock and native EMA, shadow EMA, snapshots, arm streams) as
      round 2.
R3-8. ``--updates`` defaults to the horizon (t.steps or 20000) and must not
      exceed it.

Round-2 docstring (still valid)
-------------------------------

This file is a copy of round-1 ``../trainer.py``; every round-1 difference
from the released adapter (items 1-9 of that docstring) still applies.  The
round-1 docstring items, abbreviated: (1) Interface-1 cache loader and split
validation; (2) CPU fitting allowed; (3) candidate t005 hyperparameters passed
explicitly; (4) the 11 released training streams from
SeedSequence([20261109, fold, seed]); (5) ``advance`` copied, critic
observation from arms.Features, released call sequence line for line; (6) arm
generator/critic classes; (7) recovery.pt every 2000 updates, released
snapshot fields + study fields; (8) learning_curve rows as released +
wall_seconds; (9) no StageLaw / allocation clock / GPU admission.

Round-2 changes (nothing else differs from round 1; ``base`` and round-1
flags are bitwise equal to round 1 for initial tensors and updates, tested in
test_core.py):

R1. Arms: ``r2/arms.py`` (round-1 flags copied unchanged, ``genemean@s``,
    plug-in registry).  Plug-in hooks are called here: setup and (a)
    conditions in ``Context.__init__``; (b) ``batch_condition`` in
    ``Context.batch`` with the arm's own numpy stream (created in ``build``,
    ``ctx.arm_rngs``, never touching the 11 released streams); (c) generator
    class chain + ``init_generator`` in ``build`` (released parameters are
    constructed first under torch.manual_seed(generator_init), so their
    initial tensors are unchanged); (d) feature blocks via arms.Features,
    critic width = sum of blocks (WideVectorCritic); (f) arm state in every
    checkpoint.  ``advance`` differs from round 1 only in: ``ctx.batch`` gets
    the phase ('critic' / 'generator'), ``feats.fake`` also gets ``rate`` and
    ``counts``, the step-0 gradient assertion also covers the plug-ins'
    ``generator_grad_params`` and checks each plug-in block's critic input
    columns separately.  None of these consume random numbers.
R2. Snapshots (round-2 rule, SUPERSEDED in round 3 by R3-3; round-2 fits
    keep theirs): after update u in ``snapshot_updates(U)`` (16000, 17000, 18000,
    19000 for U = 20000; for U < 20000 floor(U*f + 0.5) for f = .80, .85,
    .90, .95, dropping duplicates and values outside 1..U-1) the generator
    and prior state_dicts (CPU copies) are written atomically to
    ``snap_{u}.pt`` with fit identity fields and the arm state.  Writing a
    snapshot reads parameters only, so the training streams are untouched.  A
    resumed fit rewrites a snapshot it passes again (bitwise the same
    content); fit_receipt.json lists the snapshots and their SHA256.
R3. Splits are validated against ``r2/folds_r2.json`` when that file exists
    (inner folds 0-8; this trainer checks that its folds 0-2, development
    donors and sealed donors equal ``../folds.json``), else ``../folds.json``.
    Generation streams always use the ``../folds.json`` development-donor
    order (common random numbers identical to round 1).
R4. Default out root ``runs/improve_20261109/fits_r2``; default threads 3.
    config.json gains ``round`` = 2, ``snapshot_updates`` and ``r2_arms``
    (per plug-in flag: value and arm info); checkpoints gain ``round``,
    ``arm_streams`` and ``arm_runtime``.
R5. Repair (2026-11-10, review FID-2): ``run`` refuses an ``--out-root`` that
    is, or lies inside, a frozen round-1 tree (runs/improve_20261109/fits,
    results, results_screen48, screen_analysis, caches), checked on resolved
    paths before anything is created (``check_out_root``).  Previously only
    the config.json diff protected existing round-1 fit folders; a new
    arm/fold/seed folder under fits/ would have been written.  run_screen.sh
    applies the same check to -o.  No effect on training or streams.
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
    return '3'


if __name__ == '__main__':
    _t = _early_threads(sys.argv)
    for _k in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
        os.environ[_k] = _t

import argparse
import copy
import hashlib
import weakref
import json
import resource
import signal
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent          # .../improve_20261109/r3
STUDY_DIR = HERE.parent                         # round-1 code (read-only)
MODEL = Path('/ml2/alz/analysis/faithful097/model')
SCIENCE = Path('/ml2/codex-storage/future-workspaces/2026-11-03/task-2/alz-audit/core-comparison-20261103')
BENCH = Path('/ml2/codex-storage/future-workspaces/2026-10-27/task-2/benchmark')
for _p in (BENCH, SCIENCE, MODEL):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
# r3 first: ``arms`` / ``generate`` must be the round-3 modules even when the
# working directory is another round's folder.
if str(HERE) in sys.path:
    sys.path.remove(str(HERE))
sys.path.insert(0, str(HERE))

import numpy as np
import scipy.sparse as sp
import torch

from release_env import activate, VERSION, PENALTY_OPTIONS
activate()
from particlegan import UpdatePolicy, serial_autograd, get_recipe
from particlegan.recipes import learning_rate_scales
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

if Path(arms.__file__).resolve().parent != HERE:
    raise ImportError(f'r3 trainer imported the wrong arms module: {arms.__file__}')
for _m in ('pg_model', 'critic', 'usage', 'released_update', 'released_adapter', 'release_env'):
    if Path(sys.modules[_m].__file__).resolve().parent == HERE:
        raise ImportError(f'r3/{_m}.py shadows a released module')

STUDY = 'improve_20261109'
ROUND = 3
FOLDS_JSON_R1 = STUDY_DIR / 'folds.json'
FOLDS_JSON_R2 = HERE / 'folds_r2.json'
RUNS = Path('/ml2/alz/runs/improve_20261109')
DEFAULT_OUT_ROOT = RUNS / 'fits_r3'
SHADOW_EMA_DECAYS = arms.SHADOW_EMA_DECAYS
T005 = {'generator_lr': .012, 'critic_lr': .018, 'particle_lr': .06, 'reg_coeff': 1., 'reg_kappa': 1., 'prior_reg': .01}
STREAM_ROOT = 20261109
RELEASED_STREAM_ROOT = 20261103
RECIPE_STEPS = 20000
CHECKPOINT_EVERY = 2000
BAG = 128      # released cells per bag (t.bag changes ctx.bag)
BATCH = 32     # released donors per batch (t.batch changes ctx.batch_size)
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
    """released_adapter.training_streams with a study root (round-1 item 4)."""
    if type(fold) is not int or fold < 0 or type(seed) is not int or seed < 0:
        raise ValueError('nonnegative int fold/seed required')
    children = np.random.SeedSequence([root, fold, seed]).spawn(len(STREAM_NAMES))
    return {k: int(c.generate_state(1, dtype=np.uint64)[0]) % (2 ** 63 - 16) for k, c in zip(STREAM_NAMES, children)}


def snapshot_intervals(updates):
    """(dense, coarse) snapshot intervals for a run of ``updates`` updates (docstring R3-3)."""
    if updates >= RECIPE_STEPS:
        return 1000, 5000
    return max(1, int(np.floor(1000 * updates / RECIPE_STEPS + 0.5))), max(1, int(np.floor(5000 * updates / RECIPE_STEPS + 0.5)))


def snapshot_updates(updates):
    """Snapshot update numbers: every dense updates at u >= 0.75 U plus every coarse updates, 1 <= u < U (R3-3)."""
    dense, coarse = snapshot_intervals(updates)
    out = {u for u in range(dense, updates, dense) if 4 * u >= 3 * updates}
    out |= set(range(coarse, updates, coarse))
    return sorted(u for u in out if 1 <= u < updates)


def load_folds_r1():
    return json.loads(FOLDS_JSON_R1.read_text())


def load_folds():
    """folds_r2.json when present (checked against round 1), else ../folds.json."""
    r1 = load_folds_r1()
    if not FOLDS_JSON_R2.exists():
        return r1
    q = json.loads(FOLDS_JSON_R2.read_text())
    if list(q['development_donors']) != list(r1['development_donors']) and set(q['development_donors']) != set(r1['development_donors']):
        raise ValueError('folds_r2.json development donors differ from folds.json')
    if set(q['sealed_evaluation_donors_fold0']) != set(r1['sealed_evaluation_donors_fold0']):
        raise ValueError('folds_r2.json sealed donors differ from folds.json')
    for k in range(len(r1['inner_folds'])):
        a, b = r1['inner_folds'][k], q['inner_folds'][k]
        if a['inner_fold'] != b['inner_fold'] or set(a['train_donors']) != set(b['train_donors']) or set(a['heldout_donors']) != set(b['heldout_donors']):
            raise ValueError(f'folds_r2.json inner fold {k} differs from round 1')
    return q


def development_order():
    """Development donor -> index in ../folds.json (generation CRN key, as round 1)."""
    return {d: j for j, d in enumerate(load_folds_r1()['development_donors'])}


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
    if not 0 <= k < len(q['inner_folds']):
        raise ValueError(f'inner fold {k} not in the folds file')
    f = q['inner_folds'][k]
    if f['inner_fold'] != k or set(train_ids) != set(f['train_donors']) or (heldout_ids and set(heldout_ids) != set(f['heldout_donors'])):
        raise ValueError('cache donors differ from folds inner fold %d' % k)
    return True


def load_counts(path):
    counts = sp.load_npz(path).tocsr().astype(np.float32)
    d = counts.data
    if not np.isfinite(d).all() or (d < 0).any() or not np.array_equal(d, np.floor(d)):
        raise ValueError('counts must be finite nonnegative integers: ' + str(path))
    return counts


class Context:
    """Interface-1 cache context (round-1 Context + plug-in arm hooks + round-3 training options)."""

    def __init__(self, cache, device, flags, *, allow_nonprotocol_split=False):
        cache = Path(cache)
        flags = parse_arm(arm_name(tuple(flags)))
        self.cache = cache
        self.flags = tuple(flags)
        self.topt = arms.train_options(self.flags)
        self.batch_size = int(self.topt.batch)
        self.bag = int(self.topt.bag)
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
        age = self.raw_c[:, 6]
        self.age_mean, self.age_sd = float(age.mean()), float(age.std())
        self.space_age = (float(space['age_mean']), float(space['age_sd'])) if 'age_mean' in space else None
        self.plugins = arms.make_plugin_arms(self.flags)
        self.batch_arms = [a for a in self.plugins if a.overrides('batch_condition')]
        self.arm_rngs = {}
        self.train_seed = self.stream_root = None
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
        r1 = arms.round1_names(self.flags)
        need_Z = 'genemean' in r1 or 'donorpc' in r1 or any(a.needs_Z for a in self.plugins)
        self.Z = standardize_counts(self.X, self.mean, self.sd) if need_Z else None
        for a in self.plugins:
            a.setup(self)
        self.conditions = torch.as_tensor(self.condition_matrix(self.raw_c), device=dev)
        self.features = Features(self.flags, self, self.plugins)
        self.offset_table = self.offset_info = None
        if 'offset' in r1:
            self.offset_table, self.offset_info = arms.offset_init(self.X, self.by, stage_of(self.raw_c))

    def condition_matrix(self, raw_c):
        """Round-1 build_conditions, then plug-in hook (a) chain."""
        cond = build_conditions(raw_c, self.flags, self.age_mean, self.age_sd)
        return arms.plugin_conditions(self.plugins, self, raw_c, cond)

    def target_conditions(self, raw_c, is_heldout):
        """Default generation condition for target donors (hook a, then hook e)."""
        return arms.plugin_generation_conditions(self.plugins, self, raw_c, self.condition_matrix(raw_c), is_heldout)

    def project(self, x):
        return standardize_counts(x, self.mean, self.sd) @ self.W

    def arm_seed(self, name, purpose):
        """Seed for an arm's private randomness: SeedSequence([stream_root, 7, fold, seed, crc32(name), purpose])."""
        if self.train_seed is None:
            raise ValueError('arm_seed is available only after trainer.build (it depends on the training seed)')
        ss = np.random.SeedSequence([self.stream_root, arms.ARM_STREAM_TAG, self.fold, self.train_seed, arms.crc(name), int(purpose)])
        return int(ss.generate_state(1, dtype=np.uint64)[0]) % (2 ** 63 - 16)

    def batch(self, rng, phase='critic'):
        """Released uniform-donor batch; critic observation from the arm; hook (b)."""
        ds = rng.integers(len(self.ids), size=self.batch_size)
        idx = torch.as_tensor(np.stack([rng.choice(self.by[i], self.bag, replace=True) for i in ds]), device=self.device)
        real = self.features.real(idx, ds)
        c = self.conditions[torch.as_tensor(ds, device=self.device)]
        for a in self.batch_arms:
            c2 = a.batch_condition(self, c, ds, self.arm_rngs[a.name], phase)
            if not isinstance(c2, torch.Tensor) or c2.shape != c.shape or c2.dtype != c.dtype or c2.device != c.device:
                raise ValueError(f'{a.flag}.batch_condition must return a [B,7] tensor like its input')
            c = c2
        return real, c, self.X[idx].sum(-1)

    def arm_state(self):
        q = {'features': self.features.state(), 'features_info': self.features.info}
        if self.offset_table is not None:
            q['offset_init'] = torch.as_tensor(self.offset_table).clone()
            q['offset_info'] = self.offset_info
        if self.plugins:
            q['r2'] = {a.flag: {'arm': a.state(), 'blocks': {b.name: b.state() for a2, b, _, _ in self.features.blocks if a2 is a}}
                       for a in self.plugins}
        return q

    def plugin_info(self):
        return {a.flag: {'name': a.name, 'value': a.value, 'info': a.info} for a in self.plugins}


def recipe(topt=None):
    """Resolved native recipe: released_update.resolved_recipe(T005) for default options, else
    get_recipe with the released adapter's arguments plus arms.recipe_overrides(topt) (R3-1)."""
    if RELEASED_CONFIG != RELEASED_DEFAULT_CONFIG:
        raise ValueError('released CONFIG must be the default PG8 configuration')
    if topt is None or topt.is_default:
        return resolved_recipe(T005)
    return general_recipe(topt)


def general_recipe(topt):
    """get_recipe('gan', trainer='bispf_optimized', ...) exactly as released_update.resolved_recipe, with t. overrides.

    For default options this equals resolved_recipe(T005) field for field (test_core.py ``recipes``)."""
    q = {'reg_coeff': T005['reg_coeff'], 'reg_kappa': T005['reg_kappa'], 'prior_reg': T005['prior_reg']}
    q.update(arms.recipe_overrides(topt))   # integration: t.reg / t.kappa / t.priorreg override the three t005 fields
    return get_recipe('gan', trainer='bispf_optimized', **q)


def check_recipe(rec, topt):
    """Round-1 canonical-recipe assertion for default options; generalised for t. options (R3-1)."""
    if topt is None or topt.is_default:
        ok = (rec.name == 'bispf_optimized' and rec.loss == 'non_saturating' and rec.num_particles == 512 and rec.z_dim == 8
              and rec.ema_decay == rec.input_noise_std == rec.output_noise_std == 0 and rec.network_lr_horizon_cap is None
              and rec.lr_floor == rec.network_lr_floor == 1)
    else:
        want = arms.recipe_overrides(topt)
        ok = (rec.name == 'bispf_optimized' and rec.loss == 'non_saturating' and rec.input_noise_std == rec.output_noise_std == 0
              and rec.lr_schedule == 'cosine' and rec.serve_average == 0 and rec.optimizer_family == 'dualnorm'
              and rec.optimizer_momentum == 0 and rec.constraint_geometry_mode == 'direction_blend'
              and all(getattr(rec, k) == v for k, v in want.items())
              and (topt.anneal is not None or (rec.lr_floor == rec.network_lr_floor == 1 and rec.network_lr_horizon_cap is None))
              and (topt.ema is not None or rec.ema_decay == 0))
    if not ok:
        raise ValueError('canonical PG recipe changed')
    return rec


def expected_lrs(rec, step):
    """Native per-group learning rates for the update after ``step`` completed updates: (G, prior), (D,) (R3-1)."""
    net, pri = learning_rate_scales(step, rec)
    return [rec.lr * net, rec.lr * rec.prior_lr_mult * pri], [rec.lr * rec.d_lr_mult * net]


class ShadowEMA:
    """Read-out EMA copies of the generator and particle prior (docstring R3-2).

    ``update`` repeats UpdatePolicy.finish_step's averaging exactly (mul_(d).add_(live, alpha=1-d)
    for trainable parameters, copy for frozen parameters and buffers, prior set_sigma when present).
    No optimizer sees these tensors and no random number is drawn.
    """

    def __init__(self, G, P, decays=SHADOW_EMA_DECAYS):
        self.decays = tuple(float(d) for d in decays)
        self.models = {}
        for d in self.decays:
            g, pr = copy.deepcopy(G).eval(), copy.deepcopy(P).eval()
            g.requires_grad_(False); pr.requires_grad_(False)
            self.models[d] = (g, pr)

    @torch.no_grad()
    def update(self, G, P):
        for d, (g, pr) in self.models.items():
            for target, source in ((g, G), (pr, P)):
                for averaged, current in zip(target.parameters(), source.parameters()):
                    if not current.requires_grad:
                        averaged.copy_(current)
                    else:
                        averaged.mul_(d).add_(current, alpha=1 - d)
                for averaged, current in zip(target.buffers(), source.buffers()):
                    averaged.copy_(current)
                if target is pr and hasattr(target, 'set_sigma'):
                    target.set_sigma(source.sigma)

    def state(self, cpu=False):
        f = (lambda m: _cpu_state(m)) if cpu else (lambda m: {k: v.detach().clone() for k, v in m.state_dict().items()})
        return {arms.decay_text(d): {'generator': f(g), 'prior': f(pr)} for d, (g, pr) in self.models.items()}

    def load(self, state):
        if set(state) != {arms.decay_text(d) for d in self.decays}:
            raise ValueError(f'checkpoint shadow EMA decays {sorted(state)} differ from {self.decays}')
        for d, (g, pr) in self.models.items():
            q = state[arms.decay_text(d)]
            g.load_state_dict(q['generator'], strict=True)
            pr.load_state_dict(q['prior'], strict=True)


_SHADOWS = weakref.WeakKeyDictionary()   # policy -> ShadowEMA (one per build; never stored on native objects)


def shadow_of(bundle):
    return _SHADOWS[bundle[-1]]


def generator_kwargs(ctx, rec):
    """Constructor arguments after the (c2) chain (round-1/2 default: zd = recipe z_dim, zc = 8, h = 128)."""
    kw = {'zd': rec.z_dim, 'zc': 8, 'h': 128}
    for a in ctx.plugins:
        if a.overrides('generator_kwargs'):
            new = a.generator_kwargs(ctx, dict(kw))
            if not isinstance(new, dict) or new.get('zd') != rec.z_dim:
                raise ValueError(f'{a.flag}.generator_kwargs must return a dict keeping zd = recipe z_dim (use t.zdim)')
            kw = new
    return kw


def make_generator(ctx, rec):
    """Round-1 generator classes, or the plug-in class chain (hooks c, c2).  Global RNG = generator_init."""
    r1 = arms.round1_names(ctx.flags)
    gen_arms = [a for a in ctx.plugins if a.overrides('generator_class')]
    kw_arms = [a for a in ctx.plugins if a.overrides('generator_kwargs')]
    if 'offset' in r1:
        if kw_arms:
            raise ValueError('round-1 offset cannot be combined with generator_kwargs plug-ins')
        G = arms.make_offset_generator_class()(ctx.mu.cpu(), (ctx.theta * 1.0).cpu(), ctx.offset_table, zd=rec.z_dim, zc=8)
    elif gen_arms or kw_arms:
        cls = arms.hooked_count_generator()
        for a in gen_arms:
            new = a.generator_class(ctx, cls)
            if not (isinstance(new, type) and issubclass(new, cls)):
                raise TypeError(f'{a.flag}.generator_class must return a subclass of its input class')
            cls = new
        G = cls(ctx.mu.cpu(), (ctx.theta * 1.0).cpu(), **generator_kwargs(ctx, rec))
    else:
        G = CountGenerator(ctx.mu.cpu(), (ctx.theta * 1.0).cpu(), zd=rec.z_dim, zc=8)
    with torch.no_grad():
        for a in ctx.plugins:
            if a.overrides('init_generator'):
                a.init_generator(ctx, G)
    return G


def build(ctx, seed, *, stream_root=STREAM_ROOT):
    """released_adapter.build with arm-dependent G/D classes (same call order); (re)creates ctx.arm_rngs."""
    device = ctx.device
    streams = training_streams(ctx.fold, seed, stream_root)
    ctx.train_seed, ctx.stream_root = seed, stream_root
    ctx.arm_rngs = {a.name: np.random.default_rng(ctx.arm_seed(a.name, 0)) for a in ctx.batch_arms}
    rec = recipe(ctx.topt)
    torch.manual_seed(streams['generator_init'])
    G = make_generator(ctx, rec).to(device)
    torch.manual_seed(streams['critic_init'])
    nf = ctx.features.n_features
    crit_arms = [a for a in ctx.plugins if a.overrides('critic_module')]
    if len(crit_arms) > 1:
        raise ValueError('at most one arm may provide critic_module')
    if crit_arms:
        D = crit_arms[0].critic_module(ctx, ctx.features.layout())
        if not isinstance(D, torch.nn.Module) or not any(True for _ in D.parameters()):
            raise TypeError(f'{crit_arms[0].flag}.critic_module must return an nn.Module with parameters')
        D = D.to(device)
    else:
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
    check_recipe(rec, ctx.topt)
    if not (isinstance(og, NormalizedOptimizer) and isinstance(od, NormalizedOptimizer)):
        raise ValueError('canonical PG recipe changed')
    generators = {k: torch.Generator(device=device).manual_seed(streams[k]) for k in ['cell', 'count']}
    rng = np.random.default_rng(streams['donor_sampler'])
    torch.manual_seed(streams['gamma_global'])
    _SHADOWS[policy] = ShadowEMA(G, P)   # read-out copies (R3-2): no RNG, no optimizer
    usage = Usage(P.z)
    if ctx.batch_size != BATCH:   # usage.Usage's unique-row histogram has 33 bins (released batch 32); diagnostic only
        usage.unique_batch_histogram = [0] * (ctx.batch_size + 1)
    return (rec, G, D, P, og, od, pen, loss, reg, policy), generators, rng, usage, streams


def _critic_is_released_style(D, nf):
    phi = getattr(D, 'phi', None)
    return isinstance(D, (VectorCritic, WideVectorCritic)) and isinstance(phi[0], torch.nn.Linear) and phi[0].in_features == nf + 7


def _check_plugin_critic_blocks(D, feats, real, c):
    """Update-0 check for a plug-in critic: dD/dx nonzero on every non-released block (R3-4).  No .grad, no RNG."""
    with torch.enable_grad():
        x = real.detach().clone().requires_grad_(True)
        out = D(x, c)
        if out.shape != (len(x),):
            raise ValueError(f'plug-in critic must return [B], got {tuple(out.shape)}')
        gx = torch.autograd.grad(out.sum(), x)[0]
    for b in feats.layout().blocks:
        if b.name != 'released':
            assert bool(gx[:, b.lo:b.hi].abs().sum() > 0), f'no critic input gradient on block {b.flag}/{b.name}'


def advance(ctx, bundle, generators, rng, usage, *, execution_limit):
    """released_update.advance with the arm critic observation (round-1 item 5, docstring R1, R3-1/2/4)."""
    rec, G, D, P, og, od, pen, loss, reg, policy = bundle
    step = policy.completed_steps
    feats = ctx.features
    B, K = ctx.batch_size, ctx.bag

    def fake(z, c, lib):
        rate = G.rates(z, c, K, lib, generators['cell'])
        counts = torch.poisson(rate.detach(), generator=generators['count'])
        return feats.fake(critic_features(counts, rate, ctx.mean, ctx.sd), rate, counts)
    with serial_autograd():
        real, c, lib = ctx.batch(rng, 'critic')
        assert real.shape == (B, feats.n_features)
        noise = policy.begin_step(real, execution_limit=execution_limit)
        assert noise.input_sigma == 0 and float(noise.output_sigma) == 0
        want_g, want_d = expected_lrs(rec, step)
        assert [x['lr'] for x in og.param_groups] == want_g, ([x['lr'] for x in og.param_groups], want_g)
        assert [x['lr'] for x in od.param_groups] == want_d, ([x['lr'] for x in od.param_groups], want_d)
        D.train(); G.eval()
        with torch.no_grad():
            z, _ = P.sample(B, generator=policy.latent_generator)
            policy.observe_support(z)
            f = fake(z, c, lib)
        assert f.shape == real.shape
        policy.observe_critic_pair(real, f)
        dr, df = D(real, c), D(f, c); adv = loss.d_loss(dr, df); pen.collect_stats = step % 250 == 0
        penalty = pen(D, real, f, c); ld = adv + penalty
        od.zero_grad(); ld.backward(); finite(D, 'D gradients', True)
        if step == 0 and feats.n_extra:
            if _critic_is_released_style(D, feats.n_features):
                assert bool(D.phi[0].weight.grad[:, 150:feats.n_features].abs().sum() > 0), 'no critic gradient on added block'
                for a, blk, lo, hi in feats.blocks:
                    assert bool(D.phi[0].weight.grad[:, lo:hi].abs().sum() > 0), f'no critic gradient on block {a.flag}/{blk.name}'
            else:
                _check_plugin_critic_blocks(D, feats, real, c)
        dn = gradnorm(D) if step % 250 == 0 else None
        od.step(); finite(D, 'D parameters'); policy.after_critic_step()
        D.eval(); G.train(); D.requires_grad_(False)
        real2, c2, lib2 = ctx.batch(rng, 'generator'); z, idx = P.sample(B, generator=policy.latent_generator)
        f = fake(z, c2, lib2); lg = loss.g_loss(D(f, c2))
        raw = reg(P.z if rec.num_particles <= 1024 else P.z[torch.unique(idx)]); total = lg + rec.prior_reg * raw
        og.zero_grad(); policy.observe_sampled_rows(idx)
        constraint_geometry_backward(total, og, (lg,))
        finite(G, 'G gradients', True); finite(P, 'prior gradients', True)
        if step == 0:
            must = (list(G.step0_grad_params()) if hasattr(G, 'step0_grad_params') else [G.out[-1].weight, G.log_theta]) + [P.z] + \
                ([G.stage_offset] if hasattr(G, 'stage_offset') else [])
            for a in ctx.plugins:
                must += list(a.generator_grad_params(G))
            assert all(p.grad is not None and bool(p.grad.abs().sum() > 0) for p in must)
        gn = gradnorm(G) if step % 250 == 0 else None; pn = gradnorm(P) if step % 250 == 0 else None
        policy.after_generator_backward(loss_gan=lg.detach(), loss_critic=adv.detach())
        before_prior = P.z.detach().clone()
        og.step(); finite(G, 'G parameters'); finite(P, 'prior parameters')
        policy.after_generator_step(); D.requires_grad_(True); policy.finish_step()
        usage.observe(P.z, before_prior, idx)
    shadow_of(bundle).update(G, P)   # read-out EMA (R3-2): after the completed update, no RNG
    assert policy.completed_steps == step + 1
    return {'step': policy.completed_steps, 'D_adversarial': float(adv.detach()), 'D_penalty': float(penalty.detach()), 'D_total': float(ld.detach()),
            'G_adversarial': float(lg.detach()), 'prior_regularization_unweighted': float(raw.detach()), 'G_total': float(total.detach()),
            'G_prior_LRs': [g['lr'] for g in og.param_groups], 'D_LRs': [g['lr'] for g in od.param_groups],
            'critic_gap': float((dr - df).mean().detach()), 'D_gradient_norm': dn, 'G_gradient_norm': gn, 'prior_gradient_norm': pn,
            'penalty_stats': pen.last_stats, 'penalty_diagnostics': pen.diagnostics()}


def _cpu_state(module):
    return {k: v.detach().cpu().clone() for k, v in module.state_dict().items()}


def snapshot(ctx, bundle, generators, rng, usage, seed, streams, status, *, endpoint, stream_root, wall_seconds):
    rec, G, D, P, og, od, pen, loss, reg, policy = bundle
    if policy._phase != 'ready':
        raise ValueError('no checkpoint at a partial optimizer-pair boundary')
    q = {'schema': 1, 'method': 'pg', 'study': STUDY, 'round': ROUND, 'arm': arm_name(ctx.flags), 'flags': list(ctx.flags), 'panel': ctx.panel,
         'fold': ctx.fold, 'seed': seed, 'cache_manifest_SHA256': sha(ctx.cache / 'manifest.json'),
         'recipe': rec.to_dict(), 'policy': policy.state_dict(), 'prior_regularizer': rec.prior_regularizer_checkpoint(reg),
         'generators': {k: g.get_state() for k, g in generators.items()}, 'numpy_rng': rng.bit_generator.state,
         'usage': usage.state_dict(), 'training_streams': streams, 'stream_root': stream_root,
         'torch_cpu_rng': torch.get_rng_state(),
         'torch_device_rng': torch.cuda.get_rng_state(ctx.device) if ctx.device.type == 'cuda' else None,
         'completed_updates': policy.completed_steps, 'fixed_endpoint': endpoint, 'status': status,
         'particlegan_version': VERSION, 'penalty_options': dict(PENALTY_OPTIONS), 'variant_config': dict(RELEASED_CONFIG),
         'native_recipe': 'bispf_optimized', 'released_hyperparameters': dict(T005), 'arm_state': ctx.arm_state(),
         'arm_streams': {k: r.bit_generator.state for k, r in ctx.arm_rngs.items()},
         'arm_runtime': {a.flag: a.runtime_state() for a in ctx.plugins},
         'train_options': ctx.topt.as_dict(), 'r3_shadow': shadow_of(bundle).state(),
         'wall_seconds': wall_seconds}
    return copy.deepcopy(q)


def model_snapshot(ctx, bundle, seed, *, endpoint, stream_root):
    """snap_{u}.pt content: generator + prior state, shadow EMAs, native EMA (t.ema), all CPU, and the fit identity (R3-3)."""
    rec, G, P, policy = bundle[0], bundle[1], bundle[3], bundle[-1]
    q = {'schema': 1, 'kind': 'model_snapshot', 'study': STUDY, 'round': ROUND, 'arm': arm_name(ctx.flags),
         'flags': list(ctx.flags), 'panel': ctx.panel, 'fold': ctx.fold, 'seed': seed,
         'cache_manifest_SHA256': sha(ctx.cache / 'manifest.json'), 'stream_root': stream_root,
         'completed_updates': policy.completed_steps, 'fixed_endpoint': endpoint,
         'generator': _cpu_state(G), 'prior': _cpu_state(P), 'arm_state': ctx.arm_state(),
         'shadow': shadow_of(bundle).state(cpu=True)}
    if rec.ema_decay > 0:
        q['native_ema'] = {'decay': rec.ema_decay, 'generator': _cpu_state(policy.ema_G), 'prior': _cpu_state(policy.ema_prior)}
    return copy.deepcopy(q)


def _same(a, b):
    if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
        if isinstance(a, torch.Tensor) or isinstance(b, torch.Tensor):
            return torch.equal(torch.as_tensor(np.asarray(a) if not isinstance(a, torch.Tensor) else a).cpu(),
                               torch.as_tensor(np.asarray(b) if not isinstance(b, torch.Tensor) else b).cpu())
        a, b = np.asarray(a), np.asarray(b)
        return a.shape == b.shape and a.dtype == b.dtype and np.array_equal(a, b)
    if isinstance(a, torch.Tensor) or isinstance(b, torch.Tensor):
        # snapshots are loaded on CPU while the live arm state sits on the fit device
        return torch.equal(torch.as_tensor(a).cpu(), torch.as_tensor(b).cpu())
    if isinstance(a, dict) and isinstance(b, dict):
        return set(a) == set(b) and all(_same(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        return len(a) == len(b) and all(_same(x, y) for x, y in zip(a, b))
    return a == b


def restore(ctx, q, seed=None):
    """released_adapter.restore for this study's snapshot (+ plug-in arm streams/runtime state)."""
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
    streams_saved = q.get('arm_streams', {})
    if set(streams_saved) != set(ctx.arm_rngs):
        raise ValueError('checkpoint arm streams differ from the arm')
    for k, st in streams_saved.items():
        ctx.arm_rngs[k].bit_generator.state = st
    runtime = q.get('arm_runtime', {})
    for a in ctx.plugins:
        if a.flag in runtime and runtime[a.flag] is not None:
            a.load_runtime_state(runtime[a.flag])
    if q.get('train_options') != ctx.topt.as_dict():
        raise ValueError('checkpoint training options differ from the arm')
    if 'r3_shadow' not in q:
        raise ValueError('checkpoint has no shadow EMA state (not a round-3 checkpoint)')
    shadow_of(b).load(q['r3_shadow'])
    return b, g, r, u, seeds


def fit_dir(out_root, flags, ctx, seed):
    return Path(out_root) / arm_name(flags) / ctx.panel / f'fold{ctx.fold}' / f'seed{seed}'


def departures(ctx, rec):
    """config.json ``departures``: every t. flag (resolved recipe fields) and every g./d. plug-in flag (R3-5)."""
    out = arms.train_departures(ctx.flags, rec.to_dict() | {k: getattr(rec, k) for k in (
        'lr_schedule', 'lr_anneal_start', 'lr_floor', 'network_lr_floor', 'network_lr_horizon_cap', 'ema_decay', 'total_steps',
        'batch_size', 'num_particles', 'z_dim', 'lr', 'd_lr_mult', 'prior_lr_mult')})
    for a in ctx.plugins:
        if a.flag.startswith(('g.', 'd.')):
            out[a.flag] = {'kind': 'generator architecture' if a.flag.startswith('g.') else 'critic architecture',
                           'value': a.value, 'info': a.info}
    return out


def make_config(ctx, args, stream_root):
    rec = recipe(ctx.topt)
    shapes = {}
    if any(a.overrides('generator_kwargs') or a.overrides('generator_class') for a in ctx.plugins):
        shapes = {k: (v if isinstance(v, (int, float, str, bool)) or v is None else repr(v)) for k, v in generator_kwargs(ctx, rec).items()}
    return {'study': STUDY, 'round': ROUND, 'arm': arm_name(ctx.flags), 'flags': list(ctx.flags), 'panel': ctx.panel, 'fold': ctx.fold,
            'seed': args.seed, 'cache': str(Path(args.cache).resolve()), 'cache_manifest_SHA256': sha(ctx.cache / 'manifest.json'),
            'protocol_split': ctx.protocol_split, 'train_donors': list(ctx.ids), 'heldout_donors': ctx.heldout_ids,
            'n_genes': ctx.n_genes, 'hyperparameters': dict(T005), 'recipe': rec.to_dict(), 'native_recipe': 'bispf_optimized',
            'particlegan_version': VERSION, 'updates': args.updates, 'recipe_total_steps': rec.total_steps, 'device': args.device,
            'stream_root': stream_root, 'batch': [ctx.batch_size, ctx.bag], 'checkpoint_every': args.checkpoint_every,
            'snapshot_updates': snapshot_updates(args.updates),
            'folds_file': str(FOLDS_JSON_R2 if FOLDS_JSON_R2.exists() else FOLDS_JSON_R1),
            'critic_features': ctx.features.n_features, 'arm_info': {**ctx.features.info, **({'offset': ctx.offset_info} if ctx.offset_info else {}),
                                                                     'age_mean_train': ctx.age_mean, 'age_sd_train': ctx.age_sd,
                                                                     'age_space': ctx.space_age},
            'r2_arms': ctx.plugin_info(),
            'training_options': ctx.topt.as_dict(), 'departures': departures(ctx, rec),
            'learning_rates': {'generator': rec.lr, 'critic': rec.lr * rec.d_lr_mult, 'particle': rec.lr * rec.prior_lr_mult,
                               'schedule': {'lr_schedule': rec.lr_schedule, 'lr_anneal_start': rec.lr_anneal_start, 'lr_floor': rec.lr_floor,
                                            'network_lr_floor': rec.resolved_network_lr_floor,
                                            'network_lr_horizon_cap': rec.network_lr_horizon_cap}},
            'shadow_ema_decays': list(SHADOW_EMA_DECAYS), 'native_ema_decay': rec.ema_decay,
            'default_generation_state': 'native_ema' if rec.ema_decay > 0 else 'raw',
            'critic_layout': ctx.features.layout().as_dict(),
            'critic_module': next((a.flag for a in ctx.plugins if a.overrides('critic_module')), None),
            'generator_kwargs': shapes}


_CONFIG_MUTABLE = {'cache', 'checkpoint_every', 'folds_file'}


def peak_memory():
    q = {'max_rss_MB': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024}
    if torch.cuda.is_available() and torch.cuda.is_initialized():
        q['cuda_max_allocated_MB'] = torch.cuda.max_memory_allocated() / 2 ** 20
    return q


FROZEN_OUT_ROOTS = ('fits', 'results', 'results_screen48', 'screen_analysis', 'caches', 'fits_r2', 'results_r2', 'results_r2_interim',
                    'posthoc_r2', 'smoke_r2', 'logs', 'logs_r2', 'smoke_fits_v2', 'smoke_logs', 'smoke_logs_v2', 'smoke_results_v2',
                    'results_stale_prescreen')
ALLOWED_OUT_ROOTS = ('fits_r3', 'smoke_r3', 'results_r3', 'logs_r3')


def check_out_root(out_root):
    """R5 + R3-6: refuse frozen round-1/2 trees and any other folder under RUNS except the round-3 output roots."""
    o = Path(out_root).resolve()
    for name in FROZEN_OUT_ROOTS:
        fr = (RUNS / name).resolve()
        if o == fr or fr in o.parents:
            raise ValueError(f'refusing --out-root {o}: inside the frozen round-1/2 tree {fr}')
    runs = RUNS.resolve()
    if o == runs or runs in o.parents:
        if not any(o == (RUNS / a).resolve() or (RUNS / a).resolve() in o.parents for a in ALLOWED_OUT_ROOTS):
            raise ValueError(f'refusing --out-root {o}: round-3 outputs go under {RUNS}/{{{",".join(ALLOWED_OUT_ROOTS)}}}')
    return o


def run(args):
    check_out_root(args.out_root)
    flags = parse_arm(args.arm)
    horizon = arms.train_options(flags).steps
    if args.updates is None:
        args.updates = horizon
    if not 1 <= args.updates <= horizon:
        raise ValueError(f'updates must be in 1..{horizon} (the declared horizon)')
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
        if q['fixed_endpoint'] != args.updates or q.get('round') != ROUND:
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
        for p in out.glob('snap_*.pt'):  # stale snapshots of an unfinished fresh start
            p.unlink()
        b, g, r, u, streams = build(ctx, args.seed, stream_root=stream_root)
    policy = b[-1]
    snaps = set(snapshot_updates(args.updates))
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

    def save_model_snapshot(step):
        p = out / f'snap_{step}.pt'
        tmp = out / f'.snap_{step}.partial.pt'
        torch.save(model_snapshot(ctx, b, args.seed, endpoint=args.updates, stream_root=stream_root), tmp)
        tmp.replace(p)
    first = policy.completed_steps
    try:
        while policy.completed_steps < args.updates:
            if stops:
                raise InterruptedError('stop signal received; checkpoint at completed update')
            if args.stop_after is not None and policy.completed_steps >= args.stop_after:
                raise InterruptedError('--stop-after reached (developer interruption test)')
            metrics = advance(ctx, b, g, r, u, execution_limit=args.updates)
            if metrics['step'] in snaps:
                save_model_snapshot(metrics['step'])
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
    snap_files = {u_: (sha(out / f'snap_{u_}.pt') if (out / f'snap_{u_}.pt').exists() else None) for u_ in sorted(snaps)}
    receipt = {'status': 'training_completed', 'completed_updates': policy.completed_steps, 'endpoint': args.updates,
               'full_20k_endpoint': args.updates == RECIPE_STEPS, 'full_horizon_endpoint': args.updates == ctx.topt.steps,
               'wall_seconds': train_seconds,
               'seconds_per_update_this_session': (time.monotonic() - started) / max(new, 1), 'updates_this_session': new,
               'resumed_from': resumed_from, 'device': args.device, 'threads': args.threads, 'peak_memory': peak_memory(),
               'final_SHA256': sha(out / 'final.pt'), 'snapshots_SHA256': {str(k): v for k, v in snap_files.items()},
               'snapshots_missing': [k for k, v in snap_files.items() if v is None]}
    dump(out / 'fit_receipt.json', receipt)
    if receipt['snapshots_missing']:
        print(f'[trainer] WARNING: snapshots missing {receipt["snapshots_missing"]} in {out}', flush=True)
    del b
    generate.generate_fit(out, args.cache, device=args.device, ctx=ctx)
    return out


def parser():
    p = argparse.ArgumentParser(description=__doc__.split('\n')[0], formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--cache', required=True)
    p.add_argument('--arm', default='base')
    p.add_argument('--seed', type=int, required=True)
    p.add_argument('--out-root', default=str(DEFAULT_OUT_ROOT))
    p.add_argument('--updates', type=int, default=None, help='run length (default: the horizon, t.steps or 20000; may only shorten it)')
    p.add_argument('--device', choices=['cpu', 'cuda'], default='cpu')
    p.add_argument('--threads', type=int, default=3)
    p.add_argument('--gpu-mem-fraction', type=float, default=None)
    p.add_argument('--checkpoint-every', type=int, default=CHECKPOINT_EVERY, help=argparse.SUPPRESS)
    p.add_argument('--stop-after', type=int, default=None, help=argparse.SUPPRESS)
    p.add_argument('--stream-root', type=int, default=STREAM_ROOT, help=argparse.SUPPRESS)
    p.add_argument('--allow-nonprotocol-split', action='store_true', help='developer caches only (recorded in config)')
    return p


def main(argv=None):
    args = parser().parse_args(argv)
    horizon = arms.train_options(parse_arm(args.arm)).steps
    if args.updates is None:
        args.updates = horizon
    if not 1 <= args.updates <= horizon:
        raise ValueError(f'updates must be in 1..{horizon} (the declared horizon; t.steps@N sets it)')
    if args.device == 'cpu':
        os.environ.setdefault('CUDA_VISIBLE_DEVICES', '')
    out = run(args)
    print(f'[trainer] done: {out}', flush=True)


if __name__ == '__main__':
    sys.modules.setdefault('trainer', sys.modules['__main__'])
    main()
