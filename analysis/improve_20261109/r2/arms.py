"""Round-2 arms for improve_20261109: round-1 flags (unchanged) + a plug-in registry.

This module is a COPY of round-1 ``../arms.py`` (the round-1 flags ``offset``,
``genemean``, ``donorpc`` and ``covars`` keep their exact round-1 code paths
and semantics, see "Round-1 flags" at the end of this docstring) extended with
a registry for round-2 arms written as separate plug-in files.

Arm strings and flags
=====================

Arm string = comma-joined flags sorted by Python string order (= ``LC_ALL=C
sort``), ``base`` when there are none.  A flag is ``name`` or ``name@value``.
Names match ``[a-z][a-z0-9_]*``.  The value text must already be in the arm's
canonical form (for float values ``'%g' % v``: ``0.5`` not ``0.50``, ``2``
not ``2.0``); ``parse_arm`` refuses non-canonical text so the fit folder name
``fits_r2/{arm}/{panel}/fold{k}/seed{s}/`` is unique per arm.

* Round-1 names without a value mean exactly what they meant in round 1.
* ``genemean@s`` (core, not a plug-in): the round-1 genemean block with its
  scale multiplied by s (``s > 0``, ``s != 1``; use ``genemean`` for s = 1).
  Same reference bags, same s_gm; block = ``s * s_gm * mean_cells(F)``.
  ``covars@``, ``offset@`` and ``donorpc@`` are refused.
* Any other name is a plug-in arm: a class registered with ``@register`` in a
  file ``r2/arm_<something>.py``.  ``arm_<name>.py`` is imported first; if the
  name is still unknown every ``r2/arm_*.py`` is imported (a file may register
  several arms and modes).  Import errors of unrelated plug-in files are only
  reported when the requested name is not found.

Plug-in API (``Arm`` and ``FeatureBlock`` below; worked example:
``r2/arm_example.py``, flag ``example`` / ``example@s``, mode ``exshrink@v``)
==========================================================================

A plug-in file does::

    import arms
    from arms import Arm, FeatureBlock, register, register_mode, canonical_float

    @register
    class MyArm(Arm):
        name = 'myarm'              # flag name
        value_policy = 'optional'   # 'none' | 'optional' | 'required'
        requires = ()               # other flag NAMES that must be present
        conflicts = ()              # flag NAMES that must be absent
        needs_Z = False             # True -> ctx.Z (standardised train cells) is built
        @classmethod
        def parse_value(cls, text):  # -> (value, canonical_text); default float
            return canonical_float(text, lo=0.0, hi=1.0)

One instance is created per Context (training or generation) with
``self.value`` (parsed value or None), ``self.flag`` (canonical flag string),
``self.name`` and ``self.info`` (a JSON-serialisable dict written to
config.json ``r2_arms[flag]``).  Hooks, all optional (default = no effect).
Plug-in arms run in sorted flag order; every hook output is shape-checked.

Setup (train-only fitting)
    ``setup(self, ctx) -> None``
    Called once in ``trainer.Context.__init__`` after the released data are
    loaded and BEFORE conditions and critic features are built.  Available:
    ``ctx.X`` (train counts [cells,G] float32 tensor on ctx.device),
    ``ctx.by`` (list of int row arrays per train donor, cache order),
    ``ctx.ids``, ``ctx.raw_c`` (float64 [n_train,7], bench_core encoding),
    ``ctx.fit_labels``, ``ctx.mean``/``ctx.sd`` (gene standardisation),
    ``ctx.W`` ([G,50] = W / pc_sd), ``ctx.trainpcs`` ([cells,50]),
    ``ctx.mu``/``ctx.theta``, ``ctx.fitlibs``, ``ctx.Z`` (only when some arm
    has needs_Z or genemean/donorpc is on), ``ctx.age_mean``/``ctx.age_sd``,
    ``ctx.genes``, ``ctx.n_genes``, ``ctx.fold``, ``ctx.panel``,
    ``ctx.flags``, ``ctx.device``, ``ctx.heldout_ids``.  setup must be a
    deterministic function of the TRAIN donors of the cache (it is recomputed
    at resume and at generation and compared with the checkpoint, hook (f));
    it must NOT read held-out counts and must not use any training random
    stream or the torch global RNG.  For fixed randomness use
    ``np.random.default_rng(np.random.SeedSequence([20261109, zlib.crc32(b'myarm'), k]))``.

(a) Per-donor condition matrix
    ``conditions(self, ctx, raw_c, cond) -> np.ndarray[n,7] float32``
    ``raw_c``: float64 [n,7] raw bench_core encoding (Braak/6, Thal/5,
    CERAD/3, ADNC/3, Male, APOE4 dose, raw age) of n donors; ``cond``: the
    condition built so far (round-1 ``build_conditions`` for the flags, then
    earlier plug-ins).  Return a new array (do not modify ``cond``).  Used for
    the train donors (``ctx.conditions``) AND for generation targets (train +
    held-out donors), so it must be a function of raw_c rows plus train-fitted
    setup state.  The condition is always 7 wide (generator donor layer and
    critic take [B,7]); slot 3 (ADNC/3) carries the stage, slots 0-2 are 0 in
    every round-1 arm and are free for plug-ins, slots 4-6 are used by covars.

(b) Per-batch condition transform during training (e.g. condition dropout)
    ``batch_condition(self, ctx, c, ds, rng, phase) -> Tensor[B,7]``
    ``c``: [B,7] float32 tensor on ctx.device (the batch donors' rows of
    ctx.conditions after earlier plug-ins); ``ds``: numpy int [B] train-donor
    indices of the batch; ``rng``: this arm's own ``numpy.random.Generator``;
    ``phase``: ``'critic'`` (first batch of an update) or ``'generator'``
    (second batch).  The returned condition is used for BOTH the generator
    (fake bags) and the critic (real and fake bags) in that half-update.
    Return a new tensor.  ``rng`` is created only for arms that override this
    hook: ``default_rng(SeedSequence([stream_root, 7, fold, seed,
    crc32(name), 0]))`` (``ctx.arm_seed(name, 0)``), independent of the 11
    released training streams, so arms without this hook keep the round-1
    streams bit for bit; its bit_generator state is checkpointed in
    recovery.pt / final.pt (``arm_streams``) and restored on resume.

(c) Generator class and parameters
    ``generator_class(self, ctx, cls) -> type``
    ``cls`` starts as ``HookedCountGenerator`` (a pg_model.CountGenerator
    subclass whose ``parameters_nb`` is split into overridable steps, forward
    bitwise identical to CountGenerator):

        film(z, c) -> (scale [B,h], shift [B,h])            donor FiLM
        cell_hidden(noise, scale, shift, z, c) -> h [B,K,h]
        logits(h, z, c) -> [B,K,G]                           before softmax
        dispersion(z, c) -> theta (broadcastable to [B,K,G])

    Return a subclass of ``cls`` (call ``super()`` to compose with other
    plug-ins; each plug-in wraps the class returned by the previous one, in
    sorted flag order).  The constructor signature stays
    ``(mean, theta, zd=4, zc=8, h=128)``; capture arm data by closure.  Add
    parameters in ``__init__`` AFTER ``super().__init__`` without drawing
    from the torch global RNG (zeros, constants, or a private
    ``torch.Generator().manual_seed(ctx.arm_seed(name, k))``), so the released
    parameters get bitwise the round-1 initial tensors.  New parameters land
    in the generator optimizer group (generator learning rate, released
    DualNorm update; no new loss term).  Not combinable with round-1
    ``offset`` (refused by parse_arm).
    ``init_generator(self, ctx, G) -> None``: called once on the constructed
    generator (CPU, before ``.to(device)``), inside ``torch.no_grad``, for
    every arm that overrides it (also when no arm changes the class); may
    initialise parameters the class added.  Never use the global RNG here.
    ``generator_grad_params(self, G) -> list[nn.Parameter]``: parameters
    asserted to receive a nonzero gradient at update 0 (default none).

(d) Critic feature blocks (appended after the released 150 features)
    ``feature_blocks(self, ctx) -> list[FeatureBlock]`` (called once, after
    setup).  Critic input = [150 released bag features, round-1 genemean or
    donorpc block if that flag is on, then each plug-in's blocks in sorted
    flag order and list order]; the critic is ``WideVectorCritic(total
    width)`` (released VectorCritic layers, h=128, wider input) whenever the
    width is not 150.  A FeatureBlock has ``name`` (str, unique within the
    arm), ``width`` (int > 0), ``info`` (JSON dict) and:

        real(self, ctx, idx, ds) -> Tensor[B,width]
            idx: [B,cells] long tensor of train cell rows (use ctx.X[idx],
            ctx.Z[idx], ctx.trainpcs[idx]); ds: numpy [B] train-donor index.
        fake(self, ctx, cf, rate, counts) -> Tensor[B,width]
            cf: straight-through critic_features [B,cells,G]
            (standardised log1p CP10k of the Poisson counts with the rate
            gradient); rate: Gamma-NB rate [B,cells,G] WITH gradient;
            counts: Poisson counts [B,cells,G] (no gradient).  A
            linear-in-counts block uses ``counts + (rate - rate.detach())``.
        state(self) -> dict   tensors/arrays/numbers for the consistency check.

    Real and fake must be the same function of (counts) for the same cells;
    the trainer asserts at update 0 that every block's critic input columns
    receive a nonzero critic gradient.  Feature blocks are not used at
    generation.

(e) Generation-time hooks
    ``generation_condition(self, ctx, raw_c, cond, is_heldout) -> np.ndarray[n,7]``
    Condition used by default generation (``generated.npz``) for each target
    donor, applied after the (a) chain to the targets' raw_c; default
    identity.  ``is_heldout``: bool [n].
    Named generation modes (``generated_{mode}.npz``) are registered with::

        register_mode(name, plan, *, value_policy='none', parse_value=None,
                      prepare=None, doc='')

    ``plan(ctx, value, target, rng, states) -> (cond, state_names)``
        target: ``GenTarget`` (index, donor, is_heldout, raw_c float64[7],
        stage, dev_index, default_cond float32[7], M, K); rng: numpy
        Generator seeded from SeedSequence([20261109, 5, fold, dev_index,
        crc32(mode name)]) (depends on fold, target and mode name only:
        same draws for every arm, seed and fit); states: tuple of model
        states present in the fit (``'snap_<u>'`` ascending, then
        ``'final'``).  Return cond float32 [M,7] (per generated donor) and a
        length-M sequence of state names.  Generated donors that use the same
        state should be contiguous (each contiguous run is generated in chunks
        of 8 in order of m; the per-target streams and library draws are the
        default ones, so a plan returning (default_cond x M, 'final' x M)
        reproduces generated.npz bitwise).
    ``prepare(ctx, value, state_name, G, P) -> None`` (optional): called once
        per loaded state on a private copy of the generator and prior, inside
        ``torch.no_grad``, before generation (e.g. scale a learned gain).
    Modes never retrain.  Built-in modes (here): ``snapens`` (M split over the
    four snapshots and final, 7,7,6,6,6 in ascending update order, contiguous)
    and ``snap@U`` (all M from snapshot update U).  A mode registered in a
    plug-in file is found by importing ``arm_<mode name>.py`` or, failing
    that, every ``arm_*.py``.  Mode names share the flag syntax
    (``name@value``, canonical value text).

(f) Checkpointed arm state
    ``state(self) -> dict`` (plus every block's ``state()``): deterministic
    tensors / numpy arrays / numbers / lists recomputed from the cache by
    setup; stored in recovery.pt, final.pt and snap_*.pt under
    ``arm_state['r2'][flag] = {'arm': ..., 'blocks': {name: ...}}`` and
    compared exactly (on CPU) with the recomputed value at resume and
    generation (mismatch -> ValueError).
    ``runtime_state(self)`` / ``load_runtime_state(self, state)``: optional
    mutable state that changes during training (default None); stored in
    recovery.pt/final.pt ``arm_runtime`` and restored on resume.

Round-1 flags (copied code; semantics unchanged)
================================================

base
    Released formulation: CountGenerator (output bias = log pooled mean
    proportions, moment theta init), 150 bag features from 50 PCs
    (critic.bag_features), critic.VectorCritic(h=128), stage-only condition
    (stage_law.mask_adnc semantics), uniform donor sampling, Poisson forward
    with straight-through rate features (pg_model.critic_features).

offset
    ``OffsetCountGenerator``: a learnable table ``stage_offset`` [4, G] is
    added to the output logits before the softmax, row selected by the ADNC
    stage round(3*c[:,3]) and applied as ``one_hot(stage, 4) @ stage_offset``.
    Initialisation from TRAIN donors only (``offset_init``):

        C_d   = summed panel counts of donor d,  N_d = sum_g C_dg
        a     = 1 (pseudocount, counts per gene)
        q_g   = (sum_d C_dg + a) / (sum_d N_d + a G)          pooled proportions
        r_dg  = log((C_dg + a)/(N_d + a G)) - log q_g          donor log-ratios
        m_g   = mean over all n train donors of r_dg
        r_sg  = mean_{d in s} r_dg - m_g                       donor-averaged stage effect
        v_g   = pooled within-stage variance of r_dg
        e_s   = 1/n_s - 1/n
        tau2  = max(1e-6, mean over (s with n_s>0, g) of (r_sg^2 - v_g e_s))
        w_sg  = tau2 / (tau2 + v_g e_s)                        (0 when n_s = 0)
        offset_sg = w_sg * r_sg

genemean
    Critic input = [150 released features, s_gm * mean_cells(F)] where F is
    the SAME straight-through critic_features tensor [B, cells, G] whose PCA
    projection feeds the 150 features; for real bags F = standardize_counts
    (real counts).  s_gm = sqrt(E||f150||^2 / E||mean_cells F||^2) from 256
    real train bags (numpy default_rng(20261109)).  WideVectorCritic.

donorpc
    Critic input = [150 released features, ((mean_cells F) - mu) V / sd] with
    the top-K (K = min(16, n_train - 1)) right singular vectors of the train
    donor mean standardised profiles.  Mutually exclusive with genemean.

covars
    Condition slots: 3 = ADNC/3 (as released), 4 = Male, 5 = APOE4 dose, 6 =
    (age - mean)/sd with train-donor population mean/SD.
"""
from __future__ import annotations

import importlib
import math
import re
import sys
import zlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn

HERE = Path(__file__).resolve().parent
# Plug-ins do ``import arms``; make sure that is THIS module even if it was
# imported under another name.
sys.modules.setdefault('arms', sys.modules[__name__])

ROUND1_FLAGS = ('covars', 'donorpc', 'genemean', 'offset')
FLAGS = ROUND1_FLAGS
OFFSET_PSEUDOCOUNT = 1.0
GENEMEAN_REF_BAGS = 256
GENEMEAN_REF_SEED = 20261109
DONORPC_MAX_K = 16
ARM_STREAM_TAG = 7      # SeedSequence([stream_root, 7, fold, seed, crc32(name), purpose])
MODE_STREAM_TAG = 5     # SeedSequence([20261109, 5, fold, dev_index, crc32(mode)])
GEN_ROOT = 20261109
_NAME_RE = re.compile(r'^[a-z][a-z0-9_]*$')


def crc(name):
    return zlib.crc32(str(name).encode())


# =========================================================================== flags / values
def split_flag(flag):
    """'name@value' -> ('name', 'value'); 'name' -> ('name', None)."""
    name, sep, vtext = str(flag).partition('@')
    if not _NAME_RE.match(name) or (sep and not vtext) or '@' in vtext or ',' in vtext or any(ch.isspace() for ch in vtext):
        raise ValueError(f'bad flag {flag!r}: want name or name@value, name [a-z][a-z0-9_]*')
    return name, (vtext if sep else None)


def flag_name(flag):
    return split_flag(flag)[0]


def canonical_float(text, lo=None, hi=None, *, lo_open=False, hi_open=False):
    """Parse a float value; canonical text is '%g' % v.  Returns (v, text)."""
    try:
        v = float(text)
    except ValueError:
        raise ValueError(f'value {text!r} is not a number') from None
    if not math.isfinite(v):
        raise ValueError(f'value {text!r} must be finite')
    if lo is not None and (v < lo or (lo_open and v == lo)):
        raise ValueError(f'value {v} below {"(" if lo_open else "["}{lo}')
    if hi is not None and (v > hi or (hi_open and v == hi)):
        raise ValueError(f'value {v} above {hi}{")" if hi_open else "]"}')
    return v, '%g' % v


def canonical_int(text, lo=None, hi=None):
    if not re.fullmatch(r'-?[0-9]+', str(text)):
        raise ValueError(f'value {text!r} is not an integer')
    v = int(text)
    if (lo is not None and v < lo) or (hi is not None and v > hi):
        raise ValueError(f'value {v} out of range [{lo},{hi}]')
    return v, str(v)


# =========================================================================== plug-in API
class FeatureBlock:
    """One critic feature block (hook d).  Subclass and set name/width."""
    name = None
    width = None

    def __init__(self):
        self.info = {}

    def real(self, ctx, idx, ds):
        raise NotImplementedError

    def fake(self, ctx, cf, rate, counts):
        raise NotImplementedError

    def state(self):
        return {}


class Arm:
    """Base class of a round-2 plug-in arm.  See the module docstring."""
    name = None
    value_policy = 'none'      # 'none' | 'optional' | 'required'
    requires = ()
    conflicts = ()
    needs_Z = False

    @classmethod
    def parse_value(cls, text):
        return canonical_float(text)

    def __init__(self, value, flag):
        self.value = value
        self.flag = flag
        self.info = {}

    # ---- hooks (defaults: no effect)
    def setup(self, ctx):
        pass

    def conditions(self, ctx, raw_c, cond):
        return cond

    def batch_condition(self, ctx, c, ds, rng, phase):
        return c

    def generator_class(self, ctx, cls):
        return cls

    def init_generator(self, ctx, G):
        pass

    def generator_grad_params(self, G):
        return []

    def feature_blocks(self, ctx):
        return []

    def generation_condition(self, ctx, raw_c, cond, is_heldout):
        return cond

    def state(self):
        return {}

    def runtime_state(self):
        return None

    def load_runtime_state(self, state):
        pass

    # ---- introspection used by the trainer
    @classmethod
    def overrides(cls, hook):
        return getattr(cls, hook) is not getattr(Arm, hook)


_ARMS = {}
_MODES = {}
_PLUGIN_ERRORS = {}
_SCANNED = [False]


def register(cls):
    """Class decorator: register a plug-in Arm subclass under ``cls.name``."""
    if not (isinstance(cls, type) and issubclass(cls, Arm)):
        raise TypeError('register expects an Arm subclass')
    name = cls.name
    if not isinstance(name, str) or not _NAME_RE.match(name):
        raise ValueError(f'bad arm name {name!r}')
    if name in ROUND1_FLAGS:
        raise ValueError(f'{name!r} is a round-1 flag; plug-ins cannot redefine it')
    if cls.value_policy not in ('none', 'optional', 'required'):
        raise ValueError('value_policy must be none, optional or required')
    old = _ARMS.get(name)
    if old is not None and (old.__module__, old.__qualname__) != (cls.__module__, cls.__qualname__):
        raise ValueError(f'arm {name!r} registered twice ({old.__module__} and {cls.__module__})')
    _ARMS[name] = cls
    return cls


@dataclass(frozen=True)
class ModeSpec:
    name: str
    plan: object
    value_policy: str = 'none'
    parse_value: object = None
    prepare: object = None
    doc: str = ''
    module: str = ''


@dataclass
class GenTarget:
    """What a mode plan sees about one generation target donor."""
    index: int
    donor: str
    is_heldout: bool
    raw_c: np.ndarray
    stage: int
    dev_index: int
    default_cond: np.ndarray
    M: int
    K: int


def register_mode(name, plan, *, value_policy='none', parse_value=None, prepare=None, doc=''):
    """Register a generation mode (hook e).  See the module docstring."""
    if not isinstance(name, str) or not _NAME_RE.match(name):
        raise ValueError(f'bad mode name {name!r}')
    if value_policy not in ('none', 'optional', 'required'):
        raise ValueError('value_policy must be none, optional or required')
    mod = getattr(plan, '__module__', '')
    old = _MODES.get(name)
    if old is not None and old.module != mod:
        raise ValueError(f'mode {name!r} registered twice ({old.module} and {mod})')
    _MODES[name] = ModeSpec(name, plan, value_policy, parse_value or (lambda t: canonical_float(t)), prepare, doc, mod)
    return plan


def _import_plugin(path):
    modname = path.stem
    if modname in sys.modules:
        return
    if str(HERE) not in sys.path:
        sys.path.insert(0, str(HERE))
    try:
        importlib.import_module(modname)
    except Exception as exc:  # noqa: BLE001 - reported when the requested name is missing
        sys.modules.pop(modname, None)
        _PLUGIN_ERRORS[modname] = repr(exc)


def _scan_plugins():
    if _SCANNED[0]:
        return
    for p in sorted(HERE.glob('arm_*.py')):
        _import_plugin(p)
    _SCANNED[0] = True


def _lookup(table, name, kind):
    if name not in table:
        p = HERE / f'arm_{name}.py'
        if p.exists():
            _import_plugin(p)
    if name not in table:
        _scan_plugins()
    if name not in table:
        errs = '; '.join(f'{k}: {v}' for k, v in sorted(_PLUGIN_ERRORS.items()))
        raise ValueError(f'unknown {kind} {name!r}; registered {sorted(table)}' + (f'; plug-in import errors: {errs}' if errs else ''))
    return table[name]


def arm_class(name):
    return _lookup(_ARMS, name, 'arm flag')


def _check_value(name, vtext, policy, parser, kind):
    if vtext is None:
        if policy == 'required':
            raise ValueError(f'{kind} {name!r} needs a value ({name}@value)')
        return None
    if policy == 'none':
        raise ValueError(f'{kind} {name!r} takes no value')
    v, canon = parser(vtext)
    if canon != vtext:
        raise ValueError(f'{kind} value {vtext!r} is not canonical; write {name}@{canon}')
    return v


def flag_value(flag):
    """Parsed value of a canonical flag (None without a value)."""
    name, vtext = split_flag(flag)
    if vtext is None:
        return None
    if name == 'genemean':
        return canonical_float(vtext)[0]
    return arm_class(name).parse_value(vtext)[0]


def parse_arm(text):
    """Arm string -> sorted tuple of canonical flags; 'base' -> ().  Validates everything."""
    if text is None or text == '' or text == 'base':
        return ()
    if isinstance(text, (list, tuple)):
        text = ','.join(text)
    flags = [f.strip() for f in str(text).split(',') if f.strip()]
    if 'base' in flags:
        raise ValueError('base cannot be combined with other flags')
    names = []
    plug = []
    for f in flags:
        name, vtext = split_flag(f)
        names.append(name)
        if name in ROUND1_FLAGS:
            if vtext is None:
                continue
            if name != 'genemean':
                raise ValueError(f'round-1 flag {name!r} takes no value')
            v, canon = canonical_float(vtext, lo=0.0, lo_open=True)
            if canon != vtext:
                raise ValueError(f'value {vtext!r} is not canonical; write genemean@{canon}')
            if v == 1.0:
                raise ValueError('genemean@1 is genemean; use the round-1 name')
        else:
            cls = arm_class(name)
            _check_value(name, vtext, cls.value_policy, cls.parse_value, 'arm flag')
            plug.append(cls)
    if len(set(names)) != len(names):
        raise ValueError('duplicate arm flag name')
    if 'genemean' in names and 'donorpc' in names:
        raise ValueError('genemean and donorpc are mutually exclusive')
    nameset = set(names)
    for cls in plug:
        miss = set(cls.requires) - nameset
        if miss:
            raise ValueError(f'arm {cls.name!r} requires {sorted(miss)}')
        bad = set(cls.conflicts) & nameset
        if bad:
            raise ValueError(f'arm {cls.name!r} conflicts with {sorted(bad)}')
    if 'offset' in nameset and any(c.overrides('generator_class') for c in plug):
        raise ValueError('round-1 offset cannot be combined with a plug-in that changes the generator class')
    return tuple(sorted(flags))


def arm_name(flags):
    return ','.join(sorted(flags)) if flags else 'base'


def round1_names(flags):
    return {flag_name(f) for f in flags if flag_name(f) in ROUND1_FLAGS}


def make_plugin_arms(flags):
    """Plug-in Arm instances for the plug-in flags, in sorted flag order."""
    out = []
    for f in sorted(flags):
        name, vtext = split_flag(f)
        if name in ROUND1_FLAGS:
            continue
        cls = arm_class(name)
        out.append(cls(_check_value(name, vtext, cls.value_policy, cls.parse_value, 'arm flag'), f))
    return out


def parse_mode(text):
    """Mode string -> (ModeSpec, value, canonical text)."""
    name, vtext = split_flag(text)
    spec = _lookup(_MODES, name, 'generation mode')
    v = _check_value(name, vtext, spec.value_policy, spec.parse_value, 'mode')
    return spec, v, text


def mode_rng(fold, dev_index, mode_name):
    return np.random.default_rng(np.random.SeedSequence([GEN_ROOT, MODE_STREAM_TAG, int(fold), int(dev_index), crc(mode_name)]))


# =========================================================================== built-in modes
def snapshot_states(states):
    """Snapshot state names sorted by update."""
    snaps = [s for s in states if s.startswith('snap_')]
    return sorted(snaps, key=lambda s: int(s.split('_', 1)[1]))


def _snapens_plan(ctx, value, target, rng, states):
    snaps = snapshot_states(states)
    if len(snaps) != 4 or 'final' not in states:
        raise ValueError(f'snapens needs four snapshots and final; fit has {list(states)}')
    order = snaps + ['final']
    n = len(order)
    counts = [target.M // n + (1 if i < target.M % n else 0) for i in range(n)]
    names = [s for s, c in zip(order, counts) for _ in range(c)]
    return np.repeat(target.default_cond[None], target.M, 0), names


def _snap_plan(ctx, value, target, rng, states):
    name = f'snap_{value}'
    if name not in states:
        raise ValueError(f'snapshot {name} not in fit (have {list(states)})')
    return np.repeat(target.default_cond[None], target.M, 0), [name] * target.M


register_mode('snapens', _snapens_plan, doc='M generated donors split 7,7,6,6,6 over snap_u (ascending) and final')
register_mode('snap', _snap_plan, value_policy='required', parse_value=lambda t: canonical_int(t, lo=1),
              doc='all M generated donors from snapshot snap_<U>')


# =========================================================================== conditions
def stage_of(raw_c):
    v = np.asarray(raw_c, dtype=np.float64)[..., 3] * 3
    s = np.rint(v).astype(int)
    if np.any((s < 0) | (s > 3)) or not np.allclose(v, s, atol=1e-6, rtol=0):
        raise ValueError('ADNC must be one of four ordinal codes')
    return s


def build_conditions(raw_c, flags, age_mean=None, age_sd=None):
    """Seven-slot condition per donor as the ROUND-1 flags condition (copied)."""
    x = np.asarray(raw_c, dtype=np.float64)
    if x.ndim != 2 or x.shape[1] != 7 or not np.isfinite(x).all():
        raise ValueError('raw_c must be finite [n,7]')
    stage_of(x)
    out = np.zeros_like(x, dtype=np.float32)
    out[:, 3] = x[:, 3].astype(np.float32)  # identical to mask_adnc (float32 cast)
    if 'covars' in flags:
        if age_sd is None or not age_sd > 0:
            raise ValueError('covars arm needs a positive train age SD')
        out[:, 4] = x[:, 4]
        out[:, 5] = x[:, 5]
        out[:, 6] = (x[:, 6] - age_mean) / age_sd
    return out


def _checked_cond(arm, out, n, hook):
    out = np.asarray(out)
    if out.shape != (n, 7) or out.dtype != np.float32 or not np.isfinite(out).all():
        raise ValueError(f'{arm.flag}.{hook} must return finite float32 [{n},7], got {out.dtype} {out.shape}')
    return out


def plugin_conditions(plugins, ctx, raw_c, cond):
    """Hook (a) chain."""
    raw_c = np.asarray(raw_c, dtype=np.float64)
    for a in plugins:
        cond = _checked_cond(a, a.conditions(ctx, raw_c, cond.copy()), len(raw_c), 'conditions')
    return cond


def plugin_generation_conditions(plugins, ctx, raw_c, cond, is_heldout):
    raw_c = np.asarray(raw_c, dtype=np.float64)
    for a in plugins:
        cond = _checked_cond(a, a.generation_condition(ctx, raw_c, cond.copy(), np.asarray(is_heldout, bool)), len(raw_c), 'generation_condition')
    return cond


# =========================================================================== generator
def _count_generator_cls():
    from pg_model import CountGenerator  # imported lazily (sys.path set by trainer)
    return CountGenerator


_HOOKED = []


def hooked_count_generator():
    """HookedCountGenerator: CountGenerator with parameters_nb split into steps (hook c)."""
    if _HOOKED:
        return _HOOKED[0]
    CountGenerator = _count_generator_cls()

    class HookedCountGenerator(CountGenerator):
        """pg_model.CountGenerator; parameters_nb = film -> cell_hidden -> logits -> softmax/clamp -> dispersion.

        Same constructor (no new parameters, same RNG draws) and the same
        operations in the same order as CountGenerator.parameters_nb, so the
        forward is bitwise identical when no method is overridden.
        """

        def film(self, z, c):
            return self.donor(torch.cat([z, c], 1)).chunk(2, 1)

        def cell_hidden(self, noise, scale, shift, z, c):
            return self.cell(noise) * (1 + scale[:, None]) + shift[:, None]

        def logits(self, h, z, c):
            return self.out(h)

        def dispersion(self, z, c):
            return nn.functional.softplus(self.log_theta) + .05

        def parameters_nb(self, z, c, k, library, gen=None):
            scale, shift = self.film(z, c)
            noise = torch.randn(len(z), k, self.zc, device=z.device, generator=gen)
            h = self.cell_hidden(noise, scale, shift, z, c)
            proportions = torch.softmax(self.logits(h, z, c), dim=-1).clamp_min(1e-8)
            proportions = proportions / proportions.sum(-1, keepdim=True)
            mu = library.clamp_min(1)[..., None] * proportions
            theta = self.dispersion(z, c)
            return mu, theta

    _HOOKED.append(HookedCountGenerator)
    return HookedCountGenerator


def make_offset_generator_class():
    CountGenerator = _count_generator_cls()

    class OffsetCountGenerator(CountGenerator):
        """CountGenerator + learnable stage x gene logit offset (round-1 copy)."""

        def __init__(self, mean, theta, offset, zd=4, zc=8, h=128):
            super().__init__(mean, theta, zd=zd, zc=zc, h=h)
            offset = torch.as_tensor(offset, dtype=torch.float32)
            if offset.shape != (4, len(mean)):
                raise ValueError('offset table must be [4,G]')
            self.stage_offset = nn.Parameter(offset.clone())

        def parameters_nb(self, z, c, k, library, gen=None):
            scale, shift = self.donor(torch.cat([z, c], 1)).chunk(2, 1)
            noise = torch.randn(len(z), k, self.zc, device=z.device, generator=gen)
            h = self.cell(noise) * (1 + scale[:, None]) + shift[:, None]
            stage = torch.round(c[:, 3] * 3).long()
            onehot = nn.functional.one_hot(stage, 4).to(self.stage_offset.dtype)
            logits = self.out(h) + (onehot @ self.stage_offset)[:, None, :]
            proportions = torch.softmax(logits, dim=-1).clamp_min(1e-8)
            proportions = proportions / proportions.sum(-1, keepdim=True)
            mu = library.clamp_min(1)[..., None] * proportions
            theta = nn.functional.softplus(self.log_theta) + .05
            return mu, theta

    return OffsetCountGenerator


def offset_init(X, by, stages, pseudocount=OFFSET_PSEUDOCOUNT):
    """Shrunken per-stage pseudobulk log-ratio table [4,G] (train donors only; round-1 copy)."""
    Xt = torch.as_tensor(X)
    C = torch.stack([Xt[torch.as_tensor(r, device=Xt.device)].sum(0) for r in by]).double().cpu().numpy()
    stages = np.asarray(stages, dtype=int)
    n, G = C.shape
    a = float(pseudocount)
    N = C.sum(1)
    logq = np.log(C.sum(0) + a) - np.log(N.sum() + a * G)
    r_d = np.log(C + a) - np.log(N[:, None] + a * G) - logq
    m = r_d.mean(0)
    n_s = np.array([(stages == s).sum() for s in range(4)])
    present = [s for s in range(4) if n_s[s] > 0]
    r_s = np.zeros((4, G))
    ss = np.zeros(G)
    for s in present:
        msk = stages == s
        r_s[s] = r_d[msk].mean(0) - m
        ss += ((r_d[msk] - r_d[msk].mean(0)) ** 2).sum(0)
    df = n - len(present)
    if df < 1:
        raise ValueError('offset init needs more train donors than stages')
    v = ss / df
    e = np.array([1.0 / n_s[s] - 1.0 / n if n_s[s] else 0.0 for s in range(4)])
    excess = [r_s[s] ** 2 - v * e[s] for s in present]
    tau2 = max(1e-6, float(np.mean(excess)))
    w = np.zeros((4, G))
    for s in present:
        w[s] = tau2 / (tau2 + v * e[s])
    table = (w * r_s).astype(np.float32)
    info = {
        'pseudocount': a, 'tau2': tau2, 'donors_per_stage': n_s.tolist(), 'within_stage_df': int(df),
        'mean_within_stage_variance': float(v.mean()),
        'mean_weight_per_stage': [float(w[s].mean()) if n_s[s] else 0.0 for s in range(4)],
        'raw_rms_per_stage': [float(np.sqrt((r_s[s] ** 2).mean())) for s in range(4)],
        'shrunk_rms_per_stage': [float(np.sqrt((table[s].astype(np.float64) ** 2).mean())) for s in range(4)],
        'formula': 'offset_sg = tau2/(tau2 + v_g (1/n_s - 1/n)) * r_sg, r_sg = donor-averaged stage log-ratio minus '
                   'all-donor average (fixer revision v2); see arms.py docstring',
        'version': 'offset_init-v2-donor-averaged',
    }
    return table, info


# =========================================================================== critic
class WideVectorCritic(nn.Module):
    """critic.VectorCritic with an arbitrary input width (same layers, h=128)."""

    def __init__(self, n_features, h=128):
        super().__init__()
        self.n_features = int(n_features)
        self.phi = nn.Sequential(nn.Linear(n_features + 7, h), nn.SiLU(), nn.Linear(h, h), nn.SiLU())
        self.rho = nn.Sequential(nn.Linear(h + 7, h), nn.SiLU(), nn.Linear(h, 1))

    def forward(self, x, c):
        if x.ndim != 2 or x.shape[1] != self.n_features or c.shape != (len(x), 7):
            raise ValueError(f'Expected donor vectors [B,{self.n_features}] and conditions [B,7]')
        f = self.phi(torch.cat([x, c], 1))
        return self.rho(torch.cat([f, c], 1)).squeeze(1)


# =========================================================================== features
class Features:
    """Critic observation: released 150 + round-1 block (genemean/donorpc) + plug-in blocks.

    ``real(idx, ds=None)``: [B, cells] long tensor of train cell rows -> [B, F];
    ``fake(cf, rate=None, counts=None)``: straight-through critic_features
    [B, cells, G] (+ rate with gradient and Poisson counts) -> [B, F].  With
    round-1 flags only, both are the round-1 code path.  ``blocks`` lists
    (arm, block, first column, last column) of plug-in blocks.
    """

    def __init__(self, flags, ctx, plugins=()):
        from released_update import bag_features, CONFIG
        if CONFIG.get('fourier_feature_scale', 1.0) != 1.0:
            raise ValueError('released bag feature scale must be 1')
        self.bag_features = bag_features
        self.flags = flags
        self.ctx = ctx
        names = round1_names(flags)
        self.kind = 'genemean' if 'genemean' in names else 'donorpc' if 'donorpc' in names else 'base'
        self.W = ctx.W
        self.info = {'kind': self.kind}
        if self.kind == 'base':
            self.n_extra = 0
        elif self.kind == 'genemean':
            gm = [f for f in flags if flag_name(f) == 'genemean'][0]
            mult = flag_value(gm)
            self.n_extra = ctx.n_genes
            self.scale = self._genemean_scale()
            self.info.update(genemean_scale=self.scale, reference_bags=GENEMEAN_REF_BAGS,
                             reference_seed=GENEMEAN_REF_SEED)
            if mult is not None:  # genemean@s
                self.scale = self.scale * mult
                self.info.update(genemean_multiplier=mult, genemean_scale_effective=self.scale)
        else:
            self._fit_donorpc()
        self.n_round1 = 150 + self.n_extra
        self.blocks = []
        col = self.n_round1
        for a in plugins:
            for b in a.feature_blocks(ctx):
                if not isinstance(b, FeatureBlock) or not isinstance(b.name, str) or not (isinstance(b.width, int) and b.width > 0):
                    raise ValueError(f'{a.flag}: feature blocks must be FeatureBlock with name and positive int width')
                if any(x[0] is a and x[1].name == b.name for x in self.blocks):
                    raise ValueError(f'{a.flag}: duplicate block name {b.name}')
                self.blocks.append((a, b, col, col + b.width))
                col += b.width
        self.n_extra = col - 150
        self.n_features = col
        if self.blocks:
            self.info['r2_blocks'] = [{'flag': a.flag, 'block': b.name, 'columns': [lo, hi], 'info': b.info}
                                      for a, b, lo, hi in self.blocks]

    def _extra(self, gene_mean):
        if self.kind == 'genemean':
            return gene_mean * self.scale
        return ((gene_mean - self.pc_mu) @ self.pc_V) / self.pc_sd

    def _check(self, a, b, t):
        if t.ndim != 2 or t.shape[1] != b.width:
            raise ValueError(f'{a.flag}/{b.name} returned {tuple(t.shape)}, want [B,{b.width}]')
        return t

    def real(self, idx, ds=None):
        base = self.bag_features(self.ctx.trainpcs[idx])
        if self.kind == 'base' and not self.blocks:
            return base
        parts = [base]
        if self.kind != 'base':
            parts.append(self._extra(self.ctx.Z[idx].mean(1)))
        parts += [self._check(a, b, b.real(self.ctx, idx, ds)) for a, b, _, _ in self.blocks]
        return torch.cat(parts, dim=1)

    def fake(self, cf, rate=None, counts=None):
        base = self.bag_features(cf @ self.W)
        if self.kind == 'base' and not self.blocks:
            return base
        parts = [base]
        if self.kind != 'base':
            parts.append(self._extra(cf.mean(1)))
        parts += [self._check(a, b, b.fake(self.ctx, cf, rate, counts)) for a, b, _, _ in self.blocks]
        return torch.cat(parts, dim=1)

    @torch.no_grad()
    def _genemean_scale(self):
        ctx = self.ctx
        rng = np.random.default_rng(GENEMEAN_REF_SEED)
        num = den = 0.0
        for _ in range(GENEMEAN_REF_BAGS // 32):
            ds = rng.integers(len(ctx.by), size=32)
            idx = torch.as_tensor(np.stack([rng.choice(ctx.by[i], 128, replace=True) for i in ds]), device=ctx.device)
            num += float(self.bag_features(ctx.trainpcs[idx]).double().square().sum())
            den += float(ctx.Z[idx].mean(1).double().square().sum())
        return float(math.sqrt(num / den))

    @torch.no_grad()
    def _fit_donorpc(self):
        ctx = self.ctx
        P = torch.stack([ctx.Z[torch.as_tensor(r, device=ctx.device)].double().mean(0) for r in ctx.by]).cpu().numpy()
        n = len(P)
        K = min(DONORPC_MAX_K, n - 1)
        mu = P.mean(0)
        U, S, Vt = np.linalg.svd(P - mu, full_matrices=False)
        V = Vt[:K].T
        signs = np.sign(V[np.abs(V).argmax(0), np.arange(K)])
        V = V * signs
        scores = (P - mu) @ V
        sd = scores.std(0)
        if not (sd > 0).all():
            raise ValueError('degenerate donor PC')
        dev = ctx.device
        self.pc_mu = torch.as_tensor(mu, dtype=torch.float32, device=dev)
        self.pc_V = torch.as_tensor(V, dtype=torch.float32, device=dev)
        self.pc_sd = torch.as_tensor(sd, dtype=torch.float32, device=dev)
        self.n_extra = K
        self.info.update(K=int(K), explained_variance_fraction=(S[:K] ** 2 / (S ** 2).sum()).tolist(),
                         train_donor_score_sd=sd.tolist())

    def state(self):
        """Round-1 block tensors for the exact consistency check (plug-in blocks: Context.arm_state)."""
        if self.kind == 'genemean':
            return {'genemean_scale': self.scale}
        if self.kind == 'donorpc':
            return {'pc_mu': self.pc_mu.cpu(), 'pc_V': self.pc_V.cpu(), 'pc_sd': self.pc_sd.cpu()}
        return {}
