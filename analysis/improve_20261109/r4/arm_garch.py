"""Round-3 generator-architecture flags (work item R3, PROTOCOL_R3.md): g.h, g.zc, g.depth, g.disp, g.cdisp.

Registered through the core hooks (r3/arms.py docstring, hooks c and c2).
With none of these flags present nothing changes (the file only registers
arm classes; the trainer builds pg_model.CountGenerator as in round 1/2).
Every flag is listed by the trainer under config.json ``departures``.

Released generator (pg_model.CountGenerator, h = 128, zc = 8, zd = z_dim = 8)::

    donor  : [z (zd) , c (7)] -> Linear(zd+7, h) -> SiLU -> Linear(h, 2h) -> (scale, shift)   FiLM
    cell   : noise (zc) ~ N(0, I) -> Linear(zc, h) -> SiLU ; h1 = cell * (1 + scale) + shift
    head   : SiLU -> Linear(h, h) -> SiLU -> Linear(h, G)  -> softmax -> mu = library * p
    disp   : theta_g = softplus(log_theta_g) + .05  (one dispersion per gene)
    counts : Poisson(Gamma(theta, theta / mu))

Flags
=====

``g.h@N``      hidden width N (8..2048; 128 refused, it is the default) for
               every released hidden layer (donor MLP, cell layer, head).
               Pure constructor argument (hook c2): the released layers get
               new shapes and therefore new initial draws (under the
               released generator_init seed; no other stream).
``g.zc@N``     cell-noise dimension N (1..256; 8 refused): Linear(N, h) cell
               layer; generation draws N normals per cell (the trainer's and
               generate.py's ``torch.randn(B, K, zc)`` uses ``self.zc``).
``g.depth@N``  N hidden layers in the head instead of 1 (2..8): N-1 extra
               residual layers ``x <- x + W SiLU(x) + b`` applied to the
               FiLM-ed cell state before the released head.  W and b start
               at ZERO and are created without any random draw, so the
               released parameters keep bitwise their round-1 initial
               tensors and the initial generator computes bitwise the
               released function (tested); the extra layers then learn under
               the released generator optimizer (DualNorm, generator lr).
               ``generator_grad_params`` asserts the first extra W gets a
               nonzero gradient at update 0.
``g.disp``     donor-dependent dispersion: theta_dg = softplus(log_theta_g +
               A [z, c] + b) + .05 (A: [G, zd+7], b: [G], both zero at start,
               no random draw), i.e. the per-gene log-dispersion gets a
               linear function of the donor latent and condition.  Initial
               generator bitwise released.  ``generator_grad_params`` asserts
               A gets a gradient.
``g.cdisp``    cell-dependent dispersion: theta_cg = softplus(log_theta_g +
               A SiLU(h_c) + b) + .05 with h_c the final hidden cell state
               (the input of the released head, after any g.depth layers;
               it already carries the donor FiLM), A: [G, h], b: [G], zero at
               start, no random draw; initial generator bitwise released.
               ``logits`` keeps h_c for the ``dispersion`` call that
               HookedCountGenerator.parameters_nb makes right after it.
               Exclusive with g.disp (g.cdisp contains it as a special case).

Flags compose (the class chain runs in sorted flag order: g.cdisp, g.depth,
g.disp, g.h, g.zc), e.g. ``g.depth@2,g.h@256``.  ``g.example`` (core worked
example) and round-1 ``offset`` are refused in combination (``conflicts``).

Parameter count on hvg2000 (G = 2000): released 312,736; g.disp adds 32,000
(G x 15 + G); g.cdisp adds 258,000 (G x 128 + G); g.depth@2 adds 16,512;
g.h@256 gives 719,776 in total.

``make_generator_class(base, depth=..., disp=..., cdisp=...)`` returns the class the
flags build; ``capacity_diagnose.py probe`` uses the same factory, so the
maximum-likelihood capacity probe tests exactly the architectures these flags
produce.  ``GARCH_FLAGS`` lists the flag names.

Evidence (capacity_diagnose.py docstring, "Findings"): in a maximum-likelihood
probe of exactly these classes on the fold 0-2 train cells, width, depth and
cell-noise size do not help (h256/h512 overfit cells; depth2/3, zc16/32
within about 1 nat per cell), while g.disp improves the validation ELBO by
4.9-7.1 nats per cell in every fold and seed and cuts generated per-gene
mean-log / nonzero / log-variance errors 2-3 fold.  g.disp is therefore the
flag recommended for the screen; g.h / g.depth / g.zc exist to confirm the
negative result; g.cdisp is the cell-level variant (less consistent in the
probe).

Repair (round-3 adversarial review, 2026-11-11): ``g.disp`` opens a learned STAGE path
---------------------------------------------------------------------------------------
``g.disp`` feeds the condition c (stage slot 3) into the log-dispersion map.  Under the
GAN with DualNorm (every update moves the zero-initialised A [2000 x 15] by a full-size
normalised step: spectral 0.012 sqrt(2000/15) = 0.139, Frobenius about 0.54, against
0.012 L2 for the whole log_theta vector) the reviewer measured, in a 300-update GPU fit
(g.disp,pbulk, fold 1, seed 0), a stage column of A that grows almost linearly
(column norm 5.8 / 11.4 / 16.6 / 22.1 at u = 75 / 150 / 225 / 300; 35-81% of ||A||^2;
per-gene log-dispersion gap ADNC 0 vs 3 about 0.5 and rising).  The data's stage effect
is indistinguishable from a permutation null (stage task), and stage inflation is the
largest error term, so this is an unwanted new stage path; the maximum-likelihood probe
(Adam, cosine) that motivated g.disp is not evidence about it.  ``g.disp`` is KEPT
unchanged (it is the object the probe tested, and round-3 code must keep its meaning),
but the screen uses instead:

``g.dispz[@s]`` donor-dependent dispersion from the donor LATENT ONLY:
               theta_dg = softplus(log_theta_g + s (A z + b)) + .05, A: [G, zd], b: [G],
               both zero at start (no random draw; initial generator bitwise released),
               no condition input, so no stage path (stage modes such as stagehalf /
               stageblind / fixedstage cannot change the dispersion).  s in (0, 1]
               (bare name: s = 1; ``g.dispz@1`` refused) is a FIXED re-parametrisation
               scale: DualNorm steps are invariant to the gradient's size, so the
               effective per-update move of s A is s times the released full-size step
               (a named way to give the dispersion map a smaller effective learning rate
               without touching the native optimizer; recorded under departures).
               Conflicts with g.disp and g.cdisp.  Probe variant ``dispz``
               (capacity_diagnose.py) measures its maximum-likelihood gain.

``disp_stats(G)`` / ``python3 -B arm_garch.py dispstats FIT_DIR`` report, for every snapshot
and the final state of a g.disp / g.dispz fit, the Frobenius / spectral norm of the
dispersion map, the norm of every input column (the stage column c3 separately for
g.disp) and the per-gene RMS of the stage-column offset, so the drift of the map can be
read from every snapshot (TR-1 / E7).  ``fixedstage`` refuses g.disp (TR-4).

Random streams.  g.h / g.zc change released layer shapes (their initial
draws come from the same generator_init stream, in the released order);
g.depth / g.disp / g.cdisp draw nothing.  None of the flags uses the arm streams
(no ``batch_condition``), so the 11 released training streams are untouched.
"""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

import arms
from arms import Arm, register, canonical_int

GARCH_FLAGS = ('g.cdisp', 'g.depth', 'g.disp', 'g.dispz', 'g.h', 'g.zc')
_CONFLICTS = ('offset', 'g.example')


class ZeroResidual(nn.Module):
    """x + W SiLU(x) + b with W, b initialised to zero WITHOUT any random draw (exact identity at start)."""

    def __init__(self, h):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(int(h), int(h)))
        self.bias = nn.Parameter(torch.zeros(int(h)))

    def forward(self, x):
        return x + F.linear(F.silu(x), self.weight, self.bias)


def depth_class(base):
    """Subclass of ``base`` (a HookedCountGenerator chain) with ``depth - 1`` zero-initialised residual head layers."""

    class DepthGenerator(base):
        def __init__(self, mean, theta, *args, depth=1, **kw):
            super().__init__(mean, theta, *args, **kw)
            depth = int(depth)
            if depth < 1:
                raise ValueError('depth must be >= 1')
            h = self.out[1].in_features
            self.garch_depth = depth
            self.extra = nn.ModuleList([ZeroResidual(h) for _ in range(depth - 1)])

        def logits(self, h, z, c):
            for layer in self.extra:
                h = layer(h)
            return super().logits(h, z, c)

    DepthGenerator.__qualname__ = DepthGenerator.__name__ = f'Depth{base.__name__}'
    return DepthGenerator


def disp_class(base):
    """Subclass of ``base`` with donor-dependent log-dispersion (zero-initialised linear map of [z, c])."""

    class DispGenerator(base):
        def __init__(self, mean, theta, *args, disp=False, **kw):
            super().__init__(mean, theta, *args, **kw)
            self.garch_disp = bool(disp)
            if self.garch_disp:
                n_in = self.donor[0].in_features
                self.disp_weight = nn.Parameter(torch.zeros(len(self.log_theta), n_in))
                self.disp_bias = nn.Parameter(torch.zeros(len(self.log_theta)))

        def dispersion(self, z, c):
            if not self.garch_disp:
                return super().dispersion(z, c)
            d = F.linear(torch.cat([z, c], 1), self.disp_weight, self.disp_bias)      # [B, G]
            return (nn.functional.softplus(self.log_theta + d) + .05)[:, None, :]     # [B, 1, G]

    DispGenerator.__qualname__ = DispGenerator.__name__ = f'Disp{base.__name__}'
    return DispGenerator


def cdisp_class(base):
    """Subclass of ``base`` with CELL-dependent log-dispersion: a zero-initialised linear map of SiLU(final cell state).

    ``logits`` keeps the final hidden cell state (after any g.depth layers, since this class sits below
    DepthGenerator in the chain) for the ``dispersion`` call that follows it in ``parameters_nb``."""

    class CellDispGenerator(base):
        def __init__(self, mean, theta, *args, cdisp=False, **kw):
            super().__init__(mean, theta, *args, **kw)
            self.garch_cdisp = bool(cdisp)
            if self.garch_cdisp:
                h = self.out[1].in_features
                self.cdisp_weight = nn.Parameter(torch.zeros(len(self.log_theta), h))
                self.cdisp_bias = nn.Parameter(torch.zeros(len(self.log_theta)))

        def logits(self, h, z, c):
            if self.garch_cdisp:
                self.__dict__['_cdisp_h'] = h
            return super().logits(h, z, c)

        def dispersion(self, z, c):
            if not self.garch_cdisp:
                return super().dispersion(z, c)
            h = self.__dict__.pop('_cdisp_h', None)
            if h is None:
                raise RuntimeError('g.cdisp: dispersion() must follow logits() (parameters_nb order)')
            d = F.linear(F.silu(h), self.cdisp_weight, self.cdisp_bias)                 # [B, K, G]
            return nn.functional.softplus(self.log_theta + d) + .05

    CellDispGenerator.__qualname__ = CellDispGenerator.__name__ = f'CellDisp{base.__name__}'
    return CellDispGenerator


def dispz_class(base):
    """Subclass of ``base`` with donor-LATENT-only log-dispersion (zero-initialised, scaled map of z; no condition input)."""

    class DispZGenerator(base):
        def __init__(self, mean, theta, *args, dispz=None, **kw):
            super().__init__(mean, theta, *args, **kw)
            self.garch_dispz = None if dispz is None else float(dispz)
            if self.garch_dispz is not None:
                if not 0 < self.garch_dispz <= 1:
                    raise ValueError('dispz scale must be in (0, 1]')
                zd = self.donor[0].in_features - 7
                self.disp_weight = nn.Parameter(torch.zeros(len(self.log_theta), zd))
                self.disp_bias = nn.Parameter(torch.zeros(len(self.log_theta)))

        def dispersion(self, z, c):
            if self.garch_dispz is None:
                return super().dispersion(z, c)
            d = self.garch_dispz * F.linear(z, self.disp_weight, self.disp_bias)          # [B, G]; c is NOT used
            return (nn.functional.softplus(self.log_theta + d) + .05)[:, None, :]          # [B, 1, G]

    DispZGenerator.__qualname__ = DispZGenerator.__name__ = f'DispZ{base.__name__}'
    return DispZGenerator


def make_generator_class(base=None, *, depth=1, disp=False, cdisp=False, dispz=None):
    """Class built by the g. flags (for the capacity probe and tests).  ``base`` defaults to HookedCountGenerator.

    Same chain order as the trainer (sorted flags: g.cdisp, g.depth, g.disp).  Instantiate with
    ``cls(mean, theta, zd=8, zc=zc, h=h, **opts)`` where opts holds only the keys the chain accepts
    (``depth`` when depth > 1, ``disp`` when disp, ``cdisp`` when cdisp, ``dispz`` = scale when dispz)."""
    if sum([bool(disp), bool(cdisp), dispz is not None]) > 1:
        raise ValueError('disp, cdisp and dispz are exclusive')
    cls = arms.hooked_count_generator() if base is None else base
    if cdisp:
        cls = cdisp_class(cls)
    if depth > 1:
        cls = depth_class(cls)
    if disp:
        cls = disp_class(cls)
    if dispz is not None:
        cls = dispz_class(cls)
    return cls


class _GArch(Arm):
    value_policy = 'required'
    conflicts = _CONFLICTS
    key = None

    def setup(self, ctx):
        self.info = {self.key: self.value}

    def generator_kwargs(self, ctx, kwargs):
        return {**kwargs, self.key: self.value}

    def state(self):
        return {self.key: self.value}


@register
class GWidth(_GArch):
    """g.h@N: hidden width of every released generator layer."""
    name = 'g.h'
    key = 'h'

    @classmethod
    def parse_value(cls, text):
        v, t = canonical_int(text, lo=8, hi=2048)
        if v == 128:
            raise ValueError('g.h@128 is the released width; omit the flag')
        return v, t


@register
class GCellNoise(_GArch):
    """g.zc@N: cell-noise (cell latent) dimension."""
    name = 'g.zc'
    key = 'zc'

    @classmethod
    def parse_value(cls, text):
        v, t = canonical_int(text, lo=1, hi=256)
        if v == 8:
            raise ValueError('g.zc@8 is the released cell-noise size; omit the flag')
        return v, t


@register
class GDepth(_GArch):
    """g.depth@N: N head hidden layers (N-1 zero-initialised residual layers added)."""
    name = 'g.depth'
    key = 'depth'

    @classmethod
    def parse_value(cls, text):
        return canonical_int(text, lo=2, hi=8)

    def generator_class(self, ctx, cls):
        return depth_class(cls)

    def generator_grad_params(self, G):
        return [G.extra[0].weight]


@register
class GCellDisp(_GArch):
    """g.cdisp: cell-dependent dispersion (zero-initialised map of the final cell state)."""
    name = 'g.cdisp'
    key = 'cdisp'
    value_policy = 'none'
    conflicts = _CONFLICTS + ('g.disp', 'g.dispz')

    def setup(self, ctx):
        self.info = {'cdisp': 'cell-dependent log-dispersion, zero-initialised linear map of SiLU(final cell state)'}

    def generator_kwargs(self, ctx, kwargs):
        return {**kwargs, 'cdisp': True}

    def generator_class(self, ctx, cls):
        return cdisp_class(cls)

    def generator_grad_params(self, G):
        return [G.cdisp_weight]

    def state(self):
        return {'cdisp': True}


@register
class GDisp(_GArch):
    """g.disp: donor-dependent dispersion (zero-initialised)."""
    name = 'g.disp'
    key = 'disp'
    value_policy = 'none'

    def setup(self, ctx):
        self.info = {'disp': 'donor-dependent log-dispersion, zero-initialised linear map of [z, c]'}

    def generator_kwargs(self, ctx, kwargs):
        return {**kwargs, 'disp': True}

    def generator_class(self, ctx, cls):
        return disp_class(cls)

    def generator_grad_params(self, G):
        return [G.disp_weight]

    def state(self):
        return {'disp': True}


@register
class GDispZ(_GArch):
    """g.dispz[@s]: donor-latent-only dispersion (zero-initialised, fixed scale s; no stage input).  See module docstring."""
    name = 'g.dispz'
    key = 'dispz'
    value_policy = 'optional'
    conflicts = _CONFLICTS + ('g.disp', 'g.cdisp')

    @classmethod
    def parse_value(cls, text):
        v, t = arms.canonical_float(text, lo=0.0, hi=1.0)
        if v <= 0:
            raise ValueError('g.dispz@s needs 0 < s <= 1')
        if v == 1.0:
            raise ValueError('g.dispz@1 is g.dispz; use the bare name')
        return v, t

    @property
    def scale(self):
        return 1.0 if self.value is None else float(self.value)

    def setup(self, ctx):
        self.info = {'dispz': 'donor-latent-only log-dispersion: softplus(log_theta + s (A z + b)) + .05, A, b zero at start; '
                              'no condition input (no stage path)', 'scale': self.scale,
                     'effective_step': 'DualNorm step of A times s (fixed re-parametrisation, no optimizer change)'}

    def generator_kwargs(self, ctx, kwargs):
        return {**kwargs, 'dispz': self.scale}

    def generator_class(self, ctx, cls):
        return dispz_class(cls)

    def generator_grad_params(self, G):
        return [G.disp_weight]

    def state(self):
        return {'dispz': self.scale}


# --------------------------------------------------------------------------- diagnostics (TR-1 / E7)
def disp_stats(state, with_condition):
    """Norms of the dispersion map from a generator state_dict (or module).  None when the fit has no map.
    ``with_condition``: True for g.disp (columns [z (zd), c (7)], stage = column zd + 3), False for g.dispz."""
    sd = state.state_dict() if hasattr(state, 'state_dict') else state
    A = sd.get('disp_weight')
    if A is None:
        return None
    A = A.detach().double().cpu()
    b = sd['disp_bias'].detach().double().cpu()
    col = A.norm(dim=0)
    out = {'shape': list(A.shape), 'fro': float(A.norm()), 'spectral': float(torch.linalg.matrix_norm(A, 2)),
           'bias_norm': float(b.norm()), 'column_norms': col.tolist()}
    if with_condition:                  # g.disp: [z (zd), c (7)]; stage slot = column zd + 3
        zd = A.shape[1] - 7
        st = A[:, zd + 3]
        out.update(stage_column_norm=float(st.norm()), stage_share_of_fro2=float(st.square().sum() / max(A.square().sum(), 1e-300)),
                   stage_offset_rms_per_gene_adnc0_vs_3=float(st.square().mean().sqrt()),
                   stage_offset_max_abs=float(st.abs().max()))
    else:
        out['stage_column_norm'] = 0.0
    return out


def fit_disp_stats(fit_dir):
    """disp_stats of every snap_*.pt (key 'generator' or the snapshot's own layout) and of final.pt, in update order."""
    import json as _json
    from pathlib import Path
    fit_dir = Path(fit_dir)
    flags = str(_json.loads((fit_dir / 'config.json').read_text()).get('arm', fit_dir.parts[-4])).split(',')
    with_c = 'g.disp' in flags
    scale = ([float(f.partition('@')[2] or 1) for f in flags if f.partition('@')[0] == 'g.dispz'] or [1.0])[0]
    rows = []
    paths = sorted(fit_dir.glob('snap_*.pt'), key=lambda p: int(p.stem.split('_')[1]))
    for p in paths + ([fit_dir / 'final.pt'] if (fit_dir / 'final.pt').exists() else []):
        q = torch.load(p, map_location='cpu', weights_only=False)
        sd = _find_generator_state(q)
        r = disp_stats(sd, with_c) if sd is not None else None
        if r is not None:   # g.dispz@s: the map applied is s * (A z + b); norms above are of the raw parameter A
            r.update(scale=scale, effective_fro=scale * r['fro'], effective_spectral=scale * r['spectral'])
        rows.append({'file': p.name, 'update': int(p.stem.split('_')[1]) if p.stem.startswith('snap_') else 'final', 'stats': r})
    return rows


def _find_generator_state(q):
    if isinstance(q, dict):
        if 'disp_weight' in q:
            return q
        for k in ('generator', 'G', 'models'):
            if k in q:
                r = _find_generator_state(q[k])
                if r is not None:
                    return r
        for v in q.values():
            if isinstance(v, dict):
                r = _find_generator_state(v)
                if r is not None:
                    return r
    return None


if __name__ == '__main__':
    import json
    import sys
    if len(sys.argv) != 3 or sys.argv[1] != 'dispstats':
        raise SystemExit('usage: arm_garch.py dispstats FIT_DIR')
    print(json.dumps(fit_disp_stats(sys.argv[2]), indent=1))
