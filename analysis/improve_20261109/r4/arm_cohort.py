"""Round-4 arms ``cohort`` and ``cohortfix`` and generation mode ``asmadrc`` (r4/PROTOCOL_EXT.md).

Training data: a ``caches/hvg2000x/fold{k}`` cache (r4/build_cache_ext.py): the SEA-AD train donors of
the fold followed by the 31 MADRC ITG donors (GSE268599) as extra TRAIN donors; ``ctx.cohort`` per train
donor (trainer R4-1): 0 = SEA-AD MTG, 1 = MADRC ITG.  Every score stays on SEA-AD held-out donors.

Without either flag a hvg2000x fit is the NAIVE POOLED control: both cohorts are pooled with no cohort
information anywhere (generator and critic see the released stage-only condition), so the generator
has to fit the 55% MADRC / 45% SEA-AD donor mixture and its SEA-AD generation is that mixture.

Cohort indicator: condition SLOT 1
==================================
The 7-slot condition vector is shared by the generator's donor FiLM MLP and the critic (trainer:
``Context.batch`` returns one ``c`` used for the fake bags and for D(real, c), D(fake, c) and the
penalty).  Slot 1 is bench_core's Thal/5 slot of ``raw_c``, but the CONDITION built by
``arms.build_conditions`` writes 0 into slots 0, 1 and 2 for every flag, and no r3/r4 plug-in or mode
writes or reads slot 1 (checked by grep over r4/*.py: the only writer of slots 0-2 is the worked
example ``arm_example.py``, slot 0; stage modes touch slot 3, covariate arms/modes slots 4-6; no module
reads raw_c slots 0-2).  The two arms therefore put the cohort indicator into condition slot 1:

    c[:, 1] = cohort of the batch donor (0.0 SEA-AD, 1.0 MADRC)

through hook (b) ``batch_condition`` (after every hook (a), so it composes with ``nostage``, whose
hook (a) zeroes all slots, and with covariate arms), for BOTH half-updates.  Hook (a) cannot be used:
it sees raw_c rows only, which carry no cohort.  Hook (b) gives the arm its own numpy stream
(arms.py); these arms draw nothing from it, so every released training stream is untouched and the
donor sampling, bag cells and library sizes are bitwise those of the naive pooled fit with the same
seed.  ``ctx.conditions`` (the per-donor matrix before hook b) has slot 1 = 0; only the diagnostic
gm_diagnose.py reads it directly.  The first batch checks that slot 1 is still 0 before writing it.

Generation.  Targets are the 37 SEA-AD donors (generate.py GR4-1); their default condition is built by
hooks (a) and (e) from raw_c, so slot 1 = 0: SEA-AD targets are generated as cohort 0 in every mode
(default, snapens, stagehalf, stageblind, condmix, ema@D, biascal ...).  The diagnostic mode
``asmadrc[@R]`` generates the same targets (same streams, same depth pools) with slot 1 = 1, i.e.
"as MADRC", on the read-out states R (``stagehalf``'s read-out rule: none = primary read-out, i.e.
snapens for constant-lr fits; or final / raw / snapens / ema<D>).

``cohort`` (learned cohort path)
================================
The generator receives the indicator through its donor FiLM MLP input (like the stage), so it can
learn any cohort shift; the critic receives it in its condition input and compares MADRC real bags
with MADRC-conditioned fakes and SEA-AD with SEA-AD ("like with like").  Nothing else changes.

``cohortfix`` (fixed, non-learned per-gene cohort logit offset)
=============================================================
Why.  Rounds 2-3 showed that learned stage paths under this optimizer inflate: normalised zero-
momentum DualNorm moves every parameter by a fixed amount per update, the stage offset / anchored
tables kept growing (stageanchor gains 5-62 in fold 0) and the generators' stage deviations were 3-8x
the real ones.  The cohort difference, unlike the stage effect, is LARGE AND WELL DETERMINED in the
train donors (measured with ``python3 -B r4/arm_cohort.py measure``, train donors only, all three
folds): stage-adjusted cohort log-ratio RMS 0.64-0.68 over the 2,000 panel genes (as large as the
within-cohort donor residual SD, 0.68-0.69, and larger than the raw High-minus-Not-AD contrast,
0.47-0.59 SEA-AD / 0.38 MADRC); split-half correlation of the shrunk per-gene offset below
0.83-0.84 (mean over 50 stage-stratified donor splits, minimum 0.73; each half has half the donors,
so the full-data offset is more reliable still); the within-stage cohort effects agree across the
four stages (pairwise correlation 0.59-0.79, i.e. an additive, stage-independent offset is a fair
description); empirical-Bayes weights tau2/(tau2+se2) average 0.91 (shrunk RMS 0.58-0.62).
(Numbers above: cache v1.  Cache v2 -- mg-select-v3, <= 1,000 cells per MADRC donor -- gives raw RMS
0.62 / 0.62 / 0.65, shrunk 0.56 / 0.56 / 0.59, split-half 0.83 / 0.82 / 0.83, mean weight 0.90-0.91.  These
are estimator-specific: the comparable, noise-corrected log1p-CP10k offset is 0.30, see README_EXT.txt.)
Output: smoke_r4/cohort_offset_measure.json.  Its top genes are ambient-RNA genes (up in
MADRC: GFAP, AQP4, ALDH1L1, FGFR3, ATP1A2 - astrocyte; down: KCNA1, LINC00507, POU3F3 - neuronal),
matching the fetch task's ambient finding.  A per-gene offset estimated once from the train donors
therefore carries the cohort shift with little estimation noise and cannot grow.

What.  ``cohortfix`` = the slot-1 indicator for the CRITIC (as ``cohort``) plus a generator that
(i) does NOT see slot 1 in its donor FiLM input (slot 1 is zeroed before the FiLM MLP, so the
generator has no learned cohort path), and (ii) adds a fixed buffer ``cohort_offset`` [G] to the
output logits of MADRC-conditioned donors:

    logits = out(h) + c[:, 1] * cohort_offset        (c[:, 1] in {0, 1}; SEA-AD logits unchanged)

``cohort_offset`` is a registered BUFFER (not a parameter: no optimizer, no gradient, saved in every
checkpoint and snapshot, copied by the shadow EMA), computed in ``setup`` from the TRAIN donors only:

    C_d = summed panel counts of train donor d (float64; exact integer sums, device independent),
    r_dg = log((C_dg + 1) / (N_d + G))                                (offset_init pseudocount)
    per gene OLS  r_dg = a_g + b_g [cohort_d = 1] + sum_{s=1..3} e_sg [stage_d = s]   (all train donors)
    se_g = SE of b_g;  tau2 = max(1e-6, mean_g(b_g^2 - se_g^2));  w_g = tau2 / (tau2 + se_g^2)
    cohort_offset_g = w_g * b_g      (stage-adjusted, empirical-Bayes shrunk MADRC-minus-SEA-AD log-ratio)

Design risk (review finding F9, accepted, not a bug).  The generator's ONLY cohort path is this
additive per-gene logit offset; its dispersion and FiLM have no cohort input.  Any non-additive
cohort difference (dispersion, composition, depth-dependent detection) can only be absorbed by the
shared base and can therefore leak into SEA-AD (cohort-0) generation.  Read a cohortfix unit's
held-out scores together with (i) its SEA-AD TRAIN-target fit (evaluate_ext ``train`` block, which
is SEA-AD train donors only) against the hvg2000 control and the naive pooled arm, and (ii) the
``asmadrc`` diagnostic; a SEA-AD train-fit loss relative to the control is attributed to this.
Note also that the offset is estimated on cache v2 (<= 1,000 cells per MADRC donor).

Stage adjustment matters because the cohorts' stage mixes differ (MADRC 7/5/6/13, SEA-AD train
2-3/3-4/6/12-13).  The softmax makes the offset's gene mean irrelevant.  The fixed offset is the only
cohort path in the generator; SEA-AD generation (slot 1 = 0) is untouched by it, and ``biascal``
(which shifts G.out[-1].bias) composes.  Conflicts: ``cohort``, and generator-class arms that replace
the head (the offset is added after ``logits``).  Initial tensors of all released parameters are
unchanged (the buffer is created after super().__init__ without any random draw).

Cohort-balanced donor sampling: NOT provided.  The released rule draws donors uniformly over all
train donors (55-56 per fold: 24-25 SEA-AD, 31 MADRC), so MADRC makes about 55% of every batch.  A
balanced rule would change each donor's sampling weight by about +-11% (1/50 SEA-AD and 1/62 MADRC
instead of 1/56) - small next to the cohort shift itself, which the indicator (critic) and the
offset/FiLM (generator) are there to absorb - and would need a new sampling hook in the trainer.

Both arms refuse a cache without external donors (the indicator would be constant 0).  Composable
with ``pbulk``, ``genemean``, ``nostage`` and the ``t.``/``g.``/``d.`` flags (cohortfix: g. arms that
keep ``logits`` as the head).

``extperm`` (control arm, review finding F1, 2026-11-11)
=======================================================
Question it answers: does a hvg2000x gain come from the MADRC STAGE LABELS, or only from having 31
more donors / 3x more cells?  ``extperm`` keeps every external donor, cell, batch and random draw,
but the stage the generator and critic see for an EXTERNAL donor (condition slot 3) is the stage of
another external donor under a fixed permutation:

    c[:, 3] = raw_c[src[d], 3] (float32) for external batch donors d; SEA-AD donors unchanged.

The permutation is drawn once per fold from ``default_rng([20261111, fold, k])`` for k = 0, 1, ...,
taking the first permutation of the 31 external donors whose permuted stage index correlates with
the true one at |Pearson| < 0.1 (so the stage marginal is kept and the stage-label link is
broken); ``info['perm']`` records it.  It acts through hook (b) for both half-updates, draws no
random numbers, and does not touch generation (SEA-AD targets keep their real stage).  The first
batch checks that slot 3 still equals ADNC/3 (no other arm rewrote it).  Conflicts: ``nostage``
(slot 3 would be 0).  Pre-registered comparison (README_EXT.txt): ``cohortfix,pbulk`` vs
``cohortfix,extperm,pbulk`` on the same fold and seed (evaluate_ext pairs them); a stage gain that is
not larger than in this control is not credited to MADRC stage information.  The cohortfix offset
is estimated with the TRUE stages in both arms, so the two fits share the same offset bitwise.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import torch

import arms
from arms import Arm, register, register_mode
import mode_stagedefault as SD   # parse_readout / readout_states (stagehalf's read-out rule)

SLOT = 1
NAME = 'cohort'
NAME_FIX = 'cohortfix'
MODE = 'asmadrc'
NAME_PERM = 'extperm'
STAGE_SLOT = 3
PSEUDOCOUNT = 1.0


def _check_cache(ctx, flag):
    coh = np.asarray(getattr(ctx, 'cohort', np.zeros(len(ctx.ids), np.int64)))
    if not (coh == 1).any() or not (coh == 0).any():
        raise ValueError(f'{flag}: the cache has no external (cohort 1) train donors; use a hvg2000x cache')
    return coh


def cohort_offset(X, by, stages, cohort, pseudocount=PSEUDOCOUNT):
    """Stage-adjusted, empirical-Bayes shrunk MADRC-minus-SEA-AD per-gene log-ratio (module docstring).

    X: [cells, G] tensor (any device); by: row arrays per train donor; stages, cohort: int per donor.
    Returns (offset float32 [G], info dict).  Deterministic: integer counts summed exactly in float64."""
    Xt = torch.as_tensor(X)
    C = torch.stack([Xt[torch.as_tensor(r, device=Xt.device)].double().sum(0) for r in by]).cpu().numpy()
    stages = np.asarray(stages, dtype=int)
    cohort = np.asarray(cohort, dtype=int)
    n, G = C.shape
    a = float(pseudocount)
    N = C.sum(1)
    r = np.log(C + a) - np.log(N[:, None] + a * G)
    present = [s for s in (1, 2, 3) if (stages == s).any()]
    D = np.column_stack([np.ones(n), cohort.astype(float)] + [(stages == s).astype(float) for s in present])
    if np.linalg.matrix_rank(D) < D.shape[1] or n - D.shape[1] < 1:
        raise ValueError('cohort offset design is rank deficient')
    XtXi = np.linalg.inv(D.T @ D)
    B = XtXi @ D.T @ r
    res = r - D @ B
    df = n - D.shape[1]
    v = (res ** 2).sum(0) / df
    se2 = v * XtXi[1, 1]
    b = B[1]
    tau2 = max(1e-6, float(np.mean(b ** 2 - se2)))
    w = tau2 / (tau2 + se2)
    off = (w * b).astype(np.float32)
    info = {'formula': 'w_g b_g; b_g = cohort coefficient of per-gene OLS of donor log((C+1)/(N+G)) on cohort + stage '
                       'dummies over all train donors; w_g = tau2/(tau2+se_g^2)',
            'pseudocount': a, 'n_donors': int(n), 'n_external': int(cohort.sum()), 'df': int(df), 'tau2': tau2,
            'raw_rms': float(np.sqrt(np.mean(b ** 2))), 'shrunk_rms': float(np.sqrt(np.mean(off.astype(np.float64) ** 2))),
            'mean_weight': float(w.mean()), 'min_weight': float(w.min()),
            'median_abs_t': float(np.median(np.abs(b) / np.sqrt(se2))),
            'residual_sd_mean': float(np.sqrt(v).mean())}
    return off, info


class _CohortBase(Arm):
    value_policy = 'none'

    def setup(self, ctx):
        self.cohort = _check_cache(ctx, self.flag)
        self._cohort_dev = torch.as_tensor(self.cohort.astype(np.float32), device=ctx.device)
        self._checked = False
        self.info = {'condition_slot': SLOT, 'indicator': 'c[:, 1] = cohort of the batch donor (0 SEA-AD MTG, 1 MADRC ITG)',
                     'hook': 'batch_condition (b), both half-updates; generation targets keep slot 1 = 0 (SEA-AD)',
                     'n_train_donors_by_cohort': [int((self.cohort == 0).sum()), int((self.cohort == 1).sum())],
                     'donor_sampling': 'released uniform over all train donors (no cohort balancing)'}

    def batch_condition(self, ctx, c, ds, rng, phase):
        if not self._checked:
            if bool((c[:, SLOT] != 0).any()):
                raise ValueError(f'{self.flag}: condition slot {SLOT} is already used by another arm')
            self._checked = True
        out = c.clone()
        out[:, SLOT] = self._cohort_dev[torch.as_tensor(ds, device=c.device)]
        return out

    def state(self):
        return {'cohort': torch.as_tensor(self.cohort), 'slot': SLOT}


@register
class CohortArm(_CohortBase):
    """Learned cohort path: generator FiLM and critic see the indicator in condition slot 1."""
    name = NAME
    conflicts = (NAME_FIX, 'offset')

    def setup(self, ctx):
        super().setup(ctx)
        self.info['generator'] = 'donor FiLM MLP input includes slot 1 (learned cohort path)'


@register
class CohortFixArm(_CohortBase):
    """Critic sees the indicator; generator: no FiLM cohort input, fixed train-estimated logit offset for cohort 1."""
    name = NAME_FIX
    conflicts = (NAME, 'offset', 'covars')

    def setup(self, ctx):
        super().setup(ctx)
        self.offset, oinfo = cohort_offset(ctx.X, ctx.by, arms.stage_of(ctx.raw_c), self.cohort)
        self.info.update({'generator': 'FiLM input slot 1 zeroed; logits += c[:,1] * cohort_offset (buffer, not learned)',
                          'offset': oinfo})

    def generator_class(self, ctx, cls):
        off = torch.as_tensor(self.offset).clone()

        class CohortFixGenerator(cls):
            """Generator with a fixed per-gene cohort logit offset (arm_cohort.py docstring)."""

            def __init__(self, *a, **kw):
                super().__init__(*a, **kw)
                if off.shape != (self.out[-1].out_features,):
                    raise ValueError('cohort offset width differs from the generator gene count')
                self.register_buffer('cohort_offset', off.clone())

            def film(self, z, c):
                c0 = c.clone()
                c0[:, SLOT] = 0
                return super().film(z, c0)

            def logits(self, h, z, c):
                return super().logits(h, z, c) + c[:, SLOT, None, None] * self.cohort_offset

        return CohortFixGenerator

    def state(self):
        return {**super().state(), 'cohort_offset': torch.as_tensor(self.offset)}


def external_stage_permutation(stages, cohort, fold, max_abs_r=0.1):
    """src[d]: donor whose stage donor d shows (identity for cohort 0). Module docstring, ``extperm``."""
    stages = np.asarray(stages, int)
    ext = np.flatnonzero(np.asarray(cohort) == 1)
    st = stages[ext].astype(float)
    for k in range(10000):
        perm = np.random.default_rng([20261111, int(fold), k]).permutation(len(ext))
        r = float(np.corrcoef(st, st[perm])[0, 1]) if st.std() > 0 else 0.0
        if abs(r) < max_abs_r:
            src = np.arange(len(stages))
            src[ext] = ext[perm]
            return src, {'draw': k, 'pearson_true_vs_shown_stage': r,
                         'fraction_external_shown_own_stage': float((st == st[perm]).mean()),
                         'shown_stage_counts': np.bincount(st[perm].astype(int), minlength=4).tolist(),
                         'true_stage_counts': np.bincount(st.astype(int), minlength=4).tolist()}
    raise ValueError('no permutation with |r| below the bound')


@register
class ExtPermArm(Arm):
    """Control: external donors keep their cells but show a permuted (fixed) external stage label."""
    name = NAME_PERM
    value_policy = 'none'
    conflicts = ('nostage',)

    def setup(self, ctx):
        coh = _check_cache(ctx, self.flag)
        st = arms.stage_of(ctx.raw_c)
        self.src, pinfo = external_stage_permutation(st, coh, ctx.fold)
        shown = ctx.raw_c[self.src, STAGE_SLOT].astype(np.float32)
        self._shown = torch.as_tensor(shown, device=ctx.device)
        self._true = torch.as_tensor(ctx.raw_c[:, STAGE_SLOT].astype(np.float32), device=ctx.device)
        self._checked = False
        self.info = {'condition_slot': STAGE_SLOT, 'hook': 'batch_condition (b), both half-updates',
                     'rule': 'external donor d shows raw_c[src[d], 3]; SEA-AD donors and generation unchanged',
                     'perm': pinfo, 'src_external': [str(ctx.ids[i]) for i in self.src[coh == 1]]}

    def batch_condition(self, ctx, c, ds, rng, phase):
        idx = torch.as_tensor(ds, device=c.device)
        if not self._checked:
            if not torch.equal(c[:, STAGE_SLOT], self._true[idx]):
                raise ValueError('extperm: condition slot 3 is not ADNC/3 (another arm rewrote the stage)')
            self._checked = True
        out = c.clone()
        out[:, STAGE_SLOT] = self._shown[idx]
        return out

    def state(self):
        return {'src': torch.as_tensor(self.src)}


# ------------------------------------------------------------------ generation mode asmadrc[@R]
def _asmadrc_plan(ctx, value, target, rng, states):
    flags = {arms.flag_name(f) for f in getattr(ctx, 'flags', ())}
    if not flags & {NAME, NAME_FIX}:
        raise ValueError('asmadrc needs a cohort or cohortfix fit')
    names = SD.readout_states(ctx, value, target, states)
    cond = np.repeat(target.default_cond[None], target.M, 0).astype(np.float32)
    if (cond[:, SLOT] != 0).any():
        raise ValueError('asmadrc: default condition slot 1 is not 0')
    cond[:, SLOT] = 1.0
    return cond, names


register_mode(MODE, _asmadrc_plan, value_policy='optional', parse_value=SD.parse_readout,
              doc='diagnostic: the SEA-AD targets generated AS MADRC (condition slot 1 = 1) on the primary read-out states '
                  '(or @final/@raw/@snapens/@ema<D>); same streams and depth pools as the default generation')


# ------------------------------------------------------------------ measurement CLI (train donors only)
def measure(folds=(0, 1, 2), n_split=50, out=None):
    """Cohort offset vs stage effect on the TRAIN donors of hvg2000x folds (no held-out donor is read)."""
    import scipy.sparse as sp
    root = '/ml2/alz/runs/improve_20261109/caches/hvg2000x/fold%d/train/'
    res = {}
    for k in folds:
        m = np.load(root % k + 'metadata.npz', allow_pickle=False)
        C = sp.load_npz(root % k + 'counts.npz').tocsr()
        ids, dl = m['donors'].astype(str), m['donor'].astype(str)
        coh = m['cohort'].astype(int)
        st = arms.stage_of(m['raw_c'])
        by = [np.flatnonzero(dl == d) for d in ids]
        X = torch.as_tensor(C.toarray())
        off, info = cohort_offset(X, by, st, coh)
        P = torch.stack([X[torch.as_tensor(r)].double().sum(0) for r in by]).numpy()
        G = P.shape[1]
        r = np.log(P + 1) - np.log(P.sum(1, keepdims=True) + G)
        sea, mad = coh == 0, coh == 1
        hn = {c: r[(coh == c) & (st == 3)].mean(0) - r[(coh == c) & (st == 0)].mean(0) for c in (0, 1)}
        ws = [r[mad & (st == s)].mean(0) - r[sea & (st == s)].mean(0) for s in range(4)]
        cc = np.corrcoef(ws)
        rng = np.random.default_rng(0)
        sh = []
        for _ in range(n_split):
            h = np.zeros(len(ids), bool)
            for c in (0, 1):
                for s in range(4):
                    idx = np.flatnonzero((coh == c) & (st == s))
                    rng.shuffle(idx)
                    h[idx[:len(idx) // 2]] = True
            o1, _ = cohort_offset(X, [by[i] for i in np.flatnonzero(h)], st[h], coh[h])
            o2, _ = cohort_offset(X, [by[i] for i in np.flatnonzero(~h)], st[~h], coh[~h])
            sh.append(float(np.corrcoef(o1, o2)[0, 1]))
        order = np.argsort(-off)
        res[f'fold{k}'] = {**info,
                           'high_minus_notad_rms': {'SEA-AD': float(np.sqrt((hn[0] ** 2).mean())), 'MADRC': float(np.sqrt((hn[1] ** 2).mean()))},
                           'high_minus_notad_corr_between_cohorts': float(np.corrcoef(hn[0], hn[1])[0, 1]),
                           'within_stage_cohort_effect_corr': np.round(cc, 3).tolist(),
                           'split_half_corr_shrunk_offset': {'mean': float(np.mean(sh)), 'min': float(np.min(sh)), 'n': n_split},
                           'top_up_in_MADRC': [str(m['genes'][i]) for i in order[:15]],
                           'top_down_in_MADRC': [str(m['genes'][i]) for i in order[::-1][:15]]}
        print(json.dumps({f'fold{k}': {kk: res[f'fold{k}'][kk] for kk in ('raw_rms', 'shrunk_rms', 'mean_weight', 'residual_sd_mean',
                                                                           'high_minus_notad_rms', 'split_half_corr_shrunk_offset')}}), flush=True)
    if out:
        os.makedirs(os.path.dirname(out), exist_ok=True)
        with open(out, 'w') as f:
            json.dump(res, f, indent=1)
    return res


if __name__ == '__main__':
    sys.modules.setdefault('arm_cohort', sys.modules['__main__'])
    if len(sys.argv) > 1 and sys.argv[1] == 'measure':
        torch.set_num_threads(3)
        measure(out=sys.argv[2] if len(sys.argv) > 2 else '/ml2/alz/runs/improve_20261109/smoke_r4/cohort_offset_measure.json')
    else:
        print(__doc__)
