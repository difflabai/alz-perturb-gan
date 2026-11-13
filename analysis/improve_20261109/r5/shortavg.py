"""Round-5 work item 1: when to stop and what to average (no new training).

PROTOCOL_R5.md work item 1, exactly as pre-declared:
    candidates  snap@U and snapema@U:D, U in {5000, 10000, 15000}, D in {0.999, 0.9995}
    references  snapens (15000-19000 + final), ema@0.9995, final (the 20,000-update state)
    selection   best candidate by held-out pb_rmse, folds 0-2, seed 0, of the control pbulk
    confirm     control seed 1; g.dispz,pbulk,t.kappa@0.25 seeds 0-3; later folds 3-8 (pbulk, seeds 0-1)
    rule        replace snapens only if better on the confirmation sets by > 2 seed-noise SE and in
                both extra-fold repeats (folds 3-5 and 6-8).

Every read-out is ALSO generated stage-blind ("sb:") and with train-only bias calibration ("bc:"),
and scored with the round-3 evaluator's own functions (evaluate.load_real / gene_subsets /
pooled_gene_set / dam_context / align_generated / score_generated / headline), the same settings as
results_r3 (pc_repeats 24, cell_repeats 4, dam draws 2000).

Generation (the r5 core is not changed)
---------------------------------------
* Modes ``snap@U`` and ``snapema@U:D`` of r5/arms.py accept U = 5000, 10000, 15000 on a 20,000-update
  round-3 fit (they only need the state to exist; checked by ``test_shortavg.py modes``).
* ``generate.generate_fit`` is called in-process with ``out=`` under smoke_r5/shortavg/modes.  The r5
  copy of trainer.check_out_root still lists only the round-3 output roots (fits_r3, smoke_r3, ...),
  so it refuses smoke_r5.  This module adds 'smoke_r5' to ``trainer.ALLOWED_OUT_ROOTS`` IN THIS PROCESS
  ONLY (no file edit); the frozen-tree checks of generate._check_out are unchanged.
* Stage-blind on a snapshot state: ``stageblind@R`` only takes R in {final, raw, snapens, ema<D>}, so
  snap/snapema states cannot be composed.  This module registers the generation mode
  ``sblind@<state>`` (state = final | snapens | snap_<U> | snapema_<U>_<D> | ema_<D>) whose plan is
  mode_stagedefault's own rule, ``mixed_conditions(ctx, target, 1.0, names)`` (= condmix@1
  conditions for a single state, the round-robin layout for snapens).  Same generation streams (the
  target's default streams; generate.generate_arrays).  Checked bitwise: sblind@final ==
  condmix@1, sblind@snapens == stageblind (test_shortavg.py repro).
* Bias calibration: ``biascal@snap:U``, ``biascal@snapema:U:D``, ``biascal@ema:D``,
  ``biascal@snapens``, ``biascal`` (final) and ``biascal@sblind:<state>`` (calibrated stage-blind),
  i.e. mode_biascal unchanged with the candidate's plan as INNER.
* GPU generation (``--device cuda``, memory fraction 0.04), one process; mode files made on CUDA are
  CRN-paired with the fits' own CUDA-made mode files (generate.py FID-1).

EMA warm-up.  The shadow EMAs start at the initial weights (trainer.ShadowEMA copies G, P at
construction), so snapema@U:D keeps the share D**U on the untrained initial weights:
0.9995**5000 = 0.082, 0.999**5000 = 0.0067, 0.9995**10000 = 0.0067, 0.999**10000 = 4.5e-5.
Every score record carries it (``init_share``).

CLI (bash; run from /tmp)::

    PY="/mnt/nvme/ai/conda/anaconda3/bin/python3 -B"; R5=/ml2/alz/analysis/improve_20261109/r5
    # generate on the GPU and score with 3 CPU processes (one per fold), deleting mode files once scored
    $PY $R5/shortavg.py run --set main --arms pbulk,g.dispz,pbulk,t.kappa@0.25 --folds 0,1,2 --seeds 0,1,2,3
    $PY $R5/shortavg.py run --set curve60k --folds 0,1,2 --seeds 0
    $PY $R5/shortavg.py run --set diag --folds 0,1,2 --seeds 0,1,2,3       # post-hoc: untrained weights, debiased EMAs
    $PY $R5/shortavg.py run --set main --folds 0,1,2 --seeds 0,1,2,3 --variants plain \
        --bcsb 'snapema@5000:0.9995;snapens;ema@0.9995'                    # calibrated stage-blind for the chosen + references
    $PY $R5/shortavg.py analyze            # -> smoke_r5/shortavg/analysis.json and analysis.txt
    # folds 3-8, once the confirmation screen has finished (incomplete fits are skipped and listed):
    $PY $R5/shortavg.py run --set main --arms pbulk --folds 3,4,5,6,7,8 --seeds 0,1
    $PY $R5/shortavg.py analyze
    $PY $R5/shortavg.py keep --readout <chosen> --arm pbulk --folds 0,1,2 --seeds 0,1   # re-make the kept files

Round-5 repair (adversarial review F4, F6), analysis/report only (no score changes):
* F6: every spread statistic now carries its gene subset in its name and in the report legend:
  ``_all`` = all 2,000 panel genes, ``_well`` = the N_WELL_EXPRESSED (256) genes with the highest train
  grand-mean log CP10k.  Both subsets are reported for every spread statistic (before, the trade-off table
  mixed sd_within_target_gen_well with sd_between_targets_pred_all and the 10-character column header hid it).
* F4: pb_rmse, pc_energy_rep and cell_energy_rep all reward collapse toward the pooled profile (the
  untrained generator beats snapens on pb_rmse and cell_energy_rep and ties it on pc_energy_rep).  The
  analysis adds ``collapse_check``: init minus snapens and chosen minus snapens on the guard metrics next to
  the spread / coverage metrics (pc_spread_rep, pc_coverage80_rep, generated within-target SD over real
  between-donor SD at the same depth, both subsets).  A "no worse than 2 SE" guard on pb_rmse / pc_energy /
  cell_energy is only read together with these; a read-out that "improves" the guards while its spread
  ratio falls well below 1 is a collapse, not a better generator.

Arm lists are separated by ';' when an arm contains ',' (``--arms 'pbulk;g.dispz,pbulk,t.kappa@0.25'``).
"""
from __future__ import annotations

import os
import sys

if __name__ == '__main__':
    for _k in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
        os.environ[_k] = '3'
HERE = os.path.dirname(os.path.abspath(__file__))
if HERE in sys.path:
    sys.path.remove(HERE)
sys.path.insert(0, HERE)
sys.dont_write_bytecode = True

import argparse  # noqa: E402
import gc  # noqa: E402
import json  # noqa: E402
import subprocess  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

RUNS = Path('/ml2/alz/runs/improve_20261109')
FITS = RUNS / 'fits_r3'
CACHES = RUNS / 'caches'
BASELINES = RUNS / 'results_r3' / 'baselines'
OUT = RUNS / 'smoke_r5' / 'shortavg'
PANEL = 'hvg2000'
PY = '/mnt/nvme/ai/conda/anaconda3/bin/python3'

CONTROL = 'pbulk'
COMBO = 'g.dispz,pbulk,t.kappa@0.25'
LONG_ARMS = ('pbulk,t.steps@60000', 't.steps@60000')
CAND_U = (5000, 10000, 15000)
CAND_D = (0.999, 0.9995)
CANDIDATES = tuple([f'snap@{u}' for u in CAND_U] + [f'snapema@{u}:{d:g}' for u in CAND_U for d in CAND_D])
REFERENCES = ('snapens', 'ema@0.9995', 'final')
CURVE_EXTRA = ('ema@0.999',)            # the 20,000-update point of the D = 0.999 curve
VARIANTS = ('', 'sb', 'bc')             # plain, stage-blind, bias-calibrated
LONG_U = tuple(range(5000, 50001, 5000))
SIX = [('pc_energy_rep', 'lower'), ('cell_energy_rep', 'lower'), ('pc_coverage80_rep', 'cov80'),
       ('meanlog_rmse_gene', 'lower'), ('nonzero_rmse_gene', 'lower'), ('logvar_rmse_gene', 'lower')]
EXTRA_KEYS = ['pb_rmse', 'pb_rmse_recal', 'pc_spread_rep', 'pbdist_spread', 'pbdist_energy', 'pbdist_coverage80',
              'meanlog_bias', 'nonzero_bias', 'logvar_bias', 'pb_dev_pearson']
# repair F6: every spread statistic in both gene subsets (_all = 2,000 panel genes; _well = top-256 by train mean)
SPREAD_KEYS = [f'{k}_{tag}' for k in ('dist_pred_to_grand', 'dist_obs_to_grand', 'sd_between_targets_pred', 'sd_between_donors_obs',
                                      'sd_within_target_gen', 'sd_between_donors_realK_train', 'sd_between_donors_realK_heldout')
               for tag in ('all', 'well')]
GUARD_KEYS = ['pb_rmse', 'pc_energy_rep', 'cell_energy_rep']   # PROTOCOL_R5 "no worse than 2 SE" guards (repair F4)
MAX_PENDING = 30                        # generated, unscored mode files (25 MB each) before generation waits


# =========================================================================== read-out grammar
def split_readout(r):
    """'sb:snap@5000' -> ('sb', 'snap@5000'); 'final' -> ('', 'final')."""
    v, sep, base = r.partition(':')
    if sep and v in ('sb', 'bc', 'bcsb', 'db'):
        return v, base
    return '', r


def parse_base(base):
    """-> (kind, U, D): final | snapens | ema (D) | snap (U) | snapema (U, D)."""
    if base in ('final', 'snapens', 'init'):
        return base, None, None
    name, _, val = base.partition('@')
    if name == 'ema':
        return 'ema', None, float(val)
    if name == 'snap':
        return 'snap', int(val), None
    if name == 'snapema':
        u, _, d = val.partition(':')
        return 'snapema', int(u), float(d)
    raise ValueError(f'unknown base read-out {base!r}')


def normalise_base(base, updates):
    """snap@U at U = run length is the final state; snapema@U:D at U = run length is ema@D."""
    kind, u, d = parse_base(base)
    if kind == 'snap' and u == updates:
        return 'final'
    if kind == 'snapema' and u == updates:
        return f'ema@{d:g}'
    return base


def state_token(base):
    """Model-state name used by generate.py for a single-state base read-out (or 'snapens')."""
    kind, u, d = parse_base(base)
    if kind in ('final', 'snapens', 'init'):
        return kind
    return {'ema': lambda: f'ema_{d:g}', 'snap': lambda: f'snap_{u}', 'snapema': lambda: f'snapema_{u}_{d:g}'}[kind]()


def init_share(base, updates):
    """Weight left on the initial weights by the shadow EMA of a read-out (0 for raw states)."""
    kind, u, d = parse_base(base)
    if kind == 'init':
        return 1.0
    if kind == 'snapema':
        return float(d ** u)
    if kind == 'ema':
        return float(d ** updates)
    return 0.0


def gen_mode(variant, base):
    """Generation mode text for (variant, base)."""
    kind, u, d = parse_base(base)
    if kind == 'init':
        if variant:
            raise ValueError('init takes no variant')
        return 'winit'
    plain = {'final': 'raw', 'snapens': 'snapens'}.get(kind) or {
        'ema': lambda: f'ema@{d:g}', 'snap': lambda: f'snap@{u}', 'snapema': lambda: f'snapema@{u}:{d:g}'}[kind]()
    if variant == '':
        return plain
    if variant == 'sb':
        return f'sblind@{state_token(base)}'
    if variant == 'bc':
        if kind == 'final':
            return 'biascal'
        return 'biascal@' + plain.replace('@', ':', 1)
    if variant == 'bcsb':
        return f'biascal@sblind:{state_token(base)}'
    if variant == 'db':
        if kind not in ('snapema', 'ema'):
            raise ValueError('db: EMA read-outs only')
        return f'debias@{state_token(base)}'
    raise ValueError(variant)


def existing_file(fit, variant, base):
    """A mode file the fit already holds for this read-out (read-only use), or None."""
    fit = Path(fit)
    kind, u, d = parse_base(base)
    names = []
    if variant == '':
        names = ['generated.npz'] if kind == 'final' else [f'generated_{gen_mode("", base)}.npz']
    elif variant == 'sb':
        names = {'final': ['generated_condmix@1.npz', 'generated_stageblind@final.npz'],
                 'snapens': ['generated_stageblind.npz', 'generated_stageblind@snapens.npz']}.get(kind, [])
    for n in names:
        p = fit / n
        if p.exists() and (n == 'generated.npz' or p.with_name(p.stem + '.receipt.json').exists()):
            return p
    return None


def safe(r):
    return r.replace('@', '_at_').replace(':', '__')


def readout_id(variant, base):
    return f'{variant}:{base}' if variant else base


# =========================================================================== jobs
def fit_dir(arm, fold, seed):
    return FITS / arm / PANEL / f'fold{fold}' / f'seed{seed}'


def fit_complete(fit):
    """True for a finished round-3 fit (never touch incomplete fits)."""
    fit = Path(fit)
    try:
        rec = json.loads((fit / 'fit_receipt.json').read_text())
        cfg = json.loads((fit / 'config.json').read_text())
    except (OSError, json.JSONDecodeError):
        return False
    if rec.get('status') != 'completed' or not (fit / 'final.pt').exists() or not (fit / 'generated.npz').exists():
        return False
    try:
        prog = json.loads((fit / 'progress.json').read_text())
        if int(prog.get('step', cfg['updates'])) != int(cfg['updates']):
            return False
    except (OSError, json.JSONDecodeError, ValueError):
        pass
    return True


def score_path(arm, fold, seed, rid):
    return OUT / 'scores' / arm / f'fold{fold}' / f'seed{seed}' / f'{safe(rid)}.json'


def mode_path(arm, fold, seed, rid):
    return OUT / 'modes' / arm / f'fold{fold}_seed{seed}' / f'{safe(rid)}.npz'


def readouts_for(set_name, updates, variants=VARIANTS, bcsb=()):
    out = []
    if set_name == 'main':
        bases = list(CANDIDATES) + list(REFERENCES)
        for v in variants:
            for b in bases:
                out.append((v, b))
        for b in CURVE_EXTRA:
            out.append(('', b))
        for b in bcsb:
            out.append(('bcsb', b))
    elif set_name == 'diag':     # post-hoc diagnostics, not pre-declared
        out += [('', 'init'), ('db', 'snapema@5000:0.9995'), ('db', 'snapema@5000:0.999'), ('db', 'snapema@10000:0.9995')]
    elif set_name == 'curve60k':
        for u in LONG_U:
            out.append(('', f'snap@{u}'))
            for d in CAND_D:
                out.append(('', f'snapema@{u}:{d:g}'))
    else:
        raise ValueError(set_name)
    seen, uniq = set(), []
    for v, b in out:
        b = normalise_base(b, updates)
        if (v, b) not in seen:
            seen.add((v, b)); uniq.append((v, b))
    return uniq


def build_jobs(set_name, arms_, folds, seeds, variants=VARIANTS, bcsb=()):
    """[(job dict)], [skipped fits]."""
    jobs, skipped = [], []
    for arm in arms_:
        for f in folds:
            for s in seeds:
                fit = fit_dir(arm, f, s)
                if not fit.exists():
                    continue
                if not fit_complete(fit):
                    skipped.append(str(fit)); continue
                updates = int(json.loads((fit / 'config.json').read_text())['updates'])
                for v, b in readouts_for(set_name, updates, variants, bcsb):
                    rid = readout_id(v, b)
                    src = existing_file(fit, v, b)
                    jobs.append({'arm': arm, 'fold': f, 'seed': s, 'fit': str(fit), 'updates': updates, 'variant': v, 'base': b,
                                 'readout': rid, 'mode': gen_mode(v, b), 'source': str(src) if src else None,
                                 'out': str(mode_path(arm, f, s, rid)), 'score': str(score_path(arm, f, s, rid)),
                                 'init_share': init_share(b, updates)})
    return jobs, skipped


# =========================================================================== generation
_REGISTERED = False


def register_sblind():
    """Register generation mode sblind@<state> (stage-blind on any single state or snapens; docstring)."""
    global _REGISTERED
    import arms
    import mode_stagedefault as SD
    if _REGISTERED or 'sblind' in arms._MODES:
        _REGISTERED = True
        return

    def parse(text):
        t = str(text)
        ok = t in ('final', 'snapens') or (t.startswith(('snap_', 'snapema_', 'ema_')) and '@' not in t and ':' not in t)
        if not ok:
            raise ValueError(f'sblind state {text!r}: want final, snapens, snap_<U>, snapema_<U>_<D> or ema_<D>')
        return t, t

    def plan(ctx, value, target, rng, states):
        if value == 'snapens':
            names = list(arms._snapens_plan(ctx, None, target, None, states)[1])
        else:
            if value not in states:
                raise ValueError(f'sblind: state {value} not in the fit (have {list(states)})')
            names = [value] * target.M
        cond, _ = SD.mixed_conditions(ctx, target, 1.0, names)     # mode stageblind's own rule (lam = 1)
        return cond, names
    plan.__module__ = __name__
    arms.register_mode('sblind', plan, value_policy='required', parse_value=parse,
                       doc='round-5 shortavg: stage-blind (mode_stagedefault.mixed_conditions, lam 1) on one named state or snapens')

    # ---- post-hoc diagnostics (NOT pre-declared): what the EMA's initial-weight share does
    def initial_models(ctx):
        import trainer
        b, *_ = trainer.build(ctx, ctx.train_seed, stream_root=ctx.stream_root)   # the fit's own initial weights (generator_init / prior_init streams)
        return b[1], b[3]

    def winit_plan(ctx, value, target, rng, states):
        return np.repeat(target.default_cond[None], target.M, 0), ['final'] * target.M

    def winit_prepare(ctx, value, name, G, P):
        G0, P0 = initial_models(ctx)
        G.load_state_dict(G0.state_dict(), strict=True)
        P.load_state_dict(P0.state_dict(), strict=True)
        if hasattr(P, 'set_sigma') and hasattr(P0, 'sigma'):
            P.set_sigma(P0.sigma)

    def debias_parse(text):
        t = str(text)
        if not t.startswith(('snapema_', 'ema_')):
            raise ValueError(f'debias state {text!r}: want snapema_<U>_<D> or ema_<D>')
        return t, t

    def debias_plan(ctx, value, target, rng, states):
        if value not in states:
            raise ValueError(f'debias: state {value} not in the fit')
        ctx._shortavg_updates = int(target.updates)
        return np.repeat(target.default_cond[None], target.M, 0), [value] * target.M

    def debias_prepare(ctx, value, name, G, P):
        parts = value.split('_')
        u, d = (int(parts[1]), float(parts[2])) if parts[0] == 'snapema' else (ctx._shortavg_updates, float(parts[1]))
        s = d ** u
        G0, P0 = initial_models(ctx)
        for live, init in ((G, G0), (P, P0)):
            for p, p0 in zip(live.parameters(), init.parameters()):
                if p0.requires_grad:
                    p.sub_(s * p0.to(p.device)).div_(1 - s)
        ctx.mode_receipt_extra = {'debias': {'state': value, 'initial_share_removed': s}}

    arms.register_mode('winit', winit_plan, prepare=winit_prepare,
                       doc='round-5 shortavg diagnostic: all M from the fit\'s untrained initial weights (default conditions)')
    arms.register_mode('debias', debias_plan, value_policy='required', parse_value=debias_parse, prepare=debias_prepare,
                       doc='round-5 shortavg diagnostic: shadow EMA with the initial-weight share removed, (ema - s w0)/(1 - s), s = D**U')
    _REGISTERED = True


def allow_smoke_r5():
    """In-process only: let generate.py's output-root check accept smoke_r5 (docstring)."""
    import trainer
    if 'smoke_r5' not in trainer.ALLOWED_OUT_ROOTS:
        trainer.ALLOWED_OUT_ROOTS = tuple(trainer.ALLOWED_OUT_ROOTS) + ('smoke_r5',)


def setup_generation(device, gpu_mem_fraction):
    import torch
    import trainer  # noqa: F401  (released-module paths before generate)
    import generate  # noqa: F401
    import mode_biascal  # noqa: F401
    import arm_nostage  # noqa: F401  (registers stagehalf / stageblind)
    import arm_condshrink  # noqa: F401
    torch.set_num_threads(3)
    if device == 'cuda' and gpu_mem_fraction:
        torch.cuda.set_per_process_memory_fraction(gpu_mem_fraction)
    register_sblind()
    allow_smoke_r5()


def pending_files():
    return len(list((OUT / 'modes').rglob('*.npz'))) if (OUT / 'modes').exists() else 0


def generate_jobs(jobs, device='cuda', log=print, max_pending=MAX_PENDING):
    """Generate every job without a score, source or output yet; one fit context at a time."""
    import torch
    import generate as GEN
    from trainer import Context
    todo = [j for j in jobs if j['source'] is None and not Path(j['score']).exists() and not Path(j['out']).exists()
            and not Path(j['out'] + '.failed.json').exists()]
    byfit = {}
    rank = {'': 0, 'sb': 1, 'bc': 2, 'bcsb': 3, 'db': 4}      # all plain read-outs first (the selection needs them), then sb, bc
    for j in sorted(todo, key=lambda j: rank[j['variant']]):
        byfit.setdefault((rank[j['variant']], j['fit']), []).append(j)
    log(f'[shortavg] generating {len(todo)} mode files in {len(byfit)} (variant, fit) groups on {device}')
    for (_, fit), js in byfit.items():
        cfg = json.loads((Path(fit) / 'config.json').read_text())
        cache = CACHES / PANEL / f"fold{js[0]['fold']}"
        ctx = Context(cache, device, tuple(cfg['flags']), allow_nonprotocol_split=not cfg['protocol_split'])
        for j in js:
            while pending_files() >= max_pending:
                time.sleep(10)
            if Path(j['score']).exists() or Path(j['out']).exists():
                continue
            t0 = time.monotonic()
            ctx.mode_receipt_extra = None     # a diagnostic mode's record must not leak into the next receipt
            try:
                GEN.generate_fit(fit, cache, device=device, ctx=ctx, mode=j['mode'], out=j['out'])
                rp = Path(j['out']).with_name(Path(j['out']).stem + '.receipt.json')
                rec = json.loads(rp.read_text())
                rec['shortavg'] = {k: j[k] for k in ('arm', 'fold', 'seed', 'readout', 'init_share')}
                rp.write_text(json.dumps(rec, indent=1))
                log(f"[shortavg] {j['arm']} f{j['fold']} s{j['seed']} {j['readout']} ({j['mode']}) {time.monotonic() - t0:.1f}s")
            except Exception as e:   # noqa: BLE001  (recorded; the scorer skips it)
                Path(j['out']).parent.mkdir(parents=True, exist_ok=True)
                Path(j['out'] + '.failed.json').write_text(json.dumps({'job': j, 'error': repr(e)}, indent=1))
                log(f"[shortavg] FAILED {j['arm']} f{j['fold']} s{j['seed']} {j['readout']}: {e!r}")
        del ctx
        gc.collect()      # the Context sits in reference cycles: without this every fit context stayed on the GPU (OOM after 18)
        if device == 'cuda':
            torch.cuda.empty_cache()
            log(f'[shortavg] CUDA allocated after releasing the fit context: {torch.cuda.memory_allocated() / 2**20:.0f} MiB')


# =========================================================================== scoring
_REAL = {}


def real_fold(fold):
    import evaluate as EV
    if fold not in _REAL:
        _REAL.clear()
        rc = EV.RealCache(CACHES)
        _REAL[fold] = rc.get(PANEL, fold)
    return _REAL[fold]


def spread_extras(gen, real):
    """Between-donor spread and distance to the pooled train profile (held-out targets; see module docstring)."""
    import evaluate as EV
    import metrics as MX
    held, train = real.is_heldout, ~real.is_heldout
    pb = gen['pb_counts'].astype(np.float64)
    T, M, G = pb.shape
    L = np.stack([MX.log_cp10k(pb[t]) for t in range(T)])            # [T,M,G]
    pred = L.mean(1)
    obs = MX.log_cp10k(real.pb)
    grand = obs[train].mean(0)
    K_est = float(np.median(pb.sum(-1) / np.maximum(gen['lib'], 1e-9)))
    K = int(round(K_est)) if np.isfinite(K_est) and K_est > 0 else 256
    rK = MX.log_cp10k(EV.real_pb_subsample(real, K))
    well = np.argsort(-grand, kind='stable')[:min(EV.N_WELL_EXPRESSED, G)]
    out = {}
    for tag, idx in (('all', np.arange(G)), ('well', well)):
        out[f'dist_pred_to_grand_{tag}'] = float(np.sqrt(np.mean((pred[held][:, idx] - grand[idx]) ** 2)))
        out[f'dist_obs_to_grand_{tag}'] = float(np.sqrt(np.mean((obs[held][:, idx] - grand[idx]) ** 2)))
        out[f'sd_between_targets_pred_{tag}'] = float(pred[held][:, idx].std(0, ddof=1).mean())
        out[f'sd_between_donors_obs_{tag}'] = float(obs[held][:, idx].std(0, ddof=1).mean())
        out[f'sd_within_target_gen_{tag}'] = float(L[held][:, :, idx].std(1, ddof=1).mean())
        out[f'sd_between_donors_realK_train_{tag}'] = float(rK[train][:, idx].std(0, ddof=1).mean())
        out[f'sd_between_donors_realK_heldout_{tag}'] = float(rK[held][:, idx].std(0, ddof=1).mean())
    out['K'] = K
    return out


def score_file(path, fold, settings_kw=None):
    """Score one generated file with the r5 evaluator's own functions; returns (flat headline, extras, full result)."""
    import evaluate as EV
    real, subsets, pooled, dam = real_fold(fold)
    kw = settings_kw or {}
    settings = EV.ScoreSettings(kw.get('pc_repeats', EV.DEFAULT_PC_REPEATS), kw.get('cell_repeats', EV.DEFAULT_CELL_REPEATS), dam)
    gen = EV.align_generated(EV.load_generated(path), real)
    res, _ = EV.score_generated(gen, real, subsets, pooled, settings=settings)
    flat = {k: v for k, v in EV.headline(res).items() if '__' not in k}
    return flat, spread_extras(gen, real), res


def score_jobs(jobs, folds, watch=False, log=print, keep=()):
    """Score jobs of the given folds; delete each generated mode file once scored (unless its readout is in keep)."""
    modes_root = (OUT / 'modes').resolve()
    mine = [j for j in jobs if j['fold'] in folds]
    while True:
        left = 0
        for j in mine:
            sp = Path(j['score'])
            if sp.exists():
                continue
            src = Path(j['source']) if j['source'] else Path(j['out'])
            if j['source'] is None and Path(j['out'] + '.failed.json').exists():
                continue
            rec = src.with_name(src.stem + '.receipt.json')
            if not src.exists() or (j['source'] is None and not rec.exists()):
                left += 1
                continue
            t0 = time.monotonic()
            flat, extra, res = score_file(src, j['fold'])
            receipt = json.loads(rec.read_text()) if rec.exists() else None
            out = {'job': j, 'scored_file': str(src), 'generated_sha256': EV_sha(src), 'headline': flat, 'extras': extra,
                   'mode_receipt': {k: receipt.get(k) for k in ('mode', 'states_used', 'generation_device', 'seconds', 'mode_extra')}
                   if receipt else None,
                   'evaluator_version': res['evaluator_version'], 'score_settings': res['score_settings'],
                   'score_seconds': time.monotonic() - t0}
            if out['mode_receipt'] and isinstance(out['mode_receipt'].get('mode_extra'), dict):
                bc = out['mode_receipt']['mode_extra'].get('biascal')
                if bc:
                    bc = {k: v for k, v in bc.items() if k != 'delta'}
                    out['mode_receipt']['mode_extra'] = {'biascal': bc}
            sp.parent.mkdir(parents=True, exist_ok=True)
            tmp = sp.with_name('.' + sp.name + '.partial')
            tmp.write_text(json.dumps(out, indent=1, default=float))
            tmp.replace(sp)
            log(f"[score] {j['arm']} f{j['fold']} s{j['seed']} {j['readout']}: pb_rmse {flat['pb_rmse']:.4f} ({time.monotonic() - t0:.0f}s)")
            if j['source'] is None and j['readout'] not in keep:
                p = src.resolve()
                if modes_root in p.parents:
                    p.unlink(missing_ok=True)
                    rec.unlink(missing_ok=True)
        if not watch or left == 0:
            return left
        if not generator_alive():
            time.sleep(15)
            if not generator_alive() and all(Path(j['score']).exists() or (j['source'] is None and not Path(j['out']).exists())
                                             for j in mine):
                return left
        time.sleep(10)


def EV_sha(path):
    import evaluate as EV
    return EV.sha256(path)


def generator_alive():
    p = OUT / 'generator.pid'
    try:
        pid = int(p.read_text())
        os.kill(pid, 0)
        return True
    except (OSError, ValueError):
        return False


# =========================================================================== analysis
def load_scores():
    """{(arm, fold, seed, readout): record}."""
    out = {}
    for p in (OUT / 'scores').rglob('*.json'):
        try:
            r = json.loads(p.read_text())
        except json.JSONDecodeError:
            continue
        j = r['job']
        out[(j['arm'], j['fold'], j['seed'], j['readout'])] = r
    return out


def metric(S, arm, fold, seed, rid, m='pb_rmse'):
    r = S.get((arm, fold, seed, rid))
    if r is None:
        return None
    v = r['headline'].get(m, r['extras'].get(m))
    return None if v is None else float(v)


def fold_means(S, arm, rid, folds, seeds, m='pb_rmse'):
    out = {}
    for f in folds:
        v = [metric(S, arm, f, s, rid, m) for s in seeds]
        v = [x for x in v if x is not None]
        if v:
            out[f] = float(np.mean(v))
    return out


def paired(S, arm, rid, ref, folds, seeds, m='pb_rmse', s2_external=None):
    """rid - ref on common (fold, seed): fold-balanced mean, per-fold means, seed-noise SE, fold-scatter t.

    seed-noise SE (paired): sqrt(sum_f s2 / k_f) / F with s2 the pooled within-fold variance of the paired
    difference across seeds (folds with >= 2 seeds), else s2_external (e.g. from the control's seeds 0-1).
    evaluator-style SE (repair E5): sqrt(sum_f 2 s2_ref / k_f) / F with s2_ref the pooled within-fold seed
    variance of the reference read-out itself (treats the two read-outs as independent fits).
    fold t: mean of per-fold means / (SD of per-fold means / sqrt(F)), df F - 1.
    """
    d, refv = {}, {}
    for f in folds:
        for s in seeds:
            a, b = metric(S, arm, f, s, rid, m), metric(S, arm, f, s, ref, m)
            if a is not None and b is not None:
                d.setdefault(f, {})[s] = a - b
                refv.setdefault(f, {})[s] = b
    if not d:
        return None
    fm = {f: float(np.mean(list(v.values()))) for f, v in d.items()}
    F = len(fm)
    mean = float(np.mean(list(fm.values())))
    var = [np.var(list(v.values()), ddof=1) for v in d.values() if len(v) >= 2]
    df = sum(len(v) - 1 for v in d.values() if len(v) >= 2)
    if var:
        s2, src = float(np.mean(var)), f'within-fold seed variance of the difference (df {df})'
    elif s2_external is not None:
        s2, src = float(s2_external), 'external (control seeds 0-1)'
    else:
        s2, src = None, None
    se = float(np.sqrt(sum(s2 / len(v) for v in d.values())) / F) if s2 is not None else None
    rvar = [np.var(list(v.values()), ddof=1) for v in refv.values() if len(v) >= 2]
    se_ev = float(np.sqrt(sum(2 * float(np.mean(rvar)) / len(v) for v in d.values())) / F) if rvar else None
    t = None
    if F >= 2:
        sd = float(np.std(list(fm.values()), ddof=1))
        t = mean / (sd / np.sqrt(F)) if sd > 0 else None
    return {'mean_diff': mean, 'per_fold': fm, 'n_pairs': sum(len(v) for v in d.values()), 'n_folds': F,
            'n_folds_better': int(sum(v < 0 for v in fm.values())),
            'n_pairs_better': int(sum(x < 0 for v in d.values() for x in v.values())),
            'seed_se': se, 'seed_se_source': src, 'seed_s2': s2, 'z_seed': (mean / se) if se else None,
            'evaluator_style_se': se_ev, 'fold_t': t, 'fold_t_df': F - 1}


def baseline_flat(fold, name):
    import evaluate as EV
    p = BASELINES / PANEL / f'fold{fold}' / f'{name}.json'
    if not p.exists():
        return None
    return {k: v for k, v in EV.headline(json.loads(p.read_text())).items() if '__' not in k}


def six_vs_reference(S, arm, rid, folds, seeds, ref='resample'):
    out, n_pass = {}, 0
    for m, kind in SIX:
        a = fold_means(S, arm, rid, folds, seeds, m)
        r = {}
        for f in a:
            b = baseline_flat(f, ref)
            if b and b.get(m) is not None:
                r[f] = float(b[m])
        c = sorted(set(a) & set(r))
        if not c:
            out[m] = None
            continue
        va, vr = float(np.mean([a[f] for f in c])), float(np.mean([r[f] for f in c]))
        margin = (abs(va - .8) - abs(vr - .8)) if kind == 'cov80' else va - vr
        ok = bool(margin <= 1e-12)
        n_pass += ok
        out[m] = {'value': va, 'reference': vr, 'margin': margin, 'pass': ok}
    return {'n_pass': n_pass, 'metrics': out, 'reference': ref}


def analyze(S, folds_main=(0, 1, 2), extra_folds=(3, 4, 5, 6, 7, 8)):
    A = {'created': time.strftime('%Y-%m-%d %H:%M:%S %Z'), 'n_scores': len(S)}
    fm = list(folds_main)
    # ---- (1) selection: control, folds 0-2, seed 0
    sel = {r: fold_means(S, CONTROL, r, fm, [0]) for r in CANDIDATES + REFERENCES}
    sel_mean = {r: (float(np.mean(list(v.values()))) if len(v) == len(fm) else None) for r, v in sel.items()}
    avail = {r: v for r, v in sel_mean.items() if r in CANDIDATES and v is not None}
    chosen = min(avail, key=avail.get) if len(avail) == len(CANDIDATES) else None
    A['selection'] = {'rule': 'best candidate by held-out pb_rmse, folds 0-2, seed 0, control pbulk',
                      'per_fold': sel, 'mean': sel_mean, 'chosen': chosen}
    # ---- (2) confirmation
    s2_ctrl = None
    conf = {}
    for r in CANDIDATES + ('ema@0.9995', 'final'):
        both = paired(S, CONTROL, r, 'snapens', fm, [0, 1])
        s2_r = both['seed_s2'] if both else None
        if r == chosen:
            s2_ctrl = s2_r
        conf[r] = {'control_seed1': paired(S, CONTROL, r, 'snapens', fm, [1], s2_external=s2_r),
                   'control_seeds01': both,
                   'combination_seeds0-3': paired(S, COMBO, r, 'snapens', fm, [0, 1, 2, 3]),
                   'vs_ema0.9995_control_seed1': paired(S, CONTROL, r, 'ema@0.9995', fm, [1]),
                   'vs_ema0.9995_combination': paired(S, COMBO, r, 'ema@0.9995', fm, [0, 1, 2, 3])}
        ex = [f for f in extra_folds if any((CONTROL, f, s, r) in S for s in (0, 1))]
        if ex:
            conf[r]['control_folds3-8'] = {'repeat1 (folds 3-5)': paired(S, CONTROL, r, 'snapens', [f for f in ex if f < 6], [0, 1]),
                                           'repeat2 (folds 6-8)': paired(S, CONTROL, r, 'snapens', [f for f in ex if f >= 6], [0, 1]),
                                           'folds_present': ex}
    A['confirmation'] = conf
    A['confirmation_note'] = ('candidate minus snapens, held-out pb_rmse (negative = candidate better); control_seed1 SE uses the '
                              'within-fold variance of the paired difference over control seeds 0-1 (seed 0 is the selection set)')
    # decision under the pre-declared rule
    dec = {'chosen': chosen}
    if chosen:
        c = conf[chosen]
        def beats(p):
            return bool(p and p['seed_se'] and p['mean_diff'] < -2 * p['seed_se'])
        dec['control_seed1_beyond_2SE'] = beats(c['control_seed1'])
        dec['combination_beyond_2SE'] = beats(c['combination_seeds0-3'])
        ext = c.get('control_folds3-8')
        if ext and len(ext['folds_present']) == 6:
            dec['folds3-8_repeat1'] = beats(ext['repeat1 (folds 3-5)'])
            dec['folds3-8_repeat2'] = beats(ext['repeat2 (folds 6-8)'])
            dec['replace_snapens'] = all(dec[k] for k in ('control_seed1_beyond_2SE', 'combination_beyond_2SE',
                                                          'folds3-8_repeat1', 'folds3-8_repeat2'))
        else:
            dec['replace_snapens'] = None
            dec['note'] = 'folds 3-8 confirmation not yet available: the rule cannot be completed'
    A['decision'] = dec
    # ---- (3) stage-blind and bias-calibrated read-outs + six distribution metrics
    tabs = {}
    for arm, seeds in ((CONTROL, [0, 1]), (COMBO, [0, 1, 2, 3])):
        rows = {}
        for v in VARIANTS + ('bcsb',):
            for b in CANDIDATES + REFERENCES:
                rid = readout_id(v, b)
                fmn = fold_means(S, arm, rid, fm, seeds)
                if len(fmn) < len(fm):
                    continue
                ref = readout_id(v, 'snapens')
                rows[rid] = {'pb_rmse': float(np.mean(list(fmn.values()))), 'per_fold': fmn,
                             'vs_snapens_same_variant': paired(S, arm, rid, ref, fm, seeds) if rid != ref else None,
                             'six_vs_resample': six_vs_reference(S, arm, rid, fm, seeds, 'resample'),
                             'six_vs_resample_blind': six_vs_reference(S, arm, rid, fm, seeds, 'resample_blind'),
                             'init_share': init_share(b, 20000),
                             'pb_rmse_recal': float(np.mean(list(fold_means(S, arm, rid, fm, seeds, 'pb_rmse_recal').values())))}
                for m, _ in SIX:
                    mv = fold_means(S, arm, rid, fm, seeds, m)
                    rows[rid][m] = float(np.mean(list(mv.values()))) if mv else None
        tabs[arm] = rows
    A['readouts'] = tabs
    refs = {}
    for name in ('resample', 'resample_blind', 'grand_mean', 'ridge'):
        vals = {}
        for f in fm:
            b = baseline_flat(f, name)
            if b:
                for m in ['pb_rmse', 'pb_rmse_recal'] + [x for x, _ in SIX]:
                    if b.get(m) is not None:
                        vals.setdefault(m, []).append(float(b[m]))
        refs[name] = {m: float(np.mean(v)) for m, v in vals.items()}
    A['references_folds0-2'] = refs
    # ---- (4) curves
    curves = {}
    for arm, seeds in ((CONTROL, [0, 1]), (COMBO, [0, 1, 2, 3])):
        c = {}
        for series, fn in (('single', lambda u: 'final' if u == 20000 else f'snap@{u}'),
                           ('ema0.999', lambda u: 'ema@0.999' if u == 20000 else f'snapema@{u}:0.999'),
                           ('ema0.9995', lambda u: 'ema@0.9995' if u == 20000 else f'snapema@{u}:0.9995')):
            c[series] = {}
            for u in (5000, 10000, 15000, 20000):
                rid = fn(u)
                per = {f: fold_means(S, arm, rid, [f], seeds).get(f) for f in fm}
                c[series][u] = {'per_fold': per, 'mean': float(np.mean([v for v in per.values() if v is not None]))
                                if any(v is not None for v in per.values()) else None}
        c['snapens'] = {f: fold_means(S, arm, 'snapens', [f], seeds).get(f) for f in fm}
        curves[arm] = c
    for arm in LONG_ARMS:
        c = {}
        for series, fn in (('single', lambda u: f'snap@{u}'), ('ema0.999', lambda u: f'snapema@{u}:0.999'),
                           ('ema0.9995', lambda u: f'snapema@{u}:0.9995')):
            c[series] = {}
            for u in LONG_U:
                per = {f: metric(S, arm, f, 0, fn(u)) for f in fm}
                c[series][u] = {'per_fold': per, 'mean': float(np.mean([v for v in per.values() if v is not None]))
                                if any(v is not None for v in per.values()) else None}
            argmin = {}
            for f in fm:
                pts = {u: c[series][u]['per_fold'][f] for u in LONG_U if c[series][u]['per_fold'][f] is not None}
                argmin[f] = min(pts, key=pts.get) if pts else None
            c[series + '_argmin_by_fold'] = argmin
        curves[arm] = c
    A['curves'] = curves
    # ---- (4b) trade-off: distribution metrics and spread at 5k / 10k / 20k
    trade = {}
    keys = ['pb_rmse'] + [m for m, _ in SIX] + ['pc_spread_rep', 'pbdist_spread'] + SPREAD_KEYS
    for arm, seeds in ((CONTROL, [0, 1]), (COMBO, [0, 1, 2, 3])):
        t = {}
        for rid in ('init', 'snap@5000', 'snap@10000', 'snap@15000', 'final', 'snapema@5000:0.999', 'snapema@10000:0.999',
                    'snapema@5000:0.9995', 'db:snapema@5000:0.9995', 'db:snapema@5000:0.999', 'db:snapema@10000:0.9995',
                    'snapema@10000:0.9995', 'snapema@15000:0.999', 'snapema@15000:0.9995', 'ema@0.9995', 'snapens'):
            row = {}
            for k in keys:
                v = fold_means(S, arm, rid, fm, seeds, k)
                row[k] = float(np.mean(list(v.values()))) if len(v) == len(fm) else None
            t[rid] = row
        t['paired_10k_minus_20k'] = {k: paired(S, arm, 'snap@10000', 'final', fm, seeds, k) for k in keys}
        t['paired_5k_minus_20k'] = {k: paired(S, arm, 'snap@5000', 'final', fm, seeds, k) for k in keys}
        if chosen:
            t['paired_chosen_minus_snapens'] = {k: paired(S, arm, chosen, 'snapens', fm, seeds, k) for k in keys}
        trade[arm] = t
    A['tradeoff'] = trade
    # ---- post-hoc diagnostics (not pre-declared): untrained weights and the EMA's initial-weight share
    diag = {}
    for arm, seeds in ((CONTROL, [0, 1]), (COMBO, [0, 1, 2, 3])):
        d = {'init_pb_rmse': fold_means(S, arm, 'init', fm, seeds)}
        for b in ('snapema@5000:0.9995', 'snapema@5000:0.999', 'snapema@10000:0.9995'):
            d[f'debiased_minus_raw_ema {b}'] = paired(S, arm, 'db:' + b, b, fm, seeds)
            d[f'debiased_minus_snapens {b}'] = paired(S, arm, 'db:' + b, 'snapens', fm, seeds)
        diag[arm] = d
    A['diagnostics_posthoc'] = diag
    # ---- repair F4: the guard metrics reward collapse; read them next to spread and coverage
    cc = {'note': ('pb_rmse, pc_energy_rep and cell_energy_rep are all lower for the untrained generator than for snapens: '
                   'they reward collapse toward the pooled profile. Read every guard next to pc_spread_rep, pc_coverage80_rep '
                   '(nominal 0.80) and spread_ratio = sd_within_target_gen / sd_between_donors_realK_heldout (1 = real '
                   'between-donor spread at the same depth).')}
    for arm, seeds in ((CONTROL, [0, 1]), (COMBO, [0, 1, 2, 3])):
        rows = {}
        for rid in ('init', chosen, 'final', 'ema@0.9995', 'snapens'):
            if rid is None:
                continue
            row = {}
            for k in GUARD_KEYS + ['pc_spread_rep', 'pc_coverage80_rep']:
                v = fold_means(S, arm, rid, fm, seeds, k)
                row[k] = float(np.mean(list(v.values()))) if len(v) == len(fm) else None
            for tag in ('well', 'all'):
                g = fold_means(S, arm, rid, fm, seeds, f'sd_within_target_gen_{tag}')
                r = fold_means(S, arm, rid, fm, seeds, f'sd_between_donors_realK_heldout_{tag}')
                row[f'spread_ratio_{tag}'] = (float(np.mean([g[f] / r[f] for f in fm])) if len(g) == len(r) == len(fm) else None)
            rows[rid] = row
        pairs = {}
        for rid in ('init', chosen):
            if rid is None:
                continue
            pairs[f'{rid} - snapens'] = {k: paired(S, arm, rid, 'snapens', fm, seeds, k)
                                         for k in GUARD_KEYS + ['pc_spread_rep', 'pc_coverage80_rep',
                                                                'sd_within_target_gen_well', 'sd_within_target_gen_all']}
        cc[arm] = {'levels': rows, 'paired': pairs}
    A['collapse_check'] = cc
    return A


def fmt(v, nd=4, plus=False):
    if v is None:
        return '   -  '
    return (f'{v:+.{nd}f}' if plus else f'{v:.{nd}f}')


def report_text(A):
    L = []
    s = A['selection']
    L.append('WORK ITEM 1 (shortavg): held-out pb_rmse; selection = control pbulk, folds 0-2, seed 0')
    L.append(f"{'read-out':24s} {'fold0':>7s} {'fold1':>7s} {'fold2':>7s} {'mean':>7s}")
    for r in CANDIDATES + REFERENCES:
        pf = s['per_fold'].get(r, {})
        L.append(f"{r:24s} " + ' '.join(fmt(pf.get(f)) .rjust(7) for f in (0, 1, 2)) + f" {fmt(s['mean'].get(r)):>7s}")
    L.append(f"chosen: {s['chosen']}")
    L.append('')
    L.append('Confirmation: candidate - snapens (negative = better); mean, per fold, seed SE, fold t')
    for r, c in A['confirmation'].items():
        for k in ('control_seed1', 'combination_seeds0-3'):
            p = c.get(k)
            if p:
                L.append(f"  {r:22s} {k:22s} {fmt(p['mean_diff'], plus=True)} folds "
                         + ' '.join(fmt(p['per_fold'].get(f), plus=True) for f in sorted(p['per_fold']))
                         + f"  SE {fmt(p['seed_se'])} (ev {fmt(p['evaluator_style_se'])}) z {fmt(p['z_seed'], 2)}"
                         f"  fold t {fmt(p['fold_t'], 2)}  better {p['n_pairs_better']}/{p['n_pairs']}")
        ext = c.get('control_folds3-8')
        if ext:
            for k in ('repeat1 (folds 3-5)', 'repeat2 (folds 6-8)'):
                p = ext[k]
                if p:
                    L.append(f"  {r:22s} {k:22s} {fmt(p['mean_diff'], plus=True)}  SE {fmt(p['seed_se'])} fold t {fmt(p['fold_t'], 2)}"
                             f"  better {p['n_pairs_better']}/{p['n_pairs']}")
    L.append(f"decision: {json.dumps(A['decision'])}")
    L.append('')
    for arm, rows in A['readouts'].items():
        L.append(f'Read-outs, {arm} (folds 0-2, all seeds): pb_rmse, vs snapens of the same variant, six metrics, T2 k/6')
        L.append(f"{'read-out':28s} {'init':>6s} {'pb':>7s} {'d':>8s} {'SE':>6s} {'pcE':>6s} {'cellE':>6s} {'cov':>6s} "
                 f"{'mlog':>6s} {'nz':>6s} {'lvar':>6s} {'k/6':>4s} {'k/6b':>4s}")
        for rid, r in rows.items():
            p = r['vs_snapens_same_variant'] or {}
            L.append(f"{rid:28s} {r['init_share']:6.3f} {fmt(r['pb_rmse'])} {fmt(p.get('mean_diff'), plus=True):>8s} "
                     f"{fmt(p.get('seed_se'), 3):>6s} " + ' '.join(fmt(r[m], 3).rjust(6) for m, _ in SIX)
                     + f" {r['six_vs_resample']['n_pass']:>4d} {r['six_vs_resample_blind']['n_pass']:>4d}")
        L.append('')
    L.append('References folds 0-2: ' + json.dumps({k: {m: round(v, 4) for m, v in d.items()} for k, d in A['references_folds0-2'].items()}))
    L.append('')
    for arm, c in A['curves'].items():
        L.append(f'Curve {arm}: mean pb_rmse over folds (per fold)')
        for series in ('single', 'ema0.999', 'ema0.9995'):
            pts = c[series]
            L.append(f'  {series:10s} ' + '  '.join(f"{u // 1000}k {fmt(p['mean'], 3)}("
                                                   + '/'.join(fmt(p['per_fold'].get(f), 3) for f in (0, 1, 2)) + ')'
                                                   for u, p in pts.items()))
            if series + '_argmin_by_fold' in c:
                L.append(f"  {'':10s} argmin by fold: {c[series + '_argmin_by_fold']}")
    L.append('')
    for arm, t in A['tradeoff'].items():
        L.append(f'Trade-off {arm} (folds 0-2, all seeds)')
        keys = list(next(iter(t.values())).keys())
        rids = [r for r in t if not r.startswith('paired')]
        L.append('  (one line per statistic, full name; _all = 2,000 panel genes, _well = top-256 genes by train mean; '
                 'sd_within_target_gen = SD over the 32 generated donors of a held-out target, real counterpart '
                 'sd_between_donors_realK_heldout at the same K)')
        L.append(f"  {'statistic':38s} " + ' '.join(r[:12].rjust(12) for r in rids))
        for k in keys:
            L.append(f'  {k:38s} ' + ' '.join(fmt(t[r].get(k), 4).rjust(12) for r in rids))
        L.append('  read-out columns: ' + ', '.join(rids))
        for pk in ('paired_10k_minus_20k', 'paired_5k_minus_20k', 'paired_chosen_minus_snapens'):
            if pk in t:
                L.append(f'  {pk}: ' + '; '.join(f"{k} {fmt(p['mean_diff'], 4, True)} (SE {fmt(p['seed_se'], 4)}, t {fmt(p['fold_t'], 1)})"
                                                for k, p in t[pk].items() if p))
        L.append('')
    L.append('Post-hoc diagnostics (not pre-declared): untrained initial weights; EMA with its initial-weight share removed')
    for arm, d in A.get('diagnostics_posthoc', {}).items():
        L.append(f"  {arm}: init pb_rmse per fold {json.dumps({f: round(v, 4) for f, v in d['init_pb_rmse'].items()})}")
        for k, p in d.items():
            if k != 'init_pb_rmse' and p:
                L.append(f"    {k:42s} {fmt(p['mean_diff'], 4, True)} SE {fmt(p['seed_se'])} fold t {fmt(p['fold_t'], 2)} "
                         f"better {p['n_pairs_better']}/{p['n_pairs']}")
    cc = A.get('collapse_check')
    if cc:
        L.append('')
        L.append('Collapse check (repair F4): guard metrics next to spread and coverage (folds 0-2, all seeds)')
        L.append('  ' + cc['note'])
        for arm in (CONTROL, COMBO):
            if arm not in cc:
                continue
            L.append(f'  {arm}')
            cols = GUARD_KEYS + ['pc_spread_rep', 'pc_coverage80_rep', 'spread_ratio_well', 'spread_ratio_all']
            L.append(f"    {'read-out':22s} " + ' '.join(c[:17].rjust(17) for c in cols))
            for rid, row in cc[arm]['levels'].items():
                L.append(f'    {rid:22s} ' + ' '.join(fmt(row.get(c), 4).rjust(17) for c in cols))
            for pk, d in cc[arm]['paired'].items():
                L.append(f'    {pk}: ' + '; '.join(f"{k} {fmt(p['mean_diff'], 4, True)} (SE {fmt(p['seed_se'], 4)}, "
                                                   f"better {p['n_pairs_better']}/{p['n_pairs']})" for k, p in d.items() if p))
    return '\n'.join(L)


# =========================================================================== CLI
def parse_arms(text):
    if text is None:
        return None
    if ';' in text:
        return [a for a in text.split(';') if a]
    known = [COMBO, CONTROL] + list(LONG_ARMS)
    out, rest = [], text
    while rest:   # greedy match of known arm names (they contain commas)
        for k in sorted(known, key=len, reverse=True):
            if rest == k or rest.startswith(k + ','):
                out.append(k); rest = rest[len(k) + 1:]
                break
        else:
            raise ValueError(f'cannot split arm list {text!r}: separate arms with ";"')
    return out


def ints(t):
    return [int(x) for x in str(t).split(',') if x != '']


def job_args(a):
    arms_ = parse_arms(a.arms) or ([CONTROL, COMBO] if a.set in ('main', 'diag') else list(LONG_ARMS))
    variants = tuple(v if v != 'plain' else '' for v in a.variants.split(',')) if a.variants else VARIANTS
    bcsb = tuple(x for x in (a.bcsb or '').split(';') if x)
    return build_jobs(a.set, arms_, ints(a.folds), ints(a.seeds), variants, bcsb)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    sub = p.add_subparsers(dest='cmd', required=True)
    for name in ('run', 'generate', 'score', 'list'):
        q = sub.add_parser(name)
        q.add_argument('--set', choices=['main', 'curve60k', 'diag'], default='main')
        q.add_argument('--arms', default=None)
        q.add_argument('--folds', default='0,1,2')
        q.add_argument('--seeds', default='0,1,2,3')
        q.add_argument('--variants', default=None, help="comma list of plain,sb,bc (default all)")
        q.add_argument('--bcsb', default=None, help="';'-separated base read-outs also generated as biascal on stage-blind")
        q.add_argument('--device', default='cuda')
        q.add_argument('--gpu-mem-fraction', type=float, default=0.04)
        q.add_argument('--watch', action='store_true')
        q.add_argument('--score-folds', default=None)
    q = sub.add_parser('analyze')
    q.add_argument('--out', default=str(OUT / 'analysis'))
    q = sub.add_parser('keep', help='(re)generate the chosen read-out for the control and keep the mode files')
    q.add_argument('--readout', required=True)
    q.add_argument('--arm', default=CONTROL)
    q.add_argument('--folds', default='0,1,2')
    q.add_argument('--seeds', default='0,1')
    q.add_argument('--device', default='cuda')
    q.add_argument('--gpu-mem-fraction', type=float, default=0.04)
    a = p.parse_args(argv)
    OUT.mkdir(parents=True, exist_ok=True)
    if a.cmd == 'analyze':
        A = analyze(load_scores())
        Path(a.out + '.json').write_text(json.dumps(A, indent=1, default=float))
        txt = report_text(A)
        Path(a.out + '.txt').write_text(txt + '\n')
        print(txt)
        return
    if a.cmd == 'keep':
        setup_generation(a.device, a.gpu_mem_fraction)
        jobs = []
        for v in VARIANTS:
            j, _ = build_jobs('main', [a.arm], ints(a.folds), ints(a.seeds), (v,))
            jobs += [x for x in j if x['base'] == a.readout and x['source'] is None]
        for j in jobs:
            j['out'] = str(OUT / 'kept' / Path(j['out']).relative_to(OUT / 'modes'))
            j['score'] = j['out'] + '.never'
        generate_jobs(jobs, a.device, max_pending=10 ** 6)
        return
    jobs, skipped = job_args(a)
    if skipped:
        print(f'[shortavg] skipped {len(skipped)} incomplete fits: {skipped}', flush=True)
    if a.cmd == 'list':
        for j in jobs:
            print(j['arm'], j['fold'], j['seed'], j['readout'], j['mode'], 'EXISTING ' + j['source'] if j['source'] else '',
                  'scored' if Path(j['score']).exists() else '')
        print(len(jobs), 'jobs;', sum(j['source'] is None for j in jobs), 'to generate;',
              sum(Path(j['score']).exists() for j in jobs), 'scored')
        return
    if a.cmd == 'generate':
        (OUT / 'generator.pid').write_text(str(os.getpid()))
        try:
            setup_generation(a.device, a.gpu_mem_fraction)
            generate_jobs(jobs, a.device)
        finally:
            (OUT / 'generator.pid').unlink(missing_ok=True)
        return
    if a.cmd == 'score':
        folds = ints(a.score_folds or a.folds)
        left = score_jobs(jobs, folds, watch=a.watch)
        print(f'[score] done; {left} jobs without a file')
        return
    if a.cmd == 'run':
        logs = RUNS / 'logs_r5'
        logs.mkdir(exist_ok=True)
        (OUT / 'generator.pid').write_text(str(os.getpid()))
        base = [PY, '-B', __file__, 'score', '--set', a.set, '--folds', a.folds, '--seeds', a.seeds, '--watch']
        if a.arms:
            base += ['--arms', a.arms]
        if a.variants:
            base += ['--variants', a.variants]
        if a.bcsb:
            base += ['--bcsb', a.bcsb]
        env = dict(os.environ, OMP_NUM_THREADS='3', OPENBLAS_NUM_THREADS='3', MKL_NUM_THREADS='3', CUDA_VISIBLE_DEVICES='')
        procs = []
        for f in ints(a.folds):
            lf = open(logs / f'shortavg_{a.set}_score_fold{f}.log', 'a')
            procs.append(subprocess.Popen(base + ['--score-folds', str(f)], stdout=lf, stderr=subprocess.STDOUT, env=env, cwd='/tmp'))
        try:
            setup_generation(a.device, a.gpu_mem_fraction)
            generate_jobs(jobs, a.device)
        finally:
            (OUT / 'generator.pid').unlink(missing_ok=True)
        for pr in procs:
            pr.wait()
        left = [j for j in jobs if not Path(j['score']).exists()]
        print(f'[shortavg] run finished: {len(jobs) - len(left)} scored, {len(left)} missing', flush=True)
        for j in left:
            print('  missing', j['arm'], j['fold'], j['seed'], j['readout'])


if __name__ == '__main__':
    sys.modules.setdefault('shortavg', sys.modules['__main__'])
    main()
