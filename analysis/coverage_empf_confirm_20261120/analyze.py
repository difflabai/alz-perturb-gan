"""Analyze the native EMPF fold/seed confirmation exactly as PROTOCOL.md declares.

Question 1: readout B (cellshape@0.25) against A (cellshape@0.5), paired per fit over
the 17 confirmation fits (folds 0-8, seeds 0-1, without the pilot fold 0 seed 0):
all ten pilot gates on the mean over fits, AND both proper interval scores lower in
each repeat (folds 0-2 without the pilot, 3-5, 6-8). Paired SE over fits and the
number of fits improving are reported for every metric.

Question 2: SD-only calibration for A and for B: SD and combined coverage closer to
80% and SD and combined proper scores lower than without calibration, pooled and in
each repeat. Widths are reported.

Secondary (descriptive): native EMPF readout A against the previous BISPF count-shape
default on the same fold, seed and held-out donors, and stage-blind empirical
resampling as the interval reference.

Runs on whatever fits are scored, prints what is missing and labels the result
INCOMPLETE until all 17 confirmation fits are scored. Spread and coverage are always
printed next to RMSE and energy (RESULTS.md 8g).

Review repairs (2026-11-20, recorded before any confirmation fit was scored):
F1  Folds 0-2 secondary references are the covariance-challenge rescorings
    (results_r5/covariance_challenge_20261114/{current32,empirical32}), pinned by
    analysis/covariance_challenge_20261114/artifact_audit_complete_20261116.json. They use
    the same generation-stream namespace as the EMPF readouts, SeedSequence([20261114,6,
    fold,index,replicate]) with replicate = fit seed, and they are the references the
    owner's EMPF selection (empf_comparison_20261117) used. The empirical-challenge
    rescorings of the same fits (namespace [20261114,1,...]) are kept as labelled
    stream-sensitivity rows, and their difference from the covariance rescorings is
    reported as the previous default's own stream-to-stream spread.
F2  Question 2 also reports the mean over fits of per-fit |coverage-0.8| (pooled and per
    repeat), as question 1 already did. The protocol's literal rule (distance of the MEAN
    coverage from 0.8) governs both questions; the per-fit distance is a diagnostic.
F5  Every paired table also gives a fold-level SE: seeds are averaged within each fold
    first, then the SE is taken over folds. EMPF seeds 0 and 1 of one fold share
    generation streams, and folds 3-8 reuse one stage-blind empirical stream for both
    seeds, so the per-fit SE overstates precision. TRAIN-partition (raw_train) SD and
    combined coverage and interval scores are printed next to the held-out values.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OUT = ROOT/'runs/coverage_empf_confirm_20261120'
PIN = HERE/'source_manifest.json'
PILOT = (0, 0)
CONFIRMATION = tuple((k, s) for s in (0, 1) for k in range(9) if (k, s) != PILOT)
REPEATS = {'folds0-2_without_pilot': (0, 1, 2), 'folds3-5': (3, 4, 5), 'folds6-8': (6, 7, 8)}
GUARDS = {'pb_rmse': .005, 'meanlog_rmse_gene': .02, 'nonzero_rmse_gene': .02, 'logvar_rmse_gene': .02,
          'pc_energy_rep': .02, 'cell_energy_rep': .02}
GATE_ORDER = ('sd_coverage', 'combined_coverage', 'sd_interval_score', 'combined_interval_score', *GUARDS)
LOWER = ('pb_rmse', 'meanlog_rmse_gene', 'nonzero_rmse_gene', 'logvar_rmse_gene', 'pc_energy_rep',
         'cell_energy_rep', 'sd_interval_score', 'combined_interval_score', 'mean_interval_score')
SPREAD_COVERAGE = ('pc_spread_rep', 'pc_coverage80_rep', 'pbdist_spread', 'pbdist_coverage80')
INTERVAL_KEYS = ('coverage', 'below', 'above', 'width', 'interval_score')
PREVIOUS = ROOT/'runs/improve_20261109/results_r5'
PREVIOUS_EVIDENCE = ROOT/'analysis/improve_20261109/default_config_distribution_evidence_20261113.json'
EMPIRICAL_EVIDENCE = ROOT/'analysis/improve_20261109/empirical_interval_reference_evidence_20261113.json'
CHALLENGE_AUDIT = ROOT/'analysis/empirical_challenge_20261114/artifact_audit_complete_20261114.json'
COVARIANCE_AUDIT = ROOT/'analysis/covariance_challenge_20261114/artifact_audit_complete_20261116.json'
# Folds 0-2 reference rescorings of the previous default and stage-blind resampling (review F1):
# 'covariance' is primary (same stream namespace as EMPF, used by the owner's EMPF selection);
# 'empirical_challenge' is a stream-sensitivity row only.
FOLD02_REFERENCES = {
    'covariance': dict(dir='covariance_challenge_20261114', audit=COVARIANCE_AUDIT,
                       streams='SeedSequence([20261114,6,fold,index,replicate]), replicate = fit seed'),
    'empirical_challenge': dict(dir='empirical_challenge_20261114', audit=CHALLENGE_AUDIT,
                                streams='SeedSequence([20261114,1,fold,index,replicate]), replicate = fit seed'),
}


def challenge_pin(recipe, fold, replicate, audit_path=CHALLENGE_AUDIT):
    audit = json.loads(Path(audit_path).read_text())
    if audit_path == COVARIANCE_AUDIT and not (audit.get('status') == 'passed' and audit.get('complete')):
        return None
    return next((x['score_sha256'] for x in audit['artifacts']
                 if x['recipe'] == recipe and x['fold'] == fold and x['replicate'] == replicate), None)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n'); tmp.replace(path)


def repeat_of(fold):
    return next(name for name, folds in REPEATS.items() if fold in folds)


# --------------------------------------------------------------------------- metric rows
TRAIN_KEYS = tuple(f'train_{g}_{k}' for g in ('sd', 'combined') for k in ('coverage', 'interval_score', 'width'))


def flat(row, calibrated=False):
    """One readout's metrics as a flat dict: historical metrics plus held-out interval scores.
    Rows from score.json also carry TRAIN-partition raw intervals (train_* keys, review F5);
    reference rows do not."""
    value = {k: float(v) for k, v in row['historical'].items() if isinstance(v, (int, float))}
    intervals = row['calibrated_heldout' if calibrated else 'raw_heldout']
    for group in ('mean', 'sd', 'combined'):
        for key in INTERVAL_KEYS:
            value[f'{group}_{key}'] = float(intervals[group][key])
    if 'raw_train' in row and not calibrated:
        for group in ('sd', 'combined'):
            for key in INTERVAL_KEYS:
                value[f'train_{group}_{key}'] = float(row['raw_train'][group][key])
    return value


def mean(values):
    return sum(values)/len(values)


def se_of(values):
    n = len(values)
    if n < 2:
        return None
    m = mean(values)
    return math.sqrt(sum((v-m)**2 for v in values)/(n-1)/n)


def fold_level(d, folds):
    """Review F5: average the per-fit differences within each fold, then SE over folds."""
    by = {}
    for v, k in zip(d, folds):
        by.setdefault(k, []).append(v)
    means = [mean(v) for _, v in sorted(by.items())]
    return dict(n_folds=len(means), mean_of_fold_means=mean(means), fold_se=se_of(means))


def paired(a, b, key, folds=None):
    """b - a over fits: mean, paired SE (n-1), and how many fits improve. With ``folds``
    (one fold per pair) also a fold-level SE over fold means (review F5)."""
    d = [y[key]-x[key] for x, y in zip(a, b)]
    n = len(d); m = mean(d)
    se = se_of(d)
    if key.endswith('coverage') or key == 'pc_coverage80_rep' or key == 'pbdist_coverage80':
        improved = sum(abs(y[key]-.8) < abs(x[key]-.8) for x, y in zip(a, b)); rule = 'closer to 0.8'
    elif key in LOWER or key.endswith('interval_score'):
        improved = sum(y[key] < x[key] for x, y in zip(a, b)); rule = 'lower'
    elif key.endswith('spread') or key.endswith('spread_rep'):
        improved = sum(abs(math.log(y[key])) < abs(math.log(x[key])) for x, y in zip(a, b)); rule = 'closer to 1'
    else:
        improved = None; rule = 'descriptive'
    value = dict(n=n, mean_a=mean([x[key] for x in a]), mean_b=mean([y[key] for y in b]), mean_diff=m,
                 paired_se=se, n_improved=improved, improvement_rule=rule)
    if folds is not None:
        value['fold_level'] = fold_level(d, folds)
    return value


def gates_on_means(a, b):
    """The pilot's ten gates applied to the means over fits (a: base rows, b: candidate rows)."""
    ma = {k: mean([x[k] for x in a]) for k in a[0]}
    mb = {k: mean([x[k] for x in b]) for k in b[0]}
    gates = {}
    for group in ('sd', 'combined'):
        gates[f'{group}_coverage'] = abs(mb[f'{group}_coverage']-.8) < abs(ma[f'{group}_coverage']-.8)
        gates[f'{group}_interval_score'] = mb[f'{group}_interval_score'] < ma[f'{group}_interval_score']
    for key, limit in GUARDS.items():
        gates[key] = mb[key] <= ma[key]*(1+limit)
    return {k: gates[k] for k in GATE_ORDER}


def coverage_distance_per_fit(a, b, group):
    """Diagnostic only: mean over fits of |coverage-0.8| (the 8p-style transform)."""
    return dict(a=mean([abs(x[f'{group}_coverage']-.8) for x in a]), b=mean([abs(y[f'{group}_coverage']-.8) for y in b]))


def repeat_scores_lower(a, b, folds):
    """Mean over fits within each repeat: both proper interval scores lower for b."""
    out = {}
    for name, members in REPEATS.items():
        idx = [i for i, f in enumerate(folds) if f in members]
        if not idx:
            out[name] = None; continue
        out[name] = {g: mean([b[i][f'{g}_interval_score'] for i in idx]) < mean([a[i][f'{g}_interval_score'] for i in idx])
                     for g in ('sd', 'combined')} | {'n_fits': len(idx)}
    return out


def question1(a, b, folds, complete):
    gates = gates_on_means(a, b)
    repeats = repeat_scores_lower(a, b, folds)
    repeat_ok = all(r is not None and r['sd'] and r['combined'] for r in repeats.values())
    keys = [*GATE_ORDER[:4], 'mean_coverage', 'mean_interval_score', *[f'{g}_{k}' for g in ('sd', 'combined', 'mean')
            for k in ('width', 'below', 'above')], *GUARDS, *SPREAD_COVERAGE]
    keys += [k for k in TRAIN_KEYS if k in a[0] and k in b[0]]
    passed = all(gates.values()) and repeat_ok
    return dict(status=('PASS' if passed else 'FAIL') if complete else 'INCOMPLETE',
                provisional_on_scored_fits='PASS' if passed else 'FAIL',
                gates_on_means=gates, all_ten_gates=all(gates.values()),
                repeats_both_proper_scores_lower=repeats, repeat_condition=repeat_ok,
                coverage_rule=COVERAGE_RULE,
                coverage_distance_mean_of_per_fit=dict(sd=coverage_distance_per_fit(a, b, 'sd'),
                                                      combined=coverage_distance_per_fit(a, b, 'combined')),
                paired=[dict(metric=k, **paired(a, b, k, folds)) for k in keys])


COVERAGE_RULE = ('Governing rule (PROTOCOL.md, literal): the distance of the MEAN coverage over fits from 0.8. '
                 'coverage_distance_mean_of_per_fit (mean over fits of |coverage-0.8|) is a diagnostic only; it '
                 'shows when over- and undershooting fits cancel in the mean. Recorded before the first '
                 'confirmation fit was scored (review F2).')


def calibration_distance(raw, cal, group):
    """Diagnostic only (review F2): mean over fits of |coverage-0.8|, raw and calibrated."""
    return dict(raw=mean([abs(r[f'{group}_coverage']-.8) for r in raw]),
                calibrated=mean([abs(c[f'{group}_coverage']-.8) for c in cal]))


def calibration_check(raw, cal):
    return dict(sd_coverage_closer=abs(mean([c['sd_coverage'] for c in cal])-.8) < abs(mean([r['sd_coverage'] for r in raw])-.8),
                combined_coverage_closer=abs(mean([c['combined_coverage'] for c in cal])-.8) <
                abs(mean([r['combined_coverage'] for r in raw])-.8),
                sd_interval_score_lower=mean([c['sd_interval_score'] for c in cal]) < mean([r['sd_interval_score'] for r in raw]),
                combined_interval_score_lower=mean([c['combined_interval_score'] for c in cal]) <
                mean([r['combined_interval_score'] for r in raw]))


def question2(raw, cal, folds, complete):
    pooled = calibration_check(raw, cal)
    repeats, distance_repeats = {}, {}
    for name, members in REPEATS.items():
        idx = [i for i, f in enumerate(folds) if f in members]
        repeats[name] = (calibration_check([raw[i] for i in idx], [cal[i] for i in idx]) | {'n_fits': len(idx)}) if idx else None
        distance_repeats[name] = ({g: calibration_distance([raw[i] for i in idx], [cal[i] for i in idx], g)
                                   for g in ('sd', 'combined')} | {'n_fits': len(idx)}) if idx else None
    ok = all(pooled.values()) and all(r is not None and all(v for k, v in r.items() if k != 'n_fits')
                                      for r in repeats.values())
    keys = [f'{g}_{k}' for g in ('sd', 'combined', 'mean') for k in INTERVAL_KEYS]
    return dict(status=('CONFIRMED' if ok else 'NOT_CONFIRMED') if complete else 'INCOMPLETE',
                provisional_on_scored_fits='CONFIRMED' if ok else 'NOT_CONFIRMED',
                pooled=pooled, repeats=repeats, coverage_rule=COVERAGE_RULE,
                coverage_distance_mean_of_per_fit=dict(pooled={g: calibration_distance(raw, cal, g) for g in ('sd', 'combined')},
                                                      repeats=distance_repeats),
                paired_calibrated_minus_raw=[dict(metric=k, **paired(raw, cal, k, folds)) for k in keys])


# --------------------------------------------------------------------------- loading
def check_sources():
    pin = json.loads(PIN.read_text())
    changed = [n for n, d in {**pin['execution_sources'], **pin.get('analysis_sources', {})}.items() if sha(ROOT/n) != d]
    changed += [n for n, d in pin['inputs'].items() if sha(ROOT/n) != d]
    return pin, changed


def load_scores(pin):
    import importlib.util
    spec = importlib.util.spec_from_file_location('confirm_score', HERE/'score.py')
    S = importlib.util.module_from_spec(spec); spec.loader.exec_module(S)
    digest = S.scoring_digest(pin)
    scores, missing, problems = {}, [], []
    for fold, seed in (PILOT, *CONFIRMATION):
        path = OUT/f'scores/fold{fold}/seed{seed}/score.json'
        if not path.exists():
            missing.append(f'fold{fold}/seed{seed}'); continue
        value = json.loads(path.read_text())
        if value['hashes']['experiment_manifest_sha256'] != digest:
            problems.append(f'fold{fold}/seed{seed}: scored with different sources')
        for label, row in value['rows'].items():
            raw = path.parent/f'{label}_raw_score.json'
            if sha(raw) != row['raw_score_sha256']:
                problems.append(f'fold{fold}/seed{seed}/{label}: raw score file changed')
            for swap in row['swaps']:
                if sha(ROOT/swap['artifact']) != swap['artifact_sha256']:
                    problems.append(f'fold{fold}/seed{seed}/{label}: calibration artifact changed')
        scores[(fold, seed)] = value
    return scores, missing, problems


def score_file(path, fold, seed=None):
    value = json.loads(Path(path).read_text())
    if value.get('fold') != fold:
        raise ValueError(f'{path}: wrong fold')
    return value


def fold02_reference(recipe, fold, seed, cache_sha, variant):
    """Folds 0-2: a pinned CPU M32/K256 rescoring of the previous default (current32) or of
    stage-blind empirical resampling (empirical32), replicate = fit seed (review F1)."""
    spec = FOLD02_REFERENCES[variant]
    path = PREVIOUS/f'{spec["dir"]}/{recipe}/fold{fold}/seed{seed}.json'
    if not path.exists():
        return None, f'missing {path}'
    value = score_file(path, fold)
    challenge = value.get('challenge', {})
    if (challenge.get('recipe'), challenge.get('fold'), challenge.get('replicate')) != (recipe, fold, seed):
        return None, f'{path}: recipe/fold/replicate metadata does not match'
    if value['fingerprint']['cache_sha256'] != cache_sha:
        return None, f'{path}: different cache/held-out donors'
    if 'intervals' not in value:
        return None, f'{path}: no interval scores'
    if challenge_pin(recipe, fold, seed, spec['audit']) != sha(path):
        return None, f'{path}: hash differs from {spec["audit"].relative_to(ROOT)}'
    source = f'{spec["dir"]} {recipe} (CPU, M32/K256, streams {spec["streams"]})'
    return dict(path=str(path.relative_to(ROOT)), sha256=sha(path), source=source, pinned_by_evidence=True,
                pin=str(spec['audit'].relative_to(ROOT)), variant=variant,
                row=dict(historical=value['historical'], raw_heldout=value['intervals']['heldout']['percentile'])), None


def previous_default(fold, seed, cache_sha, variant='covariance'):
    """Previous BISPF count-shape default (pbulk fits_r3, cellshape@0.5:biascal:stagehalf), CPU M32 rescores.
    Folds 0-2: the covariance-challenge rescoring (primary, review F1) or, with
    variant='empirical_challenge', the stream-sensitivity rescoring. Folds 3-8: the count-shape
    confirmation (one rescoring exists)."""
    if fold in (0, 1, 2):
        return fold02_reference('current32', fold, seed, cache_sha, variant)
    if variant != 'covariance':
        return None, 'no stream-sensitivity rescoring exists for folds 3-8'
    path = PREVIOUS/f'architecture_pass/shape_confirmation/count_shape/fold{fold}/seed{seed}.json'
    source = 'count-shape distribution confirmation (CPU, M32/K256, streams SeedSequence([20261109,2,fold,index]))'
    if not path.exists():
        return None, f'missing {path}'
    value = score_file(path, fold)
    if value['fingerprint']['cache_sha256'] != cache_sha:
        return None, f'{path}: different cache/held-out donors'
    if 'intervals' not in value:
        return None, f'{path}: no interval scores'
    evidence = json.loads(PREVIOUS_EVIDENCE.read_text())
    pinned = next((x['score_sha256'] for x in evidence['artifact_audit']['artifacts']
                   if x['label'] == 'count_shape' and x['fold'] == fold and x['seed'] == seed), None)
    if pinned != sha(path):
        return None, f'{path}: hash differs from the pinned promotion evidence'
    return dict(path=str(path.relative_to(ROOT)), sha256=sha(path), source=source, pinned_by_evidence=True,
                pin=str(PREVIOUS_EVIDENCE.relative_to(ROOT)), variant='count_shape_confirmation',
                row=dict(historical=value['historical'], raw_heldout=value['intervals']['heldout']['percentile'])), None


def empirical_blind(fold, seed, cache_sha, variant='covariance'):
    """Stage-blind empirical resampling (RESULTS.md 8q). Folds 0-2: covariance-challenge
    empirical32 (primary, review F1) or the empirical-challenge stream-sensitivity row;
    folds 3-8: resample_blind, one stream per fold shared by both seeds."""
    if fold in (0, 1, 2):
        return fold02_reference('empirical32', fold, seed, cache_sha, variant)
    if variant != 'covariance':
        return None, 'no stream-sensitivity rescoring exists for folds 3-8'
    path = PREVIOUS/f'architecture_pass/empirical_interval_references/resample_blind/fold{fold}.json'
    source = 'RESULTS.md 8q resample_blind, ONE fixed stream per fold (shared by both seeds)'
    if not path.exists():
        return None, f'missing {path}'
    value = score_file(path, fold)
    if value['fingerprint']['cache_sha256'] != cache_sha:
        return None, f'{path}: different cache/held-out donors'
    evidence = json.loads(EMPIRICAL_EVIDENCE.read_text())
    pinned = next((x['score_sha256'] for x in evidence['artifacts'] if x['baseline'] == 'resample_blind' and x['fold'] == fold), None)
    if pinned != sha(path):
        return None, f'{path}: hash differs from the pinned empirical evidence'
    return dict(path=str(path.relative_to(ROOT)), sha256=sha(path), source=source,
                row=dict(historical=value['historical'], raw_heldout=value['intervals']['heldout']['percentile'])), None


# --------------------------------------------------------------------------- printing
def fmt(v):
    return 'nan' if v is None or (isinstance(v, float) and math.isnan(v)) else f'{v:.6f}'


def print_paired(title, rows, label_a, label_b):
    print(f'\n{title}')
    print(f'  {"metric":28s} {label_a:>11s} {label_b:>11s} {"diff":>11s} {"paired SE":>10s} {"fold SE":>10s} '
          f'{"improved":>9s}  rule')
    for r in rows:
        imp = '-' if r['n_improved'] is None else f'{r["n_improved"]}/{r["n"]}'
        fold_se = r.get('fold_level', {}).get('fold_se')
        print(f'  {r["metric"]:28s} {fmt(r["mean_a"]):>11s} {fmt(r["mean_b"]):>11s} {fmt(r["mean_diff"]):>11s} '
              f'{fmt(r["paired_se"]):>10s} {fmt(fold_se):>10s} {imp:>9s}  {r["improvement_rule"]}')


SECONDARY_KEYS = ('pb_rmse', 'meanlog_rmse_gene', 'nonzero_rmse_gene', 'logvar_rmse_gene', 'pc_energy_rep',
                  'cell_energy_rep', *SPREAD_COVERAGE,
                  *[f'{g}_{k}' for g in ('sd', 'combined', 'mean') for k in ('coverage', 'interval_score', 'width')])
STREAM_KEYS = ('pb_rmse', 'sd_interval_score', 'combined_interval_score')
REFERENCE_SETS = (('previous_default', previous_default, 'covariance'),
                  ('empirical_stage_blind', empirical_blind, 'covariance'),
                  ('previous_default_stream_sensitivity_empirical_challenge', previous_default, 'empirical_challenge'),
                  ('empirical_stage_blind_stream_sensitivity_empirical_challenge', empirical_blind, 'empirical_challenge'))


def stream_to_stream(fits):
    """Review F1: the previous default's own stream-to-stream spread on folds 0-2. The same frozen
    fits_r3 fit (replicate = fit seed) is rescored on two stream namespaces; empirical-challenge
    (namespace 1) minus covariance-challenge (namespace 6), both pinned, same cache."""
    pairs, used, notes = [], [], []
    for fold, seed in fits:
        if fold not in (0, 1, 2):
            continue
        cov_path = PREVIOUS/f'covariance_challenge_20261114/current32/fold{fold}/seed{seed}.json'
        cache_sha = json.loads(cov_path.read_text())['fingerprint']['cache_sha256'] if cov_path.exists() else None
        cov, p1 = previous_default(fold, seed, cache_sha, 'covariance')
        emp, p2 = previous_default(fold, seed, cache_sha, 'empirical_challenge')
        if cov is None or emp is None:
            notes.append(f'fold{fold}/seed{seed}: {p1 or p2}'); continue
        pairs.append((flat(cov['row']), flat(emp['row']), fold))
        used.append(dict(fold=fold, seed=seed, namespace6=cov['path'], namespace1=emp['path']))
    if not pairs:
        return dict(n=0, notes=notes)
    a, b, folds = [p[0] for p in pairs], [p[1] for p in pairs], [p[2] for p in pairs]
    rows = [dict(metric=k, **paired(a, b, k, folds)) for k in SECONDARY_KEYS]
    spread = {k: dict(max_abs_difference=max(abs(y[k]-x[k]) for x, y in zip(a, b)),
                      differences=[y[k]-x[k] for x, y in zip(a, b)]) for k in STREAM_KEYS}
    return dict(n=len(pairs), fits=used, notes=notes, namespace1_minus_namespace6=rows, spread=spread)


def secondary(scores, fits, a_label):
    """Descriptive EMPF-versus-reference comparisons on the same fold, seed and held-out donors.
    Primary references: covariance-challenge rescorings for folds 0-2 (review F1), count-shape
    confirmation and resample_blind for folds 3-8. Stream-sensitivity rows: the empirical-challenge
    rescorings for folds 0-2. Every table carries a fold-level SE (review F5)."""
    keys = SECONDARY_KEYS
    result = {}
    for name, loader, variant in REFERENCE_SETS:
        pairs, notes, used = [], [], []
        for fold, seed in fits:
            if (fold, seed) not in scores:
                continue
            if variant != 'covariance' and fold not in (0, 1, 2):
                continue
            value = scores[(fold, seed)]
            cache_sha = value['hashes']['cache_anchor']['manifest_sha256']
            ref, problem = loader(fold, seed, cache_sha, variant)
            if ref is None:
                notes.append(f'fold{fold}/seed{seed}: {problem}'); continue
            label = a_label if a_label in value['rows'] else 'A'
            pairs.append((flat(ref['row']), flat(value['rows'][label]), flat(value['rows']['B']),
                          flat(value['rows'][label], True), flat(value['rows']['B'], True), fold))
            used.append(dict(fold=fold, seed=seed, reference=ref['path'], reference_sha256=ref['sha256'],
                             source=ref['source'], pin=ref.get('pin')))
        if not pairs:
            result[name] = dict(n=0, notes=notes); continue
        ref_rows = [p[0] for p in pairs]; folds = [p[5] for p in pairs]
        comps = {}
        for idx, tag in ((1, 'empf_A'), (2, 'empf_B'), (3, 'empf_A_calibrated'), (4, 'empf_B_calibrated')):
            other = [p[idx] for p in pairs]
            use = keys if idx in (1, 2) else [k for k in keys if k.split('_')[0] in ('sd', 'combined', 'mean')]
            comps[f'{tag}_minus_reference'] = [dict(metric=k, **paired(ref_rows, other, k, folds)) for k in use]
        result[name] = dict(n=len(pairs), fits=used, notes=notes, comparisons=comps,
                            role='stream sensitivity only' if variant != 'covariance' else 'primary descriptive reference')
    sts = stream_to_stream([(k, s) for k in (0, 1, 2) for s in (0, 1)])   # all six pinned pairs, EMPF-independent
    result['previous_default_stream_to_stream'] = sts
    statement = {}
    comps = result['previous_default'].get('comparisons', {})
    for key in STREAM_KEYS:
        row = next((r for r in comps.get('empf_A_minus_reference', []) if r['metric'] == key), None)
        if row is None or not sts.get('n'):
            continue
        bound = sts['spread'][key]['max_abs_difference']
        statement[key] = dict(empf_A_minus_previous_mean=row['mean_diff'], previous_stream_to_stream_max_abs=bound,
                              within_previous_default_stream_spread=abs(row['mean_diff']) <= bound)
    result['empf_A_minus_previous_vs_stream_spread'] = dict(
        values=statement,
        note=('The previous default rescored on two stream namespaces (folds 0-2, same fits) differs by up to '
              'previous_stream_to_stream_max_abs; an EMPF-minus-previous mean difference no larger than that '
              'is within the reference\'s own Monte Carlo spread and should not be read as an effect.'))
    result['interval_availability'] = (
        'Previous default: folds 0-2 seeds 0-1 use the CPU M32 covariance-challenge rescore (current32, same stream '
        'namespace [20261114,6] as the EMPF readouts, pinned by covariance_challenge artifact_audit_complete_20261116); '
        'the CPU M32 empirical-challenge rescore (namespace [20261114,1]) of the same fits is a stream-sensitivity row. '
        'The original GPU six-fit default_pass files carry historical metrics but no interval scores. Folds 3-8 seeds '
        '0-1 use the CPU count-shape confirmation (namespace [20261109,2]). Stage-blind resampling: folds 0-2 have two '
        'streams (matched to seed by replicate) in each challenge; folds 3-8 have one stream per fold, reused for both '
        'seeds (hence the fold-level SE). No values are invented for any other fold.')
    return result


# --------------------------------------------------------------------------- main
def analyze(a_source='cpu'):
    pin, changed = check_sources()
    scores, missing, problems = load_scores(pin)
    a_label = 'A' if a_source == 'cpu' else 'A_cli'
    fits = [f for f in CONFIRMATION if f in scores]
    missing_confirmation = [f'fold{k}/seed{s}' for k, s in CONFIRMATION if (k, s) not in scores]
    complete = not missing_confirmation and not problems and not changed
    report = dict(status='COMPLETE' if complete else 'INCOMPLETE', n_confirmation_fits_scored=len(fits),
                  n_confirmation_fits_declared=len(CONFIRMATION), missing_confirmation_fits=missing_confirmation,
                  pilot_scored=PILOT in scores, provenance_problems=problems, changed_sources=changed,
                  readout_A_source=('CPU through the pilot generation path (common random numbers with B, as in the pilot)'
                                    if a_source == 'cpu' else 'archive written by the CLI (training device)'),
                  readout_A_devices={f'fold{k}/seed{s}': {lab: info.get('generation_device') for lab, info in
                                     scores[(k, s)]['readouts'].items()} for k, s in scores})
    if fits:
        rows = {(k, s): scores[(k, s)]['rows'] for k, s in fits}
        a = [flat(rows[f].get(a_label, rows[f]['A'])) for f in fits]
        b = [flat(rows[f]['B']) for f in fits]
        folds = [k for k, _ in fits]
        report['question1_B_vs_A'] = question1(a, b, folds, complete)
        if a_source == 'cpu' and any('A_cli' in rows[f] for f in fits):
            a_cli = [flat(rows[f].get('A_cli', rows[f]['A'])) for f in fits]
            report['question1_B_vs_A_cli_secondary'] = question1(a_cli, b, folds, complete)
        pick = lambda f, lab: rows[f].get(lab, rows[f]['A'])
        report['question2'] = {label: question2([flat(pick(f, lab)) for f in fits], [flat(pick(f, lab), True) for f in fits],
                                                folds, complete)
                               for label, lab in (('A', a_label), ('B', 'B'))}
        report['per_fit'] = {f'fold{k}/seed{s}': dict(
            repeat=repeat_of(k), gates_B_vs_A=scores[(k, s)]['gates_B_vs_A'],
            **{lab: {key: v for key, v in flat(row).items()
                     if key in (*GATE_ORDER, *SPREAD_COVERAGE, 'sd_width', 'combined_width', *TRAIN_KEYS)}
               for lab, row in rows[(k, s)].items()},
            calibrated={lab: {f'{g}_{m}': row['calibrated_heldout'][g][m] for g in ('sd', 'combined')
                              for m in ('coverage', 'interval_score', 'width')} for lab, row in rows[(k, s)].items()},
            split_sizes=scores[(k, s)]['split']['sizes']) for k, s in fits}
        report['secondary'] = secondary(scores, fits, a_label)
    if PILOT in scores:
        p = scores[PILOT]
        report['pilot_not_in_confirmation_set'] = dict(gates_B_vs_A=p['gates_B_vs_A'],
            reproduction=p.get('pilot_reproduction'),
            calibration_useful={lab: row['calibration_useful_on_both_primary_groups'] for lab, row in p['rows'].items()})
        report['secondary_pilot'] = secondary(scores, [PILOT], 'A')
    return report


def print_report(report):
    print(f'NATIVE EMPF CONFIRMATION: {report["status"]} '
          f'({report["n_confirmation_fits_scored"]}/{report["n_confirmation_fits_declared"]} confirmation fits scored)')
    if report['missing_confirmation_fits']:
        print('MISSING (not scored):', ', '.join(report['missing_confirmation_fits']))
    if not report['pilot_scored']:
        print('MISSING: pilot fold0/seed0 (reproduction)')
    for problem in report['provenance_problems'] + [f'changed source: {c}' for c in report['changed_sources']]:
        print('PROBLEM:', problem)
    print('Readout A:', report['readout_A_source'])
    if 'pilot_not_in_confirmation_set' in report:
        p = report['pilot_not_in_confirmation_set']
        print('\nPilot fold0/seed0 (NOT in the confirmation set): reproduction', p['reproduction'])
        print('  pilot gates B vs A:', p['gates_B_vs_A'], ' calibration useful:', p['calibration_useful'])
        for name in ('previous_default', 'empirical_stage_blind', *[n for n, _, v in REFERENCE_SETS if v != 'covariance']):
            for tag, rows in report['secondary_pilot'][name].get('comparisons', {}).items():
                if tag.startswith('empf_A_minus') or tag.startswith('empf_B_calibrated'):
                    print_paired(f'  pilot {tag} vs {name}:', rows, 'reference', tag.split('_minus')[0])
        print_stream_spread(report['secondary_pilot'], '  pilot ')
    if 'question1_B_vs_A' not in report:
        print('\nNo confirmation fit scored yet.'); return
    q = report['question1_B_vs_A']
    print(f'\nQUESTION 1 (B quarter vs A default): {q["status"]} [on scored fits: {q["provisional_on_scored_fits"]}]')
    for k, v in q['gates_on_means'].items():
        print(f'  gate {k:26s} {"pass" if v else "FAIL"}')
    for name, r in q['repeats_both_proper_scores_lower'].items():
        print(f'  repeat {name:24s} ' + ('no fits' if r is None else f'SD score lower={r["sd"]} combined score lower={r["combined"]} (n={r["n_fits"]})'))
    print('  ' + q['coverage_rule'])
    print('  diagnostic mean per-fit |coverage-0.8|:', q['coverage_distance_mean_of_per_fit'])
    print_paired('  Paired B - A over fits (spread/coverage shown with RMSE/energy; train_* = TRAIN partition):',
                 q['paired'], 'A', 'B')
    if 'question1_B_vs_A_cli_secondary' in report:
        s = report['question1_B_vs_A_cli_secondary']
        print(f'\n  Secondary, A as written by the CLI: all ten gates={s["all_ten_gates"]}, repeat condition={s["repeat_condition"]}')
    print('\n  Per fit, held-out vs TRAIN raw SD/combined coverage and interval score:')
    for fit, row in report['per_fit'].items():
        for lab in ('A', 'B'):
            r = row[lab]
            print(f'    {fit:12s} {lab}: held SD {r["sd_coverage"]:.4f}/{r["sd_interval_score"]:.4f} '
                  f'comb {r["combined_coverage"]:.4f}/{r["combined_interval_score"]:.4f} | TRAIN SD '
                  f'{r.get("train_sd_coverage", float("nan")):.4f}/{r.get("train_sd_interval_score", float("nan")):.4f} '
                  f'comb {r.get("train_combined_coverage", float("nan")):.4f}/'
                  f'{r.get("train_combined_interval_score", float("nan")):.4f}')
    for label, q2 in report['question2'].items():
        print(f'\nQUESTION 2 (SD calibration of {label}): {q2["status"]} [on scored fits: {q2["provisional_on_scored_fits"]}]')
        print('  pooled:', q2['pooled'])
        for name, r in q2['repeats'].items():
            print(f'  {name}:', r)
        print('  ' + q2['coverage_rule'])
        print('  diagnostic mean per-fit |coverage-0.8|:', q2['coverage_distance_mean_of_per_fit'])
        print_paired(f'  Calibrated - raw for {label}:', q2['paired_calibrated_minus_raw'], 'raw', 'calibrated')
    for name, _, _ in REFERENCE_SETS:
        sec = report['secondary'][name]
        print(f'\nSECONDARY (descriptive) vs {name}: n={sec["n"]} [{sec.get("role", "")}]')
        for note in sec.get('notes', []):
            print('  note:', note)
        for tag, rows in sec.get('comparisons', {}).items():
            print_paired(f'  {tag} (paired, other minus reference):', rows, 'reference', tag.split('_minus')[0])
    print_stream_spread(report['secondary'], '')
    print('\n' + report['secondary']['interval_availability'])


def print_stream_spread(sec, prefix):
    sts = sec.get('previous_default_stream_to_stream', {})
    if sts.get('n'):
        print(f'\n{prefix}Previous default stream-to-stream spread (folds 0-2, same fits, namespace 1 minus 6, n={sts["n"]}):')
        for key, v in sts['spread'].items():
            print(f'{prefix}  {key:26s} max |diff| {v["max_abs_difference"]:.6f}  diffs ' +
                  ' '.join(f'{d:+.6f}' for d in v['differences']))
    for key, v in sec.get('empf_A_minus_previous_vs_stream_spread', {}).get('values', {}).items():
        print(f'{prefix}  EMPF A - previous {key}: {v["empf_A_minus_previous_mean"]:+.6f}; within the previous default\'s '
              f'stream spread ({v["previous_stream_to_stream_max_abs"]:.6f}): {v["within_previous_default_stream_spread"]}')


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--a-source', choices=('cpu', 'cli'), default='cpu',
                   help='readout A for the paired comparison: CPU regeneration with common streams (default, as the '
                        'pilot) or the archive written by the CLI on the training device')
    p.add_argument('--out', type=Path, default=OUT/'analysis.json')
    p.add_argument('--quiet', action='store_true')
    a = p.parse_args(argv)
    report = analyze(a.a_source)
    if not a.quiet:
        print_report(report)
    write(a.out, report)
    print(f'\nwrote {a.out} ({report["status"]})')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
