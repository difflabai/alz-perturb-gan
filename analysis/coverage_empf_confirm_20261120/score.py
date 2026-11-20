"""Score one native EMPF confirmation fit with the pilot's readouts, scorer and SD calibration.

For fold K / seed S this script
  (a) verifies the fit with the application's own checks (check_fit, check_cache,
      check_weights, checkpoint/recipe/regularizer audit; the selected-model hash
      only for the selected pilot fit),
  (b) locates readout A (cellshape@0.5:biascal:stagehalf) as written by the CLI,
      or for the pilot the verification archive,
  (c) generates readout B (cellshape@0.25:biascal:stagehalf) on CPU through the
      pilot's generation path and streams (G.generate_fit with
      SeedSequence([20261114,6,fold,development index,0]); the joint bias/nonzero
      calibration is refitted inside generate_fit on that fit's TRAIN targets),
  (d) scores every readout with the pilot's scorer (architecture_score.score and
      the pilot's interval summary) on fold K's held-out donors in fold K's
      TRAIN-fitted PCA basis,
  (e) applies the pilot's cross-fitted SD-only calibration with the pilot's donor
      split rule.

The CLI writes readout A on the training device. The pilot's A and B were both
generated on CPU, so their random streams were common. When the CLI archive was
made on another device this script also regenerates A on CPU through the same
path (label A; the CLI archive is scored as A_cli), so the paired B-A comparison
uses common random numbers exactly as in the pilot. See PROTOCOL.md: "with the
pilot's code paths and generation streams".

New outputs go only to runs/coverage_empf_confirm_20261120/.

Review repairs (2026-11-20, before any confirmation fit was scored): crossfit() now
has an end-to-end leakage guard (F3) and reproduce_pilot() compares the generated
archive SHA instead of ignoring it (F4). Neither changes any score; the pilot
reproduction was rerun under the new pins. The superseded first pilot run is kept
(JSON only) under runs/coverage_empf_confirm_20261120/superseded/.
"""
import os
os.environ.update(OMP_NUM_THREADS='3', OPENBLAS_NUM_THREADS='3', MKL_NUM_THREADS='3',
                  CUDA_VISIBLE_DEVICES='', PYTHONDONTWRITEBYTECODE='1')
import argparse
import fcntl
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import sys
import time

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OUT = ROOT/'runs/coverage_empf_confirm_20261120'
PIN = HERE/'source_manifest.json'
PILOT_DIR = ROOT/'analysis/coverage_empf_20261117'
PILOT_OUT = ROOT/'runs/coverage_empf_20261117'
PILOT_COMPLETION = PILOT_DIR/'completion'
PILOT_A = ROOT/'runs/microglia_empf/verification/final_generated.npz'
MODE_A = 'cellshape@0.5:biascal:stagehalf'
MODE_B = 'cellshape@0.25:biascal:stagehalf'
STREAM_NAMESPACE = (20261114, 6)
SPLIT_SALT = ('interval-calibration-split', 20261113)
PILOT = (0, 0)
CONFIRMATION = tuple((k, s) for s in (0, 1) for k in range(9) if (k, s) != PILOT)
REPEATS = {'folds0-2_without_pilot': (0, 1, 2), 'folds3-5': (3, 4, 5), 'folds6-8': (6, 7, 8)}
GUARDS = {'pb_rmse': .005, **{k: .02 for k in ('meanlog_rmse_gene', 'nonzero_rmse_gene',
          'logvar_rmse_gene', 'pc_energy_rep', 'cell_energy_rep')}}
HOST_RACE = 'existing derived EMPF host differs'
ACTIVATION_LOCK = OUT/'activation.lock'


# --------------------------------------------------------------------------- small helpers
def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n'); tmp.replace(path)


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


def rel(path):
    path = Path(path).resolve()
    return str(path.relative_to(ROOT)) if ROOT in path.parents else str(path)


def fit_dir(cfg, fold, seed):
    """Default fit location, exactly as scripts/microglia.py resolves it for generate."""
    if (fold, seed) == (cfg['runtime']['selected_fold'], cfg['runtime']['selected_seed']):
        return ROOT/cfg['runtime']['selected_fit']
    t = cfg['training']
    return ROOT/cfg['runtime']['fit_output_root']/t['arm']/t['panel']/f'fold{fold}/seed{seed}'


def cli_readout_a(cfg, fold, seed):
    if (fold, seed) == PILOT:
        return PILOT_A
    return ROOT/cfg['runtime']['generation_output_root']/f'fold{fold}/seed{seed}'/f'generated_{MODE_A}.npz'


def receipt_of(npz):
    npz = Path(npz)
    return npz.with_name(npz.stem+'.receipt.json')


def out_dir(fold, seed, kind='scores'):
    return OUT/kind/f'fold{fold}/seed{seed}'


# --------------------------------------------------------------------------- pure, testable rules
def split_halves(donors, is_heldout, stable_seed):
    """The pilot's split: held-out donors ordered by stable_seed(salt, 20261113, donor ID),
    then np.array_split(order, 2). With 12 donors that is 6/6; with 13 the FIRST half
    (lowest seeds) has 7 donors and the second 6 (np.array_split puts the extra element
    first). Returns two integer index arrays into ``donors``."""
    donors = np.asarray(donors).astype(str)
    held = np.flatnonzero(np.asarray(is_heldout, dtype=bool))
    order = sorted(held, key=lambda i: stable_seed(SPLIT_SALT[0], SPLIT_SALT[1], str(donors[i])))
    halves = np.array_split(order, 2)
    if not all(len(h) for h in halves) or set(halves[0]) & set(halves[1]):
        raise ValueError('held-out donors cannot be split into two disjoint halves')
    return halves


def crossfit_once(low, high, observed, donors, is_heldout, halves, C, pilot_check=True):
    """One pass of the pilot's cross-fitted SD calibration (run.py analyze, unchanged
    arithmetic, including the pilot's own in-loop check). observed: [repeats, donors, 100];
    repeat 0 calibrates. Returns calibrated bounds for the held-out donors (other rows
    unchanged) and the two (cal, test, record) triples.

    The pilot's in-loop check (kept for fidelity) perturbs observed[0, test] and refits
    on observed[0, cal]; it only fails when the two halves overlap, because fit_sd never
    receives the evaluation rows. The real leakage guard is in crossfit() below."""
    donors = np.asarray(donors).astype(str)
    is_heldout = np.asarray(is_heldout, dtype=bool)
    train = donors[~is_heldout].tolist()
    lo, hi = low.copy(), high.copy(); records = []
    for swap in (0, 1):
        cal, test = halves[swap], halves[1-swap]
        if not is_heldout[cal].all() or not is_heldout[test].all():
            raise ValueError('calibration/evaluation donors must be GAN-held-out')
        record = C.fit_sd(low[cal], high[cal], observed[0, cal], donors[cal], train)
        if pilot_check:
            changed = observed[0].copy(); changed[test] = 1e6
            if record != C.fit_sd(low[cal], high[cal], changed[cal], donors[cal], train):
                raise AssertionError('evaluation donor counts changed a calibration record')
        lo[test], hi[test] = C.apply_sd(low[test], high[test], record, donors[test])
        records.append((cal, test, record))
    if not (np.array_equal(lo[:, :50], low[:, :50]) and np.array_equal(hi[:, :50], high[:, :50])):
        raise AssertionError('calibration changed mean-coordinate bounds')
    return lo, hi, records


def crossfit(low, high, observed, donors, is_heldout, halves, C, check_leakage=True):
    """The pilot's cross-fitted SD calibration plus an end-to-end leakage guard.

    Review finding F3 (2026-11-20): the pilot's in-loop check cannot detect leakage of
    evaluation-donor counts, because it compares two fits on the same calibration rows.
    With check_leakage, for each swap the whole pass is rerun with that swap's evaluation
    donors' observed summaries replaced in EVERY real repeat (observed[:, test] + 1e6);
    the swap's record and the calibrated bounds of its evaluation donors must be
    bit-identical. The returned values are those of the unperturbed pass, so the guard
    cannot change any number."""
    lo, hi, records = crossfit_once(low, high, observed, donors, is_heldout, halves, C, pilot_check=check_leakage)
    if check_leakage:
        for swap, (cal, test, record) in enumerate(records):
            changed = np.array(observed, copy=True); changed[:, test] = changed[:, test] + 1e6
            lo2, hi2, records2 = crossfit_once(low, high, changed, donors, is_heldout, halves, C, pilot_check=False)
            if records2[swap][2] != record:
                raise AssertionError(f'evaluation donor counts changed calibration record {swap}')
            if not (np.array_equal(lo2[test], lo[test]) and np.array_equal(hi2[test], hi[test])):
                raise AssertionError(f'evaluation donor counts changed their own calibrated bounds (swap {swap})')
    return lo, hi, records


def closer(new, old, target=.8):
    return abs(new-target) < abs(old-target)


def pilot_gates(base, candidate):
    """The pilot's ten gates (run.py analyze), for any pair of score rows."""
    gates = {f'{group}_{metric}': (closer(candidate['raw_heldout'][group][metric], base['raw_heldout'][group][metric])
             if metric == 'coverage' else candidate['raw_heldout'][group][metric] < base['raw_heldout'][group][metric])
             for group in ('sd', 'combined') for metric in ('coverage', 'interval_score')}
    gates.update({key: candidate['historical'][key] <= base['historical'][key]*(1+limit)
                  for key, limit in GUARDS.items()})
    return gates


def calibration_useful(row):
    return all(closer(row['calibrated_heldout'][g]['coverage'], row['raw_heldout'][g]['coverage']) and
               row['calibrated_heldout'][g]['interval_score'] < row['raw_heldout'][g]['interval_score']
               for g in ('sd', 'combined'))


# --------------------------------------------------------------------------- source pins
def scoring_digest(pin):
    """Digest of the scoring pins only; analysis pins can be added later without changing it."""
    core = {k: pin[k] for k in ('schema', 'application_manifest_sha256', 'execution_sources', 'inputs')}
    return hashlib.sha256(json.dumps(core, sort_keys=True).encode()).hexdigest()


def check_sources(R):
    text = PIN.read_bytes(); pin = json.loads(text); digest = scoring_digest(pin)
    for name, expected in pin['execution_sources'].items():
        if sha(ROOT/name) != expected:
            raise ValueError(f'confirmation source changed: {name}')
    if sha(R.SOURCE_MANIFEST) != pin['application_manifest_sha256']:
        raise ValueError('application source manifest changed')
    R.check_sources()
    for name, expected in pin['inputs'].items():
        if sha(ROOT/name) != expected:
            raise ValueError(f'confirmation input changed: {name}')
    if PIN.read_bytes() != text:
        raise ValueError('confirmation source manifest changed during checking')
    return digest


ANALYSIS_SOURCES = ('analysis/coverage_empf_confirm_20261120/analyze.py',
                    'analysis/coverage_empf_confirm_20261120/test_confirm.py')


def pin_analysis():
    """Add or refresh only the analysis pins (analyze.py, tests); scoring pins are untouched."""
    pin = json.loads(PIN.read_text()); before = scoring_digest(pin)
    pin['analysis_sources'] = {n: sha(ROOT/n) for n in ANALYSIS_SOURCES}
    assert scoring_digest(pin) == before
    write(PIN, pin)
    return pin


def pin_sources():
    """Write the scoring pins once, before any score exists."""
    if (OUT/'scores').exists() and any((OUT/'scores').rglob('*_raw_score.json')):
        raise ValueError('scores exist; scoring sources cannot be re-pinned')
    names = ['analysis/coverage_empf_confirm_20261120/PROTOCOL.md',
             'analysis/coverage_empf_confirm_20261120/score.py',
             'analysis/coverage_empf_20261117/run.py',
             'analysis/coverage_empf_20261117/calibration.py',
             'analysis/improve_20261109/r5/architecture_score.py',
             'analysis/improve_20261109/r5/interval_metrics.py',
             'analysis/improve_20261109/r5/evaluate.py',
             'analysis/improve_20261109/r5/metrics.py',
             'analysis/improve_20261109/r5/generate.py',
             'analysis/improve_20261109/r5/mode_cellshape.py',
             'scripts/microglia.py', 'scripts/microglia_runtime.py', 'configs/microglia_default.json']
    inputs = ['runs/microglia_empf/verification/final_generated.npz',
              'runs/microglia_empf/verification/final_generated.receipt.json',
              'runs/coverage_empf_20261117/quarter_generated.npz',
              'runs/coverage_empf_20261117/quarter_generated.receipt.json',
              'analysis/default_empf_20261117/selected_model.json',
              *[str(p.relative_to(ROOT)) for p in sorted(PILOT_COMPLETION.rglob('*.json'))]]
    R = load_module('application_runtime', ROOT/'scripts/microglia_runtime.py')
    value = dict(schema=1, application_manifest_sha256=sha(R.SOURCE_MANIFEST),
                 execution_sources={n: sha(ROOT/n) for n in names},
                 inputs={n: sha(ROOT/n) for n in inputs})
    write(PIN, value)
    return value


# --------------------------------------------------------------------------- activation (race-aware)
def young_runtime_processes(max_age=90.):
    """Other processes running scripts/microglia_runtime.py that started < max_age s ago;
    those are the ones that may be activating the EMPF host right now."""
    hz = os.sysconf('SC_CLK_TCK'); uptime = float(Path('/proc/uptime').read_text().split()[0]); found = []
    for proc in Path('/proc').iterdir():
        if not proc.name.isdigit() or int(proc.name) == os.getpid():
            continue
        try:
            cmd = (proc/'cmdline').read_bytes().replace(b'\0', b' ').decode(errors='replace')
            if 'microglia_runtime.py' not in cmd:
                continue
            start = int((proc/'stat').read_text().rsplit(')', 1)[1].split()[19])/hz
        except (OSError, IndexError, ValueError):
            continue
        if uptime-start < max_age:
            found.append(int(proc.name))
    return found


def activate(R, log=print):
    """Activate the EMPF host without racing my own processes; retry once on the known race."""
    OUT.mkdir(parents=True, exist_ok=True)
    with open(ACTIVATION_LOCK, 'a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        deadline = time.monotonic()+900
        while (young := young_runtime_processes()) and time.monotonic() < deadline:
            log(f'waiting for recently started runtime processes {young} before EMPF activation')
            time.sleep(15)
        try:
            return R.activate('cpu', .04)
        except RuntimeError as exc:
            if HOST_RACE not in str(exc):
                raise
            log(f'EMPF activation race ({exc}); retrying once in 10 s')
            time.sleep(10)
            return R.activate('cpu', .04)


# --------------------------------------------------------------------------- verification
def verify_fit(R, M, cfg, fold, seed, fit):
    cache = M.cache_path(cfg, fold)
    anchor = R.check_cache(cache, fold, held=True)
    receipt = json.loads((fit/'fit_receipt.json').read_text())
    if receipt.get('status') != 'training_completed':
        raise ValueError(f'fit receipt status is {receipt.get("status")!r}, not training_completed')
    M.check_fit(fit, cfg, fold, seed)
    saved = json.loads((fit/'config.json').read_text())
    if sha(cache/'manifest.json') != saved['cache_manifest_SHA256']:
        raise ValueError('fit was trained on a different cache')
    weights = R.check_weights(fit, receipt)
    selected_model = fit.resolve() == (ROOT/cfg['runtime']['selected_fit']).resolve()
    if selected_model:
        selected = json.loads((R.SELECTION/'selected_model.json').read_text())
        if weights != selected['weights_sha256']:
            raise ValueError('wrong selected native model')
        for name, expected in selected['metadata_sha256'].items():
            if sha(fit/name) != expected:
                raise ValueError('selected native fit metadata changed')
    metadata = {name: sha(fit/name) for name in ('config.json', 'fit_receipt.json', 'no_skip_audit.json',
                'checkpoint_receipt.json', 'learning_curve.jsonl', 'progress.json') if (fit/name).exists()}
    return cache, anchor, saved, dict(fit=rel(fit), selected_model=selected_model,
        selected_model_hash_checked=selected_model, weights_sha256=weights, metadata_sha256=metadata,
        receipt_status=receipt['status'], training_device=receipt.get('device'),
        cache_anchor=anchor)


def verify_cli_a(R, fold, fit, anchor, path):
    """Readout A as written by the CLI (or the pilot verification archive)."""
    rp = receipt_of(path)
    if not path.exists() or not rp.exists():
        raise FileNotFoundError(f'readout A missing: {path}')
    receipt = json.loads(rp.read_text()); digest = sha(path)
    problems = []
    if receipt.get('generated_SHA256') != digest: problems.append('archive hash')
    if receipt.get('mode') != MODE_A: problems.append('mode')
    if Path(receipt.get('fit', '')).resolve() != fit.resolve(): problems.append('fit')
    if receipt.get('generation_streams') != [*STREAM_NAMESPACE, fold, 'development donor index', 0]:
        problems.append('streams')
    if receipt.get('cache_anchor', {}).get('manifest_sha256') != anchor['manifest_sha256']: problems.append('cache')
    if receipt.get('application_default', {}).get('source_manifest_sha256') != sha(R.SOURCE_MANIFEST):
        problems.append('application source manifest')
    if problems:
        raise ValueError(f'readout A receipt does not match its archive/fit: {problems}')
    return dict(path=rel(path), sha256=digest, receipt=rel(rp), receipt_sha256=sha(rp),
                generation_device=receipt.get('generation_device'), mode=MODE_A)


# --------------------------------------------------------------------------- generation
def patch_streams(T, G):
    """Same five named streams as the pilot (run.py setup) and the application runtime."""
    def streams(fold, index):
        children = T.np.random.SeedSequence([*STREAM_NAMESPACE, fold, index, 0]).spawn(5)
        return {key: int(child.generate_state(1, dtype=T.np.uint64)[0]) % (2**63-16)
                for key, child in zip(G.GEN_STREAMS, children)}
    G.generation_streams = streams


def generate(G, fit, cache, mode, output, digest, fold, extra):
    rp = receipt_of(output)
    if output.exists():
        receipt = json.loads(rp.read_text()) if rp.exists() else {}
        if (receipt.get('generated_SHA256') == sha(output) and receipt.get('mode') == mode and
                receipt.get('experiment_manifest_sha256') == digest and
                Path(receipt.get('fit', '')).resolve() == fit.resolve()):
            print(f'reusing {rel(output)}', flush=True)
            return receipt
        raise ValueError(f'{output} exists without a matching receipt; inspect before rerunning')
    _, receipt = G.generate_fit(fit, cache, mode=mode, out=output, device='cpu')
    receipt.update(experiment='native_EMPF_fold_seed_confirmation', experiment_manifest_sha256=digest,
                   generation_streams=[*STREAM_NAMESPACE, fold, 'development donor index', 0],
                   model_weights_unchanged=True, training_updates=0, default_changed=False, **extra)
    write(rp, receipt)
    return receipt


# --------------------------------------------------------------------------- scoring
def score_readout(label, path, cache, real, observed, halves, out, fit, anchor, digest, A, I, C, W):
    """Exactly the pilot's analyze() row for one readout."""
    path = Path(path)
    row = A.score(path, cache, out/f'{label}_raw_score.json')
    generated = A.E.align_generated(A.E.load_generated(path), real)
    sample = A.E.MX.pc_bag_summary(generated['pcs'].astype(np.float32))
    low, high = I.bounds(sample, 'percentile')
    lo, hi, records = crossfit(low, high, observed, real.donors, real.is_heldout, halves, C)
    swaps = []
    for swap, (cal, test, record) in enumerate(records):
        record['binding'] = dict(fit=rel(fit), final_checkpoint_sha256=sha(fit/'final.pt'),
            source_generated_sha256=sha(path), cache_manifest_sha256=anchor['manifest_sha256'],
            experiment_manifest_sha256=digest,
            coordinate_basis=f'frozen fold{real.fold} TRAIN-fitted PCA50 mean then SD', calibration_real_repeat=0)
        artifact = out/f'calibration/{label}_half{swap}.json'; write(artifact, record)
        mask = np.isin(np.arange(len(real.donors)), test)
        swaps.append(dict(calibration_donors=real.donors[cal].tolist(), evaluation_donors=real.donors[test].tolist(),
            rank=record['rank'], artifact=rel(artifact), artifact_sha256=sha(artifact),
            raw=W.scores(low, high, observed, mask, I), calibrated=W.scores(lo, hi, observed, mask, I)))
    raw = W.scores(low, high, observed, real.is_heldout, I)
    calibrated = W.scores(lo, hi, observed, real.is_heldout, I)
    value = dict(generated_sha256=sha(path), generation_receipt_sha256=sha(receipt_of(path)),
        raw_score_sha256=sha(out/f'{label}_raw_score.json'),
        historical=row['historical'], raw_heldout=raw, calibrated_heldout=calibrated,
        raw_train=W.scores(low, high, observed, ~real.is_heldout, I), swaps=swaps,
        raw_rank_heldout=I.summarize(sample, observed, real.is_heldout, 'rank'))
    value['calibration_useful_on_both_primary_groups'] = calibration_useful(value)
    # Per-donor assembled cross-fitted scores (each held-out donor scored once, by the half it was not in).
    per_donor = {}
    for i in np.flatnonzero(real.is_heldout):
        mask = np.arange(len(real.donors)) == i
        per_donor[str(real.donors[i])] = dict(raw=W.scores(low, high, observed, mask, I),
                                              calibrated=W.scores(lo, hi, observed, mask, I))
    value['per_donor_heldout'] = per_donor
    np.savez_compressed(out/f'{label}_crossfitted_intervals.npz', donors=real.donors, is_heldout=real.is_heldout,
                        raw_low=low, raw_high=high, calibrated_low=lo, calibrated_high=hi)
    value['crossfitted_intervals_sha256'] = sha(out/f'{label}_crossfitted_intervals.npz')
    return value


# --------------------------------------------------------------------------- pilot reproduction
def numeric_leaves(value, prefix=''):
    if isinstance(value, dict):
        for k, v in value.items():
            yield from numeric_leaves(v, f'{prefix}/{k}')
    elif isinstance(value, list):
        for i, v in enumerate(value):
            yield from numeric_leaves(v, f'{prefix}[{i}]')
    else:
        yield prefix, value


def compare_tree(expected, got, ignore=()):
    """Every leaf; numbers compared exactly and by absolute difference."""
    e, g = dict(numeric_leaves(expected)), dict(numeric_leaves(got))
    numbers = diff = 0.; mismatched = []; ignored = []; count = 0
    for key in sorted(set(e) | set(g)):
        if any(key.startswith(p) for p in ignore):
            if e.get(key) != g.get(key): ignored.append(key)
            continue
        a, b = e.get(key, '<missing>'), g.get(key, '<missing>')
        if isinstance(a, (int, float)) and not isinstance(a, bool) and isinstance(b, (int, float)) and not isinstance(b, bool):
            count += 1; d = abs(float(a)-float(b)); diff = max(diff, d)
            if a != b: mismatched.append(key)
        elif a != b:
            mismatched.append(key)
    return dict(numeric_values=count, max_abs_difference=diff, mismatched=mismatched,
                differing_ignored_provenance_fields=ignored, exact=not mismatched)


def reproduce_pilot(rows, out, b_path):
    """Compare the regenerated pilot outputs with the committed pilot completion files.

    Review finding F4 (2026-11-20): the regenerated B archive is byte-identical to the
    pilot's quarter archive, so the generated-archive SHA (scorer fingerprint and
    calibration binding) is compared, not ignored. Only the archive's source path and
    the experiment-manifest SHA in the calibration binding are excluded; both must
    differ because this pass writes to its own folder under its own source pins."""
    report = {}
    with np.load(PILOT_OUT/'quarter_generated.npz', allow_pickle=False) as old, np.load(b_path, allow_pickle=False) as new:
        same = sorted(old.files) == sorted(new.files) and all(np.array_equal(old[k], new[k]) for k in old.files)
        report['readout_B_arrays_identical_to_pilot_quarter_generated'] = bool(same)
    report['readout_B_archive_byte_identical_to_pilot_quarter_generated'] = sha(b_path) == sha(PILOT_OUT/'quarter_generated.npz')
    provenance = ('/source',)
    report['baseline_raw_score.json'] = compare_tree(json.loads((PILOT_COMPLETION/'baseline_raw_score.json').read_text()),
        json.loads((out/'A_raw_score.json').read_text()), ignore=provenance)
    report['quarter_raw_score.json'] = compare_tree(json.loads((PILOT_COMPLETION/'quarter_raw_score.json').read_text()),
        json.loads((out/'B_raw_score.json').read_text()), ignore=provenance)
    for old_label, label in (('baseline', 'A'), ('quarter', 'B')):
        for swap in (0, 1):
            name = f'calibration/{old_label}_half{swap}.json'
            report[name] = compare_tree(json.loads((PILOT_COMPLETION/name).read_text()),
                json.loads((out/f'calibration/{label}_half{swap}.json').read_text()),
                ignore=('/binding/experiment_manifest_sha256',))
    comparison = json.loads((PILOT_COMPLETION/'comparison.json').read_text())
    for old_label, label in (('baseline', 'A'), ('quarter', 'B')):
        keep = ('historical', 'raw_heldout', 'calibrated_heldout', 'raw_train', 'raw_rank_heldout',
                'calibration_useful_on_both_primary_groups')
        old = {k: comparison['rows'][old_label][k] for k in keep}
        old['swaps'] = [{k: s[k] for k in ('calibration_donors', 'evaluation_donors', 'rank', 'raw', 'calibrated')}
                        for s in comparison['rows'][old_label]['swaps']]
        new = {k: rows[label][k] for k in keep}
        new['swaps'] = [{k: s[k] for k in ('calibration_donors', 'evaluation_donors', 'rank', 'raw', 'calibrated')}
                        for s in rows[label]['swaps']]
        report[f'comparison.json/rows/{old_label}'] = compare_tree(old, new)
    report['comparison.json/candidate_gates'] = compare_tree(comparison['candidate_gates'], pilot_gates(rows['A'], rows['B']))
    files = [k for k in report if k.endswith('.json') or '/' in k]
    report['all_exact'] = bool(report['readout_B_arrays_identical_to_pilot_quarter_generated'] and
                               report['readout_B_archive_byte_identical_to_pilot_quarter_generated'] and
                               all(report[k]['exact'] for k in files))
    report['max_abs_difference'] = max(report[k]['max_abs_difference'] for k in files)
    report['numeric_values_compared'] = int(sum(report[k]['numeric_values'] for k in files))
    report['note'] = ('The regenerated B archive is byte-identical to the pilot quarter archive, so generated-archive '
                      'SHAs (scorer fingerprint and calibration binding) are compared. Only two provenance fields '
                      'that must differ are excluded: the archive source path in the raw score files and the '
                      'experiment-manifest SHA inside the calibration binding.')
    return report


# --------------------------------------------------------------------------- main
def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--fold', type=int, choices=range(9))
    p.add_argument('--seed', type=int, choices=(0, 1))
    p.add_argument('--fit', type=Path, help='fit directory (default: where the CLI writes/reads it)')
    p.add_argument('--cpus', help='optional comma list of three CPUs to pin to after activation')
    p.add_argument('--pin-sources', action='store_true', help='write the scoring pins (refused once scores exist)')
    p.add_argument('--pin-analysis', action='store_true', help='add/refresh only the analysis-source pins')
    a = p.parse_args(argv)
    if a.pin_sources:
        print(json.dumps(pin_sources(), indent=2)); return 0
    if a.pin_analysis:
        print(json.dumps(pin_analysis(), indent=2)); return 0
    if a.fold is None or a.seed is None:
        p.error('--fold and --seed are required')
    fold, seed = a.fold, a.seed
    out = out_dir(fold, seed); final = out/'score.json'
    if final.exists():
        print(f'already scored: {rel(final)}'); return 0
    if (fold, seed) != PILOT:
        rep = out_dir(*PILOT)/'reproduction.json'
        if not rep.exists() or not json.loads(rep.read_text()).get('all_exact'):
            raise SystemExit('pilot reproduction (fold 0 seed 0) has not been shown exact; refusing new fits')
    R = load_module('application_runtime', ROOT/'scripts/microglia_runtime.py')
    digest = check_sources(R); M = R.application(); cfg = M.load_config()
    fit = (a.fit or fit_dir(cfg, fold, seed)).resolve()
    cache, anchor, saved, fit_record = verify_fit(R, M, cfg, fold, seed, fit)
    readout_a = verify_cli_a(R, fold, fit, anchor, cli_readout_a(cfg, fold, seed))
    started = time.time()
    T, G, provenance = activate(R)
    if a.cpus:
        os.sched_setaffinity(0, [int(c) for c in a.cpus.split(',')][:3])
    R.check_checkpoint(T.torch.load(fit/'final.pt', map_location='cpu', weights_only=False), saved)
    expected = json.loads(M.RECIPE.read_text())['native']['recipe']
    if json.loads(json.dumps(T.recipe().to_dict())) != expected:
        raise ValueError('live trainer differs from the selected native EMPF recipe')
    patch_streams(T, G)
    import architecture_score as A
    import interval_metrics as I
    C = load_module('pilot_calibration', PILOT_DIR/'calibration.py')
    W = load_module('pilot_run', PILOT_DIR/'run.py')
    gen = out_dir(fold, seed, 'generated')
    extra = dict(fit_weights_sha256=fit_record['weights_sha256'], cache_anchor=anchor, native_provenance=provenance)
    readouts = {}
    receipt_b = generate(G, fit, cache, MODE_B, gen/'B_generated.npz', digest, fold, extra)
    readouts['B'] = dict(path=rel(gen/'B_generated.npz'), sha256=receipt_b['generated_SHA256'], mode=MODE_B,
                         generation_device=receipt_b['generation_device'], receipt=rel(receipt_of(gen/'B_generated.npz')),
                         receipt_sha256=sha(receipt_of(gen/'B_generated.npz')), retained=True)
    if readout_a['generation_device'] == 'cpu':
        readouts['A'] = dict(readout_a, source='CLI' if (fold, seed) != PILOT else 'pilot verification archive',
                             retained=True)
        paths = {'A': ROOT/readout_a['path']}
    else:
        receipt_a = generate(G, fit, cache, MODE_A, gen/'A_cpu_generated.npz', digest, fold, extra)
        readouts['A'] = dict(path=rel(gen/'A_cpu_generated.npz'), sha256=receipt_a['generated_SHA256'], mode=MODE_A,
                             generation_device='cpu', source='CPU regeneration through the pilot path',
                             receipt=rel(receipt_of(gen/'A_cpu_generated.npz')),
                             receipt_sha256=sha(receipt_of(gen/'A_cpu_generated.npz')))
        readouts['A_cli'] = dict(readout_a, source='CLI', retained=True)
        paths = {'A': gen/'A_cpu_generated.npz', 'A_cli': ROOT/readout_a['path']}
    paths['B'] = gen/'B_generated.npz'
    real = A.E.load_real(cache)
    if real.fold != fold or real.manifest_sha256 != anchor['manifest_sha256']:
        raise ValueError('real-cell cache does not match the fold')
    observed = A.observed_summaries(real, 64)
    halves = split_halves(real.donors, real.is_heldout, A.E.MX.stable_seed)
    train = set(real.donors[~real.is_heldout].tolist()) | set(json.loads((cache/'manifest.json').read_text())['train_donors'])
    for h in halves:
        if set(real.donors[h].tolist()) & train:
            raise ValueError('a calibration donor belongs to the GAN TRAIN set')
    rows = {label: score_readout(label, path, cache, real, observed, halves, out, fit, anchor, digest, A, I, C, W)
            for label, path in paths.items()}
    if check_sources(R) != digest or R.check_cache(cache, fold, held=True) != anchor:
        raise ValueError('confirmation source or cache changed during execution')
    value = dict(status='scored', fold=fold, seed=seed, confirmation_fit=(fold, seed) != PILOT,
        repeat=next(name for name, folds in REPEATS.items() if fold in folds),
        fit=fit_record, readouts=readouts, rows=rows,
        gates_B_vs_A=pilot_gates(rows['A'], rows['B']),
        gates_B_vs_A_cli=pilot_gates(rows['A_cli'], rows['B']) if 'A_cli' in rows else None,
        split=dict(rule="held-out donors ordered by stable_seed('interval-calibration-split',20261113,donor ID); "
                        'np.array_split(order,2); an odd count puts the extra donor in half 0',
                   half0=real.donors[halves[0]].tolist(), half1=real.donors[halves[1]].tolist(),
                   sizes=[len(halves[0]), len(halves[1])], n_heldout=int(real.is_heldout.sum()),
                   n_train=int((~real.is_heldout).sum())),
        hashes=dict(experiment_manifest_sha256=digest, application_manifest_sha256=sha(R.SOURCE_MANIFEST),
                    sources=json.loads(PIN.read_text())['execution_sources'], inputs=json.loads(PIN.read_text())['inputs'], cache_anchor=anchor,
                    fit_weights=fit_record['weights_sha256'], fit_metadata=fit_record['metadata_sha256']),
        native_provenance=provenance, seconds=time.time()-started, training_updates=0,
        empirical_cell_resampling_used=False, calibration_changes_generated_cells=False,
        cuda_initialized=bool(T.torch.cuda.is_initialized()))
    if (fold, seed) == PILOT:
        report = reproduce_pilot(rows, out, paths['B'])
        write(out/'reproduction.json', report)
        value['pilot_reproduction'] = dict(all_exact=report['all_exact'], max_abs_difference=report['max_abs_difference'],
                                           numeric_values_compared=report['numeric_values_compared'],
                                           report_sha256=sha(out/'reproduction.json'))
        print(json.dumps({k: v for k, v in report.items() if not isinstance(v, dict) or not v.get('exact', True)}, indent=2))
    # Prune: score and hash, then delete the CPU regeneration of A (the CLI archive and B are retained).
    for label, info in readouts.items():
        if not info.get('retained'):
            path = ROOT/info['path']
            if path.exists() and sha(path) == info['sha256']:
                path.unlink(); info['pruned'] = True
    value['output_bytes'] = sum(f.stat().st_size for d in (out, gen) for f in d.rglob('*') if f.is_file())
    write(final, value)
    print(json.dumps({'fold': fold, 'seed': seed, 'gates_B_vs_A': value['gates_B_vs_A'],
                      **{label: {g: {k: row['raw_heldout'][g][k] for k in ('coverage', 'interval_score', 'width')}
                                 for g in ('sd', 'combined')} for label, row in rows.items()}}, indent=2), flush=True)
    return 0


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except (ValueError, FileNotFoundError) as exc:
        raise SystemExit(f'confirmation score: {exc}')
