"""Pre-registered selection (inner only) and sealed exploratory reporting (plan v4 section 4).

select  : reads inner only, applies the rule, writes selection_frozen.json binding every input hash,
          and appends its sha to ledger.jsonl.
explore : refuses unless the selection file's sha is supplied and matches, a ledger entry exists,
          and every bound hash (receipts, protocol, bundle, rule, scoring, calibrators, sealed files)
          still matches the current artifacts. Output is post-selection and non-confirmatory.
Limits: the ledger and mtimes are writable and are an audit aid, not independent proof of chronology.
"""
from __future__ import annotations
import argparse, hashlib, json, os, stat, sys, time
from datetime import datetime, timezone
from pathlib import Path
from .gene_evaluation import paired_delta, all_seed_deltas
from .gene_features import file_sha256, json_hash
from .gene_calibration import calibrator_sha
import math
import torch
from .paths import scoped, create_new, check_name

ARMS = ('B0', 'G', 'H', 'GH', 'shuffle', 'G_frozen')
CANDIDATES = ('G', 'H', 'GH')            # A2: G_frozen is a diagnostic probe, never selectable
RULE = ("v4 pre-registered rule: precondition all seeds complete and gate |B0-G|<=0.10 at steps 1000 and 2000 with both in "
        "[0.3,0.7]; an arm in {G,H,GH} is selected only if its inner head-AUROC paired delta (raw logits) is >0 vs both B0 "
        "and shuffle in every seed AND the all_seed_deltas (1000 draws) hierarchical CI lower bound is >0 vs both. "
        "A1: calibration status is reported, not a veto. A2: G_frozen is reported, not selectable.")
RULE_SHA256 = hashlib.sha256(RULE.encode()).hexdigest()
DRAWS = 1000


def _load(path):
    return json.loads(Path(scoped(path)).read_text())


def _arm_dir(run, arm):
    return scoped(Path(scoped(run)) / arm)


class BindingError(PermissionError):
    pass


def _need(cond, msg):
    if not cond: raise BindingError(msg)


def _verify_provenance(eval_dir):
    """Every file listed in provenance/file_hashes.json must exist (scoped) and match; nothing unlisted but the list."""
    listed = _load(eval_dir / 'provenance' / 'file_hashes.json')
    for rel, want in listed.items():
        _need(file_sha256(scoped(eval_dir / rel)) == want, f'provenance hash mismatch: {eval_dir / rel}')
    present = {str(p.relative_to(eval_dir)) for p in Path(eval_dir).rglob('*') if p.is_file()}
    _need(present - set(listed) <= {'provenance/file_hashes.json'}, f'unlisted files in {eval_dir}: {sorted(present - set(listed))}')
    return file_sha256(scoped(eval_dir / 'provenance' / 'file_hashes.json'))



def _state_hashes(state):
    """Same digest as gene_checkpoint.state_hash, computed from the saved state dicts (no model construction)."""
    out = {}
    for name, sd in state.items():
        h = hashlib.sha256()
        for k, v in sorted(sd.items()):
            h.update(k.encode()); h.update(str(v.dtype).encode()); h.update(str(tuple(v.shape)).encode())
            h.update(v.detach().cpu().contiguous().numpy().tobytes())
        out[name] = h.hexdigest()
    return out


def _verify_checkpoint(ck_path, tj_path, *, seed, arm, protocol_sha256, grant, summary):
    """HOLD 2: the checkpoint must be a real trained schema-1 checkpoint that traces to its training.json
    (embedded receipt identical), with arm/seed/steps/protocol consistent with the grant and the evaluation."""
    where = f'seed {seed} {arm}'
    want = file_sha256(scoped(ck_path))
    try:
        blob = torch.load(scoped(ck_path), map_location='cpu', weights_only=True)
    except Exception as e:  # noqa: BLE001 - dummy bytes, truncated or foreign files
        raise BindingError(f'{where}: checkpoint is not a loadable checkpoint ({type(e).__name__})')
    _need(file_sha256(scoped(ck_path)) == want, f'{where}: checkpoint changed while reading')
    _need(isinstance(blob, dict) and blob.get('schema_version') == 1, f'{where}: checkpoint schema is not 1')
    cfg, state = blob.get('config') or {}, blob.get('state') or {}
    _need(set(state) == {'generator', 'critic', 'decision', 'prior'}, f'{where}: checkpoint state is incomplete')
    tensors = [v for sd in state.values() for v in sd.values() if torch.is_tensor(v) and v.is_floating_point()]
    _need(tensors and all(bool(torch.isfinite(v).all()) for v in tensors), f'{where}: checkpoint has non-finite or no weights')
    _need(any(float(v.std()) > 0 for v in tensors if v.numel() > 1), f'{where}: checkpoint weights are constant (dummy)')
    _need(blob.get('training_protocol_sha256') == protocol_sha256 == json_hash(blob.get('protocol')),
          f'{where}: checkpoint was not trained under the bound protocol (untrained or foreign)')
    t = json.loads(Path(scoped(tj_path)).read_text())
    _need(blob.get('training_receipt') == t, f'{where}: training.json differs from the receipt embedded in the checkpoint')
    _need(cfg.get('arm') == arm == t.get('arm'), f'{where}: arm differs between directory, checkpoint and training.json')
    _need(cfg.get('seed') == seed, f'{where}: seed differs between run and checkpoint')
    steps = t.get('steps')
    _need(isinstance(steps, int) and steps > 0 and steps == cfg.get('steps') == grant.get('max_steps_per_arm'),
          f'{where}: training steps differ between training.json, checkpoint config and the grant')
    hist = t.get('history') or []
    _need([r.get('step') for r in hist] == list(range(1, steps + 1)), f'{where}: training history does not cover steps 1..{steps}')
    _need(all(isinstance(r.get('j_loss'), (int, float)) and math.isfinite(r['j_loss']) for r in hist),
          f'{where}: training history lacks finite decision-head updates')
    hashes = _state_hashes(state)
    _need(summary.get('model_hashes_before') == hashes == summary.get('model_hashes_after'),
          f'{where}: evaluation was not run on this checkpoint')
    _need(summary.get('selection_eligible') is True, f'{where}: evaluation marks the checkpoint as not selection-eligible')
    return {'checkpoint_sha256': want, 'training_json_sha256': file_sha256(scoped(tj_path)), 'steps': steps,
            'state_hashes': hashes}


def bindings(runs, receipt, *, expected_seeds, bundle_sha256, protocol_sha256):
    """Every input the selection depends on, recomputed and cross-checked from current artifacts."""
    receipt = scoped(receipt); R = json.loads(Path(receipt).read_text())
    _need(str(R.get('schema', '')).startswith('supervisor_receipt_v4'), 'unexpected supervisor receipt schema')
    _need(R.get('rc') == 0 and R.get('stop_reason') is None, 'supervisor receipt does not record a clean run')
    _need(sorted(R.get('seeds', [])) == sorted(expected_seeds) == sorted(runs), 'seeds differ between receipt, runs and expectation')
    _need(all(any(x.get('seed') == sd and x.get('arm') == a and x.get('status') == 'ok' for x in R.get('arms', []))
              for sd in expected_seeds for a in ARMS), 'receipt does not show every seed x arm completed ok')
    out = {'supervisor_receipt': {'path': str(receipt), 'sha256': file_sha256(receipt)}, 'rule_sha256': RULE_SHA256,
           'analysis_code_sha256': file_sha256(__file__), 'expected_seeds': sorted(expected_seeds),
           'scoring_mode': 'raw_logit_ranks', 'bundle_sha256': bundle_sha256, 'protocol_sha256': protocol_sha256,
           'harness_commit': R.get('static_checks', {}).get('harness_commit'), 'runs': {}}
    for seed in sorted(runs):
        run = scoped(runs[seed]); rr = R.get('runs', {}).get(str(seed), {})
        _need(rr.get('path') and scoped(rr['path']) == run, f'seed {seed}: run path is not the one in the receipt')
        _need(run.parent == Path(receipt).parent, f'seed {seed}: run is not in the receipt cohort')
        grant_p = scoped(run / 'grant.json'); gsha = file_sha256(grant_p); g = json.loads(Path(grant_p).read_text())
        _need(gsha == rr.get('grant_sha256'), f'seed {seed}: grant hash differs from the receipt')
        _need(g.get('bundle_sha256') == bundle_sha256 and g.get('protocol_sha256') == protocol_sha256, f'seed {seed}: grant bundle/protocol differ')
        _need(g.get('seeds') == [seed] and g.get('harness_commit') == out['harness_commit'], f'seed {seed}: grant seed/commit differ')
        arms = {}
        for arm in ARMS:
            d = _arm_dir(run, arm); ev = scoped(d / 'evaluation'); sp = scoped(ev / 'metrics' / 'summary.json')
            s = json.loads(Path(sp).read_text())
            _need(s.get('seed') == seed and s.get('arm') == arm, f'seed {seed} {arm}: summary seed/arm differ')
            _need(s.get('protocol_sha256') == protocol_sha256, f'seed {seed} {arm}: protocol differs')
            _need(s.get('profile_mode') == 'generated', f'seed {seed} {arm}: profile mode is not generated')
            _need(s.get('scoring', {}).get('mode') == 'raw_logit_ranks', f'seed {seed} {arm}: scoring mode is not raw_logit_ranks')
            cal = s.get('calibration', {}).get('calibrator', {})
            _need(cal.get('sha256') and calibrator_sha(cal) == cal['sha256'], f'seed {seed} {arm}: calibrator hash does not verify')
            sealed = s.get('sealed', {}).get('exploratory') or {}
            _need(set(sealed) == {'exploratory.json', 'exploratory.pt', 'exploratory.jsonl'}, f'seed {seed} {arm}: sealed outputs missing')
            for fname, h in sealed.items():
                _need(file_sha256(scoped(ev / 'sealed' / fname)) == h, f'seed {seed} {arm}: sealed {fname} changed')
            ck = _verify_checkpoint(d / 'checkpoint.pt', d / 'training.json', seed=seed, arm=arm,
                                    protocol_sha256=protocol_sha256, grant=g, summary=s)
            arms[arm] = {'summary_sha256': file_sha256(sp), 'provenance_file_hashes_sha256': _verify_provenance(ev),
                         'checkpoint_sha256': ck['checkpoint_sha256'], 'training_json_sha256': ck['training_json_sha256'],
                         'training_steps': ck['steps'], 'state_hashes': ck['state_hashes'],
                         'calibrator_sha256': cal['sha256'], 'calibration_split_sha256': cal.get('calibration_split_sha256'),
                         'calibration_status': cal.get('status'), 'sealed_sha256': sealed}
        out['runs'][str(seed)] = {'path': str(run), 'grant_sha256': gsha, 'arms': arms}
    return out


def _gate(run):
    gates = {}
    for arm in ('B0', 'G'):
        h = _load(_arm_dir(run, arm) / 'training.json')['history']
        gates[arm] = {r['step']: r['realness_weight_mean'] for r in h if 'realness_weight_mean' in r and r['step'] in (1000, 2000)}
    ok = all(st in gates['B0'] and st in gates['G'] for st in (1000, 2000))
    ok = ok and all(abs(gates['B0'][st] - gates['G'][st]) <= 0.10 for st in (1000, 2000))
    ok = ok and all(0.3 <= v <= 0.7 for a in gates for v in gates[a].values())
    return ok, gates


LEDGER = 'ledger.jsonl'


def _open_ledger(out_dir, *, flags, expect=None, create=False):
    """Open the ledger relative to a no-follow directory handle and validate what was opened:
    regular file, owned by us, single link, same dev/inode as the directory entry, and (if given) the identity
    recorded in the selection. Returns (fd, identity)."""
    d = scoped(out_dir); dfd = os.open(d, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        try:
            try:
                fd = os.open(LEDGER, flags | os.O_NOFOLLOW, dir_fd=dfd)
            except FileNotFoundError:
                if not create: raise BindingError('ledger missing')
                fd = os.open(LEDGER, os.O_WRONLY | os.O_APPEND | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o644, dir_fd=dfd)
        except BindingError:
            raise
        except OSError as e:  # ELOOP from O_NOFOLLOW on a symlink, EEXIST race, etc.
            raise BindingError(f'ledger rejected: {e.strerror}')
        st = os.fstat(fd); entry = os.stat(LEDGER, dir_fd=dfd, follow_symlinks=False)
        ident = {'dev': st.st_dev, 'ino': st.st_ino}
        problems = [m for c, m in ((stat.S_ISREG(st.st_mode), 'not a regular file'), (st.st_uid == os.geteuid(), 'not owned by us'),
                                   (st.st_nlink == 1, 'has extra hard links'),
                                   ((entry.st_dev, entry.st_ino) == (st.st_dev, st.st_ino), 'directory entry changed during open'),
                                   (expect is None or expect == ident, 'dev/inode differs from the identity bound in the selection')) if not c]
        if problems:
            os.close(fd); raise BindingError('ledger rejected: ' + ', '.join(problems))
        return fd, ident
    finally:
        os.close(dfd)


def _append(fd, entry):
    data = (json.dumps(entry, sort_keys=True) + '\n').encode()
    _need(os.write(fd, data) == len(data), 'short ledger write')
    os.fsync(fd)


def select(runs, receipt, out_dir, *, expected_seeds, bundle_sha256, protocol_sha256):
    _need(sorted(runs) == sorted(expected_seeds), 'all pre-declared seeds are required')
    b = bindings(runs, receipt, expected_seeds=expected_seeds, bundle_sha256=bundle_sha256, protocol_sha256=protocol_sha256)
    summaries = {s: {a: _load(_arm_dir(runs[s], a) / 'evaluation' / 'metrics' / 'summary.json') for a in ARMS} for s in expected_seeds}
    gate = {s: _gate(runs[s]) for s in expected_seeds}
    pre = {'all_seeds_complete': True, 'gate_ok': all(g[0] for g in gate.values())}
    deltas, selected = {}, {}
    for arm in (*CANDIDATES, 'G_frozen'):
        for ref in ('B0', 'shuffle'):
            r = all_seed_deltas({s: summaries[s][arm] for s in expected_seeds}, {s: summaries[s][ref] for s in expected_seeds},
                                'inner', expected_seeds=list(expected_seeds), draws=DRAWS)
            deltas[f'{arm}_vs_{ref}'] = {'per_seed': {str(s): {'mean': v['mean_delta'], 'ci95': v['ci95']} for s, v in r['per_seed'].items()},
                                         'pooled_mean': r['all_seed_mean_delta'], 'pooled_ci95': r['hierarchical_ci95']}
    for arm in CANDIDATES:
        selected[arm] = bool(all(pre.values()) and all(
            all(d['mean'] > 0 for d in deltas[f'{arm}_vs_{ref}']['per_seed'].values()) and deltas[f'{arm}_vs_{ref}']['pooled_ci95'][0] > 0
            for ref in ('B0', 'shuffle')))
    calib = {str(s): {a: b['runs'][str(s)]['arms'][a]['calibration_status'] for a in ARMS} for s in expected_seeds}
    fd, lid = _open_ledger(out_dir, flags=os.O_WRONLY | os.O_APPEND, create=True)
    try:
        doc = {'schema': 'selection_frozen_v4_2', 'rule': RULE, 'rule_sha256': RULE_SHA256, 'split': 'inner',
               'created_utc': datetime.now(timezone.utc).isoformat(), 'preconditions': pre,
               'gate': {str(s): g[1] for s, g in gate.items()}, 'inner_deltas': deltas, 'selected': selected,
               'calibration_status_reported_not_veto': calib, 'bindings': b, 'ledger_identity': lid,
               'wording': 'selected on inner; not confirmed' if any(selected.values()) else 'selection criterion unmet; evidence inconclusive'}
        data = (json.dumps(doc, indent=1, sort_keys=True) + '\n').encode(); digest = hashlib.sha256(data).hexdigest()
        path = create_new(out_dir, f'selection_frozen_{digest[:12]}.json', data)
        _append(fd, {'event': 'select', 'selection': str(path), 'sha256': digest, 'utc': doc['created_utc']})
    finally:
        os.close(fd)
    return path, digest, doc


def explore(selection, selection_sha256, out_dir):
    path = scoped(selection)
    if not selection_sha256 or file_sha256(path) != str(selection_sha256).lower():
        raise PermissionError('explore refused: selection file missing or its sha256 does not match')
    _need(scoped(out_dir) == Path(path).parent, 'explore refused: out-dir must be the selection directory (ledger lives there)')
    doc = json.loads(Path(path).read_text())
    _need(doc.get('schema') == 'selection_frozen_v4_2' and 'ledger_identity' in doc, 'explore refused: selection schema/ledger identity missing')
    fd, _ = _open_ledger(out_dir, flags=os.O_RDONLY, expect=doc['ledger_identity'])
    try:
        text = b''
        while chunk := os.read(fd, 1 << 20): text += chunk
    finally:
        os.close(fd)
    _need(any(json.loads(l).get('sha256') == selection_sha256 and json.loads(l).get('event') == 'select'
              for l in text.decode().splitlines() if l.strip()), 'explore refused: no ledger entry for this selection')
    b = doc['bindings']; runs = {int(s): v['path'] for s, v in b['runs'].items()}
    try:
        now = bindings(runs, b['supervisor_receipt']['path'], expected_seeds=b['expected_seeds'],
                       bundle_sha256=b['bundle_sha256'], protocol_sha256=b['protocol_sha256'])
    except BindingError as e:
        raise BindingError(f'explore refused: {e}')
    _need(now == b, 'explore refused: bound artifacts changed since selection was frozen')
    sealed = {}
    for s, run in runs.items():
        for arm in ARMS:
            sd = _arm_dir(run, arm) / 'evaluation' / 'sealed'
            summ = _load(_arm_dir(run, arm) / 'evaluation' / 'metrics' / 'summary.json')
            sealed.setdefault(s, {})[arm] = dict(summ, splits={'exploratory': _load(sd / 'exploratory.json')})
    seeds = sorted(runs)
    table = {str(s): {a: {k: sealed[s][a]['splits']['exploratory']['head'][k] for k in ('macro_auroc', 'micro_auroc', 'mrr', 'top1')}
                      for a in ARMS} for s in seeds}
    deltas = {}
    for arm in (*CANDIDATES, 'G_frozen'):
        for ref in ('B0', 'shuffle'):
            r = all_seed_deltas({s: sealed[s][arm] for s in seeds}, {s: sealed[s][ref] for s in seeds}, 'exploratory',
                                expected_seeds=seeds, draws=DRAWS)
            deltas[f'{arm}_vs_{ref}'] = {'pooled_mean': r['all_seed_mean_delta'], 'pooled_ci95': r['hierarchical_ci95']}
    out = {'header': 'POST-SELECTION, NON-CONFIRMATORY exploratory report; never confirmation; selection was frozen first',
           'selection_sha256': selection_sha256, 'selected': doc['selected'], 'exploratory_head': table, 'deltas': deltas,
           'chronology_limit': 'ledger and mtimes are writable by the owner; not independent proof of chronology'}
    data = (json.dumps(out, indent=1, sort_keys=True) + '\n').encode(); digest = hashlib.sha256(data).hexdigest()
    fd, _ = _open_ledger(out_dir, flags=os.O_WRONLY | os.O_APPEND, expect=doc['ledger_identity'])
    try:
        p = create_new(out_dir, f'exploratory_report_{digest[:12]}.json', data)
        _append(fd, {'event': 'explore', 'selection_sha256': selection_sha256, 'report': str(p), 'sha256': digest,
                     'utc': datetime.now(timezone.utc).isoformat()})
    finally:
        os.close(fd)
    return p, digest, out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__); sub = ap.add_subparsers(dest='cmd', required=True)
    s = sub.add_parser('select'); s.add_argument('--run', action='append', required=True, help='SEED=/abs/run/dir')
    s.add_argument('--receipt', required=True); s.add_argument('--out-dir', required=True)
    s.add_argument('--seeds', default='42,43,44')
    s.add_argument('--bundle-sha256', required=True); s.add_argument('--protocol-sha256', required=True)
    e = sub.add_parser('explore'); e.add_argument('--selection', required=True); e.add_argument('--selection-sha256', required=True)
    e.add_argument('--out-dir', required=True)
    a = ap.parse_args(argv)
    if a.cmd == 'select':
        runs = {int(k): v for k, v in (x.split('=', 1) for x in a.run)}
        p, d, doc = select(runs, a.receipt, a.out_dir, expected_seeds=[int(x) for x in a.seeds.split(',')],
                           bundle_sha256=a.bundle_sha256, protocol_sha256=a.protocol_sha256)
        print(json.dumps({'selection': str(p), 'sha256': d, 'selected': doc['selected'], 'wording': doc['wording']}))
    else:
        p, d, _ = explore(a.selection, a.selection_sha256, a.out_dir)
        print(json.dumps({'report': str(p), 'sha256': d}))


if __name__ == '__main__':
    sys.exit(main())
