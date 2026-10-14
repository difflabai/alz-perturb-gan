"""Pre-registered selection (inner only) and sealed exploratory reporting (plan v4 section 4).

select  : reads inner only, applies the rule, writes selection_frozen.json binding every input hash,
          and appends its sha to ledger.jsonl.
explore : refuses unless the selection file's sha is supplied and matches, a ledger entry exists,
          and every bound hash (receipts, protocol, bundle, rule, scoring, calibrators, sealed files)
          still matches the current artifacts. Output is post-selection and non-confirmatory.
Limits: the ledger and mtimes are writable and are an audit aid, not independent proof of chronology.
"""
from __future__ import annotations
import argparse, hashlib, json, os, sys, time
from datetime import datetime, timezone
from pathlib import Path
from .gene_evaluation import paired_delta, all_seed_deltas
from .gene_features import file_sha256, json_hash
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


def bindings(runs, receipt):
    """Every hash the selection depends on, recomputed from current artifacts."""
    receipt = scoped(receipt)
    out = {'supervisor_receipt': {'path': str(receipt), 'sha256': file_sha256(receipt)}, 'rule_sha256': RULE_SHA256,
           'scoring_mode': 'raw_logit_ranks', 'runs': {}}
    protocols, bundles = set(), set()
    for seed, run in sorted(runs.items()):
        grant = scoped(Path(scoped(run)) / 'grant.json') if (Path(run) / 'grant.json').exists() else None
        g = json.loads(Path(grant).read_text()) if grant else {}
        bundles.add(g.get('bundle_sha256'))
        arms = {}
        for arm in ARMS:
            d = _arm_dir(run, arm); sp = scoped(d / 'evaluation' / 'metrics' / 'summary.json'); s = json.loads(Path(sp).read_text())
            protocols.add(s['protocol_sha256'])
            if s.get('scoring', {}).get('mode') != 'raw_logit_ranks': raise ValueError(f'{arm} seed {seed}: scoring mode is not raw_logit_ranks')
            cal = s.get('calibration', {}).get('calibrator', {})
            arms[arm] = {'summary_sha256': file_sha256(sp), 'checkpoint_sha256': file_sha256(scoped(d / 'checkpoint.pt')),
                         'calibrator_sha256': cal.get('sha256'), 'calibration_split_sha256': cal.get('calibration_split_sha256'),
                         'sealed_sha256': s.get('sealed', {}).get('exploratory', {})}
        out['runs'][str(seed)] = {'path': str(scoped(run)), 'grant_sha256': file_sha256(grant) if grant else None, 'arms': arms}
    if len(protocols) != 1 or len(bundles) != 1: raise ValueError('runs disagree on protocol or bundle')
    out['protocol_sha256'], out['bundle_sha256'] = protocols.pop(), bundles.pop()
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


def select(runs, receipt, out_dir, *, expected_seeds):
    if sorted(runs) != sorted(expected_seeds): raise ValueError('all pre-declared seeds are required')
    rec = _load(receipt)
    complete = all(any(x.get('seed') == s and x['arm'] == a and x.get('status') == 'ok' for x in rec.get('arms', []))
                   for s in expected_seeds for a in ARMS) if rec.get('arms') and 'seed' in rec['arms'][0] else \
               all((Path(scoped(runs[s])) / a / 'evaluation' / 'metrics' / 'summary.json').exists() for s in expected_seeds for a in ARMS)
    summaries = {s: {a: _load(_arm_dir(runs[s], a) / 'evaluation' / 'metrics' / 'summary.json') for a in ARMS} for s in expected_seeds}
    gate = {s: _gate(runs[s]) for s in expected_seeds}
    pre = {'all_seeds_complete': bool(complete), 'gate_ok': all(g[0] for g in gate.values())}
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
    calib = {str(s): {a: summaries[s][a].get('calibration', {}).get('report', {}).get('status') for a in ARMS} for s in expected_seeds}
    doc = {'schema': 'selection_frozen_v4', 'rule': RULE, 'rule_sha256': RULE_SHA256, 'split': 'inner',
           'created_utc': datetime.now(timezone.utc).isoformat(), 'preconditions': pre,
           'gate': {str(s): g[1] for s, g in gate.items()}, 'inner_deltas': deltas, 'selected': selected,
           'calibration_status_reported_not_veto': calib, 'bindings': bindings(runs, receipt),
           'wording': 'selected on inner; not confirmed' if any(selected.values()) else 'selection criterion unmet; evidence inconclusive'}
    data = (json.dumps(doc, indent=1, sort_keys=True) + '\n').encode(); digest = hashlib.sha256(data).hexdigest()
    name = f'selection_frozen_{digest[:12]}.json'
    path = create_new(out_dir, name, data)
    _ledger(out_dir, {'event': 'select', 'selection': str(path), 'sha256': digest, 'utc': doc['created_utc']})
    return path, digest, doc


def _ledger(out_dir, entry):
    d = scoped(out_dir); dfd = os.open(d, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        fd = os.open('ledger.jsonl', os.O_WRONLY | os.O_APPEND | os.O_CREAT | os.O_NOFOLLOW, 0o644, dir_fd=dfd)
        try: os.write(fd, (json.dumps(entry, sort_keys=True) + '\n').encode())
        finally: os.close(fd)
    finally: os.close(dfd)


def explore(selection, selection_sha256, out_dir):
    path = scoped(selection)
    if not selection_sha256 or file_sha256(path) != selection_sha256:
        raise PermissionError('explore refused: selection file missing or its sha256 does not match')
    doc = json.loads(Path(path).read_text())
    ledger = scoped(Path(path).parent / 'ledger.jsonl')
    if not any(json.loads(l).get('sha256') == selection_sha256 and json.loads(l).get('event') == 'select'
               for l in Path(ledger).read_text().splitlines() if l.strip()):
        raise PermissionError('explore refused: no ledger entry for this selection')
    b = doc['bindings']; runs = {int(s): v['path'] for s, v in b['runs'].items()}
    now = bindings(runs, b['supervisor_receipt']['path'])
    if now != b: raise PermissionError('explore refused: bound artifacts changed since selection was frozen')
    sealed = {}
    for s, run in runs.items():
        for arm in ARMS:
            want = b['runs'][str(s)]['arms'][arm]['sealed_sha256']
            if not want: raise PermissionError(f'explore refused: seed {s} {arm} has no sealed exploratory outputs')
            sd = _arm_dir(run, arm) / 'evaluation' / 'sealed'
            for fname, h in want.items():
                if file_sha256(scoped(sd / fname)) != h: raise PermissionError(f'explore refused: sealed {fname} changed')
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
           'chronology_limit': 'ledger and mtimes are writable; not independent proof of chronology'}
    data = (json.dumps(out, indent=1, sort_keys=True) + '\n').encode(); digest = hashlib.sha256(data).hexdigest()
    p = create_new(out_dir, f'exploratory_report_{digest[:12]}.json', data)
    _ledger(out_dir, {'event': 'explore', 'selection_sha256': selection_sha256, 'report': str(p), 'sha256': digest,
                      'utc': datetime.now(timezone.utc).isoformat()})
    return p, digest, out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__); sub = ap.add_subparsers(dest='cmd', required=True)
    s = sub.add_parser('select'); s.add_argument('--run', action='append', required=True, help='SEED=/abs/run/dir')
    s.add_argument('--receipt', required=True); s.add_argument('--out-dir', required=True)
    s.add_argument('--seeds', default='42,43,44')
    e = sub.add_parser('explore'); e.add_argument('--selection', required=True); e.add_argument('--selection-sha256', required=True)
    e.add_argument('--out-dir', required=True)
    a = ap.parse_args(argv)
    if a.cmd == 'select':
        runs = {int(k): v for k, v in (x.split('=', 1) for x in a.run)}
        p, d, doc = select(runs, a.receipt, a.out_dir, expected_seeds=[int(x) for x in a.seeds.split(',')])
        print(json.dumps({'selection': str(p), 'sha256': d, 'selected': doc['selected'], 'wording': doc['wording']}))
    else:
        p, d, _ = explore(a.selection, a.selection_sha256, a.out_dir)
        print(json.dumps({'report': str(p), 'sha256': d}))


if __name__ == '__main__':
    sys.exit(main())
