"""Materialize the prespecified next fresh20k search after the full stage1 audit.

No activation, allocation, training or release changes. CPU qualification and
exact source/resource review remain separate, required execution steps.
"""
import argparse
import copy
import datetime
import hashlib
import json
import math
import subprocess
from pathlib import Path

ROOT = Path('/ml2/alz')
STAGE1 = ROOT / 'analysis/distribution097'
RULE = ROOT / 'analysis/distribution097_next_search_rule_20261106.json'
NEXT = ROOT / 'analysis/distribution097b'
RUNTIME = ROOT / 'distribution097b_runtime'

def read(path):
    return json.loads(Path(path).read_bytes())

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def write(path, value):
    path = Path(path)
    assert not path.exists()
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')

def derive(rows, historical):
    assert len(rows) == 3 and {r['candidate_id'] for r in rows} == {'d000', 'd001', 'd002'}
    eligible = []
    for row in rows:
        if not row['finite']:
            continue
        assert all(math.isfinite(row['metrics'][k]) for k in ('cell_energy_distance', 'energy_score', 'spread_ratio'))
        assert all(type(v) in (int, float) and math.isfinite(v) and v > 0 for v in row['hyperparameters'].values())
        if .7 <= row['metrics']['spread_ratio'] <= 1.3:
            eligible.append(row)
    assert eligible, 'No finite spread-qualified anchor; inspect outcomes rather than schedule blind adaptation'
    anchor = min(eligible, key=lambda r: (r['metrics']['cell_energy_distance'], r['metrics']['energy_score'], r['candidate_id']))
    base = anchor['hyperparameters']
    larger = [r['hyperparameters']['reg_kappa'] for r in rows + historical
              if r['finite'] and r['hyperparameters']['reg_kappa'] > base['reg_kappa']]
    assert larger, 'No larger measured kappa; revise and freeze this hypothesis before any fit'
    kappa_upper = min(1., min(larger))
    assert kappa_upper > base['reg_kappa']
    proposals = []
    for cid, multiplier in (('d003', .5), ('d004', 1.5)):
        h = dict(base)
        for key in ('generator_lr', 'critic_lr', 'particle_lr'):
            h[key] *= multiplier
        proposals.append(dict(id=cid, hyperparameters=h, reason=f'Joint native rate scale {multiplier}; all other anchor scalars fixed.'))
    h = dict(base)
    h['reg_kappa'] = math.sqrt(base['reg_kappa'] * kappa_upper)
    proposals.append(dict(id='d005', hyperparameters=h, reason=f'Geometric penalty-center midpoint between anchor {base["reg_kappa"]} and measured upper {kappa_upper}; other five scalars fixed.'))
    vectors = [p['hyperparameters'] for p in proposals]
    assert len({json.dumps(h, sort_keys=True) for h in vectors}) == 3
    assert not any(h == old['hyperparameters'] for h in vectors for old in rows + historical), 'Identical previously measured20k configuration; inspect rather than duplicate'
    return anchor, proposals

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage1-audit', type=Path, required=True)
    args = parser.parse_args()
    assert not subprocess.check_output(['git', 'status', '--porcelain'], cwd=ROOT, text=True)
    assert not NEXT.exists() and not RUNTIME.exists()
    audit = read(args.stage1_audit)
    assert audit['status'] == 'INDEPENDENT_NATIVE097_FRESH20K_STAGE1_COUNTS_SCORES_AND_SETTLEMENT_AUDIT_PASS'
    assert audit['complete_stage1'] and audit['all_new_models_from_update_zero'] and not audit['heldout_values_read']
    report = read(STAGE1 / 'stage1_report.json')
    assert report['all_new_fits_from_scratch'] and not report['full_goal_complete'] and not report['SOTA_claim']
    for row in audit['rows']:
        path = STAGE1 / 'results' / f'CAL_GPU0_V1_dist097_{row["candidate_id"]}_f0_pg_s0.json'
        assert sha(path) == row['result_SHA256']
    assert not read(ROOT / 'distribution097_runtime/plan.json')['enabled']
    assert not read(ROOT / 'artifacts/gpu1-distribution097-stage1-20261106/live_registry/registry.json')['jobs']
    old_path = ROOT / 'analysis/CAL_GPU0_V1_released_v095_bridge_f0_pg_s0-result.json'
    old = read(old_path)['candidate']
    historical = [dict(finite=True, hyperparameters=old['released_hyperparameters'])]
    anchor, proposals = derive(audit['rows'], historical)
    NEXT.mkdir()
    (NEXT / 'model').mkdir()
    (NEXT / 'proposals').mkdir()
    RUNTIME.mkdir()
    evidence = [args.stage1_audit.resolve(), STAGE1 / 'stage1_report.json', RULE, old_path, Path(__file__).resolve()]
    protocol = copy.deepcopy(read(STAGE1 / 'protocol.json'))
    protocol.update(UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                    stage='SECOND_DISTRIBUTION_SEARCH_FRESH_RATE_AND_PENALTY_REFINEMENT',
                    adaptive_rule_path=str(RULE), adaptive_rule_SHA256=sha(RULE),
                    anchor_candidate_id=anchor['candidate_id'], anchor_hyperparameters=anchor['hyperparameters'],
                    anchor_metrics=anchor['metrics'],
                    selection_evidence_SHA256={str(p): sha(p) for p in evidence})
    protocol.pop('selection_stage1')
    protocol['selection_stage2'] = dict(configurations=[dict(id=p['id'], parameters=p['hyperparameters'], reason=p['reason']) for p in proposals],
                                        selection_seed=0, every_arm_fresh20k=True, maximum_initial_fits=3,
                                        record_all_outcomes=True, intermediate_snapshots='Read-only5000/10000; fixed20k selection only, no warm starts.')
    protocol['historical_evidence'] += list(map(str, evidence))
    write(NEXT / 'protocol.json', protocol)
    write(NEXT / 'native_release.json', read(STAGE1 / 'native_release.json'))
    for proposal in proposals:
        write(NEXT / 'proposals' / (proposal['id'] + '.json'),
              dict(**proposal, native_version='0.9.7', endpoint=20000, fold=0, seed=0,
                   initial_completed_updates=0, source_checkpoint=None, from_scratch=True,
                   heldout_results_used=False, protocol_SHA256=sha(NEXT / 'protocol.json')))
    for source in (STAGE1 / 'model').glob('*.py'):
        (NEXT / 'model' / source.name).write_bytes(source.read_bytes())
    for source in (ROOT / 'distribution097_runtime').glob('*.py'):
        code = source.read_text().replace('distribution097', 'distribution097b').replace('DISTRIBUTION097', 'DISTRIBUTION097B')
        code = code.replace('stage1', 'stage2').replace('STAGE1', 'STAGE2')
        # Preserve a distinct invalid test key before renaming valid arm IDs.
        code = code.replace("('d003',0,'search')", "('d006',0,'search')")
        for old_id, new_id in (('d000', 'd003'), ('d001', 'd004'), ('d002', 'd005')):
            code = code.replace(old_id, new_id)
        code = code.replace('d00[0-2]', 'd00[3-5]')
        if source.name == 'resource_profile.py':
            code = code.replace("ROOT/'artifacts/gpu1-hpo20k-c014-20261106/live_registry/registry.json'", "ROOT/'artifacts/gpu1-distribution097-stage1-20261106/live_registry/registry.json'")
        if source.name == 'preset_scope.py':
            code += '\nSELECTION_EVIDENCE=' + repr(evidence) + '\n'
            # repr(Path) would need PosixPath imported; use plain Path calls.
            code = code.replace(repr(evidence), '[' + ','.join('Path(' + repr(str(p)) + ')' for p in evidence) + ']')
            code = code.replace("ROOT/'analysis/particlegan097_runtime_equivalence_20261106.json']", "ROOT/'analysis/particlegan097_runtime_equivalence_20261106.json',*SELECTION_EVIDENCE]")
            code = code.replace("p=read(PROTOCOL);assert p['status']", "validate_adaptation();p=read(PROTOCOL);assert p['status']")
            code += '''
def validate_adaptation():
 import importlib.util
 p=read(PROTOCOL)
 for path,pin in p['selection_evidence_SHA256'].items():assert sha(path)==pin
 proof=read(SELECTION_EVIDENCE[0]);assert proof['complete_stage1'] and proof['all_new_models_from_update_zero'] and not proof['heldout_values_read']
 spec=importlib.util.spec_from_file_location('pinned_fresh097_adaptation',ROOT/'analysis/prepare_distribution097b.py')
 materializer=importlib.util.module_from_spec(spec);spec.loader.exec_module(materializer)
 historical=read(ROOT/'analysis/CAL_GPU0_V1_released_v095_bridge_f0_pg_s0-result.json')['candidate']
 anchor,proposals=materializer.derive(proof['rows'],[dict(finite=True,hyperparameters=historical['released_hyperparameters'])])
 assert p['anchor_candidate_id']==anchor['candidate_id'] and p['anchor_hyperparameters']==anchor['hyperparameters'] and p['anchor_metrics']==anchor['metrics']
 assert p['selection_stage2']['configurations']==[dict(id=q['id'],parameters=q['hyperparameters'],reason=q['reason']) for q in proposals]
'''
        if source.name == 'qualify.py':
            code = code.replace('paths=[*HERE.glob', 'paths=[*scope.SELECTION_EVIDENCE,*HERE.glob')
        compile(code, str(RUNTIME / source.name), 'exec')
        (RUNTIME / source.name).write_text(code)
    policy = read(ROOT / 'distribution097_runtime/policy.prepared.json')
    policy['registry_path'] = str(ROOT / 'artifacts/gpu1-distribution097b-stage2-20261106/live_registry')
    policy['resource_basis']['foreign_control'] = 'NONE. Fresh native0.9.7 full20k rate/penalty refinement; same own2GiBGPU/6GiBhost/3GiBdriver, foreign-growth forecasts, floors and shared physical locks. No foreign signals, settings or formulation changes.'
    write(RUNTIME / 'policy.prepared.json', policy)
    write(RUNTIME / 'plan.json', dict(enabled=False))
    driver = (ROOT / 'analysis/run_distribution097.py').read_text()
    driver = driver.replace('distribution097', 'distribution097b').replace('DISTRIBUTION097', 'DISTRIBUTION097B').replace('stage1', 'stage2').replace('STAGE1', 'STAGE2')
    for old_id, new_id in (('d000', 'd003'), ('d001', 'd004'), ('d002', 'd005')):
        driver = driver.replace(old_id, new_id)
    compile(driver, str(ROOT / 'analysis/run_distribution097b.py'), 'exec')
    (ROOT / 'analysis/run_distribution097b.py').write_text(driver)
    write(NEXT / 'materialization.json', dict(status='PREPARED_DISABLED_REQUIRES_NEW_CPU_QUALIFICATION_AND_EXACT_SOURCE_REVIEW',
          UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(), anchor=anchor['candidate_id'],
          proposals=proposals, source_SHA256=sha(__file__), evidence_SHA256={str(p): sha(p) for p in evidence},
          activation=False, GPU_allocation_created=False, new_model_updates=0))
    print(json.dumps(dict(status='PREPARED_DISABLED', anchor=anchor['candidate_id'], proposals=proposals)))

if __name__ == '__main__':
    main()
