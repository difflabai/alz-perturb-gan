"""Audited released-API sweep readout and predeclared provisional anchor rule."""
import argparse,datetime,hashlib,json,math
from pathlib import Path
ROOT=Path('/ml2/alz')
PROTOCOL=ROOT/'analysis/released_tuning/protocol.json'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
read=lambda p:json.loads(Path(p).read_bytes())

def choose_anchor(rows,incumbent):
    """Choose a tuning anchor; never return an incumbent promotion decision."""
    valid=[]
    for row in rows:
        if row.get('state')!='audited_complete':continue
        es,ed,spread=(row[k] for k in ('energy_score','cell_energy_distance','spread_ratio'))
        if not all(type(x) in (int,float) and math.isfinite(x) for x in (es,ed,spread)):
            raise ValueError('Audited complete record has invalid metric')
        if .7<=spread<=1.3:valid.append(row)
    nondominated=[r for r in valid if not any(
        s['energy_score']<=r['energy_score'] and s['cell_energy_distance']<=r['cell_energy_distance']
        and (s['energy_score']<r['energy_score'] or s['cell_energy_distance']<r['cell_energy_distance'])
        for s in valid)]
    def key(r):
        ratios=(r['energy_score']/incumbent['energy_score'],r['cell_energy_distance']/incumbent['cell_energy_distance'])
        return max(ratios),sum(ratios),r['id']
    return min(nondominated,key=key) if nondominated else None

def summarize(destination):
    protocol=read(PROTOCOL);incumbent=read(ROOT/'pure_incumbent.json')
    assert sha(ROOT/'pure_incumbent.json')==protocol['frozen_science']['pure_incumbent_SHA256']
    settings={s['id']:s['parameters'] for s in protocol['stage1']['configs']}
    audited={};inflight={}
    for path in (ROOT/'analysis').glob('CAL_GPU0_V1_*-result.json'):
        result=read(path);candidate=result.get('candidate',{});identity=candidate.get('tuning_candidate_id')
        if identity not in settings or candidate.get('training_seed')!=0:continue
        assert identity not in audited,'Duplicate completed scientific configuration requires explicit lineage audit'
        assert result['owned_release_and_clean_settlement_verified'] is True
        assert result['independent_saved_scores_checkpoint_and_count_exemplars_verified'] is True
        assert candidate['released_hyperparameters']==settings[identity]
        assert candidate['native_recipe']=='bispf_optimized' and candidate['pure_formulation']=='released_hyperparameters'
        assert candidate['status']=='completed' and candidate['completed_updates']==candidate['fixed_endpoint']==20000
        assert candidate['integer_count_contract_passed'] and candidate['real_bag_replacements']==0
        assert candidate['training_seed']==0 and candidate['generation_replicates']==2
        assert candidate['calibration_manifest_SHA256']==protocol['frozen_science']['calibration_manifest_SHA256']
        assert candidate['trace_manifest_SHA256']==protocol['frozen_science']['trace_manifest_SHA256']
        assert sha(candidate['checkpoint_path'])==candidate['checkpoint_SHA256']
        assert candidate['preset_protocol_SHA256']==sha(PROTOCOL)
        audited[identity]={'id':identity,'parameters':settings[identity],'state':'audited_complete',
            **{k:candidate[k] for k in ('energy_score','cell_energy_distance','spread_ratio')},
            'endpoint_decision':result['decision'],'result_path':str(path),'result_SHA256':sha(path),
            'charged_GPU_seconds':result.get('charged_GPU_seconds_total_scientific_trial',result['charged_GPU_seconds']),
            'convergence_curve_path':candidate['convergence_curve_path'],'convergence_curve_SHA256':candidate['convergence_curve_SHA256']}
    # This inventory is explicitly output presence, not proof of a live job.
    # Authoritative process/ledger checks are required before saying "running".
    for path in (ROOT/'artifacts/allocations').glob('*/authorization.json'):
        auth=read(path);identity=auth.get('tuning_candidate_id')
        if identity not in settings or identity in audited or auth['cell']['seed']!=0:continue
        assert auth['released_hyperparameters']==settings[identity]
        out=Path(auth['output_dir']);progress=out/'fit/progress.json';failure=out/'failure.json'
        entry={'id':identity,'parameters':settings[identity],'state':'allocated_quality_not_audited',
            'authorization_path':str(path),'attempt':auth['attempt_id'],'issued_unix':auth['issued_unix'],
            'initial_completed_updates':auth['initial_completed_updates'],'process_liveness_checked':False}
        if progress.exists():entry['latest_saved_progress_updates']=read(progress)['step']
        receipt=out/'fit/checkpoint_receipt.json'
        if receipt.exists():entry['latest_checkpoint_complete_updates']=read(receipt)['completed_updates']
        if failure.exists():entry['state']='incomplete_operational_failure';entry['failure_path']=str(failure)
        prior=inflight.get(identity)
        if prior is None or (entry['issued_unix'],entry['attempt'])>(prior['issued_unix'],prior['attempt']):inflight[identity]=entry
    rows=[audited.get(s['id'],inflight.get(s['id'],dict(s,state='not_started'))) for s in protocol['stage1']['configs']]
    all_complete=len(audited)==len(settings)
    anchor=choose_anchor(rows,incumbent) if all_complete else None
    result={'status':'RELEASED_RATE_GRID_READOUT','utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'source_SHA256':{str(Path(__file__).resolve()):sha(__file__),str(PROTOCOL):sha(PROTOCOL)},
        'declared_configurations':len(settings),'audited_complete_configurations':len(audited),'rows':rows,
        'stage1_complete':all_complete,'stage2_anchor':anchor,'anchor_eligible_only_after_all27_audited':True,
        'incumbent_promoted':False,'stable_joint_ATH_claimed':False,
        'scope':'Fixed20k seed0 released-rate grid. Complete independently audited endpoints only; all27 retained, operational failures stay unscored. Pareto/minimax anchor is provisional for regularization, never an incumbent promotion. Four adaptive calibration donors, evaluation untouched.'}
    Path(destination).write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    return result

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    result=summarize(args.output)
    print(json.dumps({k:v for k,v in result.items() if k not in ('rows','source_SHA256')},indent=2))
