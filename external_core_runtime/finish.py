"""Independent saved-state/count/score and clean-settlement audit of one cell."""
import argparse,datetime,json,os,sys
from pathlib import Path
HERE=Path(__file__).resolve().parent;sys.path.insert(0,str(HERE))
import binding as b,preset_scope as scope

def finish(attempt):
    assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
    folder=HERE.parent/'artifacts/allocations'/attempt
    dep=b.read(folder/'deployment.json');auth=b.read(dep['authorization_path']);value=auth['cell']
    assert b.sha(dep['authorization_path'])==dep['authorization_SHA256'] and attempt==b.attempt_key(value)
    scope.validate_authorization(auth)
    for path,pin in auth['files_SHA256'].items():assert b.sha(path)==pin,path
    job=b.read(auth['ledger_path'])['jobs'][0]
    settlement=b.read(Path(auth['ledger_path']).with_name(attempt+'-settlement.json'))
    assert job['status']=='completed' and job['exit_code']==0 and job['reservation_released']
    assert settlement['uncertain'] is None and settlement['settlement']['status']=='settled'
    for key in ('child_identity','guard_identity'):
        assert not Path('/proc',str(job[key]['pid'])).exists(),'Owned processes still present'
    registry=b.read(Path(auth['registry_path'])/'registry.json');assert job['lease_token'] not in registry['jobs']
    sys.path.extend(map(str,b.native_site_roots(value,auth['files_SHA256'])))
    sys.path[:0]=[str(scope.MODEL_DIR),str(HERE.parent/'model'),str(b.SCIENCE),str(HERE.parent)]
    import numpy as np,torch
    from scoring import score_summary_arrays
    from bench_core import cell_energy
    from state_fingerprint import fingerprint
    out=Path(auth['output_dir']);completed=b.read(out/'completed.json');native=b.read(out/'native_restore.json')
    assert completed['status']=='completed' and completed['metric_role']=='final_evaluation'
    assert (completed['fold'],completed['method'],completed['seed'])==(value['fold'],value['method'],value['seed'])
    assert completed['endpoint']==auth['fixed_endpoint'] and completed['benchmark_protocol_SHA256']==b.sha(scope.PROTOCOL)
    assert completed['new_model_updates']==auth['new_model_updates'] and completed['reuse_completed_model']==auth['reuse_completed_model']
    assert completed['checkpoint_SHA256']==b.sha(completed['checkpoint_path'])==native['checkpoint_SHA256']
    assert completed['native_restore_SHA256']==b.sha(out/'native_restore.json')
    if value['method']=='pg':
        from worker_gpu0 import validate_pg
        q=torch.load(completed['checkpoint_path'],map_location='cpu',weights_only=False);validate_pg(q,auth)
        assert fingerprint(q['policy'])==native['policy_fingerprint']
        for module in q['policy']['models'].values():
            for tensor in module.values():assert torch.isfinite(tensor).all()
    else:
        from native_scvi_reload import load_native
        model,receipt,state=load_native(auth['cache_dir'],Path(completed['checkpoint_path']).parent.parent,'cpu')
        assert state==native['model_fingerprint'] and receipt['completed_epochs']==400
        assert receipt['seed']==value['seed'] and receipt['fold']==value['fold']
    pins={};records={}
    for name in ('evaluation_r0','evaluation_r1','resampling_r0','resampling_r1','stage_swap_r0'):
        dest=out/name;m=b.read(dest/'metrics.json');rep=0 if name.endswith('r0') else 1
        assert m['status']=='completed' and m['fold']==value['fold'] and m['training_seed']==value['seed'] and m['generation_replicate']==rep
        assert m['diagnostic_only'] is name.startswith('stage_swap')
        assert m['query_stage_permutation']==([2,3,0,1] if name.startswith('stage_swap') else [0,1,2,3])
        assert m['generated_cells']==320000 and m['arrays_SHA256']==b.sha(dest/'arrays.npz')
        with np.load(dest/'arrays.npz',allow_pickle=False) as a:
            recomputed=score_summary_arrays(a['stage_synthetic_summaries'],a['real_summaries'],a['stage'])
            assert a['donors'].tolist()==m['donors']
            recomputed['cell_energy_distance']=cell_energy(a['generated_cell_reservoir'],a['real_cell_reservoir'])
        for key,v in recomputed.items():
            if isinstance(v,(int,float,list)):np.testing.assert_allclose(m[key],v,rtol=0,atol=1e-10)
        if not name.startswith('resampling'):
            for stage in range(4):
                p=dest/f'stage{stage}_count_exemplar.npz'
                with np.load(p,allow_pickle=False) as a:
                    c=a['counts'];assert c.shape==(1,400,2000) and np.isfinite(c).all() and (c>=0).all() and np.equal(c,np.floor(c)).all()
                pins[str(p)]=b.sha(p)
        pins[str(dest/'arrays.npz')]=b.sha(dest/'arrays.npz');records[name]=m
    assert not torch.cuda.is_initialized()
    result=dict(status='EXTERNAL_CORE_NATIVE_CHECKPOINT_COUNTS_SCORES_CLEAN_SETTLEMENT_AUDIT_PASS',UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                attempt=attempt,cell=value,benchmark_protocol_SHA256=b.sha(scope.PROTOCOL),completed=completed,
                summary={name:{key:m[key] for key in ('energy_score','cell_energy_distance','spread_ratio')} for name,m in records.items()},
                own_processes_absent=True,owned_registry_lease_absent=True,independent_CPU_saved_score_audit=True,native_saved_state_matches_GPU_restore=True,
                charged_GPU_seconds=settlement['settlement']['charged_GPU_seconds'],files_SHA256=pins,
                conditional_partial_result_only=True,full_matrix_required_before_decision=True,SOTA_claim=False,
                source_SHA256={str(Path(__file__)):b.sha(__file__)})
    path=HERE.parent/'analysis'/(attempt+'-external-audit.json');assert not path.exists();path.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:result[k] for k in ('status','cell','summary','charged_GPU_seconds')},indent=2))
if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--attempt',required=True);finish(parser.parse_args().attempt)
