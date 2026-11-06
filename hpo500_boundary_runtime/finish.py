"""Independent checkpoint, native integer-count, score and settlement audit."""
import argparse,datetime,json,os,sys
from pathlib import Path
HERE=Path(__file__).resolve().parent;sys.path.insert(0,str(HERE))
import binding as b,preset_scope as scope
def finish(attempt):
    assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
    folder=HERE.parent/'artifacts/allocations'/attempt;dep=b.read(folder/'deployment.json');auth=b.read(dep['authorization_path'])
    assert b.sha(dep['authorization_path'])==dep['authorization_SHA256'] and attempt==b.attempt_key(auth['cell'])
    scope.validate_authorization(auth)
    for path,pin in auth['files_SHA256'].items():assert b.sha(path)==pin,path
    job=b.read(auth['ledger_path'])['jobs'][0];settlement=b.read(Path(auth['ledger_path']).with_name(attempt+'-settlement.json'))
    assert job['reservation_released'] and settlement['uncertain'] is None and settlement['settlement']['status']=='settled'
    for key in ('child_identity','guard_identity'):assert not Path('/proc',str(job[key]['pid'])).exists()
    assert job['lease_token'] not in b.read(Path(auth['registry_path'])/'registry.json')['jobs']
    logical_attempt=attempt
    out=Path(auth['output_dir'])
    base=dict(UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),attempt=logical_attempt,physical_attempt=attempt,resumed_same_trial=False,candidate_id=auth['candidate_id'],phase=auth['phase'],seed=auth['cell']['seed'],hyperparameters=auth['released_hyperparameters'],protocol_SHA256=b.sha(scope.PROTOCOL),proposal_SHA256=auth['proposal_SHA256'],owned_processes_and_lease_absent=True,clean_settlement=True,charged_GPU_seconds=settlement['settlement']['charged_GPU_seconds'],phase_charged_GPU_seconds=settlement['settlement']['charged_GPU_seconds'],calibration_only=True,endpoint=500,automatic_promotion=False)
    if job['status']!='completed' or job['exit_code']!=0:
        failure=b.read(out/'failure.json') if (out/'failure.json').exists() else {}
        assert failure.get('error','').startswith('FloatingPointError('),'Operational failure; study stops without retry'
        base.update(status='HPO500_SCIENTIFIC_NONFINITE_FAILURE_CLEAN_AUDIT',finite=False,failure=failure,metrics=None,failed_candidate_not_retried=True)
    else:
        sys.path.extend(map(str,b.native_site_roots(auth['cell'],auth['files_SHA256'])))
        sys.path[:0]=[str(scope.MODEL_DIR),str(HERE.parent/'model'),str(b.SCIENCE),str(HERE.parent)]
        import numpy as np,torch
        from calibration import score,average
        from state_fingerprint import fingerprint
        import released_adapter as model,released_update as update
        from particlegan.policy import _validate_optimizer_state
        from worker_gpu0 import validate_pg
        update.configure(scope.CFG);update.configure_hyperparameters(auth['released_hyperparameters'])
        c=b.read(out/'completed.json');assert c['from_scratch'] and c['evaluation_targets_read'] is False
        assert c['candidate_id']==auth['candidate_id'] and c['seed']==auth['cell']['seed'] and c['protocol_SHA256']==b.sha(scope.PROTOCOL)
        assert c['endpoint']==500 and c['new_model_updates']==auth['new_model_updates'] and c['initial_completed_updates']==auth['initial_completed_updates']
        checkpoint=Path(c['checkpoint_path']);assert b.sha(checkpoint)==c['checkpoint_SHA256']
        q=torch.load(checkpoint,map_location='cpu',weights_only=False);validate_pg(q,auth)
        assert q['recipe'].get('generator_gradient_response','none')=='none' and q['policy']['device']=='cuda:0'
        assert q['usage']['steps']==500 and int(q['usage']['draws'].sum())==16000 and int(q['usage']['unsampled_moved_steps'].sum())==0
        assert q['training_streams']==model.training_streams(0,auth['cell']['seed'])
        assert fingerprint(q['policy'])==c['policy_fingerprint']
        bundle,*_=model.build(torch.ones(2000),torch.ones(2000),'cpu',0,auth['cell']['seed'])
        assert q['recipe']==bundle[0].to_dict()
        for name,module in zip(('generator','critic','prior'),bundle[1:4]):
            module.load_state_dict(q['policy']['models'][name],strict=True)
            assert all(torch.isfinite(t).all() for t in module.state_dict().values())
        for opt,saved in zip(bundle[4:6],q['policy']['optimizers']):
            _validate_optimizer_state(opt,saved);opt.load_state_dict(saved);assert fingerprint(opt.state_dict())==fingerprint(saved)
        records=[];pins={str(checkpoint):b.sha(checkpoint)}
        for rep in (0,1):
            dest=out/f'r{rep}';m=b.read(dest/'metrics.json')
            assert m['status']=='completed' and m['metric_role']=='calibration_selection' and m['training_seed']==auth['cell']['seed']
            assert m['generation_replicate']==rep and m['generated_cells']==320000 and m['arrays_SHA256']==b.sha(dest/'arrays.npz')
            with np.load(dest/'arrays.npz',allow_pickle=False) as a:recomputed=score(a['stage_synthetic_summaries'],a['generated_cells_by_stage'])
            for key,value in recomputed.items():
                if isinstance(value,(int,float,list)) and key!='donors':np.testing.assert_allclose(m[key],value,rtol=0,atol=1e-10)
            for stage in range(4):
                p=dest/f'stage{stage}_exemplar.npz'
                with np.load(p,allow_pickle=False) as a:
                    counts=a['counts'];assert counts.shape==(1,400,2000) and np.isfinite(counts).all() and (counts>=0).all() and np.equal(counts,np.floor(counts)).all()
                pins[str(p)]=b.sha(p)
            pins[str(dest/'arrays.npz')]=b.sha(dest/'arrays.npz');records.append(m)
            assert m==c['replicate_metrics'][rep]
        assert average(records)==c['metrics'] and not torch.cuda.is_initialized()
        base.update(status='HPO500_NATIVE_COUNTS_SCORES_CLEAN_AUDIT_PASS',finite=True,metrics=c['metrics'],replicate_metrics=records,files_SHA256=pins,from_scratch=True,physical_start_update=auth['initial_completed_updates'],new_phase_model_updates=auth['new_model_updates'],GPU_peak_allocated_bytes=c['GPU_peak_allocated_bytes'],GPU_peak_reserved_bytes=c['GPU_peak_reserved_bytes'],GPU_allocator_limit_bytes=c['GPU_allocator_limit_bytes'],independent_saved_score_audit=True,native_saved_state_matches_GPU_restore=True)
    path=scope.STUDY/'results'/(logical_attempt+'.json');assert not path.exists();path.parent.mkdir(exist_ok=True);path.write_text(json.dumps(base,indent=2)+'\n')
    print(json.dumps({k:v for k,v in base.items() if k not in ('files_SHA256','replicate_metrics')},indent=2))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--attempt',required=True);finish(p.parse_args().attempt)
