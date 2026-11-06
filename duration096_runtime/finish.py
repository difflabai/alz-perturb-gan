"""Retain and independently audit both matched endpoints regardless of quality."""
import argparse,datetime,json,os,sys
from pathlib import Path
HERE=Path(__file__).resolve().parent;sys.path.insert(0,str(HERE))
import binding as b,preset_scope as scope
def finish(attempt):
    assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
    folder=HERE.parent/'artifacts/allocations'/attempt;dep=b.read(folder/'deployment.json');auth=b.read(dep['authorization_path'])
    assert b.sha(dep['authorization_path'])==dep['authorization_SHA256'] and attempt==b.attempt_key(scope.cell(0,'pg',0))
    scope.validate_authorization(auth)
    for path,pin in auth['files_SHA256'].items():assert b.sha(path)==pin,path
    job=b.read(auth['ledger_path'])['jobs'][0];settlement=b.read(Path(auth['ledger_path']).with_name(attempt+'-settlement.json'))
    assert job['status']=='completed' and job['exit_code']==0 and job['reservation_released']
    assert settlement['uncertain'] is None and settlement['settlement']['status']=='settled'
    for key in ('child_identity','guard_identity'):assert not Path('/proc',str(job[key]['pid'])).exists()
    assert job['lease_token'] not in b.read(Path(auth['registry_path'])/'registry.json')['jobs']
    sys.path.extend(map(str,b.native_site_roots(auth['cell'],auth['files_SHA256'])))
    sys.path[:0]=[str(scope.EXT_DIR),str(scope.MODEL_DIR),str(HERE.parent/'model'),str(b.SCIENCE),str(HERE.parent)]
    import numpy as np,torch
    from calibration import score,average
    from state_fingerprint import fingerprint
    import released_adapter as model,released_update as update
    from particlegan.policy import _validate_optimizer_state
    update.configure(scope.CFG);update.configure_hyperparameters(scope.HYPER)
    out=Path(auth['output_dir']);completed=b.read(out/'completed.json');pins={}
    assert completed['status']=='completed' and completed['metric_role']=='calibration_selection'
    assert completed['protocol_SHA256']==b.sha(scope.PROTOCOL) and completed['training_finished_before_calibration']
    assert completed['evaluation_targets_read'] is False and completed['checkpoint_selection'] is False
    assert [r['updates'] for r in completed['rows']]==[20000,30000,40000]
    assert b.sha(out/'fit/checkpoint20000.pt')==auth['source_model_SHA256']
    restore=b.read(out/'fit/native_parent_restore.json')
    assert restore['whole_parent_scientific_fingerprint']==auth['extension']['whole_parent_scientific_fingerprint']
    assert not restore['native_recipe_rewritten'] and not restore['native_or_usage_or_data_clock_reset']
    bundle,*_=model.build(torch.ones(2000),torch.ones(2000),'cpu',0,0)
    for row in completed['rows']:
        step=row['updates'];path=Path(row['checkpoint_path']);assert b.sha(path)==row['checkpoint_SHA256']
        q=torch.load(path,map_location='cpu',weights_only=False)
        assert q['completed_updates']==q['policy']['completed_steps']==q['usage']['steps']==step
        assert q['particlegan_version']=='0.9.6' and q['recipe']['ema_decay']==0
        assert q['recipe']==q['policy']['recipe']==bundle[0].to_dict() and q['policy']['device']=='cuda:0'
        assert q['released_hyperparameters']==scope.HYPER and q['variant_config']==scope.CFG
        assert q['training_streams']==model.training_streams(0,0) and q['cache_manifest_SHA256']==b.sha(Path(auth['cache_dir'])/'manifest.json')
        assert int(q['usage']['draws'].sum())==32*step and int(q['usage']['unsampled_moved_steps'].sum())==0
        assert q['fixed_endpoint']==(20000 if step==20000 else 40000)
        assert q['status']==('diagnostic' if step==30000 else 'completed')
        assert fingerprint(q['policy'])==row['policy_fingerprint']
        for name,module in zip(('generator','critic','prior'),bundle[1:4]):
            module.load_state_dict(q['policy']['models'][name],strict=True)
            assert all(torch.isfinite(t).all() for t in module.state_dict().values())
        for opt,saved in zip(bundle[4:6],q['policy']['optimizers']):
            _validate_optimizer_state(opt,saved);opt.load_state_dict(saved);assert fingerprint(opt.state_dict())==fingerprint(saved)
        records=[]
        for rep in (0,1):
            dest=out/('u%05d'%step)/('r%d'%rep);m=b.read(dest/'metrics.json')
            assert m['status']=='completed' and m['metric_role']=='calibration_selection' and m['training_seed']==0
            assert m['generation_replicate']==rep and m['generated_cells']==320000 and m['arrays_SHA256']==b.sha(dest/'arrays.npz')
            with np.load(dest/'arrays.npz',allow_pickle=False) as a:recomputed=score(a['stage_synthetic_summaries'],a['generated_cells_by_stage'])
            for key,value in recomputed.items():
                if isinstance(value,(int,float,list)) and key!='donors':np.testing.assert_allclose(m[key],value,rtol=0,atol=1e-10)
            for stage in range(4):
                p=dest/f'stage{stage}_exemplar.npz'
                with np.load(p,allow_pickle=False) as a:
                    c=a['counts'];assert c.shape==(1,400,2000) and np.isfinite(c).all() and (c>=0).all() and np.equal(c,np.floor(c)).all()
                pins[str(p)]=b.sha(p)
            pins[str(dest/'arrays.npz')]=b.sha(dest/'arrays.npz');records.append(m)
            assert m==row['replicate_metrics'][rep]
        assert average(records)==row['metrics'];pins[str(path)]=b.sha(path)
    assert not torch.cuda.is_initialized()
    metrics={str(r['updates']):r['metrics'] for r in completed['rows']}
    delta={key:metrics['40000'][key]-metrics['20000'][key] for key in ('energy_score','cell_energy_distance','spread_ratio')}
    result=dict(status='MATCHED_NATIVE096_DURATION_INDEPENDENT_AUDIT_PASS',UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),attempt=attempt,protocol_SHA256=b.sha(scope.PROTOCOL),metrics=metrics,difference40k_minus20k=delta,joint_improvement=delta['energy_score']<0 and delta['cell_energy_distance']<=0,owned_processes_and_lease_absent=True,clean_settlement=True,charged_GPU_seconds=settlement['settlement']['charged_GPU_seconds'],same_parent_all_native_scientific_and_random_state_verified=True,calibration_only=True,training_seeds=1,automatic_promotion=False,frozen_external_benchmark_changed=False,files_SHA256=pins)
    path=HERE.parent/'analysis'/(attempt+'-duration-audit.json');assert not path.exists();path.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='files_SHA256'},indent=2))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--attempt',required=True);finish(p.parse_args().attempt)
