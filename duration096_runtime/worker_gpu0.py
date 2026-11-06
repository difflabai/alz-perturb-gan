"""One fixed benchmark cell: train/reuse, native restore, evaluate and resample."""
import argparse,json,os,socket,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
import binding as b
import preset_scope as scope

def validate_pg(q,auth):
    v=auth['cell']
    b.require(q['schema']==1 and q['method']=='pg' and q['native_recipe']=='bispf_optimized','Native PG checkpoint required')
    b.require(q['fold']==v['fold'] and q['seed']==v['seed'],'Checkpoint fold/seed differs')
    b.require(q['status']=='completed' and q['completed_updates']==q['fixed_endpoint']==q['policy']['completed_steps']==20000,'Fixed20k final checkpoint required')
    b.require(q['cache_manifest_SHA256']==b.sha(Path(auth['cache_dir'])/'manifest.json'),'Checkpoint optimization cache differs')
    b.require(q['particlegan_version']=='0.9.6' and q['penalty_options']=={'input_domain':'continuous'},'Native release/domain differs')
    b.require(q['released_hyperparameters']==scope.HYPER and q['variant_config']==scope.CFG,'Frozen configuration differs')

def scientific(auth,admission,layout):
    value=b.cell(auth);seed=value['seed']
    sys.path.extend(map(str,b.native_site_roots(value,auth['files_SHA256'])))
    sys.path[:0]=[str(scope.EXT_DIR),str(scope.MODEL_DIR),str(b.HERE.parent/'model'),str(layout.science),str(b.HERE.parent)]
    def deny(self,address):raise RuntimeError('Scientific worker network disabled')
    socket.socket.connect=deny;socket.socket.connect_ex=deny
    os.environ.update(HF_HUB_OFFLINE='1',WANDB_MODE='disabled',WANDB_DISABLED='true',OTEL_SDK_DISABLED='true',DO_NOT_TRACK='1',CUBLAS_WORKSPACE_CONFIG=':4096:8')
    import torch
    torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True);torch.set_float32_matmul_precision('highest')
    torch.cuda.set_per_process_memory_fraction(3*2**30/torch.cuda.get_device_properties(0).total_memory,0)
    from calibration import evaluate,average,dump,sha
    from generation import iter_pg
    from state_fingerprint import fingerprint
    import released_update as update,released_adapter as model,extension
    assert Path(extension.__file__).resolve()==scope.EXT_DIR/'extension.py'
    update.configure(scope.CFG);update.configure_hyperparameters(scope.HYPER)
    ctx=model.Context(Path(auth['cache_dir']),'cuda:0')
    out=Path(auth['output_dir']);out.mkdir(parents=True,exist_ok=False)
    end=time.monotonic()+max(0,min(admission['authorization_expiry'],admission['ledger_started_unix']+auth['cap_seconds'])-time.time()-120)
    try:
        bundle,final=extension.fit(ctx,out/'fit',seed,auth,end_monotonic=end)
        assert bundle[-1].completed_steps==40000
        rows=[]
        for step in (20000,30000,40000):
            checkpoint=final if step==40000 else out/'fit'/('checkpoint%05d.pt'%step)
            q=torch.load(checkpoint,map_location='cpu',weights_only=False)
            if step==20000:validate_pg(q,auth)
            else:
                assert q['completed_updates']==q['policy']['completed_steps']==q['usage']['steps']==step
                assert q['fixed_endpoint']==40000 and q['particlegan_version']=='0.9.6'
                assert q['status']==('completed' if step==40000 else 'diagnostic')
            assert q['released_hyperparameters']==scope.HYPER and q['recipe']==bundle[0].to_dict()==q['policy']['recipe']
            assert q['recipe']['ema_decay']==0 and q['recipe']['total_steps']==20000 and q['policy']['device']=='cuda:0'
            bundle[-1].load_state_dict(q['policy']);saved=fingerprint(q['policy'])
            assert fingerprint(bundle[-1].state_dict())==saved
            def iterator(schedule,c,s):
                for item in iter_pg(ctx,bundle[1],bundle[3],schedule,c,s):
                    if time.monotonic()>=end:raise TimeoutError('Duration phase deadline; no partial score')
                    yield item
            records=[evaluate(ctx,'pg_duration096',seed,rep,iterator,out/('u%05d'%step)/('r%d'%rep)) for rep in (0,1)]
            rows.append(dict(updates=step,checkpoint_path=str(checkpoint),checkpoint_SHA256=sha(checkpoint),policy_fingerprint=saved,metrics=average(records),replicate_metrics=records,diagnostic_only=step==30000))
        dump(out/'completed.json',dict(status='completed',metric_role='calibration_selection',endpoint=40000,new_model_updates=20000,initial_completed_updates=20000,protocol_SHA256=sha(scope.PROTOCOL),rows=rows,training_finished_before_calibration=True,checkpoint_selection=False,EMA_decay=0,ParticleGAN_version='0.9.6',evaluation_targets_read=False))
    except BaseException as exc:
        dump(out/'failure.json',dict(status='incomplete',error=repr(exc),automatic_retry=False));raise

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--authorization',type=Path,required=True);parser.add_argument('--cpu-fixture-root',type=Path)
    args=parser.parse_args();layout=b.Layout.fixture(args.cpu_fixture_root) if args.cpu_fixture_root else b.NATIVE_LAYOUT
    auth,admission=b.require_live(args.authorization,layout)
    import project_runner_gpu0
    project_runner_gpu0.controller_gate(b.read(layout.project/'project_gpu_budget.json'),auth)
    if layout.fixture_mode:
        b.require('torch' not in sys.modules and os.environ.get('CUDA_VISIBLE_DEVICES')=='','Fixture exposed CUDA/model')
        if auth.get('fixture_fail_exit'):return 7
        out=Path(auth['output_dir']);out.mkdir(parents=True,exist_ok=False)
        (out/'CPU_FIXTURE_COMPLETE.json').write_text(json.dumps(dict(status='CPU_BINDING_FIXTURE_ONLY',allocation_id=auth['allocation_id'],attempt_id=auth['attempt_id'],model_imported=False,CUDA_VISIBLE_DEVICES=''))+'\n')
        return 0
    return scientific(auth,admission,layout)
if __name__=='__main__':raise SystemExit(main())
