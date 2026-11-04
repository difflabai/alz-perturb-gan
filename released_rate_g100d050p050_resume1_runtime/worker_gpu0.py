"""One released scalar setting; complete live binding precedes scientific imports."""
import argparse,json,os,socket,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
import binding as b

def validate_reference(q,auth):
    value=b.cell(auth)
    b.require(q.get('schema')==1 and q.get('method')=='pg','reference checkpoint schema/method differs')
    b.require(q.get('fold')==value['fold'] and q.get('seed')==value['seed'],'reference checkpoint fold/seed differs')
    b.require(q.get('status')=='completed' and q.get('completed_updates')==q.get('fixed_endpoint')==20000,'complete fixed reference checkpoint required')
    b.require(q.get('cache_manifest_SHA256')==b.sha(Path(auth['cache_dir'])/'manifest.json'),'reference preprocessing/cache differs')

def validate_curve_checkpoint(q,auth,updates):
    b.require(type(updates) is int and updates in (5000,10000,20000),'predeclared checkpoint clock required')
    b.require(q.get('schema')==1 and q.get('method')=='pg' and q.get('native_recipe')=='bispf_optimized','native fixed-family checkpoint required')
    b.require(q.get('completed_updates')==q.get('policy',{}).get('completed_steps')==updates and q.get('fixed_endpoint')==20000,'native diagnostic clock differs')
    b.require(q.get('status')==('completed' if updates==20000 else 'diagnostic'),'intermediate score cannot stand in for complete20k endpoint')
    b.require(q.get('fold')==auth['cell']['fold'] and q.get('seed')==auth['cell']['seed'],'checkpoint fold/seed differs')
    b.require(q.get('cache_manifest_SHA256')==b.sha(Path(auth['cache_dir'])/'manifest.json'),'checkpoint cache differs')
    b.require(q.get('released_hyperparameters')==auth['released_hyperparameters'] and q.get('variant_config')==auth['variant_config'],'checkpoint hyperparameters or architecture differ')

def run_scientific(auth,admission,layout):
    b.require(auth['operation']=='resume_released_hyperparameters','one released scalar setting only')
    b.require(Path(sys.executable).absolute()==b.runtime_path(layout,b.cell(auth)).absolute() and sys.version_info[:2]==(3,12),'exact native runtime required')
    sys.path.extend(map(str,b.native_site_roots(b.cell(auth),auth['files_SHA256'])))
    sys.path[:0]=[str(b.HERE.parent/'analysis/released_tuning'),str(b.HERE.parent/'model'),str(layout.science),str(b.HERE.parent)]
    def deny(self,address):raise RuntimeError('Scientific worker network disabled')
    socket.socket.connect=deny;socket.socket.connect_ex=deny
    os.environ.update(WANDB_MODE='disabled',WANDB_DISABLED='true',OTEL_SDK_DISABLED='true',DO_NOT_TRACK='1',HF_HUB_OFFLINE='1',JAX_PLATFORMS='cpu',CUBLAS_WORKSPACE_CONFIG=':4096:8')
    import torch
    torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True);torch.set_float32_matmul_precision('highest')
    torch.cuda.set_per_process_memory_fraction((3*2**30)/torch.cuda.get_device_properties(0).total_memory,0)
    from calibration import evaluate,average,dump,sha
    from generation import iter_pg
    import released_update as update
    import released_adapter as model
    from state_fingerprint import fingerprint
    for module,name in ((update,'released_update.py'),(model,'released_adapter.py')):
        b.require(Path(module.__file__).resolve()==b.HERE.parent/'analysis/released_tuning'/name,'qualified model import origin differs')
    update.configure(auth['variant_config']);update.configure_hyperparameters(auth['released_hyperparameters'])
    out=Path(auth['output_dir']);out.mkdir(parents=True,exist_ok=False)
    end=time.monotonic()+max(0,min(admission['authorization_expiry'],admission['ledger_started_unix']+auth['cap_seconds'])-time.time()-120)
    try:
        ctx=model.Context(Path(auth['cache_dir']),'cuda:0')
        from resume_training import resume_fit
        bundle,final_path=resume_fit(ctx,out/'fit',auth['cell']['seed'],auth,end_monotonic=end)
        b.require(bundle[-1].completed_steps==20000,'full fixed endpoint required before any calibration')
        curve=[];reloads=[]
        # All diagnostic generation happens after fitting. Each native snapshot
        # has its own saved clock; none is selected as an early stopping point.
        for updates in (5000,10000,20000):
            path=final_path if updates==20000 else out/'fit'/('checkpoint%05d.pt'%updates)
            q=torch.load(path,map_location='cpu',weights_only=False);validate_curve_checkpoint(q,auth,updates)
            if updates==20000:validate_reference(q,auth)
            b.require(q['recipe']==q['policy']['recipe']==bundle[0].to_dict(),'native resolved recipe differs')
            saved=fingerprint(q['policy']);bundle[-1].load_state_dict(q['policy'])
            b.require(saved==fingerprint(bundle[-1].state_dict()),'full native public CUDA policy reload differs')
            reloads.append({'updates':updates,'checkpoint_path':str(path),'checkpoint_SHA256':sha(path),
                'policy_fingerprint':saved,'native_device_metadata_preserved':q['policy']['device'],
                'public_CUDA_reload_byte_exact':True})
            del q
            def iterator(schedule,c,s):
                for item in iter_pg(ctx,bundle[1],bundle[3],schedule,c,s):
                    if time.monotonic()>=end:raise TimeoutError('Finite allocation ending; no partial diagnostic or endpoint score')
                    yield item
            dest=out if updates==20000 else out/'diagnostics'/('u%05d'%updates)
            rows=[evaluate(ctx,'pg_released_params',auth['cell']['seed'],rep,iterator,dest/f'r{rep}') for rep in (0,1)]
            metrics=average(rows)
            entry={'completed_updates':updates,'checkpoint_path':str(path),'checkpoint_SHA256':sha(path),
                'diagnostic_only':updates!=20000,'eligible_for_selection':updates==20000,'metrics':metrics,
                'replicate_directories':[str(dest/f'r{rep}') for rep in (0,1)]}
            curve.append(entry)
        dump(out/'native_policy_reload_receipts.json',reloads)
        dump(out/'convergence_curve.json',{'status':'COMPLETE_PREDECLARED_5K_10K_20K_CALIBRATION_CURVE','candidate_id':auth['tuning_candidate_id'],
            'training_finished_before_calibration':True,'best_checkpoint_selection':False,'entries':curve})
        result=dict(curve[-1]['metrics'])
        result.update(status='completed',completed_updates=20000,fixed_endpoint=20000,integer_count_contract_passed=True,
            allocation_id=admission['allocation_id'],operation=auth['operation'],checkpoint_path=str(final_path),checkpoint_SHA256=sha(final_path),
            variant_config=auth['variant_config'],released_hyperparameters=auth['released_hyperparameters'],tuning_candidate_id=auth['tuning_candidate_id'],
            method='pg_released_params',hybrid_config=None,qualification_only=False,new_model_updates=17411,initial_completed_updates=2589,continuation=auth['continuation'],continuation_protocol_SHA256=auth['continuation_protocol_SHA256'],resumed_same_model=True,
            native_recipe_schedule_horizon=20000,parent_checkpoint_SHA256=auth['continuation']['checkpoint_SHA256'],automatic_successor=False,generation='direct_iter_pg',real_bag_replacements=0,
            pure_only=True,native_recipe='bispf_optimized',pure_formulation='released_hyperparameters',preset_protocol_SHA256=auth['preset_protocol_SHA256'],
            convergence_curve_path=str(out/'convergence_curve.json'),convergence_curve_SHA256=sha(out/'convergence_curve.json'),
            native_reload_path=str(out/'native_policy_reload_receipts.json'),native_reload_SHA256=sha(out/'native_policy_reload_receipts.json'),
            native_parent_restore_path=str(out/'fit/native_parent_restore.json'),native_parent_restore_SHA256=sha(out/'fit/native_parent_restore.json'),
            learning_curve_path=str(out/'fit/learning_curve.jsonl'),learning_curve_SHA256=sha(out/'fit/learning_curve.jsonl'))
        dump(out/'completed.json',result)
    except BaseException as exc:
        dump(out/'failure.json',{'status':'incomplete','error':repr(exc),'automatic_retry':False})
        raise

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--authorization',type=Path,required=True);parser.add_argument('--cpu-fixture-root',type=Path)
    args=parser.parse_args();layout=b.Layout.fixture(args.cpu_fixture_root) if args.cpu_fixture_root else b.NATIVE_LAYOUT
    auth,admission=b.require_live(args.authorization,layout)
    import project_runner_gpu0
    project_runner_gpu0.controller_gate(b.read(layout.project/'project_gpu_budget.json'),auth)
    if layout.fixture_mode:
        b.require('torch' not in sys.modules and os.environ.get('CUDA_VISIBLE_DEVICES')=='','CPU fixture exposed model or CUDA')
        if auth.get('fixture_fail_exit'):return 7
        out=Path(auth['output_dir']);out.mkdir(parents=True,exist_ok=False)
        (out/'CPU_FIXTURE_COMPLETE.json').write_text(json.dumps({'status':'CPU_BINDING_FIXTURE_ONLY','allocation_id':auth['allocation_id'],
            'attempt_id':auth['attempt_id'],'model_imported':False,'CUDA_VISIBLE_DEVICES':os.environ['CUDA_VISIBLE_DEVICES']})+'\n')
        return 0
    return run_scientific(auth,admission,layout)
if __name__=='__main__':raise SystemExit(main())
