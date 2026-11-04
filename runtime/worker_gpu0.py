"""New GPU0 entrypoint; live complete binding precedes native package imports."""
import argparse
import json
import os
from pathlib import Path
import socket
import sys
import time
from types import SimpleNamespace

sys.path.insert(0,str(Path(__file__).resolve().parent))
import binding as b

def validate_reference(checkpoint,auth):
    value=b.cell(auth)
    b.require(checkpoint.get('schema')==1 and checkpoint.get('method')=='pg','reference checkpoint schema/method differs')
    b.require(checkpoint.get('fold')==value['fold'] and checkpoint.get('seed')==value['seed'],'reference checkpoint fold/seed differs')
    b.require(checkpoint.get('status')=='completed' and checkpoint.get('completed_updates')==checkpoint.get('fixed_endpoint')==(auth['fixed_endpoint'] if auth.get('operation')=='evaluate_pure_variant' else 20000),'complete fixed reference checkpoint required')
    b.require(checkpoint.get('cache_manifest_SHA256')==b.sha(Path(auth['cache_dir'])/'manifest.json'),'reference preprocessing/cache differs')

def run_scientific(auth,admission,layout):
    b.require(auth['operation'] in ('evaluate_reference','evaluate_hybrid','evaluate_pure_variant','qualify_pure_ensemble','evaluate_pure_ensemble','confirm_pure_ensemble','qualify_interrupted_pg8','resume_pure_ensemble','confirm_frozen_ensemble_seed2','train_variant','train_bcap','train_bispf','train_semantic','train_learned','train_network_rate','train_hybrid','qualify_continuation','continue_hybrid','train_extended_hybrid'),'registered calibration operation required')
    runtime=b.runtime_path(layout,b.cell(auth))
    b.require(Path(sys.executable).absolute()==runtime.absolute() and sys.version_info[:2]==(3,12),'exact PG scientific interpreter required')
    sys.path.extend(map(str,b.native_site_roots(b.cell(auth),auth['files_SHA256'])))
    sys.path.insert(0,str(layout.science));sys.path.insert(0,str(b.HERE.parent))
    def deny(self,address):raise RuntimeError('calibration worker network disabled')
    socket.socket.connect=deny;socket.socket.connect_ex=deny
    os.environ.update(WANDB_MODE='disabled',WANDB_DISABLED='true',OTEL_SDK_DISABLED='true',DO_NOT_TRACK='1',HF_HUB_OFFLINE='1',JAX_PLATFORMS='cpu',CUBLAS_WORKSPACE_CONFIG=':4096:8')
    import torch
    torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True);torch.set_float32_matmul_precision('highest')
    from generation import iter_pg
    from calibration import evaluate,average,dump,sha
    out=Path(auth['output_dir']);out.mkdir(parents=True,exist_ok=False)
    reserve=120 if auth['operation'] in ('resume_pure_ensemble','confirm_frozen_ensemble_seed2','confirm_pure_ensemble','qualify_interrupted_pg8','resume_pure_ensemble','confirm_frozen_ensemble_seed2','train_variant','train_bcap','train_bispf','train_semantic','train_learned','train_network_rate','train_hybrid','continue_hybrid','train_extended_hybrid') else 30
    end=time.monotonic()+max(0,min(admission['authorization_expiry'],admission['ledger_started_unix']+auth['cap_seconds'])-time.time()-reserve)
    try:
        if auth['operation'] in ('qualify_interrupted_pg8','resume_pure_ensemble','confirm_frozen_ensemble_seed2'):
            sys.path.insert(0,str(b.HERE.parent/'recovery_runtime'))
            from execution import run
            run(auth,out,end,admission,validate_reference)
            return
        if auth['operation']=='confirm_pure_ensemble':
            sys.path.insert(0,str(b.HERE.parent/'confirmation_runtime'))
            from ensemble_confirmation_execution import run
            run(auth,out,end,admission,validate_reference)
            return
        if auth['operation'] in ('qualify_pure_ensemble','evaluate_pure_ensemble'):
            from ensemble_execution import run
            run(auth,out,end,admission,validate_reference)
            return
        if auth['operation']=='qualify_continuation':
            sys.path.insert(0,str(b.HERE.parent/'model'))
            import variant_pg_update
            variant_pg_update.configure(auth['variant_config'])
            from variant_pg_adapter import Context
            from continuation import qualify
            b.require(sha(auth['checkpoint_path'])==auth['checkpoint_SHA256'],'complete parent checkpoint changed')
            checkpoint=torch.load(auth['checkpoint_path'],map_location='cpu',weights_only=False)
            validate_reference(checkpoint,auth)
            ctx=Context(Path(auth['cache_dir']),'cuda:0')
            result=qualify(ctx,checkpoint,out)
            result.update(parent_checkpoint_SHA256=auth['checkpoint_SHA256'],allocation_id=admission['allocation_id'])
            dump(out/'qualification.json',result)
            return
        if auth['operation'] in ('evaluate_reference','evaluate_hybrid','evaluate_pure_variant'):
            if auth['operation'] in ('evaluate_hybrid','evaluate_pure_variant'):
                sys.path.insert(0,str(b.HERE.parent/'model'))
                import variant_pg_update
                variant_pg_update.configure(auth['variant_config'])
                from variant_pg_adapter import Context,restore
            else:
                from pg_adapter import Context,restore
            b.require(sha(auth['checkpoint_path'])==auth['checkpoint_SHA256'],'reference checkpoint changed')
            checkpoint=torch.load(auth['checkpoint_path'],map_location='cpu',weights_only=False)
            validate_reference(checkpoint,auth)
            ctx=Context(Path(auth['cache_dir']),'cuda:0')
            bundle,_,_,_,_=restore(ctx,checkpoint,expected_endpoint=auth['fixed_endpoint']) if auth['operation']=='evaluate_pure_variant' else restore(ctx,checkpoint)
            checkpoint_path=Path(auth['checkpoint_path'])
        else:
            sys.path.insert(0,str(b.HERE.parent/'model'))
            if auth['operation']=='train_network_rate':
                import network_rate_update
                network_rate_update.configure(auth['variant_config'])
                from network_rate_adapter import Context,fit
            elif auth['operation']=='train_learned':
                import learned_pg_update
                learned_pg_update.configure(auth['variant_config'])
                learned_pg_update.configure_observation(auth['learned_observation_config'])
                from learned_pg_adapter import Context,fit
            elif auth['operation']=='train_bcap':
                import bcap_update
                bcap_update.configure(auth['variant_config'])
                from bcap_adapter import Context,fit
            elif auth['operation']=='train_semantic':
                import semantic_pg_update
                semantic_pg_update.configure(auth['variant_config'])
                semantic_pg_update.configure_semantic(auth['semantic_config'])
                from semantic_pg_adapter import Context,fit
            elif auth['operation']=='train_bispf':
                import original_bispf_update
                original_bispf_update.configure(auth['variant_config'])
                from original_bispf_adapter import Context,fit
            else:
                import variant_pg_update
                variant_pg_update.configure(auth['variant_config'])
                from variant_pg_adapter import Context,fit
            ctx=Context(Path(auth['cache_dir']),'cuda:0')
            if auth['operation'] in ('continue_hybrid','train_extended_hybrid'):
                from continuation import fit_extended
                parent=None
                if auth['operation']=='continue_hybrid':
                    b.require(sha(auth['checkpoint_path'])==auth['checkpoint_SHA256'],'complete continuation parent changed')
                    parent=torch.load(auth['checkpoint_path'],map_location='cpu',weights_only=False)
                    validate_reference(parent,auth)
                bundle,checkpoint_path=fit_extended(ctx,out/'fit',b.cell(auth)['seed'],end_monotonic=end,parent=parent,parent_SHA256=auth['checkpoint_SHA256'] if parent is not None else None)
            else:bundle,checkpoint_path=fit(ctx,out/'fit',b.cell(auth)['seed'],end_monotonic=end)
            b.require(bundle[-1].completed_steps==auth['fixed_endpoint'],'complete declared variant endpoint required')
        def iterator_for(rep):
            def iterator(schedule,c,s):
                if auth['operation'] in ('evaluate_hybrid','train_hybrid','continue_hybrid','train_extended_hybrid'):
                    from hybrid import iter_hybrid
                    stream=iter_hybrid(ctx,bundle[1],bundle[3],schedule,c,s,fraction=auth['hybrid_config']['pg_bag_fraction'],replicate=rep)
                else:stream=iter_pg(ctx,bundle[1],bundle[3],schedule,c,s)
                for item in stream:
                    if time.monotonic()>=end:raise TimeoutError('calibration allocation ending; no partial score')
                    yield item
            return iterator
        method='pg_resampling_hybrid' if auth['operation'] in ('evaluate_hybrid','train_hybrid','continue_hybrid','train_extended_hybrid') else 'pg'
        if auth['operation']=='train_bispf':method='pg_bispf'
        if auth['operation']=='train_bcap':method='pg_bcap'
        if auth['operation']=='train_network_rate':method='pg_network_rate_half'
        if auth['operation']=='train_semantic':method='pg_semantic_observation'
        if auth['operation']=='train_learned':method='pg_learned_observation'
        if auth['fixed_endpoint']==40000:method+='_'+'40k'
        records=[evaluate(ctx,method,b.cell(auth)['seed'],rep,iterator_for(rep),out/f'r{rep}') for rep in [0,1]]
        result=average(records)
        result.update(status='completed',completed_updates=auth['fixed_endpoint'],fixed_endpoint=auth['fixed_endpoint'],integer_count_contract_passed=True,
                      allocation_id=admission['allocation_id'],operation=auth['operation'],
                      checkpoint_path=str(checkpoint_path),checkpoint_SHA256=sha(checkpoint_path),
                      variant_config=auth.get('variant_config',{'z_dim':2}),
                      method=method,hybrid_config=auth.get('hybrid_config'),
                      qualification_only=auth.get('qualification_only',False),
                      new_model_updates=auth['fixed_endpoint']-auth.get('initial_completed_updates',0) if auth['operation'] in ('train_variant','train_bcap','train_bispf','train_semantic','train_learned','train_network_rate','train_hybrid','continue_hybrid','train_extended_hybrid') else 0,
                      initial_completed_updates=auth.get('initial_completed_updates',0),native_recipe_schedule_horizon=20000,
                      parent_checkpoint_SHA256=auth.get('checkpoint_SHA256') if auth['operation']=='continue_hybrid' else None,
                      automatic_successor=False)
        if auth['operation']=='evaluate_pure_variant':
            result.update(generation='direct_iter_pg',real_bag_replacements=0,pure_formulation=auth['pure_formulation'],pure_only=True)
        if auth['operation']=='train_bispf':
            result.update(generation='direct_iter_pg',real_bag_replacements=0,pure_only=True,native_recipe='bispf',pure_formulation='original_bispf')
        if auth['operation']=='train_network_rate':
            result.update(generation='direct_iter_pg',real_bag_replacements=0,pure_only=True,native_recipe='bispf_optimized',pure_formulation='network_rate_half',network_rate_config=auth['network_rate_config'])
        if auth['operation']=='train_bcap':
            result.update(generation='direct_iter_pg',real_bag_replacements=0,pure_only=True,native_recipe='bcap',pure_formulation='bcap')
        if auth['operation']=='train_learned':
            result.update(generation='direct_iter_pg',real_bag_replacements=0,pure_only=True,native_recipe='bispf_optimized',pure_formulation='learned_observation',learned_observation_config=auth['learned_observation_config'])
        if auth['operation']=='train_semantic':
            result.update(generation='direct_iter_pg',real_bag_replacements=0,pure_only=True,native_recipe='bispf_optimized',pure_formulation='semantic_observation',semantic_config=auth['semantic_config'],semantic_observation=ctx.observation.metadata(),classifier_file_SHA256=ctx.classifier_file_SHA256)
        dump(out/'completed.json',result)
    except BaseException as exc:
        dump(out/'failure.json',{'status':'incomplete','error':repr(exc),'automatic_retry':False})
        raise

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--authorization',type=Path,required=True)
    parser.add_argument('--cpu-fixture-root',type=Path)
    args=parser.parse_args();layout=b.Layout.fixture(args.cpu_fixture_root) if args.cpu_fixture_root else b.NATIVE_LAYOUT
    auth,admission=b.require_live(args.authorization,layout)
    # Repeat the controller-specific deployment/source gate, without reserving.
    import project_runner_gpu0
    project_runner_gpu0.controller_gate(b.read(layout.project/'project_gpu_budget.json'),auth)
    if layout.fixture_mode:
        b.require('torch' not in sys.modules and os.environ.get('CUDA_VISIBLE_DEVICES')=='','CPU fixture imported model or exposed CUDA')
        if auth.get('fixture_fail_exit'):return 7
        out=Path(auth['output_dir']);out.mkdir(parents=True,exist_ok=False)
        (out/'CPU_FIXTURE_COMPLETE.json').write_text(json.dumps({'status':'CPU_BINDING_FIXTURE_ONLY','allocation_id':auth['allocation_id'],'attempt_id':auth['attempt_id'],'model_imported':False,'CUDA_VISIBLE_DEVICES':os.environ['CUDA_VISIBLE_DEVICES']})+'\n')
        return 0
    return run_scientific(auth,admission,layout)

if __name__=='__main__':raise SystemExit(main())
