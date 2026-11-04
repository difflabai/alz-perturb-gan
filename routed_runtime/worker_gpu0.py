"""One bounded native routed E22 count trial; binding precedes model imports."""
import argparse
import json
import os
from pathlib import Path
import socket
import sys
import time

sys.path.insert(0,str(Path(__file__).resolve().parent))
import binding as b


def validate_reference(checkpoint,auth):
    value=b.cell(auth)
    b.require(checkpoint.get('schema')==1 and checkpoint.get('method')=='pg','reference checkpoint schema/method differs')
    b.require(checkpoint.get('fold')==value['fold'] and checkpoint.get('seed')==value['seed'],'reference checkpoint fold/seed differs')
    b.require(checkpoint.get('status')=='completed' and checkpoint.get('completed_updates')==checkpoint.get('fixed_endpoint')==20000,'complete fixed reference checkpoint required')
    b.require(checkpoint.get('cache_manifest_SHA256')==b.sha(Path(auth['cache_dir'])/'manifest.json'),'reference preprocessing/cache differs')


def run_scientific(auth,admission,layout):
    b.require(auth['operation']=='train_routed_e22','one explicit routed E22 native trial only')
    b.require(Path(sys.executable).absolute()==b.runtime_path(layout,b.cell(auth)).absolute()
              and sys.version_info[:2]==(3,12),'exact native scientific interpreter required')
    sys.path.extend(map(str,b.native_site_roots(b.cell(auth),auth['files_SHA256'])))
    sys.path[:0]=[str(b.HERE.parent/'routed_model'),str(b.HERE.parent/'model'),str(layout.science),str(b.HERE.parent)]
    def deny(self,address):raise RuntimeError('Scientific worker network disabled')
    socket.socket.connect=deny;socket.socket.connect_ex=deny
    os.environ.update(WANDB_MODE='disabled',WANDB_DISABLED='true',OTEL_SDK_DISABLED='true',DO_NOT_TRACK='1',
                      HF_HUB_OFFLINE='1',JAX_PLATFORMS='cpu',CUBLAS_WORKSPACE_CONFIG=':4096:8')
    import torch
    torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
    torch.set_float32_matmul_precision('highest')
    total=torch.cuda.get_device_properties(0).total_memory
    torch.cuda.set_per_process_memory_fraction((3*2**30)/total,0)
    from calibration import evaluate,average,dump,sha
    import routed_pg_adapter as model
    import routed_counts
    import routed_generation
    import partition
    from state_fingerprint import fingerprint
    for module in (model,routed_counts,routed_generation,partition):
        b.require(Path(module.__file__).resolve().parent==b.HERE.parent/'routed_model','Exact qualified routed model import origin required')
    b.require(auth['variant_config']==model.CONFIG and auth['routed_config']==model.ROUTED_CONFIG,'Qualified routed configuration required')
    out=Path(auth['output_dir']);out.mkdir(parents=True,exist_ok=False)
    end=time.monotonic()+max(0,min(admission['authorization_expiry'],admission['ledger_started_unix']+auth['cap_seconds'])-time.time()-120)
    try:
        ctx=model.Context(Path(auth['cache_dir']),'cuda:0')
        runtime,checkpoint_path=model.fit(ctx,out/'fit',b.cell(auth)['seed'],end_monotonic=end)
        checkpoint=torch.load(checkpoint_path,map_location='cpu',weights_only=False)
        validate_reference(checkpoint,auth)
        b.require(runtime.policy.completed_steps==20000 and checkpoint['native_recipe']=='e22_routed'
                  and checkpoint['recipe']==runtime.recipe.to_dict() and checkpoint['routed_config']==auth['routed_config'],
                  'Complete native routed recipe and checkpoint required')
        saved_policy=fingerprint(checkpoint['policy'])
        # Validate and restore all CUDA-specific controllers and RNG engines on
        # the admitted CUDA policy; CPU auditing must not relabel that metadata.
        runtime.policy.load_state_dict(checkpoint['policy'])
        b.require(saved_policy==fingerprint(runtime.policy.state_dict()),'Full native public CUDA checkpoint reload differs')
        b.require(fingerprint(runtime.fitting_rng.bit_generator.state)==fingerprint(checkpoint['fitting_rng'])
                  and fingerprint(runtime.protected_rng.bit_generator.state)==fingerprint(checkpoint['protected_rng']),
                  'Independent fitting/protection sampler state differs')
        b.require(checkpoint['partition']==ctx.partition.summary(),'Actual protected fitting partition differs')
        reload_receipt=out/'native_checkpoint_reload.json'
        dump(reload_receipt,{'status':'FULL_NATIVE_PUBLIC_CUDA_POLICY_RELOAD_AND_BYTE_FINGERPRINT_PASSED',
                             'checkpoint_path':str(checkpoint_path),'checkpoint_SHA256':sha(checkpoint_path),
                             'policy_fingerprint':saved_policy,'completed_updates':20000,
                             'native_device_metadata_preserved':checkpoint['policy']['device'],
                             'both_independent_data_sampler_states_match':True,'CUDA_rng_states_used_only_on_admitted_CUDA_policy':True})
        del checkpoint
        served=runtime.policy.served_model()
        served_reference=out/'native_served_reference.json'
        serving={'weights':'native_coupled_served_snapshot','source':served.source,'output_noise':False,
                 'perturb':False,'raw_count_half_primary':True}
        dump(served_reference,{'status':'NATIVE_COUPLED_SERVED_COUNT_REFERENCE','checkpoint_path':str(checkpoint_path),
                               'checkpoint_SHA256':sha(checkpoint_path),'completed_updates':20000,
                               'models_fingerprint':fingerprint({k:v.state_dict() for k,v in served.models.items()}),
                               'table_fingerprint':fingerprint(served.table),'serving_contract':serving})
        def iterator_for(rep):
            def iterator(schedule,c,s):
                for item in routed_generation.iter_routed_counts(ctx,served,runtime.forward,schedule,c,s):
                    if time.monotonic()>=end:raise TimeoutError('Finite allocation ending; no partial calibration score')
                    yield item
            return iterator
        rows=[evaluate(ctx,'pg_e22_routed_counts',0,rep,iterator_for(rep),out/f'r{rep}') for rep in (0,1)]
        result=average(rows)
        result.update(status='completed',completed_updates=20000,fixed_endpoint=20000,integer_count_contract_passed=True,
                      allocation_id=admission['allocation_id'],operation=auth['operation'],checkpoint_path=str(checkpoint_path),
                      checkpoint_SHA256=sha(checkpoint_path),variant_config=auth['variant_config'],routed_config=auth['routed_config'],
                      method='pg_e22_routed_counts',hybrid_config=None,qualification_only=False,new_model_updates=20000,
                      initial_completed_updates=0,native_recipe_schedule_horizon=None,parent_checkpoint_SHA256=None,
                      automatic_successor=False,generation='native_served_routed_count_forward',real_bag_replacements=0,
                      pure_only=True,native_recipe='e22_routed',pure_formulation='e22_routed_count',
                      partition=ctx.partition.summary(),serving_contract=serving,
                      native_reload_receipt_path=str(reload_receipt),native_reload_receipt_SHA256=sha(reload_receipt),
                      served_reference_path=str(served_reference),served_reference_SHA256=sha(served_reference),
                      preset_protocol_SHA256=auth['preset_protocol_SHA256'])
        dump(out/'completed.json',result)
    except BaseException as exc:
        dump(out/'failure.json',{'status':'incomplete','error':repr(exc),'automatic_retry':False})
        raise


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--authorization',type=Path,required=True)
    parser.add_argument('--cpu-fixture-root',type=Path)
    args=parser.parse_args();layout=b.Layout.fixture(args.cpu_fixture_root) if args.cpu_fixture_root else b.NATIVE_LAYOUT
    auth,admission=b.require_live(args.authorization,layout)
    import project_runner_gpu0
    project_runner_gpu0.controller_gate(b.read(layout.project/'project_gpu_budget.json'),auth)
    if layout.fixture_mode:
        b.require('torch' not in sys.modules and os.environ.get('CUDA_VISIBLE_DEVICES')=='','CPU fixture imported model or exposed CUDA')
        if auth.get('fixture_fail_exit'):return 7
        out=Path(auth['output_dir']);out.mkdir(parents=True,exist_ok=False)
        (out/'CPU_FIXTURE_COMPLETE.json').write_text(json.dumps({'status':'CPU_BINDING_FIXTURE_ONLY',
            'allocation_id':auth['allocation_id'],'attempt_id':auth['attempt_id'],'model_imported':False,
            'CUDA_VISIBLE_DEVICES':os.environ['CUDA_VISIBLE_DEVICES']})+'\n')
        return 0
    return run_scientific(auth,admission,layout)


if __name__=='__main__':raise SystemExit(main())
