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
    b.require(auth['operation']=='evaluate_ka2_ema','generation-only saved KA2 EMA evaluation required')
    runtime=b.runtime_path(layout,b.cell(auth))
    b.require(Path(sys.executable).absolute()==runtime.absolute() and sys.version_info[:2]==(3,12),'exact scientific interpreter required')
    sys.path.extend(map(str,b.native_site_roots(b.cell(auth),auth['files_SHA256'])))
    sys.path[:0]=[str(b.HERE.parent/'ema_readout_model'),str(b.HERE.parent/'model'),str(layout.science),str(b.HERE.parent)]
    def deny(self,address):raise RuntimeError('calibration worker network disabled')
    socket.socket.connect=deny;socket.socket.connect_ex=deny
    os.environ.update(WANDB_MODE='disabled',WANDB_DISABLED='true',OTEL_SDK_DISABLED='true',DO_NOT_TRACK='1',HF_HUB_OFFLINE='1',CUBLAS_WORKSPACE_CONFIG=':4096:8')
    import torch
    torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True);torch.set_float32_matmul_precision('highest')
    total=torch.cuda.get_device_properties(0).total_memory;torch.cuda.set_per_process_memory_fraction((3*2**30)/total,0)
    from calibration import evaluate,average,dump,sha
    from generation import iter_pg
    import ka2_ema
    from ka2_adapter import Context
    b.require(Path(ka2_ema.__file__).resolve()==b.HERE.parent/'ema_readout_model/ka2_ema.py','qualified EMA helper origin required')
    b.require(Path(sys.modules['ka2_adapter'].__file__).resolve()==b.HERE.parent/'model/ka2_adapter.py','qualified parent adapter origin required')
    out=Path(auth['output_dir']);out.mkdir(parents=True,exist_ok=False)
    end=time.monotonic()+max(0,min(admission['authorization_expiry'],admission['ledger_started_unix']+auth['cap_seconds'])-time.time()-120)
    try:
        ctx=Context(Path(auth['cache_dir']),'cuda:0')
        b.require(sha(auth['parent_checkpoint_path'])==auth['parent_checkpoint_SHA256'],'parent checkpoint changed')
        q=torch.load(auth['parent_checkpoint_path'],map_location='cpu',weights_only=False)
        validate_reference(q,auth)
        G,P,info=ka2_ema.load(ctx.mu,ctx.theta,'cuda:0',q,cache_sha=sha(Path(auth['cache_dir'])/'manifest.json'),fold=0,seed=0)
        b.require(info['readout']==auth['readout_config'] and info['training_updates_executed']==0,'readout contract differs')
        ref={'kind':'explicit_native_KA2_EMA_readout_reference','parent_checkpoint_path':auth['parent_checkpoint_path'],'parent_checkpoint_SHA256':auth['parent_checkpoint_SHA256'],'parent_result_path':auth['parent_result_path'],'parent_result_SHA256':auth['parent_result_SHA256'],'fold':0,'seed':0,'cache_manifest_SHA256':q['cache_manifest_SHA256'],**info}
        refpath=out/'ema_reference.json';dump(refpath,ref);del q
        def iterator_for(rep):
            def iterator(schedule,c,s):
                for item in iter_pg(ctx,G,P,schedule,c,s):
                    if time.monotonic()>=end:raise TimeoutError('finite EMA evaluation ending; no partial score')
                    yield item
            return iterator
        records=[evaluate(ctx,'pg_ka2_native_ema',0,rep,iterator_for(rep),out/f'r{rep}') for rep in (0,1)]
        result=average(records)
        result.update(status='completed',completed_updates=20000,fixed_endpoint=20000,integer_count_contract_passed=True,
                      allocation_id=admission['allocation_id'],operation=auth['operation'],checkpoint_path=str(refpath),checkpoint_SHA256=sha(refpath),checkpoint_kind='explicit_native_KA2_EMA_readout_reference',
                      variant_config=auth['variant_config'],method='pg_ka2_native_ema',hybrid_config=None,qualification_only=False,
                      new_model_updates=0,initial_completed_updates=20000,parent_training_updates_total=20000,native_recipe_schedule_horizon=20000,parent_checkpoint_SHA256=auth['parent_checkpoint_SHA256'],automatic_successor=False,
                      generation='direct_iter_pg',real_bag_replacements=0,pure_only=True,native_recipe='ka2',pure_formulation='ka2_saved_native_ema_readout',readout_config=auth['readout_config'],
                      preset_protocol_SHA256=auth['preset_protocol_SHA256'])
        dump(out/'completed.json',result)
    except BaseException as exc:
        dump(out/'failure.json',{'status':'incomplete','error':repr(exc),'automatic_retry':False});raise

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
