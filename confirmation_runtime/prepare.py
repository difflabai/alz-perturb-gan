"""Prepare one frozen seed's two fresh20k parents with the existing finite GPU0 guard."""
import argparse,copy,datetime,json,subprocess,sys,time
from pathlib import Path
ROOT=Path('/ml2/alz');sys.path[:0]=[str(ROOT/'runtime'),str(ROOT),str(Path(__file__).parent)]
import binding as b
import ensemble_confirmation_scope as scope

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--seed',required=True,type=int,choices=[1,2]);args=parser.parse_args()
    freeze=scope.frozen();ident=freeze['confirmation_identities'][str(args.seed)];value=ident['cell'];selected=freeze['seed0_candidate']
    b.require(not subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True),'Commit complete confirmation implementation/proof before deployment')
    commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    previous=b.read(ROOT/'calibration_deployment.json');auth=copy.deepcopy(b.read(previous['authorization_path']))
    auth.pop('checkpoint_path',None);auth.pop('checkpoint_SHA256',None)
    folder=ROOT/'artifacts/allocations'/ident['attempt_id'];b.require(not folder.exists(),'Existing frozen seed allocation; inspect, never repeat/rerank')
    auth_path=folder/'authorization.json'
    auth.update(cell=value,allocation_id=ident['allocation_id'],attempt_id=ident['attempt_id'],label=ident['attempt_id'],
                scheduler_allocation_id='ALZ_PURE_ENSEMBLE_CONFIRM_'+datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ'),
                expires_unix=time.time()+21630,cap_seconds=21600,operation='confirm_pure_ensemble',fixed_endpoint=40000,
                output_dir=str(b.output_path(b.NATIVE_LAYOUT,value)),ledger_path=str(b.ledger_path(b.NATIVE_LAYOUT,value)),
                variant_config=selected['variant_config'],hybrid_config=None,candidate_commit=commit,pure_formulation='pg8_learned_ensemble75',
                ensemble_config=selected['ensemble_config'],ensemble_weight=.25,qualification_only=False,initial_completed_updates=0,new_model_updates=40000,parent_updates=[20000,20000],
                confirmation_freeze_SHA256=b.sha(scope.FREEZE),confirmation_qualification_SHA256=b.sha(scope.PROOF),
                source_review='Frozen accepted pure75PG8/25learned model. BOTH original20k parent fits from scratch for exact seed1/2 only, each original fitting Context/adapter and native optimized recipe, exact seed-derived streams. Frozen wholebag composition/scorer and immutable inputs unchanged. CPU orchestration/source/refusal and existing synthetic update/serialization/guard checks qualified; independent full nativeAPI/G-D-prior/streams/count-score audit required on completion. No seed ranking/incumbent mutation or real resampling; sixhour existing GPU0 guard, no automatic successor.')
    auth['command']=b.expected_command(b.NATIVE_LAYOUT,auth_path,value)
    research={str(p):b.sha(p) for p in [*ROOT.glob('*.py'),*(ROOT/'runtime').glob('*.py'),*(ROOT/'model').glob('*.py'),*scope.HERE.glob('*.py')]}
    auth['research_source_SHA256']=research;pins=auth['files_SHA256'];pins.update(research)
    paths=[scope.FREEZE,scope.PROOF,ROOT/'pure_incumbent.json',ROOT/'incumbent.json',ROOT/'analysis/pure_confirmation_protocol.json',ROOT/'ensemble_parents.json',ROOT/'ensemble_protocol.json',ROOT/'ITERATION_AUTHORIZATION_20261103.json']
    paths.extend(Path(q['path']) for q in freeze['matched_baselines'].values());paths.extend(map(Path,scope.read(scope.PROOF)['source_SHA256']))
    for path in paths:pins[str(path)]=b.sha(path)
    b.source_closure(auth, b.NATIVE_LAYOUT)
    controller, admission, _ = b.modules(auth)
    observation = admission.snapshot()
    policy = b.read(auth['policy_path'])
    b.require(set(observation['gpu_memory']) == {p['identity']['pid'] for p in policy['protected_peers']}, 'unknown or missing GPU0 peer')
    for peer in policy['protected_peers']:
        pid = peer['identity']['pid']
        b.require(admission.protected_identity(pid) == peer['identity']
                  and set(observation['gpu_kinds'][pid]) == set(peer['gpu_kinds']), 'protected identity/membership changed')
    b.require(not b.read(Path(auth['registry_path']) / 'registry.json')['jobs'], 'resource registry has active work')
    backend = b.load_pinned(ROOT / 'runtime/project_runner_gpu0.py', research[str(ROOT / 'runtime/project_runner_gpu0.py')], 'project_runner_gpu0')
    metadata = {'status': 'REVIEWED_FOR_BOUNDED_EXECUTION', 'path': str(ROOT / 'runtime/project_runner_gpu0.py'),
                'source_SHA256': {str(p): pins[str(p)] for p in [*(ROOT / 'runtime').glob('*.py'), b.GUARD / 'gpu0_admission.py', b.GUARD / 'gpu0_runner.py']},
                'files_SHA256': pins, 'policy_SHA256': auth['policy_SHA256'],
                **{k: auth[k] for k in ['allocation_id', 'attempt_id', 'scheduler_allocation_id', 'expires_unix']}}
    control = controller.Controller(b.PROJECT)
    with control.transaction() as state:
        b.require(not state['open_reservations'], 'another canonical reservation is active')
        b.unseen(auth, b.NATIVE_LAYOUT, state); b.core_envelope(state, 21630)
        b.require(control._status(state)['available_after_reservations_GPU_hours'] >= 21630 / 3600, 'insufficient project hours')
        state['future_accounting_policy']['current_launch_admission'] = True
        state['execution_controller']['status'] = 'REVIEWED_FOR_BOUNDED_EXECUTION'
        state['execution_controller']['calibration_backend'] = metadata
        backend.controller_gate(state, auth)
    folder.mkdir(parents=True, exist_ok=False)
    auth_path.write_text(json.dumps(auth, indent=2, allow_nan=False) + '\n')
    b.verify(auth_path)
    result={'status':'ONE_USE_FROZEN_PURE_ENSEMBLE_CONFIRMATION_DEPLOYED_NO_JOB_STARTED','authorization_path':str(auth_path),
            'authorization_SHA256':b.sha(auth_path),'allocation_id':auth['allocation_id'],'attempt_id':auth['attempt_id'],
            'candidate_commit':commit,'variant_config':auth['variant_config'],'hybrid_config':None,'pure_formulation':auth['pure_formulation'],
            'ensemble_config':auth['ensemble_config'],'endpoint':40000,'qualification_only':False,'output_dir':auth['output_dir'],
            'ledger_path':auth['ledger_path'],'cap_seconds':21600,'source_pin_count':len(pins),'base_model_updates':0,'new_model_updates':40000,
            'confirmation_freeze_SHA256':b.sha(scope.FREEZE),'training_seed':args.seed}
    (folder/'deployment.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
if __name__=='__main__':main()
