"""Prepare exact recovery with reviewed ALZ-owned parallel profile; old policy intact."""
import argparse,copy,datetime,json,subprocess,sys,time
from pathlib import Path
ROOT=Path('/ml2/alz');sys.path[:0]=[str(ROOT/'runtime'),str(ROOT),str(Path(__file__).parent)]
import binding as b
import recovery_scope as scope
import parallel_profile as parallel

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--mode',choices=['qualify','resume','seed2'],required=True);parser.add_argument('--check-only',action='store_true');args=parser.parse_args()
    p,f=scope.lineage();value=scope.cell_for(args.mode)
    if not args.check_only:parallel.reviewed()
    if not args.check_only:b.require(not subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True),'Commit complete source qualification before deployment')
    commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    if not args.check_only:
        parallel.DEPLOYED.parent.mkdir(parents=True,exist_ok=True)
        if parallel.DEPLOYED.exists():b.require(parallel.DEPLOYED.read_bytes()==parallel.deployed_bytes(),'Existing parallel deployed policy differs; never replace')
        else:
            with parallel.DEPLOYED.open('xb') as stream:stream.write(parallel.deployed_bytes())
    prior=b.read(ROOT/'calibration_deployment.json');auth=copy.deepcopy(b.read(prior['authorization_path']))
    auth.pop('checkpoint_path',None);auth.pop('checkpoint_SHA256',None)
    qualify=args.mode=='qualify';seed2=args.mode=='seed2'
    folder=ROOT/'artifacts/allocations'/scope.attempt(value);b.require(not folder.exists(),'This lineage phase already exists; inspect, never repeat/reset')
    auth_path=folder/'authorization.json';cap=900 if qualify else 21600
    expiry=time.time()+21630 if seed2 else p['wall_clock_envelope']['absolute_recovery_deadline_unix']
    b.require(expiry>time.time()+150,'Original fixed deadline leaves no startup/cleanup window; no sixhour reset')
    auth.update(cell=value,allocation_id=scope.allocation(value),attempt_id=scope.attempt(value),label=scope.attempt(value),
                scheduler_allocation_id='ALZ_EXACT_RECOVERY_'+datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ'),
                expires_unix=expiry,cap_seconds=cap,operation={'qualify':'qualify_interrupted_pg8','resume':'resume_pure_ensemble','seed2':'confirm_frozen_ensemble_seed2'}[args.mode],fixed_endpoint=20000 if qualify else 40000,
                output_dir=str(b.output_path(b.NATIVE_LAYOUT,value)),ledger_path=str(b.ledger_path(b.NATIVE_LAYOUT,value)),variant_config=f['seed0_candidate']['variant_config'],hybrid_config=None,candidate_commit=commit,
                pure_formulation='pg8_learned_ensemble75',ensemble_config=p['frozen_ensemble_config'],qualification_only=qualify,
                initial_completed_updates=0 if seed2 else scope.PARENT_INITIAL if qualify else scope.INITIAL,initial_parent_updates=[0,0] if seed2 else [20000,scope.PARENT_INITIAL],new_model_updates={'qualify':2,'resume':40000-scope.INITIAL,'seed2':40000}[args.mode],
                recovery_protocol_SHA256=b.sha(scope.PROTOCOL),recovery_CPU_proof_SHA256=b.sha(scope.CPU_PROOF),
                source_review='Exact learned9237 continuation retaining completed20kPG8 and every prior interruption/qualification charge. Native restoration of retainedPG8 and learned9237/two9238 branches mandatory. Prior description for historical context: second exact saved14410 continuation after reviewed MOSSfit identity turnover. Originalfailed8942 and first8942to14410 phases/history/charges retained. Original frozen46 model/scorer sources and inputs unchanged. Fullstate CPU audit plus synthetic serialization/next-update parity and guarded/source tests qualified. Actual sameGPU initial/two-next-branch bitwise CUDA restoration mandatory before resume; replaybranch14411 never used for training. Resume to original20kPG8+fresh20klearned, frozen75/25 wholebags and2reps; original absolute6h seed1 deadline. Seed2 keeps exact originalidentity/fresh bothparents. No seedranking/incumbent mutation/foreignsignals; independently reviewed separate ALZ-owned exact current8peer profile/registry, original task11 policy/registry/admission/referenceguard and Controller untouched; separate ALZ-owned runner retains original admission/health/finite/accounting gates with exact owned-only cleanup after foreign departure.')
    if not seed2:auth.update(checkpoint_path=p['checkpoint_path'],checkpoint_SHA256=p['checkpoint_SHA256'])
    auth['command']=b.expected_command(b.NATIVE_LAYOUT,auth_path,value)
    parallel.configure(auth,check_only=args.check_only)
    research={str(path):b.sha(path) for folder_source in [ROOT,ROOT/'model',ROOT/'runtime',ROOT/'confirmation_runtime',scope.HERE] for path in folder_source.glob('*.py')};auth['research_source_SHA256']=research;pins=auth['files_SHA256'];pins.update(research)
    paths=[scope.PROTOCOL,scope.CPU_PROOF,Path(p['original_confirmation_freeze']),Path(p['original_failed_ledger']),Path(p['original_interruption_evidence']),Path(p['checkpoint_path']),ROOT/'analysis/interrupted_seed1_source_history.json',ROOT/'analysis/failed_seed1_full_state_audit.json',ROOT/'analysis/exact_interrupted_resume_cpu_qualification.json',ROOT/'pure_incumbent.json',ROOT/'incumbent.json',ROOT/'analysis/pure_confirmation_protocol.json']
    paths.extend(scope.continuation_paths(p));paths.extend(Path(q['path']) for q in f['matched_baselines'].values())
    for proof_path in [scope.CPU_PROOF,ROOT/'analysis/exact_learned_resume_cpu_qualification.json',ROOT/'analysis/exact_interrupted_resume_cpu_qualification.json',ROOT/'analysis/failed_seed1_full_state_audit.json']:
        paths.extend(map(Path,b.read(proof_path)['source_SHA256']))
    history=b.read(ROOT/'analysis/interrupted_seed1_source_history.json');paths.append(Path(history['authorization_path']));paths.extend(map(Path,history['verification_source_SHA256']))
    if not qualify:
        auth['native_restore_proof_SHA256']=b.sha(scope.NATIVE_PROOF);paths.append(scope.NATIVE_PROOF);paths.extend(map(Path,b.read(scope.NATIVE_PROOF)['verification_source_SHA256']))
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
    registry=Path(auth['registry_path'])/'registry.json'
    b.require(not registry.exists() or not b.read(registry)['jobs'], 'resource registry has active work')
    forecast=parallel.capacity(observation,admission)
    parallel.free_floor()
    if args.check_only:
        print(json.dumps({'status':'CHECK_ONLY_PARALLEL_SOURCE_PEERS_CAPACITY_PASS_INDEPENDENT_REVIEW_STILL_REQUIRED','mode':args.mode,'attempt':auth['attempt_id'],'policy_SHA256':parallel.policy_sha(),'forecast':forecast},indent=2));return
    backend = b.load_pinned(ROOT / 'runtime/project_runner_gpu0.py', research[str(ROOT / 'runtime/project_runner_gpu0.py')], 'project_runner_gpu0')
    metadata = {'status': 'REVIEWED_FOR_BOUNDED_EXECUTION', 'path': str(ROOT / 'runtime/project_runner_gpu0.py'),
                'source_SHA256': {str(p): pins[str(p)] for p in [*(ROOT / 'runtime').glob('*.py'), b.GUARD / 'gpu0_admission.py', b.GUARD / 'gpu0_runner.py']},
                'files_SHA256': pins, 'policy_SHA256': auth['policy_SHA256'],
                **{k: auth[k] for k in ['allocation_id', 'attempt_id', 'scheduler_allocation_id', 'expires_unix']}}
    control = controller.Controller(b.PROJECT)
    with control.transaction() as state:
        b.require(not state['open_reservations'], 'another canonical reservation is active')
        b.unseen(auth, b.NATIVE_LAYOUT, state); b.core_envelope(state, auth['cap_seconds']+30)
        b.require(control._status(state)['available_after_reservations_GPU_hours'] >= (auth['cap_seconds']+30) / 3600, 'insufficient project hours')
        state['future_accounting_policy']['current_launch_admission'] = True
        state['execution_controller']['status'] = 'REVIEWED_FOR_BOUNDED_EXECUTION'
        state['execution_controller']['calibration_backend'] = metadata
        backend.controller_gate(state, auth)
    folder.mkdir(parents=True, exist_ok=False)
    auth_path.write_text(json.dumps(auth, indent=2, allow_nan=False) + '\n')
    b.verify(auth_path)
    result={'status':'ONE_USE_EXACT_FROZEN_RECOVERY_PHASE_DEPLOYED_NO_JOB_STARTED','authorization_path':str(auth_path),'authorization_SHA256':b.sha(auth_path),
            'allocation_id':auth['allocation_id'],'attempt_id':auth['attempt_id'],'candidate_commit':commit,'variant_config':auth['variant_config'],'hybrid_config':None,
            'pure_formulation':auth['pure_formulation'],'ensemble_config':auth['ensemble_config'],'endpoint':auth['fixed_endpoint'],'qualification_only':qualify,
            'output_dir':auth['output_dir'],'ledger_path':auth['ledger_path'],'cap_seconds':cap,'expires_unix':expiry,'source_pin_count':len(pins),
            'base_model_updates':auth['initial_completed_updates'],'new_model_updates':auth['new_model_updates'],'training_seed':value['seed'],'recovery_protocol_SHA256':b.sha(scope.PROTOCOL)}
    (folder/'deployment.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
if __name__=='__main__':main()
