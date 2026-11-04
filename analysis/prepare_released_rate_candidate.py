"""Create a disabled, isolated member of the predeclared released rate grid.

Does not deploy, reserve, start a worker, change other sources or promote a model.
The generated runtime still needs its actual CPU qualification, source commit,
fresh capacity review and one-use native admission.
"""
import argparse,hashlib,json,re,subprocess,datetime
from pathlib import Path
ROOT=Path('/ml2/alz');TEMPLATE=ROOT/'released_rate_runtime'
PROTOCOL=ROOT/'analysis/released_tuning/protocol.json'
TEMPLATE_PROOF=ROOT/'analysis/released_rate_runtime_CPU_qualification.json'
GPUS={0:'GPU-72c1b506-891d-b8bc-b353-e020585e1c47',1:'GPU-cb4ce47d-d968-bffd-5646-e830a9fa1c69'}
read=lambda p:json.loads(Path(p).read_bytes())
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()

def assignment(source,name,value):
    source,n=re.subn(r'^'+re.escape(name)+r'=.*$',lambda _:name+'='+value,source,flags=re.M)
    assert n==1,(name,n)
    return source

def make_candidate(candidate_id,gpu,predecessor_attempt,*,alternate_first_card=False):
    protocol=read(PROTOCOL)
    settings=[s for s in protocol['stage1']['configs'] if s['id']==candidate_id]
    assert len(settings)==1,'Candidate ID is not in the registered27-setting grid'
    if candidate_id=='lr_g050_d050_p050':
        assert alternate_first_card and gpu==0,'First setting already has its own prepared GPU1 runtime; explicit unused GPU0 alternative required'
        attempt='CAL_GPU0_V1_released_rate_g050d050p050_f0_pg_s0'
        assert not (ROOT/'artifacts/allocations'/attempt).exists() and not (ROOT/'runs'/attempt).exists(),'First scientific key already prepared or consumed; no alternate-card duplicate'
    else:assert alternate_first_card is False,'Alternate-first flag is specific to the first unused setting'
    setting=settings[0];assert gpu in GPUS
    assert re.fullmatch(r'CAL_GPU0_V1_[a-z][a-z0-9_]{0,48}_f0_pg_s0',predecessor_attempt),'Exact owned calibration predecessor required'
    predecessor_auth=ROOT/'artifacts/allocations'/predecessor_attempt/'authorization.json'
    assert predecessor_auth.exists(),'Predecessor must already be concretely allocated, not guessed'
    previous=read(predecessor_auth)
    previous_registry=Path(previous['registry_path'])/'registry.json'
    assert previous_registry.is_relative_to(ROOT/'artifacts')
    proof=read(TEMPLATE_PROOF)
    assert proof['status']=='PASSED_ISOLATED_RELEASED_PARAMETER_RUNTIME_CPU_QUALIFICATION' and proof['tests_passed']==24
    for path,pin in proof['source_SHA256'].items():assert sha(path)==pin,'Qualified template or its model/input changed: '+path
    suffix=candidate_id.replace('lr_','').replace('_','')
    experiment='released_rate_'+suffix;name=experiment+'_runtime';folder=ROOT/name
    assert not folder.exists(),'Never overwrite a prepared or consumed scientific configuration'
    observation=ROOT/'analysis'/(experiment+'_initial_resource_observation.json')
    assert not observation.exists()
    query=subprocess.check_output(['nvidia-smi','-i',GPUS[gpu],'--query-gpu=uuid,memory.total,memory.used,memory.free','--format=csv,noheader,nounits'],text=True,timeout=5).strip().split(',')
    assert query[0].strip()==GPUS[gpu]
    available=int(next(line.split()[1] for line in Path('/proc/meminfo').read_text().splitlines() if line.startswith('MemAvailable:')))*1024
    obs={'status':'READ_ONLY_SELECTED_GPU_SNAPSHOT_NOT_ADMISSION','utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'gpu_uuid':GPUS[gpu],'GPU_total_bytes':int(query[1])*2**20,'GPU_used_bytes':int(query[2])*2**20,
        'GPU_free_bytes':int(query[3])*2**20,'host_available_bytes':available,'memory_pressure':Path('/proc/pressure/memory').read_text(),
        'GPU_context_created':False,'allocation_or_lease_created':False,'full_foreign_growth_capacity_forecast_required_at_admission':True}
    observation.write_text(json.dumps(obs,indent=2)+'\n')
    namespace=f'gpu{gpu}-released-{suffix}-20261104'
    scope_name='ONE_RELEASED_'+candidate_id.upper()+'_SEED0_TRIAL_ONLY'
    folder.mkdir()
    for path in TEMPLATE.glob('*.py'):
        source=path.read_text().replace('released_rate_runtime',name).replace('released_rate_g050d050p050',experiment)
        source=source.replace('lr_g050_d050_p050',candidate_id)
        source=source.replace('released_rate_owner_execution_review',experiment+'_owner_execution_review')
        source=source.replace('gpu1-released-rate050-20261104',namespace)
        source=source.replace('ONE_RELEASED_PARAMETER_RATE050_SEED0_TRIAL_ONLY',scope_name)
        if path.name=='preset_scope.py':
            source=assignment(source,'HYPER',repr(setting['parameters']))
            source=assignment(source,'RUNTIME_PROOF',"ROOT/'analysis/"+experiment+"_runtime_CPU_qualification.json'")
            source=source.replace("assert p['stage1']['configs'][0]=={'id':CANDIDATE_ID,'parameters':HYPER}",
                "assert [s for s in p['stage1']['configs'] if s['id']==CANDIDATE_ID]==[{'id':CANDIDATE_ID,'parameters':HYPER}]")
            source=source.replace("ROOT/'analysis/routed_e22_resume_prelaunch_resource_observation.json']",repr(observation)+']')
            # repr(Path) is not executable in this module; use an explicit Path.
            source=source.replace(repr(observation),"Path("+repr(str(observation))+")")
            source=source.replace(GPUS[1],GPUS[gpu]).replace("auth['dynamic_shared_gpu0'] is False","auth['dynamic_shared_gpu0'] is "+str(gpu==0))
            if gpu==0:
                source=source.replace("ROOT/'analysis/routed_e22_resume_shared_GPU1_CPU_qualification.json'","ROOT/'analysis/shared_gpu0_candidate_CPU_qualification.json'")
                source=source.replace('PASSED_CPU_UNCHANGED_SHARED_GPU1_ENGINE_FOR_ROUTED_CONTINUATION','PASSED_CPU_SHARED_GPU0_ENGINE_CANDIDATE_ONLY_WITH_CURRENT_READ_ONLY_FORECAST')
        elif path.name=='resource_profile.py':
            source=assignment(source,'PREVIOUS_REGISTRY','Path('+repr(str(previous_registry))+')')
            source=source.replace(GPUS[1],GPUS[gpu]).replace("s.replace('gpu-0.lock','gpu-1.lock')","s.replace('gpu-0.lock','gpu-"+str(gpu)+".lock')")
            source=source.replace('dynamic_shared_gpu0=False','dynamic_shared_gpu0='+str(gpu==0))
            source=source.replace("auth.get('dynamic_shared_gpu0') is False","auth.get('dynamic_shared_gpu0') is "+str(gpu==0))
            source=source.replace('Current E22 continuation must release before released rate trial','Owned predecessor must release before next registered rate trial')
        elif path.name=='test_scope.py':
            source=source.replace("**scope.HYPER,'particle_lr':.03","**scope.HYPER,'particle_lr':scope.HYPER['particle_lr']*2")
            source=source.replace("'tuning_candidate_id':'lr_g050_d050_p100'","'tuning_candidate_id':'invalid_candidate'")
            source=source.replace("['stage1']['configs'][0]['id'],scope.CANDIDATE_ID","['status'],'PROSPECTIVE_RELEASED_PARTICLEGAN_HYPERPARAMETER_SEARCH_NOT_LAUNCHED'")
            if gpu==0:source=source.replace("'shared_gpu1_admission.py'","'shared_gpu0_admission.py'")
        elif path.name=='prepare.py':
            source=source.replace("model=b.read(scope.MODEL_PROOF);pins.update(model['source_SHA256']);", "pins.update(b.read(scope.COMPAT_PROOF)['source_SHA256'])\n    model=b.read(scope.MODEL_PROOF);pins.update(model['source_SHA256']);")
            source=source.replace('G/D/prior rates=.006/.009/.015','G/D/prior nominal rates='+str(tuple(setting['parameters'][k] for k in ('generator_lr','critic_lr','particle_lr'))))
            source=source.replace('GPU1 shared guard unchanged','Selected GPU shared guard unchanged')
        elif path.name=='finish.py':
            source=source.replace('three learning rates uniformly halved via public recipe fields','registered G/D/prior learning rates applied via public recipe fields')
        (folder/path.name).write_text(source)
    policy=read(TEMPLATE/'policy.prepared.json')
    policy['gpu']=GPUS[gpu];policy['registry_path']=str(ROOT/'artifacts'/namespace/'live_registry')
    policy['locks']=[s.replace('gpu-1.lock',f'gpu-{gpu}.lock') for s in policy['locks']]
    policy['resource_basis']['observation_path']=str(observation);policy['resource_basis']['observation_SHA256']=sha(observation)
    policy['resource_basis']['foreign_control']='NONE. Selected released-grid scalar setting, same own/shared guard budgets. Initial snapshot is not admission; full fresh dynamic foreign growth forecast and predecessor release required.'
    (folder/'policy.prepared.json').write_text(json.dumps(policy,indent=2)+'\n')
    plan=read(TEMPLATE/'plan.json');plan.update(enabled=False,experiment=experiment,candidate_id=candidate_id)
    (folder/'plan.json').write_text(json.dumps(plan,indent=2)+'\n')
    import ast
    for path in folder.glob('*.py'):ast.parse(path.read_text())
    receipt={'status':'GENERATED_DISABLED_DECLARED_RATE_CANDIDATE_NOT_QUALIFIED_OR_DEPLOYED',
        'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'candidate_id':candidate_id,'parameters':setting['parameters'],
        'physical_GPU_uuid':GPUS[gpu],'runtime_directory':str(folder),'predecessor_attempt':predecessor_attempt,
        'alternative_first_card':alternate_first_card,
        'template_runtime_qualification_SHA256':sha(TEMPLATE_PROOF),'protocol_SHA256':sha(PROTOCOL),
        'generator_source_SHA256':sha(__file__),
        'source_SHA256':{str(path):sha(path) for path in folder.glob('*.py')},
        'next_required_action':'Run actual generated runtime CPU qualification; inspect changes, commit disabled source, fresh capacity/source review, then separate activation and one-use native admission. No automatic launch or promotion.'}
    dest=ROOT/'analysis'/(experiment+'_runtime_generation.json');assert not dest.exists()
    dest.write_text(json.dumps(receipt,indent=2)+'\n')
    return receipt

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--candidate-id',required=True);parser.add_argument('--gpu',type=int,choices=(0,1),required=True)
    parser.add_argument('--predecessor-attempt',required=True);parser.add_argument('--alternate-first-card',action='store_true');args=parser.parse_args()
    print(json.dumps(make_candidate(args.candidate_id,args.gpu,args.predecessor_attempt,alternate_first_card=args.alternate_first_card),indent=2))
