"""Independently remeasure saved endpoints; never start a model or change a trial."""
import argparse,datetime,gc,hashlib,json,os,sys
from pathlib import Path
ROOT=Path('/ml2/alz');STUDY=ROOT/'analysis/hpo500'
PROJECT=Path('/ml2/codex-storage/future-workspaces/2026-10-27/task-2/benchmark')
SCIENCE=Path('/ml2/codex-storage/future-workspaces/2026-11-03/task-2/alz-audit/core-comparison-20261103')
assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
sys.path[:0]=list(map(str,[STUDY,STUDY/'model',ROOT/'model',ROOT,SCIENCE,PROJECT,PROJECT/'current_base_20261029/runtime/lib/python3.12/site-packages',PROJECT/'.venv312/lib/python3.12/site-packages']))
from release_env import activate
activate()
import numpy as np,torch,search,released_adapter as model,released_update as update
from calibration import score,average

def read(path):return json.loads(Path(path).read_bytes())
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def live(identity):
    p=Path('/proc',str(identity['pid']),'stat')
    if not p.exists():return False
    s=p.read_text().rsplit(')',1)[1].split()
    return s[0]!='Z' and int(s[19])==identity['birth'] and Path('/proc/sys/kernel/random/boot_id').read_text().strip()==identity['boot']

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--minimum-selection',type=int,required=True);parser.add_argument('--name',required=True);parser.add_argument('--state-root',type=Path,default=ROOT/'artifacts/hpo500-r4');args=parser.parse_args()
    path=ROOT/'analysis'/(args.name+'.json');assert not path.exists()
    native=read(ROOT/'analysis/released_tuning_v096/native_release.json')
    for source,pin in native['official_wheel_source_SHA256'].items():assert sha(source)==pin
    assert sha(native['package_metadata_path'])==native['package_metadata_SHA256']
    assert (STUDY/'model/released_update.py').read_bytes()==(ROOT/'analysis/released_tuning_v096/released_update.py').read_bytes()
    files=sorted((STUDY/'results').glob('*.json'));records=[read(p) for p in files]
    selected=sorted([d for d in records if d['phase']=='search'],key=lambda d:d['candidate_id'])
    assert args.minimum_selection<=len(selected)<=25
    prior=[]
    for index,d in enumerate(selected):
        assert d['candidate_id']==f'c{index:03d}'
        assert read(STUDY/'proposals'/(d['candidate_id']+'.json'))==search.make_proposal(index,prior)
        prior.append(d)
    if len(selected)<25:
        pending=STUDY/'proposals'/(f'c{len(selected):03d}'+'.json')
        if pending.exists():assert read(pending)==search.make_proposal(len(selected),selected)
    config={'z_dim':8,'stage_balance':False,'cell_z_dim':8,'theta_init_scale':1.,'fourier_feature_scale':1.}
    rows=[];hashed=0;rescored=0;exemplars=0
    for source,d in zip(files,records):
        cid=d['candidate_id'];proposal=STUDY/'proposals'/(cid+'.json')
        assert d['endpoint']==500 and d['calibration_only'] and d['clean_settlement'] and d['owned_processes_and_lease_absent']
        assert d['protocol_SHA256']==sha(STUDY/'protocol.json') and d['proposal_SHA256']==sha(proposal)
        row={k:d[k] for k in ('attempt','candidate_id','seed','phase','finite','hyperparameters','charged_GPU_seconds')}
        row['result_SHA256']=sha(source)
        if not d['finite']:
            assert d['status']=='HPO500_SCIENTIFIC_NONFINITE_FAILURE_CLEAN_AUDIT'
            assert d['failure']['error'].startswith('FloatingPointError(')
            row['failure_retained']=d['failure'];rows.append(row);continue
        assert d['from_scratch'] and d['independent_saved_score_audit']
        assert d['hyperparameters']==read(proposal)['hyperparameters']
        if d.get('resumed_same_trial'):
            assert d['charged_GPU_seconds']==d['phase_charged_GPU_seconds']+d['parent_resource_interruption_GPU_seconds']
        for file,pin in d['files_SHA256'].items():assert sha(file)==pin;hashed+=1
        cp=next(file for file in d['files_SHA256'] if file.endswith('/fit/final.pt'))
        q=torch.load(cp,map_location='cpu',weights_only=False)
        assert q['status']=='completed' and q['fixed_endpoint']==q['completed_updates']==q['policy']['completed_steps']==q['usage']['steps']==500
        assert q['particlegan_version']=='0.9.6' and q['variant_config']==config and q['recipe']['ema_decay']==0
        assert q['training_streams']==model.training_streams(0,d['seed']) and q['fold']==0 and q['seed']==d['seed']
        assert q['released_hyperparameters']==d['hyperparameters']
        assert int(q['usage']['draws'].sum())==16000 and int(q['usage']['unsampled_moved_steps'].sum())==0
        update.configure(config);update.configure_hyperparameters(d['hyperparameters']);recipe=update.recipe()
        # Native0.9.6 intentionally omits this field in to_dict when it is none.
        assert recipe.generator_gradient_response=='none' and recipe.ema_decay==0
        assert q['recipe']==q['policy']['recipe']==recipe.to_dict() and recipe.total_steps==20000
        assert recipe.lr_floor==recipe.network_lr_floor==1
        out=Path(cp).parent.parent;completed=read(out/'completed.json')
        assert completed['evaluation_targets_read'] is False and completed['training_finished_before_calibration']
        reps=[]
        for rep in (0,1):
            dest=out/f'r{rep}';m=read(dest/'metrics.json')
            assert m['generated_cells']==320000 and m['generation_replicate']==rep and m['training_seed']==d['seed']
            with np.load(dest/'arrays.npz',allow_pickle=False) as a:measured=score(a['stage_synthetic_summaries'],a['generated_cells_by_stage'])
            for key,value in measured.items():
                if isinstance(value,(int,float,list)) and key!='donors':np.testing.assert_allclose(m[key],value,rtol=0,atol=1e-10)
            assert m==d['replicate_metrics'][rep]
            for stage in range(4):
                with np.load(dest/f'stage{stage}_exemplar.npz',allow_pickle=False) as a:
                    counts=a['counts'];assert counts.shape==(1,400,2000) and np.isfinite(counts).all() and (counts>=0).all() and np.equal(counts,np.floor(counts)).all()
                exemplars+=1
            reps.append(m);rescored+=1
        assert average(reps)==d['metrics']
        row['metrics']=d['metrics'];rows.append(row);del q;gc.collect()
    control=selected[0]['metrics'];valid=[d for d in selected if search.feasible(d,control)]
    best=min(valid,key=lambda d:(d['metrics']['energy_score'],d['metrics']['cell_energy_distance'],d['candidate_id']))
    current=read(args.state_root/'progress.json');driver=read(args.state_root/'driver_identity.json')
    running=None
    if current['status']=='NATIVE500STEP_FIT_OR_CALIBRATION_ACTIVE':
        auth=read(ROOT/'artifacts/allocations'/current['attempt']/'authorization.json');job=read(auth['ledger_path'])['jobs'][0]
        assert all(live(i) for i in (driver,job['guard_identity'],job['child_identity']))
        running=dict(candidate=current['candidate'],attempt=current['attempt'],GPU=auth['gpu_uuid'],driver=driver['pid'],guard=job['guard_identity']['pid'],worker=job['child_identity']['pid'])
    else:assert live(driver) or current['status'].startswith('COMPLETE_')
    assert not torch.cuda.is_initialized()
    result=dict(status='INDEPENDENT_FIXED500STEP_CHECKPOINT_AND_CALIBRATION_REMEASUREMENT_PASS',UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),endpoint=500,native_version='0.9.6',EMA_decay=0,formulation_changed=False,native_gradient_response='none; nativeRecipe.to_dict omits its default',official_native_Python_hashes_verified=len(native['official_wheel_source_SHA256']),native_update_byte_identical=True,proposal_sequence_reproduced=True,completed_selection_candidates=len(selected),maximum_selection_candidates=25,total_completed_selection_optimizer_pairs=sum(500 for d in selected if d['finite']),scientific_nonfinite_trials_retained=sum(not d['finite'] for d in records),saved_artifacts_hashed=hashed,full_calibration_replicates_independently_rescored=rescored,saved_exemplar_count_arrays_verified=exemplars,generated_cells_per_replicate=320000,rows=rows,best_provisional_candidate=best['candidate_id'],best_provisional_hyperparameters=best['hyperparameters'],best_provisional_metrics=best['metrics'],best_ES_reduction_percent=100*(1-best['metrics']['energy_score']/control['energy_score']),best_cellED_reduction_percent=100*(1-best['metrics']['cell_energy_distance']/control['cell_energy_distance']),actual_completed_trial_GPU_seconds=sum(d['charged_GPU_seconds'] for d in records),current_live_trial=running,heldout_evaluation_used_for_tuning=False,fresh_seed_confirmation_fits=sum(d['phase']=='confirm' for d in records),automatic_promotion=False,CUDA_initialized_by_audit=False,protocol_SHA256=sha(STUDY/'protocol.json'),audit_source_SHA256=sha(__file__),limitation='Best found so far on four adaptive calibration donors. Fresh seeds test initialization on those same donors; no independent cohort, global optimum, or SOTA claim.')
    path.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:result[k] for k in ('status','completed_selection_candidates','total_completed_selection_optimizer_pairs','full_calibration_replicates_independently_rescored','best_provisional_candidate','best_ES_reduction_percent','best_cellED_reduction_percent','actual_completed_trial_GPU_seconds','current_live_trial')},indent=2))
if __name__=='__main__':main()
