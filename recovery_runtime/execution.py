"""Actual guarded CUDA qualification or frozen-lineage two-parent completion."""
from pathlib import Path
import gc,sys,time
import recovery_scope as scope

def run(auth,out,end,admission,validate_reference):
    import torch
    from calibration import evaluate,average,dump,sha
    from generation import iter_pg
    from ensemble_scope import CFG,OBS
    sys.path[:0]=[str(scope.ROOT/'model'),str(scope.ROOT/'confirmation_runtime')]
    import variant_pg_update as base_update
    import learned_pg_update as learned_update
    from variant_pg_adapter import Context as BaseContext,fit as fit_base
    from learned_pg_adapter import Context as LearnedContext,fit as fit_learned
    from pure_ensemble_generation import iter_ensemble
    from ensemble_confirmation_execution import fit_pair
    from exact_resume import qualify_exact,fit_resume
    p,f,mode=scope.validate_authorization(auth)
    base_update.configure(CFG);learned_update.configure(CFG);learned_update.configure_observation(OBS)
    seed=auth['cell']['seed'];cache=Path(auth['cache_dir'])
    checkpoint=None
    if mode!='seed2':
        assert sha(auth['checkpoint_path'])==p['checkpoint_SHA256']
        checkpoint=torch.load(auth['checkpoint_path'],map_location='cpu',weights_only=False)
    if mode=='qualify':
        ctx=BaseContext(cache,'cuda:0')
        result,state=qualify_exact(ctx,checkpoint,expected_initial=scope.INITIAL,branch_directory=out/'qualification_branches')
        result.update(status='ACTUAL_CUDA_EXACT_INTERRUPTED_STATE_AND_NEXT_UPDATE_BITWISE_VERIFIED',checkpoint_SHA256=p['checkpoint_SHA256'],
                      recovery_protocol_SHA256=scope.sha(scope.PROTOCOL),allocation_id=admission['allocation_id'],training_seed=1,
                      branch_SHA256={str(out/'qualification_branches'/f'branch{i}.pt'):sha(out/'qualification_branches'/f'branch{i}.pt') for i in (0,1)})
        dump(out/'qualification.json',result)
        return
    def release():gc.collect();torch.cuda.empty_cache()
    def phase(index):dump(out/'training_phase.json',{'parent':['PG8','learned_observation'][index],'parent_index':index,'training_seed':seed,
                         'completed_parent_updates':(scope.INITIAL if mode=='resume' else 0) if index==0 else 20000,'new_updates_planned':auth['new_model_updates']})
    def resume_first(ctx,folder,seed,*,end_monotonic):
        return fit_resume(ctx,folder,seed,checkpoint=checkpoint,checkpoint_SHA256=p['checkpoint_SHA256'],expected_initial=scope.INITIAL,end_monotonic=end_monotonic)
    bundles,paths=fit_pair([lambda:BaseContext(cache,'cuda:0'),lambda:LearnedContext(cache,'cuda:0')],[resume_first if mode=='resume' else fit_base,fit_learned],seed,out,end,release,phase)
    parents=[]
    for index,path in enumerate(paths):
        q=torch.load(path,map_location='cpu',weights_only=False);validate_reference(q,auth);assert q['variant_config']==CFG
        if index==1:assert q['learned_observation_config']==OBS
        parents.append({'name':['PG8','learned_observation'][index],'checkpoint_path':str(path),'checkpoint_SHA256':sha(path),'completed_updates':20000})
        del q
    ctx=BaseContext(cache,'cuda:0')
    def deadline():
        if time.monotonic()>=end:raise TimeoutError('Finite original frozen recovery deadline ending')
    def iterator_for(rep):
        def iterator(schedule,c,s):
            stage=int(round(float(c[3])*3));factories={index:(lambda index=index:iter_pg(ctx,bundles[index][1],bundles[index][3],schedule,c,s)) for index in (0,1)}
            yield from iter_ensemble(factories,weight=.25,replicate=rep,stage=stage,check_deadline=deadline)
        return iterator
    path=out/'fit/ensemble.json';dump(path,scope.reference_payload(seed,parents))
    dump(path.parent/'checkpoint_receipt.json',{'path':path.name,'SHA256':sha(path),'status':'completed','completed_updates':40000,'kind':'complete_two_parent_references','initial_completed_updates':auth['initial_completed_updates'],'new_model_updates':auth['new_model_updates']})
    records=[evaluate(ctx,'pg_pure_ensemble75',seed,rep,iterator_for(rep),out/f'r{rep}') for rep in (0,1)]
    result=average(records)
    result.update(status='completed',completed_updates=40000,fixed_endpoint=40000,integer_count_contract_passed=True,
                  checkpoint_path=str(path),checkpoint_SHA256=sha(path),checkpoint_kind='complete_two_parent_references',operation=auth['operation'],allocation_id=admission['allocation_id'],
                  variant_config=CFG,hybrid_config=None,method='pg_pure_ensemble75',pure_only=True,pure_formulation='pg8_learned_ensemble75',real_bag_replacements=0,
                  generation='iter_pg_whole_bag_ensemble',ensemble_config=auth['ensemble_config'],parent_training_updates_total=40000,parent_completed_updates=[20000,20000],
                  effective_active_parent_training_updates=40000,new_model_updates=auth['new_model_updates'],initial_completed_updates=auth['initial_completed_updates'],
                  native_recipe='bispf_optimized',native_recipe_schedule_horizon_each=20000,qualification_only=False,recovery_protocol_SHA256=scope.sha(scope.PROTOCOL),
                  original_confirmation_freeze_SHA256=p['original_confirmation_freeze_SHA256'],original_interrupted_attempt=p['original_attempt'] if mode=='resume' else None,
                  resume_parent_checkpoint_SHA256=p['checkpoint_SHA256'] if mode=='resume' else None,native_restore_proof_SHA256=scope.sha(scope.NATIVE_PROOF),automatic_successor=False)
    dump(out/'training_phase.json',{'parent':'both_completed','training_seed':seed,'completed_parent_updates':40000})
    dump(out/'completed.json',result)
