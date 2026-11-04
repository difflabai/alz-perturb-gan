"""Two unchanged parent fits from scratch, then frozen pure ensemble generation."""
from pathlib import Path
import gc,sys,time
import ensemble_confirmation_scope as scope

def fit_pair(factories,fitters,seed,out,end,release,phase):
    """Each adapter owns its fitting Context. Never reuse pooled features as raw PCs."""
    assert type(seed) is int and seed in (1,2) and len(factories)==len(fitters)==2
    bundles=[];paths=[]
    for index in (0,1):
        phase(index)
        ctx=factories[index]()
        bundle,path=fitters[index](ctx,Path(out)/'fit'/['pg8','learned'][index],seed,end_monotonic=end)
        assert bundle[-1].completed_steps==20000,'Do not evaluate a partial parent'
        bundles.append(bundle);paths.append(path)
        del ctx
        release()
    return bundles,paths

def run(auth,out,end,admission,validate_reference):
    import torch
    from calibration import evaluate,average,dump,sha
    from generation import iter_pg
    from ensemble_scope import CFG,OBS
    sys.path.insert(0,str(scope.ROOT/'model'))
    import variant_pg_update as base_update
    import learned_pg_update as learned_update
    from variant_pg_adapter import Context as BaseContext,fit as fit_base
    from learned_pg_adapter import Context as LearnedContext,fit as fit_learned
    from pure_ensemble_generation import iter_ensemble
    base_update.configure(CFG);learned_update.configure(CFG);learned_update.configure_observation(OBS)
    seed=auth['cell']['seed'];cache=Path(auth['cache_dir'])
    def release():gc.collect();torch.cuda.empty_cache()
    def phase(index):dump(out/'training_phase.json',{'parent':['PG8','learned_observation'][index],'parent_index':index,'training_seed':seed,'completed_parent_updates':index*20000,'new_updates_planned':40000})
    bundles,paths=fit_pair([lambda:BaseContext(cache,'cuda:0'),lambda:LearnedContext(cache,'cuda:0')],[fit_base,fit_learned],seed,out,end,release,phase)
    parents=[]
    for index,path in enumerate(paths):
        q=torch.load(path,map_location='cpu',weights_only=False);validate_reference(q,auth)
        assert q['variant_config']==CFG
        if index==1:assert q['learned_observation_config']==OBS
        parents.append({'name':['PG8','learned_observation'][index],'checkpoint_path':str(path),'checkpoint_SHA256':sha(path),'completed_updates':20000})
        del q
    ctx=BaseContext(cache,'cuda:0')
    def deadline():
        if time.monotonic()>=end:raise TimeoutError('finite confirmation allocation ending; no partial metric')
    def iterator_for(rep):
        def iterator(schedule,c,s):
            stage=int(round(float(c[3])*3))
            factories={index:(lambda index=index:iter_pg(ctx,bundles[index][1],bundles[index][3],schedule,c,s)) for index in (0,1)}
            yield from iter_ensemble(factories,weight=.25,replicate=rep,stage=stage,check_deadline=deadline)
        return iterator
    checkpoint=out/'fit/ensemble.json';dump(checkpoint,scope.reference_payload(seed,parents))
    dump(checkpoint.parent/'checkpoint_receipt.json',{'path':checkpoint.name,'SHA256':sha(checkpoint),'status':'completed','completed_updates':40000,'kind':'complete_two_parent_references_not_single_model'})
    records=[evaluate(ctx,'pg_pure_ensemble75',seed,rep,iterator_for(rep),out/f'r{rep}') for rep in (0,1)]
    result=average(records)
    result.update(status='completed',completed_updates=40000,fixed_endpoint=40000,integer_count_contract_passed=True,
                  checkpoint_path=str(checkpoint),checkpoint_SHA256=sha(checkpoint),checkpoint_kind='complete_two_parent_references',
                  operation=auth['operation'],allocation_id=admission['allocation_id'],variant_config=CFG,hybrid_config=None,
                  method='pg_pure_ensemble75',pure_only=True,pure_formulation='pg8_learned_ensemble75',real_bag_replacements=0,
                  generation='iter_pg_whole_bag_ensemble',ensemble_config=auth['ensemble_config'],
                  parent_training_updates_total=40000,parent_completed_updates=[20000,20000],effective_active_parent_training_updates=40000,
                  new_model_updates=40000,initial_completed_updates=0,native_recipe='bispf_optimized',native_recipe_schedule_horizon_each=20000,
                  qualification_only=False,confirmation_freeze_SHA256=scope.sha(scope.FREEZE),automatic_successor=False)
    dump(out/'training_phase.json',{'parent':'both_completed','training_seed':seed,'completed_parent_updates':40000})
    dump(out/'completed.json',result)
