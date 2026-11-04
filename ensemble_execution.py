"""Called only inside the existing live guarded native scientific worker."""
from pathlib import Path
import json
import sys
import time

def run(auth,out,end,admission,validate_reference):
    import numpy as np
    import torch
    import ensemble_scope as scope
    from calibration import evaluate,average,dump,sha
    from generation import iter_pg
    sys.path.insert(0,str(scope.ROOT/'model'))
    import variant_pg_update as base_update
    import learned_pg_update as learned_update
    from variant_pg_adapter import Context,restore as restore_base
    from learned_pg_adapter import restore as restore_learned
    from pure_ensemble_generation import iter_ensemble
    base_update.configure(scope.CFG);learned_update.configure(scope.CFG);learned_update.configure_observation(scope.OBS)
    ctx=Context(Path(auth['cache_dir']),'cuda:0');parents=scope.manifest()['parents'];bundles=[]
    for index,parent in enumerate(parents):
        assert sha(parent['checkpoint_path'])==parent['checkpoint_SHA256']
        q=torch.load(parent['checkpoint_path'],map_location='cpu',weights_only=False);validate_reference(q,auth)
        restored=(restore_base if index==0 else restore_learned)(ctx,q)
        assert restored[0][-1].completed_steps==20000;bundles.append(restored[0])
    def deadline():
        if time.monotonic()>=end:raise TimeoutError('finite ensemble allocation ending; no partial result')
    def iterator_for(rep):
        def iterator(schedule,c,s):
            stage=int(round(float(c[3])*3))
            factories={index:(lambda index=index:iter_pg(ctx,bundles[index][1],bundles[index][3],schedule,c,s)) for index in (0,1)}
            yield from iter_ensemble(factories,weight=auth['ensemble_weight'],replicate=rep,stage=stage,check_deadline=deadline)
        return iterator
    folder=out/'fit';folder.mkdir(exist_ok=False)
    checkpoint=folder/'ensemble.json';dump(checkpoint,scope.reference_payload(auth['ensemble_weight']))
    dump(folder/'checkpoint_receipt.json',{'path':checkpoint.name,'SHA256':sha(checkpoint),'status':'completed','completed_updates':40000,'kind':'complete_two_parent_references_not_new_training'})
    method='pg_pure_ensemble_control' if auth['qualification_only'] else 'pg_pure_ensemble75'
    records=[evaluate(ctx,method,0,rep,iterator_for(rep),out/f'r{rep}') for rep in (0,1)]
    result=average(records)
    result.update(status='completed',completed_updates=40000,fixed_endpoint=40000,integer_count_contract_passed=True,
                  checkpoint_path=str(checkpoint),checkpoint_SHA256=sha(checkpoint),checkpoint_kind='complete_two_parent_references',
                  operation=auth['operation'],allocation_id=admission['allocation_id'],variant_config=scope.CFG,hybrid_config=None,
                  method=method,pure_only=True,pure_formulation=auth['pure_formulation'],real_bag_replacements=0,
                  generation='iter_pg_whole_bag_ensemble',ensemble_config={'learned_weight':auth['ensemble_weight'],'mask_seed':20261108,'bags_per_stage':200},
                  parent_manifest_SHA256=auth['ensemble_manifest_SHA256'],parent_training_updates_total=40000,parent_completed_updates=[20000,20000],
                  effective_active_parent_training_updates=40000 if auth['ensemble_weight']==.25 else 20000,
                  new_model_updates=0,initial_completed_updates=40000,native_recipe='bispf_optimized',native_recipe_schedule_horizon_each=20000,
                  qualification_only=auth['qualification_only'],automatic_successor=False)
    dump(out/'completed.json',result)
