"""Resolve documented EMA/external-cap requirements without disabling controls.

Follow-up to the exact existing-loop ABI probe. CPU initialization only; neither
successful construction nor native refusal is an ALZ quality measurement.
"""
import copy,datetime,json,os,sys
from pathlib import Path
ROOT=Path('/ml2/alz');sys.path.insert(0,str(ROOT/'analysis'))
import check_published_recipe_construction as base

def main():
    assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
    base.torch.set_num_threads(1);rows=[]
    for name in ('ka2','k3p','e22','atlas','e22_routed'):
        continuous=name in ('e22','atlas','e22_routed')
        row={'published_preset':name,'total_steps':None if continuous else 20000,'external_execution_limit':20000,'critic_EMA_supplied':True,'row_semantics':'conditional','optimizer_pairs_executed':0}
        try:
            rec=base.get_recipe(name,total_steps=None if continuous else 20000,batch_size=32,num_particles=512,z_dim=8)
            row['resolved_recipe']=rec.to_dict()
            base.torch.manual_seed(0)
            G=base.CountGenerator(base.torch.ones(2000),base.torch.ones(2000),zd=8,zc=8)
            D=base.VectorCritic(150,h=128);ema=copy.deepcopy(D).requires_grad_(False).eval()
            P=rec.make_prior(generator=base.torch.Generator().manual_seed(0))
            og,od=rec.make_optimizers(G,D,P,ema_critic=ema)
            pen=rec.make_critic_penalty(od);loss=rec.make_loss();reg=rec.make_prior_regularizer(prior=P,weight=1.)
            policy=base.UpdatePolicy(rec,G,D,prior=P,generator_optimizer=og,critic_optimizer=od,penalty=pen,seed=0,row_semantics='conditional')
            assert policy.completed_steps==0
            row.update(status='NATIVE_REQUIREMENTS_COMPONENT_CONSTRUCTION_SUCCEEDED_NO_UPDATE_QUALIFICATION',completed_updates=0,
                       native_component_classes={k:type(v).__module__+'.'+type(v).__name__ for k,v in {'G_optimizer':og,'D_optimizer':od,'penalty':pen,'loss':loss,'regularizer':reg,'policy':policy}.items()})
        except Exception as exc:
            row.update(status='NATIVE_COMPONENT_REQUIREMENT_REFUSED',exception_type=type(exc).__name__,reason=str(exc))
        rows.append(row)
    assert not base.torch.cuda.is_initialized()
    prior=ROOT/'analysis/published_recipe_construction_inventory.json'
    result={'status':'CPU_NATIVE_PRESET_REQUIREMENTS_CHECK_NOT_TRAINING_OR_QUALITY','utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
            'existing_loop_inventory_path':str(prior),'existing_loop_inventory_SHA256':base.sha(prior),'rows':rows,
            'row_controls_disabled':False,'RoutedRows_contract_invented':False,'CUDA_initialized':False,'GPU_allocation_created':False,
            'scope':'Native required critic EMA supplied and continuous schedule horizon left None with a planned external20000 cap. Existing ALZ conditional row semantics preserved. No forward/backward/update, no data access, no new trained comparison. Successful construction needs preset-specific EMA/noise/serving/update and checkpoint qualification before any native trial.',
            'source_SHA256':{str(p):base.sha(p) for p in [Path(__file__).resolve(),Path(base.__file__).resolve(),base.SITE/'particlegan/recipes.py',base.SITE/'particlegan/policy.py']}}
    out=ROOT/'analysis/native_preset_requirements_inventory.json';assert not out.exists();out.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps([{k:v for k,v in row.items() if k in ('published_preset','status','reason')} for row in rows],indent=2))

if __name__=='__main__':main()
