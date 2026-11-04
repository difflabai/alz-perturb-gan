"""CPU-only construction of published presets with the current ALZ model ABI.

Construction is not training, update-loop qualification, count serving validation
or quality evidence. Incompatible native requirements are retained as failures;
no row control or other native mechanism is disabled to force compatibility.
"""
import datetime,hashlib,inspect,json,os,sys
from pathlib import Path
ROOT=Path('/ml2/alz')
B=Path('/ml2/codex-storage/future-workspaces/2026-10-27/task-2/benchmark')
S=Path('/ml2/codex-storage/future-workspaces/2026-11-03/task-2/alz-audit/core-comparison-20261103')
SITE=B/'current_base_20261029/runtime/lib/python3.12/site-packages'
sys.path[:0]=list(map(str,[ROOT/'model',ROOT,S,B,SITE,B/'.venv312/lib/python3.12/site-packages']))
assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
import torch
import particlegan
from particlegan import get_recipe,UpdatePolicy
from pg_model import CountGenerator
from critic import VectorCritic

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def main():
    torch.set_num_threads(1)
    assert not torch.cuda.is_initialized()
    leader=json.loads((ROOT/'analysis/pure_formulations_leaderboard.json').read_text())
    coverage=set(leader['native_recipe_families_tested']);rows=[]
    for name in ('ka2','k3p','bcap_adam','bcap','bispf','bispf_optimized','e22','atlas','e22_routed','halloween'):
        row={'published_preset':name,'ALZ_native_quality_comparison_completed':name in coverage,'optimizer_pairs_executed':0}
        try:
            rec=get_recipe(name,total_steps=20000,batch_size=32,num_particles=512,z_dim=8)
            row['resolved_recipe']=rec.to_dict()
            torch.manual_seed(0)
            G=CountGenerator(torch.ones(2000),torch.ones(2000),zd=8,zc=8)
            D=VectorCritic(150,h=128)
            P=rec.make_prior(generator=torch.Generator().manual_seed(0))
            og,od=rec.make_optimizers(G,D,P)
            pen=rec.make_critic_penalty(od);loss=rec.make_loss()
            reg=rec.make_prior_regularizer(prior=P,weight=1.)
            policy=UpdatePolicy(rec,G,D,prior=P,generator_optimizer=og,critic_optimizer=od,penalty=pen,seed=0,row_semantics='conditional')
            row.update(status='COMPONENT_CONSTRUCTION_SUCCEEDED_NO_UPDATE_QUALIFICATION',
                       native_component_classes={k:type(v).__module__+'.'+type(v).__name__ for k,v in {'prior':P,'G_optimizer':og,'D_optimizer':od,'penalty':pen,'loss':loss,'regularizer':reg,'policy':policy}.items()},
                       conditional_row_semantics=policy.row_semantics,completed_updates=policy.completed_steps)
            assert policy.completed_steps==0
            del policy,reg,loss,pen,og,od,P,D,G
        except Exception as exc:
            row.update(status='NATIVE_COMPONENT_REQUIREMENT_REFUSED',exception_type=type(exc).__name__,reason=str(exc))
        rows.append(row)
    assert not torch.cuda.is_initialized()
    result={'status':'CPU_PUBLISHED_PRESET_CONSTRUCTION_INVENTORY_NOT_TRAINING_OR_QUALITY','utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
            'task_overrides':{'total_steps':20000,'batch_size':32,'num_particles':512,'z_dim':8,'row_semantics':'conditional'},
            'model_initialization':'Synthetic all-one2000gene mean/dispersion; actual ALZ CountGenerator and VectorCritic classes on CPU only. No fitting or calibration/evaluation data.',
            'rows':rows,'completed_native_recipe_family_quality_coverage':sorted(coverage),'CUDA_initialized':False,'GPU_allocation_created':False,'native_sources_or_controls_modified':False,
            'scope':'Native factory/policy initialization only. A successful construction still needs correct preset-specific update ordering, noise/EMA/serving/regularizer integration, synthetic replay/gradient checks, guarded native qualification, fixed-count training and full quality audit. A refusal is not a failed trained model. No incompatible native mechanism silently disabled.',
            'source_SHA256':{str(p):sha(p) for p in [Path(__file__).resolve(),SITE/'particlegan/recipes.py',SITE/'particlegan/policy.py',Path(inspect.getfile(CountGenerator)),Path(inspect.getfile(VectorCritic)),ROOT/'analysis/pure_formulations_leaderboard.json']}}
    out=ROOT/'analysis/published_recipe_construction_inventory.json';assert not out.exists();out.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps([{k:v for k,v in row.items() if k in ('published_preset','status','reason','ALZ_native_quality_comparison_completed')} for row in rows],indent=2))

if __name__=='__main__':main()
