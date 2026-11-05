"""Audit the completed external comparator without changing any incumbent."""
import argparse,json,os,sys,datetime
from pathlib import Path
ROOT=Path('/ml2/alz');HERE=Path(__file__).resolve().parent;sys.path[:0]=[str(HERE),str(ROOT)]
import binding as b,preset_scope as scope
def finish(attempt):
 assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
 folder=ROOT/'artifacts/allocations'/attempt;dep=b.read(folder/'deployment.json');auth=b.read(dep['authorization_path']);scope.validate_authorization(auth)
 assert b.sha(dep['authorization_path'])==dep['authorization_SHA256'] and attempt==b.attempt_key(scope.CELL)
 sys.path.extend(map(str,b.native_site_roots(scope.CELL,auth['files_SHA256'])));sys.path[:0]=[str(b.SCIENCE),str(b.PROJECT)]
 import numpy as np,torch
 torch.set_num_threads(1);torch.set_num_interop_threads(1)
 from calibration import score,average
 from scvi_adapter import load_final
 from state_fingerprint import fingerprint
 ledger=Path(auth['ledger_path']);job=b.read(ledger)['jobs'][0];settle=b.read(ledger.with_name(attempt+'-settlement.json'))
 assert job['status']=='completed' and job['exit_code']==0 and job['reservation_released'] is True
 assert settle['uncertain'] is None and settle['settlement']['status']=='settled'
 assert not b.read(Path(auth['registry_path'])/'registry.json')['jobs']
 out=Path(auth['output_dir']);candidate=b.read(out/'completed.json')
 assert candidate['completed_epochs']==candidate['fixed_endpoint']==400 and candidate['evaluation_used'] is False
 assert candidate['scvi_architecture']==auth['scvi_architecture'] and candidate['scvi_training_plan']==auth['scvi_training_plan']
 for pathkey,hashkey in [('fit_receipt_path','fit_receipt_SHA256'),('native_reload_path','native_reload_SHA256')]:assert b.sha(candidate[pathkey])==candidate[hashkey]
 from worker_gpu0 import validate_fit
 receipt=b.read(candidate['fit_receipt_path']);validate_fit(receipt,auth)
 model=load_final(auth['cache_dir'],out/'fit',device='cpu')
 assert fingerprint(model.module.state_dict())==b.read(candidate['native_reload_path'])['model_state_fingerprint']
 assert not torch.cuda.is_initialized()
 rows=[];arraypins={}
 for rep in (0,1):
  d=out/f'r{rep}';m=b.read(d/'metrics.json');assert m['arrays_SHA256']==b.sha(d/'arrays.npz')
  q=np.load(d/'arrays.npz',allow_pickle=False);actual=score(q['stage_synthetic_summaries'],q['generated_cells_by_stage'])
  for k,v in actual.items():assert m[k]==v,(rep,k)
  assert m['generated_cells']==320000 and m['training_seed']==0 and m['generation_replicate']==rep
  for stage in range(4):
   path=d/f'stage{stage}_exemplar.npz';x=np.load(path,allow_pickle=False)['counts'];assert x.shape==(1,400,2000)
   assert np.isfinite(x).all() and (x>=0).all() and np.equal(x,np.floor(x)).all();arraypins[str(path)]=b.sha(path)
  rows.append(m);arraypins[str(d/'arrays.npz')]=b.sha(d/'arrays.npz')
 for k,v in average(rows).items():assert candidate[k]==v,k
 assert candidate['comparator_protocol_SHA256']==auth['comparator_protocol_SHA256']==b.sha(scope.PROTOCOL)
 result={'status':'MATCHED_SCVI400_CALIBRATION_INDEPENDENT_AUDIT_PASS','utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'candidate':candidate,'owned_release_and_clean_settlement_verified':True,'native_checkpoint_CPU_reload_matches_saved_CUDA_state':True,'independent_saved_scores_and_count_exemplars_verified':True,'charged_GPU_seconds':settle['settlement']['charged_GPU_seconds'],'authorization_SHA256':b.sha(dep['authorization_path']),'model_files_SHA256':receipt['model_files_SHA256'],'array_files_SHA256':arraypins,'incumbent_changed':False,'SOTA_claim':False,'evaluation_used':False,'scope':'One fixed-recipe external scVI comparator on four adaptive calibration donors. Many prior PG tuning trials disclosed; not equally tuned systems or completed3fold5seed final benchmark.'}
 destination=ROOT/'analysis'/(attempt+'-result.json');assert not destination.exists();destination.write_text(json.dumps(result,indent=2)+'\n')
 print(json.dumps({k:candidate[k] for k in ('energy_score','cell_energy_distance','spread_ratio')},indent=2))
if __name__=='__main__':
 parser=argparse.ArgumentParser();parser.add_argument('--attempt',required=True);finish(parser.parse_args().attempt)
