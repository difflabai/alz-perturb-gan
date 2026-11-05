"""Independent saved-checkpoint/count/score audit of fixed matchedCFGen."""
import argparse,datetime,json,os,sys
from pathlib import Path
ROOT=Path('/ml2/alz');HERE=Path(__file__).resolve().parent;sys.path.insert(0,str(HERE))
import binding as b,preset_scope as scope
def finish(attempt):
 assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
 folder=ROOT/'artifacts/allocations'/attempt;dep=b.read(folder/'deployment.json');auth=b.read(dep['authorization_path']);scope.validate_authorization(auth)
 assert b.sha(dep['authorization_path'])==dep['authorization_SHA256'] and attempt==b.attempt_key(scope.CELL)
 sys.path.extend(map(str,b.native_site_roots(scope.CELL,auth['files_SHA256'])));sys.path[:0]=[str(scope.MODEL_DIR),str(b.SCIENCE),str(ROOT)]
 from fit_adapter import restore,validate_complete,torch
 import numpy as np
 from state_fingerprint import fingerprint
 from contracts import checked_cache
 from calibration import score,average
 ledger=Path(auth['ledger_path']);job=b.read(ledger)['jobs'][0];settle=b.read(ledger.with_name(attempt+'-settlement.json'))
 assert job['status']=='completed' and job['exit_code']==0 and job['reservation_released'] is True
 assert settle['uncertain'] is None and settle['settlement']['status']=='settled'
 assert not b.read(Path(auth['registry_path'])/'registry.json')['jobs']
 out=Path(auth['output_dir']);candidate=b.read(out/'completed.json');receipt=b.read(out/'fit/fit_receipt.json');checkpoint=Path(receipt['FM_checkpoint_path'])
 validate_complete(receipt,auth['cache_dir'],checkpoint);_,fold=checked_cache(auth['cache_dir'],'train')
 assert receipt['training_donors']==fold['optimization_donors']
 assert candidate['AE_epochs']==300 and candidate['FM_epochs']==1500 and candidate['total_optimizer_updates']==candidate['new_model_updates']==102600
 assert candidate['evaluation_used'] is False and candidate['real_bag_replacements']==0 and candidate['query_information']==['stage']
 assert candidate['checkpoint_SHA256']==b.sha(checkpoint) and candidate['fit_receipt_SHA256']==b.sha(out/'fit/fit_receipt.json')
 assert candidate['comparator_protocol_SHA256']==auth['comparator_protocol_SHA256']==b.sha(scope.PROTOCOL)
 for phase,epochs,updates in [('AE',300,17100),('FM',1500,85500)]:
  phase_receipt=b.read(out/'fit'/phase/'receipt.json');assert phase_receipt['completed_epochs']==epochs and phase_receipt['optimizer_updates']==updates
  assert phase_receipt['checkpoint_SHA256']==b.sha(phase_receipt['checkpoint_path'])
  assert receipt[phase+'_receipt_SHA256']==b.sha(out/'fit'/phase/'receipt.json')
  if phase=='FM':
   assert phase_receipt['frozen_encoder_checks'] and all(x['encoder_fingerprint']==receipt['encoder_fingerprint'] for x in phase_receipt['frozen_encoder_checks'])
 ae=torch.load(out/'fit/AE/final.ckpt',map_location='cpu',weights_only=False)
 assert fingerprint(ae['state_dict'])==receipt['encoder_fingerprint']
 model=restore(checkpoint,'cpu');native=b.read(out/'native_reload_receipt.json');assert b.sha(out/'native_reload_receipt.json')==candidate['native_reload_SHA256']
 assert fingerprint(model.state_dict())==native['registered_state_fingerprint']
 assert fingerprint({n:f.state_dict() for n,f in model.feature_embeddings.items()})==native['external_state_fingerprint']
 assert fingerprint(model.encoder_model.state_dict())==receipt['encoder_fingerprint']==native['encoder_fingerprint']
 assert native['registered_tensor_devices']==native['external_tensor_devices']==['cuda:0']
 for v in model.state_dict().values():
  if v.is_floating_point():assert torch.isfinite(v).all()
 rows=[];pins={}
 for rep in (0,1):
  d=out/f'r{rep}';q=b.read(d/'metrics.json');assert q['arrays_SHA256']==b.sha(d/'arrays.npz');arr=np.load(d/'arrays.npz',allow_pickle=False);actual=score(arr['stage_synthetic_summaries'],arr['generated_cells_by_stage'])
  assert all(q[k]==v for k,v in actual.items());assert q['generated_cells']==320000 and q['training_seed']==0 and q['generation_replicate']==rep
  for stage in range(4):
   path=d/f'stage{stage}_exemplar.npz';x=np.load(path,allow_pickle=False)['counts'];assert x.shape==(1,400,2000) and np.isfinite(x).all() and (x>=0).all() and np.equal(x,np.floor(x)).all();pins[str(path)]=b.sha(path)
  rows.append(q);pins[str(d/'arrays.npz')]=b.sha(d/'arrays.npz')
 assert all(candidate[k]==v for k,v in average(rows).items())
 assert not torch.cuda.is_initialized()
 result={'status':'MATCHED_CFGEN300AE1500FM_CALIBRATION_INDEPENDENT_AUDIT_PASS','utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'candidate':candidate,'owned_release_and_clean_settlement_verified':True,'native_CPU_checkpoint_reload_matches_fresh_GPU_state':True,'fixed_AE300_FM1500_endpoints_and_encoder_freeze_verified':True,'independent_saved_scores_and_count_exemplars_verified':True,'charged_GPU_seconds':settle['settlement']['charged_GPU_seconds'],'array_files_SHA256':pins,'authorization_SHA256':b.sha(dep['authorization_path']),'incumbent_changed':False,'SOTA_claim':False,'scope':'One fresh matched nativeCFGen fixedrecipe on33fittingdonors/4adaptivecalibrationdonors,fold0seed0,2generationreplicates. Fixeddecoder/flow architecture and nativecaller adapter/frozenAE disclosed; unequal historicaltuning. No final3fold5seed/evaluation or SOTA claim.'}
 p=ROOT/'analysis'/(attempt+'-result.json');assert not p.exists();p.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:candidate[k] for k in ('energy_score','cell_energy_distance','spread_ratio')},indent=2))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--attempt',required=True);finish(p.parse_args().attempt)
