"""Freeze a verified pure seed0 winner for fixed seeds1/2; never allocate a GPU."""
import argparse,datetime,hashlib,json,math,re,subprocess
from pathlib import Path
from aggregate_validation import verified_deployment,scientific_sources

ROOT=Path('/ml2/alz')
PROTOCOL=ROOT/'analysis/pure_confirmation_protocol.json'

def read(path):return json.loads(Path(path).read_bytes())
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def confirmation_identities(seed0_attempt,checkpoint_sha256):
 if not isinstance(seed0_attempt,str) or re.fullmatch(r'CAL_GPU0_V1_[a-z][a-z0-9_]{0,48}_f0_pg_s0',seed0_attempt) is None:
  raise ValueError('Exact first-fold seed0 attempt identity required')
 if not isinstance(checkpoint_sha256,str) or re.fullmatch(r'[0-9a-f]{64}',checkpoint_sha256) is None:
  raise ValueError('Complete selected checkpoint SHA256 required')
 identity=json.dumps({'seed0_attempt':seed0_attempt,'checkpoint_SHA256':checkpoint_sha256},sort_keys=True).encode()
 experiment='pure_confirm_'+hashlib.sha256(identity).hexdigest()[:20]
 return {str(seed):{'cell':{'fold':0,'method':'pg','seed':seed,'experiment':experiment},
                    'attempt_id':f'CAL_GPU0_V1_{experiment}_f0_pg_s{seed}',
                    'allocation_id':f'ALZ_CAL_V1_{experiment}_f0_pg_s{seed}'} for seed in (1,2)}

def seed0_gate(candidate,baseline):
 reasons=[]
 for key in ['metric_role','donors','training_seed','generation_replicates','calibration_manifest_SHA256','trace_manifest_SHA256','sampling_contract']:
  if candidate.get(key)!=baseline.get(key):reasons.append('unmatched baseline: '+key)
 if candidate.get('metric_role')!='calibration_selection' or candidate.get('generation_replicates')!=2:reasons.append('registered calibration role and two replicates required')
 if candidate.get('training_seed')!=0:reasons.append('seed0 selection required')
 if candidate.get('pure_only') is not True or candidate.get('real_bag_replacements')!=0 or candidate.get('hybrid_config') is not None:reasons.append('pure generation required')
 if candidate.get('completed_updates')!=candidate.get('fixed_endpoint') or candidate.get('completed_updates') not in (20000,40000):reasons.append('fixed endpoint incomplete')
 if candidate.get('integer_count_contract_passed') is not True:reasons.append('count validity missing')
 values=[candidate.get(k) for k in ['energy_score','cell_energy_distance','spread_ratio']]+[baseline.get(k) for k in ['energy_score','cell_energy_distance']]
 if any(type(v) not in (float,int) or not math.isfinite(v) for v in values) or baseline.get('energy_score',0)<=0:
  return {'ready':False,'reasons':reasons+['invalid/nonfinite candidate or baseline metric'],'seed0_ES_relative_improvement':None}
 improvement=1-candidate['energy_score']/baseline['energy_score']
 if not improvement>=.05:reasons.append('seed0 ES improvement is below5percent')
 if not candidate['cell_energy_distance']<=baseline['cell_energy_distance']:reasons.append('seed0 cell distance exceeds matched resampling')
 if not .7<=candidate['spread_ratio']<=1.3:reasons.append('seed0 spread is outside registered band')
 return {'ready':not reasons,'reasons':reasons,'seed0_ES_relative_improvement':improvement}

def main():
 parser=argparse.ArgumentParser();parser.add_argument('--check-only',action='store_true');parser.add_argument('--output',type=Path);args=parser.parse_args()
 assert args.check_only != bool(args.output),'Choose read-only readiness check OR a new freeze output'
 selected_path=ROOT/'pure_incumbent.json';selected=read(selected_path);baseline0=read(ROOT/'analysis/baseline_calibration_seed0.json')
 gate=seed0_gate(selected,baseline0)
 if args.check_only:
  print(json.dumps({'status':'PURE_SEED0_TARGET_READY' if gate['ready'] else 'PURE_SEED0_TARGET_NOT_READY',**gate,'scope':'Target readiness only. No freeze, allocation, seed training or confirmation claim.'},indent=2));return
 assert gate['ready'],gate['reasons']
 assert not args.output.exists(),'Preserve the existing frozen protocol'
 assert not subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True),'Commit complete confirmation implementation before freezing'
 matches=[p.parent.name for p in (ROOT/'artifacts/allocations').glob('*/deployment.json') if read(p)['allocation_id']==selected['allocation_id']]
 assert len(matches)==1,'Selected seed0 allocation is ambiguous'
 attempt=matches[0];deployment,auth,candidate=verified_deployment(attempt)
 assert candidate=={k:v for k,v in selected.items() if k!='candidate_commit'}
 assert candidate.get('qualification_only') is False,'A replay-only selection requires its explicit verified fitting-lineage bridge'
 decision_path=ROOT/'analysis'/(attempt+'-result.json');decision=read(decision_path)
 assert decision['decision']['accept'] and decision['independent_saved_scores_checkpoint_and_count_exemplars_verified']
 assert decision['candidate']==candidate
 protocol=read(PROTOCOL);assert protocol['training_seeds']==[0,1,2] and protocol['ranking_seeds_allowed'] is False
 baselines={}
 for seed in (0,1,2):
  path=ROOT/'analysis'/f'baseline_calibration_seed{seed}.json';baseline=read(path)
  assert baseline['training_seed']==seed
  for key in ['metric_role','donors','generation_replicates','calibration_manifest_SHA256','trace_manifest_SHA256','sampling_contract']:assert baseline[key]==baseline0[key]
  baselines[str(seed)]={'path':str(path),'SHA256':sha(path)}
 assert baseline0['both_original_seed0_arrays_bitwise_reproduced']
 sources=scientific_sources(auth)
 assert sources and all(sha(path)==pin for path,pin in sources.items()),'Selected model/scorer sources changed; qualify exact parity before confirmation'
 assert sha(ROOT/'data/calibration_fold0/manifest.json')==candidate['calibration_manifest_SHA256']
 result={'status':'FROZEN_PURE_THREE_SEED_CALIBRATION_CONFIRMATION','utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
         'git_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
         'seed0_attempt':attempt,'seed0_candidate':candidate,'pure_incumbent_SHA256':sha(selected_path),
         'seed0_decision_SHA256':sha(decision_path),'seed0_authorization_SHA256':sha(deployment['authorization_path']),
         'pure_confirmation_protocol_SHA256':sha(PROTOCOL),'training_seeds':[0,1,2],'confirmation_training_seeds':[1,2],
         'confirmation_identities':confirmation_identities(attempt,candidate['checkpoint_SHA256']),
         'scientific_source_SHA256':sources,'matched_baselines':baselines,'ranking_seeds_allowed':False,
         'native_allocation_created':False,'confirmation_complete':False,
         'scope':'Frozen same-model calibration seed confirmation only. Runtime admission must separately bind this freeze and exact seed1/2 cells; this file does not authorize an unreviewed backend or expand existing device scope.'}
 args.output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n');print(json.dumps({k:result[k] for k in ['status','seed0_attempt','confirmation_training_seeds','native_allocation_created','confirmation_complete']},indent=2))

if __name__=='__main__':main()
