"""Cost-only, capacity-aware placement of six fresh scalar-calibration fits."""
import hashlib,json,math,statistics,sys
from pathlib import Path
ROOT=Path('/ml2/alz');sys.path.insert(0,str(ROOT/'analysis'))
from released097_remaining_runtime_balance import balance_counts
from dual_coordination import GPUS,valid_cell

def choose_next(pending,active,gpu,costs,*,eligible=None):
 assert gpu in GPUS and gpu not in active
 if not pending:return None,dict(reason='queue_empty')
 eligible=set(GPUS if eligible is None else eligible);assert eligible<=set(GPUS)
 if gpu not in eligible:return None,dict(reason='card_has_no_verified_capacity')
 initial=[0.,0.]
 for card,h in active.items():initial[GPUS.index(card)]=max(0.,costs['pg'][GPUS.index(card)]-h.get('elapsed_seconds',0.))
 if len(eligible)==1:
  counts=[0,0];counts[GPUS.index(gpu)]=len(pending)
  proposal=dict(counts_per_lane={'pg':counts},predicted_lane_seconds=[initial[i]+counts[i]*costs['pg'][i] for i in (0,1)],reason='only_card_with_current_capacity')
  selected=pending[0];valid_cell(selected);return selected,proposal
 proposal=balance_counts({'pg':len(pending)},costs,initial=initial)
 if proposal['counts_per_lane']['pg'][GPUS.index(gpu)]==0:return None,proposal
 selected=pending[0];valid_cell(selected);return selected,proposal

def verified_replaced_capacity_failure(root,path,job):
 """One declared failed PG attempt remains charged; use its fresh replacement for cost."""
 if path.stem!='CAL_GPU0_V1_external097_f2_pg_s1':return False
 def read(p):return json.loads(Path(p).read_bytes())
 def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
 matrix=read(root/'analysis/benchmark097/matrix_complete.json')
 amendment=read(root/'analysis/benchmark097_infra1/protocol_v2.json')
 replacement='CAL_GPU0_V1_external097infra1_f2_pg_s1';result=root/'analysis/benchmark097_infra1/results'/(replacement+'.json')
 assert matrix['matrix_complete'] and matrix['complete_cells']==matrix['required_cells']==45 and matrix['physical_attempts']==47 and len(matrix['result_SHA256'])==45
 assert matrix['logical_to_physical_attempt']['f2_pg_s1']==replacement
 assert matrix['failed_result_SHA256'][path.stem]==amendment['known_failed_attempts'][path.stem]==sha(path)
 assert matrix['result_SHA256'][replacement]==sha(result)
 assert job['status']=='failed' and 'Protected/registered GPU growth no longer fits' in job['error']
 return True

def settled_cost_rows(root=ROOT):
 """Parse authorization/ledger metadata, never result quality JSON or arrays."""
 root=Path(root);rows=[]
 for phase in ('benchmark097','benchmark097_infra1','scalarcal097'):
  for path in sorted((root/'analysis'/phase/'results').glob('*.json')):
   folder=root/'artifacts/allocations'/path.stem;dep=json.loads((folder/'deployment.json').read_text());a=json.loads(Path(dep['authorization_path']).read_text())
   if a['cell']['method']!='pg':continue
   assert a['fixed_endpoint']==a['new_model_updates']==20000 and a['initial_completed_updates']==0 and a['source_model_path'] is None
   ledger=Path(a['ledger_path']);job=json.loads(ledger.read_text())['jobs'][0];closed=json.loads(ledger.with_name(path.stem+'-settlement.json').read_text())
   if not job['reservation_released'] or closed['uncertain'] is not None or closed['settlement']['status']!='settled':raise ValueError('Incomplete PG cost source retained; never select or retry by quality')
   if job['status']!='completed' or job['exit_code']!=0:
    if phase=='benchmark097' and verified_replaced_capacity_failure(root,path,job):continue
    raise ValueError('Incomplete PG cost source retained; never select or retry by quality')
   seconds=closed['settlement']['charged_GPU_seconds'];assert type(seconds) in (int,float) and math.isfinite(seconds) and seconds>0
   rows.append(dict(gpu_uuid=a['gpu_uuid'],seconds=seconds,checked_unix=closed['checked_unix'],attempt=path.stem,source_phase=phase,effective_seconds=seconds*(1.15 if phase in ('benchmark097','benchmark097_infra1') else 1.)))
 return rows

def cost_table(rows):
 values=[];evidence=[]
 for gpu in GPUS:
  selected=sorted([r for r in rows if r['gpu_uuid']==gpu],key=lambda r:r['checked_unix'])[-3:]
  if not selected:raise ValueError('At least one measured full20k PG cost sample per card required')
  values.append(statistics.median(r['effective_seconds'] for r in selected));evidence.append([r['attempt'] for r in selected])
 return {'pg':values},{'pg':evidence}
