"""Quality-blind dispatch using only settled method/card wall costs."""
import json,math,statistics,sys,time
from pathlib import Path
ROOT=Path('/ml2/alz');sys.path.insert(0,str(ROOT/'analysis'))
from released097_remaining_runtime_balance import balance_counts
from dual_coordination import GPUS,valid_cell

def choose_next(pending,active,gpu,costs):
 assert gpu in GPUS and gpu not in active
 counts={m:sum(v['method']==m for v in pending) for m in ('pg','scvi','cfgen')}
 initial=[0.,0.]
 for card,h in active.items():
  estimate=costs[h['cell']['method']][GPUS.index(card)]
  initial[GPUS.index(card)]=max(0.,estimate-h.get('elapsed_seconds',0.))
 allocation=balance_counts(counts,costs,initial=initial)
 index=GPUS.index(gpu);methods=[m for m in counts if allocation['counts_per_lane'][m][index]>0]
 if not methods:return None,allocation
 method=max(methods,key=lambda m:(costs[m][index],m))
 selected=next(v for v in pending if v['method']==method);valid_cell(selected)
 return selected,allocation

def settled_cost_rows(root=ROOT):
 """Do not parse result JSON: completion is backed by immutable ledger closure."""
 root=Path(root);rows=[]
 for path in sorted((root/'analysis/benchmark097/results').glob('*.json')):
  folder=root/'artifacts/allocations'/path.stem
  dep=json.loads((folder/'deployment.json').read_text());a=json.loads(Path(dep['authorization_path']).read_text())
  ledger=Path(a['ledger_path']);job=json.loads(ledger.read_text())['jobs'][0];closed=json.loads(ledger.with_name(path.stem+'-settlement.json').read_text())
  if job['status']!='completed' or job['exit_code']!=0 or not job['reservation_released'] or closed['uncertain'] is not None or closed['settlement']['status']!='settled':raise ValueError('Incomplete/uncertain original scientific cell cannot be retried or silently skipped')
  value=closed['settlement']['charged_GPU_seconds'];assert type(value) in (int,float) and math.isfinite(value) and value>0
  rows.append(dict(method=a['cell']['method'],gpu_uuid=a['gpu_uuid'],seconds=value,checked_unix=closed['checked_unix'],attempt=path.stem))
 return rows

def cost_table(rows):
 costs={};evidence={}
 for method in ('pg','scvi','cfgen'):
  costs[method]=[];evidence[method]=[]
  for gpu in GPUS:
   selected=sorted([r for r in rows if r['method']==method and r['gpu_uuid']==gpu],key=lambda r:r['checked_unix'])[-3:]
   if not selected:raise ValueError('At least one measured full-endpoint sample on each method/card required')
   costs[method].append(statistics.median(r['seconds'] for r in selected));evidence[method].append([r['attempt'] for r in selected])
 return costs,evidence
