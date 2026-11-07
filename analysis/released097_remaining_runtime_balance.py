"""Metadata-only allocation forecasts; never an authorization or model selector."""
from itertools import product
import statistics

def balance_counts(counts,costs,*,initial=(0.,0.)):
 """Exact integer count allocation for two cards and method-specific costs."""
 if set(counts)!=set(costs) or len(initial)!=2 or any(v<0 for v in initial):raise ValueError('matching methods and two nonnegative lane times required')
 methods=sorted(counts)
 for m in methods:
  if type(counts[m]) is not int or counts[m]<0 or len(costs[m])!=2 or any(not isinstance(v,(int,float)) or not 0<v<float('inf') for v in costs[m]):raise ValueError('nonnegative counts and finite positive method costs required')
 candidates=[]
 for n0 in product(*(range(counts[m]+1) for m in methods)):
  loads=[initial[0]+sum(n*costs[m][0] for n,m in zip(n0,methods)),initial[1]+sum((counts[m]-n)*costs[m][1] for n,m in zip(n0,methods))]
  candidates.append((max(loads),sum(loads),n0,loads))
 _,_,n0,loads=min(candidates)
 return dict(counts_per_lane={m:[n,counts[m]-n] for m,n in zip(methods,n0)},predicted_lane_seconds=loads,predicted_makespan_seconds=max(loads),conditional_on_cost_assumptions=True,not_a_native_admission=True)

def measured_costs(rows,gpus):
 if len(gpus)!=2:raise ValueError('two physical cards required')
 costs={};samples={}
 for method in ('pg','scvi','cfgen'):
  costs[method]=[];samples[method]=[]
  for gpu in gpus:
   values=[r['charged_GPU_seconds'] for r in rows if r['method']==method and r['gpu_uuid']==gpu]
   if not values:raise ValueError('no same-card method samples; do not infer speed from quality')
   costs[method].append(statistics.median(values));samples[method].append(len(values))
 return costs,samples
