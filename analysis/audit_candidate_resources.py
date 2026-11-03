"""Attribute sampled health to the exact candidate, excluding stale shared-guard data."""
import argparse,datetime,hashlib,json,math
from pathlib import Path

ROOT=Path('/ml2/alz')

def main():
 parser=argparse.ArgumentParser();parser.add_argument('--attempt',required=True);parser.add_argument('--require-settled',action='store_true');args=parser.parse_args()
 deployment=json.loads((ROOT/'artifacts/allocations'/args.attempt/'deployment.json').read_text())
 ledger=Path(deployment['ledger_path']);ledger_raw=ledger.read_bytes();job=json.loads(ledger_raw)['jobs'][0]
 assert job['label']==args.attempt and type(job['child_pid']) is int and job['child_identity']['pid']==job['child_pid']
 settlement_path=ledger.with_name(args.attempt+'-settlement.json')
 settlement=json.loads(settlement_path.read_text()) if settlement_path.exists() else None
 settled=bool(settlement and settlement['uncertain'] is None and settlement['settlement']['status']=='settled' and job.get('reservation_released'))
 if args.require_settled:assert settled,'Resource audit is not a completed/released allocation'
 start=job['started_unix'];end=settlement['checked_unix'] if settled else float('inf')
 history=ROOT/'artifacts/monitor'/args.attempt/'history.jsonl';raw=history.read_bytes();samples={};discarded={};partial=False
 for index,line in enumerate(raw.splitlines()):
  try:q=json.loads(line)
  except json.JSONDecodeError:
   assert index==len(raw.splitlines())-1 and not raw.endswith(b'\n');partial=True;continue
  assert q['attempt']==args.attempt
  h=q.get('health')
  reason=None
  if not h:reason='missing_health'
  elif h['checked_unix']<start:reason='health_precedes_this_allocation'
  elif h['checked_unix']>end:reason='health_follows_settlement'
  elif job['child_pid'] not in h['members']:reason='exact_worker_not_in_health_members'
  if reason:discarded[reason]=discarded.get(reason,0)+1;continue
  assert all(type(h[k]) is int and h[k]>=0 for k in ['own_gpu_bytes','own_ram_bytes'])
  assert math.isfinite(h['checked_unix'])
  samples[h['checked_unix']]=h
 assert samples,'No health sample can be attributed to this allocation'
 values=list(samples.values())
 result={'status':'SETTLED_CANDIDATE_SAMPLED_RESOURCE_AUDIT' if settled else 'LIVE_CANDIDATE_SAMPLED_RESOURCE_AUDIT',
         'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'attempt':args.attempt,'candidate_commit':deployment['candidate_commit'],
         'exact_worker_identity':job['child_identity'],'started_unix':start,'settled_and_owned_release_verified':settled,
         'unique_attributable_health_samples':len(values),'discarded_monitor_records':discarded,'ignored_partial_final_line':partial,
         'sampled_GPU_peak_bytes':max(q['own_gpu_bytes'] for q in values),'sampled_host_RAM_peak_bytes':max(q['own_ram_bytes'] for q in values),
         'any_own_GPU_budget_exceeded':any(q['own_gpu_budget_exceeded'] for q in values),
         'any_own_RAM_reservation_exceeded':any(q['own_ram_above_reservation'] for q in values),
         'all_sampled_host_growth_forecasts_fit':all(q['host_growth_forecast_fits'] for q in values),
         'source_SHA256':{str(Path(__file__).resolve()):hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),str(history):hashlib.sha256(raw).hexdigest(),str(ledger):hashlib.sha256(ledger_raw).hexdigest()},
         'limits':'Discrete monitor samples, not a continuous hardware peak measurement. Raw history is retained unchanged. Attribution requires time within this job and its exact child PID in the guard membership; old shared health is excluded. No process control or GPU imports.'}
 path=ROOT/'analysis'/(args.attempt+'-resources'+('' if settled else '-live')+'.json')
 path.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n');print(json.dumps({k:v for k,v in result.items() if k not in ['source_SHA256','limits']},indent=2))

if __name__=='__main__':main()
