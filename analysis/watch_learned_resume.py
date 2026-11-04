"""Read-only, finite observer of one existing learned continuation; no launch/control."""
import argparse,datetime,json,time
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--attempt',required=True);args=p.parse_args();root=Path('/ml2/alz')
d=json.loads((root/'artifacts/allocations'/args.attempt/'deployment.json').read_text());out=Path(d['output_dir']);ledger=Path(d['ledger_path']);folder=root/'artifacts/monitor'/args.attempt;folder.mkdir(parents=True,exist_ok=False)
def read(p):
 try:return json.loads(p.read_text())
 except FileNotFoundError:return None
while time.time()<d['expires_unix']+90:
 state=read(ledger);job=(state or {}).get('jobs',[{}])[0];progress=read(out/'fit/learned/progress.json');settlement=read(ledger.with_name(args.attempt+'-settlement.json'));health=read(ledger.parent/'cooperative_live.json')
 q={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'attempt':args.attempt,'gpu_uuid':(state or {}).get('gpu_uuid'),'job':job,'retained_PG8_updates':20000,'learned_progress':progress,'learned_checkpoint':read(out/'fit/learned/checkpoint_receipt.json'),'quality_result':read(out/'completed.json'),'failure':read(out/'failure.json'),'fit_failure':read(out/'fit/learned/failure.json'),'settlement':settlement,'health':health if job.get('child_pid') in (health or {}).get('members',[]) else None}
 tmp=folder/'latest.tmp';tmp.write_text(json.dumps(q,indent=2)+'\n');tmp.replace(folder/'latest.json')
 with (folder/'history.jsonl').open('a') as f:f.write(json.dumps(q)+'\n')
 if settlement:
  (folder/'terminal.json').write_text(json.dumps(q,indent=2)+'\n');break
 time.sleep(20)
