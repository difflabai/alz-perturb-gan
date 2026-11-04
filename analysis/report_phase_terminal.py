"""Finite observer reports one existing phase's actual settlement; never launches successors."""
import argparse,datetime,hashlib,json,subprocess,time
from pathlib import Path
parser=argparse.ArgumentParser();parser.add_argument('--attempt',required=True);args=parser.parse_args();ROOT=Path('/ml2/alz');ATTEMPT=args.attempt
folder=ROOT/'artifacts/allocations'/ATTEMPT;d=json.loads((folder/'deployment.json').read_text());ledger=Path(d['ledger_path']);terminal=ledger.with_name(ATTEMPT+'-settlement.json');out=Path(d['output_dir'])
def read(path):return json.loads(path.read_text()) if path.exists() else None
end=min(d['expires_unix']+90,time.time()+d['cap_seconds']+90)
while time.time()<end:
 if terminal.exists():break
 time.sleep(20)
s=read(terminal);l=read(ledger);job=(l or {}).get('jobs',[{}])[0];checkpoint=read(out/'fit/learned/checkpoint_receipt.json');complete=read(out/'completed.json')
summary=('ALZ sharedGPU1 phase ended successfully; quality output requires independent CPU audit.' if s and job.get('status')=='completed' and complete else 'ALZ sharedGPU1 phase stopped; inspect terminal checkpoint and canonical settlement before continuing.')
details={'observed_UTC':datetime.datetime.now(datetime.timezone.utc).isoformat(),'attempt':ATTEMPT,'job_status':job.get('status'),'settlement':s,'learned_checkpoint':checkpoint,'worker_quality_output_present':bool(complete),'owner_five_day_deadline_UTC':'2026-11-09T06:26:46.606656+00:00','incumbent_changed':False,'automatic_successor_launched':False,'next_step':'CPU checkpoint/score/count/cleanup audit if complete; otherwise preserve fullstate and all charges before fresh exact continuation within owner-authorized five-day window.'}
notice={'recipient':'bits255.dots.ada_20261002','conversation_id':'report_auto_a20a79ed947ac6c51b74dbd821b796e2_ember','task_ref':'01a1a22b-f310-7190-a7c6-8697ecabae6c','event_ref':'alz-phase-terminal-'+hashlib.sha256(ATTEMPT.encode()).hexdigest()[:24],'kind':'experiment_transition','from_state':'five_day_shared_GPU1_scientific_phase','to_state':'terminal_requires_independent_CPU_audit','summary':summary,'details':json.dumps(details)}
cmd=['/mnt/nvme/ai/conda/anaconda3/bin/python3','/home/mikkel/.local/lib/agent-babel/account_session_helper.py','tool','--client','codex','--account-file','/home/mikkel/.config/255bits/agent-babel/trusted-account.json','--native-session','01a1a22b-f310-7190-a7c6-8697ecabae6c','--tool','stage_report']
r=subprocess.run(cmd,input=json.dumps(notice),text=True,capture_output=True,timeout=25);(folder/'terminal_report_receipt.json').write_text(json.dumps({'returncode':r.returncode,'helper_output':r.stdout,'helper_error':r.stderr,'facts':details},indent=2)+'\n')
