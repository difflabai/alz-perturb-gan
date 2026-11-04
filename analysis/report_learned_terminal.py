"""Finite terminal observer: report actual settlement to coordinator; never launch/control."""
import datetime,json,subprocess,time
from pathlib import Path
ROOT=Path('/ml2/alz');ATTEMPT='CAL_GPU0_V1_pure_learned_a5b327c4c32e01fa8555_f0_pg_s1'
d=json.loads((ROOT/'artifacts/allocations'/ATTEMPT/'deployment.json').read_text());ledger=Path(d['ledger_path']);terminal=ledger.with_name(ATTEMPT+'-settlement.json');out=Path(d['output_dir'])
def read(path):return json.loads(path.read_text()) if path.exists() else None
while time.time()<d['expires_unix']+90:
 if terminal.exists():break
 time.sleep(20)
s=read(terminal);l=read(ledger);job=(l or {}).get('jobs',[{}])[0];checkpoint=read(out/'fit/learned/checkpoint_receipt.json');completed=read(out/'completed.json')
summary=('ALZ sharedGPU1 worker ended successfully; actual quality output requires independent CPU audit.' if s and job.get('status')=='completed' and completed else 'ALZ sharedGPU1 worker stopped; inspect actual settlement/checkpoint before continuing.')
details={'observed_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'attempt':ATTEMPT,'job_status':job.get('status'),'settlement':s,'learned_checkpoint':checkpoint,'worker_quality_output_present':bool(completed),'incumbent_changed':False,'automatic_successor_launched':False,'next_step':'CPU audit and finish if complete; otherwise retain checkpoint/charges and resolve original wall deadline with coordinator.'}
args={'recipient':'bits255.dots.ada_20261002','conversation_id':'report_auto_a20a79ed947ac6c51b74dbd821b796e2_ember','task_ref':'01a1a22b-f310-7190-a7c6-8697ecabae6c','event_ref':'alz-learned-gpu1-terminal-a5b327c4c32e01fa8555','kind':'experiment_transition','from_state':'GPU1_exact_learned9237_scientific_resume','to_state':'guard_terminal_CPU_audit_required','summary':summary,'details':json.dumps(details),'reply_to':'task11-GPU1-profile-two-concrete-findings-0545-20261104'}
command=['/mnt/nvme/ai/conda/anaconda3/bin/python3','/home/mikkel/.local/lib/agent-babel/account_session_helper.py','tool','--client','codex','--account-file','/home/mikkel/.config/255bits/agent-babel/trusted-account.json','--native-session','01a1a22b-f310-7190-a7c6-8697ecabae6c','--tool','stage_report']
result=subprocess.run(command,input=json.dumps(args),text=True,capture_output=True,timeout=25)
receipt=ROOT/'artifacts/allocations'/ATTEMPT/'terminal_report_receipt.json';receipt.write_text(json.dumps({'returncode':result.returncode,'helper_output':result.stdout,'helper_error':result.stderr,'facts':details},indent=2)+'\n')
