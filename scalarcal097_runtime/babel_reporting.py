"""Existing authenticated mailbox; reporting cannot restart or stop model jobs."""
import json,subprocess
from pathlib import Path
ROOT=Path('/ml2/alz')
def report(kind,event,summary,details,*,log_root):
 request=dict(recipient='bits255.dots.ada_20261002',conversation_id='report_auto_a20a79ed947ac6c51b74dbd821b796e2_ember',task_ref='01a1a22b-f310-7190-a7c6-8697ecabae6c',event_ref=event,kind=kind,summary=summary,details=json.dumps(details))
 if kind=='experiment_transition':
  request.update(from_state=details['from_state'],to_state=details['to_state'])
 helper=['/mnt/nvme/ai/conda/anaconda3/bin/python3','/home/mikkel/.local/lib/agent-babel/account_session_helper.py','tool','--client','codex','--account-file','/home/mikkel/.config/255bits/agent-babel/trusted-account.json','--native-session','01a1a22b-f310-7190-a7c6-8697ecabae6c','--tool','stage_report']
 try:
  p=subprocess.run(helper,input=json.dumps(request),capture_output=True,text=True,cwd=ROOT,timeout=30)
  root=Path(log_root);root.mkdir(exist_ok=True,parents=True);(root/(event+'-report.stdout')).write_text(p.stdout);(root/(event+'-report.stderr')).write_text(p.stderr)
 except (OSError,subprocess.TimeoutExpired) as exc:
  (Path(log_root)/(event+'-report-error.json')).write_text(json.dumps(dict(error=repr(exc),computation_status_unchanged=True)))
