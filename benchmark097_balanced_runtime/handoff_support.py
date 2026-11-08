"""Handoff evidence classification and narrowly bound CPU-only stop request."""
import json,os,signal,time
from pathlib import Path
from dual_coordination import identity

def classify(*,has_result,has_authorization,has_ledger,job_status=None,settled=False,uncertain=False,guard_live=False,matching_process_live=False,reservation=False,history_consumed=False):
 if uncertain:return 'RECONCILE_UNCERTAIN_NO_RETRY'
 if guard_live or matching_process_live:return 'WAIT_ORIGINAL_PROCESSES'
 if reservation:return 'RECONCILE_RESERVATION_NO_RETRY'
 if has_ledger:
  if not settled:return 'WAIT_ORIGINAL_SETTLEMENT'
  if job_status!='completed':return 'RETAIN_INCOMPLETE_NO_MODEL_RETRY'
  return 'VERIFY_RETAINED_RESULT' if has_result else 'FINISH_ORIGINAL_READONLY_AUDIT'
 if history_consumed:return 'RECONCILE_CONSUMED_KEY_NO_RETRY'
 if has_result:return 'RECONCILE_RESULT_WITHOUT_LEDGER'
 if has_authorization:return 'PREPARED_UNSTARTED_ZERO_WORK_RECONCILIATION'
 return 'UNTOUCHED_CELL'

def command_matches(pid,script,authorization=None):
 try:
  p=Path('/proc',str(pid))
  if p.stat().st_uid!=os.getuid():return False
  args=(p/'cmdline').read_bytes().split(b'\0');wanted=os.fsencode(str(script))
  if wanted not in args:return False
  return authorization is None or os.fsencode(str(authorization)) in args
 except FileNotFoundError:return False

def matching_processes(scripts,authorization=None):
 found=[]
 for p in Path('/proc').iterdir():
  if not p.name.isdigit():continue
  try:
   if p.stat().st_uid!=os.getuid():continue
   if any(command_matches(int(p.name),script,authorization) for script in scripts):
    ident=identity(int(p.name))
    if ident is not None:found.append(ident)
  except FileNotFoundError:continue
 return found

def signal_dispatcher_only(expected,script,*,fixture_root=None):
 """Signal one exact CPU PID, never a process group or a training guard."""
 script=Path(script).resolve()
 if fixture_root is None:
  if script!=Path('/ml2/alz/analysis/run_benchmark097_dual.py'):raise ValueError('Native handoff binds only the original CPU dispatcher')
 else:
  root=Path(fixture_root).resolve()
  if not root.is_relative_to(Path('/tmp')) or not script.is_relative_to(root):raise ValueError('CPU fixture signal must be confined to its own temporary source')
 if identity(expected['pid'])!=expected or not command_matches(expected['pid'],script):raise ValueError('Exact original CPU dispatcher identity/command missing')
 if fixture_root is None and Path('/proc',str(expected['pid']),'wchan').read_text().strip()!='hrtimer_nanosleep':raise ValueError('Wait for a quiescent CPU dispatch loop before stopping admissions')
 os.kill(expected['pid'],signal.SIGINT)
 return dict(signalled_identity=expected,signal='SIGINT',single_PID_only=True,training_or_foreign_process_signalled=False)
