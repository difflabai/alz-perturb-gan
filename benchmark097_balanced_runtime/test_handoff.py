"""Actual CPU PID signal isolation and authoritative handoff evidence cases."""
import errno,json,os,signal,subprocess,sys,tempfile,time,unittest
from unittest.mock import patch
from pathlib import Path
from handoff_support import classify,signal_dispatcher_only
from dual_coordination import identity
class Handoff(unittest.TestCase):
 def evidence(self,**changes):
  q=dict(has_result=False,has_authorization=False,has_ledger=False,job_status=None,settled=False,uncertain=False,guard_live=False,matching_process_live=False,reservation=False,history_consumed=False);q.update(changes);return classify(**q)
 def test_missing_process_or_state_alone_is_never_success(self):
  self.assertEqual(self.evidence(has_ledger=True,job_status='active'),'WAIT_ORIGINAL_SETTLEMENT')
  self.assertEqual(self.evidence(has_result=True),'RECONCILE_RESULT_WITHOUT_LEDGER')
  self.assertEqual(self.evidence(reservation=True),'RECONCILE_RESERVATION_NO_RETRY')
 def test_original_guard_or_partial_cli_keeps_same_handle(self):
  self.assertEqual(self.evidence(guard_live=True),'WAIT_ORIGINAL_PROCESSES')
  self.assertEqual(self.evidence(matching_process_live=True,has_authorization=True),'WAIT_ORIGINAL_PROCESSES')
 def test_closed_native_fit_is_readonly_audited_not_retrained(self):
  self.assertEqual(self.evidence(has_ledger=True,job_status='completed',settled=True),'FINISH_ORIGINAL_READONLY_AUDIT')
  self.assertEqual(self.evidence(has_ledger=True,job_status='completed',settled=True,has_result=True),'VERIFY_RETAINED_RESULT')
 def test_failures_uncertainty_and_consumed_keys_never_become_untouched(self):
  self.assertEqual(self.evidence(uncertain=True),'RECONCILE_UNCERTAIN_NO_RETRY')
  self.assertEqual(self.evidence(has_ledger=True,job_status='failed',settled=True),'RETAIN_INCOMPLETE_NO_MODEL_RETRY')
  self.assertEqual(self.evidence(history_consumed=True,has_authorization=True),'RECONCILE_CONSUMED_KEY_NO_RETRY')
 def test_zero_work_preparation_is_distinct_from_worker_restart(self):
  self.assertEqual(self.evidence(has_authorization=True),'PREPARED_UNSTARTED_ZERO_WORK_RECONCILIATION')
  self.assertEqual(self.evidence(),'UNTOUCHED_CELL')
 def test_signal_is_one_exact_CPU_pid_and_separate_worker_finishes(self):
  with tempfile.TemporaryDirectory(prefix='alz-cpu-handoff-',dir='/tmp') as folder:
   root=Path(folder);worker=root/'worker.py';parent=root/'parent.py'
   worker.write_text("import json,os,time\nfrom pathlib import Path\np=Path(__file__).parent\nfor i in range(5):\n time.sleep(.3)\n (p/'worker-progress.json').write_text(json.dumps({'completed':i+1}))\n(p/'worker-complete.json').write_text(json.dumps({'completed':5,'CUDA_used':False}))\n")
   parent.write_text("import json,os,subprocess,sys,time\nfrom pathlib import Path\np=Path(__file__).parent\nc=subprocess.Popen([sys.executable,'-I','-S','-B',str(p/'worker.py')],start_new_session=True)\n(p/'child.json').write_text(json.dumps({'pid':c.pid}))\ntry:\n while True:time.sleep(.1)\nexcept KeyboardInterrupt:pass\n")
   p=subprocess.Popen([sys.executable,'-I','-S','-B',str(parent)],stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True);child=None
   try:
    deadline=time.time()+5
    while not (root/'child.json').exists():
     self.assertLess(time.time(),deadline);time.sleep(.01)
    child=json.loads((root/'child.json').read_text())['pid'];child_id=identity(child);parent_id=identity(p.pid)
    wrong={**parent_id,'birth':parent_id['birth']+1}
    with self.assertRaises(ValueError):signal_dispatcher_only(wrong,parent,fixture_root=root)
    self.assertIsNone(p.poll())
    result=signal_dispatcher_only(parent_id,parent,fixture_root=root);self.assertTrue(result['single_PID_only']);self.assertFalse(result['training_or_foreign_process_signalled'])
    p.wait(timeout=5);self.assertEqual(identity(child),child_id)
    while not (root/'worker-complete.json').exists():
     self.assertLess(time.time(),deadline);time.sleep(.02)
    self.assertEqual(json.loads((root/'worker-complete.json').read_text())['completed'],5)
   finally:
    if p.poll() is None:p.terminate()
    p.wait(timeout=5)
    if child and identity(child)==child_id:os.kill(child,signal.SIGTERM)
    p.stdout.close();p.stderr.close()
 def test_exiting_proc_cmdline_race_is_not_a_guard_failure(self):
  from handoff_support import command_matches
  with patch.object(Path,'read_bytes',side_effect=ProcessLookupError(errno.ESRCH,'No such process')):
   self.assertFalse(command_matches(os.getpid(),Path('/tmp/CPU-fixture-script.py')))

if __name__=='__main__':unittest.main(verbosity=2)
