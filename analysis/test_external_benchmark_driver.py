"""Ensure the finite queue retains negatives and requires all audited identities."""
import copy,sys,unittest,subprocess,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
import external_benchmark_driver as d
def record(value):
    fold,method,seed=value
    endpoint=20000 if method=='pg' else 400
    return dict(status='EXTERNAL_CORE_NATIVE_CHECKPOINT_COUNTS_SCORES_CLEAN_SETTLEMENT_AUDIT_PASS',cell=dict(experiment='external_b1',fold=fold,method=method,seed=seed),benchmark_protocol_SHA256=d.PROTOCOL_SHA,
                own_processes_absent=True,owned_registry_lease_absent=True,independent_CPU_saved_score_audit=True,native_saved_state_matches_GPU_restore=True,
                completed=dict(status='completed',metric_role='final_evaluation',endpoint=endpoint,new_model_updates=0 if method=='scvi' and fold==seed==0 else endpoint,PG_configuration=d.read(d.PROTOCOL)['PG']['hyperparameters'] if method=='pg' else None))
class Queue(unittest.TestCase):
    def test_exact_finite_matrix_and_only_registered_reused_fit(self):
        values=d.matrix();self.assertEqual(len(values),30);self.assertEqual(len(set(values)),30)
        self.assertEqual(sum(m=='pg' for _,m,_ in values),15)
        self.assertEqual(sum(record(v)['completed']['new_model_updates']==0 for v in values),1)
        for v in values:self.assertTrue(d.validate_audit(record(v),v))
    def test_negative_quality_result_is_retained_and_not_filtered(self):
        q=record((0,'pg',0));q['summary']={'evaluation_r0':{'energy_score':99999.,'cell_energy_distance':99999.,'spread_ratio':0.}}
        self.assertTrue(d.validate_audit(q,(0,'pg',0)))
    def test_wrong_identity_endpoint_or_cleanup_cannot_advance_queue(self):
        original=record((0,'pg',0))
        for key,value in [('status','partial'),('benchmark_protocol_SHA256','wrong'),('own_processes_absent',False),('owned_registry_lease_absent',False),('independent_CPU_saved_score_audit',False)]:
            q=copy.deepcopy(original);q[key]=value
            with self.assertRaises(AssertionError):d.validate_audit(q,(0,'pg',0))
        for key,value in [('endpoint',10000),('new_model_updates',0),('metric_role','calibration_selection')]:
            q=copy.deepcopy(original);q['completed'][key]=value
            with self.assertRaises(AssertionError):d.validate_audit(q,(0,'pg',0))
        with self.assertRaises(AssertionError):d.validate_audit(original,(1,'pg',0))
    def test_import_does_not_expose_models_or_cuda(self):self.assertNotIn('torch',sys.modules)
    def test_settled_zombie_direct_child_is_reaped_before_audit(self):
        child=subprocess.Popen([sys.executable,'-I','-S','-c','pass'])
        path=Path('/proc',str(child.pid),'stat')
        identity=dict(pid=child.pid,birth=int(path.read_text().rsplit(')',1)[1].split()[19]),boot=Path('/proc/sys/kernel/random/boot_id').read_text().strip())
        for _ in range(100):
            if path.read_text().rsplit(')',1)[1].split()[0]=='Z':break
            time.sleep(.01)
        self.assertTrue(d.reap_settled_guard(identity));self.assertFalse(path.exists())
        child.wait()
    def test_live_guard_is_never_signalled_or_reaped_early(self):
        child=subprocess.Popen([sys.executable,'-I','-S','-c','import time;time.sleep(10)'])
        try:
            path=Path('/proc',str(child.pid),'stat')
            identity=dict(pid=child.pid,birth=int(path.read_text().rsplit(')',1)[1].split()[19]),boot=Path('/proc/sys/kernel/random/boot_id').read_text().strip())
            self.assertFalse(d.reap_settled_guard(identity));self.assertTrue(path.exists())
        finally:child.terminate();child.wait()
if __name__=='__main__':unittest.main(verbosity=2)
