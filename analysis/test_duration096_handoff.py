"""Actual process tests for an exact supervisor-only scheduling handoff."""
import importlib.util,os,subprocess,sys,tempfile,time,unittest
from pathlib import Path
spec=importlib.util.spec_from_file_location('duration_handoff',Path(__file__).with_name('run_duration096_then_resume_benchmark.py'))
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
class Handoff(unittest.TestCase):
    def test_birth_mismatch_refuses_without_signalling(self):
        p=subprocess.Popen([sys.executable,'-c','import time;time.sleep(20)'])
        try:
            expected=m.identity(p.pid);expected['birth']+=1
            with self.assertRaises(AssertionError):m.stop_exact_supervisor(expected)
            self.assertIsNone(p.poll())
        finally:p.terminate();p.wait()
    def test_exact_supervisor_stops_and_its_independent_worker_survives(self):
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'child'
            code='import subprocess,sys,time;from pathlib import Path;p=subprocess.Popen([sys.executable,"-c","import time;time.sleep(20)"],start_new_session=True);Path(sys.argv[1]).write_text(str(p.pid));time.sleep(20)'
            p=subprocess.Popen([sys.executable,'-c',code,str(path)]);child=None
            try:
                end=time.monotonic()+5
                while not path.exists() and time.monotonic()<end:time.sleep(.01)
                child=int(path.read_text());worker=m.identity(child)
                m.stop_exact_supervisor(m.identity(p.pid));p.wait(timeout=5)
                self.assertEqual(p.returncode,-15)
                self.assertTrue(m.same(m.identity(child),worker));self.assertNotEqual(m.identity(child)['state'],'Z')
            finally:
                if p.poll() is None:p.terminate();p.wait()
                if child and Path('/proc',str(child)).exists():os.kill(child,15)
if __name__=='__main__':unittest.main(verbosity=2)
