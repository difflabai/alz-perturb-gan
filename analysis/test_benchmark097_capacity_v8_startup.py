"""Exercise ledger publication delay without CUDA or a scientific model."""
import errno
import importlib.util
import json
import pathlib
import subprocess
import sys
import tempfile
import time
import unittest

ROOT = pathlib.Path('/ml2/alz')
spec = importlib.util.spec_from_file_location('startup_v8', ROOT / 'analysis/continue_benchmark097_after_capacity_v8.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


class StartupAccounting(unittest.TestCase):
    def handle(self):
        identity = dict(pid=17, birth=23, boot='CPU_FIXTURE')
        return dict(identity=identity, cell=dict(method='scvi'), attempt='CPU_FIXTURE',
                    auth=dict(ledger_path='/CPU_FIXTURE/ledger.json'), admission_requested_unix=1.)

    @staticmethod
    def absent(path):
        raise FileNotFoundError(errno.ENOENT, 'not published', str(path))

    def test_exact_live_startup_defers_without_inventing_elapsed_time(self):
        h = self.handle()
        self.assertIsNone(m.active_timing(h, clock=lambda: 2., get_identity=lambda pid: h['identity'], read_json=self.absent))

    def test_overdue_or_negative_startup_time_fails_closed(self):
        h = self.handle()
        for now in (31., 100., 0.):
            with self.assertRaises(RuntimeError):
                m.active_timing(h, clock=lambda: now, get_identity=lambda pid: h['identity'], read_json=self.absent)

    def test_absent_reused_or_foreign_guard_cannot_authorize_deferral(self):
        h = self.handle()
        for identity in (None, h['identity'] | dict(birth=99)):
            with self.assertRaises(RuntimeError):
                m.active_timing(h, clock=lambda: 2., get_identity=lambda pid: identity, read_json=self.absent)

    def test_retained_handle_without_launch_evidence_never_defers(self):
        h = self.handle()
        del h['admission_requested_unix']
        with self.assertRaises(RuntimeError):
            m.active_timing(h, clock=lambda: 2., get_identity=lambda pid: h['identity'], read_json=self.absent)

    def test_published_ledger_needs_exact_owner_and_uses_real_start(self):
        h = self.handle()
        value = dict(jobs=[dict(guard_identity=h['identity'], started_unix=1.5)])
        self.assertEqual(m.active_timing(h, clock=lambda: 3., read_json=lambda path: value)['elapsed_seconds'], 1.5)
        value['jobs'][0]['guard_identity'] = h['identity'] | dict(birth=99)
        with self.assertRaises(AssertionError):
            m.active_timing(h, clock=lambda: 3., read_json=lambda path: value)

    def test_missing_authorization_is_not_a_startup_ledger_exception(self):
        h = self.handle()
        del h['auth']
        with self.assertRaises(FileNotFoundError):
            m.active_timing(h, clock=lambda: 2., get_identity=lambda pid: h['identity'], read_json=self.absent)

    def test_real_delayed_process_publishes_then_becomes_timed(self):
        with tempfile.TemporaryDirectory() as folder:
            ledger = pathlib.Path(folder) / 'ledger.json'
            code = ('import json,pathlib,sys,time;ident=json.loads(sys.stdin.readline());'
                    'p=pathlib.Path(sys.argv[1]);time.sleep(.25);'
                    'temp=p.with_suffix(".partial");'
                    'temp.write_text(json.dumps(dict(jobs=[dict(guard_identity=ident,started_unix=time.time())])));'
                    'temp.replace(p);time.sleep(.5)')
            child = subprocess.Popen([sys.executable, '-I', '-S', '-B', '-c', code, str(ledger)],
                                     stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
            try:
                identity = m.identity(child.pid)
                self.assertIsNotNone(identity)
                child.stdin.write(json.dumps(identity) + '\n')
                child.stdin.flush()
                h = dict(identity=identity, cell=dict(method='scvi'), attempt='CPU_FIXTURE',
                         auth=dict(ledger_path=str(ledger)), admission_requested_unix=time.time())
                self.assertIsNone(m.active_timing(h))
                deadline = time.monotonic() + 3
                while not ledger.exists() and time.monotonic() < deadline:
                    time.sleep(.02)
                self.assertTrue(ledger.exists())
                self.assertGreaterEqual(m.active_timing(h)['elapsed_seconds'], 0)
                self.assertEqual(child.wait(timeout=5), 0)
            finally:
                if child.poll() is None:
                    child.kill()  # This exact confined CPU fixture only.
                    child.wait(timeout=5)
                child.stdin.close()
                child.stderr.close()


if __name__ == '__main__':
    unittest.main()
