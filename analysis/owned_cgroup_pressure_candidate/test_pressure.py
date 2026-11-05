"""CPU cgroup telemetry counterexamples and unchanged guarded-run semantics."""
import ast
from pathlib import Path
import sys
import tempfile
import unittest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from owned_pressure import OwnedPressure, PressureRefusal, full_avg10


def psi(value):
    return f'some avg10={value} avg60=0 avg300=0 total=0\nfull avg10={value} avg60=0 avg300=0 total=0\n'


class Pressure(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='alz-private-cgroup-pressure-')
        self.root = Path(self.temp.name)
        self.proc = self.root / 'proc'
        self.cg = self.root / 'cgroup'
        for folder in (self.proc / 'self', self.proc / '123', self.proc / 'pressure', self.cg / 'owned'):
            folder.mkdir(parents=True, exist_ok=True)
        for pid in ('self', '123'):
            (self.proc / pid / 'cgroup').write_text('0::/owned\n')
        (self.proc / 'pressure/memory').write_text(psi(20))
        (self.cg / 'owned/memory.pressure').write_text(psi(0))

    def tearDown(self):
        self.temp.cleanup()

    def reader(self):
        return OwnedPressure(self.proc, self.cg)

    def test_unrelated_global_stall_does_not_report_owned_stall(self):
        probe = self.reader()
        probe.check_worker(123)
        self.assertEqual(probe.full_pressure(), 0)
        self.assertEqual(probe.sample()['global_full_avg10'], 20)
        self.assertEqual(probe.sample()['cgroup_relative_path'], '/owned')

    def test_owned_stall_is_reported_even_with_quiet_global(self):
        (self.proc / 'pressure/memory').write_text(psi(0))
        (self.cg / 'owned/memory.pressure').write_text(psi(25))
        self.assertEqual(self.reader().full_pressure(), 25)

    def test_unknown_malformed_or_changed_scope_never_falls_back_to_zero(self):
        for value in ('nan', 'inf', '-1', '101', 'missing'):
            with self.subTest(value=value):
                (self.cg / 'owned/memory.pressure').write_text(psi(value))
                with self.assertRaises(PressureRefusal):
                    self.reader().sample()
        (self.cg / 'owned/memory.pressure').write_text('full avg10=0\nfull avg10=0\n')
        with self.assertRaises(PressureRefusal):
            self.reader().sample()
        (self.cg / 'owned/memory.pressure').write_text(psi(0))
        probe = self.reader()
        (self.proc / 'self/cgroup').write_text('0::/other\n')
        with self.assertRaises(PressureRefusal):
            probe.sample()
        for membership in ('0::/../outside\n', '1:memory:/owned\n', '0::/owned\n0::/owned\n'):
            (self.proc / 'self/cgroup').write_text(membership)
            with self.assertRaises(PressureRefusal):
                self.reader()
        (self.proc / 'self/cgroup').write_text('0::/owned\n')
        probe = self.reader()
        (self.cg / 'owned/memory.pressure').unlink()
        with self.assertRaises(PressureRefusal):
            probe.sample()

    def test_worker_scope_must_match_before_launch_barrier(self):
        probe = self.reader()
        probe.check_worker(123)
        (self.proc / '123/cgroup').write_text('0::/other\n')
        with self.assertRaises(PressureRefusal):
            probe.check_worker(123)
        (self.proc / '123/cgroup').unlink()
        with self.assertRaises(PressureRefusal):
            probe.check_worker(123)

    def test_guarded_run_oom_budget_locks_ownership_and_cleanup_ast_unchanged(self):
        old = Path('/ml2/alz/recovery_runtime/shared_gpu1_runner.py')
        new = HERE / 'cgroup_shared_gpu1_runner.py'
        def functions(path):
            return {node.name: node for node in ast.parse(path.read_text()).body if isinstance(node, ast.FunctionDef)}
        before, after = functions(old), functions(new)
        for name in ('run', 'oom_count', 'cleanup_owned_targets', 'signal_owned'):
            self.assertEqual(ast.dump(before[name], include_attributes=False), ast.dump(after[name], include_attributes=False), name)
        after_gate = after['gate_spawn']
        removed = 0
        for node in ast.walk(after_gate):
            for attr in ('body', 'orelse', 'finalbody'):
                body = getattr(node, attr, None)
                if not isinstance(body, list):
                    continue
                kept = []
                for statement in body:
                    if isinstance(statement, ast.Expr) and isinstance(statement.value, ast.Call) and isinstance(statement.value.func, ast.Attribute) and statement.value.func.attr == 'check_worker':
                        removed += 1
                    else:
                        kept.append(statement)
                setattr(node, attr, kept)
        self.assertEqual(removed, 1)
        self.assertEqual(ast.dump(before['gate_spawn'], include_attributes=False), ast.dump(after_gate, include_attributes=False))
        text = new.read_text()
        self.assertLess(text.index('pressure_reader().check_worker(worker.pid)'), text.index("lease.attach(w,observe());os.write(wr,b'1')"))
        self.assertIn('psi>=10', text)
        self.assertIn('now-pressure_since>=30', text)


if __name__ == '__main__':
    unittest.main()
