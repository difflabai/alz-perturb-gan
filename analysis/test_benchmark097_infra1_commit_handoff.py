"""A completed parent and pending output commit must not consume a native key."""
import json
import pathlib
import sys
import tempfile
import unittest

ROOT = pathlib.Path('/ml2/alz')
sys.path[:0] = [str(ROOT / 'analysis'), str(ROOT / 'benchmark097_infra1_runtime')]
import run_benchmark097_infra1_pipeline as m


class CompletedParent(unittest.TestCase):
    def fixture(self, folder):
        base = pathlib.Path(folder)
        parent = dict(pid=17, birth=23, boot='CPU_FIXTURE')
        receipt = dict(status='ALL45_ORIGINAL_ATTEMPTS_CLOSED_42_COMPLETE_THREE_INFRA_FAILURES_FRESH_REPLACEMENTS_REMAIN',
                       original_scientific_cells=45, successful_original_models=42,
                       failed_original_attempts=list(m.lineage.FAILED))
        handoff = base / 'closed.json'
        handoff.write_text(json.dumps(receipt))
        (base / 'completed.json').write_text(json.dumps(receipt))
        return base, parent, receipt, handoff

    def test_live_exact_parent_remains_a_wait_dependency(self):
        with tempfile.TemporaryDirectory() as folder:
            base, parent, _, handoff = self.fixture(folder)
            self.assertEqual(m.upstream_state(parent, get_identity=lambda pid: parent, upstream=base, handoff=handoff), 'live')

    def test_verified_closed_parent_is_accepted_without_restarting_it(self):
        with tempfile.TemporaryDirectory() as folder:
            base, parent, _, handoff = self.fixture(folder)
            self.assertEqual(m.upstream_state(parent, get_identity=lambda pid: None, upstream=base, handoff=handoff), 'completed')

    def test_missing_wrong_or_conflicting_completion_refuses(self):
        with tempfile.TemporaryDirectory() as folder:
            base, parent, receipt, handoff = self.fixture(folder)
            (base / 'failure.json').write_text('{}')
            with self.assertRaises(AssertionError):
                m.upstream_state(parent, get_identity=lambda pid: None, upstream=base, handoff=handoff)
            (base / 'failure.json').unlink()
            (base / 'completed.json').unlink()
            with self.assertRaises(AssertionError):
                m.upstream_state(parent, get_identity=lambda pid: None, upstream=base, handoff=handoff)
            receipt['successful_original_models'] = 41
            (base / 'completed.json').write_text(json.dumps(receipt))
            handoff.write_text(json.dumps(receipt))
            with self.assertRaises(AssertionError):
                m.upstream_state(parent, get_identity=lambda pid: None, upstream=base, handoff=handoff)


class CommitBoundary(unittest.TestCase):
    def test_dirty_outputs_wait_until_clean_without_native_work(self):
        clean = iter([False, False, True])
        clock = [0.]
        waits = []
        m.wait_clean_handoff(clean=lambda: next(clean), budget_free=lambda: True, deadline=10,
                             clock=lambda: clock[0], sleep=lambda n: clock.__setitem__(0, clock[0] + n),
                             on_wait=lambda: waits.append(clock[0]))
        self.assertEqual(waits, [0., 2.])

    def test_live_reservation_is_not_hidden_by_clean_tree(self):
        with self.assertRaises(AssertionError):
            m.wait_clean_handoff(clean=lambda: True, budget_free=lambda: False, deadline=10)

    def test_expiry_ends_commit_wait_without_admission(self):
        with self.assertRaises(TimeoutError):
            m.wait_clean_handoff(clean=lambda: False, budget_free=lambda: True, deadline=10,
                                 clock=lambda: 10, sleep=lambda n: self.fail('No wait after expiry'))

    def test_fresh_admission_defers_only_unconsumed_dirty_refusals(self):
        calls = []
        self.assertIsNone(m.fresh_admission(lambda: calls.append(True), clean=lambda: False, untouched=lambda: True))
        self.assertFalse(calls)
        handle = object()
        self.assertIs(m.fresh_admission(lambda: handle, clean=lambda: True, untouched=lambda: True), handle)
        def dirty():
            raise AssertionError(m.DIRTY_START)
        self.assertIsNone(m.fresh_admission(dirty, clean=lambda: True, untouched=lambda: True))
        def prepare_race():
            raise RuntimeError('Traceback from native preimport\nbinding.Refusal: ' + m.DIRTY_PREPARE)
        self.assertIsNone(m.fresh_admission(prepare_race, clean=lambda: True, untouched=lambda: True))

    def test_consumed_keys_and_unrelated_errors_are_never_retried(self):
        with self.assertRaises(AssertionError):
            m.fresh_admission(lambda: self.fail('Consumed key started'), clean=lambda: False, untouched=lambda: False)
        def drift():
            raise RuntimeError('Native source hash differs')
        with self.assertRaises(RuntimeError):
            m.fresh_admission(drift, clean=lambda: True, untouched=lambda: True)
        states = iter([True, False])
        def dirty_after_consumption():
            raise AssertionError(m.DIRTY_START)
        with self.assertRaises(AssertionError):
            m.fresh_admission(dirty_after_consumption, clean=lambda: True, untouched=lambda: next(states))


if __name__ == '__main__':
    unittest.main()
