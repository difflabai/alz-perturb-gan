"""Resource-only scheduling checks: full45 retained, no native/model execution."""
import importlib.util
from pathlib import Path
import unittest

ROOT = Path('/ml2/alz')
spec = importlib.util.spec_from_file_location('capacity_v7_method_routing', ROOT/'analysis/continue_benchmark097_after_capacity_v7.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


class MethodRouting(unittest.TestCase):
    def setUp(self):
        self.costs = dict(pg=[4., 1.], scvi=[.5, .1], cfgen=[8., 2.])
        self.values = [m.scope.cell(1, method, 1) for method in self.costs]
        self.exclusions = {(m.GPUS[0], 'cfgen')}

    def choose(self, gpu, values=None, eligible=None, active=None):
        return m.choose_with_exclusions(self.values if values is None else values,
                {} if active is None else active, gpu, self.costs,
                set(m.GPUS) if eligible is None else eligible, self.exclusions)

    def test_no_exclusions_preserves_previous_cost_only_choice(self):
        for gpu in m.GPUS:
            self.assertEqual(m.choose_with_exclusions(self.values, {}, gpu, self.costs, set(m.GPUS)),
                             m.choose_available(self.values, {}, gpu, self.costs, set(m.GPUS)))

    def test_CFGen_never_selected_for_excluded_card(self):
        for eligible in ({m.GPUS[0]}, set(m.GPUS)):
            selected, _ = self.choose(m.GPUS[0], eligible=eligible)
            self.assertIn(selected['method'], ('pg', 'scvi'))

    def test_CFGen_only_waits_without_dropping_or_consuming_key(self):
        pending = [m.scope.cell(0, 'cfgen', seed) for seed in range(5)]
        before = list(pending)
        self.assertIsNone(self.choose(m.GPUS[0], values=pending, eligible={m.GPUS[0]})[0])
        self.assertEqual(pending, before)
        self.assertIn(self.choose(m.GPUS[1], values=pending)[0], pending)

    def test_constrained_forecast_allocates_zero_CFGen_to_excluded_card(self):
        _, report = self.choose(m.GPUS[1])
        self.assertEqual(report['counts_per_card']['cfgen'], [0, 1])
        self.assertFalse(report['quality_values_read'])

    def test_live_healthy_peer_is_preserved_in_cost_choice(self):
        active = {m.GPUS[1]: dict(cell=m.scope.cell(0, 'cfgen', 3), elapsed_seconds=.5)}
        saved = repr(active)
        selected, _ = self.choose(m.GPUS[0], active=active)
        self.assertNotEqual(selected['method'], 'cfgen')
        self.assertEqual(repr(active), saved)

    def test_busy_or_unregistered_card_refused(self):
        with self.assertRaises(AssertionError):
            self.choose(m.GPUS[1], active={m.GPUS[1]: dict(cell=self.values[0], elapsed_seconds=0)})
        with self.assertRaises(AssertionError):
            self.choose(m.GPUS[1], eligible={'unknown_gpu'})

    def test_entire_frozen45_schedules_once_without_excluded_binding(self):
        pending = m.old.matrix()
        starts = []
        clock = [0]
        def start(cell, gpu, count):
            starts.append((m.b.attempt_key(cell), gpu, cell['method']))
            return dict(cell=cell, gpu=gpu, identity={}, attempt=m.b.attempt_key(cell))
        def select(pending, active, gpu):
            timed = {g:dict(cell=h['cell'], elapsed_seconds=0) for g,h in active.items()}
            return m.choose_with_exclusions(pending,timed,gpu,self.costs,set(m.GPUS),self.exclusions)[0]
        result = m.drive(pending, [], {}, key=m.b.attempt_key, start=start,
                ready=lambda h: True, finish=lambda h:dict(cell=h['cell'], complete=True),
                select=select, observe=lambda *a:None, sleep=lambda seconds:clock.__setitem__(0,clock[0]+1),
                now=lambda:clock[0], expiry=10000)
        self.assertFalse(result['errors'])
        self.assertEqual(len(result['rows']),45)
        self.assertEqual(len({key for key,_,_ in starts}),45)
        self.assertFalse(any((gpu,method) in self.exclusions for _,gpu,method in starts))


if __name__ == '__main__':
    unittest.main()
