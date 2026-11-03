"""Read-only real readiness refusal and synthetic scientific-gate edge cases."""
import copy,json,unittest
from pathlib import Path
from freeze_pure_confirmation import seed0_gate,sha,confirmation_identities

ROOT=Path('/ml2/alz')

class ConfirmationGate(unittest.TestCase):
 def setUp(self):
  self.paths=[ROOT/'pure_incumbent.json',ROOT/'incumbent.json',ROOT/'analysis/baseline_calibration_seed0.json']
  self.pins={str(p):sha(p) for p in self.paths}
  self.c=json.loads((ROOT/'runs/CAL_GPU0_V1_pure_latent8_v1_f0_pg_s0/completed.json').read_text());self.b=json.loads(self.paths[2].read_text())
  # Counterfactual test numbers, never a model result or a selection candidate.
  self.good={**self.c,'energy_score':.94*self.b['energy_score'],'cell_energy_distance':.99*self.b['cell_energy_distance'],'spread_ratio':1.}
 def tearDown(self):self.assertEqual(self.pins,{str(p):sha(p) for p in self.paths})
 def test_actual_PG8_not_ready_for_joint_target(self):
  self.assertEqual(self.c['checkpoint_SHA256'],'444c22a5829b79d8ed0cd300be255d9410916d6c2248f930cf4cc0f886744da2')
  q=seed0_gate(self.c,self.b);self.assertFalse(q['ready']);self.assertEqual(len(q['reasons']),2)
 def test_synthetic_joint_target_passes_without_writing_or_allocating(self):self.assertTrue(seed0_gate(self.good,self.b)['ready'])
 def test_synthetic_individual_regressions_and_identity_changes_refuse(self):
  for change in [{'energy_score':.96*self.b['energy_score']},{'cell_energy_distance':1.01*self.b['cell_energy_distance']},
                 {'spread_ratio':.69},{'spread_ratio':1.31},{'training_seed':1},{'pure_only':False},
                 {'real_bag_replacements':1},{'integer_count_contract_passed':False},{'completed_updates':19999},
                 {'hybrid_config':{'pg_bag_fraction':.75}},{'trace_manifest_SHA256':'0'*64},
                 {'metric_role':'evaluation'},{'generation_replicates':1}]:
   with self.subTest(change=change):self.assertFalse(seed0_gate({**self.good,**change},self.b)['ready'])
 def test_nonfinite_or_invalid_baseline_refuses(self):
  for metric in ['energy_score','cell_energy_distance','spread_ratio']:
   for value in [float('nan'),float('inf'),float('-inf'),True,None]:
    with self.subTest(metric=metric,value=value):self.assertFalse(seed0_gate({**self.good,metric:value},self.b)['ready'])
  self.assertFalse(seed0_gate(self.good,{**self.b,'energy_score':0.})['ready'])
 def test_confirmation_attempts_are_prospectively_unique_and_deterministic(self):
  attempt='CAL_GPU0_V1_network_rate_half_v1_f0_pg_s0';pin='a'*64
  q=confirmation_identities(attempt,pin)
  self.assertEqual(q,confirmation_identities(attempt,pin))
  self.assertEqual(set(q),{'1','2'})
  self.assertNotEqual(q['1']['attempt_id'],q['2']['attempt_id'])
  self.assertNotEqual(q,confirmation_identities(attempt,'b'*64))
  self.assertNotEqual(q,confirmation_identities('CAL_GPU0_V1_another_v1_f0_pg_s0',pin))
  for seed in (1,2):
   self.assertEqual(q[str(seed)]['cell']['seed'],seed)
   self.assertRegex(q[str(seed)]['cell']['experiment'],r'^pure_confirm_[0-9a-f]{20}$')
 def test_malformed_selection_identity_or_incomplete_checkpoint_digest_refuses(self):
  for attempt,pin in [('CAL_GPU0_V1_test_f0_pg_s1','a'*64),('../selection','a'*64),
                      ('CAL_GPU0_V1_test_f0_pg_s0','a'*63),('CAL_GPU0_V1_test_f0_pg_s0','z'*64)]:
   with self.subTest(attempt=attempt,pin=pin),self.assertRaises(ValueError):confirmation_identities(attempt,pin)

if __name__=='__main__':unittest.main()
