"""Read-only real readiness refusal and synthetic scientific-gate edge cases."""
import copy,json,unittest
from pathlib import Path
from freeze_pure_confirmation import seed0_gate,sha

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

if __name__=='__main__':unittest.main()
