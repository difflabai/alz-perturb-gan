"""Frozen settings must exactly match completed native-trainer CPU prefix cases."""
import json,os,sys,unittest
import preset_scope as s
assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
s.validate_protocol()
class Factory(unittest.TestCase):
 def test_exact_model_and_native_trainer_source_pins(self):
  q=s.read(s.GRID_MODEL_PROOF);self.assertEqual(q['actual_cases'],24);self.assertEqual(q['actual_optimizer_pairs_per_implementation'],72);self.assertFalse(q['native_package_modified']);self.assertFalse(q['CUDA_initialized'])
  for p,pin in q['source_SHA256'].items():self.assertEqual(s.sha(p),pin)
  for p,pin in s.read(s.MODEL_PROOF)['source_SHA256'].items():self.assertEqual(s.sha(p),pin)
 def test_every_actual_frozen_setting_seed_has_exact_native_update_prefix(self):
  p=s.read(s.PROTOCOL);anchor=s.read(s.STUDY/'frozen_design.json')['anchor_candidate_id'];proof=s.read(s.GRID_MODEL_PROOF);rows=[]
  for candidate in p['candidates']:
   for seed in (0,1):
    q=next(r for r in proof['rows'] if r['anchor_id']==anchor and r['candidate_id']==candidate['id'] and r['seed']==seed)
    self.assertEqual(q['hyperparameters'],candidate['hyperparameters']);self.assertEqual(q['optimizer_pairs_per_loop'],3);self.assertTrue(q['exact_models_optimizers_rng_and_update_clock']);self.assertTrue(q['exact_all_losses_and_penalty']);self.assertTrue(q['from_scratch']);rows.append(q)
  self.assertEqual(len(rows),2*len(p['candidates']));self.assertNotIn('torch',sys.modules);print('NATIVE_LOCAL_ROWS '+json.dumps(rows))
if __name__=='__main__':unittest.main(verbosity=2)
