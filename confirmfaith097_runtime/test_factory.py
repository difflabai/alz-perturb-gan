"""Actual native-trainer prefix parity for every frozen confirmation setting/seed."""
import importlib.util,json,os,sys,unittest
from pathlib import Path
import preset_scope as s
assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
s.validate_protocol()
path=s.ROOT/'analysis/faithful097/native_trainer_parity.py'
spec=importlib.util.spec_from_file_location('actual_released_parity',path);native=importlib.util.module_from_spec(spec);spec.loader.exec_module(native)
import torch,released_update as update
class Factory(unittest.TestCase):
 def tearDown(self):self.assertFalse(torch.cuda.is_initialized())
 def test_qualified_update_source_and_nonEMA_full_horizon(self):
  for p,pin in s.read(s.MODEL_PROOF)['source_SHA256'].items():self.assertEqual(s.sha(p),pin)
  update.configure(s.CFG);update.configure_hyperparameters(s.read(s.PROTOCOL)['selected_configurations'][0]['hyperparameters']);r=update.recipe();self.assertEqual((r.total_steps,r.ema_decay,r.generator_gradient_response),(20000,0,'none'))
 def test_every_frozen_setting_and_seed_matches_actual_native_trainer(self):
  rows=[]
  for c in s.read(s.PROTOCOL)['selected_configurations']:
   for seed in (2,3,4):
    q=native.run(c['hyperparameters'],seed,pairs=3);q['candidate_id']=c['candidate_id'];rows.append(q)
  self.assertEqual(len(rows),3*len(s.ENABLED['candidate_ids']));print('NATIVE_CONFIRMATION_ROWS '+json.dumps(rows))
if __name__=='__main__':unittest.main(verbosity=2)
