"""Read-only native learned observation source gates, with a virtual enabled plan; no allocation."""
import copy,hashlib,json
from pathlib import Path
import unittest
from unittest.mock import patch
import binding as b

class LearnedBinding(unittest.TestCase):
 def setUp(self):
  self.root=b.HERE.parent
  prior=b.read(self.root/'calibration_deployment.json')
  self.auth=copy.deepcopy(b.read(prior['authorization_path']))
  self.plan=self.root/'learned_observation_plan.json'
  self.enabled={'enabled':True,'cell_hidden':64,'cell_features':25,'feature_scale':.2,'pool_epsilon':1e-6,'layernorm_epsilon':1e-5}
  self.plan_pin=hashlib.sha256((json.dumps(self.enabled,indent=2)+'\n').encode()).hexdigest()
  research={str(p):b.sha(p) for p in [*self.root.glob('*.py'),*b.HERE.glob('*.py'),*(self.root/'model').glob('*.py')]}
  self.auth.update(operation='train_learned',pure_formulation='learned_observation',fixed_endpoint=20000,cap_seconds=21600,qualification_only=False,hybrid_config=None,
                   variant_config={'z_dim':8,'stage_balance':False,'cell_z_dim':8,'theta_init_scale':1.,'fourier_feature_scale':1.},
                   learned_observation_config=self.enabled,learned_observation_probe_SHA256=b.sha(self.root/'analysis/learned_observation_fitting_probe.json'),
                   research_source_SHA256=research,learned_observation_plan_SHA256=self.plan_pin,
                   learned_observation_qualification_SHA256=b.sha(self.root/'analysis/learned_observation_cpu_qualification.json'))
  self.auth['cell']={'fold':0,'method':'pg','seed':0,'experiment':'learned_source_cpu_only'}
  pins=self.auth['files_SHA256'];pins.update(research)
  for p in [self.root/'candidate.json',self.root/'incumbent.json',self.root/'pure_incumbent.json',self.root/'learned_observation_protocol.json',self.root/'analysis/learned_observation_cpu_qualification.json',self.root/'analysis/run_learned_observation_cpu.py',self.root/'analysis/learned_observation_fitting_probe.json',self.root/'analysis/probe_learned_observation_fitting.py']:pins[str(p)]=b.sha(p)
  pins[str(self.plan)]=self.plan_pin
  self.real_read=b.read;self.real_sha=b.sha
 def check(self,auth):
  def read(path):return copy.deepcopy(self.enabled) if Path(path)==self.plan else self.real_read(path)
  def sha(path):return self.plan_pin if Path(path)==self.plan else self.real_sha(path)
  with patch.object(b,'read',read),patch.object(b,'sha',sha):b.source_closure(auth,b.NATIVE_LAYOUT)
 def test_complete_source_closure_with_virtual_enabled_plan(self):self.check(self.auth)
 def test_wrong_seed_config_endpoint_hybrid_or_proof_refuses(self):
  for change in [{'cell':{**self.auth['cell'],'seed':1}},{'variant_config':{**self.auth['variant_config'],'z_dim':2}},
                 {'fixed_endpoint':40000},{'cap_seconds':900},{'hybrid_config':{'pg_bag_fraction':.75}},{'qualification_only':True},
                 {'learned_observation_plan_SHA256':'0'*64},{'learned_observation_qualification_SHA256':'0'*64},{'learned_observation_probe_SHA256':'0'*64},{'learned_observation_config':{**self.enabled,'cell_features':26}}]:
   with self.subTest(change=change),self.assertRaises(b.Refusal):self.check({**self.auth,**change})
 def test_disabled_plan_refuses_and_no_model_imports(self):
  self.enabled['enabled']=False
  with self.assertRaisesRegex(b.Refusal,'disabled or changed'):self.check(self.auth)
  import sys
  self.assertNotIn('torch',sys.modules)

if __name__=='__main__':unittest.main()
