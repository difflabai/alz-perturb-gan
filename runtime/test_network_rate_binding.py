"""Read-only native NetworkRate source gates, with a virtual enabled plan; no allocation."""
import copy,hashlib,json
from pathlib import Path
import unittest
from unittest.mock import patch
import binding as b

class NetworkRateBinding(unittest.TestCase):
 def setUp(self):
  self.root=b.HERE.parent
  prior=b.read(self.root/'calibration_deployment.json')
  self.auth=copy.deepcopy(b.read(prior['authorization_path']))
  self.plan=self.root/'network_rate_plan.json'
  self.enabled={'enabled':True,'native_recipe':'bispf_optimized','lr':.006,'prior_lr_mult':5.,'fixed_endpoint':20000,'pure_only':True,'training_seed':0}
  self.plan_pin=hashlib.sha256((json.dumps(self.enabled,indent=2)+'\n').encode()).hexdigest()
  research={str(p):b.sha(p) for p in [*self.root.glob('*.py'),*b.HERE.glob('*.py'),*(self.root/'model').glob('*.py')]}
  self.auth.update(operation='train_network_rate',pure_formulation='network_rate_half',network_rate_config={'lr':.006,'prior_lr_mult':5.},fixed_endpoint=20000,cap_seconds=21600,qualification_only=False,hybrid_config=None,
                   variant_config={'z_dim':8,'stage_balance':False,'cell_z_dim':8,'theta_init_scale':1.,'fourier_feature_scale':1.},
                   research_source_SHA256=research,network_rate_plan_SHA256=self.plan_pin,
                   network_rate_qualification_SHA256=b.sha(self.root/'analysis/network_rate_cpu_qualification.json'))
  self.auth['cell']={'fold':0,'method':'pg','seed':0,'experiment':'network_rate_source_cpu_only'}
  pins=self.auth['files_SHA256'];pins.update(research)
  for p in [self.root/'candidate.json',self.root/'incumbent.json',self.root/'pure_incumbent.json',self.root/'network_rate_protocol.json',self.root/'analysis/network_rate_cpu_qualification.json',self.root/'analysis/run_network_rate_cpu.py']:pins[str(p)]=b.sha(p)
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
                 {'network_rate_config':{'lr':.006,'prior_lr_mult':2.5}},{'network_rate_plan_SHA256':'0'*64},{'network_rate_qualification_SHA256':'0'*64}]:
   with self.subTest(change=change),self.assertRaises(b.Refusal):self.check({**self.auth,**change})
 def test_disabled_plan_refuses_and_no_model_imports(self):
  self.enabled['enabled']=False
  with self.assertRaisesRegex(b.Refusal,'disabled or changed'):self.check(self.auth)
  import sys
  self.assertNotIn('torch',sys.modules)

if __name__=='__main__':unittest.main()
