import copy,json,sys,unittest
from pathlib import Path
sys.path.insert(0,'/ml2/alz')
import pure_suite as p
class PureScope(unittest.TestCase):
 def auth(self,label):
  q=p.load_entry(label);receipt=str(Path(q['checkpoint_path']).parent/'checkpoint_receipt.json');manifest=str(p.ROOT/'pure_formulations.json')
  return {**q,'pure_formulation':label,'cap_seconds':900,'qualification_only':True,'hybrid_config':None,'fixed_endpoint':q['completed_updates'],'cell':{'seed':0},'pure_manifest_SHA256':p.sha(manifest),'files_SHA256':{receipt:p.sha(receipt),manifest:p.sha(manifest),q['checkpoint_path']:q['checkpoint_SHA256']}}
 def test_all_retained_receipts(self):
  for label in p.read(p.ROOT/'pure_formulations.json')['entries']:p.validate_authorization(self.auth(label))
 def test_reject_mixture_wrong_endpoint_config_and_seed(self):
  for key,value in [('hybrid_config',{'pg_bag_fraction':.75}),('fixed_endpoint',40000),('variant_config',{'z_dim':2}),('cell',{'seed':1}),('cap_seconds',21600),('qualification_only',False)]:
   q=self.auth('latent8');q[key]=value
   with self.assertRaises(AssertionError):p.validate_authorization(q)
 def test_reject_changed_receipt_pin(self):
  q=self.auth('continue40k');q['files_SHA256'][str(Path(q['checkpoint_path']).parent/'checkpoint_receipt.json')]='0'*64
  with self.assertRaises(AssertionError):p.validate_authorization(q)
if __name__=='__main__':unittest.main()
