"""CPU orchestration and real frozen-source refusal. No native deployment or models."""
import copy,gc,json,sys,tempfile,unittest,weakref
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
ROOT=Path('/ml2/alz');sys.path[:0]=[str(Path(__file__).parent),str(ROOT/'runtime'),str(ROOT),str(ROOT/'analysis')]
import binding as b
import ensemble_confirmation_scope as scope
from ensemble_confirmation_execution import fit_pair
from aggregate_pure_confirmation import stable_target

class Orchestration(unittest.TestCase):
 def run_pair(self,seed=1,incomplete=False,error=False):
  events=[];refs=[]
  def factory(index):
   def make():
    if refs:self.assertIsNone(refs[-1](),'First fitting Context must be released before second allocation')
    class C:pass
    ctx=C();ctx.index=index;refs.append(weakref.ref(ctx));events.append(('context',index));return ctx
   return make
  def fitter(index):
   def fit(ctx,out,seed,*,end_monotonic):
    self.assertEqual(ctx.index,index);events.append(('fit',index,str(out),seed,end_monotonic))
    if error:raise InterruptedError('fixture finite deadline')
    return (SimpleNamespace(completed_steps=19999 if incomplete else 20000),),out/'final.pt'
   return fit
  bundles,paths=fit_pair([factory(0),factory(1)],[fitter(0),fitter(1)],seed,Path('/tmp/NOT-A-NATIVE-OUTPUT'),123.,lambda:(gc.collect(),events.append(('release',))),lambda i:events.append(('phase',i)))
  self.assertEqual(len(bundles),2);self.assertEqual(len(paths),2)
  return events
 def test_both_seeds_use_each_own_original_context_and_same_deadline(self):
  for seed in (1,2):
   events=self.run_pair(seed);fits=[v for v in events if v[0]=='fit'];self.assertEqual([v[1] for v in fits],[0,1]);self.assertEqual([v[3] for v in fits],[seed,seed]);self.assertEqual([v[4] for v in fits],[123.,123.])
   self.assertTrue(fits[0][2].endswith('/fit/pg8'));self.assertTrue(fits[1][2].endswith('/fit/learned'))
   self.assertEqual([v for v in events if v[0]=='phase'],[('phase',0),('phase',1)])
 def test_incomplete_parent_refuses_before_second_fit(self):
  with self.assertRaisesRegex(AssertionError,'partial parent'):self.run_pair(incomplete=True)
 def test_deadline_error_is_not_restarted_or_evaluated(self):
  with self.assertRaises(InterruptedError):self.run_pair(error=True)
 def test_seed0_boolean_and_unplanned_seeds_refuse(self):
  for seed in (0,False,3):
   with self.assertRaises(AssertionError):self.run_pair(seed)

class FrozenBinding(unittest.TestCase):
 def setUp(self):
  previous=b.read(ROOT/'calibration_deployment.json');self.auth=copy.deepcopy(b.read(previous['authorization_path']))
  self.auth.pop('checkpoint_path',None);self.auth.pop('checkpoint_SHA256',None)
  f=scope.frozen();ident=f['confirmation_identities']['1'];self.auth.update(cell=ident['cell'],allocation_id=ident['allocation_id'],attempt_id=ident['attempt_id'],label=ident['attempt_id'],operation='confirm_pure_ensemble',qualification_only=False,
   cap_seconds=21600,fixed_endpoint=40000,initial_completed_updates=0,new_model_updates=40000,parent_updates=[20000,20000],hybrid_config=None,variant_config=f['seed0_candidate']['variant_config'],pure_formulation='pg8_learned_ensemble75',ensemble_config=f['seed0_candidate']['ensemble_config'],ensemble_weight=.25,confirmation_freeze_SHA256=scope.sha(scope.FREEZE))
  research={str(p):b.sha(p) for p in [*ROOT.glob('*.py'),*(ROOT/'runtime').glob('*.py'),*(ROOT/'model').glob('*.py'),*scope.HERE.glob('*.py')]};self.auth['research_source_SHA256']=research;pins=self.auth['files_SHA256'];pins.update(research)
  paths=[scope.FREEZE,ROOT/'pure_incumbent.json',ROOT/'analysis/pure_confirmation_protocol.json',ROOT/'ensemble_parents.json',ROOT/'ensemble_protocol.json'];paths.extend(Path(q['path']) for q in f['matched_baselines'].values())
  for p in paths:pins[str(p)]=scope.sha(p)
 def test_exact_frozen_scientific_parity_and_both_identity_scopes(self):
  for seed in (1,2):
   f=scope.frozen();i=f['confirmation_identities'][str(seed)];a={**self.auth,'cell':i['cell'],'allocation_id':i['allocation_id'],'attempt_id':i['attempt_id'],'label':i['attempt_id']};scope.validate_authorization(a,require_proof=False)
  self.assertNotIn('torch',sys.modules)
 def test_frozen_configuration_and_update_limits_cannot_drift(self):
  for change in [{'ensemble_weight':.5},{'ensemble_config':{'learned_weight':.25,'mask_seed':1,'bags_per_stage':200}},{'cap_seconds':900},{'fixed_endpoint':20000},{'new_model_updates':20000},{'initial_completed_updates':20000},{'parent_updates':[10000,30000]},{'qualification_only':True},{'hybrid_config':{'pg_bag_fraction':.75}},{'confirmation_freeze_SHA256':'0'*64},{'pure_formulation':'latent8'}]:
   with self.subTest(change=change),self.assertRaises(AssertionError):scope.validate_authorization({**self.auth,**change},require_proof=False)
 def test_unplanned_repeat_seed_identity_or_warm_start_refuses(self):
  for change in [{'cell':{**self.auth['cell'],'seed':0}},{'cell':{**self.auth['cell'],'experiment':'repeat_seed1'}},{'checkpoint_path':'seed0.pt'},{'checkpoint_SHA256':'0'*64}]:
   with self.subTest(change=change),self.assertRaises(AssertionError):scope.validate_authorization({**self.auth,**change},require_proof=False)
 def test_changed_scientific_or_extension_source_pin_refuses(self):
  for path in [next(iter(scope.frozen()['scientific_source_SHA256'])),str(scope.HERE/'ensemble_confirmation_execution.py')]:
   auth=copy.deepcopy(self.auth);auth['research_source_SHA256'][path]='0'*64
   with self.assertRaises(AssertionError):scope.validate_authorization(auth,require_proof=False)
 def test_full_real_native_source_closure_with_private_proof_wire_fixture(self):
  # Private fixture supplies proof wire metadata only, never a native permission or allocation.
  with tempfile.TemporaryDirectory(prefix='alz-confirm-proof-fixture-') as temp:
   p=Path(temp)/'fixture-proof.json';p.write_text(json.dumps({'status':'PASSED_CPU_FROZEN_TWO_PARENT_CONFIRMATION','freeze_SHA256':scope.sha(scope.FREEZE),'native_confirmation_completed':False,'GPU_allocation_created':False,'source_SHA256':{str(scope.HERE/'ensemble_confirmation_execution.py'):scope.sha(scope.HERE/'ensemble_confirmation_execution.py')}}))
   auth=copy.deepcopy(self.auth);auth['files_SHA256'][str(p)]=scope.sha(p);auth['confirmation_qualification_SHA256']=scope.sha(p)
   with patch.object(scope,'PROOF',p):
    b.source_closure(auth,b.NATIVE_LAYOUT)
    changed=copy.deepcopy(auth);changed['files_SHA256'][str(scope.HERE/'ensemble_confirmation_execution.py')]='0'*64
    with self.assertRaises(b.Refusal):b.source_closure(changed,b.NATIVE_LAYOUT)
   self.assertNotIn('torch',sys.modules)
 def test_bad_confirmation_seed_remains_and_fails_stable_target(self):
  records=[{'seed':s,'energy_score':1.8,'matched_baseline_energy_score':2.,'cell_energy_distance':.2 if s!=2 else .4,'matched_baseline_cell_energy_distance':.3,'spread_ratio':1.} for s in (0,1,2)]
  result=stable_target(records);self.assertFalse(result['stable_joint_target_passed']);self.assertEqual(result['per_seed_joint_improvement'],[True,True,False])
if __name__=='__main__':unittest.main()
