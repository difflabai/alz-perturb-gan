"""Exercise real scorer/count verifier on confined synthetic arrays; no models/native jobs."""
import contextlib,copy,importlib.util,io,json,sys,unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
ROOT=Path('/ml2/alz');sys.path[:0]=[str(ROOT/'analysis'),str(ROOT/'recovery_runtime'),str(ROOT)]
import test_recovery_confirmation_artifacts as fixture_base
import aggregate_validation as common
import aggregate_pure_confirmation as base_aggregate
import ensemble_confirmation_scope as original_scope
import recovery_scope as scope
import calibration
spec=importlib.util.spec_from_file_location('recovery_saved_array_verifier_target',ROOT/'recovery_runtime/finish.py');finish=importlib.util.module_from_spec(spec);spec.loader.exec_module(finish)

class SavedArrayVerification(unittest.TestCase):
 def setUp(self):
  self.fx=fixture_base.RecoveryArtifacts('test_complete_private_wire_artifacts_aggregate_without_touching_real_incumbents');self.fx.setUp();self.addCleanup(self.fx.tearDown)
  f=self.fx;r=f.root;self.data=r/'data/calibration_fold0';rng=np.random.default_rng(20261109)
  summary=rng.normal(size=(4,26)).astype(np.float32);summary[:,:6]=np.sqrt(rng.dirichlet(np.ones(6),size=4))
  np.savez_compressed(self.data/'real_summaries.npz',summary=summary,stage=np.array([3,1,0,2]),donors=np.array(['A','B','C','D']))
  cells=rng.normal(size=(4,375,50)).astype(np.float32);np.savez_compressed(self.data/'cell_reservoir.npz',by_stage=cells,equal_stage=cells.reshape(-1,50))
  files={p.name:common.sha(p) for p in [self.data/'real_summaries.npz',self.data/'cell_reservoir.npz']};f.write(self.data/'manifest.json',{'status':'CPU_CALIBRATION_PROJECTED_WITHOUT_REFITTING','files_SHA256':files,'fixture':'SYNTHETIC_ARRAYS_NOT_ALZ_DATA'})
  data_pin=common.sha(self.data/'manifest.json');out=Path(f.deployments[1]['output_dir']);self.records=[]
  with patch.object(calibration,'DATA',self.data):
   for rep in (0,1):
    folder=out/f'r{rep}';folder.mkdir()
    y=rng.normal(size=(4,200,26)).astype(np.float32);y[:,:,:6]=np.sqrt(rng.dirichlet(np.ones(6),size=(4,200)))
    generated=rng.normal(size=(4,375,50)).astype(np.float32);np.savez_compressed(folder/'arrays.npz',stage_synthetic_summaries=y,generated_cells_by_stage=generated)
    metrics=calibration.score(y,generated);metrics.update(status='completed',method='pg_pure_ensemble75',training_seed=1,generation_replicate=rep,
       trace_manifest_SHA256=f.candidates[1]['trace_manifest_SHA256'],sampling_contract=f.candidates[1]['sampling_contract'],generated_zero_count_rate=0.,arrays_SHA256=common.sha(folder/'arrays.npz'))
    f.write(folder/'metrics.json',metrics);self.records.append(metrics)
    for stage in range(4):np.savez_compressed(folder/f'stage{stage}_exemplar.npz',counts=np.ones((1,400,2000),dtype=np.float32))
   avg=calibration.average(self.records)
  f.candidates[1].update(avg)
  for candidate in f.candidates:candidate['calibration_manifest_SHA256']=data_pin
  for seed,baseline in enumerate(f.baselines):
   baseline['calibration_manifest_SHA256']=data_pin;f.write(r/'analysis'/f'baseline_calibration_seed{seed}.json',baseline)
  f.write(r/'pure_incumbent.json',{**f.candidates[0],'candidate_commit':'PRIVATE_SAVED_ARRAY_FIXTURE'})
  decision=r/'analysis'/(f.attempt0+'-result.json');f.write(decision,{'candidate':f.candidates[0],'decision':{'accept':True},'independent_saved_scores_checkpoint_and_count_exemplars_verified':True})
  data_paths=[self.data/'manifest.json',self.data/'real_summaries.npz',self.data/'cell_reservoir.npz']
  auth0=f.auths[0]
  for p in data_paths:auth0['files_SHA256'][str(p)]=common.sha(p)
  f.write(f.deployments[0]['authorization_path'],auth0);f.deployments[0]['authorization_SHA256']=common.sha(f.deployments[0]['authorization_path']);f.write(Path(f.deployments[0]['authorization_path']).with_name('deployment.json'),f.deployments[0])
  frozen=common.read(f.freeze);frozen.update(seed0_candidate=f.candidates[0],pure_incumbent_SHA256=common.sha(r/'pure_incumbent.json'),seed0_decision_SHA256=common.sha(decision),seed0_authorization_SHA256=common.sha(f.deployments[0]['authorization_path']))
  frozen['matched_baselines']={str(seed):{'path':str(r/'analysis'/f'baseline_calibration_seed{seed}.json'),'SHA256':common.sha(r/'analysis'/f'baseline_calibration_seed{seed}.json')} for seed in (0,1,2)};f.write(f.freeze,frozen)
  history_path=r/'analysis/interrupted_seed1_source_history.json';history=common.read(history_path);old_auth=common.read(history['authorization_path'])
  for p in [*data_paths,f.freeze,r/'pure_incumbent.json']:old_auth['files_SHA256'][str(p)]=common.sha(p)
  f.write(history['authorization_path'],old_auth);history.update(authorization_SHA256=common.sha(history['authorization_path']),original_freeze_SHA256=common.sha(f.freeze));f.write(history_path,history)
  protocol=common.read(f.recovery_protocol);protocol.update(original_confirmation_freeze_SHA256=common.sha(f.freeze),frozen_baseline_pins=frozen['matched_baselines']);f.write(f.recovery_protocol,protocol)
  proof=common.read(f.recovery_cpu);proof['recovery_protocol_SHA256']=common.sha(f.recovery_protocol);f.write(f.recovery_cpu,proof)
  for seed in (1,2):
   candidate=f.candidates[seed];dep=f.deployments[seed];auth=f.auths[seed]
   ref=common.read(candidate['checkpoint_path']);ref.update(recovery_protocol_SHA256=common.sha(f.recovery_protocol),original_confirmation_freeze_SHA256=common.sha(f.freeze));f.write(candidate['checkpoint_path'],ref)
   candidate.update(checkpoint_SHA256=common.sha(candidate['checkpoint_path']),original_confirmation_freeze_SHA256=common.sha(f.freeze),recovery_protocol_SHA256=common.sha(f.recovery_protocol),qualification_only=False,generation='iter_pg_whole_bag_ensemble')
   auth.update(recovery_protocol_SHA256=common.sha(f.recovery_protocol),recovery_CPU_proof_SHA256=common.sha(f.recovery_cpu))
   paths=[*data_paths,f.freeze,r/'pure_incumbent.json',history_path,Path(history['authorization_path']),f.recovery_protocol,f.recovery_cpu,*[Path(q['path']) for q in frozen['matched_baselines'].values()]]
   for p in paths:auth['files_SHA256'][str(p)]=common.sha(p)
   f.write(dep['authorization_path'],auth);dep['authorization_SHA256']=common.sha(dep['authorization_path']);f.write(Path(dep['authorization_path']).with_name('deployment.json'),dep);f.write(Path(dep['output_dir'])/'completed.json',candidate)
   audit=r/'analysis'/f'checkpoint-seed{seed}-fixture.json';q=common.read(audit);q.update(checkpoint_SHA256=candidate['checkpoint_SHA256'],recovery_protocol_SHA256=common.sha(f.recovery_protocol));f.write(audit,q)
   # finish.py uses its deterministic independent-audit filename.
   f.write(r/'analysis'/(f.ids[str(seed)]['attempt_id']+'-checkpoint-audit.json'),q)
  self.attempt=f.ids['1']['attempt_id'];self.output=r/'analysis'/(self.attempt+'-confirmation.json');self.output.unlink();self.out=out
 @contextlib.contextmanager
 def environment(self):
  f=self.fx
  values=[(finish,'ROOT',f.root),(common,'ROOT',f.root),(base_aggregate,'ROOT',f.root),(original_scope,'ROOT',f.root),(original_scope,'HERE',f.extension),(original_scope,'FREEZE',f.freeze),(original_scope,'PROOF',f.proof),
   (scope,'ROOT',f.root),(scope,'HERE',f.recovery),(scope,'PROTOCOL',f.recovery_protocol),(scope,'CPU_PROOF',f.recovery_cpu),(scope,'NATIVE_PROOF',f.native_restore),(calibration,'DATA',self.data)]
  with contextlib.ExitStack() as stack:
   for module,name,value in values:stack.enter_context(patch.object(module,name,value))
   stack.enter_context(contextlib.redirect_stdout(io.StringIO()));yield
 def run_verifier(self):
  with self.environment():finish.verify(self.attempt)
 def test_real_score_recomputation_and_count_checks_complete_without_selection(self):
  self.run_verifier();q=common.read(self.output)
  self.assertTrue(q['independently_verified_saved_scores_counts_and_checkpoint']);self.assertTrue(q['incumbent_unchanged']);self.assertEqual(len(q['array_and_exemplar_SHA256']),10)
  self.assertAlmostEqual(q['candidate']['energy_score'],np.mean([r['energy_score'] for r in self.records]),places=12)
  with self.assertRaises(AssertionError):self.run_verifier()
 def test_falsified_metric_score_refuses_without_confirmation(self):
  p=self.out/'r0/metrics.json';q=common.read(p);q['energy_score']+=.1;self.fx.write(p,q)
  with self.assertRaises(AssertionError):self.run_verifier()
  self.assertFalse(self.output.exists())
 def test_noninteger_count_exemplar_refuses(self):
  p=self.out/'r0/stage0_exemplar.npz'
  with np.load(p,allow_pickle=False) as arrays:x=arrays['counts'].copy()
  x[0,0,0]=.5;np.savez_compressed(p,counts=x)
  with self.assertRaises(AssertionError):self.run_verifier()
  self.assertFalse(self.output.exists())
 def test_missing_sampling_replicate_refuses(self):
  (self.out/'r1/arrays.npz').unlink()
  with self.assertRaises(FileNotFoundError):self.run_verifier()
  self.assertFalse(self.output.exists())
 def test_changed_donor_order_refuses(self):
  p=self.out/'r0/metrics.json';q=common.read(p);q['donors']=q['donors'][::-1];self.fx.write(p,q)
  with self.assertRaises(AssertionError):self.run_verifier()
  self.assertFalse(self.output.exists())
 def test_falsified_average_score_refuses(self):
  p=self.out/'completed.json';q=common.read(p);q['energy_score']+=.1;self.fx.write(p,q)
  with self.assertRaises(AssertionError):self.run_verifier()
  self.assertFalse(self.output.exists())
if __name__=='__main__':unittest.main()
