"""Previously measured controls stay separate from new optimizer trajectories."""
import copy,importlib.util,unittest
from pathlib import Path
from test_released097_selection import fixture,H
path=Path(__file__).parent/'localfaith097/selection.py';spec=importlib.util.spec_from_file_location('prospective_local_selector',path);selector=importlib.util.module_from_spec(spec);spec.loader.exec_module(selector)
def inputs():
 _,raw=fixture();controls=[]
 for row in raw:
  q=copy.deepcopy(row);q['candidate_id']='f000' if q['candidate_id']=='a' else 'f001';q.update(read_only_previously_measured_reference=True,new_model_fitted=False,new_GPU_charge=0,source_result_path='/fixture/'+q['candidate_id']+str(q['seed'])+'.json',source_result_SHA256='fixture',charged_GPU_seconds=1400.);controls.append(q)
 candidates=[dict(id='t000',hyperparameters=dict(H,prior_reg=.003)),dict(id='t001',hyperparameters=dict(H,prior_reg=.03))];rows=[]
 for c in candidates:
  for seed in (0,1):
   q=copy.deepcopy(raw[seed]);q.update(candidate_id=c['id'],hyperparameters=c['hyperparameters'],charged_GPU_seconds=1400.);q['metrics']['cell_energy_distance']=.15 if c['id']=='t000' else .25;rows.append(q)
 return candidates,rows,controls
class LocalSelection(unittest.TestCase):
 def test_complete_new_pairs_rank_with_labeled_zero_new_charge_controls(self):
  q=selector.summarize_local(*inputs());self.assertEqual(q['required_new_fits'],4);self.assertEqual(q['previously_measured_control_fits'],4);self.assertEqual(q['ranking'][0],'t000');self.assertTrue(all(c['previously_measured_control']==(c['candidate_id'] in ('f000','f001')) for c in q['candidates']))
 def test_reference_refitting_or_new_charge_cannot_be_silently_counted(self):
  for key,val in [('new_model_fitted',True),('new_GPU_charge',1400),('read_only_previously_measured_reference',False)]:
   c,r,b=inputs();b[0][key]=val
   with self.assertRaises(ValueError):selector.summarize_local(c,r,b)
 def test_missing_control_or_new_seed_or_duplicate_refused(self):
  c,r,b=inputs()
  for new,controls in [(r[:-1],b),(r,b[:-1]),(r+[r[0]],b)]:
   with self.assertRaises(ValueError):selector.summarize_local(c,new,controls)
 def test_old_result_cannot_replace_a_new_fit(self):
  for key,val in [('read_only_previously_measured_reference',True),('new_model_fitted',False),('charged_GPU_seconds',0),('from_scratch',False)]:
   c,r,b=inputs();r[0][key]=val
   with self.assertRaises(ValueError):selector.summarize_local(c,r,b)
 def test_failed_seed_is_retained_and_cannot_win(self):
  c,r,b=inputs();r[0].update(finite=False,metrics=None,failed_candidate_not_retried=True,failure=dict(error='FloatingPointError(nonfinite)'));q=selector.summarize_local(c,r,b);self.assertNotIn('t000',q['ranking']);self.assertEqual(next(x for x in q['candidates'] if x['candidate_id']=='t000')['failed_seeds'],[0])
if __name__=='__main__':unittest.main(verbosity=2)
