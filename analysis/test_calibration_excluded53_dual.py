"""Current-calibration exclusion cannot substitute or omit original outer donors."""
import copy,unittest
from score_calibration_excluded53_dual import validate_plan

def plan():
 ids=[f'd{i}' for i in range(57)];cal=ids[20:24];folds=[]
 for i,group in enumerate((ids[:20],ids[20:40],ids[40:])):folds.append(dict(fold=i,original_donors=group,retained_donors=[d for d in group if d not in cal],excluded_calibration_donors=[d for d in group if d in cal]))
 return dict(status='PRESPECIFIED_CALIBRATION_EXCLUDED53_SENSITIVITY_BEFORE_NEW_BENCHMARK_VALUES',preserve_original57_donor_primary_comparison=True,preserve_all45_fresh_model_fits=True,current_global_calibration_donors=cal,folds=folds,no_tuning_on_supplementary_scores=True,no_new_training=True,no_new_generated_counts=True),cal
class Exclusion(unittest.TestCase):
 def test_exact53subset_and_full_original57_are_preserved(self):
  p,c=plan();q=validate_plan(p,c);self.assertEqual([len(f['retained_donors']) for f in q['folds']],[20,16,17])
 def test_direct_calibration_leakage_is_rejected(self):
  p,c=plan();p['folds'][1]['retained_donors'].append(c[0])
  with self.assertRaises(ValueError):validate_plan(p,c)
 def test_favorable_additional_donor_removal_is_rejected(self):
  p,c=plan();p['folds'][0]['retained_donors'].pop()
  with self.assertRaises(ValueError):validate_plan(p,c)
 def test_primary_scope_replacement_or_target_driven_tuning_is_rejected(self):
  for key in ('preserve_original57_donor_primary_comparison','preserve_all45_fresh_model_fits','no_tuning_on_supplementary_scores','no_new_training'):
   p,c=plan();p[key]=False
   with self.assertRaises(ValueError):validate_plan(p,c)
 def test_changed_calibration_ids_or_duplicate_original_fold_donor_rejected(self):
  p,c=plan()
  with self.assertRaises(ValueError):validate_plan(p,c[:-1]+['unknown'])
  p,c=plan();p['folds'][2]['original_donors'][0]=p['folds'][0]['original_donors'][0];p['folds'][2]['retained_donors'][0]=p['folds'][0]['original_donors'][0]
  with self.assertRaises(ValueError):validate_plan(p,c)
if __name__=='__main__':unittest.main(verbosity=2)
