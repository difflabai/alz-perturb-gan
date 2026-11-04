"""CPU-only refusals for cumulative-charge proposal arithmetic and canonical evidence."""
import copy,json,math,sys,tempfile,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).parent))
import prepare_compute_time_alternative as p
class ComputeBudget(unittest.TestCase):
 def setUp(self):
  self.expected=['original','qual1','resume1','qual2','resume2','qual3','resume3'];self.rows=[{'attempt':a,'charged_GPU_seconds':v} for a,v in zip(self.expected,[637.598511926,9.511720639,1407.534308955,7.851645384,1063.812267228,14.921552764,1952.25])]
 def test_all_segments_and_qualifications_reduce_single_sixhour_allowance(self):
  b=p.budget(self.rows,self.expected);self.assertAlmostEqual(b['retained_canonical_charge_GPU_seconds'],5093.480006896);self.assertLess(b['safe_training_cap_before_actual_qualification_settlement_seconds'],21600-5093-900);self.assertLessEqual(b['retained_canonical_charge_GPU_seconds']+900+b['safe_training_cap_before_actual_qualification_settlement_seconds']+30,21600)
 def test_no_qualification_or_training_segment_may_be_dropped(self):
  for i in range(len(self.rows)):
   with self.assertRaises(AssertionError):p.budget(self.rows[:i]+self.rows[i+1:],self.expected)
 def test_duplicate_or_reordered_charge_refuses(self):
  for rows in [self.rows+[self.rows[0]],list(reversed(self.rows))]:
   with self.assertRaises(AssertionError):p.budget(rows,self.expected)
  with self.assertRaises(AssertionError):p.budget(self.rows+self.rows[:1],self.expected+self.expected[:1])
 def test_negative_nonfinite_string_and_bool_charges_refuse(self):
  for value in [-1,float('nan'),float('inf'),'2',True]:
   rows=copy.deepcopy(self.rows);rows[0]['charged_GPU_seconds']=value
   with self.assertRaises(AssertionError):p.budget(rows,self.expected)
 def test_no_fresh_sixhour_grant_when_cumulative_allowance_exhausted(self):
  rows=copy.deepcopy(self.rows);rows[0]['charged_GPU_seconds']=21600
  with self.assertRaises(AssertionError):p.budget(rows,self.expected)
 def test_fractional_charges_never_round_up_training_to_exceed_limit(self):
  rows=[{'attempt':'one','charged_GPU_seconds':1000.00001}];b=p.budget(rows,['one']);self.assertEqual(b['safe_training_cap_before_actual_qualification_settlement_seconds'],19669)
class CanonicalEvidence(unittest.TestCase):
 def setUp(self):
  self.tmp=tempfile.TemporaryDirectory();self.root=Path(self.tmp.name);self.ledger=self.root/'ledger.json';self.settlement=self.root/'settlement.json';self.l={'jobs':[{'label':'one','status':'failed','reservation_released':True,'charged_seconds':12.5}]};self.s={'attempt_id':'one','uncertain':None,'settlement':{'status':'settled','charged_GPU_seconds':12.5}}
 def tearDown(self):self.tmp.cleanup()
 def check(self):
  self.ledger.write_text(json.dumps(self.l));self.settlement.write_text(json.dumps(self.s));return p.canonical_row('one',self.ledger,self.settlement)
 def test_exact_failed_but_released_settled_charge_is_retained(self):self.assertEqual(self.check()['charged_GPU_seconds'],12.5)
 def test_unreleased_or_active_job_refuses(self):
  for change in [{'reservation_released':False},{'status':'active'}]:
   original=copy.deepcopy(self.l);self.l['jobs'][0].update(change)
   with self.assertRaises(AssertionError):self.check()
   self.l=original
 def test_uncertain_or_unsettled_canonical_record_refuses(self):
  for change in [{'uncertain':'missingworker'},{'settlement':{'status':'reserved','charged_GPU_seconds':12.5}}]:
   original=copy.deepcopy(self.s);self.s.update(change)
   with self.assertRaises(AssertionError):self.check()
   self.s=original
 def test_mismatched_attempt_or_charge_refuses(self):
  for change in [{'attempt_id':'other'},{'settlement':{'status':'settled','charged_GPU_seconds':0}}]:
   original=copy.deepcopy(self.s);self.s.update(change)
   with self.assertRaises(AssertionError):self.check()
   self.s=original
if __name__=='__main__':unittest.main()
