import sys,unittest
import prepare,preset_scope as s
class Prepared(unittest.TestCase):
 def test_actual_unfinished_paired_study_cannot_admit_new_followup(self):
  self.assertEqual(s.read(s.PROTOCOL)['status'],'PREPARED_PENDING_ALL20_PAIRED_REFINEMENT_ENDPOINTS_AND_AUDIT');self.assertFalse(s.read(s.PLAN)['enabled'])
  with self.assertRaisesRegex(AssertionError,'All20 paired selection endpoints'):prepare.build_authorization(check_only=True)
  self.assertNotIn('torch',sys.modules);self.assertNotIn('scvi',sys.modules)
if __name__=='__main__':unittest.main(verbosity=2)
