"""Real unfrozen confirmation must refuse without any model/data import."""
import sys,unittest
import preset_scope as s
class Prepared(unittest.TestCase):
 def test_actual_unfrozen_confirmation_refuses_before_model_or_allocation(self):
  self.assertEqual(s.read(s.PROTOCOL)['status'],'PREPARED_PENDING_COMPLETE_ORIGINAL12_AND_FOLLOWUP_DESIGN')
  with self.assertRaisesRegex(AssertionError,'Complete original12fit audit'):s.cell('u000',0,'search')
  self.assertFalse(s.read(s.PLAN)['enabled']);self.assertNotIn('torch',sys.modules)
if __name__=='__main__':unittest.main(verbosity=2)
