"""Real premature admission must fail before model imports or allocation."""
import sys,unittest
import prepare,preset_scope as s
class Prepared(unittest.TestCase):
 def test_actual_pending_reference_confirmation_refuses(self):
  self.assertEqual(s.read(s.PROTOCOL)['status'],'PREPARED_PENDING_COMPLETE_REFERENCE_CONFIRMATION_AUDIT')
  self.assertFalse(s.read(s.PLAN)['enabled'])
  with self.assertRaisesRegex(AssertionError,'Complete original reference confirmation'):prepare.build_authorization(check_only=True)
  self.assertNotIn('torch',sys.modules);self.assertNotIn('scvi',sys.modules)
if __name__=='__main__':unittest.main(verbosity=2)
