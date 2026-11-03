import tempfile
from pathlib import Path
import unittest

from role_lease import RoleLease


class RoleProtection(unittest.TestCase):
    def test_second_alz_role_refuses_and_first_remains_held(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'alz-role.lock'
            with RoleLease(path) as first:
                with self.assertRaises(BlockingIOError):
                    with RoleLease(path):
                        self.fail('duplicate ALZ role admitted')
                first.check()
            with RoleLease(path) as released:
                released.check()

    def test_symlink_is_refused_without_touching_target(self):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / 'target'
            target.write_text('preserve')
            link = Path(directory) / 'role'
            link.symlink_to(target)
            with self.assertRaises(OSError):
                with RoleLease(link):
                    self.fail('role symlink admitted')
            self.assertEqual(target.read_text(), 'preserve')

    def test_replaced_path_invalidates_only_owned_role(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'role'
            with RoleLease(path) as lease:
                path.rename(path.with_suffix('.old'))
                path.touch()
                with self.assertRaisesRegex(RuntimeError, 'identity changed'):
                    lease.check()


if __name__ == '__main__':
    unittest.main()
