"""Process exits may defer observation; missing pinned inputs still stop admission."""
import errno
import importlib.util
import pathlib
import unittest

ROOT = pathlib.Path('/ml2/alz')
spec = importlib.util.spec_from_file_location('metadata_v7', ROOT / 'analysis/continue_benchmark097_after_capacity_v7.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


class MissingMetadata(unittest.TestCase):
    def test_numeric_process_leaf_exits_are_narrowly_identified(self):
        for leaf in ('stat', 'status', 'cmdline', 'exe'):
            self.assertTrue(m.transient_proc_exit(FileNotFoundError(errno.ENOENT, 'gone', '/proc/123/' + leaf)))

    def test_missing_pinned_inputs_and_other_paths_are_not_deferred(self):
        for path in ('/ml2/alz/analysis/protocol.json', '/proc/meminfo', '/proc/123/other', '/proc/self/stat', '/proc/0/stat', '/proc/123/status/extra'):
            self.assertFalse(m.transient_proc_exit(FileNotFoundError(errno.ENOENT, 'missing', path)))

    def test_unknown_missing_filename_is_not_assumed_to_be_process_exit(self):
        self.assertFalse(m.transient_proc_exit(FileNotFoundError(errno.ENOENT, 'missing')))

    def test_wrong_error_kind_is_not_a_transient_exit(self):
        self.assertFalse(m.transient_proc_exit(PermissionError(errno.EACCES, 'denied', '/proc/123/stat')))
        self.assertFalse(m.transient_proc_exit(FileNotFoundError(errno.EACCES, 'denied', '/proc/123/stat')))

    def run_choice_failure(self, exception):
        old = dict(cell=dict(id='healthy'), gpu=m.GPUS[1], identity={}, attempt='healthy')
        starts = []
        def choice(*args):
            raise exception
        result = m.drive([dict(id='untouched')], [], {m.GPUS[1]: old}, key=lambda v: v['id'],
                         start=lambda *args: starts.append(args), ready=lambda h: True,
                         finish=lambda h: dict(cell=h['cell'], complete=True), select=choice,
                         observe=lambda *args: None, sleep=lambda seconds: None, now=lambda: 0, expiry=1000)
        self.assertFalse(starts)
        self.assertEqual(result['pending'], [dict(id='untouched')])
        self.assertEqual(len(result['rows']), 1)
        self.assertFalse(result['unsettled'])
        self.assertEqual(result['errors'][0]['stage'], 'choice')
        self.assertIn('FileNotFoundError', result['errors'][0]['traceback'])
        return result['errors'][0]

    def test_pinned_missing_file_records_path_and_drains_healthy_job(self):
        path = '/ml2/alz/analysis/pinned-protocol.json'
        error = self.run_choice_failure(FileNotFoundError(errno.ENOENT, 'missing', path))
        self.assertEqual(error['filename'], path)

    def test_unknown_missing_file_still_stops_without_a_native_retry(self):
        error = self.run_choice_failure(FileNotFoundError(errno.ENOENT, 'missing'))
        self.assertIsNone(error['filename'])


if __name__ == '__main__':
    unittest.main()
