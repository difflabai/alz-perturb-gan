"""A transient resource forecast can wait; uncertainty and expiry cannot retry."""
import importlib.util,json,tempfile,unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
path=Path('/ml2/alz/analysis/run_hpo500_r4.py')
spec=importlib.util.spec_from_file_location('hpo500_r4_wait_test',path)
driver=importlib.util.module_from_spec(spec);spec.loader.exec_module(driver)

class Wait(unittest.TestCase):
    def metadata(self,path):
        if str(path).endswith('protocol.json'):return {'research_window_path':'/unused/window.json'}
        if str(path).endswith('parent420_CPU_audit.json'):return {'parent_expiry_unix':900}
        return {'expires_unix':1000}
    def test_capacity_only_repeats_readonly_preflight_with_original_expiry(self):
        responses=[SimpleNamespace(returncode=1,stderr='AssertionError: Exact parallel GPU growth forecasts do not fit'),SimpleNamespace(returncode=0,stderr='')]
        with tempfile.TemporaryDirectory() as folder,patch.object(driver,'STATE',Path(folder)),patch.object(driver,'read',side_effect=self.metadata),patch.object(driver.time,'time',return_value=100),patch.object(driver.time,'sleep') as sleep,patch.object(driver.subprocess,'run',side_effect=responses) as run:
            driver.wait_for_capacity('c008',0,'search',True)
            self.assertEqual(run.call_count,2);sleep.assert_called_once_with(15)
            self.assertEqual(run.call_args_list[0].args,run.call_args_list[1].args)
            args=run.call_args.args[0];self.assertIn('--check-only',args);self.assertIn('--resume',args)
            state=json.loads((Path(folder)/'progress.json').read_text())
            self.assertEqual(state['original_expiry_unix'],900)
            self.assertEqual(state['status'],'WAITING_FOR_SHARED_CAPACITY_NO_ALLOCATION')
    def test_source_or_telemetry_error_is_terminal_without_retry(self):
        with patch.object(driver,'read',side_effect=self.metadata),patch.object(driver.time,'time',return_value=100),patch.object(driver.time,'sleep') as sleep,patch.object(driver.subprocess,'run',return_value=SimpleNamespace(returncode=1,stderr='Source pin differs')) as run:
            with self.assertRaisesRegex(RuntimeError,'Source pin differs'):driver.wait_for_capacity('c008',0,'search',True)
            self.assertEqual(run.call_count,1);sleep.assert_not_called()
    def test_original_continuation_expiry_refuses_before_preflight(self):
        with patch.object(driver,'read',side_effect=self.metadata),patch.object(driver.time,'time',return_value=751),patch.object(driver.subprocess,'run') as run:
            with self.assertRaisesRegex(AssertionError,'Original trial'):driver.wait_for_capacity('c008',0,'search',True)
            run.assert_not_called()
if __name__=='__main__':unittest.main(verbosity=2)
