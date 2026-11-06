"""Resume only the never-launched tail of the audited, source-identical R4 queue."""
import hashlib,importlib.util,json,os,sys
from pathlib import Path
ROOT=Path('/ml2/alz');SOURCE=ROOT/'analysis/run_hpo500_r4.py'
EXPECTED='f512c1628f3efe45f05c5323b2fa1e9aa189a289952d8eb4b7e293e3fe89a6c6'
def main():
    assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
    assert hashlib.sha256(SOURCE.read_bytes()).hexdigest()==EXPECTED
    proof=json.loads((ROOT/'analysis/hpo500_queue_r4_continuation_review_20261106.json').read_text())
    assert proof['status']=='INSPECTED_QUEUE_ONLY_STOP_NO_SCIENTIFIC_RETRY'
    assert proof['old_driver_guard_worker_absent'] and proof['own_resource_registry_empty'] and proof['canonical_open_reservations']==0
    assert not (ROOT/'artifacts/allocations/CAL_GPU0_V1_h5_c021_search_f0_pg_s0').exists()
    spec=importlib.util.spec_from_file_location('hpo500_original_r4_queue',SOURCE)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    module.STATE=ROOT/'artifacts/hpo500-r4q1'
    original_report=module.report
    def report(kind,event,summary,details):
        return original_report(kind,event.replace('hpo500-r4-','hpo500-r4q1-'),summary,details)
    module.report=report
    # No changes to run_cell, search, budgets, deadlines, qualifiers or worker.
    module.main()
if __name__=='__main__':main()
