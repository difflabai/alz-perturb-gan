"""Use unchanged57/53 numerical analyses after the disclosed47-attempt lineage gate."""
import argparse,json,os,sys
from pathlib import Path
ROOT=Path('/ml2/alz');sys.path.insert(0,str(ROOT/'analysis'))
import benchmark097_recovered_lineage as lineage
def main():
 p=argparse.ArgumentParser();p.add_argument('--sensitivity',action='store_true');args=p.parse_args();assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
 # This metadata-only gate runs before importing scientific target loaders.
 lineage.load_complete_audits()
 import aggregate_benchmark097_balanced as core
 core.load_complete_audits=lineage.load_complete_audits
 if args.sensitivity:
  import score_calibration_excluded53_balanced as sensitivity
  sys.argv=[sys.argv[0]];sensitivity.main();path=ROOT/'analysis/benchmark097_calibration_excluded53_report.json'
 else:core.main();path=ROOT/'analysis/benchmark097/comparison.json'
 q=lineage.read(path);q.update(physical_attempts=47,successful_fresh_models=45,infrastructure_failed_attempts=list(lineage.FAILED),failed_infrastructure_GPU_seconds=lineage.read(ROOT/'analysis/benchmark097/matrix_complete.json')['failed_infrastructure_GPU_seconds'],scientific_protocol_unchanged=True,recovery_quality_reselection=False)
 q['source_SHA256'].update({str(p):lineage.sha(p) for p in (Path(__file__).resolve(),Path(lineage.__file__).resolve())});path.write_text(json.dumps(q,indent=2,allow_nan=False)+'\n')
if __name__=='__main__':main()
