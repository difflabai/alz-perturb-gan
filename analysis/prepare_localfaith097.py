"""Freeze a bounded coordinate study from all four corrected baseline endpoints."""
import argparse,datetime,hashlib,json,subprocess,sys,time
from pathlib import Path
ROOT=Path('/ml2/alz');STUDY=ROOT/'analysis/localfaith097';sys.path.insert(0,str(ROOT/'analysis'))
from released097_local_search_design import design

def read(p):return json.loads(Path(p).read_bytes())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 parser=argparse.ArgumentParser();parser.add_argument('--baseline-audit',type=Path,required=True);args=parser.parse_args()
 assert not subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True)
 p=read(STUDY/'protocol.json');assert p['status']=='PREPARED_PENDING_COMPLETE_CORRECTED_BASELINES_AND_LOCAL_DESIGN'
 assert not read(ROOT/'localfaith097_runtime/plan.json')['enabled'] and not list((STUDY/'proposals').glob('*.json'))
 a=read(args.baseline_audit);now=time.time();window=ROOT/'analysis/research_window_20261104_five_days.json';forecast=ROOT/'analysis/benchmark097_historical_cost_forecast_20261107.json'
 frozen=design(a,now,read(window)['expires_unix'],read(forecast)['estimated_matrix_GPU_hours'])
 assert frozen['candidates'],'No forecast slack for useful complete local study; prioritize fresh confirmation and full45fit matrix'
 paths=[args.baseline_audit.resolve(),Path(__file__).resolve(),ROOT/'analysis/released097_local_search_design.py',ROOT/'analysis/released097_local_search_design_CPU_qualification_v2.json',ROOT/'analysis/released097_selection.py',ROOT/'analysis/released097_external_selection.py',forecast,window,STUDY/'selection.py']
 for path,pin in a['source_SHA256'].items():assert sha(path)==pin;paths.append(Path(path))
 for key in ('protocol','report'):
  path=Path(a[key+'_path']);assert sha(path)==a[key+'_SHA256'];paths.append(path)
 controls=[]
 for attempt,pin in a['result_SHA256'].items():
  path=Path(a['report_path']).parent/'results'/(attempt+'.json');assert sha(path)==pin;paths.append(path);r=read(path)
  controls.append(dict(r,read_only_previously_measured_reference=True,new_model_fitted=False,new_GPU_charge=0,source_result_path=str(path),source_result_SHA256=sha(path)))
 assert len(controls)==4 and {(r['candidate_id'],r['seed']) for r in controls}=={(c,s) for c in ('f000','f001') for s in (0,1)}
 native_proof=Path(a['model_parity_proof_path']);assert sha(native_proof)==a['model_parity_proof_SHA256'];paths.append(native_proof)
 for path,pin in read(ROOT/'analysis/released097_local_search_design_CPU_qualification_v2.json')['source_SHA256'].items():assert sha(path)==pin
 project=Path('/ml2/codex-storage/future-workspaces/2026-10-27/task-2/benchmark');assert not read(project/'project_gpu_budget.json')['open_reservations']
 dest=STUDY/'frozen_design.json';assert not dest.exists();dest.write_text(json.dumps(frozen,indent=2)+'\n')
 p.update(status='FROZEN_RELEASED097_FULL20K_LOCAL_COORDINATE_SEARCH',UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),baseline_audit_path=str(args.baseline_audit.resolve()),
   design_now_unix=now,matrix_cost_forecast_path=str(forecast),candidates=frozen['candidates'],reference_controls=sorted(controls,key=lambda r:(r['candidate_id'],r['seed'])),maximum_new_fits=frozen['required_new_fits'],
   design_evidence_SHA256={str(x):sha(x) for x in paths},selection_rule='Compare every frozen neighbor at both full20k seeds0/1; include explicitly previously measured corrected f000/f001 controls. Rank complete paired mean cell distance then donor energy; retain failures, sampleSD, spread and coverage. New weights/optimizers at0 for every new fit.',
   confirmation_rule='Distinct primary-distance, joint-quality and released-reference choices require fresh full20k initialization seeds2..4; no exploratory score is independent confirmation.')
 (STUDY/'protocol.json').write_text(json.dumps(p,indent=2)+'\n')
 for c in frozen['candidates']:
  q=dict(id=c['id'],hyperparameters=c['hyperparameters'],native_version='0.9.7',endpoint=20000,fold=0,training_seeds=[0,1],initial_completed_updates=0,from_scratch=True,source_checkpoint=None,heldout_results_used=False,protocol_SHA256=sha(STUDY/'protocol.json'))
  path=STUDY/'proposals'/(c['id']+'.json');assert not path.exists();path.write_text(json.dumps(q,indent=2)+'\n')
 print(json.dumps(dict(status=p['status'],required_new_fits=p['maximum_new_fits'],reference_fits_refitted=False,activation=False,GPU_allocation_created=False,forecast=frozen['forecast'])))
if __name__=='__main__':main()
