"""Qualify the real frozen candidate and all45 cells; never a synthetic selection."""
import datetime,json,os,re,subprocess,sys
from pathlib import Path
HERE=Path(__file__).resolve().parent;sys.path.insert(0,str(HERE))
import binding as b,preset_scope as s
def main():
 assert os.environ.get('CUDA_VISIBLE_DEVICES')=='' and not s.read(s.PLAN)['enabled']
 protocol=s.protocol();assert not s.RUNTIME_PROOF.exists()
 results=[];count=0
 for name in ('test_integration.py','test_scope_actual.py','test_factory.py'):
  cmd=[sys.executable,'-I','-S','-B','-c','import sys,runpy;sys.path.insert(0,sys.argv[1]);p=sys.argv[2];sys.argv=[p];runpy.run_path(p,run_name="__main__")',str(HERE),str(HERE/name)]
  p=subprocess.run(cmd,capture_output=True,text=True,timeout=300);print(p.stderr)
  results.append(dict(file=name,returncode=p.returncode,stdout=p.stdout,stderr=p.stderr));assert p.returncode==0,name
  count+=int(re.search(r'Ran (\d+) tests?',p.stderr).group(1))
 assert count==24 and 'torch' not in sys.modules
 paths=[*HERE.glob('*.py'),*s.MODEL_DIR.glob('*.py'),*s.CFGEN_DIR.glob('*.py'),HERE/'policy.prepared.json',
        *s.protocol_paths(protocol),s.ROOT/'analysis/run_benchmark097.py',s.ROOT/'analysis/duration096/native_fixture.py',
        s.ROOT/'analysis/benchmark097/retained_PCs_CPU_qualification.json',s.ROOT/'analysis/benchmark097/aggregation_CPU_qualification.json']
 # Plan activation is deliberately a separate one-file commit; all other
 # source/evidence must remain byte-identical after this actual qualification.
 paths=[p for p in paths if p!=s.PLAN]
 q=dict(status='EXTERNAL097_ALL45_FRESH_CELLS_SCOPE_ACCOUNTING_CPU_PASS',UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),
   actual_tests=count,actual_frozen_PG_candidate_qualified=True,PG_candidate_id=protocol['PG_candidate_id'],
   actual_all45_authorizations_and_source_pins_checked=True,all_new_models_from_update_zero=True,
   fullwidth_native_optimizer_pairs=30,actual_all15_PG_fold_seed_initializations_qualified=True,
   native_model_source_unchanged=True,identical_next_pair_state_at500_and20000_limits=True,
   actual_CFGen_role_confined_controller_settlement_test=True,tests=results,GPU_allocation_created=False,CUDA_initialized=False,
   source_SHA256={str(p):b.sha(p) for p in paths},
   limitation='Native CPU optimizer qualification and actual authorization/source closure; full fresh CUDA model endpoints and external outcomes remain unmeasured.')
 s.RUNTIME_PROOF.write_text(json.dumps(q,indent=2,allow_nan=False)+'\n')
 print(json.dumps({k:q[k] for k in ('status','actual_tests','PG_candidate_id','fullwidth_native_optimizer_pairs','GPU_allocation_created')}))
if __name__=='__main__':main()
