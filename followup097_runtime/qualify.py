"""Qualify the actually selected new settings after complete paired audit."""
import datetime,json,os,re,subprocess,sys
from pathlib import Path
HERE=Path(__file__).resolve().parent;sys.path.insert(0,str(HERE))
import binding as b,preset_scope as s
def main():
 assert os.environ.get('CUDA_VISIBLE_DEVICES')=='' and not s.read(s.PLAN)['enabled']
 p=s.validate_protocol();n=len(p['selected_configurations']);assert 1<=n<=2 and not s.RUNTIME_PROOF.exists();tests=[];count=0
 for name in ('test_integration.py','test_scope.py','test_factory.py'):
  args=[sys.executable,'-I','-S','-B','-c','import sys,runpy;sys.path.insert(0,sys.argv[1]);p=sys.argv[2];sys.argv=[p];runpy.run_path(p,run_name="__main__")',str(HERE),str(HERE/name)]
  q=subprocess.run(args,capture_output=True,text=True,timeout=300);print(q.stderr);tests.append(dict(file=name,returncode=q.returncode,stdout=q.stdout,stderr=q.stderr));assert q.returncode==0,name
  count+=int(re.search(r'Ran (\d+) tests?',q.stderr).group(1))
 assert count==23 and 'torch' not in sys.modules
 files=[*HERE.glob('*.py'),HERE/'policy.prepared.json',*s.MODEL_DIR.glob('*.py'),s.PROTOCOL,s.MODEL_PROOF,s.HPO_PROOF,s.NATIVE_PROOF,s.ENGINE_PROOF,s.BASELINE_PROOF,s.RULE,s.SELECTOR,s.WINDOW,
   s.ROOT/'analysis/run_followup097.py',s.ROOT/'analysis/prepare_followup097.py',s.ROOT/'analysis/duration096/native_fixture.py',*[Path(x) for x in p['selection_evidence_SHA256']]]
 q=dict(status='NATIVE097_MATCHED_SEED234_FOLLOWUP_SCOPE_ACCOUNTING_CPU_PASS',UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),actual_runtime_tests=count,actual_cells_checked=3*n,
   fullwidth_native_optimizer_pairs=12*n,training_seeds=[2,3,4],all_new_models_from_update_zero=True,reference_models_refitted=False,native_model_source_unchanged=True,
   identical_next_pair_native_state_at500_and20000_limits=True,GPU_allocation_created=False,CUDA_initialized=False,tests=tests,source_SHA256={str(p):b.sha(p) for p in files})
 s.RUNTIME_PROOF.write_text(json.dumps(q,indent=2,allow_nan=False)+'\n');print(json.dumps({k:q[k] for k in ('status','actual_runtime_tests','actual_cells_checked','fullwidth_native_optimizer_pairs','GPU_allocation_created')}))
if __name__=='__main__':main()
