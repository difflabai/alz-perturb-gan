"""Actual frozen20cell scope, accounting and native copied-model qualification."""
import datetime,json,os,re,subprocess,sys
from pathlib import Path
HERE=Path(__file__).resolve().parent;sys.path.insert(0,str(HERE))
import binding as b,preset_scope as s
def main():
 assert os.environ.get('CUDA_VISIBLE_DEVICES')=='' and not s.read(s.PLAN)['enabled']
 protocol=s.validate_protocol();assert not s.RUNTIME_PROOF.exists();results=[];count=0;native_rows=[]
 for name in ('test_integration.py','test_scope.py','test_factory.py'):
  cmd=[sys.executable,'-I','-S','-B','-c','import sys,runpy;sys.path.insert(0,sys.argv[1]);p=sys.argv[2];sys.argv=[p];runpy.run_path(p,run_name="__main__")',str(HERE),str(HERE/name)]
  p=subprocess.run(cmd,capture_output=True,text=True,timeout=300);print(p.stderr);results.append(dict(file=name,returncode=p.returncode,stdout=p.stdout,stderr=p.stderr));assert p.returncode==0,name
  count+=int(re.search(r'Ran (\d+) tests?',p.stderr).group(1))
  if name=='test_factory.py':native_rows=json.loads(next(line.removeprefix('NATIVE_CONFIRMATION_ROWS ') for line in p.stdout.splitlines() if line.startswith('NATIVE_CONFIRMATION_ROWS ')))
 assert count==23 and 'torch' not in sys.modules and len(native_rows)==3*len(protocol['selected_configurations'])
 files=[*HERE.glob('*.py'),HERE/'policy.prepared.json',*s.MODEL_DIR.glob('*.py'),s.PROTOCOL,s.MODEL_PROOF,s.HPO_PROOF,s.NATIVE_PROOF,s.ENGINE_PROOF,s.BASELINE_PROOF,s.SCALAR_PROOF,s.DESIGN,s.SELECTOR,s.WINDOW,
   s.ROOT/'analysis/run_confirmfaith097.py',s.ROOT/'analysis/prepare_confirmfaith097.py',s.ROOT/'analysis/faithful097/native_trainer_parity.py',*[Path(p) for p in protocol['selection_evidence_SHA256']]]
 q=dict(status='RELEASED097_MATCHED_CONFIRMATION_SCOPE_ACCOUNTING_CPU_PASS',UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),actual_runtime_tests=count,actual_native_trainer_parity_rows=native_rows,
   actual_cells_checked=3*len(protocol['selected_configurations']),fullwidth_native_optimizer_pairs=18*len(protocol['selected_configurations']),selection_seeds=[0,1],all_new_models_from_update_zero=True,native_model_source_unchanged=False,actual_released_trainer_parity=True,
   actual_native_trainer_prefix_parity_for_all_selected_seeds=True,GPU_allocation_created=False,CUDA_initialized=False,tests=results,source_SHA256={str(p):b.sha(p) for p in files})
 s.RUNTIME_PROOF.write_text(json.dumps(q,indent=2,allow_nan=False)+'\n');print(json.dumps({k:q[k] for k in ('status','actual_runtime_tests','actual_cells_checked','fullwidth_native_optimizer_pairs','GPU_allocation_created')}))
if __name__=='__main__':main()
