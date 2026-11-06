"""Qualify the actual selected configurations; refuse the prepared empty plan."""
import datetime,json,os,re,subprocess,sys
from pathlib import Path
HERE=Path(__file__).resolve().parent;sys.path.insert(0,str(HERE))
import binding as b,preset_scope as s
def main():
 assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
 p=s.validate_protocol();assert 1<=len(p['selected_configurations'])<=3
 tests=[];count=0
 for name in ('test_integration.py','test_scope.py','test_factory.py'):
  cmd=[sys.executable,'-I','-S','-B','-c','import sys,runpy;sys.path.insert(0,sys.argv[1]);p=sys.argv[2];sys.argv=[p];runpy.run_path(p,run_name="__main__")',str(HERE),str(HERE/name)]
  r=subprocess.run(cmd,capture_output=True,text=True,timeout=300);print(r.stderr);assert r.returncode==0,name
  n=int(re.search(r'Ran (\d+) tests?',r.stderr).group(1));count+=n;tests.append(dict(file=name,tests=n,returncode=r.returncode,stdout=r.stdout,stderr=r.stderr))
 assert count==22 and 'torch' not in sys.modules
 files=[*HERE.glob('*.py'),*s.MODEL_DIR.glob('*.py'),s.PROTOCOL,s.MODEL_PROOF,s.HPO_PROOF,s.NATIVE_PROOF,s.ENGINE_PROOF,s.BASELINE_PROOF,s.RULE,s.SELECTOR,s.WINDOW,s.ROOT/'analysis/run_confirmation097.py',s.ROOT/'analysis/duration096/native_fixture.py',*[Path(x) for x in p['selection_evidence_SHA256']]]
 q=dict(status='NATIVE097_FRESH20K_CONFIRMATION_SCOPE_ACCOUNTING_CPU_PASS',UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        actual_runtime_tests=count,all_frozen_configuration_seed_authorizations_checked=True,confirmation_training_seeds=[1,2,3,4],
        fullwidth_native_optimizer_pairs=16*len(p['selected_configurations']),identical_next_pair_native_state_across500_and20000_execution_limits=True,
        native_model_source_unchanged=True,tests=tests,GPU_allocation_created=False,CUDA_initialized=False,
        source_SHA256={str(f):b.sha(f) for f in files})
 assert not s.RUNTIME_PROOF.exists();s.RUNTIME_PROOF.write_text(json.dumps(q,indent=2)+'\n')
 print(json.dumps({k:q[k] for k in ('status','actual_runtime_tests','fullwidth_native_optimizer_pairs','GPU_allocation_created')}))
if __name__=='__main__':main()
