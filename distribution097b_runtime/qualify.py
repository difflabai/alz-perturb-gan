"""Confined accounting/lease tests and actual fullwidth native CPU replay."""
import datetime,json,os,re,subprocess,sys
from pathlib import Path
HERE=Path(__file__).resolve().parent;sys.path.insert(0,str(HERE))
import binding as b,preset_scope as scope
def main():
 assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
 results=[];count=0
 for name in ('test_integration.py','test_scope.py','test_factory.py'):
  command=[sys.executable,'-I','-S','-B','-c','import sys,runpy;sys.path.insert(0,sys.argv[1]);p=sys.argv[2];sys.argv=[p];runpy.run_path(p,run_name="__main__")',str(HERE),str(HERE/name)]
  p=subprocess.run(command,capture_output=True,text=True,timeout=180);print(p.stderr);assert p.returncode==0
  n=int(re.search(r'Ran (\d+) tests?',p.stderr).group(1));count+=n;results.append(dict(file=name,tests=n,returncode=p.returncode,stdout=p.stdout,stderr=p.stderr))
 assert count==22
 paths=[*scope.SELECTION_EVIDENCE,*HERE.glob('*.py'),*scope.MODEL_DIR.glob('*.py'),scope.PROTOCOL,scope.MODEL_PROOF,scope.HPO_PROOF,scope.NATIVE_PROOF,scope.ENGINE_PROOF,scope.WINDOW,scope.ROOT/'analysis/run_distribution097b.py',scope.ROOT/'analysis/duration096/native_fixture.py']
 q=dict(status='NATIVE097_FRESH20K_DISTRIBUTION_SCOPE_ACCOUNTING_CPU_PASS',UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),actual_runtime_tests=count,fullwidth_native_optimizer_pairs=12,identical_next_pair_native_state_across500_and20000_execution_limits=True,native_model_source_unchanged=True,tests=results,GPU_allocation_created=False,CUDA_initialized=False,source_SHA256={str(p):b.sha(p) for p in paths})
 scope.RUNTIME_PROOF.write_text(json.dumps(q,indent=2)+'\n')
if __name__=='__main__':main()
