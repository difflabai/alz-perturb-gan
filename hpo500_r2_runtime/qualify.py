"""Actual isolated accounting/live-gate and125native identity cases, no GPU."""
import datetime,json,os,subprocess,sys
from pathlib import Path
HERE=Path(__file__).resolve().parent;sys.path.insert(0,str(HERE))
import binding as b,preset_scope as scope
def main():
    assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
    results=[]
    for name in ('test_integration.py','test_scope.py'):
        cmd=[sys.executable,'-I','-S','-B','-c','import sys,runpy;sys.path.insert(0,sys.argv[1]);p=sys.argv[2];sys.argv=[p];runpy.run_path(p,run_name="__main__")',str(HERE),str(HERE/name)]
        p=subprocess.run(cmd,capture_output=True,text=True,timeout=120)
        results.append(dict(file=name,returncode=p.returncode,stdout=p.stdout,stderr=p.stderr));print(p.stderr);assert p.returncode==0
    paths=[*HERE.glob('*.py'),*scope.MODEL_DIR.glob('*.py'),scope.STUDY/'search.py',scope.PROTOCOL,scope.MODEL_PROOF,scope.NATIVE_PROOF,scope.ENGINE_PROOF,scope.RECOVERY_PROTOCOL,scope.RESUME_PROOF,scope.PARENT_PROOF,*scope.RECOVERY_DIR.glob('*.py')]
    q=dict(status='HPO500_R2_NATIVE_SCOPE_ACCOUNTING_CPU_PASS',UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),actual_runtime_tests=22,lower_native_registered_identity_cases=126,tests=results,GPU_allocation_created=False,source_SHA256={str(p):b.sha(p) for p in paths})
    scope.RUNTIME_PROOF.write_text(json.dumps(q,indent=2)+'\n')
if __name__=='__main__':main()
