"""Isolated actual native binding/accounting fixtures for duration096."""
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
        results.append(dict(file=name,returncode=p.returncode,stdout=p.stdout,stderr=p.stderr));print(p.stderr)
        assert p.returncode==0
    paths=[*HERE.glob('*.py'),*scope.EXT_DIR.glob('*.py'),*scope.MODEL_DIR.glob('*.py'),scope.PROTOCOL,scope.MODEL_PROOF,scope.NATIVE_PROOF,scope.ENGINE_PROOF,scope.EXT_PROOF,scope.PARENT_PROOF]
    q=dict(status='DURATION096_CPU_RUNTIME_AND_NATIVE_EXTENSION_CHECKS_PASS',UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),actual_runtime_tests=18,tests=results,GPU_allocation_created=False,source_SHA256={str(p):b.sha(p) for p in paths})
    scope.RUNTIME_PROOF.write_text(json.dumps(q,indent=2)+'\n')
if __name__=='__main__':main()
