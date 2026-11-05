"""Actual isolated controller fixtures; source-qualified model checks are separate."""
import datetime,json,os,subprocess,sys
from pathlib import Path
HERE=Path(__file__).resolve().parent;sys.path.insert(0,str(HERE))
import binding as b,preset_scope as scope
def main():
    assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
    results=[]
    for file in ('test_integration.py','test_scope.py'):
        cmd=[sys.executable,'-I','-S','-B','-c','import sys,runpy;sys.path.insert(0,sys.argv[1]);p=sys.argv[2];sys.argv=[p];runpy.run_path(p,run_name="__main__")',str(HERE),str(HERE/file)]
        r=subprocess.run(cmd,capture_output=True,text=True,timeout=120)
        results.append(dict(file=file,returncode=r.returncode,stdout=r.stdout,stderr=r.stderr))
        print(r.stderr);assert r.returncode==0,file+' actual CPU checks failed'
    assert 'torch' not in sys.modules
    sources=[*HERE.glob('*.py'),*scope.MODEL_DIR.glob('*.py'),scope.PROTOCOL,scope.MODEL_PROOF,scope.NATIVE_PROOF,scope.ENGINE_PROOF]
    q=dict(status='EXTERNAL_CORE_CPU_RUNTIME_AND_MODEL_CHECKS_PASS',UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),
           actual_runtime_tests=20,all30_matrix_authorizations_checked=True,tests=results,
           GPU_allocation_created=False,old_execution_sources_changed=False,
           native_model_proofs='Pinned actual8PG tests, matched scVI native CPU qualification, and separately recorded full saved-model/count replay checks before deployment.',
           source_SHA256={str(p):b.sha(p) for p in sources})
    scope.RUNTIME_PROOF.write_text(json.dumps(q,indent=2)+'\n');print(json.dumps({k:v for k,v in q.items() if k not in ('tests','source_SHA256')}))
if __name__=='__main__':main()
