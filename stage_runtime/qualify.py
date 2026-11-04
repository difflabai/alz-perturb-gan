"""Run actual isolated CPU source/lease/accounting fixtures and record exact code."""
import datetime,json,os,subprocess,sys
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=HERE.parent;sys.path.insert(0,str(HERE))
import binding as b
import preset_scope as scope
assert os.environ.get('CUDA_VISIBLE_DEVICES')==''

def main():
    results=[]
    for name in ('test_integration.py','test_scope.py'):
        command=[sys.executable,'-I','-S','-B','-c',"import sys,runpy;sys.path.insert(0,sys.argv[1]);target=sys.argv[2];sys.argv=[target];runpy.run_path(target,run_name='__main__')",str(HERE),str(HERE/name)]
        result=subprocess.run(command,capture_output=True,text=True,timeout=60)
        print(result.stdout);print(result.stderr)
        assert result.returncode==0,'Actual isolated CPU checks failed: '+name
        results.append({'test_file':str(HERE/name),'returncode':result.returncode,'stdout':result.stdout,'stderr':result.stderr})
    assert 'torch' not in sys.modules
    sources=[*HERE.glob('*.py'),ROOT/'stage_model/stage_pg_adapter.py',ROOT/'stage_model/stage_pg_update.py',ROOT/'stage_model/stage_prior.py',scope.PROTOCOL,scope.MODEL_PROOF,scope.ENGINE_PROOF,ROOT/'analysis/shared_gpu0_candidate/shared_gpu0_admission.py',ROOT/'analysis/shared_gpu0_candidate/shared_gpu0_runner.py']
    result={'status':'PASSED_ISOLATED_STAGE_PRIOR_RUNTIME_CPU_QUALIFICATION','utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
            'native_profile_wire_and_refusals_verified':True,'old_execution_sources_unchanged':True,'GPU_allocation_created':False,
            'test_processes':results,'source_SHA256':{str(p):b.sha(p) for p in sources},
            'scope':'Actual confined CPU owner/worker/controller/lease/cleanup/accounting and private native profile/source-wire fixtures. No canonical policy/allocation/lease/model created. No GPUdeployment is performed bythesechecks; activeplan is sourcebound separately. Model native factories/gradient/replay qualification is separately pinned. Owner-authorized exact source/resource review and fresh native capacity/admission still required.'}
    scope.RUNTIME_PROOF.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k not in ('test_processes','source_SHA256')},indent=2))

if __name__=='__main__':main()
