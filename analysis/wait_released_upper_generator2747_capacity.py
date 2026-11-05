"""Wait for unchanged capacity, then run CPU checks; never start a GPU job."""
import datetime
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

ROOT=Path('/ml2/alz')
HERE=ROOT/'released_rate_g100d050p050_resume2_runtime'
sys.path.insert(0,str(HERE))
import binding as b
import prepare
import preset_scope as scope
import resource_profile as profile

FOLDER=ROOT/'artifacts/diagnostics/released_upper_generator2747_capacity'
FOLDER.mkdir(parents=True,exist_ok=True)


def dump(name,value):
    path=FOLDER/name
    temp=path.with_suffix('.partial.json')
    temp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
    temp.replace(path)


def qualified():
    if not scope.RUNTIME_PROOF.exists():return False
    proof=b.read(scope.RUNTIME_PROOF)
    assert proof['status']=='PASSED_EXACT_RELEASED_PARAMETER_RESUME_RUNTIME_CPU_QUALIFICATION'
    assert proof['tests_passed']==28
    for path,pin in proof['source_SHA256'].items():assert b.sha(path)==pin,path
    return True


def main():
    assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
    assert b.read(scope.PLAN)['enabled'] is False
    assert not (ROOT/'artifacts/allocations'/b.attempt_key(scope.CELL)).exists()
    stable=None
    qualification=qualified()
    while time.time()<scope.CONTINUATION['original_absolute_deadline_unix']-180:
        auth,_=prepare.build_authorization(check_only=True)
        controller,ad,_=b.modules(auth)
        obs=ad.snapshot()
        pressure=Path('/proc/pressure/memory').read_text()
        full=float(re.search(r'^full avg10=([0-9.]+)',pressure,re.M)[1])
        record={'status':'WAITING_UNCHANGED_SHARED_CAPACITY_AND_PSI',
                'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
                'observer_PID':os.getpid(),'host_available_bytes':obs['available_ram'],
                'full_memory_PSI_avg10':full,'CPU_qualification_passed':qualification,
                'original_absolute_deadline_unix':scope.CONTINUATION['original_absolute_deadline_unix'],
                'checkpoint_updates':2747,'GPU_job_or_allocation_started':False,
                'foreign_controls_or_guard_changes':False}
        try:
            profile.previous_trial_released()
            assert not controller.Controller(b.PROJECT)._read()['open_reservations'],'Another canonical phase is active'
            record['forecast']=profile.capacity(obs,ad)
            profile.free_floor()
            assert full<10,'Existing full memory PSI threshold is not clear'
            stable=time.monotonic() if stable is None else stable
            record['stable_capacity_seconds']=time.monotonic()-stable
        except (AssertionError,b.Refusal) as error:
            stable=None
            record['gate_refusal']=str(error)
            dump('progress.json',record)
            if not any(s in str(error) for s in ('growth forecasts do not fit','host available below',
                        'GPU free below','memory PSI','canonical phase is active')):
                raise
            time.sleep(10)
            continue
        dump('progress.json',record)
        if record['stable_capacity_seconds']>=30:
            if not qualification:
                marker=FOLDER/'CPU_qualification_attempt.json'
                assert not marker.exists(),'Inspect the previous CPU qualification attempt instead of automatic retries'
                marker.write_text(json.dumps({'utc':record['utc'],'status':'CPU_QUALIFICATION_STARTING_AFTER_30S_CAPACITY'},indent=2)+'\n')
                env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',
                         OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1')
                with (FOLDER/'qualification.stdout').open('xb') as out,(FOLDER/'qualification.stderr').open('xb') as err:
                    result=subprocess.run(['/usr/bin/python3','-I','-S','-B',str(HERE/'qualify.py')],
                        cwd=ROOT,env=env,stdin=subprocess.DEVNULL,stdout=out,stderr=err,timeout=120)
                assert result.returncode==0,'Actual continuation CPU checks failed; inspect captured output'
                qualification=qualified()
                assert qualification
                stable=None
            else:
                record.update(status='CPU_QUALIFIED_AND_UNCHANGED_SHARED_CAPACITY_READY_NO_ADMISSION',
                              runtime_CPU_tests_passed=28,runtime_proof_SHA256=b.sha(scope.RUNTIME_PROOF))
                dump('ready.json',record)
                print(json.dumps({k:v for k,v in record.items() if k!='forecast'},indent=2))
                return
        assert 'torch' not in sys.modules
        time.sleep(10)
    raise TimeoutError('Original absolute deadline too close; preserve checkpoint, no new GPU admission')


if __name__=='__main__':
    try:main()
    except BaseException as error:
        dump('observer_failure.json',{'status':'READINESS_OR_CPU_CHECK_FAILED_NO_GPU_ADMISSION',
             'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'error':repr(error),
             'original_deadline_preserved':True,'GPU_job_or_allocation_started':False})
        raise
