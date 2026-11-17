"""Fixed two-lane release pilot: serial GPU fitting and matched CPU scoring."""
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import common as C
import host


def call(lane,phase):
    command=[sys.executable,'-B',str(host.HERE/'worker.py'),'--phase',phase,'--lane',lane]
    env=dict(os.environ,OMP_NUM_THREADS='3',OPENBLAS_NUM_THREADS='3',MKL_NUM_THREADS='3')
    log=host.RUNS/f'logs/{phase}_{lane}.log';log.parent.mkdir(parents=True,exist_ok=True)
    with log.open('a') as stream:
        child=subprocess.Popen(command,stdout=stream,stderr=subprocess.STDOUT,env=env,cwd=host.ROOT)
        C.write(host.RUNS/'status.json',{'status':'running','lane':lane,'phase':phase,
            'runner_pid':os.getpid(),'worker_pid':child.pid,'command':command,
            'source_manifest_sha256':host.sha(C.PIN),'time':time.time(),'default_changed':False})
        try:
            code=child.wait(timeout=14400 if phase=='train' else 3600)
        except subprocess.TimeoutExpired:
            child.terminate()
            try:child.wait(timeout=30)
            except subprocess.TimeoutExpired:child.kill();child.wait()
            raise RuntimeError(f'{lane}/{phase} timed out; inspect checkpoint before retry')
    if code:
        raise RuntimeError(f'{lane}/{phase} exit{code}; inspect {log}')
    C.check_sources();C.check_disk()
    print(f'completed {lane}/{phase}',flush=True)


def report():
    assert all(C.valid_score(n) for n in ('compat','rhbi'))
    oldroot=host.ROOT/'runs/improve_20261109/results_r5/covariance_challenge_20261114'
    rows={n:json.loads((oldroot/n/'fold0/seed0.json').read_text())
          for n in ('current32','original32','empirical32')}
    rows.update({n:json.loads((host.RUNS/f'results/{n}.json').read_text()) for n in ('compat','rhbi')})
    summary={n:{'profile_RMSE':r['historical']['pb_rmse'],
        **{f'held_{k}_{m}':r['intervals']['heldout']['percentile'][k][m]
           for k in ('sd','combined') for m in ('interval_score','coverage','width')},
        'historical':r['historical']} for n,r in rows.items()}
    compat=rows['compat']; current=rows['current32']
    exact={'historical':compat['historical']==current['historical'],
           'intervals':compat['intervals']==current['intervals'],
           'generated_sha256':compat['fingerprint']['generated_sha256']==current['fingerprint']['generated_sha256']}
    value={'status':'completed_fixed_two_lane_release_pilot','fold':0,'seed':0,
        'summary':summary,'compatibility_full20k_exact':exact,
        'source_manifest_sha256':host.sha(C.PIN),'default_changed':False,
        'new_default_adopted':False,'sealed_or_external_donors_used':False,
        'rhbi_recipe':'rhbi_optimized_alz_conditional_no_simplex',
        'rhbi_exact_upstream_default':False,
        'scores_sha256':{str(host.RUNS/f'results/{n}.json'):host.sha(host.RUNS/f'results/{n}.json')
                         for n in ('compat','rhbi')},
        'scope':'one fixed development fold/seed; software/quality pilot, no population or default qualification'}
    C.write(host.RUNS/'pilot_comparison.json',value)
    return value


def main():
    host.RUNS.mkdir(parents=True,exist_ok=True)
    with (host.RUNS/'pilot.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        C.preflight();C.check_disk()
        root=host.RUNS/'verification'
        compatibility=json.loads((root/'compatibility.json').read_text())
        assert compatibility['status']=='passed_exact_initial_and_five_updates_on_shared_GPU1'
        for lane in ('compat','rhbi'):
            assert json.loads((root/f'fresh_process_{lane}.json').read_text())['full_checkpoint_and_metrics_exact']
            name='verification_0.10.0'+('_rhbi' if lane=='rhbi' else '')+'.json'
            proof=json.loads((root/name).read_text())
            assert proof['full_checkpoint_and_metrics_exact'] and proof['provenance']['host_derivation_sha256']==host.sha(host.HERE/'host.py')
            if lane=='rhbi':assert proof['rhbi_semantics']['conditional_transport_calls']==300
        try:
            for lane in ('compat','rhbi'):
                if not C.valid_fit(lane):call(lane,'train')
                assert C.valid_fit(lane)
                if not C.valid_score(lane):call(lane,'generate')
                assert C.valid_score(lane)
            comparison=report()
            C.write(host.RUNS/'status.json',{'status':'completed','time':time.time(),
                'comparison_sha256':host.sha(host.RUNS/'pilot_comparison.json'),'default_changed':False})
            print(json.dumps(comparison,indent=2),flush=True)
        except BaseException as exc:
            C.write(host.RUNS/'status.json',{'status':'execution_error','error':repr(exc),
                'time':time.time(),'default_changed':False})
            raise


if __name__=='__main__':main()
