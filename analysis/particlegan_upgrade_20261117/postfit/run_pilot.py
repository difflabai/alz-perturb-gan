"""Run the unchanged pilot with explicit registration in generation children."""
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import readout_fix as FIX
C=FIX.C;host=FIX.host
spec=importlib.util.spec_from_file_location('frozen_release_pilot',host.HERE/'run_pilot.py')
Q=importlib.util.module_from_spec(spec);spec.loader.exec_module(Q)
original_call=Q.call


def call(lane,phase):
    FIX.check()
    if phase=='train':return original_call(lane,phase)
    command=[sys.executable,'-B',str(FIX.HERE/'readout_fix.py'),'--lane',lane]
    with (host.RUNS/f'logs/generate_{lane}_registered.log').open('a') as log:
        child=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,cwd=host.ROOT)
        C.write(host.RUNS/'status.json',{'status':'running','lane':lane,'phase':'generate',
            'runner_pid':os.getpid(),'worker_pid':child.pid,'command':command,
            'source_manifest_sha256':host.sha(C.PIN),
            'additive_execution_manifest_sha256':host.sha(FIX.HERE/'source_manifest.json'),
            'time':time.time(),'default_changed':False})
        try:code=child.wait(timeout=3600)
        except subprocess.TimeoutExpired:
            child.terminate()
            try:child.wait(timeout=30)
            except subprocess.TimeoutExpired:child.kill();child.wait()
            raise RuntimeError('registered generation timed out')
    assert code==0, f'fixed {lane}/generation exited{code}'
    FIX.check();FIX.supplement(lane)
    print('completed registered generation',lane,flush=True)


if __name__=='__main__':
    FIX.check();Q.call=call
    Q.main()
    for lane in ('compat','rhbi'):FIX.supplement(lane)
    C.write(host.RUNS/'readout_registration_completion.json',{
        'status':'completed_fixed_pilot_and_registered_readout',
        'additive_execution_manifest_sha256':host.sha(FIX.HERE/'source_manifest.json'),
        'pilot_comparison_sha256':host.sha(host.RUNS/'pilot_comparison.json'),
        'default_changed':False})
