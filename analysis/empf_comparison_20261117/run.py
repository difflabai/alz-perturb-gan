"""Fixed EMPF pilots, serial GPU fits and registered matched CPU scoring."""
import argparse
import fcntl
import importlib
import json
import os
import subprocess
import sys
import time
import host
import common as C


def worker(lane,phase):
    W=host.load('frozen_release_worker',host.BASE/'worker.py')
    if phase=='train':
        W.train(lane)
        T=sys.modules['trainer'];fit=C.fit_path(lane)
        q=T.torch.load(fit/'final.pt',map_location='cpu',weights_only=False)
        record=q['policy']['optimizers'][1]['regularizer']['record'];packet=record['empf']
        assert record['calls']==record['observed_steps']==q['completed_updates']==20000
        assert q['recipe']['reg_every']==1 and q['recipe']['reg_coeff']==1.
        assert packet['schema']==2 and not packet['proto_initialized']
        assert packet['proto_sum'] is packet['proto_count'] is None
        C.write(fit/'no_skip_audit.json',dict(status='passed_actual_full_EMPF_no_skips',lane=lane,
            source_manifest_sha256=host.sha(C.PIN),final_checkpoint_sha256=host.sha(fit/'final.pt'),
            penalty_calls=record['calls'],observed_steps=record['observed_steps'],
            completed_updates=q['completed_updates'],lazy_k=1,native_highdim_prototype_bank_inactive=True))
        assert C.valid_fit(lane)
        return
    original=host.activate
    def registered(selected):
        result=original(selected)
        mode=importlib.import_module('mode_cellshape')
        assert str(mode.__file__)==str(host.OLD/'mode_cellshape.py')
        result[0].arms.parse_mode(C.MODE)
        return result
    host.activate=registered
    try:return W.generate(lane)
    finally:host.activate=original


def report():
    assert all(C.valid_score(n) for n in ('native','matched'))
    prior=json.loads((host.BASE/'completion/pilot_comparison.json').read_text())
    rows={n:json.loads((host.RUNS/f'results/{n}.json').read_text()) for n in ('native','matched')}
    summary=dict(prior['summary'])
    for n,r in rows.items():
        summary['empf_'+n]={'profile_RMSE':r['historical']['pb_rmse'],
            **{f'held_{k}_{m}':r['intervals']['heldout']['percentile'][k][m]
               for k in ('sd','combined') for m in ('interval_score','coverage','width')},'historical':r['historical']}
    value=dict(status='completed_fixed_EMPF_release_and_matched_comparison',fold=0,seed=0,
        summary=summary,source_manifest_sha256=host.sha(C.PIN),default_changed=False,
        lazy_k=1,new_default_adopted=False,sealed_or_external_donors_used=False,
        scores_sha256={str(host.RUNS/f'results/{n}.json'):host.sha(host.RUNS/f'results/{n}.json') for n in rows},
        scope='one fixed development fold/seed; two prospectively fixed EMPF recipes; no default/population qualification')
    C.write(host.RUNS/'comparison.json',value)
    return value


def main():
    host.RUNS.mkdir(parents=True,exist_ok=True)
    with (host.RUNS/'pilot.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);C.preflight();C.check_disk()
        try:
            for lane in ('native','matched'):
                for phase,valid in (('train',C.valid_fit),('generate',C.valid_score)):
                    if not valid(lane):
                        command=[sys.executable,'-B',__file__,'--lane',lane,'--phase',phase]
                        log=host.RUNS/f'logs/{phase}_{lane}.log';log.parent.mkdir(parents=True,exist_ok=True)
                        with log.open('a') as stream:
                            child=subprocess.Popen(command,stdout=stream,stderr=subprocess.STDOUT,cwd=host.ROOT)
                            C.write(host.RUNS/'status.json',dict(status='running',lane=lane,phase=phase,runner_pid=os.getpid(),worker_pid=child.pid,command=command,time=time.time(),source_manifest_sha256=host.sha(C.PIN),default_changed=False))
                            try:code=child.wait(timeout=14400 if phase=='train' else 3600)
                            except subprocess.TimeoutExpired:
                                child.terminate()
                                try:child.wait(timeout=30)
                                except subprocess.TimeoutExpired:child.kill();child.wait()
                                raise RuntimeError(f'EMPF {lane}/{phase} timed out')
                        assert code==0,f'EMPF {lane}/{phase} exited{code}; inspect {log}'
                    assert valid(lane);C.check_sources();C.check_disk()
                    print(f'completed {lane}/{phase}',flush=True)
            value=report();C.write(host.RUNS/'status.json',dict(status='completed',comparison_sha256=host.sha(host.RUNS/'comparison.json'),default_changed=False,time=time.time()))
            print(json.dumps(value,indent=2),flush=True)
        except BaseException as error:
            C.write(host.RUNS/'status.json',dict(status='execution_error',error=repr(error),default_changed=False,time=time.time()))
            raise


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--lane',choices=('native','matched'));p.add_argument('--phase',choices=('train','generate'))
    a=p.parse_args()
    if bool(a.lane)!=bool(a.phase):p.error('lane and phase are supplied together')
    worker(a.lane,a.phase) if a.phase else main()
