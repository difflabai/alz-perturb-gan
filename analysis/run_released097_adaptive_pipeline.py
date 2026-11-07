"""Resume the existing exact12-fit driver; optional paired follow-up then full45.

The superseded CPU pipeline/observer must be explicitly reconciled first.
This process never replaces, restarts or signals a GPU training process.
"""
import argparse,ast,datetime,fcntl,hashlib,json,os,subprocess,sys,time
from pathlib import Path
ROOT=Path('/ml2/alz');sys.path.insert(0,str(ROOT/'analysis'))
import run_released097_research_pipeline as engine
STATE=ROOT/'artifacts/released097-adaptive-research-pipeline'
engine.STATE=STATE
engine.STAGES['followup']=dict(runtime='followfaith097_runtime',study='analysis/followfaith097',
    driver='analysis/run_followfaith097.py',state='artifacts/followfaith097-paired-search',
    review_scope='RELEASED097_FRESH20K_FOLLOWUP_COORDINATE_SEARCH')

def immutable_sources():
    pins=engine.immutable_sources()
    paths=[Path(__file__).resolve(),ROOT/'analysis/after_released097_adaptive_calibration53.py',
        ROOT/'analysis/released097_followup_design.py',
        ROOT/'analysis/released097_followup_design_CPU_qualification_20261107.json',
        ROOT/'analysis/prepare_followfaith097.py',ROOT/'analysis/run_followfaith097.py',
        ROOT/'analysis/audit_followfaith097.py',
        ROOT/'analysis/followfaith097/possible_settings_native_CPU_qualification.json',
        ROOT/'analysis/followfaith097/prepared_accounting_CPU_qualification.json',
        *list((ROOT/'analysis/followfaith097').glob('*.py'))]
    pins.update({str(p):engine.sha(p) for p in paths})
    return pins

def validate_preparation():
    old=engine.read(ROOT/'artifacts/released097-research-pipeline/identity.json')
    engine.require_sources(old['source_SHA256'])
    parent=engine.read(ROOT/'artifacts/localfaith097-paired-search/identity.json')
    assert engine.alive(parent),'Retain original driver; this continuation must start while it is live'
    assert parent['activation']=='c454a89b08bc3b396146f760546299a78afd957f'
    for runtime in ('followfaith097_runtime','confirmfaith097_runtime','benchmark097_runtime'):
        assert not engine.read(ROOT/runtime/'plan.json')['enabled']
    q=engine.read(ROOT/'analysis/followfaith097/possible_settings_native_CPU_qualification.json')
    assert q['status']=='ALL4_POSSIBLE_FOLLOWUP_SETTING_SEED_NATIVE_TRAINER_PREFIX_CASES_PASS'
    assert q['actual_cases']==4 and not q['CUDA_initialized'] and not q['real_ALZ_data_read']
    engine.require_sources(q['source_SHA256'])
    pins=immutable_sources()
    for p in pins:
        if p.endswith('.py'):ast.parse(Path(p).read_text())
    return old,parent,pins

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--check-only',action='store_true');args=parser.parse_args()
    old,parent,pins=validate_preparation()
    if args.check_only:
        print(json.dumps(dict(status='ADAPTIVE_PIPELINE_PREPARATION_SOURCE_CHECK_PASS',
            original_local_driver=parent['pid'],pinned_files=len(pins),
            handoff_performed=False,followup_frozen=False,GPU_allocation_created=False)));return
    handoff=engine.read(ROOT/'analysis/released097_adaptive_CPU_handoff_20261107.json')
    assert handoff['status']=='ORIGINAL_CPU_PIPELINE_SUPERSEDED_GPU_HANDLES_PRESERVED'
    assert handoff['original_pipeline']=={k:old[k] for k in ('pid','birth','boot')}
    assert not engine.alive(old) and not engine.alive(handoff['original_sensitivity_observer'])
    assert handoff['GPU_processes_signalled'] is False and handoff['native_session_replaced'] is False
    STATE.mkdir(exist_ok=True)
    with (STATE/'lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        assert not (STATE/'identity.json').exists(),'One-use continuation already consumed'
        expiry=engine.read(engine.WINDOW)['expires_unix']+90
        engine.write(STATE/'identity.json',dict(pid=os.getpid(),birth=int(Path('/proc',str(os.getpid()),'stat').read_text().rsplit(')',1)[1].split()[19]),
            boot=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),source_SHA256=pins,
            parent={k:parent[k] for k in ('pid','birth','boot')},original_pipeline=handoff['original_pipeline'],
            finite_expiry_unix=expiry,foreign_signals_permitted=False,checkpoint_warmstarts_permitted=False,
            full45_required=True,full_goal_complete=False))
        try:
            engine.write(STATE/'progress.json',dict(status='WAITING_ORIGINAL_EXACT12_DRIVER',driver=parent,full_goal_complete=False))
            engine.wait_exact(parent,expiry);engine.require_sources(pins)
            local=ROOT/'artifacts/localfaith097-paired-search'
            assert not (local/'failure.json').exists() and engine.read(local/'progress.json')['activation_reverted']
            assert not engine.read(ROOT/'localfaith097_runtime/plan.json')['enabled']
            selection=engine.audit_phase('local','audit_localfaith097.py','INDEPENDENT_RELEASED097_PAIRED_SELECTION_AUDIT_PASS')
            response=engine.command('freeze-optional-followup',[engine.PG,'-I','-S','-B',ROOT/'analysis/prepare_followfaith097.py','--selection-audit',selection])
            result=json.loads(response)
            if result['status']=='FOLLOWUP_SKIPPED':
                engine.commit('followup-skip',[ROOT/'analysis/followfaith097/frozen_design.json'],
                    'Retain complete-scope forecast and skip optional follow-up when confirmation has priority')
                engine.write(STATE/'followup_skipped.json',result)
            else:
                assert result['required_new_fits'] in (2,4)
                engine.run_stage('followup',pins,expiry)
                selection=engine.audit_phase('followup','audit_followfaith097.py','INDEPENDENT_RELEASED097_PAIRED_SELECTION_AUDIT_PASS')
            engine.command('freeze-fresh-confirmation',[engine.PG,'-I','-S','-B',ROOT/'analysis/prepare_confirmfaith097.py','--selection-audit',selection])
            engine.run_stage('confirmation',pins,expiry)
            confirmation=engine.audit_phase('confirmation','audit_confirmfaith097.py','INDEPENDENT_RELEASED097_MATCHED_CONFIRMATION_AUDIT_PASS')
            engine.command('freeze-external-PG-before-evaluation',[engine.PG,'-I','-S','-B',ROOT/'analysis/prepare_benchmark097.py',
                '--selection-audit',selection,'--confirmation-audit',confirmation,
                '--confirmation-protocol',ROOT/'analysis/confirmfaith097/protocol.json',
                '--confirmation-report',ROOT/'analysis/confirmfaith097/report.json'])
            engine.run_stage('benchmark',pins,expiry)
            engine.command('full45-paired-distribution-comparison',[engine.PG,'-I','-S','-B',ROOT/'analysis/aggregate_benchmark097.py'],timeout=1800)
            comparison=ROOT/'analysis/benchmark097/comparison.json';assert engine.read(comparison)['complete_cells']==45
            gp=ROOT/'analysis/current_research_goal.json';g=engine.read(gp)
            g.update(study_status='FULL45_COMPARISON_READY_FINAL_REQUIREMENT_AUDIT_PENDING',
                external_comparison_path=str(comparison),full_goal_complete=False,native_goal_status='active')
            engine.write(gp,g);engine.commit('comparison',[comparison,gp],
                'Retain full45 fresh-model comparison after bounded adaptive calibration; final scope review remains')
            completed=dict(status='FULL_RESEARCH_PIPELINE_COMPARISON_READY_FOR_FINAL_AGENT_AUDIT',
                UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),selection_audit_path=str(selection),
                confirmation_audit_path=str(confirmation),comparison_path=str(comparison),
                full_goal_complete=False,SOTA_claim=False,
                required_final_review='Verify original objective, every fresh endpoint, real artifacts/statistics, calibration/evaluation separation limits, actual costs, prespecified53 sensitivity and bounded conclusions.')
            engine.write(STATE/'completed.json',completed)
            engine.report('completed','released097-full45-pipeline-comparison-ready-20261107',
                'Complete fresh45-model comparison ready for final objective audit; no automatic superiority or goal-complete claim.',completed)
        except BaseException as exc:
            failure=dict(status='ADAPTIVE_RESEARCH_PIPELINE_STOPPED_RETAINING_ORIGINAL_HANDLES',
                UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),error=repr(exc),
                automatic_retry=False,original_GPU_jobs_not_signaled=True,full_goal_complete=False,SOTA_claim=False)
            engine.write(STATE/'failure.json',failure)
            engine.report('failed','released097-adaptive-research-pipeline-stop-20261107',
                'Adaptive research handoff stopped on an actual source/qualification/endpoint/audit issue; original evidence retained.',failure)
            raise
if __name__=='__main__':main()
