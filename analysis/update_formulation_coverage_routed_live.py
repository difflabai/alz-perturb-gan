"""Reconcile completed quality evidence and one genuinely live routed trial."""
import datetime
import hashlib
import json
from pathlib import Path

ROOT=Path('/ml2/alz')
read=lambda p:json.loads(Path(p).read_bytes())
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    previous=ROOT/'analysis/published_formulation_coverage_20261104T1538.json'
    leaderboard=ROOT/'analysis/pure_formulations_leaderboard_after_Halloween.json'
    release=ROOT/'analysis/particlegan_release_reverification_20261104T1538.json'
    model=ROOT/'analysis/routed_e22_model_CPU_qualification.json'
    runtime=ROOT/'analysis/routed_e22_runtime_CPU_qualification.json'
    protocol=ROOT/'analysis/routed_e22_protocol.json'
    confirmation=ROOT/'analysis/pure_ensemble_confirmation_v6_all_seeds.json'
    attempt='CAL_GPU0_V1_routed_e22_v1_f0_pg_s0'
    folder=ROOT/'artifacts/allocations'/attempt
    authpath=folder/'authorization.json';auth=read(authpath);dep=read(folder/'deployment.json')
    ledger=Path(dep['ledger_path']);job=read(ledger)['jobs'][0]
    assert sha(authpath)==dep['authorization_SHA256'] and job['status']=='active'
    worker=job['child_identity'];stat=Path(f"/proc/{worker['pid']}/stat").read_text()
    assert int(stat.rsplit(')',1)[1].split()[19])==worker['birth']
    assert Path('/proc/sys/kernel/random/boot_id').read_text().strip()==worker['boot']
    progresspath=ROOT/'runs'/attempt/'fit/progress.json';progress=read(progresspath)
    assert progress['step']>=500 and not (ROOT/'runs'/attempt/'completed.json').exists()
    old,d=read(previous),read(leaderboard)
    assert d['completed_seed0_comparisons']==len(d['rows'])==23
    rows={r.get('formulation',r.get('pure_formulation',r.get('method'))):r for r in d['rows']}
    completed=old['completed_published_trainer_presets']
    for row in completed:
        current=rows[row['formulation']]
        for key in ('energy_score','cell_energy_distance','spread_ratio'):assert current[key]==row[key]
        row['evidence']={'leaderboard_path':str(leaderboard),'leaderboard_SHA256':sha(leaderboard),'leaderboard_row':row['formulation']}
    resultpath=ROOT/'analysis/CAL_GPU0_V1_halloween_native_v1_f0_pg_s0-result.json'
    result=read(resultpath);candidate=result['candidate']
    assert not result['decision']['accept'] and result['owned_release_and_clean_settlement_verified']
    assert candidate['completed_updates']==candidate['fixed_endpoint']==20000 and candidate['generation_replicates']==2
    completed.append({'published_trainer_preset':'halloween','formulation':'halloween','completed_count_trial':True,
                      'completed_updates':20000,**{k:candidate[k] for k in ('energy_score','cell_energy_distance','spread_ratio')},
                      'evidence':{'path':str(resultpath),'SHA256':sha(resultpath)}})
    assert set(r['published_trainer_preset'] for r in completed)==set(d['completed_native_preset_names'])
    assert len(completed)==7 and not read(confirmation)['stable_joint_target_passed']
    assert sha(ROOT/'pure_incumbent.json')==d['selected_pure_incumbent_SHA256']
    pins=read(model)['official_wheel_source_SHA256']
    assert len(pins)==51
    for p,pin in pins.items():assert sha(p)==pin
    assert read(model)['tests_passed']==19 and read(runtime)['tests_passed']==27
    paths=[Path(__file__).resolve(),previous,leaderboard,release,model,runtime,protocol,confirmation,
           resultpath,ROOT/'analysis/CAL_GPU0_V1_stage_prior_v1_f0_pg_s0-result.json',
           ROOT/'pure_incumbent.json',authpath,folder/'execution_source_archive.json']
    utc=datetime.datetime.now(datetime.timezone.utc).isoformat()
    output=ROOT/'analysis/published_formulation_coverage_routed_live_20261104'
    report={'status':'SOURCE_BOUND_COMPLETED_PRESETS_AND_VERIFIED_LIVE_ROUTED_E22',
            'utc':utc,'native_release':'0.9.4','release_last_online_check_utc':read(release)['checked_utc'],
            'release_recheck_path':str(release),'release_recheck_SHA256':sha(release),
            'official_wheel_python_source_files_reverified_now':51,'package_changed':False,
            'completed_pure_count_seed0_comparisons':23,'completed_published_trainer_presets':completed,
            'additional_completed_native_prior_family':'mog',
            'additional_controls_and_task_variants':'Fixed bank, native KA2 saved G+P EMA, architecture/observation/rate variants, generated whole-bag ensembles, cell-energy auxiliary and stage-conditioned prior are included among23comparisons; they do not add published native trainer preset names.',
            'not_quality_tested':[{'name':'e22_independent_rows','status':'Original native conditional row-semantics refusal remains; no independently trained formulation result. Routed E22 has a distinct conditional evidence/restructuring law.'},
                                  {'name':'atlas','status':'Original conditional-contract refusal; no trained count-quality result.'},
                                  {'name':'AE/VAE and DDGAN model families','status':'Not trained; faithful count task adapters remain absent.'}],
            'currently_running':{'attempt':attempt,'published_trainer_preset':'e22_routed','job_status':'active','worker_identity':worker,
                                 'latest_recorded_progress':progress,'fixed_endpoint':20000,'phase_cap_seconds':21600,
                                 'model_CPU_checks_passed':19,'runtime_CPU_checks_passed':27,
                                 'native_API':read(protocol)['native_API'],'quality_complete':False,
                                 'live_observation_note':'Exact boot/starttime worker identity checked through /proc and active ledger. Progress is a timestamped observation, not a live document. Full quality/count/checkpoint/served-state/release/charge audit remains pending.'},
            'last_negative_trials':{'stage_prior':{'energy_score':rows['stage_conditioned_particle_prior_task_extension']['energy_score'],
                                                'cell_energy_distance':rows['stage_conditioned_particle_prior_task_extension']['cell_energy_distance'],
                                                'negative_commit':'c9931c9','actual_activation_revert':'77f97a6'},
                                    'halloween':{'energy_score':candidate['energy_score'],'cell_energy_distance':candidate['cell_energy_distance'],
                                                 'negative_commit':'8a0a999','actual_activation_revert':'98fb947'}},
            'incumbent_unchanged':True,'three_seed_stable_joint_target_passed':False,
            'new_training_or_generation_by_this_report':False,'CUDA_context_created_by_this_report':False,
            'source_SHA256':{str(p):sha(p) for p in paths},'installed_official_wheel_source_SHA256':pins,
            'scope':'Four adaptive calibration donors,33fittingdonors,20evaluationdonors excluded.23comparisons are not23publishedrecipes or biologicalreplications. Seven completed native trainer presets plus one currently training is not exhaustive ParticleGAN model-family coverage. All3seed joint confirmation remains failed. No stablejointATH, externalSOTA or generalization claim. Historical coverage notes retained.'}
    assert not output.with_suffix('.json').exists() and not output.with_suffix('.txt').exists()
    output.with_suffix('.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    lines=[f'ParticleGAN ALZ formulation coverage, {utc}','',
           'Official ParticleGAN0.9.4, last online release check15:37UTC2026-11-04. All51installed Python files rehashed now and unchanged from the official wheel.',
           '23completed seed0 pure-count comparisons cover seven published trainer presets. These counts include task variants and generated ensembles; they are not23independent published recipes or biological replications.','',
           'Preset | Updates | Donor energy score | Pooled cell energy distance (both lower is better)',
           '--- | --- | --- | ---']
    for r in completed:lines.append(f"{r['published_trainer_preset']} | {r['completed_updates']} | {r['energy_score']:.9f} | {r['cell_energy_distance']:.9f}")
    lines+=['','Routed E22 is now implemented, CPU-qualified and training through the public e22_routed API with its native controls enabled.',
            f"Actual live observation: {progress['step']}updates; {progress['routing_counters']['probes']}native probes, {progress['routing_counters']['proposals']}proposals, {progress['routing_counters']['guard_rejections']}guard rejections, {progress['routing_counters']['moves']}accepted moves. Endpoint20k/phasecap6h; no quality score yet.",
            'It uses complete Gamma-Poisson count/feature callbacks, per-context64bit seeds, 512x8dense learned bank, fit/protected cell pools11825/2937within33donors and one coupled native served snapshot. NativehorizonNone/probeinterval20 are declared before training.',
            'Common preprocessing predates this protection split and includes the original fitting-cell inputs; the protected pool is not independent biological validation.','',
            'Stage-prior and Halloween are completed negatives, recorded then activation-reverted. Halloween had a severe count-distribution/spread collapse.',
            'Native MoG prior family, fixed-bank control and KA2 saved G+P EMA readout also completed; they add no published trainer preset names.','',
            'Remaining untrained coverage: independent-row E22 and Atlas retain their native conditional-contract refusals; AE/VAE/DDGAN need faithful count adapters. Routed E22 is a distinct conditional formulation, not evidence those independent-row mechanisms were trained.','',
            'Selected pure whole-bag ensemble remains ES1.935532040/cellED0.231844609 with40k total parent updates. Its fixed three-seed confirmation failed the joint cell criteria on seeds1and2. No stable joint ATH.',
            'Four repeatedly used adaptive calibration donors;20evaluationdonors remain excluded.',
            f'Source-bound evidence: {output.with_suffix(".json")}']
    output.with_suffix('.txt').write_text('\n'.join(lines)+'\n')
    print(json.dumps({'coverage_path':str(output.with_suffix('.txt')),'completed_comparisons':23,'completed_native_presets':7,'live_routed_updates':progress['step'],'source_files_unchanged':51},indent=2))


if __name__=='__main__':main()
