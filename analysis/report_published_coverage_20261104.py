"""Reconcile historical CPU construction coverage with actual completed results."""
import datetime
import hashlib
import json
from pathlib import Path

ROOT = Path('/ml2/alz')


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    leaderboard = ROOT / 'analysis/pure_formulations_leaderboard_after_KA2_EMA.json'
    data = json.loads(leaderboard.read_text())
    by = {x.get('formulation', x.get('pure_formulation', x.get('method'))): x for x in data['rows']}
    names = {'bispf_optimized': 'latent8', 'bispf': 'original_bispf', 'bcap': 'bcap',
             'bcap_adam': 'bcap_adam', 'ka2': 'ka2', 'k3p': 'k3p'}
    rows = []
    for name, form in names.items():
        x = by[form]
        assert x['completed_updates'] == 20000
        evidence = x.get('decision_evidence')
        proof = ({'path': evidence, 'SHA256': sha(evidence)} if evidence else
                 {'leaderboard_row': form, 'leaderboard_SHA256': sha(leaderboard)})
        rows.append({'published_trainer_preset': name, 'completed_count_trial': True,
                     'formulation': form, 'completed_updates': 20000,
                     **{k: x[k] for k in ('energy_score', 'cell_energy_distance', 'spread_ratio')},
                     'evidence': proof})
    not_run = [
        {'name': 'e22', 'status': 'Native conditional row semantics refused with row controls enabled; no trained quality result.'},
        {'name': 'atlas', 'status': 'Native conditional row semantics refused with row controls enabled; no trained quality result.'},
        {'name': 'e22_routed', 'status': 'Requires explicit RoutedRows full-forward contract; no qualified implementation or trained quality result.'},
        {'name': 'halloween', 'status': 'CPU component construction only; not trained.'},
        {'name': 'AE/VAE and DDGAN model families', 'status': 'Not trained; would require additional faithful count adapters.'},
    ]
    release = ROOT / 'analysis/particlegan_release_reverification_20261104T1538.json'
    audit = ROOT / 'analysis/CAL_GPU0_V1_stage_prior_v1_f0_pg_s0-checkpoint-3000-portability.json'
    out = {
        'status': 'SOURCE_BOUND_PUBLISHED_PRESET_COVERAGE_WITH_CURRENT_COMPLETED_RESULTS',
        'utc': datetime.datetime.now(datetime.timezone.utc).isoformat(), 'native_release': '0.9.4',
        'release_recheck_path': str(release), 'release_recheck_SHA256': sha(release),
        'completed_pure_count_seed0_comparisons': len(data['rows']),
        'completed_published_trainer_presets': rows,
        'additional_completed_native_prior_family': 'mog',
        'additional_controls': 'Fixed finite particle bank and explicit KA2 saved generator+prior EMA readout completed. These add no published trainer names; EMA readout added zero training updates. Other count-adapter variants and fixed whole-bag ensembles are included in the21 comparisons, not21 independent published recipes.',
        'not_quality_tested': not_run,
        'currently_running': {
            'attempt': 'CAL_GPU0_V1_stage_prior_v1_f0_pg_s0',
            'kind': 'Task-specific stage-conditioned prior extension, not a new native preset',
            'checkpoint3000_audit_path': str(audit), 'checkpoint3000_audit_SHA256': sha(audit),
            'quality_complete': False,
        },
        'source_SHA256': {str(p): sha(p) for p in [
            Path(__file__).resolve(), leaderboard,
            ROOT / 'analysis/native_preset_requirements_inventory.json',
            ROOT / 'analysis/published_recipe_construction_inventory.json',
        ]},
        'incumbent_unchanged': True, 'three_seed_stable_joint_target_passed': False,
        'scope': 'Four repeatedly used adaptive calibration donors;20evaluationdonors excluded. Comparison counts are not independent biological replication. All-six refers to six completed trainer preset names, not all published ParticleGAN model families. Original ensemble fixed-seed confirmation failed2of3jointcellgates; no stable joint ATH.',
    }
    dest = ROOT / 'analysis/published_formulation_coverage_20261104T1538.json'
    assert not dest.exists()
    dest.write_text(json.dumps(out, indent=2) + '\n')
    lines = [
        'Actual ParticleGAN ALZ formulation coverage, ' + out['utc'], '',
        'Pinned and rechecked official release:0.9.4. All51installed Python sources remain exact official-wheel matches.',
        '21 completed seed0 pure-count comparisons cover six published trainer preset names. The21 include architecture/sampling/observation variants and generated ensembles; they are not21 independent published recipes or biological replications.', '',
        'Preset | Updates | Donor energy score | Pooled cell energy distance (both lower is better)',
        '--- | --- | --- | ---',
    ]
    for x in rows:
        lines.append(f"{x['published_trainer_preset']} | {x['completed_updates']} | {x['energy_score']:.9f} | {x['cell_energy_distance']:.9f}")
    lines += [
        '', 'Native MoG prior family, fixed finite-prior control and explicit native KA2 G+P EMA readout also completed; EMA added zero training updates. None replaced the joint incumbent.',
        'Selected pure whole-bag ensemble: ES1.935532040, cellED0.231844609;40k total parent updates. Its fixed three-seed confirmation failed joint cell criteria on seeds1and2. No stable joint ATH.',
        '', 'Untrained or unsupported coverage:',
        *[x['name'] + ': ' + x['status'] for x in not_run], '',
        'Current stage-conditioned prior trial is a task extension using native component APIs, with512particles split128perknownADNCstage. Native20k endpoint and two-replica quality evaluation remain pending. Actual3000update recovery checkpoint passed CPU portability/state/learned-particle/ownership audit; this is not a quality result.',
        '', '33 fitting donors; four adaptive calibration donors;20evaluation donors excluded from selection.',
        'Source-bound evidence: ' + str(dest),
        'Historical CPU-only construction inventories remain unchanged; this dated report supersedes their old coverage counts without changing their original findings.',
    ]
    text_dest = dest.with_suffix('.txt')
    assert not text_dest.exists()
    text_dest.write_text('\n'.join(lines) + '\n')
    print('\n'.join(lines[:13]))


if __name__ == '__main__':
    main()
