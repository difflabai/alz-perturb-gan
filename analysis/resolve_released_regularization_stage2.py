"""Resolve the declared twelve regularization settings only after all rate audits."""
import argparse
from decimal import Decimal
import datetime
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path('/ml2/alz')
sys.path.insert(0, str(ROOT / 'analysis'))
from summarize_released_search import choose_anchor


class Refusal(ValueError):
    pass


def read(path):
    return json.loads(Path(path).read_bytes())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def require(condition, reason):
    if not condition:
        raise Refusal(reason)


def resolve(readout_path):
    protocol_path = ROOT / 'analysis/released_tuning/protocol.json'
    incumbent_path = ROOT / 'pure_incumbent.json'
    chooser_path = ROOT / 'analysis/summarize_released_search.py'
    protocol, incumbent, readout = map(read, (protocol_path, incumbent_path, readout_path))
    declared = {item['id']: item['parameters'] for item in protocol['stage1']['configs']}
    require(len(declared) == 27, 'Declared rate grid must retain all27 unique configurations')
    observed = readout.get('audited_complete_configurations', 0)
    require(readout.get('stage1_complete') is True and observed == 27,
            f'Stage2 requires all27 independently audited fixed20k rate configurations; observed{observed}')
    require(readout['status'] == 'RELEASED_RATE_GRID_READOUT', 'Unexpected rate readout schema')
    require(readout['source_SHA256'].get(str(protocol_path)) == sha(protocol_path), 'Rate protocol changed')
    require(readout['source_SHA256'].get(str(chooser_path)) == sha(chooser_path), 'Anchor chooser source changed')
    require(sha(incumbent_path) == protocol['frozen_science']['pure_incumbent_SHA256'], 'Frozen incumbent changed')
    rows = readout['rows']
    require(len(rows) == len({row['id'] for row in rows}) == 27 and {row['id'] for row in rows} == set(declared),
            'Rate configuration IDs are missing, duplicated or substituted')
    pins = {str(path): sha(path) for path in
            (Path(__file__).resolve(), readout_path, protocol_path, incumbent_path, chooser_path)}
    candidates = {}
    for row in rows:
        require(row['state'] == 'audited_complete' and row['parameters'] == declared[row['id']],
                'Every declared parameter setting must have an audited endpoint')
        path = Path(row['result_path'])
        require(path.is_relative_to(ROOT / 'analysis') and sha(path) == row['result_SHA256'], 'Audited result hash/path differs')
        result = read(path)
        for flag in ('owned_release_and_clean_settlement_verified',
                     'independent_saved_scores_checkpoint_and_count_exemplars_verified',
                     'predeclared_all_three_checkpoint_saved_scores_and_counts_verified',
                     'native_public_CUDA_reload_and_optimizer_state_byte_exact_verified',
                     'full_native_recipe_model_and_seed_streams_verified'):
            require(result.get(flag) is True, 'Missing completed native/score/count audit: ' + flag)
        candidate = result['candidate']
        require(candidate['tuning_candidate_id'] == row['id'] and candidate['released_hyperparameters'] == declared[row['id']],
                'Actual result parameter identity differs from declared grid')
        require(candidate['status'] == 'completed' and candidate['completed_updates'] == candidate['fixed_endpoint'] == 20000,
                'Partial or nonstandard endpoint cannot select a regularization anchor')
        require(candidate['training_seed'] == 0 and candidate['generation_replicates'] == 2,
                'Seed and generation replication must remain fixed')
        require(candidate['native_recipe'] == 'bispf_optimized' and candidate['pure_formulation'] == 'released_hyperparameters'
                and candidate['pure_only'] is True and candidate['real_bag_replacements'] == 0
                and candidate['integer_count_contract_passed'] is True, 'Released pure-count contract changed')
        require(candidate['preset_protocol_SHA256'] == sha(protocol_path), 'Actual result used a different protocol')
        for key in ('calibration_manifest_SHA256', 'trace_manifest_SHA256'):
            require(candidate[key] == protocol['frozen_science'][key], 'Frozen scientific input differs: ' + key)
        for key in ('energy_score', 'cell_energy_distance', 'spread_ratio'):
            require(candidate[key] == row[key], 'Readout metric differs from audited endpoint: ' + key)
        checkpoint = Path(candidate['checkpoint_path'])
        curve_path = Path(candidate['convergence_curve_path'])
        require(sha(checkpoint) == candidate['checkpoint_SHA256'], 'Actual endpoint checkpoint changed')
        require(sha(curve_path) == candidate['convergence_curve_SHA256'], 'Actual convergence curve changed')
        curve = read(curve_path)
        require(curve['training_finished_before_calibration'] is True and curve['best_checkpoint_selection'] is False,
                'Intermediate checkpoint selection or within-training calibration is forbidden')
        require([entry['completed_updates'] for entry in curve['entries']] == [5000, 10000, 20000], 'Missing declared checkpoints')
        require([entry['eligible_for_selection'] for entry in curve['entries']] == [False, False, True], 'Endpoint selection contract changed')
        auth_path = ROOT / 'artifacts/allocations' / result['attempt'] / 'authorization.json'
        require(sha(auth_path) == result['authorization_SHA256'], 'Actual execution authorization hash differs')
        pins.update({str(p): sha(p) for p in (path, checkpoint, curve_path, auth_path)})
        candidates[row['id']] = candidate
    anchor = choose_anchor(rows, incumbent)
    require(anchor is not None, 'No finite noncollapsed rate endpoint; investigate before stage2')
    require(readout['stage2_anchor'] == anchor, 'Saved anchor differs from the declared deterministic Pareto/minimax rule')
    base = dict(declared[anchor['id']])
    settings = []
    for field in ('reg_coeff', 'reg_kappa', 'prior_reg'):
        for factor, suffix in (('0.25', '025'), ('0.5', '050'), ('2', '200'), ('4', '400')):
            parameters = dict(base)
            parameters[field] = float(Decimal(str(base[field])) * Decimal(factor))
            require([key for key in base if parameters[key] != base[key]] == [field], 'Exactly one regularization field must change')
            settings.append({'id': 'reg_' + field + '_x' + suffix, 'changed_field': field,
                             'factor': float(factor), 'parameters': parameters})
    require(len(settings) == len({json.dumps(x['parameters'], sort_keys=True) for x in settings}) == 12,
            'Stage2 must retain twelve distinct one-field configurations')
    candidate = candidates[anchor['id']]
    return {
        'status': 'PROSPECTIVE_RELEASED_REGULARIZATION_STAGE2_RESOLVED_AFTER_ALL27_AUDITS_NOT_LAUNCHED',
        'utc': datetime.datetime.now(datetime.timezone.utc).isoformat(), 'cpu_fixture': False,
        'stage1_readout_path': str(readout_path), 'stage1_audited_rate_configurations': 27,
        'anchor': anchor, 'anchor_checkpoint_path': candidate['checkpoint_path'],
        'anchor_checkpoint_SHA256': candidate['checkpoint_SHA256'],
        'anchor_parameters': base, 'configs': settings, 'fixed_endpoint': 20000,
        'fold': 0, 'training_seed': 0, 'generation_replicates': 2,
        'checkpoints_at_updates': [5000, 10000, 20000],
        'native_recipe': 'bispf_optimized', 'frozen_science': protocol['frozen_science'],
        'source_SHA256': pins, 'GPU_job_started': False, 'GPU_allocation_created': False,
        'incumbent_promoted': False, 'stable_joint_ATH_claimed': False,
        'scope': 'Declared regularization stage only. Deterministic tuning anchor is not promotion. Each of12 settings is a fresh fixed20k same-family comparison, not a warm start or selected early checkpoint. Native source qualification, exact scope/parameter binding, fresh capacity review and one-use admission are still required before any GPU update. Four adaptive calibration donors; evaluation remains excluded.',
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--readout', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    require(not args.output.exists(), 'Preserve the existing prospective manifest; never overwrite')
    result = resolve(args.readout)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(json.dumps({'status': result['status'], 'configs': len(result['configs']), 'anchor_id': result['anchor']['id']}))


if __name__ == '__main__':
    try:
        main()
    except Refusal as error:
        print(str(error), file=sys.stderr)
        raise SystemExit(2)
