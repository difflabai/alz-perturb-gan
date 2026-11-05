"""Descriptive paired donor deltas from audited endpoints, with no new scoring."""
import datetime
import hashlib
import json
from pathlib import Path

ROOT = Path('/ml2/alz')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    board_path = ROOT / 'analysis/pure_formulations_leaderboard_after_fourth_released_rate.json'
    board = json.loads(board_path.read_bytes())
    best = next(row for row in board['rows'] if row.get('formulation') == 'lr_g050_d050_p050')
    sources = [Path(__file__).resolve(), board_path, ROOT / 'pure_incumbent.json']
    comparisons = []
    for row in board['rows']:
        name = row.get('formulation', '')
        if not (name.startswith('lr_') or name == 'network_rate_half'):
            continue
        evidence = Path(row['decision_evidence'])
        result = json.loads(evidence.read_bytes())
        candidate = result['candidate']
        assert candidate['completed_updates'] == candidate['fixed_endpoint'] == 20000
        assert candidate['energy_score'] == row['energy_score']
        assert candidate['cell_energy_distance'] == row['cell_energy_distance']
        for key in ('donors', 'training_seed', 'calibration_manifest_SHA256',
                    'trace_manifest_SHA256', 'sampling_contract', 'generation_replicates'):
            assert candidate[key] == best[key], key
        assert candidate['training_seed'] == 0 and candidate['generation_replicates'] == 2
        assert len(candidate['donors']) == 4 and candidate['integer_count_contract_passed']
        if name.startswith('lr_'):
            assert result['predeclared_all_three_checkpoint_saved_scores_and_counts_verified']
            assert result['owned_release_and_clean_settlement_verified']
            parameters = candidate['released_hyperparameters']
        else:
            assert candidate['network_rate_config'] == {'lr': .006, 'prior_lr_mult': 5.0}
            parameters = {**best['released_hyperparameters'], 'particle_lr': .03}
        changed = {k: {'reference': best['released_hyperparameters'][k], 'compared': value}
                   for k, value in parameters.items() if value != best['released_hyperparameters'][k]}
        assert len(changed) <= 1, 'This readout is a matched one-field comparison'
        deltas = [a - b for a, b in zip(candidate['per_donor_energy_score'], best['per_donor_energy_score'])]
        assert abs(sum(deltas) / 4 - (candidate['energy_score'] - best['energy_score'])) < 1e-12
        comparisons.append({
            'candidate_id': name, 'changed_public_scalar_fields': changed,
            'energy_score': candidate['energy_score'],
            'cell_energy_distance': candidate['cell_energy_distance'],
            'spread_ratio': candidate['spread_ratio'],
            'energy_difference_vs_reference': candidate['energy_score'] - best['energy_score'],
            'pooled_cell_ED_difference_vs_reference': candidate['cell_energy_distance'] - best['cell_energy_distance'],
            'per_donor_energy_score': dict(zip(candidate['donors'], candidate['per_donor_energy_score'])),
            'per_donor_energy_difference_vs_reference': dict(zip(candidate['donors'], deltas)),
            'donors_improved_vs_reference': sum(x < 0 for x in deltas),
            'donors_worsened_vs_reference': sum(x > 0 for x in deltas),
            'decision': result['decision'], 'result_path': str(evidence), 'result_SHA256': sha(evidence),
        })
        sources.append(evidence)
    assert len(comparisons) == 5  # four released settings plus the historical P=.03 control
    out = {
        'status': 'DESCRIPTIVE_MATCHED_DONOR_DELTAS_FROM_FOUR_AUDITED_RELEASED_RATES',
        'utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'reference_candidate_id': best['formulation'],
        'reference_parameters': best['released_hyperparameters'],
        'comparisons': comparisons,
        'source_SHA256': {str(path): sha(path) for path in sources},
        'interpretation': 'Negative deltas favor the compared setting. A lower aggregate energy score does not imply every calibration donor improved. One fixed training seed, four adaptive calibration donors and two generation replicates; no biological replication or statistical significance claim. Pooled cell ED is retained as scored, not reconstructed by averaging stage distances. No new training, generation, checkpoint selection, incumbent promotion, protocol change or evaluation access.',
        'stable_joint_ATH_claimed': False,
    }
    destination = ROOT / 'analysis/released_rate_four_completed_donor_deltas.json'
    assert not destination.exists()
    destination.write_text(json.dumps(out, indent=2, allow_nan=False) + '\n')
    print(json.dumps({'path': str(destination), 'compared_endpoints': len(comparisons)}))


if __name__ == '__main__':
    main()
