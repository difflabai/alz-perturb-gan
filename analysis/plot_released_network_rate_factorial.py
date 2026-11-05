"""Describe the four audited fixed-endpoint cells of one G/D rate slice."""
import argparse
import datetime
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path('/ml2/alz')
read = lambda path: json.loads(Path(path).read_bytes())
sha = lambda path: hashlib.sha256(Path(path).read_bytes()).hexdigest()


def plot(paths, prefix):
    assert len(paths) == 4, 'Four completed audited endpoints required'
    protocol_path = ROOT / 'analysis/released_tuning/protocol.json'
    incumbent_path = ROOT / 'pure_incumbent.json'
    protocol, incumbent = read(protocol_path), read(incumbent_path)
    assert sha(incumbent_path) == protocol['frozen_science']['pure_incumbent_SHA256']
    settings = {row['id']: row['parameters'] for row in protocol['stage1']['configs']}
    records = []
    sources = [Path(__file__).resolve(), protocol_path, incumbent_path]
    for path in paths:
        result = read(path)
        for key in ('predeclared_all_three_checkpoint_saved_scores_and_counts_verified',
                    'independent_saved_scores_checkpoint_and_count_exemplars_verified',
                    'native_public_CUDA_reload_and_optimizer_state_byte_exact_verified',
                    'owned_release_and_clean_settlement_verified'):
            assert result[key] is True, key
        candidate = result['candidate']
        assert candidate['status'] == 'completed' and candidate['completed_updates'] == candidate['fixed_endpoint'] == 20000
        assert candidate['training_seed'] == 0 and candidate['generation_replicates'] == 2
        assert candidate['pure_only'] and candidate['real_bag_replacements'] == 0 and candidate['integer_count_contract_passed']
        assert candidate['native_recipe'] == 'bispf_optimized' and candidate['pure_formulation'] == 'released_hyperparameters'
        assert candidate['released_hyperparameters'] == settings[candidate['tuning_candidate_id']]
        assert candidate['preset_protocol_SHA256'] == sha(protocol_path)
        assert sha(candidate['checkpoint_path']) == candidate['checkpoint_SHA256']
        curve_path = Path(candidate['convergence_curve_path'])
        assert sha(curve_path) == candidate['convergence_curve_SHA256']
        curve = read(curve_path)
        assert curve['training_finished_before_calibration'] is True and curve['best_checkpoint_selection'] is False
        assert curve['entries'][-1]['eligible_for_selection'] and curve['entries'][-1]['completed_updates'] == 20000
        for key in ('energy_score', 'cell_energy_distance', 'spread_ratio'):
            assert candidate[key] == curve['entries'][-1]['metrics'][key]
        records.append({'candidate_id': candidate['tuning_candidate_id'],
                        'parameters': candidate['released_hyperparameters'],
                        'metrics': {key: candidate[key] for key in ('energy_score', 'cell_energy_distance', 'spread_ratio')},
                        'decision': result['decision'], 'result_path': str(path), 'result_SHA256': sha(path)})
        sources.extend([path, curve_path])
        if len(records) > 1:
            first = read(paths[0])['candidate']
            for key in ('donors', 'calibration_manifest_SHA256', 'trace_manifest_SHA256', 'sampling_contract', 'variant_config'):
                assert candidate[key] == first[key], 'Unmatched scientific input: ' + key
    assert len({row['candidate_id'] for row in records}) == 4
    generators = sorted({row['parameters']['generator_lr'] for row in records})
    critics = sorted({row['parameters']['critic_lr'] for row in records})
    assert len(generators) == len(critics) == 2
    assert generators == [.006, .012] and critics == [.009, .018], 'This is the declared half/full network-rate slice'
    constants = {key: records[0]['parameters'][key] for key in ('particle_lr', 'reg_coeff', 'reg_kappa', 'prior_reg')}
    assert all(all(row['parameters'][key] == value for key, value in constants.items()) for row in records)
    lookup = {(row['parameters']['generator_lr'], row['parameters']['critic_lr']): row for row in records}
    assert set(lookup) == {(g, d) for g in generators for d in critics}, 'Retain every cell of the two-by-two comparison'
    for ext in ('.png', '.pdf', '.json'):
        assert not prefix.with_suffix(ext).exists(), 'Preserve published readouts'
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 11,
                         'axes.spines.top': False, 'axes.spines.right': False})
    fig, axes = plt.subplots(1, 2, figsize=(11.6, 6.8))
    fig.subplots_adjust(left=.08, right=.97, top=.77, bottom=.27, wspace=.32)
    fig.suptitle('Generator and critic rates: observed interaction', x=.08, y=.96, ha='left', fontsize=17, weight='bold')
    fig.text(.08, .90, f"Released BISPF | particles {constants['particle_lr']:g} | fixed 20,000 updates | fold 0, seed 0", fontsize=11)
    fig.text(.08, .84, 'All four completed endpoints independently audited. Lower is better for both metrics.', fontsize=10.5, color='#444444')
    contrasts = {}
    for axis, key, title in zip(axes, ('energy_score', 'cell_energy_distance'), ('Donor energy score', 'Pooled cell energy distance')):
        values = np.array([[lookup[g, d]['metrics'][key] for d in critics] for g in generators])
        axis.imshow(values, cmap='YlOrRd', aspect='auto')
        for i, g in enumerate(generators):
            for j, d in enumerate(critics):
                value = values[i, j]
                relative = 100 * (value / incumbent[key] - 1)
                color = 'white' if value > values.min() + .7 * (values.max() - values.min()) else '#222222'
                axis.text(j, i, f'{value:.5f}\n{relative:+.2f}% vs incumbent', ha='center', va='center', color=color, fontsize=12)
        axis.set_xticks(range(2), [f'{value:g}' for value in critics])
        axis.set_yticks(range(2), [f'{value:g}' for value in generators])
        axis.set_xlabel('Critic learning rate')
        axis.set_ylabel('Generator learning rate')
        axis.set_title(title, loc='left', fontsize=12, pad=12)
        contrasts[key] = {
            'G_full_minus_half_at_D_half': float(values[1, 0] - values[0, 0]),
            'G_full_minus_half_at_D_full': float(values[1, 1] - values[0, 1]),
            'difference_of_differences': float(values[1, 1] - values[0, 1] - values[1, 0] + values[0, 0]),
            'frozen_incumbent_value': incumbent[key],
        }
    fig.text(.08, .17, 'Each cell is a final endpoint; a promising energy value alone cannot select a model.', fontsize=10.5, weight='bold')
    fig.text(.08, .12, 'Four adaptive calibration donors; two generation replicas per model. These are descriptive fixed-seed contrasts.', fontsize=10)
    fig.text(.08, .075, 'Parameter-dependent random draws can diverge. Evaluation donors remain excluded; no stable joint ATH is claimed.', fontsize=10, color='#444444')
    for ext in ('.png', '.pdf'):
        fig.savefig(prefix.with_suffix(ext), dpi=180, facecolor='white')
    plt.close(fig)
    result = {'status': 'ALL_FOUR_AUDITED_FIXED20K_NETWORK_RATE_FACTORIAL_ENDPOINTS_DESCRIPTIVE_ONLY',
              'utc': datetime.datetime.now(datetime.timezone.utc).isoformat(), 'records': records,
              'constant_parameters': constants, 'generator_rates': generators, 'critic_rates': critics,
              'descriptive_contrasts': contrasts, 'new_training_or_generation': False,
              'incumbent_changed': False, 'stable_joint_ATH_claimed': False,
              'source_SHA256': {str(path): sha(path) for path in sources},
              'output_SHA256': {str(prefix.with_suffix(ext)): sha(prefix.with_suffix(ext)) for ext in ('.png', '.pdf')}}
    prefix.with_suffix('.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(json.dumps({'PNG': str(prefix.with_suffix('.png')), 'audited_endpoints': 4, 'contrasts': contrasts}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--result', type=Path, action='append', required=True)
    parser.add_argument('--output-prefix', type=Path, required=True)
    args = parser.parse_args()
    plot(args.result, args.output_prefix)
