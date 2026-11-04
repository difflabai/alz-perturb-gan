"""Compare independently audited convergence at matched G/D settings."""
import argparse
import datetime
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path('/ml2/alz')
read = lambda p: json.loads(Path(p).read_bytes())
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()


def plot(paths, prefix):
    assert len(paths) == 2
    records = []
    sources = [Path(__file__).resolve(), ROOT/'pure_incumbent.json', ROOT/'analysis/baseline_calibration_seed0.json']
    for path in paths:
        result = read(path)
        assert result['predeclared_all_three_checkpoint_saved_scores_and_counts_verified'] is True
        assert result['independent_saved_scores_checkpoint_and_count_exemplars_verified'] is True
        assert result['owned_release_and_clean_settlement_verified'] is True
        candidate = result['candidate']
        assert candidate['status'] == 'completed' and candidate['completed_updates'] == candidate['fixed_endpoint'] == 20000
        assert candidate['training_seed'] == 0 and candidate['generation_replicates'] == 2
        assert candidate['native_recipe'] == 'bispf_optimized' and candidate['real_bag_replacements'] == 0
        assert candidate['integer_count_contract_passed'] is True
        curvepath = Path(candidate['convergence_curve_path'])
        assert sha(curvepath) == candidate['convergence_curve_SHA256']
        assert sha(candidate['checkpoint_path']) == candidate['checkpoint_SHA256']
        curve = read(curvepath)
        assert curve['training_finished_before_calibration'] is True and curve['best_checkpoint_selection'] is False
        assert [e['completed_updates'] for e in curve['entries']] == [5000, 10000, 20000]
        assert [e['eligible_for_selection'] for e in curve['entries']] == [False, False, True]
        assert all(curve['entries'][-1]['metrics'][k] == candidate[k] for k in ('energy_score', 'cell_energy_distance', 'spread_ratio'))
        records.append({'candidate': candidate, 'entries': curve['entries'], 'result_path': str(path)})
        sources.extend([path, curvepath])
    first, second = [r['candidate'] for r in records]
    for key in ('donors', 'calibration_manifest_SHA256', 'trace_manifest_SHA256', 'sampling_contract', 'variant_config', 'preset_protocol_SHA256'):
        assert first[key] == second[key], 'Unmatched scientific input: '+key
    a, b = first['released_hyperparameters'], second['released_hyperparameters']
    assert {key for key in a if a[key] != b[key]} == {'particle_lr'}, 'Requires exactly one changed rate'
    protocol = ROOT/'analysis/released_tuning/protocol.json'
    assert sha(protocol) == first['preset_protocol_SHA256']
    assert sha(ROOT/'pure_incumbent.json') == read(protocol)['frozen_science']['pure_incumbent_SHA256']
    sources.append(protocol)
    incumbent = read(ROOT/'pure_incumbent.json')
    baseline = read(ROOT/'analysis/baseline_calibration_seed0.json')
    plt.rcParams.update({'font.family':'DejaVu Sans', 'font.size':10,
                         'axes.spines.top':False, 'axes.spines.right':False})
    fig, axes = plt.subplots(1, 2, figsize=(12.3, 5.9))
    fig.subplots_adjust(left=.07, right=.98, top=.77, bottom=.26, wspace=.26)
    fig.suptitle('Particle learning rate: matched convergence comparison', x=.07, y=.96, ha='left', fontsize=16, weight='bold')
    fig.text(.07, .895, f"Released bispf_optimized | G {a['generator_lr']:g}, D {a['critic_lr']:g} | fold 0, seed 0", fontsize=10)
    fig.text(.07, .84, 'Same four calibration donors; two generation replicas per checkpoint. Both runs complete 20,000 updates.', fontsize=10, color='#444444')
    for ax, key, title in zip(axes, ('energy_score', 'cell_energy_distance'), ('Donor energy score', 'Pooled cell energy distance')):
        ax.axhline(incumbent[key], color='#b66d16', linewidth=1.2, label='Frozen ensemble incumbent (40k total)')
        ax.axhline(baseline[key], color='#777777', linestyle='--', linewidth=1, label='Matched resampling, seed 0')
        for record, color in zip(records, ('#176ba0', '#21854c')):
            entries = record['entries']; rate = record['candidate']['released_hyperparameters']['particle_lr']
            ax.plot([e['completed_updates'] for e in entries], [e['metrics'][key] for e in entries],
                    color=color, marker='o', linewidth=1.8, label=f'Particle rate {rate:g}')
            dy = (9 if color=='#176ba0' else -15) if key=='energy_score' else 9
            ax.annotate(f"{entries[-1]['metrics'][key]:.5f}", (20000, entries[-1]['metrics'][key]),
                        xytext=(-5, dy), textcoords='offset points', ha='right', fontsize=9, color=color)
        ax.set_title(title+' (lower is better)', loc='left', fontsize=11)
        ax.set_xlim(4000, 21000); ax.set_xticks([5000, 10000, 15000, 20000]); ax.grid(alpha=.18)
        ax.set_xlabel('Completed optimizer pairs'); ax.legend(fontsize=8, loc='best')
    fig.text(.07, .15, 'Only the final 20k endpoint can select a model; 5k and 10k scores diagnose convergence.', fontsize=10, weight='bold')
    fig.text(.07, .10, 'Adaptive calibration, one training seed. These curves do not establish a stable joint ATH or evaluation performance.', fontsize=9.5)
    fig.text(.07, .055, 'Parameter-dependent random draws may diverge despite fixed seed streams. The 20 evaluation donors remain excluded.', fontsize=9.5, color='#444444')
    for ext in ('.png', '.pdf', '.json'):
        assert not prefix.with_suffix(ext).exists(), 'Never replace a published readout'
    for ext in ('.png', '.pdf'):
        fig.savefig(prefix.with_suffix(ext), dpi=180, facecolor='white')
    plt.close(fig)
    record = {'status':'TWO_AUDITED_MATCHED_GD_PARTICLE_RATE_CONVERGENCE_READOUT',
              'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
              'changed_hyperparameter':'particle_lr', 'records':records,
              'incumbent_changed':False, 'stable_joint_ATH_claimed':False,
              'source_SHA256':{str(p):sha(p) for p in sources},
              'output_SHA256':{str(prefix.with_suffix(ext)):sha(prefix.with_suffix(ext)) for ext in ('.png', '.pdf')}}
    prefix.with_suffix('.json').write_text(json.dumps(record, indent=2, allow_nan=False)+'\n')
    print(json.dumps({'PNG':str(prefix.with_suffix('.png')), 'completed_models':2}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--result', type=Path, action='append', required=True)
    parser.add_argument('--output-prefix', type=Path, required=True)
    args = parser.parse_args()
    plot(args.result, args.output_prefix)
