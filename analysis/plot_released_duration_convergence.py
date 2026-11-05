"""Plot audited parent and declared extension checkpoints without early picking."""
import hashlib
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path('/ml2/alz')
read = lambda p: json.loads(Path(p).read_bytes())
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    summary_path = ROOT / 'analysis/released_best_40k_duration_comparison.json'
    summary = read(summary_path)
    assert summary['status'] == 'INDEPENDENTLY_AUDITED_RELEASED_20K_TO40K_DURATION_COMPARISON'
    protocol_path = ROOT / 'analysis/released_best_rate_extension40k_protocol.json'
    protocol = read(protocol_path)
    parent_path = Path(protocol['parent_result_path'])
    result_path = ROOT / 'analysis' / (summary['attempt'] + '-result.json')
    parent, result = read(parent_path), read(result_path)
    parent_curve_path = Path(parent['candidate']['convergence_curve_path'])
    curve_path = Path(result['candidate']['convergence_curve_path'])
    assert sha(parent_curve_path) == parent['candidate']['convergence_curve_SHA256']
    assert sha(curve_path) == result['candidate']['convergence_curve_SHA256']
    entries = read(parent_curve_path)['entries'] + read(curve_path)['entries']
    assert [e['completed_updates'] for e in entries] == [5000, 10000, 20000, 30000, 40000]
    for source in (parent_path, result_path, protocol_path):
        assert summary['sources_SHA256'][str(source)] == sha(source)
    incumbent = read(ROOT / 'pure_incumbent.json')
    baseline = read(ROOT / 'analysis/baseline_calibration_seed0.json')
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False})
    fig, axes = plt.subplots(1, 3, figsize=(14, 5.5))
    fig.subplots_adjust(left=.065, right=.98, top=.72, bottom=.24, wspace=.33)
    fig.suptitle('Released ParticleGAN: does longer training improve the best rate setting?',
                 x=.065, y=.96, ha='left', fontsize=16, weight='bold')
    fig.text(.065, .875, 'Same model and seed | generator LR 0.006, critic LR 0.009, particle LR 0.015', fontsize=11)
    fig.text(.065, .815, 'Parent 20k results informed the duration test. Only the declared 40k extension endpoint selects it.', fontsize=10)
    fields = [('energy_score', 'Donor energy score'), ('cell_energy_distance', 'Pooled cell energy distance'), ('spread_ratio', 'Donor spread ratio')]
    for ax, (key, label) in zip(axes, fields):
        x = [e['completed_updates'] / 1000 for e in entries]
        y = [e['metrics'][key] for e in entries]
        ax.axvspan(20, 40, color='#e8f0f7', alpha=.65)
        ax.axhline(incumbent[key], color='#ad6e17', label='Frozen incumbent')
        ax.axhline(baseline[key], color='#777777', linestyle='--', label='Matched resampling')
        ax.plot(x, y, color='#176ba0', marker='o', linewidth=1.5, label='Audited same-model checkpoints')
        ax.scatter([40], [y[-1]], s=70, marker='s', color='#176ba0', zorder=4)
        for xx, yy in zip(x, y):
            ax.annotate(f'{yy:.4f}', (xx, yy), xytext=(0, 7), textcoords='offset points', ha='center', fontsize=8)
        ax.margins(y=.25)
        ax.set_xlim(0, 43); ax.set_xticks([5, 10, 20, 30, 40]); ax.set_xlabel('Completed updates (thousands)')
        ax.set_title(label + (' (lower is better)' if key != 'spread_ratio' else ''), fontsize=10, loc='left')
        ax.grid(alpha=.17)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower left', bbox_to_anchor=(.06, .115), ncol=3, frameon=False, fontsize=9)
    fig.text(.065, .065, 'Four adaptive calibration donors; two generation replicates per checkpoint. Evaluation donors excluded.', fontsize=9)
    fig.text(.065, .03, 'Shading: new 20k→40k phase. Square: fixed 40k endpoint. Other points retain their original diagnostic/historical roles.', fontsize=9)
    prefix = ROOT / 'analysis/released_best_rate_20k_to40k_audited_convergence'
    for suffix in ('.png', '.pdf', '.json'):
        assert not prefix.with_suffix(suffix).exists()
    fig.savefig(prefix.with_suffix('.png'), dpi=150); fig.savefig(prefix.with_suffix('.pdf')); plt.close(fig)
    metadata = {'status': 'AUDITED_SAME_MODEL_DURATION_CONVERGENCE', 'eligible_extension_endpoint': 40000,
                'checkpoints': [{'updates': e['completed_updates'], **{key: e['metrics'][key] for key, _ in fields}} for e in entries],
                'sources_SHA256': {str(p): sha(p) for p in (Path(__file__), summary_path, protocol_path, parent_path, result_path, parent_curve_path, curve_path)},
                'parent_calibration_informed_extension': True, 'incumbent_changed': False, 'stable_joint_ATH_claimed': False}
    prefix.with_suffix('.json').write_text(json.dumps(metadata, indent=2, allow_nan=False) + '\n')
    print(prefix.with_suffix('.png'))


if __name__ == '__main__':
    main()
