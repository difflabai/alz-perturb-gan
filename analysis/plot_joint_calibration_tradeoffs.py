"""Plot every saved complete seed-0 score; descriptive, never a selection step."""
import argparse
import datetime
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.ticker import ScalarFormatter
import numpy as np

ROOT = Path('/ml2/alz')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def plot(leaderboard, prefix):
    data = json.loads(leaderboard.read_bytes())
    baseline_path = ROOT / 'analysis/baseline_calibration_seed0.json'
    baseline = json.loads(baseline_path.read_bytes())
    # The incumbent and baseline in the leaderboard are the fixed scientific
    # references. No best-checkpoint, refitting, score averaging or new ranking.
    assert sha(ROOT / 'pure_incumbent.json') == data['selected_pure_incumbent_SHA256']
    points = [dict(name=r.get('formulation', r.get('pure_formulation', r.get('method'))),
                   ES=r['energy_score'], cell_ED=r['cell_energy_distance']) for r in data['rows']]
    by = {p['name']: p for p in points}
    assert len(by) == len(points) == data['completed_seed0_comparisons']
    assert all(np.isfinite([p['ES'], p['cell_ED']]).all() and min(p['ES'], p['cell_ED']) > 0 for p in points)
    incumbent = data['selected_pure_incumbent']
    selected = by['pure_ensemble75']
    assert selected['ES'] == incumbent['energy_score'] and selected['cell_ED'] == incumbent['cell_energy_distance']
    baseline_es, baseline_ed = baseline['energy_score'], baseline['cell_energy_distance']
    native_names = {'latent8', 'original_bispf', 'bcap', 'bcap_adam', 'ka2', 'k3p', 'halloween',
                    'native_routed_E22_count_adapter'}
    native = [p for p in points if p['name'] in native_names]
    rates = [p for p in points if p['name'].startswith('lr_')]
    pareto = [p for p in points if not any(q['ES'] <= p['ES'] and q['cell_ED'] <= p['cell_ED'] and
               (q['ES'] < p['ES'] or q['cell_ED'] < p['cell_ED']) for q in points)]
    limits = (1.85, 2.345, .218, .89)
    focused = [p for p in points if limits[0] <= p['ES'] <= limits[1] and limits[2] <= p['cell_ED'] <= limits[3]]
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10,
                         'axes.spines.top': False, 'axes.spines.right': False})
    fig, (full, zoom) = plt.subplots(1, 2, figsize=(12.3, 6.8), gridspec_kw={'width_ratios': [1, 1.4]})
    fig.subplots_adjust(left=.075, right=.975, top=.81, bottom=.25, wspace=.30)
    fig.suptitle('Measured energy and distribution tradeoffs', x=.075, y=.965, ha='left', fontsize=17, weight='bold')
    fig.text(.075, .915, f'{len(points)} completed pure-count seed-0 comparisons. Lower is better on both axes.', fontsize=11)
    fig.text(.075, .87, 'Released-API rate settings appear only after independent audits of their completed endpoints.', fontsize=10, color='#444444')
    for ax in (full, zoom):
        ax.scatter([p['ES'] for p in points], [p['cell_ED'] for p in points], s=35, color='#a8b2be', label='Other count-model variants', zorder=2)
        ax.scatter([p['ES'] for p in native], [p['cell_ED'] for p in native], s=43, color='#176ba0', marker='s', label='Native trainer presets / count adapters', zorder=3)
        ax.scatter([p['ES'] for p in rates], [p['cell_ED'] for p in rates], s=65, color='#21854c', marker='D', label='Completed released-API rate settings', zorder=5)
        ax.scatter([p['ES'] for p in pareto], [p['cell_ED'] for p in pareto], s=125, facecolors='none', edgecolors='#333333', linewidths=.8, zorder=4)
        ax.scatter(selected['ES'], selected['cell_ED'], s=160, marker='*', color='#d77a16', edgecolors='#5f390f', linewidths=.6, label='Frozen ensemble incumbent (40k total)', zorder=5)
        ax.scatter(baseline_es, baseline_ed, s=65, marker='X', color='#925c9e', label='Matched resampling, seed 0', zorder=4)
        ax.set_yscale('log'); ax.minorticks_off(); ax.grid(alpha=.15)
        ax.yaxis.set_major_formatter(ScalarFormatter())
        ax.set_xlabel('Donor energy score')
        ax.set_ylabel('Pooled cell energy distance (log scale)')
    full.set(title='All completed scores', xlim=(1.80, max(p['ES'] for p in points)*1.07),
             ylim=(.20, max(p['cell_ED'] for p in points)*1.4))
    full.set_yticks([.25, .5, 1, 2, 5])
    full.add_patch(Rectangle((limits[0], limits[2]), limits[1]-limits[0], limits[3]-limits[2],
                   fill=False, linestyle='--', edgecolor='#777777', linewidth=.8))
    full.text(2.54, .51, 'Dashed box:\nexpanded at right', fontsize=9, color='#555555')
    if 'halloween' in by:
        p = by['halloween']
        full.annotate('Halloween\nsevere collapse', (p['ES'], p['cell_ED']), xytext=(-100, -72),
                      textcoords='offset points', fontsize=9, arrowprops={'arrowstyle': '-', 'linewidth': .6})
    full.legend(loc='upper left', fontsize=7.8, frameon=False)
    zoom.set(title=f'Expanded view: {len(focused)} scores', xlim=limits[:2], ylim=limits[2:])
    zoom.set_yticks([.23, .3, .4, .6, .8])
    zoom.axvline(selected['ES'], color='#d77a16', linewidth=.7, linestyle=':', alpha=.6)
    zoom.axhline(selected['cell_ED'], color='#d77a16', linewidth=.7, linestyle=':', alpha=.6)
    labels = {'lr_g050_d050_p050': ('G/D/P half rates', (8, 26)),
              'network_rate_half': ('Half network rates', (-44, 28)),
              'pure_ensemble75': ('Frozen ensemble', (36, 8)),
              'native_routed_E22_count_adapter': ('Routed E22', (12, -14)),
              'bcap_adam': ('BCAP / Adam', (-85, -10))}
    for name, (label, offset) in labels.items():
        if name in by:
            p = by[name]
            zoom.annotate(label, (p['ES'], p['cell_ED']), xytext=offset, textcoords='offset points',
                          fontsize=8.5, arrowprops={'arrowstyle': '-', 'color': '#777777', 'linewidth': .6})
    fig.text(.075, .16, 'The first rate setting improves energy, but its cell distance remains above the frozen incumbent.', fontsize=10, weight='bold')
    fig.text(.075, .115, 'Seed-0 tuning only. The original ensemble failed joint confirmation on seeds 1 and 2; new rates are unconfirmed.', fontsize=9.5)
    fig.text(.075, .073, 'Four adaptive calibration donors; 20 evaluation donors excluded. Pareto rings describe saved scores, not promotion.', fontsize=9.5, color='#444444')
    for suffix in ('.png', '.pdf', '.json'):
        assert not prefix.with_suffix(suffix).exists(), 'Never overwrite a published readout'
    for suffix in ('.png', '.pdf'):
        fig.savefig(prefix.with_suffix(suffix), dpi=180, facecolor='white')
    plt.close(fig)
    out = {'status': 'SAVED_COMPLETE_SEED0_SCORES_DESCRIPTIVE_ONLY',
           'utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
           'all_points': points, 'focused_point_count': len(focused), 'nondominated_points': pareto,
           'new_training_or_generation': False, 'incumbent_changed': False, 'stable_joint_ATH_claimed': False,
           'source_SHA256': {str(p): sha(p) for p in (Path(__file__).resolve(), leaderboard, baseline_path, ROOT/'pure_incumbent.json')},
           'output_SHA256': {str(prefix.with_suffix(s)): sha(prefix.with_suffix(s)) for s in ('.png', '.pdf')}}
    prefix.with_suffix('.json').write_text(json.dumps(out, indent=2, allow_nan=False)+'\n')
    print(json.dumps({'points': len(points), 'focused_points': len(focused), 'PNG': str(prefix.with_suffix('.png'))}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--leaderboard', type=Path, required=True)
    parser.add_argument('--output-prefix', type=Path, required=True)
    args = parser.parse_args()
    plot(args.leaderboard, args.output_prefix)
