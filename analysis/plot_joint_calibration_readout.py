"""Plot completed saved metrics only; no model generation or target selection."""
import argparse
import datetime
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
import numpy as np

ROOT = Path('/ml2/alz')


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def plot(leaderboard, prefix):
    confirmation = ROOT / 'analysis/pure_ensemble_confirmation_v6_all_seeds.json'
    d, c = json.loads(leaderboard.read_text()), json.loads(confirmation.read_text())
    rows = d['rows']
    assert len(rows) >= 21 and c['stable_joint_target_passed'] is False
    points = [{'name': x.get('formulation', x.get('pure_formulation', x.get('method'))),
               'ES': x['energy_score'], 'cell_ED': x['cell_energy_distance']} for x in rows]
    assert all(p['name'] and np.isfinite([p['ES'], p['cell_ED']]).all() for p in points)
    pareto = [p for p in points if not any(
        q['ES'] <= p['ES'] and q['cell_ED'] <= p['cell_ED']
        and (q['ES'] < p['ES'] or q['cell_ED'] < p['cell_ED']) for q in points)]
    presets = {'latent8', 'original_bispf', 'bcap', 'bcap_adam', 'ka2', 'k3p'}
    native = [p for p in points if p['name'] in presets]
    assert len(native) == 6
    incumbent = d['selected_pure_incumbent']
    selected = next(p for p in points if p['name'] == 'pure_ensemble75')
    assert selected['ES'] == incumbent['energy_score'] and selected['cell_ED'] == incumbent['cell_energy_distance']
    recs = c['records']
    assert [x['seed'] for x in recs] == [0, 1, 2]
    relative = [{'seed': x['seed'],
                 'ES_improvement_percent': 100 * (1 - x['energy_score'] / x['matched_baseline_energy_score']),
                 'cell_ED_improvement_percent': 100 * (1 - x['cell_energy_distance'] / x['matched_baseline_cell_energy_distance']),
                 'joint_pass': passed} for x, passed in zip(recs, c['per_seed_joint_improvement'])]
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10,
                         'axes.spines.top': False, 'axes.spines.right': False})
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(12.7, 6.2), gridspec_kw={'width_ratios': [1.12, 1]})
    fig.subplots_adjust(left=.075, right=.98, top=.80, bottom=.23, wspace=.30)
    fig.suptitle('Donor energy gains often come with worse cell distribution match',
                 x=.075, y=.965, ha='left', fontsize=16, weight='bold')
    fig.text(.075, .915, f'{len(points)} completed seed-0 pure-count comparisons; fixed confirmation uses the original three seeds.', color='#444444')
    ax.scatter([p['ES'] for p in points], [p['cell_ED'] for p in points],
               s=40, color='#acb4bf', alpha=.85, label='Other pure-count variants', zorder=2)
    ax.scatter([p['ES'] for p in native], [p['cell_ED'] for p in native],
               s=55, marker='s', color='#176ba0', label='Six native trainer presets', zorder=3)
    ax.scatter([p['ES'] for p in pareto], [p['cell_ED'] for p in pareto],
               s=145, facecolors='none', edgecolors='#333333', linewidths=1, zorder=4)
    ax.scatter(selected['ES'], selected['cell_ED'], s=190, marker='*', color='#d77a16',
               edgecolors='#5f390f', linewidths=.7, label='Selected pure ensemble', zorder=5)
    base = recs[0]
    ax.scatter(base['matched_baseline_energy_score'], base['matched_baseline_cell_energy_distance'],
               marker='X', s=90, color='#925c9e', label='Matched resampling, seed 0', zorder=4)
    labels = {'network_rate_half': ('Half network rate', (20, 24)),
              'native_MoG_prior_family': ('Native MoG', (30, 8)),
              'pure_ensemble75': ('Selected ensemble', (30, -2)),
              'bcap_adam': ('Native BCAP/Adam', (-112, -8))}
    for p in points:
        if p['name'] in labels:
            text, offset = labels[p['name']]
            ax.annotate(text, (p['ES'], p['cell_ED']), xytext=offset,
                        textcoords='offset points', fontsize=8.5,
                        arrowprops={'arrowstyle': '-', 'color': '#777777', 'linewidth': .6})
    ax.set(title='Completed seed-0 scores', xlabel='Donor energy score  (lower is better)',
           ylabel='Pooled cell energy distance\n(log scale; lower is better)')
    ax.set_yscale('log')
    ax.set_yticks([.23, .26, .30, .35, .5, .8])
    ax.yaxis.set_major_formatter(ScalarFormatter())
    ax.minorticks_off()
    ax.set_ylim(.218, .88)
    ax.set_xlim(min(p['ES'] for p in points) - .035, max(p['ES'] for p in points) + .025)
    ax.grid(alpha=.15)
    ax.legend(loc='upper left', fontsize=8, frameon=False)
    bx.set_xlim(0, 10)
    bx.set_ylim(-26, 11)
    bx.axhspan(0, 11, color='#e7f2e9', zorder=0)
    bx.axhline(0, color='#666666', linewidth=.8)
    bx.text(.5, 8.5, 'Both metrics improve', color='#2b6a3d', fontsize=9)
    for x in relative:
        color = '#268044' if x['joint_pass'] else '#bd4938'
        bx.scatter(x['ES_improvement_percent'], x['cell_ED_improvement_percent'], s=90, color=color, zorder=2)
        bx.annotate(f"Seed {x['seed']}", (x['ES_improvement_percent'], x['cell_ED_improvement_percent']),
                    xytext=(7, 6), textcoords='offset points', fontsize=10, color=color)
    bx.set(title='Selected ensemble: fixed-seed confirmation',
           xlabel='Donor ES improvement vs matched resampling (%)',
           ylabel='Cell ED improvement vs matched resampling (%)')
    bx.grid(alpha=.15)
    fig.text(.075, .13,
             f"Confirmation mean ES improves {100*c['mean_ES_relative_improvement']:.2f}%, but seeds 1 and 2 worsen cell match. Stable joint test: FAILED.",
             fontsize=10, weight='bold')
    fig.text(.075, .085,
             'Same four adaptive calibration donors throughout; seeds are training repeats, not independent biological donors.',
             fontsize=9, color='#444444')
    fig.text(.075, .05,
             '20 evaluation donors remain excluded. Pending stage-prior/Halloween work has no point on this figure. No SOTA or generalization claim.',
             fontsize=9, color='#444444')
    for ext in ('.png', '.pdf'):
        assert not prefix.with_suffix(ext).exists()
        fig.savefig(prefix.with_suffix(ext), dpi=180, facecolor='white')
    plt.close(fig)
    out = {'status': 'SAVED_METRIC_READOUT_NO_NEW_SELECTION_OR_QUALITY_MEASUREMENT',
           'utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
           'points': points, 'nondominated_seed0_points': pareto,
           'fixed_seed_confirmation_relative_metrics': relative,
           'stable_joint_target_passed': c['stable_joint_target_passed'],
           'source_SHA256': {str(p): sha(p) for p in [Path(__file__).resolve(), leaderboard, confirmation]},
           'output_SHA256': {str(prefix.with_suffix(ext)): sha(prefix.with_suffix(ext)) for ext in ('.png', '.pdf')},
           'incumbent_changed': False,
           'scope': 'Saved metrics only. Four adaptive calibration donors; no new generation, fitting, target access, decision threshold change, seed replacement or ranking of confirmation seeds. Nondomination is descriptive, not a replacement for the fixed joint selection protocol.'}
    assert not prefix.with_suffix('.json').exists()
    prefix.with_suffix('.json').write_text(json.dumps(out, indent=2, allow_nan=False) + '\n')
    print(json.dumps({'points': len(points), 'nondominated': pareto, 'confirmation': relative,
                      'PNG': str(prefix.with_suffix('.png')), 'PDF': str(prefix.with_suffix('.pdf'))}, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--leaderboard', type=Path, required=True)
    parser.add_argument('--output-prefix', type=Path, required=True)
    args = parser.parse_args()
    plot(args.leaderboard, args.output_prefix)
