"""Render the frozen interval/architecture diagnostics as a standalone figure."""
from pathlib import Path
import json
import sys
sys.dont_write_bytecode = True
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

STUDY = Path(__file__).resolve().parents[1]
RUNS = Path('/ml2/alz/runs/improve_20261109/results_r5/interval_architecture')


def main():
    audit = json.loads((RUNS / 'audit.json').read_text())
    probe = json.loads((RUNS / 'noise_probe.json').read_text())
    fig, axes = plt.subplots(1, 3, figsize=(13.7, 4.8), gridspec_kw={'width_ratios': [1, 1, 1.5]})
    plt.rcParams.update({'font.size': 10})
    colors = ['#265a80', '#c55a44', '#65767d']
    groups = ['PC means', 'PC standard deviations', 'all coordinates']
    coverage = [100 * audit['summary']['heldout']['raw'][g]['coverage'] for g in groups]
    bars = axes[0].bar(np.arange(3), coverage, color=colors, width=.62)
    axes[0].set_xticks(np.arange(3), ['Expression\nmeans', 'Cellular\nspread', 'Combined'])
    axes[0].set_ylim(0, 102)
    axes[0].axhline(80, color='#555', linestyle='--', linewidth=1)
    axes[0].bar_label(bars, labels=[f'{x:.1f}%' for x in coverage], padding=4)
    axes[0].set_ylabel('Observed coordinate coverage (%)')
    axes[0].set_title('A. What the combined score hides', loc='left', fontsize=11)
    control = audit['projection_control']
    ratios = [100 * np.mean([v[method]['heldout_to_train_ratio'] for v in control.values()])
              for method in ('train_PCA', 'fixed_random')]
    bars = axes[1].bar([0, 1], ratios, color=['#c55a44', '#265a80'], width=.6)
    axes[1].set_xticks([0, 1], ['Train-selected\nPC directions', 'Fixed random\ndirections'])
    axes[1].set_ylim(0, 115)
    axes[1].axhline(100, color='#555', linestyle='--', linewidth=1)
    axes[1].bar_label(bars, labels=[f'{x:.1f}%' for x in ratios], padding=4)
    axes[1].set_ylabel('Real held-out / train cellular SD (%)')
    axes[1].set_title('B. Feature-space generalization', loc='left', fontsize=11)
    variants = ['baseline', 'mean_contract_half', 'theta_times4', 'poisson_limit']
    values = [100 * probe['summary'][v]['SD_coverage'] for v in variants]
    bars = axes[2].bar(np.arange(4), values, color=['#65767d', '#265a80', '#c55a44', '#c55a44'], width=.63)
    axes[2].set_xticks(np.arange(4), ['Baseline', 'Half network\nmean variation', '4× NB\nshape', 'Poisson\nlimit'])
    axes[2].set_ylim(0, 102)
    axes[2].axhline(80, color='#555', linestyle='--', linewidth=1)
    axes[2].bar_label(bars, labels=[f'{x:.1f}%' for x in values], padding=4)
    axes[2].set_ylabel('Cellular-spread coverage (%)')
    axes[2].set_title('C. Frozen decoder interventions', loc='left', fontsize=11)
    for ax in axes:
        ax.spines[['top', 'right']].set_visible(False)
        ax.grid(axis='y', alpha=.15)
        ax.set_axisbelow(True)
    fig.suptitle('The interval shortfall is concentrated in cellular spread', x=.05, ha='left', fontsize=15)
    fig.text(.05, .02, 'A/B: 18 fits, nine overlapping development folds. C: six fits, original folds 0–2; local interventions only.\n'
             'All reuse 37 development donors. Dashed 80% is the nominal target; finite M=32 adds percentile-estimation bias.', fontsize=9, color='#444')
    fig.tight_layout(rect=[0, .1, 1, .91])
    fig.savefig(STUDY / 'interval_architecture_20261113.png', dpi=160, facecolor='white')
    print(STUDY / 'interval_architecture_20261113.png')


if __name__ == '__main__':
    main()
