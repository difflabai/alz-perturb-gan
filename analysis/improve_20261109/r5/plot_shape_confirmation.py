"""Plot the completed fixed-readout confirmation, without partial selection."""
import os
os.environ.setdefault('MPLBACKEND', 'Agg')
import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
BASE = HERE.parents[2]/'runs/improve_20261109/results_r5/architecture_pass/shape_confirmation'


def main():
    result = json.loads((BASE/'comparison.json').read_text())
    if not result['complete'] or result['n_pairs'] != 12:
        raise ValueError('plot requires the full12-pair confirmation')
    reference, candidate = (result['means'][key] for key in ('default', 'count_shape'))
    blue, green = '#7c96ad', '#267b65'
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))
    fig.subplots_adjust(left=.12, right=.985, bottom=.22, top=.78, wspace=.35)
    fig.suptitle('Count-shape readout: completed development comparison', fontsize=15, y=.98)
    fig.text(.5, .89, '12 paired readouts on six overlapping splits of the same 37 donors; M = 32, K = 256',
             ha='center', fontsize=10)
    names = ('pb_rmse', 'meanlog_rmse_gene', 'nonzero_rmse_gene', 'logvar_rmse_gene')
    labels = ('Profile error', 'Mean-expression error', 'Detection error', 'Variance error')
    ratios = [100*candidate['historical'][m]/reference['historical'][m] for m in names]
    axes[0].barh(np.arange(4), ratios, color=green, height=.6)
    axes[0].set_yticks(np.arange(4), labels)
    axes[0].invert_yaxis()
    axes[0].axvline(100, color=blue, linestyle='--', linewidth=1.3)
    axes[0].set_xlim(0, max(115, max(ratios)+10))
    axes[0].set_xlabel('Error as % of bias-calibrated reference')
    axes[0].set_title('Gene-level errors\n(lower is better)', fontsize=11)
    for row, ratio in enumerate(ratios):
        axes[0].text(ratio+1, row, f'{ratio:.2f}%', va='center', fontsize=9)
    groups = ('mean', 'sd', 'combined')
    x = np.arange(3)
    for offset, source, color, label in ((-.18, reference, blue, 'Bias calibration'),
                                          (.18, candidate, green, 'Count-shape')):
        values = [100*source['intervals'][group]['coverage'] for group in groups]
        bars = axes[1].bar(x+offset, values, width=.36, color=color, label=label)
        axes[1].bar_label(bars, fmt='%.1f', fontsize=8, padding=2)
    axes[1].axhline(80, color='#b25c3a', linestyle='--', linewidth=1.2)
    axes[1].set_xticks(x, ('Pattern\nmeans', 'Cellular\nvariability', 'Combined'))
    axes[1].set_ylim(0, 100)
    axes[1].set_ylabel('Observed summaries covered (%)')
    axes[1].set_title('Held-out marginal ranges\nnominal target = 80%', fontsize=11)
    axes[1].legend(loc='lower center', fontsize=8)
    for label, color, marker, text in (('default', blue, 'o', 'Bias calibration'),
                                        ('count_shape', green, 's', 'Count-shape')):
        values = [np.mean([json.loads((BASE/label/f'fold{f}/seed{s}.json').read_text())
                  ['intervals']['heldout']['percentile']['sd']['interval_score'] for s in (0, 1)])
                  for f in range(3, 9)]
        axes[2].plot(range(3, 9), values, color=color, marker=marker, label=text)
    axes[2].set_xticks(range(3, 9), [f'F{f}' for f in range(3, 9)])
    axes[2].set_ylabel('Central-80% interval score')
    axes[2].set_title('Variability interval score\n(lower is better)', fontsize=11)
    axes[2].set_ylim(bottom=0)
    axes[2].legend(fontsize=8)
    width = [source['intervals']['sd']['width'] for source in (reference, candidate)]
    fig.text(.065, .09, f'Mean variability-range width: {width[0]:.3f} → {width[1]:.3f}. '
             'Interval score combines width and penalties for misses.', fontsize=9)
    train = []
    for label in ('default', 'count_shape'):
        rows = [json.loads((BASE/label/f'fold{f}/seed{s}.json').read_text())
                ['intervals']['train']['percentile']['sd'] for f in range(3, 9) for s in (0, 1)]
        train.append({key: np.mean([row[key] for row in rows]) for key in ('coverage', 'interval_score')})
    fig.text(.065, .055, f'Training variability fit worsens: coverage {100*train[0]["coverage"]:.1f}% → '
             f'{100*train[1]["coverage"]:.1f}%; interval score {train[0]["interval_score"]:.3f} → '
             f'{train[1]["interval_score"]:.3f}.', fontsize=9, color='#a94e2e')
    fig.text(.065, .02, 'Panels show held-out development donors; intervals describe projected cell-population summaries. '
             'These overlapping splits do not provide independent biological validation.', fontsize=8)
    for ax in axes:
        ax.spines[['top', 'right']].set_visible(False)
        ax.tick_params(labelsize=9)
    for suffix in ('png', 'svg'):
        destination = HERE.parent/f'count_shape_confirmation_20261113.{suffix}'
        fig.savefig(destination, dpi=180)
        if suffix == 'svg':
            destination.write_text('\n'.join(line.rstrip() for line in destination.read_text().splitlines())+'\n')
    plt.close(fig)


if __name__ == '__main__':
    main()
