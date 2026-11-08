"""Portable figure from committed frozen-model aggregates; no model/data access."""
from pathlib import Path
import importlib.util
import json
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('report_style', ROOT / 'scripts/report/build_report.py')
style = importlib.util.module_from_spec(spec)
spec.loader.exec_module(style)
style.style()
plt = style.plt
q = json.loads((ROOT / 'analysis/PPARG_PCA_blindspot_CPU_20261108.json').read_bytes())
fig, axes = plt.subplots(1, 2, figsize=(11, 4.7), sharey=True)
stages = ['Not AD', 'Low', 'Intermediate', 'High']
panels = [
    ('Actual PPARG expression', 'observed_logmean', 'generated_logmean'),
    ('PPARG reconstructed from the 50 PCs', 'real_PC_reconstructed_logmean', 'generated_PC_reconstructed_logmean')]
for ax, (title, real, generated) in zip(axes, panels):
    for seed in range(5):
        ax.plot(range(4), [s['seed_means'][str(seed)][generated] for s in q['summary']], color=style.TEAL, alpha=.24, lw=1)
    ax.plot(range(4), [s[real] for s in q['summary']], color=style.INK, marker='s', lw=2, markersize=4, label='Observed')
    ax.plot(range(4), [s[generated] for s in q['summary']], color=style.TEAL, marker='o', lw=2.3, markersize=4, label='Generated')
    ax.set_title(title, fontsize=12, pad=15)
    ax.set_xticks(range(4), stages)
    ax.set_ylim(0, 1.6)
    ax.grid(axis='y', alpha=.2, color=style.MUTED)
    ax.legend(frameon=False, loc='upper left', fontsize=9, labelcolor=style.INK)
axes[0].set_ylabel('Mean log1p(panel CP10k)')
fig.suptitle('An aggregate match can hide a gene-level failure', fontsize=17, fontweight='bold', x=.065, ha='left', y=.99)
fig.text(.065, .895, 'Five frozen models · matched SEA-AD fitting depths · four sampling probes per model', color=style.MUTED, fontsize=10)
fig.text(.065, .025, 'Pooled cells; different averaging from the report’s subtype-balanced explorer. Faint amber: individual models.\nRight: a linear projection of expression, not another model prediction or a biological measurement.', color=style.MUTED, fontsize=9)
fig.tight_layout(rect=(.04, .115, 1, .87), w_pad=3)
folder = ROOT / 'analysis/figures'
folder.mkdir(exist_ok=True)
fig.savefig(folder / 'PPARG-PCA-blindspot-20261108.png', dpi=170, facecolor=style.BG)
svg = folder / 'PPARG-PCA-blindspot-20261108.svg'
fig.savefig(svg, facecolor=style.BG, metadata={'Date': None})
svg.write_text('\n'.join(line.rstrip() for line in svg.read_text().splitlines()) + '\n')
plt.close(fig)
print(folder / 'PPARG-PCA-blindspot-20261108.png')
