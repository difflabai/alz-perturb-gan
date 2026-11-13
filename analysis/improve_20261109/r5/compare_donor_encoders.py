"""Compare encoder seeds without interpreting raw, unaligned latent axes."""
import json
from pathlib import Path
import numpy as np

ROOT=Path('/ml2/alz/runs/improve_20261109')


def main():
    items=[json.loads((ROOT/f'smoke_r5/architecture_pass/encoder/fold0/seed{s}/evaluation.json').read_text()) for s in (0,1)]
    records=[{r['donor']:r for r in item['records']} for item in items]
    donors=sorted(set(records[0])&set(records[1]))
    held=np.array([records[0][d]['heldout'] for d in donors])
    z=[np.array([rec[d]['latent_a'] for d in donors]) for rec in records]
    # Alignment is fitted on TRAIN donors only; no held-out coordinate fitting.
    normalized=[(a-a[~held].mean(0))/a[~held].std(0).clip(1e-6) for a in z]
    u,_,vt=np.linalg.svd(normalized[0][~held].T@normalized[1][~held],full_matrices=False)
    aligned=normalized[0]@(u@vt)
    alignment_r2=1-np.sum((aligned[held]-normalized[1][held])**2)/np.sum(normalized[1][held]**2)
    profiles=[np.array([rec[d]['predicted_profile'] for d in donors]) for rec in records]
    deviations=[a-a[~held].mean(0) for a in profiles]
    corr=float(np.corrcoef(deviations[0][held].ravel(),deviations[1][held].ravel())[0,1])
    value={'fold':0,'n_heldout_donors':int(held.sum()),
           'seeds':[{k:item[k] for k in ('seed','steps','synthetic_validation_mse','synthetic_null_mse',
                                        'aggregate','passes_experimental_checks')} for item in items],
           'train_fitted_orthogonal_alignment_heldout_latent_r2':float(alignment_r2),
           'heldout_predicted_gene_deviation_correlation_across_seeds':corr,
           'decision':'experimental; neither encoder passes the real reconstruction guard',
           'scope':'two frozen generator seeds, one development fold, no biological axis identification'}
    path=ROOT/'results_r5/architecture_pass/encoder_comparison.json'
    path.write_text(json.dumps(value,indent=2)+'\n')
    print(json.dumps(value,indent=2))


if __name__=='__main__':main()
