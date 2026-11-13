"""Summarize the two fixed inverse-fit follow-ups and their across-seed check."""
import json
from pathlib import Path
import numpy as np

ROOT=Path('/ml2/alz/runs/improve_20261109')


def main():
    results=[json.loads((ROOT/f'smoke_r5/architecture_pass/inverse/fold0/seed{s}/evaluation.json').read_text()) for s in (0,1)]
    rows=[{r['donor']:r for r in value['records']} for value in results]
    donors=sorted(set(rows[0])&set(rows[1]))
    held=np.array([rows[0][d]['heldout'] for d in donors])
    profiles=[np.array([r[d]['predicted_profile'] for d in donors]) for r in rows]
    deviations=[a-a[~held].mean(0) for a in profiles]
    correlation=float(np.corrcoef(deviations[0][held].ravel(),deviations[1][held].ravel())[0,1])
    keys=('fold','seed','steps','aggregate','fit_diagnostics','synthetic_latent_standardized_mse',
          'synthetic_null_mse','passes_initial_checks','GAN_weights_unchanged','wall_seconds',
          'fit_final_sha256','calibration_receipt_sha256')
    value={'seeds':[{k:r[k] for k in keys} for r in results],
           'heldout_predicted_gene_deviation_correlation_across_seeds':correlation,
           'decision':'stop this inverse-fit screen; neither seed passes both real reconstruction and repeatability',
           'scope':'final-state latent inversion; input and target cell halves disjoint; no GAN parameter updates'}
    path=ROOT/'results_r5/architecture_pass/inverse_comparison.json'
    path.write_text(json.dumps(value,indent=2)+'\n')
    print(json.dumps(value,indent=2))


if __name__=='__main__':main()
