"""Explain the completed Fourier128 cell-score failure using saved calibration arrays.

This is descriptive, not a new ensemble screen or candidate selection. The
existing exact energy-kernel decomposition is reused without changing metrics.
"""
import datetime,json,sys
from pathlib import Path
import numpy as np
ROOT=Path('/ml2/alz');sys.path[:0]=[str(ROOT),str(ROOT/'analysis')]
from calibration import targets
from diagnose_confirmation_cell_match import decompose,sha,read

def main():
    output=ROOT/'analysis/fourier128_cell_failure_decomposition.json'
    assert not output.exists()
    proof_path=ROOT/'analysis/CAL_GPU0_V1_fourier128_v1_f0_pg_s0-result.json'
    proof=read(proof_path)
    assert proof['status']=='VERIFIED_FIXED_FOURIER128_NATIVE_RESULT'
    assert proof['independent_saved_scores_checkpoint_and_count_exemplars_verified']
    assert proof['owned_release_and_clean_settlement_verified'] and not proof['decision']['accept']
    manifest,_,target_cells=targets();real=target_cells['by_stage']
    np.testing.assert_array_equal(real.reshape(-1,50),target_cells['equal_stage'])
    pins={str(proof_path):sha(proof_path),str(ROOT/'data/calibration_fold0/cell_reservoir.npz'):sha(ROOT/'data/calibration_fold0/cell_reservoir.npz')}
    folders={
        'fourier128':ROOT/'runs/CAL_GPU0_V1_fourier128_v1_f0_pg_s0',
        'selected_pure_incumbent':ROOT/'runs/CAL_GPU0_V1_ensemble75_v1_f0_pg_s0',
        'original_PG8':ROOT/'runs/CAL_GPU0_V1_latent8_v1_f0_pg_s0',
        'matched_resampling':ROOT/'artifacts/calibration_resampling_seed0_validation_replay',
    }
    records={}
    for label,folder in folders.items():
        rows=[]
        for rep in (0,1):
            path=folder/f'r{rep}/arrays.npz';metrics_path=path.with_name('metrics.json');metrics=read(metrics_path)
            assert sha(path)==metrics['arrays_SHA256']
            assert metrics['training_seed']==0 and metrics['generation_replicate']==rep
            for key in ('donors','calibration_manifest_SHA256','trace_manifest_SHA256','sampling_contract'):
                assert metrics[key]==proof['candidate'][key]
            if label=='matched_resampling':
                original=ROOT/'artifacts/calibration_resampling_seed0'/f'r{rep}/arrays.npz'
                with np.load(original,allow_pickle=False) as old,np.load(path,allow_pickle=False) as replay:
                    assert old.files==replay.files
                    for key in old.files:np.testing.assert_array_equal(old[key],replay[key])
                pins[str(original)]=sha(original)
            if label=='fourier128':assert proof['array_and_exemplar_SHA256'][str(path)]==sha(path)
            with np.load(path,allow_pickle=False) as arrays:
                generated=arrays['generated_cells_by_stage']
                row=decompose(generated,real)
            np.testing.assert_allclose(row['pooled_cell_energy_distance'],metrics['cell_energy_distance'],rtol=0,atol=1e-10)
            np.testing.assert_allclose(row['per_stage_cell_energy_distance'],metrics['per_stage_cell_energy_distance'],rtol=0,atol=1e-10)
            flat=generated.reshape(-1,50);ref=real.reshape(-1,50)
            row.update(pooled_PC_mean_error=(flat.mean(0)-ref.mean(0)).tolist(),
                       pooled_PC_std_ratio=(flat.std(0)/ref.std(0)).tolist())
            rows.append(row);pins[str(path)]=sha(path);pins[str(metrics_path)]=sha(metrics_path)
        mean={k:np.mean([r[k] for r in rows],axis=0).tolist() for k in rows[0]}
        records[label]={'replicates':rows,'mean':mean}
    for label,summary in [('fourier128',proof['candidate']),('selected_pure_incumbent',proof['incumbent'])]:
        np.testing.assert_allclose(records[label]['mean']['pooled_cell_energy_distance'],summary['cell_energy_distance'],rtol=0,atol=1e-10)
    differences={}
    candidate=records['fourier128']['mean']
    for label in ('selected_pure_incumbent','original_PG8','matched_resampling'):
        other=records[label]['mean']
        differences[label]={k:(np.asarray(candidate[k])-np.asarray(other[k])).tolist() for k in candidate}
    result={'status':'DESCRIPTIVE_AUDITED_FOURIER128_CELL_FAILURE_DECOMPOSITION',
            'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'training_seed':0,
            'records':records,'fourier128_minus_reference':differences,
            'input_SHA256':pins,'source_SHA256':{str(Path(__file__).resolve()):sha(__file__),str(ROOT/'analysis/diagnose_confirmation_cell_match.py'):sha(ROOT/'analysis/diagnose_confirmation_cell_match.py')},
            'metrics_or_selection_changed':False,'candidate_or_ensemble_created':False,
            'scope':'Same four adaptive calibration donors and existing two-replica saved reservoirs. Exact symmetric stage-kernel identity checked against each saved pooled and per-stage cell ED. PC moment diagnostics are descriptive and do not replace energy distance. No new sampling, fitting, evaluation donors, masks/weights, model correction or causal attribution.'}
    output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps({'status':result['status'],'fourier128_minus_reference':{
        label:{key:row[key] for key in ('pooled_cell_energy_distance','per_stage_cell_energy_distance','diagonal_contribution_to_pooled','off_diagonal_contribution_to_pooled','pooled_PC_centroid_error_L2')} for label,row in differences.items()}},indent=2))

if __name__=='__main__':main()
