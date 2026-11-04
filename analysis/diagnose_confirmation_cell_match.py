"""Decompose an audited calibration cell ED into fixed stage error terms.

This reads existing samples only. No weights, masks, generation, donor selection,
model selection or evaluation-donor access. Off-diagonal terms measure alignment
of stage-wise distribution residuals in the energy-distance kernel.
"""
import argparse,datetime,hashlib,json,sys
from pathlib import Path
import numpy as np
from scipy.spatial.distance import cdist
ROOT=Path('/ml2/alz');sys.path[:0]=[str(ROOT),str(ROOT/'analysis')]
from calibration import targets

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path):return json.loads(Path(path).read_text())

def decompose(synthetic,real):
    assert synthetic.shape==real.shape==(4,375,50)
    assert np.isfinite(synthetic).all() and np.isfinite(real).all()
    # Symmetric energy kernel: diagonal is each conditional ED; the matrix
    # mean is exactly ED of the equal-stage pooled distributions.
    xy=np.array([[cdist(synthetic[s],real[t]).mean() for t in range(4)] for s in range(4)])
    xx=np.array([[cdist(synthetic[s],synthetic[t]).mean() for t in range(4)] for s in range(4)])
    yy=np.array([[cdist(real[s],real[t]).mean() for t in range(4)] for s in range(4)])
    kernel=xy+xy.T-xx-yy
    pooled=float(2*cdist(synthetic.reshape(-1,50),real.reshape(-1,50)).mean()-cdist(synthetic.reshape(-1,50),synthetic.reshape(-1,50)).mean()-cdist(real.reshape(-1,50),real.reshape(-1,50)).mean())
    np.testing.assert_allclose(kernel.mean(),pooled,rtol=0,atol=1e-10)
    np.testing.assert_allclose(kernel,kernel.T,rtol=0,atol=1e-10)
    assert np.linalg.eigvalsh(kernel).min()>-1e-9
    delta=synthetic.mean(1)-real.mean(1)
    return {'pooled_cell_energy_distance':pooled,'per_stage_cell_energy_distance':np.diag(kernel).tolist(),
            'stage_residual_energy_kernel':kernel.tolist(),
            'diagonal_contribution_to_pooled':float(np.trace(kernel)/16),
            'off_diagonal_contribution_to_pooled':float((kernel.sum()-np.trace(kernel))/16),
            'stage_PC_centroid_error_L2':np.linalg.norm(delta,axis=1).tolist(),
            'pooled_PC_centroid_error_L2':float(np.linalg.norm(delta.mean(0))),
            'stage_PC_centroid_error_dot_products':(delta@delta.T).tolist()}

def diagnose(proof_path,output):
    assert not output.exists(),'Preserve prior diagnostic evidence'
    proof=read(proof_path)
    assert proof['status']=='EXACT_LINEAGE_ENSEMBLE_SEED_CONFIRMATION_RECORDED_NO_MODEL_SELECTION'
    assert proof['independently_verified_saved_scores_counts_and_checkpoint'] and proof['owned_release_and_clean_settlement_verified']
    seed=proof['candidate']['training_seed'];assert seed in (1,2)
    manifest,_,cells=targets();real=cells['by_stage']
    assert np.array_equal(real.reshape(-1,50),cells['equal_stage'])
    pins={str(proof_path):sha(proof_path),str(ROOT/'data/calibration_fold0/cell_reservoir.npz'):sha(ROOT/'data/calibration_fold0/cell_reservoir.npz')}
    records={}
    for label,folder,summary in [('candidate',ROOT/'runs'/proof['attempt'],proof['candidate']),('baseline',ROOT/'artifacts'/f'calibration_resampling_seed{seed}',proof['baseline'])]:
        rows=[]
        for rep in (0,1):
            path=folder/f'r{rep}/arrays.npz';metrics_path=path.with_name('metrics.json');metrics=read(metrics_path)
            assert sha(path)==metrics['arrays_SHA256']
            assert metrics['training_seed']==seed and metrics['generation_replicate']==rep and metrics['calibration_manifest_SHA256']==summary['calibration_manifest_SHA256']
            assert metrics['trace_manifest_SHA256']==summary['trace_manifest_SHA256'] and metrics['sampling_contract']==summary['sampling_contract']
            if label=='candidate':assert proof['array_and_exemplar_SHA256'][str(path)]==sha(path)
            with np.load(path,allow_pickle=False) as arrays:row=decompose(arrays['generated_cells_by_stage'],real)
            np.testing.assert_allclose(row['pooled_cell_energy_distance'],metrics['cell_energy_distance'],rtol=0,atol=1e-10)
            np.testing.assert_allclose(row['per_stage_cell_energy_distance'],metrics['per_stage_cell_energy_distance'],rtol=0,atol=1e-10)
            rows.append(row);pins[str(path)]=sha(path);pins[str(metrics_path)]=sha(metrics_path)
        mean={key:np.mean([r[key] for r in rows],axis=0).tolist() for key in rows[0]}
        np.testing.assert_allclose(mean['pooled_cell_energy_distance'],summary['cell_energy_distance'],rtol=0,atol=1e-10)
        np.testing.assert_allclose(mean['per_stage_cell_energy_distance'],summary['per_stage_cell_energy_distance'],rtol=0,atol=1e-10)
        records[label]={'replicates':rows,'mean':mean}
    c=records['candidate']['mean'];b=records['baseline']['mean']
    result={'status':'VERIFIED_EXISTING_CALIBRATION_SAMPLES_CELL_ED_DECOMPOSITION','utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
            'training_seed':seed,'attempt':proof['attempt'],'records':records,
            'candidate_minus_baseline':{k:(np.asarray(c[k])-np.asarray(b[k])).tolist() for k in c},
            'all_four_conditional_cell_EDs_improve':bool(np.all(np.asarray(c['per_stage_cell_energy_distance'])<b['per_stage_cell_energy_distance'])),
            'pooled_cell_ED_improves':c['pooled_cell_energy_distance']<b['pooled_cell_energy_distance'],
            'input_SHA256':pins,'source_SHA256':{str(Path(__file__).resolve()):sha(__file__)},
            'scope':'Same four calibration donors and fixed saved reservoirs only. Symmetric residual energy kernel identity verified against exact pooled score in each replica. Off-diagonal terms may be negative; conditional improvement does not imply pooled improvement. No sample correction, new generation, evaluation access, seed ranking or change to frozen confirmation/selection.'}
    output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ('records','input_SHA256','source_SHA256')},indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--confirmation',required=True,type=Path);p.add_argument('--output',required=True,type=Path);args=p.parse_args();diagnose(args.confirmation,args.output)
