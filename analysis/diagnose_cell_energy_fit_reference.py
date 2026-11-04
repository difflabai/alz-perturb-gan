"""Read-only fitting-population comparison of already-saved calibration generations."""
import datetime,json,os,sys
from pathlib import Path
ROOT=Path('/ml2/alz');sys.path[:0]=[str(ROOT/'model'),str(ROOT),str(ROOT/'analysis')]
import numpy as np
import torch
from variant_pg_adapter import Context
from variant_pg_update import configure
from diagnose_confirmation_cell_match import decompose,sha,read
from calibration import targets

def main():
    assert os.environ.get('CUDA_VISIBLE_DEVICES')=='';torch.set_num_threads(1)
    dest=ROOT/'analysis/cell_energy_fit_reference_diagnostic.json';assert not dest.exists()
    protocol_path=ROOT/'analysis/cell_energy_fit_reference_protocol.json';protocol=read(protocol_path)
    for path,pin in protocol['input_SHA256'].items():assert sha(path)==pin
    decomposition=read(ROOT/'analysis/cell_energy_cell_failure_decomposition.json')
    assert decomposition['metrics_or_selection_changed'] is False
    auth_path=ROOT/'artifacts/allocations/CAL_GPU0_V1_mog_prior_v1_f0_pg_s0/authorization.json';auth=read(auth_path)
    configure({'z_dim':8,'stage_balance':False,'cell_z_dim':8,'theta_init_scale':1.,'fourier_feature_scale':1.})
    ctx=Context(auth['cache_dir'],'cpu');assert len(ctx.ids)==33
    stage=np.rint(ctx.fitc[:,3]*3).astype(int);groups=[np.flatnonzero(stage==s) for s in range(4)]
    assert [len(x) for x in groups]==[3,4,8,18]
    real=[];indices=[];donor_indices=[]
    for s,group in enumerate(groups):
        rng=np.random.default_rng(np.random.SeedSequence([20261104,0,2026110405,s]))
        ds=rng.choice(group,size=375,replace=True);idx=np.asarray([rng.choice(ctx.by[int(d)]) for d in ds],dtype=np.int64)
        indices.append(idx);donor_indices.append(ds);real.append(ctx.trainpcs[idx].detach().cpu().numpy())
    fit_real=np.stack(real);assert fit_real.shape==(4,375,50)
    _,_,calib_cells=targets();calib_real=calib_cells['by_stage']
    artifact=ROOT/'artifacts/diagnostics/cell_energy_fit_reference_v1';artifact.mkdir(parents=True,exist_ok=False)
    reservoir=artifact/'fitting_reference.npz';np.savez_compressed(reservoir,PCs=fit_real,cell_indices=np.stack(indices),donor_indices=np.stack(donor_indices),fitting_donor_ids=ctx.ids)
    pins={str(protocol_path):sha(protocol_path),str(reservoir):sha(reservoir),str(Path(auth['cache_dir'])/'manifest.json'):sha(Path(auth['cache_dir'])/'manifest.json'),str(ROOT/'data/calibration_fold0/cell_reservoir.npz'):sha(ROOT/'data/calibration_fold0/cell_reservoir.npz')}
    models={'original_PG8':'CAL_GPU0_V1_latent8_v1_f0_pg_s0','cell_energy_task_extension':'CAL_GPU0_V1_cell_energy_aux_v1_f0_pg_s0','selected_pure_incumbent':'CAL_GPU0_V1_ensemble75_v1_f0_pg_s0'}
    records={}
    for name,attempt in models.items():
        rows=[]
        for rep in (0,1):
            path=ROOT/'runs'/attempt/f'r{rep}/arrays.npz';mp=path.with_name('metrics.json');m=read(mp)
            assert sha(path)==m['arrays_SHA256']==decomposition['input_SHA256'][str(path)]
            assert sha(mp)==decomposition['input_SHA256'][str(mp)]
            assert m['training_seed']==0 and m['generation_replicate']==rep
            with np.load(path,allow_pickle=False) as arrays:
                generated=arrays['generated_cells_by_stage'];calib=decompose(generated,calib_real);fit=decompose(generated,fit_real)
            np.testing.assert_allclose(calib['pooled_cell_energy_distance'],m['cell_energy_distance'],rtol=0,atol=1e-10)
            np.testing.assert_allclose(calib['per_stage_cell_energy_distance'],m['per_stage_cell_energy_distance'],rtol=0,atol=1e-10)
            rows.append({'calibration_reference':calib,'fitting_reference':fit});pins[str(path)]=sha(path);pins[str(mp)]=sha(mp)
        mean={reference:{key:np.mean([r[reference][key] for r in rows],axis=0).tolist() for key in rows[0][reference]} for reference in rows[0]}
        records[name]={'replicates':rows,'mean':mean}
    differences={reference:{key:(np.asarray(records['cell_energy_task_extension']['mean'][reference][key])-np.asarray(records['original_PG8']['mean'][reference][key])).tolist() for key in records['original_PG8']['mean'][reference]} for reference in ('calibration_reference','fitting_reference')}
    assert not torch.cuda.is_initialized()
    result={'status':'DESCRIPTIVE_FIXED_FITTING_REFERENCE_COMPARISON_OF_SAVED_CALIBRATION_GENERATIONS','utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'records':records,'cell_energy_task_extension_minus_PG8':differences,'fitting_reference_donor_count':33,'fitting_reference_points_per_stage':375,'PC_projection_device':'CPU float32 with original fixed training-only transform','input_SHA256':pins,'source_SHA256':{str(p):sha(p) for p in [Path(__file__).resolve(),ROOT/'model/variant_pg_adapter.py',ROOT/'analysis/diagnose_confirmation_cell_match.py']},'CUDA_initialized':False,'model_generation_or_training_performed':False,'selection_or_scorer_changed':False,'evaluation_donors_accessed':False,'scope':protocol['interpretation']+' Four original adaptive calibration donors; fitting reference samples33donors with replacement and is not an independent validation cohort. No causal attribution or new model selection.'}
    dest.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps({'status':result['status'],'differences':{r:{k:v for k,v in row.items() if k in ('pooled_cell_energy_distance','per_stage_cell_energy_distance','diagonal_contribution_to_pooled','off_diagonal_contribution_to_pooled')} for r,row in differences.items()}},indent=2))

if __name__=='__main__':main()
