"""Audit signed blocks independently via two-stage union distances.

Also reconstruct both real stage reservoirs and all40 fitting-only bootstrap
reservoir traces from cached PCs and row identities; no model/count generation.
"""
import datetime
import hashlib
import itertools
import json
import os
import statistics
import sys
from pathlib import Path

ROOT=Path('/ml2/alz')
PROJECT=Path('/ml2/codex-storage/future-workspaces/2026-10-27/task-2/benchmark')
SCIENCE=Path('/ml2/codex-storage/future-workspaces/2026-11-03/task-2/alz-audit/core-comparison-20261103')
sys.path[:0]=[str(ROOT/'analysis'),str(ROOT),
    str(PROJECT/'current_base_20261029/runtime/lib/python3.12/site-packages'),
    str(PROJECT/'.venv312/lib/python3.12/site-packages')]
import numpy as np
from audit_released097_completed_studies import independent_cell_ED
from calibration import targets


def read(p):return json.loads(Path(p).read_bytes())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def equal_multiset(a,b):
    ua,na=np.unique(a,axis=0,return_counts=True)
    ub,nb=np.unique(b,axis=0,return_counts=True)
    np.testing.assert_array_equal(ua,ub)
    np.testing.assert_array_equal(na,nb)


def main():
    assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
    result=ROOT/'analysis/t005_stage_mixture_energy_CPU_result_20261108.json'
    output=ROOT/'analysis/t005_stage_mixture_energy_CPU_root_audit_20261108.json'
    assert not output.exists()
    q=read(result)
    assert q['arrays_independently_decomposed']==40 and q['training_seeds']==list(range(5)) and q['saved_updates']==[5000,10000,20000]
    assert q['no_new_training_or_generation'] and not q['external_evaluation_target_values_read']
    for path,pin in {**q['source_SHA256'],**q['input_SHA256'],**q['figure_SHA256']}.items():assert sha(path)==pin,path
    manifest,ys,cells=targets()
    try:real=cells['by_stage'].copy();equal=cells['equal_stage'].copy()
    finally:ys.close();cells.close()
    assert np.array_equal(real.reshape(-1,50),equal)
    inputs={str(result):sha(result)}
    # Reconstruct the actual calibration stage mapping, not a convenient sorted
    # donor order. Stage3 has208 source cells and legitimately uses replacement.
    data=ROOT/'data/calibration_fold0'
    with np.load(data/'metadata.npz',allow_pickle=False) as z:
        donor_labels=z['donor'].copy();donors=z['donors'].copy();stages=z['stage'].copy()
        np.testing.assert_array_equal(stages,np.rint(z['raw_c'][:,3]*3).astype(int))
    pcs=np.load(data/'real_pcs.npy',allow_pickle=False)
    rng=np.random.default_rng(20261103)
    for stage in range(4):
        donor=donors[np.flatnonzero(stages==stage)[0]];pool=np.flatnonzero(donor_labels==donor)
        selected=rng.choice(pool,size=375,replace=len(pool)<375)
        np.testing.assert_array_equal(real[stage],pcs[selected])
        assert manifest['cell_reservoir_sampling'][stage]['donor']==donor
    for p in (data/'manifest.json',data/'metadata.npz',data/'real_pcs.npy',data/'cell_reservoir.npz'):inputs[str(p)]=sha(p)
    cache=Path(manifest['source_fold_cache']);cache_manifest=read(cache/'manifest.json')
    assert sha(cache/'manifest.json')==manifest['source_fold_cache_SHA256']
    with np.load(cache/'train/metadata.npz',allow_pickle=False) as z:
        fit_ids=z['donors'].copy();fit_labels=z['donor'].copy();fit_names=z['cell'].copy()
        fit_stages=np.rint(z['raw_c'][:,3]*3).astype(int)
    fit_pc=np.load(cache/'train/real_pcs.npy',allow_pickle=False)
    assert set(fit_ids.tolist())==set(cache_manifest['train_donors']) and not set(fit_ids.tolist())&set(donors.tolist())
    trace_manifest=read(SCIENCE/'traces/manifest.json')
    assert trace_manifest['fold_cache_manifest_SHA256']['0']==sha(cache/'manifest.json')
    for p in (cache/'manifest.json',cache/'train/metadata.npz',cache/'train/real_pcs.npy',SCIENCE/'traces/manifest.json'):inputs[str(p)]=sha(p)
    stage_distances_checked=union_blocks_checked=pooled_checked=baseline_reservoirs_checked=0
    summaries={};assignment_checks=0
    for row in q['array_records']:
        path=Path(row['arrays_path']);assert sha(path)==row['arrays_SHA256']
        with np.load(path,allow_pickle=False) as a:x=a['generated_cells_by_stage'].copy()
        d=np.array([[independent_cell_ED(x[s],real[t]) for t in range(4)] for s in range(4)])
        np.testing.assert_allclose(d,row['generated_stage_to_target_stage_ED'],rtol=0,atol=1e-10)
        stage_distances_checked+=16
        signed=np.diag(np.diag(d))
        for s in range(4):
            for t in range(s+1,4):
                union=independent_cell_ED(np.concatenate([x[s],x[t]]),np.concatenate([real[s],real[t]]))
                signed[s,t]=signed[t,s]=2*union-(d[s,s]+d[t,t])/2
                union_blocks_checked+=1
        np.testing.assert_allclose(signed,row['signed_stage_blocks'],rtol=0,atol=1e-10)
        pooled=independent_cell_ED(x.reshape(-1,50),equal)
        assert np.isclose(pooled,row['pooled_ED'],rtol=0,atol=1e-10)
        assert np.isclose(np.trace(d)/4,row['mean_within_stage_ED'],rtol=0,atol=1e-10)
        assert np.isclose(np.trace(signed)/16,row['pooled_diagonal_contribution'],rtol=0,atol=1e-10)
        assert np.isclose(pooled-np.trace(signed)/16,row['pooled_cross_stage_contribution'],rtol=0,atol=1e-10)
        assignments=[(float(sum(d[s,p[s]] for s in range(4))/4),p) for p in itertools.permutations(range(4))]
        best,permutation=min(assignments)
        assert np.isclose(best,row['lowest_assignment_diagnostic_ED'],rtol=0,atol=1e-10)
        assert list(permutation)==row['lowest_assignment_diagnostic_permutation']
        assert row['identity_assignment_is_minimum']==bool(np.isclose(best,np.trace(d)/4,rtol=0,atol=1e-10))
        assignment_checks+=1;pooled_checked+=1
        summaries.setdefault((row['seed'],row['checkpoint_or_control']),[]).append((pooled,float(np.trace(d)/4)))
        if row['checkpoint_or_control']=='resampling':
            seed,rep=row['seed'],row['replicate']
            for stage in range(4):
                trace=SCIENCE/'traces'/f'f0_s{seed}_r{rep}_stage{stage}.npz'
                assert sha(trace)==trace_manifest['files_SHA256'][trace.name]
                with np.load(trace,allow_pickle=False) as z:
                    np.testing.assert_array_equal(z['source_donors'],fit_ids)
                    np.testing.assert_array_equal(z['source_cells'],fit_names)
                    sampled_donors=z['donors'];rows=z['rows']
                    assert int(z['template_seed'])==20261104+1000*seed+100*rep+stage
                    assert np.all(fit_stages[sampled_donors]==stage)
                    np.testing.assert_array_equal(fit_labels[rows],np.broadcast_to(fit_ids[sampled_donors,None],rows.shape))
                    draws=fit_pc[rows].reshape(-1,50)
                # All80k priority keys reproduce the streaming reservoir's
                # selected multiset; arbitrary argpartition order is irrelevant.
                keys=np.random.default_rng(20261106+10*rep+stage).random(len(draws))
                selected=np.argpartition(keys,374)[:375]
                equal_multiset(draws[selected],x[stage])
                baseline_reservoirs_checked+=1;inputs[str(trace)]=sha(trace)
    assert (stage_distances_checked,union_blocks_checked,pooled_checked,baseline_reservoirs_checked,assignment_checks)==(640,240,40,40,40)
    means={key:tuple(statistics.mean(row[i] for row in values) for i in (0,1)) for key,values in summaries.items()}
    assert all(len(v)==2 for v in summaries.values()) and len(means)==20
    for step in ['5000','10000','20000']:
        for i,key in enumerate(['pooled_ED','mean_within_stage_ED']):
            assert np.isclose(statistics.mean(means[s,step][i] for s in range(5)),q['checkpoint_means'][step][key],rtol=0,atol=1e-10)
    for i,key in enumerate(['pooled_ED','mean_within_stage_ED']):
        assert np.isclose(statistics.mean(means[s,'resampling'][i] for s in range(5)),q['matched_resampling_mean'][key],rtol=0,atol=1e-10)
    assert 'torch' not in sys.modules
    record=dict(status='INDEPENDENT_UNION_DISTANCE_STAGE_BLOCKS_AND_ALL40_BASELINE_STAGE_RESERVOIRS_EXACT_AUDIT_PASS',
        UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),pooled_distances_checked=40,
        stage_to_stage_ED_values_independently_recomputed=640,two_stage_union_signed_blocks_independently_recomputed=240,
        four_calibration_real_reservoir_stage_labels_reconstructed=True,
        all40_resampling_stage_reservoirs_replayed_from_actual_fitting_row_traces=True,
        resampling_traces_stage_correct=True,calibration_real_reservoir_stage_order_correct=True,
        observed_best_assignment_is_not_evidence_of_label_code_reversal=True,
        assignment_enumerations_independently_checked=40,
        figure_visually_checked_after_correcting_colorbar_margin=True,
        initial_unpublished_cropped_render_retained='/ml2/alz/artifacts/t005-stage-mixture-render-v1-20261108',
        no_model_import_new_count_generation_or_training=True,no_GPU_allocation=True,
        frozen45_and_queued_LR_metric_unchanged=True,external_evaluation_target_values_read=False,
        source_SHA256={str(Path(__file__).resolve()):sha(__file__),str(ROOT/'analysis/audit_released097_completed_studies.py'):sha(ROOT/'analysis/audit_released097_completed_studies.py')},
        input_SHA256=inputs,full_goal_complete=False,SOTA_claim=False)
    output.write_text(json.dumps(record,indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:record[k] for k in ('status','stage_to_stage_ED_values_independently_recomputed',
        'two_stage_union_signed_blocks_independently_recomputed','resampling_traces_stage_correct','calibration_real_reservoir_stage_order_correct')}))


if __name__=='__main__':main()
