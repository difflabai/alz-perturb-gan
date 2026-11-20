"""Recompute coverage/calibration from pinned cells and archives without fitting models."""
import os
os.environ.update(OMP_NUM_THREADS='3',OPENBLAS_NUM_THREADS='3',MKL_NUM_THREADS='3',CUDA_VISIBLE_DEVICES='')
import json
from pathlib import Path
import run as W


def main():
    R,T,G,cache,fit,anchor,digest,provenance = W.setup()
    import architecture_score as A
    import numpy as np
    real = A.E.load_real(cache)
    comparison = json.loads((W.OUT/'comparison.json').read_text())
    assert comparison['experiment_manifest_sha256']==digest
    assert comparison['application_manifest_sha256']==W.sha(R.SOURCE_MANIFEST)
    observed = []
    for repeat in range(A.E.DEFAULT_PC_REPEATS):
        bags = []
        for donor,pc in zip(real.donors,real.pcs):
            indices=A.E.MX.det_subsample(len(pc),64,A.E.MX.stable_seed('pc-summary-rep',donor,repeat))
            bag=pc[indices].astype(np.float64)
            bags.append(np.r_[bag.mean(0),bag.std(0)])
        observed.append(bags)
    observed=np.asarray(observed)
    values=0;artifacts=0
    def verify_metrics(lo,hi,mask,expected):
        nonlocal values
        for name,sl in [('mean',slice(0,50)),('sd',slice(50,100)),('combined',slice(0,100))]:
            l,u,y=lo[mask,sl],hi[mask,sl],observed[:,mask,sl]
            width=u-l
            got=dict(coverage=float(np.mean((y>=l)&(y<=u))),
                below=float(np.mean(y<l)),above=float(np.mean(y>u)),width=float(np.mean(width)),
                interval_score=float(np.mean(width+10*np.maximum(l-y,0)+10*np.maximum(y-u,0))))
            for key,value in got.items():
                assert abs(value-expected[name][key])<1e-12,(name,key,value,expected[name][key])
                values+=1
    for label,path in [('baseline',W.BASE),('quarter',W.OUT/'quarter_generated.npz')]:
        row=comparison['rows'][label]
        assert W.sha(path)==row['generated_sha256']
        with np.load(path,allow_pickle=False) as f:
            target=f['targets'].astype(str);pc=f['pcs'].astype(np.float64)
        pos={d:i for i,d in enumerate(target)}
        pc=pc[[pos[d] for d in real.donors]]
        summaries=np.concatenate([pc.mean(-2),pc.std(-2)],-1)
        low,high=np.quantile(summaries,[.1,.9],axis=1)
        lo,hi=low.copy(),high.copy()
        for swap in row['swaps']:
            artifact=W.ROOT/swap['artifact'];assert W.sha(artifact)==swap['artifact_sha256']
            record=json.loads(artifact.read_text());artifacts+=1
            cal=np.array([real.index[d] for d in swap['calibration_donors']])
            test=np.array([real.index[d] for d in swap['evaluation_donors']])
            assert len(cal)==len(test)==record['n_calibration_donors']==record['rank']==6
            assert not set(cal)&set(test)
            assert real.is_heldout[cal].all() and real.is_heldout[test].all()
            assert set(record['training_donors'])==set(real.donors[~real.is_heldout])
            discrepancy=np.maximum.reduce([np.zeros_like(low[cal,50:]),
                low[cal,50:]-observed[0,cal,50:],observed[0,cal,50:]-high[cal,50:]])
            correction=np.max(discrepancy,axis=0)
            assert np.array_equal(correction,np.asarray(record['sd_correction']))
            assert record['binding']['final_checkpoint_sha256']==W.sha(fit/'final.pt')
            assert record['binding']['source_generated_sha256']==W.sha(path)
            assert record['binding']['experiment_manifest_sha256']==digest
            lo[test,50:]=np.maximum(0,low[test,50:]-correction)
            hi[test,50:]=high[test,50:]+correction
            mask=np.isin(np.arange(len(real.donors)),test)
            verify_metrics(low,high,mask,swap['raw'])
            verify_metrics(lo,hi,mask,swap['calibrated'])
        assert np.array_equal(lo[:,:50],low[:,:50]) and np.array_equal(hi[:,:50],high[:,:50])
        with np.load(W.OUT/f'{label}_crossfitted_intervals.npz',allow_pickle=False) as f:
            for name,array in [('raw_low',low),('raw_high',high),('calibrated_low',lo),('calibrated_high',hi)]:
                assert np.array_equal(f[name],array),name
        verify_metrics(low,high,real.is_heldout,row['raw_heldout'])
        verify_metrics(lo,hi,real.is_heldout,row['calibrated_heldout'])
        verify_metrics(low,high,~real.is_heldout,row['raw_train'])
    frozen=json.loads((W.ROOT/'analysis/empf_comparison_20261117/completion/results/native.json').read_text())
    assert frozen['historical']==comparison['rows']['baseline']['historical']
    base,candidate=comparison['rows']['baseline'],comparison['rows']['quarter']
    gates={}
    for group in ('sd','combined'):
        b,c=base['raw_heldout'][group],candidate['raw_heldout'][group]
        gates[group+'_coverage']=abs(c['coverage']-.8)<abs(b['coverage']-.8)
        gates[group+'_interval_score']=c['interval_score']<b['interval_score']
    for key,limit in [('pb_rmse',.005),('meanlog_rmse_gene',.02),('nonzero_rmse_gene',.02),
                      ('logvar_rmse_gene',.02),('pc_energy_rep',.02),('cell_energy_rep',.02)]:
        gates[key]=candidate['historical'][key]<=base['historical'][key]*(1+limit)
    assert gates==comparison['candidate_gates'] and all(gates.values())==comparison['candidate_advances']
    for row in comparison['rows'].values():
        useful=all(abs(row['calibrated_heldout'][g]['coverage']-.8)<abs(row['raw_heldout'][g]['coverage']-.8)
                   and row['calibrated_heldout'][g]['interval_score']<row['raw_heldout'][g]['interval_score']
                   for g in ('sd','combined'))
        assert useful==row['calibration_useful_on_both_primary_groups']
    W.finish(R,cache,anchor,digest)
    size=sum(p.stat().st_size for p in W.OUT.rglob('*') if p.is_file())
    assert size<500*1024**2
    report=dict(status='passed_direct_array_and_cell_summary_recomputation',
        auditor_source_sha256=W.sha(__file__),experiment_manifest_sha256=digest,
        comparison_sha256=W.sha(W.OUT/'comparison.json'),numeric_values_recomputed=values,
        decision_checks_recomputed=len(gates)+3,
        calibration_artifacts_verified=artifacts,calibration_donors_per_artifact=6,
        mean_bounds_exactly_unchanged=True,baseline_historical_scores_exactly_reproduced=True,
        cache_anchor=anchor,output_bytes=size,model_updates=0,cuda_initialized=T.torch.cuda.is_initialized(),
        scope='Same-agent numerical verification with direct formulas; not an independent-agent or external-cohort review.')
    assert report['cuda_initialized'] is False
    W.write(W.OUT/'audit.json',report)
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
