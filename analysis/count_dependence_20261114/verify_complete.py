"""Audit retained TRAIN diagnostic evidence without generating model/data draws."""
from datetime import datetime, timezone
import json
import os
import joint_common as J
import diagnose_joint as D
import gamma_head as H

B=J.B; np=J.np


def main():
    J.torch.set_num_threads(3)
    os.sched_setaffinity(0, sorted(os.sched_getaffinity(0))[:3])
    report=B.P.read(J.ROOT/'report.json')
    proof=B.P.read(J.ROOT/'zero_replay_fidelity.json')
    sources=B.P.read(J.HERE/'source_manifest_prelaunch.json')
    assert len(sources)==207 and J.sources()==sources==report['sources']==proof['sources']
    assert report['status']=='completed_TRAIN_count_dependence_diagnostic_only'
    assert B.sha(J.HERE/'source_manifest_prelaunch.json')=='103cc98aea29ce16fc8d4bec3961bc0aac7b801413da7db37305e09681c53f69'
    assert B.sha(J.ROOT/'compact_moments.npz')==report['compact_moments_sha256']
    assert B.sha(J.ROOT/'zero_replay_fidelity.json')==report['zero_replay_fidelity_sha256']
    D.prerequisite()
    cache=B.W.RUNS/'caches/hvg2000/fold0'
    anchor=B.P.verify_cache(0,payloads=False)
    assert report['cache_manifest_sha256']==anchor['manifest_sha256']
    assert all(v==anchor['payload_sha256'][k] for k,v in report['TRAIN_payload_sha256'].items())
    permitted={str((cache/name).resolve()) for name in ('manifest.json','space.npz','train/counts.npz','train/metadata.npz')}
    assert set(report['cache_files_opened'])==permitted
    assert report['heldout_payloads_opened'] is False and report['default_changed'] is False
    default=J.HERE.parent/'improve_20261109/default_config.json'
    cli=J.HERE.parents[1]/'scripts/microglia.py'
    assert B.sha(default)=='fecbfce1e92d31a5d8c3d646f0a6dfb036c975b599dbdc880c332c1023bb3c23'
    assert B.sha(cli)=='40d6102e77fa6a4ece2c64cd431db3d907449d7bd69335cb597370d5f860b80e'
    assert proof['passed'] and proof['array_fingerprints']==report['baseline_array_fingerprints']
    assert proof['train_donors']==report['train_donors']==anchor['train_donors']
    assert proof['state_names']==B.A.STATES and (proof['fold'],proof['fitseed'],proof['M'],proof['K'])==(0,0,16,256)
    receipt=B.MODES/'current32/fold0/seed0.receipt.json'
    cal=B.P.read(receipt)
    assert B.sha(receipt)==proof['calibration_receipt_sha256']==report['calibration_receipt_sha256']
    assert proof['complete_delta']==cal['mode_extra']['biascal']['delta'] and len(proof['complete_delta'])==2000
    assert proof['complete_s']==cal['mode_extra']['zcal']['s'] and len(proof['complete_s'])==2000
    assert proof['pruned_archive_sha256']=={k:v for k,v in report['pruned_archives_sha256'].items() if k!='fixed_response'}
    assert not any(p.is_file() for p in J.SMOKE.rglob('*'))
    for label in ('baseline','response'):
        info=report[label+'_replay_info']
        assert info['targets']==25 and info['heldout_targets']==0 and not info['heldout_dir_present']
        assert info['M']==16 and info['K']==256 and info['generated_cells']==102400
        assert info['generation_device']=='cpu' and info['mode_extra']==cal['mode_extra']
        name='zero_control' if label=='baseline' else 'fixed_response'
        assert info['generated_SHA256']==report['pruned_archives_sha256'][name]
    with np.load(J.ROOT/'compact_moments.npz',allow_pickle=False) as z:
        c={k:z[k].copy() for k in z.files}
    assert all(np.isfinite(v).all() for v in c.values() if v.dtype.kind in 'fiu')
    assert c['train_donors'].tolist()==report['train_donors'] and c['genes'].shape==(2000,)
    assert c['real_draw_indices'].shape==(2,25,16,64)
    for donor in range(25):
        assert set(c['real_draw_indices'][0,donor].ravel()).isdisjoint(c['real_draw_indices'][1,donor].ravel())
        for bag in c['real_draw_indices'][:,donor].reshape(-1,64):assert len(set(bag.tolist()))==64
    with np.load(cache/'space.npz',allow_pickle=False) as z:
        loadings=z['W'].copy();sd=z['sd'].copy();pc_sd=z['pc_sd'].copy()
    assert np.array_equal(c['gene_groups'],J.groups(loadings))
    assert report['group_counts']==np.bincount(c['gene_groups'],minlength=100).tolist()
    basis,_=np.linalg.qr(np.random.default_rng(20261113).normal(size=(2000,50)))
    assert np.array_equal(c['random_basis'],basis)
    assert np.array_equal(c['projection'][:,:50],loadings.astype(np.float64)/pc_sd.astype(np.float64))
    assert np.array_equal(c['projection'][:,50:],basis/c['random_scale'])
    assert np.array_equal(c['raw_projection'],c['projection']/sd.astype(np.float64)[:,None])
    real={k[5:]:v for k,v in c.items() if k.startswith('real_') and k!='real_draw_indices'}
    moments={label:{k[len(label)+1:]:v for k,v in c.items() if k.startswith(label+'_') and '_signed_gene_error' not in k}
             for label in ('baseline','response')}
    def equal(a,b):
        def scalar(value):
            if isinstance(value,np.generic):return value.item()
            raise TypeError(type(value).__name__)
        return json.loads(json.dumps(a,default=scalar))==json.loads(json.dumps(b,default=scalar))
    decision=D.decide(moments['baseline'],moments['response'],real)
    assert equal(decision,report['decision']) and decision['passes_TRAIN_mechanism_only'] is False
    assert len(decision['gates'])==9 and sum(decision['gates'].values())==6
    for label in ('baseline','response'):
        assert int(c[label+'_total_generated_cells'])==102400 and int(c[label+'_total_pc_cells'])==25600
        assert c[label+'_covariance'].shape==(25,100,100) and c[label+'_sd_center'].shape==(25,100)
        for half in (0,1):
            for key in ('meanlog','detection','logvariance','raw_gene_mean','raw_gene_variance'):
                name=f'half{half}_{key}'
                vec=c[f'{label}_{name}_signed_gene_error'];summary=report['TRAIN_gene_errors'][label][name]
                assert float(np.sqrt(np.square(vec).mean()))==summary['rmse_gene']
                assert np.isclose(float(vec.mean()),summary['bias'],rtol=1e-14,atol=1e-14)
                if key.startswith('raw_'):
                    error=moments[label][key]-real[key][half]
                    assert np.array_equal(error.mean(0),vec)
                    assert equal(D.gene_error_summary(error),summary)
    assert c['point_mu'].shape==c['point_theta'].shape==(25,16,2000)
    assert (c['point_mu']>0).all() and (c['point_theta']>0).all()
    assert c['point_library'].shape==c['point_state'].shape==(25,16)
    assert np.array_equal(c['point_state'],np.asarray(proof['state_index']))
    assert report['conditional_points']==400 and report['coupling']==.5
    for i in range(25):
        for j in range(16):
            mu=c['point_mu'][i,j];theta=c['point_theta'][i,j]
            raw=H.projected_kernel(mu,theta,c['gene_groups'],c['raw_projection'])
            jac=H.logcp_jacobian(mu,c['projection'],sd)
            norm=H.projected_kernel(mu,theta,c['gene_groups'],jac)
            assert np.array_equal(raw,c['analytic_raw_projected_delta'][i,j])
            assert np.array_equal(norm,c['analytic_delta_method_logcp_projected_delta'][i,j])
    alignment=D.alignment_tables(moments['baseline'],moments['response'],real,c)
    assert equal(alignment,report['raw_and_normalized_alignment'])
    numerical=report['gamma_numerical'];assert set(numerical)==set(B.A.STATES)
    assert sum(v['shared_increment_draws'] for v in numerical.values())==204800000
    assert sum(v['independent_draws'] for v in numerical.values())==204800000
    assert sum(v['rate_error_bound_sum'] for v in numerical.values())==report['global_rate_error_bound_sum']<1e-8
    assert max(v['rate_error_bound_max'] for v in numerical.values())==report['global_rate_error_bound_max']<1e-12
    assert sum(v['final_cast_lost_rate_mass'] for v in numerical.values())==report['final_cast_lost_rate_mass']
    assert report['runtime']['compute_threads']==3 and report['runtime']['cpu_affinity']==[0,1,2]
    assert J.sources()==sources and J.disk_bytes()<3000000000
    evidence={'status':'passed_retained_evidence_audit','recorded_utc':datetime.now(timezone.utc).isoformat(),
        'report_sha256':B.sha(J.ROOT/'report.json'),'compact_sha256':report['compact_moments_sha256'],
        'source_manifest_sha256':B.sha(J.HERE/'source_manifest_prelaunch.json'),'execution_sources':207,
        'zero_proof_sha256':report['zero_replay_fidelity_sha256'],'exact_decision_recomputation':True,
        'exact_alignment_recomputation':True,'exact_400_raw_and_delta_kernels_recomputed':True,
        'signed_gene_rmse_and_bias_verified':True,'raw_gene_full_summaries_recomputed':True,
        'logcp_gene_donor_rmse_limit':'Full donor errors were pruned; only producer summaries retained for these three metrics',
        'passed_gates':6,'required_gates':9,'mechanism_passes':False,'all_model_archives_pruned':True,
        'model_draws_or_training_executed':False,'sealed_or_held_payloads_opened':False,
        'default_sha256':B.sha(default),'public_cli_sha256':B.sha(cli),'default_changed':False,
        'logical_task_bytes':J.disk_bytes()}
    B.write(J.HERE/'retained_evidence_audit_20261114.json',evidence)
    print(json.dumps(evidence),flush=True)


if __name__=='__main__':main()
