"""Verify actual parent controls after clean native completion; never select."""
import argparse, json, sys
from pathlib import Path
import numpy as np
ROOT=Path('/ml2/alz');sys.path.insert(0,str(ROOT))
from ensemble_scope import manifest,read,sha,verify_reference
from calibration import score,average

def verify(attempt):
    dep=read(ROOT/'artifacts/allocations'/attempt/'deployment.json');auth=read(dep['authorization_path'])
    out=Path(dep['output_dir']);ledger=Path(dep['ledger_path']);settlement=read(ledger.with_name(attempt+'-settlement.json'));job=read(ledger)['jobs'][0]
    assert dep['qualification_only'] is True and auth['operation']=='qualify_pure_ensemble'
    assert dep['authorization_SHA256']==sha(dep['authorization_path'])
    assert settlement['uncertain'] is None and settlement['settlement']['status']=='settled' and job['reservation_released']
    assert job['status']=='completed' and job['exit_code']==0
    q=read(out/'completed.json');weight=auth['ensemble_weight'];assert weight in (0.,1.)
    assert q['qualification_only'] is True and q['pure_only'] and q['real_bag_replacements']==q['new_model_updates']==0
    assert q['completed_updates']==q['fixed_endpoint']==40000 and q['ensemble_config']['learned_weight']==weight
    assert q['parent_manifest_SHA256']==sha(ROOT/'ensemble_parents.json')
    verify_reference(q);parent=manifest()['parents'][int(weight)];old_root=Path(parent['score_output']);records=[]
    for rep in (0,1):
        folder=out/f'r{rep}';metrics=read(folder/'metrics.json');assert metrics['arrays_SHA256']==sha(folder/'arrays.npz')
        with np.load(folder/'arrays.npz',allow_pickle=False) as a,np.load(old_root/f'r{rep}/arrays.npz',allow_pickle=False) as b:
            assert set(a.files)==set(b.files)
            for key in a.files:
                assert a[key].dtype==b[key].dtype and a[key].tobytes()==b[key].tobytes()
            recalculated=score(a['stage_synthetic_summaries'],a['generated_cells_by_stage'])
        for key in ['energy_score','cell_energy_distance','spread_ratio','per_donor_energy_score','per_stage_cell_energy_distance','coverage80']:
            np.testing.assert_allclose(metrics[key],recalculated[key],rtol=0,atol=1e-10)
        for stage in range(4):
            with np.load(folder/f'stage{stage}_exemplar.npz',allow_pickle=False) as a,np.load(old_root/f'r{rep}/stage{stage}_exemplar.npz',allow_pickle=False) as b:
                x=a['counts'];assert x.shape==(1,400,2000) and np.isfinite(x).all() and (x>=0).all() and np.equal(x,np.floor(x)).all()
                assert x.dtype==b['counts'].dtype and x.tobytes()==b['counts'].tobytes()
        records.append(metrics)
    avg=average(records)
    for key in ['energy_score','cell_energy_distance','spread_ratio','coverage80']:np.testing.assert_allclose(avg[key],q[key],rtol=0,atol=1e-10)
    label='pg8' if weight==0 else 'learned';dest=ROOT/'analysis'/f'ensemble_control_{label}_verification.json';assert not dest.exists()
    result={'status':'VERIFIED_NATIVE_PURE_ENSEMBLE_PARENT_CONTROL','attempt':attempt,'control_weight':weight,
            'parent_manifest_SHA256':sha(ROOT/'ensemble_parents.json'),'cpu_qualification_SHA256':sha(ROOT/'analysis/ensemble_cpu_qualification.json'),
            'complete_clean_settlement_verified':True,'arrays_and_integer_exemplars_bitwise_equal':True,
            'independent_saved_scores_verified':True,'new_model_updates':0,'candidate_selection_performed':False,
            'qualified_execution_source_SHA256':auth['research_source_SHA256'],'charged_GPU_seconds':settlement['settlement']['charged_GPU_seconds'],
            'authorization_SHA256':sha(dep['authorization_path']),'verification_source_SHA256':{str(Path(__file__).resolve()):sha(Path(__file__).resolve())},
            'scope':'Actual guarded GPU0 parent control only. Complete branch replay, integer exemplars and independently recomputed scores; no new model training or candidate ranking.'}
    dest.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k!='qualified_execution_source_SHA256'},indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--attempt',required=True);a=p.parse_args();verify(a.attempt)
