"""Predeclared descriptive donor-matched cell ED; original scores stay intact.

All45 logical models,47 retained attempts and both original reports are mandatory
before numerical target/quality reads. This script imports no model or Torch.
"""
import argparse
import datetime
import hashlib
import json
import math
import os
import statistics
import sys
from pathlib import Path

ROOT=Path('/ml2/alz')
PROJECT=Path('/ml2/codex-storage/future-workspaces/2026-10-27/task-2/benchmark')
SCIENCE=Path('/ml2/codex-storage/future-workspaces/2026-11-03/task-2/alz-audit/core-comparison-20261103')
PLAN=ROOT/'analysis/benchmark097_donor_matched_cell_distance_supplement_plan_20261108.json'
METHODS=('pg','scvi','cfgen','hierarchical_resampling')
sys.path.insert(0,str(ROOT/'analysis'))


def read(p):return json.loads(Path(p).read_bytes())
def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def required_paths(root=ROOT):
    from scalarcal097_recovered_handoff import paths
    return paths(root)


def readiness(root=ROOT):
    missing=[str(p) for p in required_paths(root) if not p.is_file()]
    return dict(status='WAITING_FOR_ALL45_AND_BOTH_ORIGINAL_REPORTS' if missing else 'PATHS_PRESENT_HASH_AND_LINEAGE_GATE_STILL_REQUIRED',
        missing_paths=missing,expression_or_quality_values_read=False,actual_supplementary_results=0,
        full_goal_complete=False,SOTA_claim=False)


def numerical_sites():
    for p in (PROJECT/'current_base_20261029/runtime/lib/python3.12/site-packages',PROJECT/'.venv312/lib/python3.12/site-packages'):
        if str(p) not in sys.path:sys.path.append(str(p))


def weighted_ED(x,y,a,b):
    import numpy as np
    from scipy.spatial.distance import cdist
    x,y,a,b=map(np.asarray,(x,y,a,b))
    assert x.ndim==y.ndim==2 and x.shape[1]==y.shape[1] and len(x)>0 and len(y)>0
    assert a.shape==(len(x),) and b.shape==(len(y),)
    assert all(np.isfinite(z).all() for z in (x,y,a,b)) and (a>=0).all() and (b>=0).all()
    assert np.isclose(a.sum(),1,rtol=0,atol=1e-12) and np.isclose(b.sum(),1,rtol=0,atol=1e-12)
    a,b=a.astype(np.float64),b.astype(np.float64)
    return float(2*a.dot(cdist(x,y).dot(b))-a.dot(cdist(x,x).dot(a))-b.dot(cdist(y,y).dot(b)))


def real_reference(labels,donors,stage_map,cohort,fold,*,size=1500):
    import numpy as np
    labels=np.asarray(labels);donors=list(donors);d=len(donors)
    assert labels.ndim==1 and d>0 and d<=size and len(set(donors))==d
    assert set(donors)<=set(labels.tolist()) and all(stage_map[x] in range(4) for x in donors)
    allocation=np.full(d,size//d,dtype=np.int64)
    extra=np.random.default_rng(np.random.SeedSequence([20261108,97,271,cohort,fold,0,0])).permutation(d)[:size%d]
    allocation[extra]+=1
    indices=[];weights=[];donor_labels=[];stage_labels=[];replacement=[]
    for pos,(donor,n) in enumerate(zip(donors,allocation)):
        pool=np.flatnonzero(labels==donor);assert len(pool)>0
        rng=np.random.default_rng(np.random.SeedSequence([20261108,97,271,cohort,fold,1,pos]))
        replace=len(pool)<n;indices.extend(rng.choice(pool,int(n),replace=replace).tolist())
        weights.extend([1/(d*int(n))]*int(n));donor_labels.extend([donor]*int(n));stage_labels.extend([stage_map[donor]]*int(n))
        replacement.append(bool(replace))
    result=dict(indices=np.asarray(indices,dtype=np.int64),weights=np.asarray(weights,dtype=np.float64),
        donors=np.asarray(donors),allocation=allocation,labels=np.asarray(donor_labels),
        stages=np.asarray(stage_labels,dtype=np.int64),replacement=np.asarray(replacement,dtype=bool))
    assert len(result['indices'])==size and allocation.max()-allocation.min()<=1
    np.testing.assert_array_equal(labels[result['indices']],result['labels'])
    for donor in donors:assert np.isclose(result['weights'][result['labels']==donor].sum(),1/d,rtol=0,atol=1e-12)
    return result


def generated_row_plan(reference,cohort,fold,seed,rep,*,pool_size=80000):
    import numpy as np
    assert seed in range(5) and rep in (0,1)
    stages=reference['stages'];plan=np.empty(len(stages),dtype=np.int64)
    for stage in range(4):
        slots=np.flatnonzero(stages==stage)
        assert len(slots)<=pool_size
        rng=np.random.default_rng(np.random.SeedSequence([20261108,97,272,cohort,fold,seed,rep,stage]))
        chosen=rng.choice(pool_size,len(slots),replace=False);plan[slots]=chosen
        assert len(set(chosen.tolist()))==len(slots)
    return plan


def select_generated(reference,rows,loader):
    import numpy as np
    assert rows.shape==reference['indices'].shape
    result=np.empty((len(rows),50),dtype=np.float32)
    for stage in range(4):
        slots=np.flatnonzero(reference['stages']==stage)
        if len(slots):
            values=loader(stage,rows[slots])
            assert values.shape==(len(slots),50) and values.dtype==np.float32
            result[slots]=values
    assert np.isfinite(result).all()
    return result


def summarize(points):
    assert len(points)==120 and {(p['method'],p['fold'],p['seed'],p['replicate']) for p in points}=={
        (m,f,s,r) for m in METHODS for f in range(3) for s in range(5) for r in (0,1)}
    assert all(math.isfinite(p['donor_matched_cell_energy_distance']) for p in points)
    result={}
    for method in METHODS:
        rows=[dict(fold=f,seed=s,donor_matched_cell_energy_distance=statistics.mean(
            p['donor_matched_cell_energy_distance'] for p in points if (p['method'],p['fold'],p['seed'])==(method,f,s))) for f in range(3) for s in range(5)]
        result[method]=dict(donor_matched_cell_energy_distance=statistics.mean(x['donor_matched_cell_energy_distance'] for x in rows),fold_seed_rows=rows)
    return result


def run(root=ROOT):
    root=Path(root)
    assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
    ready=readiness(root);assert not ready['missing_paths'],'All45 models and both original reports required before numerical reads'
    from scalarcal097_recovered_handoff import ready as completed_barrier
    from benchmark097_recovered_lineage import build_matrix,metadata
    pins=completed_barrier(root)
    assert root==ROOT
    matrix=read(root/'analysis/benchmark097/matrix_complete.json')
    assert matrix==build_matrix(),'Actual47-attempt closure and native audits must agree with the full45 matrix'
    plan=read(PLAN);assert plan['status']=='PREDECLARED_DESCRIPTIVE_DONOR_MATCHED_CELL_ED_SUPPLEMENT_PRIMARY_UNCHANGED'
    assert plan['primary_and_53_reports_preserved'] and plan['selection_and_claims']['descriptive_only']
    assert plan['cohorts']==['original57','calibration_excluded53'] and plan['methods']==list(METHODS)
    assert plan['folds']==list(range(3)) and plan['training_seeds']==list(range(5)) and plan['generation_replicates']==[0,1]
    assert plan['real_reference']['cells_per_fold_cohort']==plan['generated_reference']['cells_per_model_replicate']==1500
    assert plan['real_reference']['seed_namespace']==[20261108,97,271] and plan['generated_reference']['seed_namespace']==[20261108,97,272]
    numerical_sites()
    import numpy as np
    out=root/'artifacts/benchmark097-donor-matched-cell-distance-20261108'
    report_path=root/'analysis/benchmark097_donor_matched_cell_distance_supplement_20261108.json'
    assert not out.exists() and not report_path.exists()
    fold_path=SCIENCE.parent/'matched-benchmark-20261103/folds.json';folds=read(fold_path)
    assert sha(fold_path)=='763d081e387c2d00c6412f07c03af43babca06430a49d5e3f3b5d081c327ade3'
    p53=read(root/'analysis/benchmark097_calibration_excluded53_plan.json')
    calibrators=set(read(root/'data/calibration_fold0/manifest.json')['donors']);stage_map=folds['stage_by_fitting_donor']
    assert len(calibrators)==4 and set(p53['current_global_calibration_donors'])==calibrators
    kept={f['fold']:f['retained_donors'] for f in p53['folds']}
    for f in folds['folds']:assert kept[f['fold']]==[d for d in f['evaluation_donors'] if d not in calibrators]
    assert sum(len(f['evaluation_donors']) for f in folds['folds'])==57 and sum(map(len,kept.values()))==53
    audit_rows={};sources={};out.mkdir()
    for f in range(3):
        for seed in range(5):
            for method in METHODS[:3]:
                key=matrix['logical_to_physical_attempt'][f'f{f}_{method}_s{seed}'];meta=metadata(key,complete=True)
                assert meta['result_SHA256']==matrix['result_SHA256'][key]
                q=read(meta['result_path']);assert q['complete'] and q['independent_CPU_saved_score_audit'] and q['native_saved_state_matches_GPU_restore']
                for p,pin in q['files_SHA256'].items():assert sha(p)==pin,p
                audit_rows[f,method,seed]=q;sources[f,method,seed]=root/'runs'/key
                pins[meta['result_path']]=meta['result_SHA256']
    traces=read(SCIENCE/'traces/manifest.json')
    pins.update({str(p):sha(p) for p in (PLAN,fold_path,root/'analysis/benchmark097_calibration_excluded53_plan.json',SCIENCE/'traces/manifest.json')})
    derived={};cohort_reports={};reference_records=[]
    for ci,cohort in enumerate(plan['cohorts']):
        assert cohort in ('original57','calibration_excluded53')
        points=[]
        for f in folds['folds']:
            fold=f['fold'];cache=SCIENCE/'caches'/f'fold{fold}';manifest=read(cache/'manifest.json')
            assert manifest['evaluation_donors']==f['evaluation_donors'] and manifest['train_donors']==f['optimization_donors']
            assert manifest['folds_SHA256']==sha(fold_path) and traces['fold_cache_manifest_SHA256'][str(fold)]==sha(cache/'manifest.json')
            pins[str(cache/'manifest.json')]=sha(cache/'manifest.json')
            for name in ('evaluation/metadata.npz','evaluation/real_pcs.npy','train/metadata.npz','train/real_pcs.npy'):
                path=cache/name;assert sha(path)==manifest['files_SHA256'][name];pins[str(path)]=sha(path)
            with np.load(cache/'evaluation/metadata.npz',allow_pickle=False) as m:
                labels=m['donor'].copy();assert m['donors'].tolist()==f['evaluation_donors']
            with np.load(cache/'train/metadata.npz',allow_pickle=False) as m:
                train_ids=m['donors'].copy();train_names=m['cell'].copy()
            donors=f['evaluation_donors'] if cohort=='original57' else kept[fold]
            reference=real_reference(labels,donors,stage_map,ci,fold)
            real_pc=np.load(cache/'evaluation/real_pcs.npy',mmap_mode='r',allow_pickle=False)
            train_pc=np.load(cache/'train/real_pcs.npy',mmap_mode='r',allow_pickle=False)
            assert real_pc.shape==(len(labels),50) and real_pc.dtype==train_pc.dtype==np.float32
            real=np.asarray(real_pc[reference['indices']]);assert np.isfinite(real).all()
            dest=out/f'{cohort}_f{fold}_reference.npz';np.savez_compressed(dest,real_cells=real,**reference);derived[str(dest)]=sha(dest)
            reference_records.append(dict(cohort=cohort,fold=fold,donors=list(donors),allocation=reference['allocation'].tolist(),replacement=reference['replacement'].tolist(),path=str(dest),SHA256=sha(dest)))
            for seed in range(5):
                for rep in (0,1):
                    selected=generated_row_plan(reference,ci,fold,seed,rep)
                    original_traces={}
                    for stage in range(4):
                        path=SCIENCE/'traces'/f'f{fold}_s{seed}_r{rep}_stage{stage}.npz';assert sha(path)==traces['files_SHA256'][path.name]
                        with np.load(path,allow_pickle=False) as t:
                            np.testing.assert_array_equal(t['source_donors'],train_ids);np.testing.assert_array_equal(t['source_cells'],train_names)
                            assert int(t['template_seed'])==20261104+100000*fold+1000*seed+100*rep+stage
                            original_traces[stage]=t['rows'].reshape(-1).copy()
                        assert len(original_traces[stage])==80000;pins[str(path)]=sha(path)
                    for method in METHODS:
                        pools={}
                        if method=='hierarchical_resampling':
                            def loader(stage,index):return np.asarray(train_pc[original_traces[stage][index]])
                            attempt=None
                        else:
                            q=audit_rows[fold,method,seed];attempt=q['attempt']
                            for stage in range(4):
                                path=sources[fold,method,seed]/f'evaluation_r{rep}/stage{stage}_all_generated_PCs.npy'
                                assert sha(path)==q['files_SHA256'][str(path)];pins[str(path)]=sha(path)
                                full=np.load(path,mmap_mode='r',allow_pickle=False);assert full.shape==(200,400,50) and full.dtype==np.float32;pools[stage]=full.reshape(-1,50)
                            def loader(stage,index):return np.asarray(pools[stage][index])
                        generated=select_generated(reference,selected,loader)
                        value=weighted_ED(generated,real,reference['weights'],reference['weights'])
                        dest=out/f'{cohort}_f{fold}_{method}_s{seed}_r{rep}.npz'
                        np.savez_compressed(dest,generated_cells=generated,source_stage=reference['stages'],source_pool_row=selected,reference_path=np.asarray(str(out/f'{cohort}_f{fold}_reference.npz')))
                        derived[str(dest)]=sha(dest)
                        points.append(dict(method=method,fold=fold,seed=seed,replicate=rep,donor_matched_cell_energy_distance=value,source_attempt=attempt,generated_path=str(dest),generated_SHA256=sha(dest)))
            del real_pc,train_pc
        methods=summarize(points)
        cohort_reports[cohort]=dict(methods=methods,ranking_by_supplementary_distance=sorted(METHODS,key=lambda m:methods[m]['donor_matched_cell_energy_distance']),all120_method_seed_replicate_points=points,descriptive_only=True)
    for p,pin in pins.items():assert sha(p)==pin,'An input changed during read-only supplementary scoring: '+p
    assert 'torch' not in sys.modules
    result=dict(status='COMPLETE_DESCRIPTIVE_DONOR_MATCHED_CELL_DISTANCE_SUPPLEMENT_SAME45_MODELS',UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),cohorts=cohort_reports,real_reference_records=reference_records,complete_frozen_models=45,physical_attempts=47,scored_method_seed_replicate_points=240,original57_primary_preserved=True,prespecified53_report_preserved=True,weighted_equal_donor_measures_within_fold=True,equal_fold_summary=True,statistical_significance_or_SOTA_not_inferred=True,new_training_or_count_generation=0,source_SHA256={str(Path(__file__).resolve()):sha(__file__),str(ROOT/'analysis/scalarcal097_recovered_handoff.py'):sha(ROOT/'analysis/scalarcal097_recovered_handoff.py'),str(ROOT/'analysis/benchmark097_recovered_lineage.py'):sha(ROOT/'analysis/benchmark097_recovered_lineage.py')},input_SHA256=pins,derived_SHA256=derived,limitations='Descriptive secondary target-law analysis, not a replacement primary, independent biological confirmation, or a prospective externally validated model selection. Generated donor blocks allocate weights to stage-only draws; they are not additional biological donors.',full_goal_complete=False,SOTA_claim=False)
    report_path.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n');return result


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--check-readiness',action='store_true');parser.add_argument('--run',action='store_true');args=parser.parse_args()
    if args.check_readiness:print(json.dumps(readiness(),indent=2));return
    assert args.run,'Explicit --run required'
    result=run();print(json.dumps(dict(status=result['status'],cohort_rankings={k:v['ranking_by_supplementary_distance'] for k,v in result['cohorts'].items()},SOTA_claim=False)))


if __name__=='__main__':main()
