"""Qualify weighted math and all six clinical-reference allocation cases.

Clinical row identities/covariates only; no expression, output-PC or quality read.
"""
import datetime
import json
import os
import pathlib
import subprocess
import sys
ROOT=pathlib.Path('/ml2/alz');PROJECT=pathlib.Path('/ml2/codex-storage/future-workspaces/2026-10-27/task-2/benchmark')
SCIENCE=pathlib.Path('/ml2/codex-storage/future-workspaces/2026-11-03/task-2/alz-audit/core-comparison-20261103')
PG=PROJECT/'current_base_20261029/runtime/bin/python'
sys.path[:0]=[str(ROOT/'analysis'),str(PROJECT/'current_base_20261029/runtime/lib/python3.12/site-packages'),str(PROJECT/'.venv312/lib/python3.12/site-packages')]
import numpy as np
import score_benchmark097_donor_matched_supplement as m


def main():
    assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
    state=ROOT/'artifacts/benchmark097-donor-matched-supplement-CPU-preparation-v5-20261108'
    output=ROOT/'analysis/benchmark097_donor_matched_supplement_CPU_preparation_v5_20261108.json'
    assert not state.exists() and not output.exists();state.mkdir()
    tests=subprocess.run([str(PG),'-I','-S','-B',str(ROOT/'analysis/test_donor_matched_supplement.py')],cwd=ROOT,capture_output=True,text=True)
    (state/'tests.stdout').write_text(tests.stdout);(state/'tests.stderr').write_text(tests.stderr)
    assert tests.returncode==0 and 'Ran 14 tests' in tests.stderr and '\nOK\n' in tests.stderr
    readiness=m.readiness();assert readiness['missing_paths'] and not readiness['expression_or_quality_values_read']
    refused=subprocess.run([str(PG),'-I','-S','-B',str(ROOT/'analysis/score_benchmark097_donor_matched_supplement.py'),'--run'],cwd=ROOT,capture_output=True,text=True)
    (state/'premature.stdout').write_text(refused.stdout);(state/'premature.stderr').write_text(refused.stderr)
    assert refused.returncode!=0 and 'All45 models and both original reports required before numerical reads' in refused.stderr
    assert not(ROOT/'artifacts/benchmark097-donor-matched-cell-distance-20261108').exists()
    folds_path=SCIENCE.parent/'matched-benchmark-20261103/folds.json';folds=m.read(folds_path)
    assert m.sha(folds_path)=='763d081e387c2d00c6412f07c03af43babca06430a49d5e3f3b5d081c327ade3'
    p53=ROOT/'analysis/benchmark097_calibration_excluded53_plan.json';kept={f['fold']:f['retained_donors'] for f in m.read(p53)['folds']}
    inputs={str(p):m.sha(p) for p in (folds_path,p53)};records=[];cases=0
    for ci,cohort in enumerate(('original57','calibration_excluded53')):
        for f in folds['folds']:
            fold=f['fold'];cache=SCIENCE/'caches'/f'fold{fold}';manifest=m.read(cache/'manifest.json');path=cache/'evaluation/metadata.npz'
            assert m.sha(path)==manifest['files_SHA256']['evaluation/metadata.npz']
            with np.load(path,allow_pickle=False) as z:labels=z['donor'].copy();assert z['donors'].tolist()==f['evaluation_donors']
            donors=f['evaluation_donors'] if cohort=='original57' else kept[fold]
            reference=m.real_reference(labels,donors,folds['stage_by_fitting_donor'],ci,fold)
            expected=np.bincount([folds['stage_by_fitting_donor'][d] for d in donors],minlength=4)/len(donors)
            actual=np.bincount(reference['stages'],weights=reference['weights'],minlength=4)
            np.testing.assert_allclose(actual,expected,rtol=0,atol=1e-12)
            for seed in range(5):
                for rep in (0,1):
                    rows=m.generated_row_plan(reference,ci,fold,seed,rep)
                    assert rows.shape==(1500,) and (rows>=0).all() and (rows<80000).all()
                    for stage in range(4):assert len(np.unique(rows[reference['stages']==stage]))==int((reference['stages']==stage).sum())
                    cases+=1
            dest=state/f'{cohort}_f{fold}_reference_metadata.npz';np.savez_compressed(dest,**reference)
            records.append(dict(cohort=cohort,fold=fold,donors=len(donors),allocation=reference['allocation'].tolist(),replacement_donors=int(reference['replacement'].sum()),exact_real_and_generated_stage_mass=actual.tolist(),reference_metadata_path=str(dest),reference_metadata_SHA256=m.sha(dest)))
            for p in (cache/'manifest.json',path):inputs[str(p)]=m.sha(p)
    assert len(records)==6 and cases==60 and 'torch' not in sys.modules
    sources=[pathlib.Path(__file__).resolve(),ROOT/'analysis/score_benchmark097_donor_matched_supplement.py',ROOT/'analysis/test_donor_matched_supplement.py',ROOT/'analysis/after_benchmark097_donor_matched_supplement.py',ROOT/'analysis/after_released097_final_audits.py',ROOT/'analysis/released097_final_audits_observer_CPU_check_v5_20261108.json',ROOT/'analysis/scalarcal097_recovered_handoff.py',ROOT/'analysis/benchmark097_recovered_lineage.py',m.PLAN]
    proof=dict(status='DONOR_MATCHED_SUPPLEMENT_ESTIMATOR_SAMPLING_AND_OPAQUE_BARRIER_CPU_PASS',UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),actual_tests=14,clinical_reference_allocations=6,generated_row_plans_checked=60,clinical_cases=records,actual_readiness=readiness,premature_scoring_refused_before_expression_reads=True,source_SHA256={str(p):m.sha(p) for p in sources},input_SHA256=inputs,log_SHA256={str(p):m.sha(p) for p in state.iterdir() if p.suffix in ('.stdout','.stderr')},actual_supplementary_results=0,expression_counts_or_PCs_or_target_summaries_or_quality_results_decoded=False,clinical_and_row_metadata_only=True,no_GPU_allocation=True,no_native_source_or_model_change=True,full_goal_complete=False,SOTA_claim=False)
    output.write_text(json.dumps(proof,indent=2,allow_nan=False)+'\n');print(json.dumps({k:proof[k] for k in ('status','actual_tests','clinical_reference_allocations','generated_row_plans_checked','actual_supplementary_results')}))


if __name__=='__main__':main()
