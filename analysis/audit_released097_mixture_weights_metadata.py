"""Reconstruct frozen mixture weights using clinical/row metadata only.

Expression counts, projected expression, summary targets, model outputs and
quality-result JSONs are never decoded. No metric or current study is changed.
"""
import ast
import datetime
import hashlib
import json
import os
import sys
from pathlib import Path

ROOT=Path('/ml2/alz')
PROJECT=Path('/ml2/codex-storage/future-workspaces/2026-10-27/task-2/benchmark')
SCIENCE=Path('/ml2/codex-storage/future-workspaces/2026-11-03/task-2/alz-audit/core-comparison-20261103')
sys.path[:0]=[str(PROJECT/'current_base_20261029/runtime/lib/python3.12/site-packages'),
              str(PROJECT/'.venv312/lib/python3.12/site-packages')]
import numpy as np


def read(p):return json.loads(Path(p).read_bytes())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
    output=ROOT/'analysis/released097_mixture_weights_metadata_audit_20261108.json'
    assert not output.exists()
    scoring=SCIENCE/'scoring.py';evaluation=PROJECT/'evaluation.py'
    auth_path=ROOT/'artifacts/allocations/CAL_GPU0_V1_external097_f2_cfgen_s2/authorization.json'
    auth=read(auth_path)
    assert sha(scoring)==auth['files_SHA256'][str(scoring)]
    assert sha(evaluation)==auth['files_SHA256'][str(evaluation)]
    # Execute only the unchanged source class on anonymous stage-code columns,
    # without importing evaluation.py or any Torch/model dependency.
    source=evaluation.read_text();tree=ast.parse(source)
    node=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='Reservoir')
    namespace={'np':np}
    exec(compile(ast.Module(body=[node],type_ignores=[]),str(evaluation),'exec'),namespace)
    Reservoir=namespace['Reservoir']
    folds_path=SCIENCE.parent/'matched-benchmark-20261103/folds.json'
    assert sha(folds_path)=='763d081e387c2d00c6412f07c03af43babca06430a49d5e3f3b5d081c327ade3'
    folds=read(folds_path);stage_map=folds['stage_by_fitting_donor']
    assert len(stage_map)==57
    pins={str(p):sha(p) for p in (folds_path,auth_path,scoring,evaluation)}
    rows=[]
    for f in folds['folds']:
        cache=SCIENCE/'caches'/f"fold{f['fold']}"
        manifest=read(cache/'manifest.json');metadata=cache/'evaluation/metadata.npz'
        assert sha(metadata)==manifest['files_SHA256']['evaluation/metadata.npz']
        with np.load(metadata,allow_pickle=False) as m:
            donors=m['donors'].copy();cell_donors=m['donor'].copy()
        assert donors.tolist()==f['evaluation_donors']==manifest['evaluation_donors']
        assert set(cell_donors.tolist())==set(donors.tolist())
        counts=np.bincount([stage_map[d] for d in donors],minlength=4)
        cell_stages=np.asarray([stage_map[d] for d in cell_donors],dtype=np.int64)
        cell_counts=np.bincount(cell_stages,minlength=4)
        reference_indices=np.random.default_rng(20261026).choice(len(cell_donors),min(1500,len(cell_donors)),replace=False)
        real_stage_counts=np.bincount(cell_stages[reference_indices],minlength=4)
        # Actual scorer adds each80k-cell stage ensemble once per queried donor.
        pool=Reservoir()
        for stage,count in enumerate(counts):
            values=np.full((200,400,1),stage,dtype=np.float32)
            for _ in range(int(count)):pool.add(values)
        actual_generated_stage_counts=np.bincount(pool.cells[:,0].astype(int),minlength=4)
        # Independent full-priority reconstruction verifies the isolated native
        # streaming class, without supplying any molecular coordinate values.
        total=80000*len(donors)
        keys=np.random.default_rng(20261026).random(total)
        labels=np.repeat(np.arange(4),80000*counts)
        expected=np.bincount(labels[np.argpartition(keys,1499)[:1500]],minlength=4)
        np.testing.assert_array_equal(actual_generated_stage_counts,expected)
        query_weights=counts/len(donors);reference_weights=cell_counts/len(cell_donors)
        actual_model_weights=actual_generated_stage_counts/1500
        actual_real_weights=real_stage_counts/len(reference_indices)
        donor_sizes=[int((cell_donors==d).sum()) for d in donors]
        rows.append(dict(fold=f['fold'],evaluation_donors=len(donors),evaluation_cells=len(cell_donors),
            query_donors_per_stage=counts.tolist(),query_stage_probabilities=query_weights.tolist(),
            observed_cells_per_stage=cell_counts.tolist(),observed_cell_stage_probabilities=reference_weights.tolist(),
            actual_generated_1500_reservoir_stage_counts=actual_generated_stage_counts.tolist(),
            actual_real_reference_reservoir_stage_counts=real_stage_counts.tolist(),
            actual_real_reference_size=len(reference_indices),
            generated_stage_reservoir_matches_original_streaming_class=True,
            expected_query_vs_cell_stage_total_variation=float(abs(query_weights-reference_weights).sum()/2),
            actual_model_vs_real_reference_stage_total_variation=float(abs(actual_model_weights-actual_real_weights).sum()/2),
            evaluation_cells_per_donor_min=min(donor_sizes),evaluation_cells_per_donor_max=max(donor_sizes),
            raw_donor_cell_counts=donor_sizes,donor_order=donors.tolist()))
        for p in (cache/'manifest.json',metadata):pins[str(p)]=sha(p)
    assert sum(row['evaluation_donors'] for row in rows)==57
    assert 'torch' not in sys.modules
    record=dict(status='FROZEN_GENERATED_DONOR_WEIGHTS_AND_REAL_CELL_REFERENCE_WEIGHTS_METADATA_RECONSTRUCTION_COMPLETE',
        UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),folds=rows,
        calibration_stage_weights=[.25]*4,
        frozen_benchmark_generated_mixture='Stage ensembles are repeated once per evaluation donor at that stage, followed by the fixed1500-cell priority reservoir. Stage probabilities are approximately donor counts per stage.',
        frozen_benchmark_real_reference='The original fixed-seed reference samples1500 observed cells uniformly across the evaluation cohort. Donors and stages are weighted by observed cell counts, not equal biological-donor weights.',
        equal_stage_statement_correction='Only the calibration mixture has explicit equal375-cell per-stage weighting. The earlier current-goal note and commentary incorrectly extended this to the frozen external benchmark.',
        scope='This is a definition/metadata audit. All methods use the same frozen scoring procedure; numerical values remain valid under that declared procedure. This is not proof of a method-ranking change or a newly discovered molecular-label bug.',
        implications=['Stage-balanced training is a hypothesis for the equal-stage calibration objective; it is not automatically aligned with the frozen donor-weighted external query mixture.',
                      'An observed-cell reference and an equal-query-donor model mixture can have different stage proportions even for perfectly fitted conditional models.',
                      'Preserve the original57 primary and prespecified53 sensitivity. A donor-matched cell-distance analysis would be supplementary and cannot replace the frozen primary or automatically establish SOTA.'],
        recommendation='After the full frozen matrix and both reports exist, consider a clearly declared supplementary donor-matched reference comparison using the same saved samples. Do not select a new metric, model, endpoint or sampling law from partial held-out scores.',
        frozen_primary_or_training_studies_changed=False,new_model_generation_or_training=False,
        expression_counts_or_PCs_or_summary_targets_decoded=False,comparison_quality_results_decoded=False,
        clinical_and_row_identity_metadata_only=True,no_GPU_allocation=True,
        source_SHA256={str(Path(__file__).resolve()):sha(__file__)},input_SHA256=pins,
        full_goal_complete=False,SOTA_claim=False)
    output.write_text(json.dumps(record,indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(status=record['status'],fold_stage_weight_comparisons=[{k:x[k] for k in (
        'fold','query_donors_per_stage','actual_generated_1500_reservoir_stage_counts',
        'actual_real_reference_reservoir_stage_counts','actual_model_vs_real_reference_stage_total_variation')} for x in rows])))


if __name__=='__main__':main()
