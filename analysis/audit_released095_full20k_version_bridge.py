"""Compare actually completed release versions without rewriting native state."""
import datetime
import hashlib
import json
import os
from pathlib import Path
import sys

ROOT=Path('/ml2/alz')
PROJECT=Path('/ml2/codex-storage/future-workspaces/2026-10-27/task-2/benchmark')
assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
sys.path[:0]=list(map(str,[ROOT/'released_v095_bridge_runtime',
    PROJECT/'current_base_20261029/runtime/lib/python3.12/site-packages',
    PROJECT/'.venv312/lib/python3.12/site-packages']))
import numpy as np
import torch
from state_fingerprint import fingerprint

read=lambda p:json.loads(Path(p).read_bytes())
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    old_path=ROOT/'analysis/CAL_GPU0_V1_released_rate_g050d050p050_resume1_f0_pg_s0-result.json'
    new_path=ROOT/'analysis/CAL_GPU0_V1_released_v095_bridge_f0_pg_s0-result.json'
    old,new=read(old_path),read(new_path)
    for result in (old,new):
        for key in ('owned_release_and_clean_settlement_verified',
                    'independent_saved_scores_checkpoint_and_count_exemplars_verified',
                    'predeclared_all_three_checkpoint_saved_scores_and_counts_verified',
                    'native_public_CUDA_reload_and_optimizer_state_byte_exact_verified'):
            assert result[key] is True,key
    assert new['official095_origin_version_and_continuous_domain_verified'] is True
    oc,nc=old['candidate'],new['candidate']
    assert nc['particlegan_version']=='0.9.5' and nc['penalty_options']=={'input_domain':'continuous'}
    for key in ('released_hyperparameters','variant_config','donors','generation_replicates',
                'training_seed','calibration_manifest_SHA256','trace_manifest_SHA256','sampling_contract'):
        assert oc[key]==nc[key],key
    assert nc['completed_updates']==nc['fixed_endpoint']==oc['completed_updates']==oc['fixed_endpoint']==20000
    old_curve_path,new_curve_path=Path(oc['convergence_curve_path']),Path(nc['convergence_curve_path'])
    assert sha(old_curve_path)==oc['convergence_curve_SHA256']
    assert sha(new_curve_path)==nc['convergence_curve_SHA256']
    old_entries,new_entries=read(old_curve_path)['entries'],read(new_curve_path)['entries']
    assert [e['completed_updates'] for e in old_entries]==[5000,10000,20000]
    assert [e['completed_updates'] for e in new_entries]==[5000,10000,20000]
    rows=[];pins={}
    for oe,ne in zip(old_entries,new_entries):
        for entry in (oe,ne):
            cp=Path(entry['checkpoint_path']);assert sha(cp)==entry['checkpoint_SHA256'];pins[str(cp)]=sha(cp)
        oq=torch.load(oe['checkpoint_path'],map_location='cpu',weights_only=False)
        nq=torch.load(ne['checkpoint_path'],map_location='cpu',weights_only=False)
        assert nq['particlegan_version']=='0.9.5' and nq['penalty_options']=={'input_domain':'continuous'}
        assert 'particlegan_version' not in oq and 'penalty_options' not in oq
        assert nq['policy']['device']==oq['policy']['device']=='cuda:0'
        assert nq['released_hyperparameters']==oq['released_hyperparameters']==nc['released_hyperparameters']
        # Exclude only the two explicit caller-owned version/domain declarations.
        # Native recipe, optimizers, clocks, CUDA/CPU RNG bytes and every other
        # scientific field are compared as saved, without migration or relabeling.
        normalized={k:v for k,v in nq.items() if k not in ('particlegan_version','penalty_options')}
        differences=[k for k in sorted(oq.keys()|normalized.keys())
            if k not in oq or k not in normalized or fingerprint(oq[k])!=fingerprint(normalized[k])]
        arrays=[]
        for od,nd in zip(oe['replicate_directories'],ne['replicate_directories']):
            op,npth=Path(od)/'arrays.npz',Path(nd)/'arrays.npz'
            with np.load(op,allow_pickle=False) as oa,np.load(npth,allow_pickle=False) as na:
                equal=oa.files==na.files and all(np.array_equal(oa[k],na[k]) for k in oa.files)
            exemplars=[]
            for stage in range(4):
                op,npth=Path(od)/f'stage{stage}_exemplar.npz',Path(nd)/f'stage{stage}_exemplar.npz'
                with np.load(op,allow_pickle=False) as oa,np.load(npth,allow_pickle=False) as na:
                    exemplars.append(np.array_equal(oa['counts'],na['counts']))
            arrays.append({'saved_projection_arrays_exact':equal,'all_four_raw_count_exemplars_exact':all(exemplars)})
        rows.append({'updates':oe['completed_updates'],
            'full_native_policy_state_byte_exact':fingerprint(oq['policy'])==fingerprint(nq['policy']),
            'all_common_scientific_state_byte_exact':not differences,
            'different_common_scientific_fields':differences,
            'historical_common_state_fingerprint':fingerprint(oq),
            'latest_common_state_fingerprint':fingerprint(normalized),
            'metric_deltas':{k:ne['metrics'][k]-oe['metrics'][k] for k in ('energy_score','cell_energy_distance','spread_ratio')},
            'replicate_saved_array_comparisons':arrays})
    assert not torch.cuda.is_initialized()
    result={'status':'ACTUAL_AUDITED_094_TO095_FULL20K_NATIVE_AND_COUNT_BRIDGE_COMPARISON',
        'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'historical_result_path':str(old_path),'historical_result_SHA256':sha(old_path),
        'latest_result_path':str(new_path),'latest_result_SHA256':sha(new_path),
        'compared_checkpoints':rows,'checkpoint_source_SHA256':pins,
        'only_removed_metadata_fields':['particlegan_version','penalty_options'],
        'native_recipe_or_device_metadata_rewritten':False,
        'all_three_full_scientific_checkpoints_match':all(x['all_common_scientific_state_byte_exact'] for x in rows),
        'all_saved_projection_arrays_and_count_exemplars_match':all(all(a.values()) for x in rows for a in x['replicate_saved_array_comparisons']),
        'latest_fixed_endpoint_joint_acceptance':new['decision'],
        'new0_9_5_completed_endpoints':1,'historical0_9_4_rate_results_not_reclassified':True,
        'CUDA_initialized':False,'source_SHA256':{str(Path(__file__)):sha(__file__)},
        'scope':'Actual independent20k executions and their saved5k/10k/20k states/arrays. Historical eight rate results retain their original version; even exact bridge reproduction does not establish a stable multi-seed ATH or make those old runs new0.9.5executions.'}
    dest=ROOT/'analysis/released095_vs094_full20k_version_bridge_comparison.json';assert not dest.exists()
    dest.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ('compared_checkpoints','checkpoint_source_SHA256')},indent=2))


if __name__=='__main__':main()
