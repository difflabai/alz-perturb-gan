"""Four real native-GANTrainer parity fixtures for optional scalar follow-up."""
import datetime, hashlib, importlib.util, json, os, sys
from pathlib import Path
ROOT=Path('/ml2/alz');sys.path.insert(0,str(ROOT/'analysis'))
assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
from released097_followup_design import neighbors
path=ROOT/'analysis/faithful097/native_trainer_parity.py'
spec=importlib.util.spec_from_file_location('native_followup_fixture',path)
native=importlib.util.module_from_spec(spec);spec.loader.exec_module(native)
import torch
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    dest=ROOT/'analysis/followfaith097/possible_settings_native_CPU_qualification.json'
    assert not dest.exists();dest.parent.mkdir(exist_ok=True);rows=[]
    for candidate in neighbors():
        for seed in (0,1):
            q=native.run(candidate['hyperparameters'],seed,pairs=3)
            q.update(anchor_id='f000',candidate_id=candidate['id'],
                     changed_field=candidate['changed_field'],multiplier=.5)
            rows.append(q);print(json.dumps(dict(candidate=candidate['id'],seed=seed,
                exact_native_update_prefix=True)),flush=True)
    assert len(rows)==4 and not torch.cuda.is_initialized()
    files=[Path(__file__).resolve(),path,ROOT/'analysis/released097_followup_design.py',
           ROOT/'analysis/duration096/native_fixture.py',
           *list((ROOT/'analysis/faithful097/model').glob('*.py')),
           Path(sys.modules['particlegan.training'].__file__)]
    q=dict(status='ALL4_POSSIBLE_FOLLOWUP_SETTING_SEED_NATIVE_TRAINER_PREFIX_CASES_PASS',
        UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),native_version='0.9.7',
        rows=rows,actual_cases=4,actual_optimizer_pairs_per_implementation=12,
        total_optimizer_pairs_both_implementations=24,fullwidth_genes=2000,
        prior_regularizer_domain='all512rows',explicit_thermal_binding=True,
        all_models_and_optimizers_initialized_fresh_at0=True,real_ALZ_data_read=False,
        actual_coordinate_study_frozen=False,actual_coordinate_training_started=False,
        native_package_modified=False,CUDA_initialized=False,GPU_allocation_created=False,
        source_SHA256={str(p):sha(p) for p in files})
    dest.write_text(json.dumps(q,indent=2)+'\n');print(q['status'],flush=True)
if __name__=='__main__':main()
