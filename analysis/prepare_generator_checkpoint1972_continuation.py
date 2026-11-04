"""Render a disabled exact continuation of the controlled GPU1 handoff."""
import ast
import datetime
import hashlib
import io
import json
import re
import subprocess
import tokenize
from pathlib import Path

ROOT = Path('/ml2/alz')
read = lambda p: json.loads(Path(p).read_bytes())
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
PARENT = 'CAL_GPU0_V1_released_rate_g025d050p050_f0_pg_s0'
N = 1972
NEW = ROOT/'released_rate_g025d050p050_resume1_runtime'
PROTO = ROOT/'analysis/released_rate_g025d050p050_resume1_protocol.json'


def assignment(source, name, value):
    source, count = re.subn('^'+re.escape(name)+'=.*$', lambda _: name+'='+value, source, flags=re.M)
    assert count == 1, (name, count)
    return source


def replace_update_clock(source):
    """Change clock literals and prose while preserving all pinned SHA strings."""
    tokens=[]
    for token in tokenize.generate_tokens(io.StringIO(source).readline):
        value=token.string
        if token.type==tokenize.NUMBER and value in ('10842','9158'):
            value={'10842':str(N),'9158':str(20000-N)}[value]
        elif token.type==tokenize.STRING:
            try: literal=ast.literal_eval(value)
            except (ValueError,SyntaxError): literal=None  # f-string prose
            if not (isinstance(literal,str) and re.fullmatch('[0-9a-f]{64}',literal)):
                value=value.replace('10842',str(N)).replace('9158',str(20000-N))
        tokens.append(token._replace(string=value))
    return tokenize.untokenize(tokens)


def main():
    assert not NEW.exists() and not PROTO.exists()
    authpath = ROOT/'artifacts/allocations'/PARENT/'authorization.json'
    auth = read(authpath)
    stop_path = ROOT/'analysis'/(PARENT+'-incomplete-GPU-handoff-stop.json')
    audit_path = ROOT/'analysis'/(PARENT+'-checkpoint1972-native-state-audit.json')
    stop, audit = read(stop_path), read(audit_path)
    assert stop['completed_updates'] == audit['completed_updates'] == N
    assert stop['owned_processes_absent'] and stop['owned_registry_empty'] and stop['canonical_reservation_released']
    assert audit['CUDA_initialized'] is False and audit['model_floating_values'] == 371177
    assert auth['released_hyperparameters'] == {'generator_lr': .003, 'critic_lr': .009,
        'particle_lr': .015, 'reg_coeff': 1., 'reg_kappa': 1., 'prior_reg': .01}
    assert read(ROOT/'released_rate_g025d050p050_runtime/plan.json')['enabled'] is False
    immutable = ROOT/'artifacts/diagnostics'/PARENT/'terminal1972'
    copies = read(immutable/'copy_receipt.json')['files']
    assert set(copies) == {'recovery.pt', 'learning_curve.jsonl'}
    for entry in copies.values(): assert sha(entry['immutable_copy_path']) == entry['SHA256']
    protocol = {'status':'PROSPECTIVE_EXACT_RELEASED_RATE_1972_TO_20000_CONTINUATION',
        'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(), 'parent_attempt':PARENT,
        'parent_authorization_path':str(authpath), 'parent_authorization_SHA256':sha(authpath),
        'checkpoint_path':copies['recovery.pt']['immutable_copy_path'],
        'checkpoint_SHA256':copies['recovery.pt']['SHA256'], 'completed_updates':N,
        'original_absolute_deadline_unix':stop['original_absolute_training_deadline_unix'],
        'parent_charged_GPU_seconds':stop['charged_GPU_seconds'], 'diagnostic_checkpoints':[],
        'training_log':copies['learning_curve.jsonl'],
        'whole_parent_scientific_fingerprint':audit['whole_scientific_checkpoint_fingerprint'],
        'parent_stop_path':str(stop_path), 'parent_stop_SHA256':sha(stop_path),
        'parent_audit_path':str(audit_path), 'parent_audit_SHA256':sha(audit_path),
        'parent_reverted_plan_path':str(ROOT/'released_rate_g025d050p050_runtime/plan.json'),
        'parent_reverted_plan_SHA256':sha(ROOT/'released_rate_g025d050p050_runtime/plan.json'),
        'parent_stop_evidence_commit':'df20aac', 'parent_activation_actual_revert':'4e65b3e',
        'same_frozen_model_source_directory':str(ROOT/'analysis/released_tuning'),
        'new_model_updates':20000-N, 'fixed_endpoint':20000, 'training_seed':0,
        'new_training_seed':False, 'new_training_model':False,
        'source_physical_GPU':1, 'destination_physical_GPU':0,
        'continuation_scope':'Same native policy/models/optimizers/samplers/usage/numpy/cell/count/CPU-CUDA Gamma state and11original streams. Original deadline and total20k endpoint retained. Preserve original log prefix; create pending5k/10k diagnostic snapshots during continuation and score all3checkpoints after full training. Byte-exact CUDA policy and complete scientific snapshot restore required before any update. No fresh model/seed, selection change, foreign control, or guard relaxation.'}
    PROTO.write_text(json.dumps(protocol, indent=2)+'\n')
    obs_path = ROOT/'analysis/released_rate_g025d050p050_resume1_initial_resource_observation.json'
    raw = subprocess.check_output(['nvidia-smi','-i','0','--query-gpu=uuid,memory.total,memory.used,memory.free',
                                  '--format=csv,noheader,nounits'], text=True, timeout=5).strip().split(',')
    assert raw[0].strip() == 'GPU-72c1b506-891d-b8bc-b353-e020585e1c47'
    observation = {'status':'READ_ONLY_GPU0_INITIAL_SNAPSHOT_NOT_ADMISSION',
        'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(), 'gpu_uuid':raw[0].strip(),
        **{key:int(value.strip())*2**20 for key,value in zip(('GPU_total_bytes','GPU_used_bytes','GPU_free_bytes'),raw[1:])},
        'memory_pressure':Path('/proc/pressure/memory').read_text(),
        'GPU_context_created':False, 'allocation_or_lease_created':False}
    assert not obs_path.exists(); obs_path.write_text(json.dumps(observation,indent=2)+'\n')
    NEW.mkdir()
    for path in (ROOT/'released_rate_resume_runtime').glob('*.py'):
        source = path.read_text().replace('released_rate_resume_runtime',NEW.name)
        source = source.replace('released_rate_resume_protocol.json',PROTO.name)
        source = source.replace('released_rate_resume_runtime_CPU_qualification','released_rate_g025d050p050_resume1_runtime_CPU_qualification')
        source = source.replace('released_rate_resume_owner_execution_review','released_rate_g025d050p050_resume1_owner_execution_review')
        source = source.replace('released_rate_g050d050p050','released_rate_g025d050p050')
        source = source.replace('lr_g050_d050_p050','lr_g025_d050_p050')
        source = replace_update_clock(source)
        source = source.replace('gpu0-released-g050d050p050-resume1-20261104',
                                'gpu0-released-g025d050p050-resume1-20261104')
        source = source.replace("'tests_passed':27","'tests_passed':28").replace("proof['tests_passed']==27","proof['tests_passed']==28")
        source = source.replace('native1972_parent_all_scientific_streams_and_diagnostics_lineage_verified',
                                'native_parent_all_scientific_streams_and_diagnostics_lineage_verified')
        if path.name == 'preset_scope.py':
            source = assignment(source,'HYPER',repr(auth['released_hyperparameters']))
            source = source.replace('released_rate_g025d050p050_initial_resource_observation.json',obs_path.name)
        elif path.name == 'resource_profile.py':
            source = assignment(source,'PREVIOUS_REGISTRY',repr(str(Path(auth['registry_path'])/'registry.json')))
            source = source.replace("PREVIOUS_REGISTRY='", "PREVIOUS_REGISTRY=Path('")
            source = re.sub(r"^(PREVIOUS_REGISTRY=Path\('[^']*')$",r'\1)',source,flags=re.M)
        elif path.name == 'prepare.py':
            source = source.replace('(0.006, 0.009, 0.015)','(0.003, 0.009, 0.015)')
            source = source.replace('original5k/10k checkpoints/log prefix retained',
                                    'original log prefix retained and pending5k/10k checkpoints created at their original clocks')
        elif path.name == 'resume_training.py':
            source = (ROOT/'analysis/released_continuation_support/resume_training.py').read_text()
        elif path.name == 'test_resume.py':
            source = (ROOT/'analysis/released_continuation_support/test_resume.py').read_text()
            source = re.sub(r'^scope=SimpleNamespace\(.*$', 'import preset_scope as scope',source,flags=re.M)
            source = source.replace("'completed_updates':1000","'completed_updates':1972").replace("'completed_steps':1000","'completed_steps':1972").replace("'steps':1000","'steps':1972")
        if path.name == 'finish.py':
            source = source.replace('original5k/10k snapshots','original journal plus pending5k/10k snapshots')
            marker="    assert restored['native_or_usage_or_data_clock_reset'] is False"
            source = source.replace(marker, marker+"\n    assert restored['original_diagnostic_checkpoints_and_log_prefix_byte_exact'] is True")
        ast.parse(source); (NEW/path.name).write_text(source)
    policy = read(ROOT/'released_rate_resume_runtime/policy.prepared.json')
    policy['registry_path'] = str(ROOT/'artifacts/gpu0-released-g025d050p050-resume1-20261104/live_registry')
    policy['resource_basis']['observation_path'] = str(obs_path)
    policy['resource_basis']['observation_SHA256'] = sha(obs_path)
    policy['resource_basis']['foreign_control']='NONE. Exact1972-to20k same-model handoff; original deadline and unchanged shared guards retained. Fresh complete capacity forecast required.'
    (NEW/'policy.prepared.json').write_text(json.dumps(policy,indent=2)+'\n')
    plan = read(ROOT/'released_rate_resume_runtime/plan.json')
    plan.update(enabled=False,experiment='released_rate_g025d050p050_resume1',candidate_id='lr_g025_d050_p050')
    (NEW/'plan.json').write_text(json.dumps(plan,indent=2)+'\n')
    receipt={'status':'DISABLED_EXACT_1972_CONTINUATION_RENDERED_NOT_QUALIFIED_OR_DEPLOYED',
        'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(), 'runtime':str(NEW),
        'protocol_path':str(PROTO), 'protocol_SHA256':sha(PROTO), 'generator_source_SHA256':sha(__file__),
        'source_SHA256':{str(p):sha(p) for p in NEW.glob('*.py')},
        'next_action':'Run actual28CPUchecks, commit disabled source, fresh exact sharedGPU0 source/capacity review, separate activation and one-use admission. Actual CUDA/scientific restore equality remains mandatory; no reset or automatic promotion.'}
    (ROOT/'analysis/released_rate_g025d050p050_resume1_generation.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps({k:v for k,v in receipt.items() if k!='source_SHA256'},indent=2))


if __name__=='__main__': main()
