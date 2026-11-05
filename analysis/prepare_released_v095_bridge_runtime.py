"""Isolate a latest-release bridge runtime with the established shared-GPU guard."""
import ast
import datetime
import hashlib
import json
from pathlib import Path
import re
import subprocess

ROOT = Path('/ml2/alz')
OLD = ROOT / 'released_rate_g025d050p025_runtime'
NEW = ROOT / 'released_v095_bridge_runtime'
MODEL = ROOT / 'analysis/released_tuning_v095'
read = lambda p: json.loads(Path(p).read_bytes())
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()


def assignment(source, name, value):
    source, n = re.subn('^' + name + '=.*$', lambda _: name + '=' + value, source, flags=re.M)
    assert n == 1, name
    return source


def main():
    assert not NEW.exists()
    proof = read(MODEL / 'CPU_qualification.json')
    assert proof['tests_passed'] == 8 and proof['native_version'] == '0.9.5'
    for p, h in proof['source_SHA256'].items(): assert sha(p) == h
    parameters = read(MODEL / 'protocol.json')['version_bridge']['parameters']
    control = dict(status='QUALIFIED_RELEASED095_PARAMETERIZATION_WITH_EXPLICIT_CONTINUOUS_DOMAIN',
        utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        protocol_SHA256=sha(MODEL/'protocol.json'), CPU_qualification_SHA256=sha(MODEL/'CPU_qualification.json'),
        native_version='0.9.5', penalty_options={'input_domain':'continuous'},
        checkpoint_version_and_domain_refusal_verified=True, historical_environment_modified=False,
        historical0_9_4_metrics_counted_as_new0_9_5_results=False, model_CPU_checks=8,
        scope='New private0.9.5 cohort; unchanged ALZ architecture/data/seeds/update body. Public factory continuous domain saved outside Recipe as required for component loops. All27 new IDs retained; historical8 old-version settings remain contextual. First new20k bridge uses best prior rates, then scalar tuning under the prospective new-version protocol.')
    (MODEL / 'rate_parameterization_and_controls.json').write_text(json.dumps(control, indent=2)+'\n')
    observation = ROOT/'analysis/released_v095_bridge_initial_resource_observation.json'
    assert not observation.exists()
    query = subprocess.check_output(['nvidia-smi','-i','0','--query-gpu=uuid,memory.total,memory.used,memory.free','--format=csv,noheader,nounits'],text=True).strip().split(',')
    observation.write_text(json.dumps({'status':'READ_ONLY_GPU0_SNAPSHOT_NOT_ADMISSION','utc':control['utc'],
        'gpu_uuid':query[0].strip(),'GPU_total_bytes':int(query[1])*2**20,'GPU_used_bytes':int(query[2])*2**20,
        'GPU_free_bytes':int(query[3])*2**20,'host_available_bytes':int(next(x.split()[1] for x in Path('/proc/meminfo').read_text().splitlines() if x.startswith('MemAvailable:')))*1024,
        'memory_pressure':Path('/proc/pressure/memory').read_text(),'GPU_context_created':False,'allocation_or_lease_created':False},indent=2)+'\n')
    parent = read(ROOT/'artifacts/allocations/CAL_GPU0_V1_released_best_extend40k_f0_pg_s0/authorization.json')
    NEW.mkdir()
    for path in OLD.glob('*.py'):
        source = path.read_text().replace('released_rate_g025d050p025','released_v095_bridge')
        source = source.replace('analysis/released_tuning','analysis/released_tuning_v095')
        source = source.replace('lr_g025_d050_p025','v095_lr_g050_d050_p050')
        source = source.replace('gpu0-released-g025d050p025-20261104','gpu0-released-v095-bridge-20261105')
        source = source.replace('ONE_RELEASED_LR_G025_D050_P025_SEED0_TRIAL_ONLY','ONE_OFFICIAL095_BEST_RATE_BRIDGE_SEED0_TRIAL_ONLY')
        if path.name == 'preset_scope.py':
            source = assignment(source,'HYPER',repr(parameters))
            source = assignment(source,'NATIVE_PROOF',"MODEL_DIR/'native_release.json'")
            source = assignment(source,'COMPAT_PROOF',"ROOT/'analysis/released_tuning/historical_native_checkpoint_CPU_audit.json'")
            source = source.replace("proof['tests_passed']==6", "proof['tests_passed']==8")
            source = source.replace("assert len(native)==51", "assert len(native)==68")
            source = source.replace("    assert auth['operation']==", "    assert auth['particlegan_version']=='0.9.5' and auth['penalty_options']=={'input_domain':'continuous'}\n    assert auth['operation']==",1)
            source = source.replace("    assert proof['tests_passed']==8", "    assert proof['native_version']=='0.9.5' and proof['explicit_continuous_domain_arithmetic_and_version_bound_restore_verified'] is True\n    assert proof['tests_passed']==8")
        elif path.name == 'resource_profile.py':
            source = assignment(source,'PREVIOUS_REGISTRY',"Path("+repr(str(Path(parent['registry_path'])/'registry.json'))+")")
        elif path.name == 'binding.py':
            source = source.replace("return [PROJECT/'current_base_20261029/runtime/lib/python3.12/site-packages',PG_SHARED_SITE]", "return [HERE.parent/'artifacts/native-releases/particlegan-0.9.5/site-packages',PROJECT/'current_base_20261029/runtime/lib/python3.12/site-packages',PG_SHARED_SITE]")
        elif path.name == 'prepare.py':
            source = source.replace("auth.update(cell=value,", "auth.update(particlegan_version='0.9.5',penalty_options={'input_domain':'continuous'},cell=value,")
            source = source.replace('G/D/prior nominal rates=(0.003, 0.009, 0.0075)', 'Official0.9.5 version bridge, continuous150D, G/D/prior rates=(0.006,0.009,0.015)')
            source = source.replace("model=b.read(scope.MODEL_PROOF);", "pins[str(b.read(scope.NATIVE_PROOF)['package_metadata_path'])]=b.read(scope.NATIVE_PROOF)['package_metadata_SHA256']\n    model=b.read(scope.MODEL_PROOF);")
        elif path.name == 'worker_gpu0.py':
            source = source.replace("    b.require(q.get('schema')==1", "    b.require(q.get('particlegan_version')==auth['particlegan_version'] and q.get('penalty_options')==auth['penalty_options'],'checkpoint native version or domain differs')\n    b.require(q.get('schema')==1")
            source = source.replace('    import torch\n', '    from release_env import activate,VERSION,PENALTY_OPTIONS\n    native=activate()\n    b.require(VERSION==auth[\'particlegan_version\'] and PENALTY_OPTIONS==auth[\'penalty_options\'],\'native package identity differs\')\n    import torch\n')
            source = source.replace("    end=time.monotonic()", "    native_receipt={'version':VERSION,'package_origin':str(Path(native.__file__).resolve()),'penalty_options':PENALTY_OPTIONS,'official_wheel_SHA256':b.read(b.HERE.parent/'analysis/released_tuning_v095/native_release.json')['official_wheel_SHA256']}\n    dump(out/'native_release_receipt.json',native_receipt)\n    end=time.monotonic()")
            source = source.replace("result.update(status='completed',", "result.update(particlegan_version=VERSION,penalty_options=PENALTY_OPTIONS,native_release_receipt_path=str(out/'native_release_receipt.json'),native_release_receipt_SHA256=sha(out/'native_release_receipt.json'),status='completed',")
            source = source.replace("q.get('particlegan_version')==auth['particlegan_version'] and q.get('penalty_options')==auth['penalty_options']", "q.get('particlegan_version')==auth.get('particlegan_version')=='0.9.5' and q.get('penalty_options')==auth.get('penalty_options')=={'input_domain':'continuous'}")
        elif path.name == 'finish.py':
            source = source.replace('    import numpy as np', '    from release_env import activate,VERSION,PENALTY_OPTIONS\n    native=activate()\n    import numpy as np')
            source = source.replace("    assert q['policy']['device']=='cuda:0'", "    assert candidate['particlegan_version']==q['particlegan_version']==auth['particlegan_version']==VERSION=='0.9.5'\n    assert candidate['penalty_options']==q['penalty_options']==auth['penalty_options']==PENALTY_OPTIONS=={'input_domain':'continuous'}\n    nr=Path(candidate['native_release_receipt_path']);assert b.sha(nr)==candidate['native_release_receipt_SHA256']\n    assert b.read(nr)['version']==VERSION and b.read(nr)['package_origin']==str(Path(native.__file__).resolve())\n    assert q['policy']['device']=='cuda:0'")
            source = source.replace("        assert cq['completed_updates']", "        assert cq['particlegan_version']==VERSION and cq['penalty_options']==PENALTY_OPTIONS\n        assert cq['completed_updates']")
            source = source.replace("'full_native_recipe_model_and_seed_streams_verified':True", "'official095_origin_version_and_continuous_domain_verified':True,'full_native_recipe_model_and_seed_streams_verified':True")
        elif path.name == 'test_integration.py':
            source = source.replace("auth=copy.deepcopy(self.auth)\n        checkpoint={'schema':1", "auth=copy.deepcopy(self.auth);auth.update(particlegan_version='0.9.5',penalty_options={'input_domain':'continuous'})\n        checkpoint={'particlegan_version':'0.9.5','penalty_options':{'input_domain':'continuous'},'schema':1")
            source = source.replace("[('completed_updates',19000)", "[('particlegan_version','0.9.4'),('penalty_options',{'input_domain':'auto'}),('completed_updates',19000)")
        elif path.name == 'test_scope.py':
            source = source.replace("auth={'cell':dict(scope.CELL)", "auth={'particlegan_version':'0.9.5','penalty_options':{'input_domain':'continuous'},'cell':dict(scope.CELL)")
            source = source.replace("q={'schema':1", "q={'particlegan_version':'0.9.5','penalty_options':{'input_domain':'continuous'},'schema':1")
            extra = '''    def test_native095_version_domain_and_import_root_refuse_mismatch_without_import(self):
        for change in ({'particlegan_version':'0.9.4'},{'penalty_options':{'input_domain':'auto'}}):
            auth=self.auth();auth.update(change);rd,sh=self.active(auth)
            with rd,sh,self.assertRaises(AssertionError):scope.validate_authorization(auth,require_runtime_proof=False)
        auth=self.auth();sites=b.native_site_roots(scope.CELL,auth['files_SHA256'])
        self.assertEqual(sites[0],scope.ROOT/'artifacts/native-releases/particlegan-0.9.5/site-packages')
        self.assertNotIn('torch',sys.modules)

'''
            source = source.replace("    def test_model_proof_whole_preset_scope_and_owner_window", extra+"    def test_model_proof_whole_preset_scope_and_owner_window")
        elif path.name == 'qualify.py':
            source = source.replace("'tests_passed':24", "'tests_passed':25,'native_version':'0.9.5','penalty_options':{'input_domain':'continuous'},'native095_scope_and_import_root_refusals_verified':True")
        ast.parse(source); (NEW/path.name).write_text(source)
    policy=read(OLD/'policy.prepared.json')
    policy['registry_path']=str(ROOT/'artifacts/gpu0-released-v095-bridge-20261105/live_registry')
    policy['resource_basis']['observation_path']=str(observation);policy['resource_basis']['observation_SHA256']=sha(observation)
    (NEW/'policy.prepared.json').write_text(json.dumps(policy,indent=2)+'\n')
    plan=read(OLD/'plan.json');plan.update(enabled=False,experiment='released_v095_bridge',candidate_id='v095_lr_g050_d050_p050')
    (NEW/'plan.json').write_text(json.dumps(plan,indent=2)+'\n')
    print(json.dumps({'status':'GENERATED_DISABLED_OFFICIAL095_GPU0_BRIDGE_RUNTIME','runtime':str(NEW),'model_checks_passed':8,'runtime_qualification_pending':True,'GPU_job_started':False}))


if __name__ == '__main__':
    main()
