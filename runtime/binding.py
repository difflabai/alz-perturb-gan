"""Read-only source/device/scientific-cell bindings for the proposed GPU0 owner.

Native defaults are fixed. Fixtures have a separate status, confined paths and
hidden CUDA; they cannot select the canonical project or scientific data.
"""
from dataclasses import dataclass
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import socket
import sys
import time

HERE=Path(__file__).resolve().parent
PROJECT=Path('/ml2/codex-storage/future-workspaces/2026-10-27/task-2/benchmark')
SCIENCE=Path('/ml2/codex-storage/future-workspaces/2026-11-03/task-2/alz-audit/core-comparison-20261103')
GUARD=Path('/ml2/codex-storage/future-workspaces/2026-11-03/task-11/gpu0-coexistence/v2')
GPU0='GPU-72c1b506-891d-b8bc-b353-e020585e1c47'
GPU1='GPU-cb4ce47d-d968-bffd-5646-e830a9fa1c69'
def device(auth):
    value=auth.get('gpu_uuid')
    require(value==GPU0 or (value==GPU1 and auth.get('parallel_profile') is True),'unapproved device')
    return value
SCIENCE_REVIEW='2d5f0add12b0fad41eb907594d9f6a4921147c26aa80229da384b8af7e203519'
GUARD_REVIEW='091589c19caa1898be29acc8ce3e2ab8b79a8f9b9a4cf443219ef13bd55dc08b'
CONTROLLER=PROJECT/'budget_executor_v2_20261103_r2/budget_controller.py'
CONTROLLER_SHA='c8520f426e6b7a94741dd96804b96d9d22a224bff973b7e85d765e5b83c09848'
PROJECT_BUDGET_SHA='85ea495a54dfc524fb493a34002305b534964cf1992fc659d07d2c38951e4872'
PG_SHARED_PTH=PROJECT/'current_base_20261029/runtime/lib/python3.12/site-packages/shared_readonly_dependencies.pth'
PG_SHARED_PTH_SHA='785d4c58af69d916f2557779723c4889b7f5a1481761aa0f4be22fddb10d67ed'
PG_SHARED_SITE=PROJECT/'.venv312/lib/python3.12/site-packages'
NATIVE_LAYOUT=None

class Refusal(RuntimeError):pass

def sha(path):
    value=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):value.update(block)
    return value.hexdigest()

def read(path):return json.loads(Path(path).read_bytes())
def require(value,message):
    if not value:raise Refusal(message)

@dataclass(frozen=True)
class Layout:
    project:Path
    science:Path
    fixture_root:Path|None=None
    @classmethod
    def fixture(cls,root):
        root=Path(root).resolve()
        require(root.name=='fixture' and (root.is_relative_to(Path('/tmp')) or root.is_relative_to(HERE)), 'CPU fixture root must be confined to task/tmp fixture directory')
        require(not root.is_relative_to(PROJECT) and not root.is_relative_to(SCIENCE),'CPU fixture cannot target canonical project/science')
        require(all((root/name).resolve().is_relative_to(root) for name in ('project','science')),'CPU fixture project/science alias escapes root')
        return cls(root/'project',root/'science',root)
    @property
    def fixture_mode(self):return self.fixture_root is not None

NATIVE_LAYOUT=Layout(PROJECT,SCIENCE)

def cell(auth):
    value=auth.get('cell',{})
    require(type(value.get('fold')) is int and value['fold'] in range(3) and value.get('method') in ('pg','scvi') and type(value.get('seed')) is int and value['seed'] in range(5),'unregistered scientific matrix cell')
    if not auth.get('cpu_fixture'):
        import re
        require(isinstance(value.get('experiment'),str) and re.fullmatch(r'[a-z][a-z0-9_]{0,48}',value['experiment']),'safe unique calibration experiment identity required')
        require(value['fold']==0 and value['method']=='pg','calibration experiments are first-fold PG only')
    return value

def allocation_key(value):return f"ALZ_CAL_V1_{value.get('experiment','fixture')}_f{value['fold']}_{value['method']}_s{value['seed']}"
def attempt_key(value):return f"CAL_GPU0_V1_{value.get('experiment','fixture')}_f{value['fold']}_{value['method']}_s{value['seed']}"
def output_path(layout,value):return (layout.science/'runs_calibration' if layout.fixture_mode else Path('/ml2/alz/runs'))/attempt_key(value)
def ledger_path(layout,value):return layout.project/'calibration_gpu0_v1_ledgers'/(attempt_key(value)+'.json')
def cache_path(layout,value):return layout.science/'caches'/f"fold{value['fold']}"

def runtime_path(layout,value):
    if layout.fixture_mode:return Path(sys.executable).absolute()
    if value['method']=='pg':return PROJECT/'current_base_20261029/runtime/bin/python'
    return SCIENCE.parent/'isolated/cfgen-cu124-py311/bin/python'

def expected_command(layout,auth_path,value):
    command=[str(runtime_path(layout,value)),'-I','-S','-B',str(HERE/'worker_gpu0.py'),'--authorization',str(Path(auth_path).resolve())]
    if layout.fixture_mode:command+=['--cpu-fixture-root',str(layout.fixture_root)]
    return command

def native_site_roots(value,pins):
    """Reproduce approved path-only runtime dependencies without executing.pth."""
    if value['method']=='pg':
        require(pins.get(str(PG_SHARED_PTH))==PG_SHARED_PTH_SHA and sha(PG_SHARED_PTH)==PG_SHARED_PTH_SHA,'reviewed PG path-only dependency cohort changed')
        require(PG_SHARED_PTH.read_text().splitlines()==[str(PG_SHARED_SITE)],'PG dependency file is not the one reviewed path-only line')
        return [PROJECT/'current_base_20261029/runtime/lib/python3.12/site-packages',PG_SHARED_SITE]
    return [SCIENCE.parent/'isolated/cfgen-cu124-py311/lib/python3.11/site-packages']

def pinned_inventory(path,expected,pins,label):
    path=Path(path)
    require(pins.get(str(path))==expected and sha(path)==expected,label+' inventory absent or changed')
    data=read(path)
    return data

def require_members(pins,members,label):
    require(bool(members),label+' is empty')
    for path,pin in members.items():
        require(pins.get(str(path))==pin and sha(path)==pin,label+' member absent or changed: '+str(path))

def source_closure(auth,layout):
    """Check complete scientific/native/guard/controller/input closure; no import."""
    pins=auth.get('files_SHA256',{})
    review=layout.science/'REVIEW_MANIFEST.json'
    expected=auth.get('fixture_science_review_SHA256') if layout.fixture_mode else SCIENCE_REVIEW
    science=pinned_inventory(review,expected,pins,'scientific review')
    require_members(pins,science['source_SHA256'],'scientific source')
    guard=pinned_inventory(GUARD/'REVIEW_MANIFEST.json',GUARD_REVIEW,pins,'GPU0 guard review')
    require_members(pins,guard['source_SHA256'],'GPU0 guard source')
    native_path=layout.science/'NATIVE_SOURCE_PINS.json'
    native=pinned_inventory(native_path,science['native_source_inventory_SHA256'],pins,'native source')
    require_members(pins,native['files_SHA256'],'native source')
    value=cell(auth);cache=cache_path(layout,value);cache_manifest=cache/'manifest.json'
    require(pins.get(str(cache_manifest))==science['cache_manifest_SHA256'][str(value['fold'])], 'reviewed fold cache manifest pin absent')
    require(pins.get(str(layout.science/'CORE_PROTOCOL.json'))==science['protocol_SHA256'],'reviewed fixed scientific protocol pin absent')
    require(pins.get(str(layout.science/'traces/manifest.json'))==science['shared_trace_manifest_SHA256'],'reviewed common trace manifest pin absent')
    for manifest in [cache_manifest,layout.science/'traces/manifest.json']:
        for relative,pin in read(manifest)['files_SHA256'].items():
            path=(manifest.parent/relative).resolve()
            require(path.is_relative_to(manifest.parent.resolve()),'input inventory path escapes its root')
            require(pins.get(str(path))==pin and sha(path)==pin,'complete immutable input pin absent/changed: '+str(path))
    required=[*HERE.glob('*.py'),CONTROLLER,layout.project/'project_budget.py',runtime_path(layout,value),Path(auth['policy_path']),cache_manifest,layout.science/'CORE_PROTOCOL.json',layout.science/'traces/manifest.json']
    require(all(str(p) in pins for p in required),'complete integration/controller/runtime/input/policy pins required')
    require(pins.get(str(CONTROLLER))==CONTROLLER_SHA,'unchanged canonical Controller source pin differs')
    if not layout.fixture_mode:
        require(pins.get(str(PROJECT/'project_budget.py'))==PROJECT_BUDGET_SHA,'unchanged canonical accounting source differs')
        venv_cfg=runtime_path(layout,value).parent.parent/'pyvenv.cfg'
        require(str(venv_cfg) in pins,'exact scientific runtime configuration pin required')
        require(all(p.is_dir() for p in native_site_roots(value,pins)),'reviewed scientific import root absent')
    if not layout.fixture_mode:
        extra=auth.get('research_source_SHA256',{})
        require_members(pins,extra,'calibration research source')
        require(all(str(p) in extra for p in HERE.parent.glob('*.py')),'complete calibration scorer/orchestrator source pins required')
        data=HERE.parent/'data/calibration_fold0'
        manifest=pinned_inventory(data/'manifest.json',auth.get('calibration_manifest_SHA256'),pins,'calibration input')
        require(manifest.get('status')=='CPU_CALIBRATION_PROJECTED_WITHOUT_REFITTING','calibration provenance differs')
        require_members(pins,{str(data/name):pin for name,pin in manifest['files_SHA256'].items()},'calibration arrays')
        require(auth.get('operation') in ('evaluate_reference','evaluate_hybrid','evaluate_pure_variant','qualify_pure_ensemble','evaluate_pure_ensemble','confirm_pure_ensemble','qualify_interrupted_pg8','resume_pure_ensemble','confirm_frozen_ensemble_seed2','train_variant','train_bcap','train_bispf','train_semantic','train_learned','train_network_rate','train_hybrid','qualify_continuation','continue_hybrid','train_extended_hybrid'),'registered calibration operation required')
        if auth['operation']=='qualify_continuation':
            require(auth['cap_seconds']==900 and auth.get('qualification_only') is True and auth.get('qualification_steps')==1 and auth.get('caller_execution_limit')==40000,'finite one-step continuation qualification required')
            checkpoint=Path(auth['checkpoint_path'])
            require(pins.get(str(checkpoint))==auth.get('checkpoint_SHA256'),'complete parent checkpoint pin absent')
            config=HERE.parent/'candidate.json'
            require(str(config) in pins and read(config)==auth.get('variant_config'),'exact qualification architecture differs')
            require(all(str(p) in extra for p in (HERE.parent/'model').glob('*.py')),'complete qualification model source pins required')
        elif auth['operation'] in ('qualify_interrupted_pg8','resume_pure_ensemble','confirm_frozen_ensemble_seed2'):
            sys.path.insert(0,str(HERE.parent/'recovery_runtime'))
            from recovery_scope import validate_authorization
            try:validate_authorization(auth)
            except (AssertionError,KeyError,ValueError) as exc:raise Refusal('exact frozen recovery/seed2 refuses: '+str(exc)) from exc
            require(all(str(p) in extra for p in (HERE.parent/'model').glob('*.py')),'complete original frozen model source pins required')
        elif auth['operation']=='confirm_pure_ensemble':
            sys.path.insert(0,str(HERE.parent/'confirmation_runtime'))
            from ensemble_confirmation_scope import validate_authorization
            try:validate_authorization(auth)
            except (AssertionError,KeyError,ValueError) as exc:raise Refusal('frozen two-parent confirmation refuses: '+str(exc)) from exc
            require(all(str(p) in extra for p in (HERE.parent/'model').glob('*.py')),'complete frozen confirmation model source pins required')
        elif auth['operation'] in ('qualify_pure_ensemble','evaluate_pure_ensemble'):
            sys.path.insert(0,str(HERE.parent))
            from ensemble_scope import validate_authorization
            try:validate_authorization(auth)
            except (AssertionError,KeyError,ValueError) as exc:raise Refusal('pure ensemble scope/proof refuses: '+str(exc)) from exc
            require(all(str(p) in extra for p in (HERE.parent/'model').glob('*.py')),'complete ensemble model source pins required')
        elif auth['operation'] in ('evaluate_reference','evaluate_hybrid','evaluate_pure_variant'):
            checkpoint=Path(auth['checkpoint_path'])
            require(pins.get(str(checkpoint))==auth.get('checkpoint_SHA256'),'complete reference checkpoint pin absent')
            if auth['operation']=='evaluate_pure_variant':
                sys.path.insert(0,str(HERE.parent))
                from pure_suite import validate_authorization
                validate_authorization(auth)
                require(all(str(p) in extra for p in (HERE.parent/'model').glob('*.py')),'complete pure model source pins required')
            if auth['operation']=='evaluate_hybrid':
                require(auth['cap_seconds']==900,'finite hybrid sampling cap required')
                config=HERE.parent/'hybrid.json'
                require(str(config) in pins and read(config)==auth.get('hybrid_config'),'exact hybrid config absent or changed')
                require(all(str(p) in extra for p in (HERE.parent/'model').glob('*.py')),'complete hybrid base-model source pins required')
        else:
            if auth['operation']=='train_learned':
                plan=HERE.parent/'learned_observation_plan.json';proof_path=HERE.parent/'analysis/learned_observation_cpu_qualification.json';probe_path=HERE.parent/'analysis/learned_observation_fitting_probe.json'
                options={'enabled':True,'cell_hidden':64,'cell_features':25,'feature_scale':.2,'pool_epsilon':1e-6,'layernorm_epsilon':1e-5}
                require(pins.get(str(plan))==auth.get('learned_observation_plan_SHA256')==sha(plan) and read(plan)==auth.get('learned_observation_config')==options,'learned observation plan disabled or changed')
                require(auth.get('hybrid_config') is None and auth['cell']['seed']==0 and auth['variant_config']=={'z_dim':8,'stage_balance':False,'cell_z_dim':8,'theta_init_scale':1.,'fourier_feature_scale':1.},'pure matched seed0 learned observation scope only')
                require(auth.get('fixed_endpoint')==20000 and auth.get('pure_formulation')=='learned_observation' and auth.get('qualification_only') is False,'complete fixed20k learned observation trial required')
                require(pins.get(str(proof_path))==auth.get('learned_observation_qualification_SHA256')==sha(proof_path) and pins.get(str(probe_path))==auth.get('learned_observation_probe_SHA256')==sha(probe_path),'learned observation CPU qualification/probe pins absent')
                proof=read(proof_path);probe=read(probe_path)
                checks=['disabled_complete_state_and_next_update_bitwise_equal','native_recipe_exact_API_factory','same_initial_G_prior_original_head','fixed100_moments_and_permutations_verified','both_real_and_fake_adversarial_encoder_gradients_verified','head_only_penalty150_vector_inputs_verified','finite_G_dispersion_prior_gradients_through_frozen_encoder','full_serialized_restore_next_update_bitwise_equal','only_declared_observation_update_AST_delta']
                require(proof['status']=='PASSED_SYNTHETIC_CPU_LEARNED_OBSERVATION_QUALIFICATION' and all(proof[k] for k in checks),'learned observation qualification incomplete')
                require(probe['status']=='PASSED_FITTING_ONLY_LEARNED_OBSERVATION_PROBE_NO_TRAINING' and probe['optimizer_updates']==probe['completed_policy_steps']==0 and probe['calibration_or_evaluation_targets_read'] is False and probe['CUDA_initialized'] is False,'learned observation fitting-only probe invalid')
                require(probe['cache_manifest_SHA256']==sha(Path(auth['cache_dir'])/'manifest.json') and probe['learned_observation_config']==options and probe['observation_dimensions']==150 and probe['first100_moment_coordinates_exact'] and probe['new_feature_block_norm_bound_verified'],'learned observation probe/cache differs')
                gradients=probe['isolated_learned50_generator_gradients']
                require(len(gradients)==3 and {q['name'] for q in gradients}=={'G_output_weight','G_dispersion','particle_prior'} and all(math.isfinite(q['norm']) and q['norm']>0 for q in gradients),'isolated learned-feature gradient path unproven')
                for evidence in [proof,probe]:
                    for path,pin in evidence['source_SHA256'].items():require(pins.get(path)==pin==sha(path),'qualified learned observation source changed: '+path)
            if auth['operation']=='train_network_rate':
                plan=HERE.parent/'network_rate_plan.json';proof_path=HERE.parent/'analysis/network_rate_cpu_qualification.json';protocol=HERE.parent/'network_rate_protocol.json'
                options={'lr':.006,'prior_lr_mult':5.}
                require(pins.get(str(plan))==auth.get('network_rate_plan_SHA256')==sha(plan),'network-rate prospective plan pin absent')
                require(read(plan)=={'enabled':True,'native_recipe':'bispf_optimized',**options,'fixed_endpoint':20000,'pure_only':True,'training_seed':0},'network-rate plan disabled or changed')
                require(auth.get('network_rate_config')==options and auth.get('hybrid_config') is None and auth['cell']['seed']==0 and auth['variant_config']=={'z_dim':8,'stage_balance':False,'cell_z_dim':8,'theta_init_scale':1.,'fourier_feature_scale':1.},'pure matched seed0 network-rate scope only')
                require(auth.get('fixed_endpoint')==20000 and auth.get('pure_formulation')=='network_rate_half' and auth.get('qualification_only') is False,'complete fixed20k network-rate trial required')
                require(pins.get(str(proof_path))==auth.get('network_rate_qualification_SHA256')==sha(proof_path) and pins.get(str(protocol))==sha(protocol),'network-rate qualification/protocol pins absent')
                proof=read(proof_path)
                checks=['only_two_official_recipe_fields_change','native_api_component_types_verified','same_initial_complete_G_D_prior_models','optimizer_state_only_intended_network_rates_differ','actual_UpdatePolicy_rates_and_native_schedule_verified','full_serialized_restore_next_update_bitwise_equal','update_observation_context_and_fit_AST_unchanged','finite_nonzero_native_gradients_verified']
                require(proof['status']=='PASSED_SYNTHETIC_CPU_NETWORK_RATE_QUALIFICATION' and proof['network_rate_config']==options and all(proof[k] for k in checks) and proof['CUDA_initialized'] is False,'network-rate CPU qualification incomplete')
                for path,pin in proof['source_SHA256'].items():require(pins.get(path)==pin==sha(path),'network-rate qualified source changed: '+path)
            if auth['operation']=='train_bcap':
                plan=HERE.parent/'bcap_plan.json';proof_path=HERE.parent/'analysis/bcap_cpu_qualification.json';protocol=HERE.parent/'bcap_protocol.json'
                require(pins.get(str(plan))==auth.get('bcap_plan_SHA256')==sha(plan),'native BCAP prospective plan pin absent')
                require(read(plan)=={'enabled':True,'native_recipe':'bcap','fixed_endpoint':20000,'pure_only':True,'training_seed':0},'native BCAP plan disabled or changed')
                require(auth.get('hybrid_config') is None and auth['cell']['seed']==0 and auth['variant_config']=={'z_dim':8,'stage_balance':False,'cell_z_dim':8,'theta_init_scale':1.,'fourier_feature_scale':1.},'pure matched seed0 BCAP scope only')
                require(auth.get('fixed_endpoint')==20000 and auth.get('pure_formulation')=='bcap' and auth.get('qualification_only') is False,'complete fixed20k BCAP trial required')
                require(pins.get(str(proof_path))==auth.get('bcap_qualification_SHA256')==sha(proof_path) and pins.get(str(protocol))==sha(protocol),'native BCAP qualification/protocol pins absent')
                proof=read(proof_path)
                require(proof['status']=='PASSED_SYNTHETIC_CPU_BCAP_QUALIFICATION' and all(proof[k] for k in ['full_serialized_restore_next_update_bitwise_equal','native_api_component_types_verified','same_initial_generator_critic_and_prior_rows_verified','optimized_update_AST_unchanged']) and proof['CUDA_initialized'] is False,'native BCAP qualification incomplete')
                for path,pin in proof['source_SHA256'].items():require(pins.get(path)==pin==sha(path),'native BCAP qualified source changed: '+path)
            if auth['operation']=='train_semantic':
                plan=HERE.parent/'semantic_plan.json';proof_path=HERE.parent/'analysis/semantic_cpu_qualification.json';probe_path=HERE.parent/'analysis/semantic_fitting_gradient_probe.json'
                require(pins.get(str(plan))==auth.get('semantic_plan_SHA256')==sha(plan) and read(plan)==auth.get('semantic_config')=={'enabled':True,'temperature':1.,'epsilon':1e-4},'semantic plan remains disabled or changed')
                require(auth.get('hybrid_config') is None and auth['cell']['seed']==0 and auth['variant_config']=={'z_dim':8,'stage_balance':False,'cell_z_dim':8,'theta_init_scale':1.,'fourier_feature_scale':1.},'matched pure PG8 seed0 semantic scope only')
                require(pins.get(str(proof_path))==auth.get('semantic_qualification_SHA256')==sha(proof_path) and pins.get(str(probe_path))==auth.get('semantic_probe_SHA256')==sha(probe_path),'semantic qualification/probe pins absent')
                proof=read(proof_path);probe=read(probe_path)
                require(proof['status']=='PASSED_SYNTHETIC_CPU_SEMANTIC_QUALIFICATION' and all(proof[k] for k in ['disabled_complete_state_and_next_update_bitwise_equal','hard_forward_and_permutation_verified','finite_nonzero_semantic_input_gradients','frozen_classifier_state_and_gradients_verified','enabled_complete_saved_next_update_bitwise_equal','native_update_AST_and_recipe_unchanged']),'semantic CPU qualification incomplete')
                require(probe['status']=='PASSED_FITTING_ONLY_SEMANTIC_GRADIENT_PROBE_NO_TRAINING' and probe['optimizer_updates']==0 and probe['calibration_or_evaluation_targets_read'] is False and probe['cache_manifest_SHA256']==sha(Path(auth['cache_dir'])/'manifest.json') and probe['classifier_file_SHA256']==sha(Path(auth['cache_dir'])/'type_classifier.pt'),'semantic fitting-only probe/cache differs')
                for evidence in [proof,probe]:
                    for path,pin in evidence['source_SHA256'].items():require(pins.get(path)==pin==sha(path),'qualified semantic source changed: '+path)
            if auth['operation']=='train_bispf':
                plan=HERE.parent/'original_bispf_plan.json';proof_path=HERE.parent/'analysis/original_bispf_cpu_qualification.json'
                require(pins.get(str(plan))==auth.get('original_bispf_plan_SHA256')==sha(plan),'original BISPF prospective plan pin absent')
                require(read(plan)=={'enabled':True,'native_recipe':'bispf','fixed_endpoint':20000,'pure_only':True,'training_seed':0},'original BISPF plan remains disabled or changed')
                require(auth.get('hybrid_config') is None and auth['cell']['seed']==0,'pure seed0 original BISPF scope only')
                require(pins.get(str(proof_path))==auth.get('original_bispf_qualification_SHA256')==sha(proof_path),'original BISPF CPU qualification pin absent')
                proof=read(proof_path)
                require(proof['status']=='PASSED_SYNTHETIC_CPU_ORIGINAL_BISPF_QUALIFICATION' and proof['full_serialized_restore_next_update_bitwise_equal'] and proof['native_api_component_types_verified'] and not proof['CUDA_initialized'],'original BISPF qualification incomplete')
                for path,pin in proof['source_SHA256'].items():require(pins.get(path)==pin==sha(path),'original BISPF qualified source changed: '+path)
            if auth['operation'] in ('continue_hybrid','train_extended_hybrid'):
                plan=HERE.parent/'training_plan.json';protocol=HERE.parent/'continuation_protocol.json'
                require(pins.get(str(plan))==sha(plan) and read(plan)=={'fixed_endpoint':40000} and pins.get(str(protocol))==auth.get('continuation_protocol_SHA256')==sha(protocol),'registered prospective40k endpoint protocol absent')
                require(auth.get('initial_completed_updates')==(20000 if auth['operation']=='continue_hybrid' else 0),'explicit extended initial endpoint differs')
                proof_path=HERE.parent/'analysis/continuation_native_qualification.json'
                require(pins.get(str(proof_path))==auth.get('continuation_qualification_SHA256')==sha(proof_path),'native continuation qualification pin absent')
                proof=read(proof_path)
                require(proof['status']=='VERIFIED_NATIVE_GPU0_QUALIFICATION_ONLY' and proof['clean_settlement_and_owned_release_verified'] and proof['proof']['complete_next_state_and_metrics_bitwise_equal'],'native continuation qualification invalid')
                if auth['operation']=='continue_hybrid':require(proof['proof']['parent_checkpoint_SHA256']==auth['checkpoint_SHA256'],'qualified parent checkpoint differs')
                if auth['operation']=='continue_hybrid':
                    checkpoint=Path(auth['checkpoint_path'])
                    require(pins.get(str(checkpoint))==auth.get('checkpoint_SHA256'),'complete continuation parent pin absent')
            require(auth['cap_seconds']==21600,'fixed training pipeline cap required')
            config=HERE.parent/'candidate.json'
            require(str(config) in pins and read(config)==auth.get('variant_config'),'exact selected candidate config absent or changed')
            require(all(str(p) in extra for p in (HERE.parent/'model').glob('*.py')),'complete model variant source pins required')
            if auth['operation'] in ('train_hybrid','continue_hybrid','train_extended_hybrid'):
                config=HERE.parent/'hybrid.json'
                require(str(config) in pins and read(config)==auth.get('hybrid_config'),'exact training-hybrid mixture config absent or changed')
    for path,pin in pins.items():require(sha(path)==pin,'source/input/runtime changed: '+path)

def verify(auth_path,layout=NATIVE_LAYOUT):
    if layout.fixture_mode:
        require(layout==Layout.fixture(layout.fixture_root),'fixture Layout fields differ from confined factory roots')
        require(Path(auth_path).resolve().is_relative_to(layout.fixture_root),'CPU fixture authorization escapes root')
    else:require(layout==NATIVE_LAYOUT,'native fixed layout differs')
    auth_path=Path(auth_path).resolve();raw=auth_path.read_bytes();auth=json.loads(raw)
    wanted='APPROVED_CPU_FIXTURE_ONLY' if layout.fixture_mode else 'APPROVED_FOR_BOUNDED_EXECUTION'
    require(auth.get('status')==wanted,'source review and one-use GPU0 allocation remain held')
    require(auth.get('cpu_fixture') is layout.fixture_mode,'native/fixture status differs')
    value=cell(auth)
    require(auth.get('allocation_id')==allocation_key(value) and auth.get('attempt_id')==attempt_key(value),'scientific one-use key or distinct GPU0 attempt identity differs')
    require(auth.get('label')==attempt_key(value),'immutable GPU0 attempt label differs')
    require(auth.get('gpu_uuid')==device(auth) and auth.get('machine_id')==socket.gethostname(),'exact GPU0/machine binding differs')
    require(auth.get('source_review_passed') is True and bool(auth.get('scheduler_allocation_id')),'independent source review/scheduler assignment missing')
    expiry=auth.get('expires_unix')
    require(type(expiry) in (float,int) and math.isfinite(expiry) and expiry>time.time(),'finite one-use scheduler expiry missing or expired')
    cap=auth.get('cap_seconds')
    require((type(cap) is int and 11<=cap<=15) if layout.fixture_mode else cap in (900,21600),'fixed finite scientific job cap differs')
    endpoint=auth.get('fixed_endpoint')
    expected=(endpoint if auth.get('operation') in ('evaluate_pure_variant','qualify_pure_ensemble','evaluate_pure_ensemble') and endpoint in (20000,40000) else (40000 if auth.get('operation') in ('continue_hybrid','train_extended_hybrid','confirm_pure_ensemble','resume_pure_ensemble','confirm_frozen_ensemble_seed2') else 20000)) if value['method']=='pg' else 400
    require(endpoint==expected and auth.get('core_GPU_hour_ceiling')==180,'fixed scientific endpoint/core envelope changed')
    wanted_paths={'project_root':layout.project,'science_root':layout.science,'cache_dir':cache_path(layout,value),'output_dir':output_path(layout,value),'ledger_path':ledger_path(layout,value)}
    require(all(auth.get(k)==str(v) for k,v in wanted_paths.items()),'immutable project/cache/output/ledger identity differs')
    require(auth.get('role')=='alz_core_'+value['method'] and auth.get('command')==expected_command(layout,auth_path,value),'reviewed runtime/worker command or role differs')
    policy_path=Path(auth['policy_path']).resolve()
    if layout.fixture_mode:require(policy_path.is_relative_to(layout.fixture_root),'CPU fixture policy escapes root')
    elif auth.get('parallel_profile') is True:
        sys.path.insert(0,str(HERE.parent/'recovery_runtime'))
        import parallel_profile
        try:parallel_profile.validate(auth)
        except (AssertionError,KeyError,ValueError,FileNotFoundError) as exc:raise Refusal('exact parallel profile/review refuses: '+str(exc)) from exc
    else:require(policy_path==Path('/ml2/codex-storage/future-workspaces/2026-11-03/task-11/gpu0-coexistence/integration/policy.GPU0.CORE.PREPARED.json'),'native exact GPU0 policy path differs')
    policy=read(policy_path)
    require(sha(policy_path)==auth.get('policy_SHA256'),'exact reviewed policy changed')
    require(policy.get('gpu')==device(auth) and auth['role'] in policy.get('roles',{}),'policy device/role differs')
    require(policy.get('status')==('PREPARED_FIXTURE_ONLY' if layout.fixture_mode else 'REVIEWED_GPU0_DEPLOYED'),'GPU0 resource policy remains prepared')
    require(policy.get('protected_peers')==auth.get('protected_peers') and bool(auth['protected_peers']),'complete exact protected allowlist differs')
    require(auth.get('registry_path')==policy.get('registry_path'),'exact GPU0 registry binding differs')
    if layout.fixture_mode:
        root=layout.fixture_root
        paths=[auth_path,policy_path,*wanted_paths.values(),Path(auth['registry_path']),*map(Path,policy['locks'])]
        require(all(p.resolve().is_relative_to(root) for p in paths),'CPU fixture path escapes its isolated root')
    else:
        expected_registry=parallel_profile.REGISTRY if auth.get('parallel_profile') is True else PROJECT/'cooperative_gpu0_20261103_v2/live_registry'
        require(layout==NATIVE_LAYOUT and Path(auth['registry_path']).resolve()==expected_registry,'native GPU0 root/registry differs')
    source_closure(auth,layout)
    return auth,hashlib.sha256(raw).hexdigest()

def load_pinned(path,expected,name):
    path=Path(path).resolve();require(sha(path)==expected,'import source pin differs: '+str(path))
    existing=sys.modules.get(name)
    if existing is not None:
        require(Path(existing.__file__).resolve()==path,'imported module origin differs: '+name)
        return existing
    spec=importlib.util.spec_from_file_location(name,path);module=importlib.util.module_from_spec(spec)
    sys.modules[name]=module
    try:spec.loader.exec_module(module)
    except BaseException:sys.modules.pop(name,None);raise
    return module

def modules(auth):
    pins=auth['files_SHA256']
    controller=load_pinned(CONTROLLER,CONTROLLER_SHA,'alz_gpu0_unchanged_budget_controller')
    admission=load_pinned(GUARD/'gpu0_admission.py',pins[str(GUARD/'gpu0_admission.py')],'gpu0_admission')
    guard=load_pinned(GUARD/'gpu0_runner.py',pins[str(GUARD/'gpu0_runner.py')],'alz_gpu0_guard_v2')
    if auth.get('parallel_profile') is True and device(auth)==GPU1:
        ap=HERE.parent/'recovery_runtime/gpu1_admission.py'
        admission=load_pinned(ap,pins[str(ap)],'gpu1_admission')
        path=HERE.parent/'recovery_runtime/gpu1_runner.py'
        guard=load_pinned(path,pins[str(path)],'alz_owned_gpu1_parallel_guard')
    elif auth.get('parallel_profile') is True:
        path=HERE.parent/'recovery_runtime/parallel_runner.py'
        guard=load_pinned(path,pins[str(path)],'alz_owned_gpu0_parallel_guard')
    return controller,admission,guard

def core_envelope(state,prospective_seconds=0):
    keys={allocation_key({'fold':f,'method':m,'seed':s}) for f in range(3) for m in ('pg','scvi') for s in range(5)}
    records=[r for r in state.get('allocation_history',[])+state.get('open_reservations',[]) if r['allocation_id'].startswith(('ALZ_CORE_V1_','ALZ_CAL_V1_'))]
    require(len({r['allocation_id'] for r in records})==len(records),'duplicate scientific core allocation history')
    require(not any(r.get('status')=='uncertain' for r in records),'uncertain core work blocks new admission')
    total=sum(r.get('charged_GPU_seconds',0) if 'charged_GPU_seconds' in r else r['remaining_wall_seconds']*r['device_count'] for r in records)+prospective_seconds
    require(total<=180*3600,'charged plus reserved core work exceeds180GPU-hour envelope')
    return total

def unseen(auth,layout,state):
    value=cell(auth);key=allocation_key(value)
    require(not any(r['allocation_id']==key for r in state.get('allocation_history',[])+state.get('open_reservations',[])),'scientific matrix cell already consumed/reserved, including failed or uncertain history')
    legacy=layout.science/'runs'/f"f{value['fold']}_{value['method']}_s{value['seed']}"
    require(not legacy.exists() and not output_path(layout,value).exists() and not ledger_path(layout,value).exists(),'existing GPU1/GPU0 scientific output or ledger; inspect instead of duplicate')

def require_live(auth_path,layout=NATIVE_LAYOUT):
    """Read-only child gate, called before site packages or model imports."""
    auth,_=verify(auth_path,layout)
    required=['ALZ_PROJECT_RESERVATION_TOKEN','ALZ_PROJECT_BUDGET_ROOT','ALZ_BUDGET_LEDGER','COOPERATIVE_GPU_TOKEN']
    require(all(os.environ.get(k) for k in required),'no live canonical reservation/cooperative lease; refuse before model import')
    require(os.environ['ALZ_PROJECT_BUDGET_ROOT']==str(layout.project) and os.environ['ALZ_BUDGET_LEDGER']==auth['ledger_path'],'live project/ledger binding differs')
    require(os.environ.get('CUDA_VISIBLE_DEVICES')==('' if layout.fixture_mode else device(auth)),'live device visibility differs')
    controller,admission,_=modules(auth)
    state=read(layout.project/'project_gpu_budget.json');core_envelope(state)
    backend=load_pinned(HERE/'project_runner_gpu0.py',auth['files_SHA256'][str(HERE/'project_runner_gpu0.py')],'project_runner_gpu0')
    backend.controller_gate(state,auth)
    require(state['future_accounting_policy']['current_launch_admission'] is True and state['execution_controller']['status']=='REVIEWED_FOR_BOUNDED_EXECUTION','canonical project admission remains held')
    rows=[r for r in state['open_reservations'] if r['token']==os.environ['ALZ_PROJECT_RESERVATION_TOKEN']]
    require(len(rows)==1,'exact live canonical reservation absent')
    reservation=rows[0];ledger=read(auth['ledger_path']);jobs=ledger.get('jobs',[])
    require(len(jobs)==1,'one fresh owned resource attempt required')
    job=jobs[0];limit=time.monotonic()+2
    while 'child_identity' not in job and time.monotonic()<limit:
        time.sleep(.01);job=read(auth['ledger_path'])['jobs'][0]
    require(reservation['allocation_id']==auth['allocation_id'] and reservation['status']=='reserved' and reservation['gpu_uuids']==[device(auth)] and reservation['device_count']==1 and reservation['declared_wall_seconds']==reservation['job_cap_seconds']==auth['cap_seconds'],'live scientific cell/device/deadline differs')
    require(reservation['label']==auth['label'] and reservation['machine_id']==auth['machine_id'] and reservation['ledger_path']==str(Path(auth['ledger_path']).relative_to(layout.project)),'live allocation label/machine/ledger differs')
    require(job['guard_identity']==reservation['owner'] and admission.identity(reservation['owner']['pid'])==reservation['owner'] and job['status']=='active' and job['role']==auth['role'] and job['lease_token']==os.environ['COOPERATIVE_GPU_TOKEN'] and ledger['cap_seconds']==auth['cap_seconds'] and ledger['gpu_uuid']==device(auth),'live owned guardian/lease/ledger differs')
    require(job['label']==auth['label'] and job['command']==auth['command'],'live exact worker command/label differs')
    registry=read(Path(auth['registry_path'])/'registry.json');lease=registry['jobs'].get(job['lease_token'])
    require(registry.get('schema')==2 and registry.get('gpu')==device(auth) and registry.get('policy_sha256')==auth['policy_SHA256'],'physical GPU0 registry/policy differs')
    require(lease is not None and lease['owner']==reservation['owner'] and lease['worker']==job.get('child_identity') and lease['status']=='running','bound physical GPU0 lease absent')
    require(lease['role']==auth['role'] and lease['request']==read(auth['policy_path'])['roles'][auth['role']],'live resource role/request differs')
    pid=os.getpid();bound=False
    while pid>1:
        if admission.identity(pid)==job.get('child_identity'):bound=True;break
        try:pid=int(Path(f'/proc/{pid}/stat').read_text().rsplit(')',1)[1].split()[1])
        except FileNotFoundError:break
    require(bound,'worker is outside the registered owned session')
    for peer in auth['protected_peers']:require(admission.protected_identity(peer['identity']['pid'])==peer['identity'],'protected peer identity changed before model import')
    value=cell(auth)
    require(not output_path(layout,value).exists() and not (layout.science/'runs'/f"f{value['fold']}_{value['method']}_s{value['seed']}").exists(),'scientific output already exists before model import')
    return auth,{'allocation_id':auth['allocation_id'],'authorization_expiry':auth['expires_unix'],'ledger_started_unix':job['started_unix']}
