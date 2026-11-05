"""One separately declared completed20k->40k native extension, after live binding."""
import argparse
import os
from pathlib import Path
import socket
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parent))
import binding as b


def validate_curve_checkpoint(q, auth, updates):
    b.require(type(updates) is int and updates in (30000, 40000), 'Predeclared30k/40k extension checkpoint required')
    b.require(q.get('schema') == 1 and q.get('method') == 'pg' and q.get('native_recipe') == 'bispf_optimized', 'Released extension schema differs')
    b.require(q.get('completed_updates') == q.get('policy', {}).get('completed_steps') == updates and q.get('fixed_endpoint') == 40000, 'Extension clock or external endpoint differs')
    b.require(q.get('initial_completed_updates') == 20000 and q.get('native_recipe_schedule_horizon') == 20000, 'Original initial clock or recipe horizon differs')
    b.require(q.get('recipe', {}).get('total_steps') == 20000 and q['policy'].get('recipe') == q['recipe'] and q['policy'].get('device') == 'cuda:0', 'Native recipe/device metadata was rewritten')
    b.require(q.get('status') == ('completed' if updates == 40000 else 'diagnostic'), 'Intermediate extension checkpoint cannot stand in for40k')
    b.require(q.get('fold') == auth['cell']['fold'] and q.get('seed') == auth['cell']['seed'], 'Extension fold/seed differs')
    b.require(q.get('cache_manifest_SHA256') == b.sha(Path(auth['cache_dir']) / 'manifest.json'), 'Checkpoint cache differs')
    b.require(q.get('released_hyperparameters') == auth['released_hyperparameters'] and q.get('variant_config') == auth['variant_config'], 'Checkpoint parameters/architecture differ')


def validate_reference(q, auth):
    # The existing private controller/lease integration fixtures retain their
    # synthetic20k PG contract; native extension inputs always use40k.
    if auth.get('cpu_fixture') is True:
        value = b.cell(auth)
        b.require(q.get('schema') == 1 and q.get('method') == 'pg', 'Fixture schema/method differs')
        b.require(q.get('fold') == value['fold'] and q.get('seed') == value['seed'], 'Fixture cell differs')
        b.require(q.get('status') == 'completed' and q.get('completed_updates') == q.get('fixed_endpoint') == 20000, 'Complete confined fixture required')
        b.require(q.get('cache_manifest_SHA256') == b.sha(Path(auth['cache_dir']) / 'manifest.json'), 'Fixture cache differs')
        return
    validate_curve_checkpoint(q, auth, 40000)


def run_scientific(auth, admission, layout):
    b.require(auth['operation'] == 'extend_released_hyperparameters', 'Only the separately declared40k extension is allowed')
    b.require(Path(sys.executable).absolute() == b.runtime_path(layout, b.cell(auth)).absolute() and sys.version_info[:2] == (3, 12), 'Exact native runtime required')
    sys.path.extend(map(str, b.native_site_roots(b.cell(auth), auth['files_SHA256'])))
    root = b.HERE.parent
    sys.path[:0] = [str(root / 'analysis/released_extension40k'), str(root / 'analysis/released_tuning'),
                    str(root / 'model'), str(layout.science), str(root)]
    def deny(self, address):
        raise RuntimeError('Scientific worker network disabled')
    socket.socket.connect = deny;socket.socket.connect_ex = deny
    os.environ.update(WANDB_MODE='disabled', WANDB_DISABLED='true', OTEL_SDK_DISABLED='true', DO_NOT_TRACK='1', HF_HUB_OFFLINE='1', JAX_PLATFORMS='cpu', CUBLAS_WORKSPACE_CONFIG=':4096:8')
    import torch
    torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True);torch.set_float32_matmul_precision('highest')
    torch.cuda.set_per_process_memory_fraction((3 * 2**30) / torch.cuda.get_device_properties(0).total_memory, 0)
    from calibration import evaluate, average, dump, sha
    from generation import iter_pg
    from state_fingerprint import fingerprint
    import released_update as update
    import released_adapter as model
    import extension
    b.require(Path(extension.__file__).resolve() == root / 'analysis/released_extension40k/extension.py', 'Qualified extension import origin differs')
    for module, name in ((update, 'released_update.py'), (model, 'released_adapter.py')):
        b.require(Path(module.__file__).resolve() == root / 'analysis/released_tuning' / name, 'Frozen model import origin differs')
    update.configure(auth['variant_config']);update.configure_hyperparameters(auth['released_hyperparameters'])
    out = Path(auth['output_dir']);out.mkdir(parents=True, exist_ok=False)
    end = time.monotonic() + max(0, min(admission['authorization_expiry'], admission['ledger_started_unix'] + auth['cap_seconds']) - time.time() - 120)
    try:
        ctx = model.Context(Path(auth['cache_dir']), 'cuda:0')
        bundle, final_path = extension.fit(ctx, out / 'fit', auth['cell']['seed'], auth, end_monotonic=end)
        b.require(bundle[-1].completed_steps == 40000 and bundle[0].total_steps == 20000, 'Full external40k endpoint and unchanged20k recipe horizon required before new calibration')
        curve, reloads = [], []
        for updates in (30000, 40000):
            path = final_path if updates == 40000 else out / 'fit' / ('checkpoint%05d.pt' % updates)
            q = torch.load(path, map_location='cpu', weights_only=False)
            validate_curve_checkpoint(q, auth, updates)
            if updates == 40000:
                validate_reference(q, auth)
            b.require(q['recipe'] == q['policy']['recipe'] == bundle[0].to_dict(), 'Original native recipe differs')
            saved = fingerprint(q['policy']);bundle[-1].load_state_dict(q['policy'])
            b.require(saved == fingerprint(bundle[-1].state_dict()), 'Public native CUDA reload above schedule horizon differs')
            reloads.append({'updates': updates, 'checkpoint_path': str(path), 'checkpoint_SHA256': sha(path),
                            'policy_fingerprint': saved, 'native_device_metadata_preserved': q['policy']['device'],
                            'native_recipe_schedule_horizon': 20000, 'public_CUDA_reload_byte_exact': True})
            del q
            def iterator(schedule, conditions, seed):
                for item in iter_pg(ctx, bundle[1], bundle[3], schedule, conditions, seed):
                    if time.monotonic() >= end:
                        raise TimeoutError('Finite extension allocation ending; no partial score')
                    yield item
            destination = out if updates == 40000 else out / 'diagnostics' / ('u%05d' % updates)
            rows = [evaluate(ctx, 'pg_released_params', auth['cell']['seed'], rep, iterator, destination / f'r{rep}') for rep in (0, 1)]
            curve.append({'completed_updates': updates, 'checkpoint_path': str(path), 'checkpoint_SHA256': sha(path),
                          'diagnostic_only': updates != 40000, 'eligible_for_selection': updates == 40000,
                          'metrics': average(rows), 'replicate_directories': [str(destination / f'r{rep}') for rep in (0, 1)]})
        dump(out / 'native_policy_reload_receipts.json', reloads)
        dump(out / 'convergence_curve.json', {'status': 'COMPLETE_PREDECLARED_30K_40K_EXTENSION_CALIBRATION_CURVE',
            'candidate_id': auth['tuning_candidate_id'], 'extension_training_finished_before_new_calibration': True,
            'parent_calibration_informed_extension': True, 'parent_result_path': auth['extension']['parent_result_path'],
            'parent_result_SHA256': auth['extension']['parent_result_SHA256'], 'best_checkpoint_selection': False, 'entries': curve})
        result = dict(curve[-1]['metrics'])
        result.update(status='completed', completed_updates=40000, fixed_endpoint=40000, integer_count_contract_passed=True,
            allocation_id=admission['allocation_id'], operation=auth['operation'], checkpoint_path=str(final_path), checkpoint_SHA256=sha(final_path),
            variant_config=auth['variant_config'], released_hyperparameters=auth['released_hyperparameters'], tuning_candidate_id=auth['tuning_candidate_id'],
            method='pg_released_params', hybrid_config=None, qualification_only=False, new_model_updates=20000, initial_completed_updates=20000,
            native_recipe_schedule_horizon=20000, parent_checkpoint_SHA256=auth['extension']['checkpoint_SHA256'], automatic_successor=False,
            generation='direct_iter_pg', real_bag_replacements=0, pure_only=True, native_recipe='bispf_optimized', pure_formulation='released_horizon40k',
            preset_protocol_SHA256=auth['preset_protocol_SHA256'], extension_protocol_SHA256=auth['extension_protocol_SHA256'],
            extension=auth['extension'], resumed_same_model=True,
            native_parent_restore_path=str(out / 'fit/native_parent_restore.json'), native_parent_restore_SHA256=sha(out / 'fit/native_parent_restore.json'),
            convergence_curve_path=str(out / 'convergence_curve.json'), convergence_curve_SHA256=sha(out / 'convergence_curve.json'),
            native_reload_path=str(out / 'native_policy_reload_receipts.json'), native_reload_SHA256=sha(out / 'native_policy_reload_receipts.json'),
            learning_curve_path=str(out / 'fit/learning_curve.jsonl'), learning_curve_SHA256=sha(out / 'fit/learning_curve.jsonl'))
        dump(out / 'completed.json', result)
    except BaseException as error:
        dump(out / 'failure.json', {'status': 'incomplete', 'error': repr(error), 'automatic_retry': False})
        raise


def main():
    parser = argparse.ArgumentParser();parser.add_argument('--authorization', type=Path, required=True);parser.add_argument('--cpu-fixture-root', type=Path)
    args = parser.parse_args();layout = b.Layout.fixture(args.cpu_fixture_root) if args.cpu_fixture_root else b.NATIVE_LAYOUT
    auth, admission = b.require_live(args.authorization, layout)
    import project_runner_gpu0
    project_runner_gpu0.controller_gate(b.read(layout.project / 'project_gpu_budget.json'), auth)
    if layout.fixture_mode:
        b.require('torch' not in sys.modules and os.environ.get('CUDA_VISIBLE_DEVICES') == '', 'CPU fixture exposed model or CUDA')
        if auth.get('fixture_fail_exit'):
            return 7
        out = Path(auth['output_dir']);out.mkdir(parents=True, exist_ok=False)
        (out / 'CPU_FIXTURE_COMPLETE.json').write_text(__import__('json').dumps({'status': 'CPU_BINDING_FIXTURE_ONLY', 'allocation_id': auth['allocation_id'], 'attempt_id': auth['attempt_id'], 'model_imported': False, 'CUDA_VISIBLE_DEVICES': os.environ['CUDA_VISIBLE_DEVICES']}) + '\n')
        return 0
    return run_scientific(auth, admission, layout)


if __name__ == '__main__':
    raise SystemExit(main())
