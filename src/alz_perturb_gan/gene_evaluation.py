"""Fixed-input zero-update evaluation and four coherent artifact groups."""
from __future__ import annotations
from contextlib import contextmanager
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
import hashlib
import importlib.metadata
import json
import math
import os
import subprocess
import sys
import time
import torch
from .gene_checkpoint import model_hashes, state_hash, zero_update_guard
from .gene_features import file_sha256, json_hash
from .metrics import phenocopy_rank, expression_report


def _tensor_hash(x):
    return hashlib.sha256(x.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def _auc(scores, positive_indices=None, labels=None):
    """Tie-aware rank AUROC, avoiding quadratic cross-target pair matrices."""
    s = scores.detach().cpu().double().flatten()
    if labels is None:
        if scores.ndim != 1: raise ValueError('single-target scores required')
        labels = torch.zeros(len(s), dtype=torch.bool); labels[positive_indices] = True
    y = labels.detach().cpu().bool().flatten()
    npos, nneg = int(y.sum()), int((~y).sum())
    if not npos or not nneg: return None
    order = torch.argsort(s, stable=True); values = s[order]
    _, counts = torch.unique_consecutive(values, return_counts=True)
    end = torch.cumsum(counts, 0); avg = (end.double() + (end - counts).double() + 1) / 2
    ranks = torch.repeat_interleave(avg, counts)
    return float((ranks[y[order]].sum() - npos * (npos + 1) / 2) / (npos * nneg))


def negative_indices(library, target_names, seed):
    """One deterministic keyed negative per target, independent of split/order."""
    output = []
    for name in target_names:
        keyed = int(hashlib.sha256(f'{seed}|{name}'.encode()).hexdigest()[:16], 16) % (2**63 - 1)
        generator = torch.Generator().manual_seed(keyed)
        offset = int(torch.randint(1, len(library), (1,), generator=generator))
        output.append((library.index(name) + offset) % len(library))
    return torch.tensor(output)


def score_summary(scores, target_names, library, negatives, *, probabilities):
    if scores.shape != (len(target_names), len(library)) or not bool(torch.isfinite(scores).all()):
        raise ValueError('finite scored pairs must align to the common library')
    positives = torch.tensor([library.index(n) for n in target_names]); labels = torch.zeros_like(scores, dtype=torch.bool)
    labels[torch.arange(len(positives)), positives] = True
    rows = []
    for i, (name, pos, neg) in enumerate(zip(target_names, positives.tolist(), negatives.tolist())):
        if neg == pos: raise ValueError('negative identity equals its target')
        row = {'target': name, 'positive': library[pos], 'negative': library[neg],
               'auroc': _auc(scores[i], pos), **phenocopy_rank(scores[i], pos)}
        if probabilities:
            pair = scores[i, [pos, neg]]; truth = torch.tensor([1., 0.])
            row.update(balanced_accuracy=float(((pair >= .5) == (truth >= .5)).float().mean()),
                       balanced_brier=float((pair - truth).square().mean()))
        rows.append(row)
    out = {'per_target': rows, 'micro_auroc': _auc(scores, labels=labels),
           'macro_auroc': sum(r['auroc'] for r in rows) / len(rows),
           **{k: sum(r[k] for r in rows) / len(rows) for k in ('top1', 'mrr', 'recall_at_3')},
           'target_count': len(rows), 'candidate_count': len(library), 'full_positive_prior': 1 / len(library)}
    if probabilities:
        if bool(((scores < 0) | (scores > 1)).any()): raise ValueError('invalid head probability')
        out.update(balanced_accuracy=sum(r['balanced_accuracy'] for r in rows) / len(rows),
                   balanced_brier=sum(r['balanced_brier'] for r in rows) / len(rows),
                   full_brier=float((scores - labels.float()).square().mean()),
                   balanced_constant_half_brier=.25, full_constant_prior_brier=(1 / len(library)) * (1 - 1 / len(library)),
                   probability_status='raw sigmoid; no claim of library-prior calibration')
        flat, truth = scores.flatten(), labels.flatten().float(); bins = []
        for lo in range(10):
            mask = (flat >= lo / 10) & ((flat < (lo + 1) / 10) if lo < 9 else (flat <= 1))
            bins.append({'lower': lo / 10, 'count': int(mask.sum()),
                         'mean_score': float(flat[mask].mean()) if mask.any() else None,
                         'positive_rate': float(truth[mask].mean()) if mask.any() else None})
        out['calibration_bins'] = bins
    else: out['calibration'] = 'not_applicable: L2 score is not a probability'
    return out


@contextmanager
def _evaluation_modes(models):
    modes = [(module, module.training) for root in models.modules().values() for module in root.modules()]
    for root in models.modules().values(): root.eval()
    try: yield
    finally:
        for module, mode in modes: module.training = mode


def _make_inputs(models, study, profile_mode, check_budget=lambda: None):
    protocol, device = study.protocol, models.device
    ids = torch.arange(len(protocol.library), device=device)
    control = study.control.to(device).expand(len(ids), -1)
    latent_indices = []
    if profile_mode == 'generated':
        generator = torch.Generator(device=device).manual_seed(protocol.evaluation_seed)
        draws = []
        for _ in range(protocol.draws):
            check_budget()
            z, ix = models.prior.sample(len(ids), generator=generator)
            draws.append(models.generator(z, control, ids)); latent_indices.append(ix.cpu())
        candidates = torch.stack(draws).mean(0).cpu()
    elif profile_mode == 'independent_real_oracle':
        candidates = torch.stack([study.candidate_cells[n].mean(0) for n in protocol.library])
    else: raise ValueError('profile mode must be generated or independent_real_oracle')
    return {'candidates': candidates, 'control': study.control.clone(), 'library': list(protocol.library),
            'gene_ids': list(protocol.gene_ids), 'profile_mode': profile_mode,
            'latent_indices': torch.stack(latent_indices) if latent_indices else torch.empty(0, len(ids), dtype=torch.long),
            'queries': {n: study.query_profiles[n].clone() for n in protocol.library},
            'candidate_cell_ids': study.candidate_ids, 'query_cell_ids': study.query_ids,
            'protocol_sha256': protocol.fingerprint, 'generator_hash': state_hash(models.generator),
            'prior_hash': state_hash(models.prior)}


def evaluate(models, study, *, output=None, profile_mode='generated', fixed_inputs=None, invocation=None, max_wall_seconds=None):
    """No optimizers; one candidate dictionary reused for every target split.

    fixed_inputs enables an explicitly frozen-generator head comparison. It must
    retain exact query/candidate construction and model-generator state hashes.
    """
    if output is not None and Path(output).exists(): raise FileExistsError('new output directory required')
    models.validate_study(study)
    if study.protocol.data_kind != 'synthetic' and not models.checkpoint_provenance:
        raise ValueError('real scoring requires a hash-bound saved checkpoint')
    legacy = models.checkpoint_provenance.get('schema_version') == 'legacy'
    if models.training_protocol_sha256 and models.training_protocol_sha256 != study.protocol.fingerprint:
        raise ValueError('checkpoint FIT/validation protocol differs')
    before = model_hashes(models); started = time.monotonic()
    def check_budget():
        if max_wall_seconds is not None and time.monotonic() - started >= max_wall_seconds:
            raise TimeoutError('fixed evaluation budget exhausted')
    check_budget()
    result = {'protocol_sha256': study.protocol.fingerprint, 'seed': models.config.seed, 'arm': models.config.arm,
              'profile_mode': fixed_inputs['profile_mode'] if fixed_inputs is not None else profile_mode,
              'selection_split': 'inner' if models.training_protocol_sha256 and not legacy else 'none',
              'selection_eligible': bool(models.training_protocol_sha256) and not legacy,
              'legacy_inner_is_not_independent': legacy, 'exploratory_is_not_untouched_test': True, 'splits': {}}
    with zero_update_guard() as guard, _evaluation_modes(models), torch.inference_mode():
        inputs = _make_inputs(models, study, profile_mode, check_budget) if fixed_inputs is None else fixed_inputs
        if inputs['protocol_sha256'] != study.protocol.fingerprint or inputs['library'] != list(study.protocol.library):
            raise ValueError('fixed scoring input protocol/library mismatch')
        if inputs['generator_hash'] != before['generator'] or inputs['prior_hash'] != before['prior']:
            raise ValueError('fixed inputs do not belong to this frozen generator/prior')
        if not torch.equal(inputs['control'], study.control) or inputs['gene_ids'] != list(study.protocol.gene_ids):
            raise ValueError('fixed control or stable gene IDs changed')
        if inputs['candidates'].shape != (len(study.protocol.library),len(study.control)) or not torch.isfinite(inputs['candidates']).all():
            raise ValueError('fixed candidate dimensions/values changed')
        for name in study.protocol.library:
            if set(inputs['candidate_cell_ids'][name]) & set(inputs['query_cell_ids'][name]):
                raise ValueError('oracle candidate/query cells overlap')
            if tuple(inputs['candidate_cell_ids'][name]) != study.candidate_ids[name] or tuple(inputs['query_cell_ids'][name]) != study.query_ids[name]:
                raise ValueError('fixed candidate/query cell partition changed')
            if not torch.equal(inputs['queries'][name], study.query_profiles[name]):
                raise ValueError('fixed scoring target changed')
        candidates = inputs['candidates']; device = models.device
        pairs = {}
        for split, names in study.protocol.groups().items():
            queries = torch.stack([inputs['queries'][n] for n in names])
            matrix = []
            for query in queries:
                check_budget()
                count = len(candidates)
                matrix.append(models.decision(study.control.to(device).expand(count, -1), candidates.to(device),
                    query.to(device).expand(count, -1), torch.arange(count, device=device)).cpu())
            scores = torch.stack(matrix)
            distances = -torch.cdist(queries, candidates)
            negatives = negative_indices(list(study.protocol.library), list(names), study.protocol.negative_seed)
            pairs[split] = {'head_scores': scores, 'l2_scores': distances, 'negative_indices': negatives,
                            'targets': list(names)}
            result['splits'][split] = {'head': score_summary(scores, list(names), list(study.protocol.library), negatives, probabilities=True),
                                      'l2': score_summary(distances, list(names), list(study.protocol.library), negatives, probabilities=False),
                                      'target_tensor_sha256': _tensor_hash(queries),
                                      'negative_identities': [study.protocol.library[i] for i in negatives.tolist()],
                                      'expression': _clean(expression_report(candidates[[study.protocol.library.index(n) for n in names]], queries, study.control))}
            missing = set(models.feature_provenance.get('missing_names', []))
            if 'missing_names' in models.feature_provenance:
                coverage = {}
                for label, flag in (('observed_feature', False), ('missing_feature', True)):
                    selected = [i for i,n in enumerate(names) if (n in missing) == flag]
                    coverage[label] = {'target_count':len(selected)}
                    if selected:
                        subset = [names[i] for i in selected]
                        coverage[label].update(head=score_summary(scores[selected],subset,list(study.protocol.library),negatives[selected],probabilities=True),
                                               l2=score_summary(distances[selected],subset,list(study.protocol.library),negatives[selected],probabilities=False))
                result['splits'][split]['coverage_strata'] = coverage
    after = model_hashes(models)
    if before != after: raise RuntimeError('evaluation changed parameters or buffers')
    result.update(optimizer_step_calls=guard['optimizer_step_calls'], model_hashes_before=before, model_hashes_after=after,
                  candidate_tensor_sha256=_tensor_hash(inputs['candidates']), wall_seconds=time.monotonic() - started,
                  parameter_counts=models.counts(), magnitude_heuristic={'auroc': .65, 'is_success_rule': False})
    check_budget()
    if output is not None: _write_groups(Path(output), models, study, inputs, pairs, result, invocation or {})
    check_budget()
    return {'summary': result, 'inputs': inputs, 'pairs': pairs}


def _clean(value):
    if isinstance(value, dict): return {k:_clean(v) for k,v in value.items()}
    if isinstance(value, (list,tuple)): return [_clean(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value): return None
    return value


def _json(path, value):
    path.write_text(json.dumps(_clean(value), indent=2, sort_keys=True, allow_nan=False) + '\n')


def _write_groups(output, models, study, inputs, pairs, result, invocation):
    if output.exists(): raise FileExistsError('new output directory required; never overwrite baseline receipts')
    output.mkdir(parents=True)
    for group in ('inputs', 'pairs', 'metrics', 'provenance'): (output / group).mkdir()
    torch.save(inputs, output / 'inputs/scored_tensors.pt')
    _json(output / 'inputs/protocol.json', asdict(study.protocol))
    _json(output / 'inputs/perturbation_dictionary.json', {'names': list(study.protocol.library), 'gene_ids': list(study.protocol.gene_ids),
        'index': {n:i for i,n in enumerate(study.protocol.library)}, 'features': models.feature_provenance})
    torch.save(pairs, output / 'pairs/matrices.pt')
    negatives = {split:result['splits'][split]['negative_identities'] for split in pairs}
    _json(output / 'pairs/negative_identities.json', negatives)
    with (output / 'pairs/scores.jsonl').open('w') as stream:
        for split, block in pairs.items():
            for row, name in enumerate(block['targets']):
                for col, candidate in enumerate(study.protocol.library):
                    stream.write(json.dumps({'split':split, 'target':name, 'candidate':candidate, 'label':int(name==candidate),
                        'head_score':float(block['head_scores'][row,col]), 'l2_score':float(block['l2_scores'][row,col])})+'\n')
    _json(output / 'metrics/summary.json', result)
    import alz_perturb_gan, particlegan, numpy
    root = Path(__file__).resolve().parents[2]
    lease_fds = tuple(int(x) for x in os.environ.get('ALZ_LEASE_FDS','').split(',') if x)
    git = subprocess.run(['git','rev-parse','HEAD'],cwd=root,text=True,capture_output=True,pass_fds=lease_fds)
    status = subprocess.run(['git','status','--porcelain'],cwd=root,text=True,capture_output=True,pass_fds=lease_fds)
    files = {str(p.relative_to(root)):file_sha256(p) for p in (root/'src/alz_perturb_gan').glob('*.py')}
    safe_env = {k:os.environ.get(k) for k in ('PYTHONPATH','CUDA_VISIBLE_DEVICES','OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS','PYTHONDONTWRITEBYTECODE','ALZ_LEASE_FDS')}
    manifest = {'schema_version':1, 'completed_utc':datetime.now(timezone.utc).isoformat(), 'argv':list(sys.argv),
        'invocation':invocation, 'environment':safe_env, 'git_sha':git.stdout.strip(), 'git_status':status.stdout.splitlines(),
        'source_sha256':files, 'imports':{'alz_perturb_gan':alz_perturb_gan.__file__, 'particlegan':particlegan.__file__, 'numpy':numpy.__file__, 'torch':torch.__file__},
        'versions':{k:importlib.metadata.version(k) for k in ('torch','numpy','particlegan')},
        'data_provenance':study.source_provenance, 'feature_provenance':models.feature_provenance,
        'checkpoint_provenance':models.checkpoint_provenance,
        'model_config':asdict(models.config), 'device':str(models.device), 'protocol_sha256':study.protocol.fingerprint,
        'zero_optimizer_step_calls':result['optimizer_step_calls'], 'model_hashes':result['model_hashes_after'],
        'groups':['inputs','pairs','metrics','provenance']}
    _json(output/'provenance/manifest.json', manifest)
    _json(output/'provenance/file_hashes.json', {str(p.relative_to(output)):file_sha256(p) for p in sorted(output.rglob('*')) if p.is_file()})


def paired_delta(candidate, baseline, split, *, metric='auroc', draws=1000, seed=800):
    """Paired target bootstrap for a fixed trained seed; no pair independence."""
    a, b = candidate['splits'][split], baseline['splits'][split]
    if candidate.get('profile_mode') != baseline.get('profile_mode'):
        raise ValueError('paired comparisons require the same generated/oracle profile mode')
    if candidate['protocol_sha256'] != baseline['protocol_sha256'] or candidate['seed'] != baseline['seed']:
        raise ValueError('baseline must be paired by protocol and training seed')
    if a['target_tensor_sha256'] != b['target_tensor_sha256'] or a['negative_identities'] != b['negative_identities']:
        raise ValueError('baseline scoring targets/negative identities differ')
    aa, bb = a['head']['per_target'], b['head']['per_target']
    if [r['target'] for r in aa] != [r['target'] for r in bb]: raise ValueError('target order differs')
    delta = torch.tensor([x[metric]-y[metric] for x,y in zip(aa,bb)], dtype=torch.float64)
    rng = torch.Generator().manual_seed(seed)
    estimates = delta[torch.randint(len(delta),(draws,len(delta)),generator=rng)].mean(1)
    return {'metric':metric, 'mean_delta':float(delta.mean()), 'ci95':estimates.quantile(torch.tensor([.025,.975],dtype=torch.float64)).tolist(),
            'unit':'targets; paired within a fixed trained seed', 'targets':[r['target'] for r in aa], 'per_target_delta':delta.tolist()}


def all_seed_deltas(candidates, baselines, split, *, expected_seeds, draws=1000, seed=801):
    """Report every seed; resample seeds and shared target identities hierarchically."""
    if set(candidates)!=set(expected_seeds) or set(baselines)!=set(expected_seeds):
        raise ValueError('all predeclared seeds and their matched baselines are required')
    reports = {s:paired_delta(candidates[s],baselines[s],split,draws=draws,seed=seed) for s in expected_seeds}
    first = reports[expected_seeds[0]]['targets']
    if any(r['targets']!=first for r in reports.values()): raise ValueError('targets differ across seeds')
    delta = torch.tensor([reports[s]['per_target_delta'] for s in expected_seeds],dtype=torch.float64)
    rng=torch.Generator().manual_seed(seed); estimates=[]
    for _ in range(draws):
        si=torch.randint(len(expected_seeds),(len(expected_seeds),),generator=rng)
        ti=torch.randint(len(first),(len(first),),generator=rng)
        estimates.append(delta[si][:,ti].mean())
    ci=torch.stack(estimates).quantile(torch.tensor([.025,.975],dtype=torch.float64)).tolist()
    return {'per_seed':reports,'all_seed_mean_delta':float(delta.mean()),'hierarchical_ci95':ci,
            'unit':'training seeds and shared targets; candidate pairs are not independent',
            'limits':'few seeds estimate training variability poorly; not biological replication or untouched-test evidence',
            'selected_seed':None,'success_by_two_of_three':False}
