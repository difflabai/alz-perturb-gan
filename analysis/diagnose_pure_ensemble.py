"""One preregistered ensemble estimate from saved forecasts; never select a model."""
import datetime, hashlib, json
from pathlib import Path
import numpy as np
from scipy.spatial.distance import cdist

ROOT = Path('/ml2/alz')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def donor_complementarity(p, q):
    assert p.shape == q.shape and p.ndim == 2 and len(p) > 1
    n = len(p); cross = cdist(p, q)
    # Same trace index shares a library query across models; exclude those pairs.
    cross_off = (cross.sum() - np.trace(cross)) / (n * (n - 1))
    return float(2 * cross_off - cdist(p, p).sum() / (n * (n - 1)) - cdist(q, q).sum() / (n * (n - 1)))


def expected_fixed_mask_score(es_p, es_q, distance, n, q_count):
    assert type(q_count) is int and 0 <= q_count <= n and n > 1
    weight = q_count / n
    return (1 - weight) * es_p + weight * es_q - .5 * (q_count * (n - q_count) / (n * (n - 1))) * distance


def cell_complementarity(p, q):
    return float(2 * cdist(p, q).mean() - cdist(p, p).mean() - cdist(q, q).mean())


def main():
    protocol_path = ROOT / 'analysis/pure_ensemble_hypothesis.json'
    protocol = json.loads(protocol_path.read_text()); weights = [m['weight'] for m in protocol['models']]
    assert weights == [.75, .25]
    folders = [ROOT / 'runs/CAL_GPU0_V1_latent8_v1_f0_pg_s0', ROOT / 'runs/CAL_GPU0_V1_learned_obs_v1_f0_pg_s0']
    pins = {str(protocol_path): sha(protocol_path)}; rows = []; component_scores = [[], []]
    for folder, model in zip(folders, protocol['models']):
        path = folder / 'fit/final.pt'; assert sha(path) == model['checkpoint_SHA256']; pins[str(path)] = sha(path)
    for rep in (0, 1):
        arrays = []; metrics = []
        for component, folder in enumerate(folders):
            path = folder / f'r{rep}/arrays.npz'; meta = path.with_name('metrics.json')
            q = json.loads(meta.read_text()); assert q['status'] == 'completed' and sha(path) == q['arrays_SHA256']
            pins.update({str(path): sha(path), str(meta): sha(meta)})
            with np.load(path, allow_pickle=False) as z:
                y = z['stage_synthetic_summaries'].copy(); cells = z['generated_cells_by_stage'].copy()
            assert y.shape == (4, 200, 26) and cells.shape == (4, 375, 50)
            assert np.isfinite(y).all() and np.isfinite(cells).all()
            arrays.append((y, cells)); metrics.append(q); component_scores[component].append(q)
        for key in ['donors', 'stage', 'training_seed', 'generation_replicate', 'sampling_contract', 'calibration_manifest_SHA256', 'trace_manifest_SHA256']:
            assert metrics[0][key] == metrics[1][key]
        assert set(metrics[0]['stage']) == set(range(4))
        donor_distances = [donor_complementarity(arrays[0][0][s], arrays[1][0][s]) for s in range(4)]
        es = expected_fixed_mask_score(metrics[0]['energy_score'], metrics[1]['energy_score'], float(np.mean(donor_distances)), 200, 50)
        cell_distance = cell_complementarity(arrays[0][1].reshape(-1, 50), arrays[1][1].reshape(-1, 50))
        ed = .75 * metrics[0]['cell_energy_distance'] + .25 * metrics[1]['cell_energy_distance'] - .75 * .25 * cell_distance
        rows.append({'replicate': rep, 'donor_between_forecast_U_distance_by_stage': donor_distances,
                     'expected_energy_score_over_uniform_fixed50of200_masks': es,
                     'weighted_empirical_cell_reservoir_energy_distance': ed,
                     'between_forecast_cell_V_distance': cell_distance,
                     'PG8_saved_energy_score': metrics[0]['energy_score'],
                     'PG8_saved_cell_energy_distance': metrics[0]['cell_energy_distance'],
                     'learned_saved_energy_score': metrics[1]['energy_score'],
                     'learned_saved_cell_energy_distance': metrics[1]['cell_energy_distance']})
    p_proof = ROOT / 'analysis/pure_latent8_verification.json'
    q_proof = ROOT / 'analysis/CAL_GPU0_V1_learned_obs_v1_f0_pg_s0-result.json'
    proofs = [json.loads(p_proof.read_text()), json.loads(q_proof.read_text())]
    assert proofs[0]['status'] == 'VERIFIED_PURE_PG_ONLY' and proofs[0]['control_arrays_and_count_exemplars_bitwise_equal']
    assert proofs[1]['independent_saved_scores_checkpoint_and_count_exemplars_verified']
    for i, proof in enumerate([proofs[0], proofs[1]['candidate']]):
        assert proof['checkpoint_SHA256'] == protocol['models'][i]['checkpoint_SHA256']
        for key in ['energy_score', 'cell_energy_distance']:
            np.testing.assert_allclose(np.mean([q[key] for q in component_scores[i]]), proof[key], rtol=0, atol=1e-12)
    pins.update({str(p_proof): sha(p_proof), str(q_proof): sha(q_proof)})
    result = {'status': 'ANALYTIC_PURE_ENSEMBLE_COMPLEMENTARITY_ESTIMATE_NOT_A_MODEL_RESULT',
              'utc': datetime.datetime.now(datetime.timezone.utc).isoformat(), 'registered_weights': weights, 'records': rows,
              'mean_expected_energy_score_over_random_masks': float(np.mean([q['expected_energy_score_over_uniform_fixed50of200_masks'] for q in rows])),
              'mean_weighted_empirical_cell_reservoir_energy_distance': float(np.mean([q['weighted_empirical_cell_reservoir_energy_distance'] for q in rows])),
              'source_checkpoint_array_metrics_protocol_SHA256': pins,
              'scope': 'Already saved generated forecasts and scalar scores only. No calibration or evaluation target arrays loaded; no new generation, fitting, GPU allocation or selection. Independent masks were not scanned.',
              'limits': 'Donor estimate averages over random fixed-count masks, not the one registered native mask. Cell estimate combines empirical saved reservoirs, not a newly drawn mixture reservoir. Sampling error, donor spread and native count/control replay remain unverified. Two20k parents use40k total updates; not equal compute to one20k model.',
              'native_ensemble_result_verified': False, 'incumbents_changed': False}
    result['source_SHA256'] = {str(p): sha(p) for p in [Path(__file__).resolve(), ROOT / 'analysis/test_ensemble_estimate.py']}
    (ROOT / 'analysis/pure_ensemble_complementarity.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(json.dumps({k: result[k] for k in ['status', 'mean_expected_energy_score_over_random_masks', 'mean_weighted_empirical_cell_reservoir_energy_distance', 'native_ensemble_result_verified']}, indent=2))


if __name__ == '__main__':
    main()
