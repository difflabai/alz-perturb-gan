"""Prospectively frozen calibration acceptance rule, with explicit regressions."""
import math


def outside_spread_band(value):
    return max(0.7 - value, 0.0, value - 1.3)


def decide(candidate, incumbent):
    reasons = []
    identity = ['metric_role', 'donors', 'generation_replicates', 'training_seed',
                'calibration_manifest_SHA256', 'trace_manifest_SHA256', 'sampling_contract']
    if any(candidate.get(k) != incumbent.get(k) for k in identity):
        reasons.append('incomparable calibration data or sampling contract')
    if candidate.get('metric_role') != 'calibration_selection':
        reasons.append('evaluation results cannot select a candidate')
    if candidate.get('status') != 'completed' or candidate.get('completed_updates') != 20000:
        reasons.append('incomplete fixed endpoint')
    keys = ['energy_score', 'cell_energy_distance', 'spread_ratio', 'generated_zero_count_rate']
    if any(type(candidate.get(k)) not in (int, float) or not math.isfinite(candidate[k]) for k in keys):
        reasons.append('invalid or nonfinite metric')
    if reasons:
        return {'accept': False, 'reasons': reasons}
    if candidate['energy_score'] >= incumbent['energy_score'] - 1e-9:
        reasons.append('energy score did not improve')
    if candidate['cell_energy_distance'] > incumbent['cell_energy_distance'] + 1e-9:
        reasons.append('cell distribution match worsened')
    if outside_spread_band(candidate['spread_ratio']) > outside_spread_band(incumbent['spread_ratio']) + 1e-9:
        reasons.append('donor spread moved farther outside the acceptable band')
    if not candidate.get('integer_count_contract_passed'):
        reasons.append('count validity not verified')
    return {'accept': not reasons, 'reasons': reasons}
