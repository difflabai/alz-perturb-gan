"""Side-by-side 0.9.7 vs native EMPF recomputation of the README's headline biology (PROTOCOL.md part C).

The frozen 0.9.7 numbers are recomputed from the immutable stage-replicate summaries and must
reproduce the published values first. The EMPF numbers use the Monte Carlo replicates written
by generate_cells.py. Nothing in docs/ or earlier evidence is modified.
"""
from __future__ import annotations

import os
os.environ.update(OMP_NUM_THREADS='3', OPENBLAS_NUM_THREADS='3', MKL_NUM_THREADS='3')
import argparse
import json
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import common as C  # noqa: E402

STATE097 = C.ROOT / 'artifacts/primamente-t005-gene-inference-recovery-v1-20261108'
RANKING097 = C.ROOT / 'analysis/primamente_gene_inference_exploratory_ranking_20261108.json'
PROFILES097 = C.ROOT / 'docs/evidence/report-data.json'
FOCUS = ['CLEC5A', 'PPARG', 'CDKN1A', 'SPP1', 'GPNMB', 'LPL', 'PLIN2']
PUBLISHED = dict(model_effect=0.454, real_effect=0.489, spearman=0.502, direction=0.659)


def finite(x):
    return None if x is None or not np.isfinite(x) else float(x)


def agreement(model, real):
    ok = np.isfinite(model) & np.isfinite(real)
    return dict(genes=int(ok.sum()), descriptive_spearman=C.spearman(model, real),
                direction_agreement=C.direction_agreement(model, real),
                model_to_real_effect_L2_ratio=finite(np.linalg.norm(model[ok]) / np.linalg.norm(real[ok])))


def run097():
    completion = json.loads((STATE097 / 'completed.json').read_text())
    with np.load(STATE097 / 'real_training_donor_summaries.npz', allow_pickle=False) as z:
        genes, sums, n, stages, donors = z['genes'].astype(str), z['sums'], z['n'], z['stages'], z['donors'].astype(str)
    real, used = C.within_contrast_observed(sums, n, stages, groups=list(range(1, 7)))
    pooled_real, _ = C.within_contrast_observed(sums, n, stages, groups=[0])
    model = np.full((5, 8, 4, 7, len(genes)), np.nan)
    counts = np.zeros((5, 8, 4, 7), np.int64)
    for row in completion['stage_replicates']:
        p = Path(row['path'])
        if C.sha(p) != row['SHA256']:
            raise ValueError(f'frozen replicate changed: {p}')
        with np.load(p, allow_pickle=False) as z:
            counts[row['seed'], row['replicate'], row['stage']] = z['n']
            model[row['seed'], row['replicate'], row['stage']] = z['sums'] / np.maximum(z['n'][:, None], 1)

    def model_effects(groups):
        means = model[:, :, :, groups, :].mean(3)
        valid = (counts[:, :, :, groups] >= C.MIN_SYN_CELLS).all(3)
        eff = np.where((valid[:, :, 3] & valid[:, :, 0])[:, :, None], means[:, :, 3] - means[:, :, 0], np.nan)
        return eff, means, valid
    eff, means, valid = model_effects(used)
    seedmean = np.nanmean(eff, axis=1)
    avg = seedmean.mean(0)
    peff, pmeans, _ = model_effects([0])
    pavg = np.nanmean(peff, axis=1).mean(0)
    profile_model = np.where(valid.all(1)[:, :, None], means.mean(1), np.nan)  # [seed,stage,G]
    profile_real, eligible = C.observed_stage_means(sums, n, stages)
    return dict(genes=genes, donors=donors, stages=stages, sums=sums, n=n, real=real, groups=used,
                model=avg, seed_effects=seedmean, replicate_effects=eff, pooled_real=pooled_real,
                pooled_model=pavg, profile_model=profile_model.mean(0), profile_model_by_seed=profile_model,
                profile_real=profile_real, eligible=eligible)


def load_replicates(paths):
    reps = []
    for p in paths:
        with np.load(p, allow_pickle=False) as z:
            reps.append({k: z[k] for k in ('subtype_sums', 'subtype_n', 'stage', 'targets', 'genes', 'replicate',
                                           'subtype_conf', 'is_heldout', 'mode')})
        receipt = json.loads(Path(p).with_suffix('.receipt.json').read_text())
        if receipt['output_sha256'] != C.sha(p):
            raise ValueError(f'replicate receipt mismatch: {p}')
    genes = reps[0]['genes'].astype(str)
    if any(not np.array_equal(r['genes'].astype(str), genes) or not np.array_equal(r['targets'], reps[0]['targets'])
           for r in reps):
        raise ValueError('replicates differ in genes or targets')
    if len({str(r['mode']) for r in reps}) != 1:
        raise ValueError('replicates use different readouts')
    return reps, genes


def run_empf(reps, observed, donors33):
    stage = reps[0]['stage']
    within = []; pooled = []; cells = []
    for r in reps:
        m, c = C.synthetic_stage_means(r['subtype_sums'], r['subtype_n'], stage)
        within.append(m); cells.append(c)
        # pooled: equal target weight (every target has M*K cells)
        per_target = r['subtype_sums'][:, 0] / r['subtype_n'][:, 0, None]
        pooled.append(np.stack([per_target[stage == s].mean(0) for s in range(4)]))
    within, pooled, cells = np.stack(within), np.stack(pooled), np.stack(cells)
    eff = within[:, 3] - within[:, 0]
    peff = pooled[:, 3] - pooled[:, 0]
    refs = {}
    names = observed['donors']
    sets = {'same_33_donors_as_frozen_report': np.isin(names, donors33),
            'fold0_GAN_TRAIN_25_donors': ~observed['is_heldout'],
            'all_37_development_donors': np.ones(len(names), bool),
            'fold0_heldout_12_donors': observed['is_heldout']}
    for label, keep in sets.items():
        sums, n, st = observed['subtype_sums'][keep], observed['subtype_n'][keep], observed['stage'][keep]
        real, used = C.within_contrast_observed(sums, n, st, groups=list(range(1, 7)))
        preal, _ = C.within_contrast_observed(sums, n, st, groups=[0])
        prof, elig = C.observed_stage_means(sums, n, st)
        refs[label] = dict(real=real, groups=used, pooled_real=preal, profile=prof, eligible=elig.tolist(),
                           donors_by_stage=np.bincount(st, minlength=4).tolist(), sums=sums, n=n, stage=st)
    def effects_for(groups):
        out = []
        for r in reps:
            m, _ = C.synthetic_stage_means(r['subtype_sums'], r['subtype_n'], stage, groups=groups)
            out.append(m[3] - m[0])
        return np.stack(out)
    return dict(within=within, pooled=pooled, cells=cells, effects=eff, pooled_effects=peff, effects_for=effects_for,
                model=eff.mean(0), pooled_model=peff.mean(0), profile=np.nanmean(within, axis=0),
                refs=refs, classifier_confidence=float(np.mean([r['subtype_conf'].mean() for r in reps])))


def gene_row(j, model, reps, real, boot):
    row = dict(model_effect=finite(model[j]))
    if reps is not None:
        v = reps[:, j]; v = v[np.isfinite(v)]
        row.update(mc_replicates=len(v), mc_sd=finite(v.std(ddof=1)) if len(v) > 1 else None,
                   mc_min=finite(v.min()) if len(v) else None, mc_max=finite(v.max()) if len(v) else None,
                   mc_sign_agreement_with_observed=(finite((np.sign(v) == np.sign(real[j])).mean())
                                                    if real is not None and len(v) else None))
    if real is not None:
        row['observed_effect'] = finite(real[j])
    if boot is not None and len(boot):
        row['observed_donor_bootstrap_ci95'] = [finite(np.quantile(boot[:, j], .025)), finite(np.quantile(boot[:, j], .975))]
        row['observed_bootstrap_valid_draws'] = int(len(boot))
    return row


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--replicates', type=Path, nargs='*', default=[])
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--check-097-only', action='store_true')
    a = p.parse_args(argv)
    old = run097()
    j0 = list(old['genes']).index('CLEC5A')
    ranking = json.loads(RANKING097.read_text())
    pub = {r['gene']: r for r in ranking['within_subtype_ranking']}
    pub_model = np.array([np.nan if pub[g]['model_high_minus_NotAD_mean_log1p_panelCP10k'] is None else
                          pub[g]['model_high_minus_NotAD_mean_log1p_panelCP10k'] for g in old['genes']])
    pub_real = np.array([pub[g]['real_high_minus_NotAD_mean_log1p_panelCP10k'] for g in old['genes']])
    reproduction = dict(
        clec5a_model=float(old['model'][j0]), clec5a_real=float(old['real'][j0]),
        max_abs_model_diff_vs_frozen_ranking=float(np.nanmax(np.abs(old['model'] - pub_model))),
        max_abs_real_diff_vs_frozen_ranking=float(np.nanmax(np.abs(old['real'] - pub_real))),
        **{f'within_{k}': v for k, v in agreement(old['model'], old['real']).items()})
    frozen_profile = json.loads(PROFILES097.read_text())['stage_profiles']['CLEC5A']
    reproduction['clec5a_profile_real_matches_report'] = bool(np.allclose(old['profile_real'][:, j0], frozen_profile['real'], atol=1e-7))
    reproduction['clec5a_profile_model_matches_report'] = bool(np.allclose(old['profile_model_by_seed'][:, :, j0], frozen_profile['model_by_seed'], atol=1e-7))
    ok = (round(reproduction['clec5a_model'], 3) == PUBLISHED['model_effect'] and
          round(reproduction['clec5a_real'], 3) == PUBLISHED['real_effect'] and
          round(reproduction['within_descriptive_spearman'], 3) == PUBLISHED['spearman'] and
          round(reproduction['within_direction_agreement'], 3) == PUBLISHED['direction'] and
          reproduction['clec5a_profile_real_matches_report'] and reproduction['clec5a_profile_model_matches_report'])
    reproduction['reproduces_published_values'] = bool(ok)
    print(json.dumps(reproduction, indent=2), flush=True)
    if not ok:
        raise SystemExit('0.9.7 values were not reproduced; stop before side-by-side reporting')
    if a.check_097_only:
        return 0

    reps, genes = load_replicates(a.replicates)
    observed = C.load_observed(bags=False)
    if not np.array_equal(observed['genes'], genes):
        raise ValueError('observed and generated gene axes differ')
    new = run_empf(reps, observed, old['donors'])
    j1 = list(genes).index('CLEC5A')
    shared = sorted(set(genes) & set(old['genes']))
    i_old = np.array([list(old['genes']).index(g) for g in shared])
    i_new = np.array([list(genes).index(g) for g in shared])

    def summary_for(ref_label):
        ref = new['refs'][ref_label]
        if ref['real'] is None:
            return dict(estimable=False, reason='fewer than 3 eligible donors in Not AD or High for every subtype',
                        donors_by_stage=ref['donors_by_stage'], eligible_donors_by_stage_group=ref['eligible'])
        boot = C.bootstrap_observed_contrast(ref['sums'], ref['n'], ref['stage'], ref['groups'])
        # model uses the reference's eligible groups, as in the frozen aggregate rule
        eff = new['effects'] if ref['groups'] == C.GROUPS else new['effects_for'](ref['groups'])
        model = eff.mean(0)
        pboot = C.bootstrap_observed_contrast(ref['sums'], ref['n'], ref['stage'], [0])
        prof = ref['profile'][:, j1]
        return dict(
            estimable=True, groups=[C.TYPES[g - 1] for g in ref['groups']], donors_by_stage=ref['donors_by_stage'],
            eligible_donors_by_stage_group=ref['eligible'],
            model_groups=[C.TYPES[g - 1] for g in ref['groups']],
            CLEC5A_within=gene_row(j1, model, eff, ref['real'], boot),
            CLEC5A_pooled=gene_row(j1, new['pooled_model'], new['pooled_effects'], ref['pooled_real'], pboot),
            CLEC5A_observed_stage_profile=[finite(v) for v in prof],
            CLEC5A_observed_peak=(C.STAGES[int(np.nanargmax(prof))] if np.isfinite(prof).any() else None),
            within_agreement_all_panel_genes=agreement(model, ref['real']),
            within_agreement_shared_genes=agreement(model[i_new], ref['real'][i_new]),
            pooled_agreement_all_panel_genes=agreement(new['pooled_model'], ref['pooled_real']),
            focus_genes={g: gene_row(list(genes).index(g), model, eff, ref['real'], boot)
                         for g in FOCUS if g in genes})

    empf_profile = new['profile'][:, j1]
    old_profile = old['profile_model'][:, j0]
    by_ref = {label: summary_for(label) for label in new['refs']}
    primary = by_ref['same_33_donors_as_frozen_report']
    result = dict(
        status='completed_side_by_side_recomputation_frozen_report_untouched',
        protocol='analysis/query_tool_20261120/PROTOCOL.md',
        units='High minus Not AD difference in mean per-cell log1p(panel CP10k); not a log fold-change',
        caveats=['EMPF: one fit (fold0/seed0, 25 GAN TRAIN donors); fit-to-fit variability unmeasured. 0.9.7: five fits on 33 donors.',
                 'Panels differ (EMPF fold0 HVG panel vs 0.9.7 panel; 1613 shared genes); CP10k uses each panel total.',
                 'Generated subtypes are classifier predictions; observed subtypes are atlas labels.',
                 'Across-gene agreement uses correlated genes; descriptive only, not held-out prediction accuracy.',
                 'Stage-blind empirical resampling predicts zero stage contrast for every gene (reference).'],
        reproduction_097=reproduction,
        model_097=dict(models=5, replicates_per_model=8, panel_genes=len(old['genes']),
                       CLEC5A_within=dict(model_effect=float(old['model'][j0]),
                                          seed_effects=old['seed_effects'][:, j0].tolist(),
                                          replicate_sign_agreement=float(np.mean(np.sign(old['replicate_effects'][:, :, j0]) == np.sign(old['real'][j0]))),
                                          observed_effect=float(old['real'][j0])),
                       CLEC5A_pooled=dict(model_effect=float(old['pooled_model'][j0]), observed_effect=float(old['pooled_real'][j0])),
                       CLEC5A_model_stage_profile=[finite(v) for v in old_profile],
                       CLEC5A_model_peak=C.STAGES[int(np.nanargmax(old_profile))],
                       CLEC5A_observed_stage_profile=[finite(v) for v in old['profile_real'][:, j0]],
                       CLEC5A_observed_peak=C.STAGES[int(np.nanargmax(old['profile_real'][:, j0]))],
                       within_agreement_all_panel_genes=agreement(old['model'], old['real']),
                       within_agreement_single_model_by_seed_posthoc=[agreement(old['seed_effects'][s], old['real'])
                                                                       for s in range(5)],
                       within_agreement_shared_genes=agreement(old['model'][i_old], old['real'][i_old]),
                       pooled_agreement_all_panel_genes=agreement(old['pooled_model'], old['pooled_real'])),
        model_empf=dict(fit='runs/empf_comparison_20261117/fits/native/pbulk/hvg2000/fold0/seed0',
                        readout=str(reps[0]['mode']), models=1, replicates=len(reps),
                        replicate_ids=[int(r['replicate']) for r in reps], panel_genes=len(genes),
                        shared_genes_with_097=len(shared),
                        synthetic_cells_per_stage_group_min=int(new['cells'].min()),
                        generated_classifier_mean_confidence=new['classifier_confidence'],
                        CLEC5A_model_stage_profile=[finite(v) for v in empf_profile],
                        CLEC5A_model_profile_by_replicate=[[finite(v) for v in w[:, j1]] for w in new['within']],
                        CLEC5A_model_peak=(C.STAGES[int(np.nanargmax(empf_profile))] if np.isfinite(empf_profile).any() else None),
                        references=by_ref),
        headline=dict(
            CLEC5A_within_097=dict(model=float(old['model'][j0]), observed=float(old['real'][j0])),
            CLEC5A_within_empf=dict(model=primary.get('CLEC5A_within', {}).get('model_effect'),
                                    observed_same_33_donors=primary.get('CLEC5A_within', {}).get('observed_effect'),
                                    observed_ci95=primary.get('CLEC5A_within', {}).get('observed_donor_bootstrap_ci95'),
                                    mc_sd=primary.get('CLEC5A_within', {}).get('mc_sd'),
                                    mc_sign_agreement=primary.get('CLEC5A_within', {}).get('mc_sign_agreement_with_observed')),
            low_stage_peak_recovered_097=C.STAGES[int(np.nanargmax(old_profile))] == 'Low',
            low_stage_peak_recovered_empf=(bool(C.STAGES[int(np.nanargmax(empf_profile))] == 'Low')
                                           if np.isfinite(empf_profile).any() else None),
            within_spearman_097=reproduction['within_descriptive_spearman'],
            within_direction_097=reproduction['within_direction_agreement'],
            within_spearman_empf_same33=primary.get('within_agreement_all_panel_genes', {}).get('descriptive_spearman'),
            within_direction_empf_same33=primary.get('within_agreement_all_panel_genes', {}).get('direction_agreement')),
        inputs=dict(replicates={str(p): C.sha(p) for p in a.replicates}, state097_completion=C.sha(STATE097 / 'completed.json'),
                    cache_manifest=C.sha(C.CACHE / 'manifest.json')),
        source_sha256=C.sha(__file__), common_sha256=C.sha(C.__file__))
    C.write_json(a.out, result)
    np.savez_compressed(a.out.with_suffix('.npz'), genes=genes, empf_within_effects=new['effects'],
                        empf_pooled_effects=new['pooled_effects'], empf_within_profile=new['within'],
                        observed_same33_within=(new['refs']['same_33_donors_as_frozen_report']['real']
                                                if new['refs']['same_33_donors_as_frozen_report']['real'] is not None else np.full(len(genes), np.nan)))
    print(json.dumps(result['headline'], indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
