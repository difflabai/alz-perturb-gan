"""Retrospective stage decomposition of40 existing calibration-array pairs.

No fit, generation, endpoint selection, condition relabeling or metric change.
Cross-stage terms are signed contributions to a pooled distance, not distances
in their own right. All existing seeds/checkpoints and matched controls remain.
"""
import datetime
import hashlib
import itertools
import json
import os
import statistics
import sys
from pathlib import Path

ROOT = Path('/ml2/alz')
PROJECT = Path('/ml2/codex-storage/future-workspaces/2026-10-27/task-2/benchmark')
sys.path[:0] = [str(ROOT),
    str(PROJECT/'current_base_20261029/runtime/lib/python3.12/site-packages'),
    str(PROJECT/'.venv312/lib/python3.12/site-packages')]
import numpy as np
from scipy.spatial.distance import cdist
from calibration import targets


def read(path):
    return json.loads(Path(path).read_bytes())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def blocks(x, y):
    """Equal stage weights, using the same V-statistic as the fixed primary."""
    x, y = np.asarray(x), np.asarray(y)
    assert x.ndim == y.ndim == 3 and x.shape[0] == y.shape[0] and x.shape[2] == y.shape[2]
    assert min(x.shape) > 0 and min(y.shape) > 0 and np.isfinite(x).all() and np.isfinite(y).all()
    k = len(x)
    xy = np.array([[cdist(a, b).mean() for b in y] for a in x])
    xx = np.array([[cdist(a, b).mean() for b in x] for a in x])
    yy = np.array([[cdist(a, b).mean() for b in y] for a in y])
    signed = xy + xy.T - xx - yy
    per_stage = np.diag(signed)
    pooled = float(signed.mean())
    diagonal = float(np.trace(signed) / k**2)
    distances = 2*xy - np.diag(xx)[:, None] - np.diag(yy)[None, :]
    # Relabeling is only a diagnostic enumeration; no output is relabeled.
    assignments = [(float(np.mean([distances[s, p[s]] for s in range(k)])), p)
                   for p in itertools.permutations(range(k))]
    best, permutation = min(assignments)
    return dict(pooled_ED=pooled, per_stage_ED=per_stage.tolist(),
        mean_within_stage_ED=float(per_stage.mean()),
        pooled_diagonal_contribution=diagonal,
        pooled_cross_stage_contribution=pooled-diagonal,
        signed_stage_blocks=signed.tolist(),
        generated_stage_to_target_stage_ED=distances.tolist(),
        lowest_assignment_diagnostic_ED=best,
        lowest_assignment_diagnostic_permutation=list(permutation),
        identity_assignment_is_minimum=bool(np.isclose(best, per_stage.mean(), rtol=0, atol=1e-10)))


def analytic_checks():
    y = np.array([[[-1.], [-1.]], [[1.], [1.]]])
    same, swapped = blocks(y, y), blocks(y[::-1], y)
    assert same['pooled_ED'] == same['mean_within_stage_ED'] == 0.
    assert swapped['pooled_ED'] == 0. and swapped['mean_within_stage_ED'] == 4.
    assert swapped['pooled_diagonal_contribution'] == 2. and swapped['pooled_cross_stage_contribution'] == -2.
    assert swapped['lowest_assignment_diagnostic_permutation'] == [1, 0]
    return 2


def average(records):
    fields = ('pooled_ED', 'mean_within_stage_ED', 'pooled_diagonal_contribution',
              'pooled_cross_stage_contribution', 'per_stage_ED', 'signed_stage_blocks')
    return {k: np.mean([r[k] for r in records], axis=0).tolist() for k in fields}


def plot(result, stem):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    plt.rcParams.update({'font.family':'DejaVu Sans', 'font.size':10,
        'axes.spines.top':False, 'axes.spines.right':False, 'svg.fonttype':'none'})
    fig, axes = plt.subplots(1, 3, figsize=(14.5, 6.1))
    fig.subplots_adjust(left=.065, right=.925, top=.77, bottom=.30, wspace=.42)
    x = [5000, 10000, 20000]
    for ax, key, title in zip(axes[:2], ('pooled_ED', 'mean_within_stage_ED'),
                            ('Pooled cell ED: fixed primary', 'Mean stage-matched ED: diagnostic')):
        for seed in range(5):
            values = [result['by_seed_and_checkpoint'][str(seed)][str(step)][key] for step in x]
            ax.plot(x, values, c='#186fa5', alpha=.25, lw=1)
        means = [result['checkpoint_means'][str(step)][key] for step in x]
        ax.plot(x, means, c='#186fa5', marker='o', lw=2)
        baseline = result['matched_resampling_mean'][key]
        ax.axhline(baseline, c='#c27724', ls='--', lw=1.8)
        for a, b in zip(x, means):
            ax.annotate(f'{b:.4f}', (a,b), xytext=(0,8), textcoords='offset points', ha='center', fontsize=9)
        ax.set_xticks(x, ['5k','10k','20k']);ax.set_xlim(3000,22500)
        ax.set_xlabel('Existing saved training updates');ax.set_ylabel('Energy distance (lower is better)')
        ax.set_title(title, fontsize=11);ax.grid(axis='y', alpha=.15)
        bottom, top = ax.get_ylim();ax.set_ylim(bottom-.04*(top-bottom),top+.18*(top-bottom))
    delta = np.asarray(result['twenty_k_PG_minus_resampling_weighted_stage_blocks'])
    vmax = float(abs(delta).max())
    im = axes[2].imshow(delta, cmap='RdBu_r', vmin=-vmax, vmax=vmax)
    for s in range(4):
        for t in range(4):
            axes[2].text(t,s,f'{delta[s,t]:+.4f}',ha='center',va='center',fontsize=9,
                color='white' if abs(delta[s,t])>.65*vmax else '#152231')
    axes[2].set_xticks(range(4));axes[2].set_yticks(range(4))
    axes[2].set_xlabel('Stage');axes[2].set_ylabel('Stage')
    axes[2].set_title('20k PG minus resampling\nSigned contributions to pooled ED',fontsize=11)
    fig.colorbar(im, ax=axes[2], fraction=.046, pad=.04)
    fig.suptitle('Pooled and stage-matched calibration distances give different rankings',x=.065,y=.96,ha='left',fontsize=16)
    fig.text(.065,.85,'Means across five existing PG initializations; two CPU sampling replicates per checkpoint. All checkpoints retained.',fontsize=11,color='#3c4b59')
    fig.legend(handles=[Line2D([],[],c='#186fa5',lw=2,marker='o',label='ParticleGAN: five-seed mean'),
        Line2D([],[],c='#186fa5',lw=1,alpha=.25,label='Individual existing seeds'),
        Line2D([],[],c='#c27724',lw=1.8,ls='--',label='Matched resampling: five-seed mean')],
        loc='lower center',bbox_to_anchor=(.5,.15),ncol=3,frameon=False)
    fig.text(.065,.085,'Four adaptively reused calibration donors, one per stage. Signed cross-stage terms can cancel stage-specific errors.',fontsize=10,color='#3c4b59')
    fig.text(.065,.035,'No new training, generation, relabeling, endpoint selection or primary-metric change. No independent biological or SOTA claim.',fontsize=10,color='#3c4b59')
    paths = [stem.with_suffix('.png'),stem.with_suffix('.svg')]
    assert not any(p.exists() for p in paths)
    for p in paths:
        fig.savefig(p,dpi=150,facecolor='white')
    plt.close(fig)
    return {str(p):sha(p) for p in paths}


def main():
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == ''
    output = ROOT/'analysis/t005_stage_mixture_energy_CPU_result_20261108.json'
    assert not output.exists()
    checks = analytic_checks()
    source = ROOT/'analysis/t005_checkpoint_CPU_multiseed_result_20261108.json'
    old = read(source)
    assert old['training_seeds'] == list(range(5)) and old['saved_updates'] == [5000,10000,20000]
    assert old['new_training_fits'] == old['new_optimizer_updates'] == 0 and not old['external_evaluation_target_values_read']
    manifest, ys, cells = targets()
    try:
        real = cells['by_stage'].copy()
        assert np.array_equal(real.reshape(-1,50),cells['equal_stage'])
        donors = dict(zip(map(str,ys['stage'].tolist()),ys['donors'].tolist()))
    finally:
        ys.close();cells.close()
    records, by_seed, baselines, pins = [], {}, {}, {str(source):sha(source)}
    for seed in range(5):
        old_seed = old['all_seed_records'][str(seed)]
        groups = {}
        for name, pin in old_seed['output_SHA256'].items():
            path = Path(name)
            if path.name != 'arrays.npz':
                continue
            metric_path = path.with_name('metrics.json')
            assert sha(path) == pin and sha(metric_path) == old_seed['output_SHA256'][str(metric_path)]
            metric = read(metric_path)
            assert metric['metric_role'] == 'calibration_selection' and metric['training_seed'] == seed
            assert metric['arrays_SHA256'] == pin
            with np.load(path,allow_pickle=False) as a:
                x = a['generated_cells_by_stage']
                assert x.shape == (4,375,50)
                decomposition = blocks(x,real)
            np.testing.assert_allclose(decomposition['pooled_ED'],metric['cell_energy_distance'],rtol=0,atol=1e-10)
            np.testing.assert_allclose(decomposition['per_stage_ED'],metric['per_stage_cell_energy_distance'],rtol=0,atol=1e-10)
            group = 'resampling' if path.parent.name.startswith('resampling') else str(int(path.parent.name.split('_')[0].removeprefix('update')))
            groups.setdefault(group,[]).append(decomposition)
            records.append(dict(seed=seed,checkpoint_or_control=group,replicate=metric['generation_replicate'],
                arrays_path=str(path),arrays_SHA256=pin,**decomposition))
            pins.update({str(p):sha(p) for p in (path,metric_path)})
        assert set(groups) == {'5000','10000','20000','resampling'} and all(len(v)==2 for v in groups.values())
        by_seed[str(seed)] = {str(step):average(groups[str(step)]) for step in (5000,10000,20000)}
        baselines[str(seed)] = average(groups['resampling'])
        for step, original in zip((5000,10000,20000),old_seed['rows']):
            assert original['completed_updates'] == step
            np.testing.assert_allclose(by_seed[str(seed)][str(step)]['pooled_ED'],original['mean']['cell_energy_distance'],rtol=0,atol=1e-10)
    assert len(records) == 40
    means = {str(step):average([by_seed[str(s)][str(step)] for s in range(5)]) for step in (5000,10000,20000)}
    control = average(list(baselines.values()))
    delta = {k:means['20000'][k]-control[k] for k in ('pooled_ED','mean_within_stage_ED',
             'pooled_diagonal_contribution','pooled_cross_stage_contribution')}
    assert np.isclose(delta['pooled_ED'],delta['pooled_diagonal_contribution']+delta['pooled_cross_stage_contribution'],rtol=0,atol=1e-10)
    stages = np.asarray(means['20000']['per_stage_ED'])-np.asarray(control['per_stage_ED'])
    block_delta = (np.asarray(means['20000']['signed_stage_blocks'])-np.asarray(control['signed_stage_blocks']))/16
    assert np.isclose(block_delta.sum(),delta['pooled_ED'],rtol=0,atol=1e-10)
    result = dict(status='ALL40_EXISTING_CALIBRATION_ARRAYS_STAGE_AND_MIXTURE_ENERGY_DECOMPOSITION_COMPLETE',
        UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),analytic_closed_form_checks=checks,
        arrays_independently_decomposed=40,training_seeds=list(range(5)),saved_updates=[5000,10000,20000],
        stage_to_calibration_donor=donors,by_seed_and_checkpoint=by_seed,matched_resampling_by_seed=baselines,
        checkpoint_means=means,matched_resampling_mean=control,twenty_k_PG_minus_resampling=delta,
        twenty_k_PG_minus_resampling_per_stage_ED=stages.tolist(),
        twenty_k_PG_minus_resampling_weighted_stage_blocks=block_delta.tolist(),
        twenty_k_stage_ED_win_counts_across_five_seeds=[sum(by_seed[str(s)]['20000']['per_stage_ED'][t]<baselines[str(s)]['per_stage_ED'][t] for s in range(5)) for t in range(4)],
        twenty_k_pooled_ED_wins_across_five_seeds=sum(by_seed[str(s)]['20000']['pooled_ED']<baselines[str(s)]['pooled_ED'] for s in range(5)),
        twenty_k_mean_within_stage_ED_wins_across_five_seeds=sum(by_seed[str(s)]['20000']['mean_within_stage_ED']<baselines[str(s)]['mean_within_stage_ED'] for s in range(5)),
        twenty_k_vs10k_mean_within_stage_ED_change=means['20000']['mean_within_stage_ED']-means['10000']['mean_within_stage_ED'],
        array_records=records,
        formula='For equal four-stage weights, pooled ED = mean over16 blocks C_st; C_st=E|X_s-Y_t|+E|X_t-Y_s|-E|X_s-X_t|-E|Y_s-Y_t|. Diagonal contribution = mean within-stage ED /4. Cross-stage contributions are signed and are not distances themselves.',
        interpretation='The fixed pooled primary tests the marginal equal-stage mixture. Mean stage-matched ED tests conditional matching, a distinct diagnostic. One can improve while the other worsens because marginal mixtures discard stage labels. This does not make either computed score invalid.',
        retrospective_calibration_only=True,reused_donors=4,one_calibration_donor_per_stage=True,
        no_new_training_or_generation=True,new_optimizer_updates=0,new_GPU_allocations=0,
        no_condition_relabeling_or_new_endpoint_selection=True,frozen45_and_queued_LR_metric_unchanged=True,
        external_evaluation_target_values_read=False,full_goal_complete=False,SOTA_claim=False,
        limitations='Five existing initializations and sampling replicates are not new biological donors. Four adaptively reused donors confound stage with donor. Assignment enumeration is descriptive only and is not a proposed relabeling, selected result, new primary metric or independent test.',
        source_SHA256={str(Path(__file__).resolve()):sha(__file__),str(ROOT/'calibration.py'):sha(ROOT/'calibration.py')},
        input_SHA256=pins)
    result['figure_SHA256'] = plot(result,ROOT/'analysis/t005_stage_mixture_energy_CPU_20261108')
    assert 'torch' not in sys.modules
    output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:result[k] for k in ('status','arrays_independently_decomposed',
        'twenty_k_PG_minus_resampling','twenty_k_stage_ED_win_counts_across_five_seeds',
        'twenty_k_pooled_ED_wins_across_five_seeds','twenty_k_mean_within_stage_ED_wins_across_five_seeds')}))


if __name__ == '__main__':
    main()
