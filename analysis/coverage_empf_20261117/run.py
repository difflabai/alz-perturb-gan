"""Fixed native EMPF count-readout and disjoint-donor SD calibration pass."""
import os
os.environ.update(OMP_NUM_THREADS='3', OPENBLAS_NUM_THREADS='3', MKL_NUM_THREADS='3',
                  CUDA_VISIBLE_DEVICES='', PYTHONDONTWRITEBYTECODE='1')
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OUT = ROOT/'runs/coverage_empf_20261117'
PIN = HERE/'source_manifest.json'
BASE = ROOT/'runs/microglia_empf/verification/final_generated.npz'
MODE = 'cellshape@0.25:biascal:stagehalf'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    path = Path(path);path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n');tmp.replace(path)


def runtime():
    spec = importlib.util.spec_from_file_location('application_runtime', ROOT/'scripts/microglia_runtime.py')
    R = importlib.util.module_from_spec(spec);spec.loader.exec_module(R)
    return R


def check_sources(R):
    digest = sha(PIN);pin = json.loads(PIN.read_text())
    for name, expected in pin['execution_sources'].items():
        if sha(ROOT/name) != expected:raise ValueError(f'coverage source changed: {name}')
    if sha(R.SOURCE_MANIFEST) != pin['application_manifest_sha256']:
        raise ValueError('application source manifest changed')
    R.check_sources()
    for name, expected in pin['inputs'].items():
        if sha(ROOT/name) != expected:raise ValueError(f'coverage input changed: {name}')
    if sha(PIN) != digest:raise ValueError('coverage source manifest changed during checking')
    return digest


def setup():
    R = runtime();digest = check_sources(R);M = R.application();cfg = M.load_config()
    cache = M.cache_path(cfg, 0);fit = ROOT/cfg['runtime']['selected_fit']
    anchor = R.check_cache(cache, 0, held=True)
    M.check_fit(fit, cfg, 0, 0)
    weights = R.check_weights(fit, json.loads((fit/'fit_receipt.json').read_text()))
    selected = json.loads((R.SELECTION/'selected_model.json').read_text())
    if weights != selected['weights_sha256']:raise ValueError('wrong selected native model')
    for name, expected in selected['metadata_sha256'].items():
        if sha(fit/name) != expected:raise ValueError('selected native fit metadata changed')
    T,G,provenance = R.activate('cpu', .04)
    R.check_checkpoint(T.torch.load(fit/'final.pt',map_location='cpu',weights_only=False),
                       json.loads((fit/'config.json').read_text()))
    def streams(fold,index):
        children=T.np.random.SeedSequence([20261114,6,fold,index,0]).spawn(5)
        return {key:int(child.generate_state(1,dtype=T.np.uint64)[0])%(2**63-16)
                for key,child in zip(G.GEN_STREAMS,children)}
    G.generation_streams = streams
    return R,T,G,cache,fit,anchor,digest,provenance


def finish(R,cache,anchor,digest):
    if check_sources(R) != digest or R.check_cache(cache,0,held=True) != anchor:
        raise ValueError('coverage source or cache changed during execution')


def generate():
    R,T,G,cache,fit,anchor,digest,provenance = setup()
    output = OUT/'quarter_generated.npz'
    if output.exists():raise ValueError('candidate archive already exists; inspect before rerunning')
    _,receipt = G.generate_fit(fit,cache,mode=MODE,out=output,device='cpu')
    finish(R,cache,anchor,digest)
    receipt.update(experiment='fixed_quarter_contraction_native_EMPF',
                   experiment_manifest_sha256=digest, application_manifest_sha256=sha(R.SOURCE_MANIFEST),
                   cache_anchor=anchor, native_provenance=provenance,
                   generation_streams=[20261114,6,0,'development donor index',0],
                   model_weights_unchanged=True,training_updates=0,default_changed=False)
    write(output.with_suffix('.receipt.json'),receipt)
    print(json.dumps({'status':'candidate_generated','sha256':sha(output),'mode':MODE}),flush=True)


def scores(low,high,observed,mask,I):
    result = {}
    for name,sl in [('mean',slice(0,50)),('sd',slice(50,100)),('combined',slice(0,100))]:
        lo,hi,y = low[mask,sl],high[mask,sl],observed[:,mask,sl]
        result[name] = dict(coverage=float(((y>=lo)&(y<=hi)).mean()),
            below=float((y<lo).mean()),above=float((y>hi).mean()),
            width=float((hi-lo).mean()),interval_score=float(I.interval_score(lo,hi,y).mean()))
    return result


def analyze():
    R,T,G,cache,fit,anchor,digest,provenance = setup()
    import numpy as np
    import architecture_score as A
    import interval_metrics as I
    import calibration as C
    real = A.E.load_real(cache)
    observed = A.observed_summaries(real,64)
    train = real.donors[~real.is_heldout].tolist()
    held = np.flatnonzero(real.is_heldout)
    order = sorted(held,key=lambda i:A.E.MX.stable_seed('interval-calibration-split',20261113,str(real.donors[i])))
    halves = np.array_split(order,2)
    rows = {}
    for label,path,mode in [('baseline',BASE,'cellshape@0.5:biascal:stagehalf'),
                            ('quarter',OUT/'quarter_generated.npz',MODE)]:
        receipt_path = path.with_name(path.stem+'.receipt.json')
        receipt = json.loads(receipt_path.read_text())
        if sha(path) != receipt['generated_SHA256'] or receipt['mode'] != mode:
            raise ValueError('generation receipt differs from the analyzed archive')
        if label=='quarter' and receipt['experiment_manifest_sha256'] != digest:
            raise ValueError('candidate generation used different experiment sources')
        row = A.score(path,cache,OUT/f'{label}_raw_score.json')
        generated = A.E.align_generated(A.E.load_generated(path),real)
        sample = A.E.MX.pc_bag_summary(generated['pcs'].astype(np.float32))
        low,high = I.bounds(sample,'percentile')
        lo,hi = low.copy(),high.copy();swaps = []
        for swap in (0,1):
            cal,test = halves[swap],halves[1-swap]
            record = C.fit_sd(low[cal],high[cal],observed[0,cal],real.donors[cal],train)
            changed = observed[0].copy();changed[test]=1e6
            assert record == C.fit_sd(low[cal],high[cal],changed[cal],real.donors[cal],train)
            lo[test],hi[test] = C.apply_sd(low[test],high[test],record,real.donors[test])
            record['binding'] = dict(fit=str(fit.relative_to(ROOT)),
                final_checkpoint_sha256=sha(fit/'final.pt'),source_generated_sha256=sha(path),
                cache_manifest_sha256=anchor['manifest_sha256'],experiment_manifest_sha256=digest,
                coordinate_basis='frozen fold0 TRAIN-fitted PCA50 mean then SD',calibration_real_repeat=0)
            artifact = OUT/f'calibration/{label}_half{swap}.json';write(artifact,record)
            mask = np.isin(np.arange(len(real.donors)),test)
            swaps.append(dict(calibration_donors=real.donors[cal].tolist(),
                evaluation_donors=real.donors[test].tolist(),rank=record['rank'],
                artifact=str(artifact.relative_to(ROOT)),artifact_sha256=sha(artifact),
                raw=scores(low,high,observed,mask,I),calibrated=scores(lo,hi,observed,mask,I)))
        assert np.array_equal(lo[:,:50],low[:,:50]) and np.array_equal(hi[:,:50],high[:,:50])
        raw = scores(low,high,observed,real.is_heldout,I)
        calibrated = scores(lo,hi,observed,real.is_heldout,I)
        useful = all(abs(calibrated[g]['coverage']-.8)<abs(raw[g]['coverage']-.8) and
                     calibrated[g]['interval_score']<raw[g]['interval_score'] for g in ('sd','combined'))
        rows[label] = dict(generated_sha256=sha(path),generation_receipt_sha256=sha(receipt_path),
            historical=row['historical'],raw_heldout=raw,calibrated_heldout=calibrated,
            raw_train=scores(low,high,observed,~real.is_heldout,I),
            calibration_useful_on_both_primary_groups=useful,swaps=swaps,
            raw_rank_heldout=I.summarize(sample,observed,real.is_heldout,'rank'))
        np.savez_compressed(OUT/f'{label}_crossfitted_intervals.npz',donors=real.donors,
            is_heldout=real.is_heldout,raw_low=low,raw_high=high,calibrated_low=lo,calibrated_high=hi)
    base,candidate = rows['baseline'],rows['quarter']
    gates = {f'{group}_{metric}':(abs(candidate['raw_heldout'][group][metric]-.8)<abs(base['raw_heldout'][group][metric]-.8)
             if metric=='coverage' else candidate['raw_heldout'][group][metric]<base['raw_heldout'][group][metric])
             for group in ('sd','combined') for metric in ('coverage','interval_score')}
    guards = {'pb_rmse':.005,**{k:.02 for k in ('meanlog_rmse_gene','nonzero_rmse_gene','logvar_rmse_gene','pc_energy_rep','cell_energy_rep')}}
    gates.update({key:candidate['historical'][key]<=base['historical'][key]*(1+limit) for key,limit in guards.items()})
    finish(R,cache,anchor,digest)
    value=dict(status='completed_fixed_native_EMPF_coverage_pass',rows=rows,candidate_gates=gates,
        candidate_advances=all(gates.values()),experiment_manifest_sha256=digest,
        application_manifest_sha256=sha(R.SOURCE_MANIFEST),cache_anchor=anchor,
        default_changed=False,training_updates=0,scope='one development fold/seed,25 GAN TRAIN donors and12 held-out donors; two6/6 calibration/test swaps, reused cohort, no independent coverage guarantee',
        calibration_changes_generated_cells=False,empirical_cell_resampling_used=False)
    write(OUT/'comparison.json',value)
    print(json.dumps({label:{k:v for k,v in row.items() if k in ('raw_heldout','calibrated_heldout','calibration_useful_on_both_primary_groups')}
                     for label,row in rows.items()}|{'candidate_gates':gates,'candidate_advances':all(gates.values())},indent=2),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase',choices=('generate','analyze'),required=True)
    args=parser.parse_args()
    generate() if args.phase=='generate' else analyze()
