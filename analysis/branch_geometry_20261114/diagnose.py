"""One fixed TRAIN-only replay; bag geometry and signed gene errors only."""
from contextlib import contextmanager,ExitStack
import fcntl
import json
from pathlib import Path
import sys
import time
from types import SimpleNamespace

HERE=Path(__file__).resolve().parent
PREVIOUS=HERE.parent/'paired_challenge_20261114'
sys.path.insert(0,str(PREVIOUS))
import pair_common as B
import pair_cells as K
import generate_pair as PAIR
from scipy.sparse import csr_matrix
np=B.np
ROOT=B.W.RUNS/'results_r5/branch_geometry_20261114'
SMOKE=B.W.RUNS/'smoke_r5/branch_geometry_20261114'
PRIOR_SHA='f6feb5844ac3088677c9d7f4324a5630a35df20c98c2e3f75c7aafe887876255'
TASKS=('architecture_pass','empirical_challenge_20261114','hybrid_challenge_20261114',
       'dispersion_challenge_20261114','paired_challenge_20261114','branch_geometry_20261114')


def sources():
    manifest=PREVIOUS/'source_manifest_prelaunch.json'
    B.require(B.sha(manifest)==PRIOR_SHA,'wrong prior192 pins')
    expected=B.P.read(manifest);B.require(B.source_manifest()==expected,'prior192 dependency changed')
    return {**expected,**{str(HERE/n):B.sha(HERE/n) for n in ('PROTOCOL.txt','diagnose.py','predecessor_evidence.json')}}


def disk_bytes():
    return sum(p.stat().st_size for root in ('fits_r5','smoke_r5','results_r5','logs_r5')
               for name in TASKS for p in (B.W.RUNS/root/name).rglob('*') if p.is_file())


def prerequisite():
    pinned=B.P.read(HERE/'predecessor_evidence.json')
    B.require(set(pinned['runtime_sha256'])=={'status.json','artifact_audit.json','screen_comparison.json'},
              'exact predecessor runtime links required')
    for name,digest in pinned['runtime_sha256'].items():B.require(B.sha(B.ROOT/name)==digest,'prior runtime changed')
    status=B.P.read(B.ROOT/'status.json');comparison=B.P.read(B.ROOT/'screen_comparison.json')
    B.require(status['status']=='completed' and status['screen_passes'] is False
              and comparison['phase']=='screen' and comparison['folds']==[0,1,2] and comparison['candidate']=='paired32'
              and comparison['verdict']['status']=='complete' and comparison['verdict']['passes'] is False,
              'completed failed paired predecessor required')
    proof=B.P.read(B.ROOT/'artifact_audit.json')
    B.require(proof['complete'] and proof['status']=='passed' and proof['verified_artifacts']==proof['required_artifacts']==24
              and proof['screen_passes'] is False and proof['single_cell_identity_verified']
              and proof['prelaunch_manifest_sha256']==PRIOR_SHA,'complete failed fourth-study proof required')
    expected={(n,f,r) for n in B.RECIPES for f in range(3) for r in (0,1)}
    B.require(len(proof['artifacts'])==24 and {(a['recipe'],a['fold'],a['replicate']) for a in proof['artifacts']}==expected,
              'wrong prior artifact identities')
    for item in proof['artifacts']:
        n,f,r=item['recipe'],item['fold'],item['replicate']
        B.require(B.sha(B.ROOT/n/f'fold{f}/seed{r}.json')==item['score_sha256']
                  and B.sha(B.MODES/n/f'fold{f}/seed{r}.receipt.json')==item['receipt_sha256'],'prior artifact changed')
    B.require(B.sha(B.ROOT/'single_cell_identity.json')==proof['single_cell_identity_sha256'],'prior identity proof changed')
    PAIR.verify_identity(B.P.read(PREVIOUS/'source_manifest_prelaunch.json'))


def read_guard(cache,opened):
    cache=Path(cache).resolve();cache_root=cache.parent.parent;allowed=cache/'train/counts.npz'
    permitted={cache/n for n in ('manifest.json','train/counts.npz','train/metadata.npz','space.npz')}
    def guard(event,args):
        if event!='open' or not isinstance(args[0],(str,bytes)):return
        p=Path(__import__('os').fsdecode(args[0])).resolve()
        if cache_root in p.parents and p not in permitted:
            raise AssertionError('only selected fold TRAIN/space/manifest cache opens permitted')
        if p.name=='counts.npz' and p!=allowed:raise AssertionError('non-TRAIN count payload forbidden')
        if cache_root in p.parents:opened.add(str(p))
    return guard


@contextmanager
def train_targets_only():
    original=B.G.targets;B.G.targets=B.D.ZC.train_targets
    try:yield
    finally:B.G.targets=original


def train_real(ctx):
    s=ctx.space
    real=SimpleNamespace(panel='hvg2000',fold=0,donors=ctx.ids.astype(str),
        is_heldout=np.zeros(len(ctx.ids),bool),stage=B.G.stage_of(ctx.raw_c),genes=np.asarray(ctx.genes),
        mean=np.asarray(s['mean'],np.float64),sd=np.asarray(s['sd'],np.float64),
        W=np.asarray(s['W'],np.float64),pc_sd=np.asarray(s['pc_sd'],np.float64),cells=[],libs=[],pcs=[],
        mean_log=[],nonzero=[],log_var=[])
    for rows in ctx.by:
        x=ctx.X[rows].cpu().numpy().astype(np.float64);log=B.AS.E.MX.log_cp10k(x)
        real.cells.append(csr_matrix(x.astype(np.float32)));real.libs.append(x.sum(1))
        real.pcs.append((((log-real.mean)/real.sd)@real.W/real.pc_sd).astype(np.float32))
        real.mean_log.append(log.mean(0));real.nonzero.append((x>0).mean(0))
        real.log_var.append(log.var(0,ddof=1))
    for key in ('mean_log','nonzero','log_var'):setattr(real,key,np.asarray(getattr(real,key)))
    return real


def trace_parts(y):
    y=np.asarray(y,np.float64);flat=y.reshape(-1,y.shape[-1])
    within=float(y.var(1,ddof=0).sum(-1).mean())
    between=float(y.mean(1).var(0,ddof=0).sum());total=float(flat.var(0,ddof=0).sum())
    B.require(abs(total-within-between)<1e-9*max(1.,total),'within/between-target trace identity failed')
    return {'total_population_trace':total,'within_target_trace':within,'between_target_trace':between}


def geometry(model,empirical):
    B.require(model.shape==empirical.shape and model.shape[-1]==100,'wrong equal component summaries')
    output={}
    for label,sl in (('means',slice(0,50)),('sds',slice(50,100)),('combined',slice(None))):
        left=np.asarray(model[...,sl],np.float64);right=np.asarray(empirical[...,sl],np.float64)
        a,b=trace_parts(left),trace_parts(right);mixed=trace_parts(np.concatenate((left,right),axis=1))
        delta=left.mean((0,1))-right.mean((0,1));separation=float(delta@delta/4)
        predicted=.5*(a['total_population_trace']+b['total_population_trace'])+separation
        B.require(abs(predicted-mixed['total_population_trace'])<1e-9*max(1.,predicted),'mixture trace identity failed')
        target_delta=left.mean(1)-right.mean(1)
        output[label]={'model':a,'empirical':b,'mixture':mixed,'component_center_trace':separation,
                       'component_center_fraction':separation/predicted if predicted else 0.,
                       'per_target_component_center_trace':(np.square(target_delta).sum(1)/4).tolist(),
                       'mean_per_target_component_center_trace':float(np.square(target_delta).sum(1).mean()/4)}
    return output


def gene_errors(arrays,real):
    ml=arrays['mean_log'].astype(np.float64);sq=arrays['sq_log'].astype(np.float64)
    values={'meanlog':ml.mean(1),'detection':arrays['nonzero'].astype(np.float64).mean(1),
            'logvariance':np.maximum(0.,(sq-ml**2)*256/255).mean(1)}
    truth={'meanlog':real.mean_log,'detection':real.nonzero,'logvariance':real.log_var}
    compact={};summaries={}
    for key,value in values.items():
        compact[key]=(value-truth[key]).mean(0)
        summaries[key]=B.AS.E.MX.gene_error_metrics(value,truth[key],real.mean_log.mean(0),[.01,.05,.1])
    return compact,summaries


def main():
    started=time.monotonic();B.torch.set_num_threads(3)
    ROOT.mkdir(parents=True,exist_ok=True);SMOKE.mkdir(parents=True,exist_ok=True)
    cache=B.W.RUNS/'caches/hvg2000/fold0';fit=B.W.RUNS/'fits_r3/pbulk/hvg2000/fold0/seed0'
    opened=set();sys.addaudithook(read_guard(cache,opened))
    with ExitStack() as stack:
        for root in (ROOT,B.ROOT,B.D.ROOT,B.H.ROOT,B.W.ROOT):
            lock=stack.enter_context((root/'queue.lock').open('a'));fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        B.require(not (ROOT/'report.json').exists() and not (ROOT/'compact_moments.npz').exists()
                  and not any(p.is_file() for p in SMOKE.rglob('*')),'partial/existing diagnostic; inspect first')
        prerequisite();expected=B.P.read(HERE/'source_manifest_prelaunch.json')
        B.require(sources()==expected,'source changed after prelaunch freeze')
        B.require(disk_bytes()<2_800_000_000,'insufficient cap headroom')
        anchor=B.P.verify_cache(0,payloads=False)
        checked={name:B.sha(cache/name) for name in ('train/counts.npz','train/metadata.npz','space.npz')}
        B.require(all(checked[n]==anchor['payload_sha256'][n] for n in checked),'TRAIN payload changed')
        fit_hashes=B.P.verify_fit(0,0);cfg=B.P.read(fit/'config.json')
        ctx=B.T.Context(cache,'cpu',tuple(cfg['flags']));real=train_real(ctx)
        B.require(real.donors.tolist()==anchor['train_donors'] and not real.is_heldout.any(),'wrong TRAIN-only scope')
        calibration=B.MODES/'current32/fold0/seed0.receipt.json'
        B.G.generation_streams=lambda f,j:B.streams(f,j,0)
        model_path=SMOKE/'model_train.npz'
        with train_targets_only():model_info=B.W.replay(fit,cache,calibration,model_path,16)
        model=B.H.read_arrays(model_path);model_hash=B.sha(model_path);model_path.unlink()
        unpaired,one=K.generate(real,0,16,'component',parts=1)
        paired,two=K.generate(real,0,16,'component',parts=2)
        B.require(one['target_bag_source_donors']==two['target_bag_source_donors'],'paired source cycles changed')
        arrays={'model':model,'empirical_single':unpaired,'empirical_pair':paired,
                'mixture_single':B.H.combine(model,unpaired),'mixture_pair':B.H.combine(model,paired)}
        summaries={name:B.AS.E.MX.pc_bag_summary(a['pcs'].astype(np.float32)) for name,a in arrays.items()}
        observed=np.stack([np.stack([B.AS.E.MX.pc_bag_summary(real.pcs[i][B.AS.E.MX.det_subsample(
            len(real.pcs[i]),64,B.AS.E.MX.stable_seed('pc-summary-rep',d,r))]) for i,d in enumerate(real.donors)])
            for r in range(B.AS.E.DEFAULT_PC_REPEATS)])
        compact={'genes':real.genes,'train_donors':real.donors,'observed_bag_summaries':observed}
        errors={};spread={}
        for name,a in arrays.items():
            y=summaries[name];compact[name+'_bag_summaries']=y
            vectors,errors[name]=gene_errors(a,real)
            for key,value in vectors.items():compact[name+'_'+key+'_signed_gene_error']=value
            numerator=float(y.astype(np.float64).reshape(-1,100).var(0,ddof=1).sum())
            spread[name]={'train_repeated_spread':float(np.mean([np.sqrt(numerator/o.astype(np.float64).var(0,ddof=1).sum())
                                                               for o in observed])),
                          'generated_summary_ddof1_trace':numerator}
        out=ROOT/'compact_moments.npz';B.AS.E.save_npz(out,**compact)
        B.require(sources()==expected and disk_bytes()<3_000_000_000,'source/cap changed')
        permitted={str((cache/n).resolve()) for n in ('manifest.json','train/counts.npz','train/metadata.npz','space.npz')}
        B.require(opened==permitted,'recorded cache opens differ from exact TRAIN-only payload set')
        report={'status':'completed_training_diagnostic_only','fold':0,'replicate':0,'train_donors':real.donors.tolist(),
                'heldout_payloads_opened':False,'cache_files_opened':sorted(opened),'TRAIN_payload_hashes':checked,
                'cache_manifest_sha256':anchor['manifest_sha256'],'fit_hashes':fit_hashes,'sources':expected,
                'calibration_receipt_sha256':B.sha(calibration),'model_replay_info':model_info,
                'pruned_model_train_npz_sha256':model_hash,'generated_TRAIN_array_fingerprints':{n:B.array_fingerprints(a) for n,a in arrays.items()},
                'compact_moments_sha256':B.sha(out),'geometry':{name:geometry(summaries['model'],summaries[name])
                    for name in ('empirical_single','empirical_pair')},'TRAIN_spread':spread,'signed_gene_error_summaries':errors,
                'seconds_total':time.monotonic()-started,'logical_task_bytes':disk_bytes(),'default_changed':False,
                'limitations':['Fixed f0/r0 TRAIN-only diagnostic, not fresh heldout performance or default eligibility',
                    'TRAIN-restricted replay, not hash reproduction or independent rescoring of prior pruned full-target archives',
                    'Bag-summary variance and within-bag gene variance are separate; no parameters were fitted',
                    'Paired cells are artificial pooled cells; no biological/patient-axis validity claim']}
        B.write(ROOT/'report.json',report)
        print(json.dumps({'status':report['status'],'train_targets':len(real.donors),'seconds':report['seconds_total']}))


if __name__=='__main__':main()
