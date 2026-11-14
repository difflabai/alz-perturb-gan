"""One fixed TRAIN-only joint-count diagnostic, never a default selector."""
from contextlib import contextmanager,ExitStack,nullcontext
import fcntl
import hashlib
from pathlib import Path
import sys
import time
import joint_common as J
import factor_model as F
import gamma_head as H
import generate_amplitude as AMP
B=J.B;np=J.np;torch=J.torch;G=B.G


class ScopeInfeasible(ValueError):pass


class SummaryRoute(F.FactorRoute):
    def __init__(self,random_projection,raw_projection):
        super().__init__([0.],lambda donor:1)
        self.random_projection=torch.as_tensor(random_projection,dtype=torch.float32)
        self.raw_projection=torch.as_tensor(raw_projection,dtype=torch.float64)

    @torch.no_grad()
    def generate(self,ctx,models,plans=None,*,T=None):
        targets=G.targets(ctx) if T is None else T
        if plans is None:plans,_=G.default_plans(ctx,targets)
        n=len(targets);gn=ctx.n_genes
        arrays={'pb_counts':np.zeros((n,G.M,gn),np.float32),'mean_log':np.zeros((n,G.M,gn),np.float32),
                'nonzero':np.zeros((n,G.M,gn),np.float32),'pcs':np.zeros((n,G.M,G.PC_CELLS,50),np.float16),
                'lib':np.zeros((n,G.M),np.float32),'sq_log':np.zeros((n,G.M,gn),np.float16)}
        arrays.update(random_pcs=np.zeros((n,G.M,G.PC_CELLS,50),np.float32),
            diagnostic_pcs=np.zeros((n,G.M,G.PC_CELLS,50),np.float32),
            raw_cov=np.zeros((n,G.M,100,100),np.float64),raw_mean=np.zeros((n,G.M,gn),np.float64),
            raw_sq=np.zeros((n,G.M,gn),np.float64),pc_total_mean=np.zeros((n,G.M),np.float64),
            pc_total_var=np.zeros((n,G.M),np.float64),pc_zero_total_cells=np.zeros((n,G.M),np.int64),
            all_zero_total_cells=np.zeros((n,G.M),np.int64))
        zero=0
        iterator=self.iterate(ctx,models,plans,targets)
        try:
            for ti,start,num,model,z,cond,libs,cell,count in iterator:
                rate=model.rates(z,cond,G.K,libs,cell)
                counts=torch.poisson(rate,generator=count)
                B.require(bool(torch.isfinite(counts).all()) and bool((counts>=0).all()),'invalid generated counts')
                total=counts.sum(-1);zero+=int((total==0).sum())
                logcp=torch.log1p(counts*(1e4/total.clamp_min(1))[...,None])
                sl=slice(start,start+num)
                arrays['pb_counts'][ti,sl]=counts.sum(1).cpu().numpy()
                arrays['mean_log'][ti,sl]=logcp.mean(1).cpu().numpy()
                arrays['sq_log'][ti,sl]=logcp.square().mean(1).cpu().numpy().astype(np.float16)
                arrays['nonzero'][ti,sl]=(counts>0).float().mean(1).cpu().numpy()
                pcs=G.standardize_counts(counts[:,:G.PC_CELLS],ctx.mean,ctx.sd)@ctx.W
                B.require(bool(torch.isfinite(pcs).all()),'nonfinite generated PC coordinates')
                arrays['pcs'][ti,sl]=pcs.cpu().numpy().astype(np.float16)
                arrays['diagnostic_pcs'][ti,sl]=pcs.cpu().numpy()
                arrays['lib'][ti,sl]=total.mean(1).cpu().numpy()
                normalized=G.standardize_counts(counts[:,:G.PC_CELLS],ctx.mean,ctx.sd)
                random=normalized@self.random_projection
                B.require(bool(torch.isfinite(random).all()),'nonfinite random coordinates')
                arrays['random_pcs'][ti,sl]=random.cpu().numpy()
                raw=counts[:,:G.PC_CELLS].double();points=raw@self.raw_projection
                centered=points-points.mean(1,keepdim=True)
                arrays['raw_cov'][ti,sl]=(centered.transpose(1,2)@centered/G.PC_CELLS).cpu().numpy()
                arrays['raw_mean'][ti,sl]=raw.mean(1).cpu().numpy()
                arrays['raw_sq'][ti,sl]=raw.square().mean(1).cpu().numpy()
                totals=raw.sum(-1)
                arrays['pc_total_mean'][ti,sl]=totals.mean(1).cpu().numpy()
                arrays['pc_total_var'][ti,sl]=totals.var(1,unbiased=False).cpu().numpy()
                arrays['pc_zero_total_cells'][ti,sl]=(totals==0).sum(1).cpu().numpy()
                arrays['all_zero_total_cells'][ti,sl]=(total==0).sum(1).cpu().numpy()
        finally:iterator.close()
        raw=np.stack([t[2] for t in targets])
        arrays.update(targets=np.array([t[0] for t in targets]),is_heldout=np.array([t[1] for t in targets],dtype=bool),
                      stage=G.stage_of(raw).astype(np.int64),genes=np.asarray(ctx.genes).astype(str))
        stats={'targets':n,'heldout_targets':int(arrays['is_heldout'].sum()),'M':G.M,'K':G.K,
               'generated_cells':n*G.M*G.K,'zero_count_cells':zero,'zero_count_cell_rate':zero/(n*G.M*G.K),
               'pcs_float16_max_abs':float(np.abs(arrays['pcs'].astype(np.float32)).max()),
               'sq_log_float16_max':float(arrays['sq_log'].astype(np.float32).max()),
               'heldout_dir_present':bool(arrays['is_heldout'].any())}
        return arrays,stats


def real_indices(n,donor):
    if n<128:raise ScopeInfeasible(f'{donor}: each real TRAIN half must contain64 cells; no fallback (n={n})')
    rng=np.random.default_rng(B.AS.E.MX.stable_seed('count-dependence-real-halves',20261114,donor))
    order=rng.permutation(n);halves=(order[:n//2],order[n//2:])
    bags=np.stack([np.stack([np.random.default_rng(B.AS.E.MX.stable_seed(
        'count-dependence-real-bag',20261114,donor,half,bag)).choice(pool,64,replace=False)
        for bag in range(16)]) for half,pool in enumerate(halves)])
    return halves,bags


def covariance(values):
    x=np.asarray(values,np.float64);centered=x-x.mean(-2,keepdims=True)
    return np.swapaxes(centered,-1,-2)@centered/x.shape[-2]


def model_moments(arrays):
    points=np.concatenate((arrays['diagnostic_pcs'],arrays['random_pcs']),axis=-1).astype(np.float64)
    return {'covariance':covariance(points).mean(1),'sd_center':points.std(-2,ddof=0).mean(1),
            'raw_covariance':arrays['raw_cov'].mean(1),'total_mean':arrays['pc_total_mean'].mean(1),
            'total_variance':arrays['pc_total_var'].mean(1),
            'zero_total_cells':int(arrays['all_zero_total_cells'].sum()),
            'total_generated_cells':int(np.prod(points.shape[:2])*G.K),
            'pc_zero_total_cells':int(arrays['pc_zero_total_cells'].sum()),
            'total_pc_cells':int(np.prod(points.shape[:-1])),
            'raw_gene_mean':arrays['raw_mean'].mean(1),
            'raw_gene_variance':np.maximum(0,arrays['raw_sq']-arrays['raw_mean']**2).mean(1)*points.shape[-2]/(points.shape[-2]-1)}


def decide(baseline,response,real):
    gates={};measures={}
    for half in (0,1):
        for label,sl in (('PCA',slice(0,50)),('fixed_random',slice(50,100))):
            truth=real['covariance'][half][:,sl,sl];before=baseline['covariance'][:,sl,sl];after=response['covariance'][:,sl,sl]
            gap=truth-before;change=after-before
            old_error=np.square(gap).sum((-2,-1));new_error=np.square(truth-after).sum((-2,-1))
            sd_truth=real['sd_center'][half][:,sl]
            old_sd=np.square(baseline['sd_center'][:,sl]-sd_truth)
            new_sd=np.square(response['sd_center'][:,sl]-sd_truth)
            prefix=f'half{half}/{label}'
            gates[prefix+'/covariance_error_reduced']=bool(new_error.mean()<old_error.mean())
            gates[prefix+'/sd_center_error_reduced']=bool(new_sd.mean()<old_sd.mean())
            inner=float((gap*change).sum((-2,-1)).mean())
            denominator=float(np.sqrt(np.square(gap).sum((-2,-1)).mean()*np.square(change).sum((-2,-1)).mean()))
            measures[prefix]={'baseline_covariance_error':float(old_error.mean()),'response_covariance_error':float(new_error.mean()),
                'baseline_sd_center_mse':float(old_sd.mean()),'response_sd_center_mse':float(new_sd.mean()),
                'gap_response_inner_product':inner,'gap_response_cosine':inner/denominator if denominator else None,
                'per_donor_covariance_error_change':(new_error-old_error).tolist(),
                'per_donor_sd_mse_change':(new_sd-old_sd).mean(-1).tolist(),
                'per_direction_sd_error_change':(new_sd-old_sd).mean(0).tolist(),
                'per_direction_signed_sd_response':(response['sd_center'][:,sl]-baseline['sd_center'][:,sl]).mean(0).tolist()}
    zero_before=baseline['zero_total_cells']/baseline['total_generated_cells']
    zero_after=response['zero_total_cells']/response['total_generated_cells']
    gates['zero_total_rate_nonincrease']=zero_after<=zero_before
    return {'passes_TRAIN_mechanism_only':all(gates.values()),'gates':gates,'measurements':measures,
            'baseline_zero_total_rate':zero_before,'response_zero_total_rate':zero_after,
            'baseline_matching64_zero_total_rate':baseline['pc_zero_total_cells']/baseline['total_pc_cells'],
            'response_matching64_zero_total_rate':response['pc_zero_total_cells']/response['total_pc_cells'],
            'interpretation':'TRAIN sampling-stability diagnostic; no held, biological or default eligibility inference'}


def prerequisite():
    pin=B.P.read(J.HERE/'predecessor_evidence.json')
    B.require(set(pin['runtime_sha256'])=={'status.json','artifact_audit.json','screen_comparison.json'},'wrong predecessor links')
    for name,digest in pin['runtime_sha256'].items():B.require(B.sha(B.ROOT/name)==digest,'fifth completion evidence changed')
    status=B.P.read(B.ROOT/'status.json');proof=B.P.read(B.ROOT/'artifact_audit.json')
    comparison=B.P.read(B.ROOT/'screen_comparison.json')
    B.require(status['status']=='completed' and status['screen_passes'] is False and proof['status']=='passed'
              and proof['complete'] and proof['verified_artifacts']==proof['required_artifacts']==24
              and proof['prelaunch_manifest_sha256']==J.PRIOR_SHA and proof['constant_amplitude_fidelity_verified']
              and comparison['candidate']=='amplitude32' and comparison['verdict']['status']=='complete'
              and comparison['verdict']['passes'] is False,'complete failed fifth screen required')
    required={(name,f,r) for name in B.RECIPES for f in range(3) for r in (0,1)}
    B.require(len(proof['artifacts'])==24 and {(x['recipe'],x['fold'],x['replicate']) for x in proof['artifacts']}==required,
              'wrong fifth artifact identities')
    for item in proof['artifacts']:
        name,f,r=item['recipe'],item['fold'],item['replicate']
        B.require(B.sha(B.ROOT/name/f'fold{f}/seed{r}.json')==item['score_sha256']
                  and B.sha(B.MODES/name/f'fold{f}/seed{r}.receipt.json')==item['receipt_sha256'],'fifth artifact changed')
    B.require(B.sha(B.ROOT/'constant_amplitude_fidelity.json')==pin['constant_amplitude_fidelity_sha256']
              ==proof['constant_amplitude_fidelity_sha256'],'fifth fidelity link changed')
    AMP.verify_fidelity(B.P.read(J.PREVIOUS/'source_manifest_prelaunch.json'))


def projections(ctx):
    rng=np.random.default_rng(20261113)
    basis,_=np.linalg.qr(rng.normal(size=(ctx.n_genes,50)))
    total=np.zeros(50);squared=np.zeros(50);n=0
    mean=np.asarray(ctx.space['mean'],np.float64);sd=np.asarray(ctx.space['sd'],np.float64)
    for rows in ctx.by:
        x=ctx.X[torch.as_tensor(rows)].numpy().astype(np.float64)
        z=(B.AS.E.MX.log_cp10k(x)-mean)/sd;values=z@basis
        n+=len(values);total+=values.sum(0);squared+=np.square(values).sum(0)
    scale=np.maximum(np.sqrt(np.maximum(0,squared/n-(total/n)**2)),1e-6)
    random=basis/scale;pca=np.asarray(ctx.space['W'],np.float64)/np.asarray(ctx.space['pc_sd'],np.float64)
    projection=np.concatenate((pca,random),axis=1)
    return projection,projection/sd[:,None],basis,scale


def real_moments(ctx,projection,raw_projection):
    n=len(ctx.ids);g=ctx.n_genes
    out={key:np.zeros(shape,np.float64) for key,shape in {
        'covariance':(2,n,100,100),'sd_center':(2,n,100),'raw_covariance':(2,n,100,100),
        'total_mean':(2,n),'total_variance':(2,n),'meanlog':(2,n,g),'detection':(2,n,g),
        'logvariance':(2,n,g),'raw_gene_mean':(2,n,g),'raw_gene_variance':(2,n,g)}.items()}
    selections=[]
    mean=np.asarray(ctx.space['mean'],np.float64);sd=np.asarray(ctx.space['sd'],np.float64)
    for index,donor in enumerate(ctx.ids):
        x=ctx.X[torch.as_tensor(ctx.by[index])].numpy().astype(np.float64)
        halves,bags=real_indices(len(x),str(donor));selections.append(bags)
        log=B.AS.E.MX.log_cp10k(x);points=(((log-mean)/sd)@projection).astype(np.float32)
        raw=x@raw_projection;totals=x.sum(-1)
        for half in (0,1):
            selected=points[bags[half]];pool=halves[half]
            out['covariance'][half,index]=covariance(selected).mean(0)
            out['sd_center'][half,index]=selected.std(1,ddof=0,dtype=np.float64).mean(0)
            out['raw_covariance'][half,index]=covariance(raw[bags[half]]).mean(0)
            out['total_mean'][half,index]=totals[bags[half]].mean()
            out['total_variance'][half,index]=totals[bags[half]].var(1,ddof=0).mean()
            out['meanlog'][half,index]=log[pool].mean(0);out['detection'][half,index]=(x[pool]>0).mean(0)
            out['logvariance'][half,index]=log[pool].var(0,ddof=1)
            out['raw_gene_mean'][half,index]=x[pool].mean(0);out['raw_gene_variance'][half,index]=x[pool].var(0,ddof=1)
    return out,np.stack(selections,axis=1)


@contextmanager
def replay_hooks(groups,coupling,heads):
    targets=B.G.targets;load=B.G.load_models
    def loader(*args,**kwargs):
        models=load(*args,**kwargs)
        for name,(model,_) in models.items():
            head=H.GammaHead(groups,coupling);head.patch(model);heads[name]=head
        return models
    B.G.targets=B.ZC.train_targets
    if coupling is not None:B.G.load_models=loader
    try:yield
    finally:B.G.targets=targets;B.G.load_models=load


def replay(fit,cache,calibration,groups,coupling,projection,raw_projection,label,custom):
    path=J.SMOKE/(label+'.npz');heads={}
    B.require(not path.exists(),'partial diagnostic archive; inspect before resuming')
    route=SummaryRoute(projection[:,50:],raw_projection)
    with replay_hooks(groups,coupling,heads),route.installed() if custom else nullcontext():
        info=B.W.replay(fit,cache,calibration,path,16)
    with np.load(path,allow_pickle=False) as loaded:arrays={k:loaded[k].copy() for k in loaded.files}
    B.require(arrays['targets'].tolist()==B.P.read(cache/'manifest.json')['train_donors'] and not arrays['is_heldout'].any(),
              'non-TRAIN replay target')
    return arrays,info,heads,path


@torch.no_grad()
def analytic_points(ctx,fit,calibration,projection,raw_projection,groups):
    cfg=B.P.read(fit/'config.json');targets=B.ZC.train_targets(ctx)
    plans,states=AMP.layout(ctx,fit,cfg,targets);models=G.load_models(ctx,fit,cfg,states)
    reference=B.P.read(calibration)
    for model,_ in models.values():
        B.BC.head(model).add_(torch.tensor(reference['mode_extra']['biascal']['delta'],dtype=torch.float32))
        B.CS.patch_generator(model,.5)
        B.ZC.patch_generator(model,'cpu')['s'].copy_(torch.tensor(reference['mode_extra']['zcal']['s'],dtype=torch.float32))
    n=len(targets);g=ctx.n_genes
    point_mu=np.zeros((n,16,g),np.float32);point_theta=np.zeros_like(point_mu);point_lib=np.zeros((n,16),np.float32)
    point_state=np.zeros((n,16),np.int8);seen=np.zeros((n,16),bool)
    iterator=F.ORIGINAL_ITERATE(ctx,models,plans,targets)
    try:
        for ti,start,num,model,z,c,libs,cell,count in iterator:
            # Separate fixed-stream pass, exactly one NB call per chunk. This
            # does not add a call/draw to either actual count-generation pass.
            mu,theta=model.parameters_nb(z,c,G.K,libs,cell)
            B.require(theta.numel()==g,'gene-only theta required at analytic points')
            sl=slice(start,start+num);point_mu[ti,sl]=mu[:,0].numpy();point_theta[ti,sl]=theta.reshape(-1).numpy()
            point_lib[ti,sl]=libs[:,0].numpy();point_state[ti,sl]=[states.index(s) for s in plans[ti][1][sl]];seen[ti,sl]=True
    finally:iterator.close()
    B.require(seen.all(),'missing fixed conditional points')
    raw=np.zeros((n,16,100,100),np.float64);normalized=np.zeros_like(raw)
    for donor in range(n):
        for bag in range(16):
            mu=point_mu[donor,bag];theta=point_theta[donor,bag]
            raw[donor,bag]=H.projected_kernel(mu,theta,groups,raw_projection)
            jac=H.logcp_jacobian(mu,projection,ctx.space['sd'])
            normalized[donor,bag]=H.projected_kernel(mu,theta,groups,jac)
    B.require(np.isfinite(raw).all() and np.isfinite(normalized).all(),'nonfinite analytic response')
    return {'point_mu':point_mu,'point_theta':point_theta,'point_library':point_lib,'point_state':point_state,
            'analytic_raw_projected_delta':raw,'analytic_delta_method_logcp_projected_delta':normalized}


def gene_changes(arrays,moments,real):
    ml=arrays['mean_log'].astype(np.float64)
    values={'meanlog':ml.mean(1),'detection':arrays['nonzero'].astype(np.float64).mean(1),
            'logvariance':np.maximum(0,(arrays['sq_log'].astype(np.float64)-ml**2)*256/255).mean(1),
            'raw_gene_mean':moments['raw_gene_mean'],'raw_gene_variance':moments['raw_gene_variance']}
    summaries={};compact={}
    for half in (0,1):
        for key,value in values.items():
            error=value-real[key][half];name=f'half{half}_{key}'
            summaries[name]=gene_error_summary(error)
            compact[name+'_signed_gene_error']=error.mean(0)
    return summaries,compact


def gene_error_summary(error):
    error=np.asarray(error,np.float64)
    return {'rmse_gene':float(np.sqrt(np.square(error.mean(0)).mean())),
            'rmse_donor_gene':float(np.sqrt(np.square(error).mean())),'bias':float(error.mean())}


def alignment_tables(baseline,response,real,analytic):
    def inner(a,b):return float((a*b).sum((-2,-1)).mean())
    def cosine(a,b):
        denominator=np.sqrt(inner(a,a)*inner(b,b))
        return inner(a,b)/denominator if denominator else None
    out={}
    raw_predicted=analytic['analytic_raw_projected_delta'].mean(1)
    norm_predicted=analytic['analytic_delta_method_logcp_projected_delta'].mean(1)
    for half in (0,1):
        for label,sl in (('PCA',slice(0,50)),('fixed_random',slice(50,100))):
            raw_gap=real['raw_covariance'][half][:,sl,sl]-baseline['raw_covariance'][:,sl,sl]
            raw_change=response['raw_covariance'][:,sl,sl]-baseline['raw_covariance'][:,sl,sl]
            norm_gap=real['covariance'][half][:,sl,sl]-baseline['covariance'][:,sl,sl]
            norm_change=response['covariance'][:,sl,sl]-baseline['covariance'][:,sl,sl]
            raw_approx=raw_predicted[:,sl,sl];norm_approx=norm_predicted[:,sl,sl]
            out[f'half{half}/{label}']={
                'raw_gap_actual_response_inner_product':inner(raw_gap,raw_change),
                'raw_gap_actual_response_cosine':cosine(raw_gap,raw_change),
                'raw_gap_conditional_kernel_inner_product':inner(raw_gap,raw_approx),
                'raw_actual_conditional_kernel_cosine':cosine(raw_change,raw_approx),
                'normalized_gap_delta_approximation_inner_product':inner(norm_gap,norm_approx),
                'normalized_actual_delta_approximation_cosine':cosine(norm_change,norm_approx),
                'delta_approximation_actual_response_squared_error':inner(norm_change-norm_approx,norm_change-norm_approx),
                'per_donor_raw_response_trace':np.trace(raw_change,axis1=-2,axis2=-1).tolist(),
                'per_direction_raw_kernel_response':np.diagonal(raw_approx,axis1=-2,axis2=-1).mean(0).tolist(),
                'per_direction_normalized_delta_approximation':np.diagonal(norm_approx,axis1=-2,axis2=-1).mean(0).tolist(),
                'interpretation':'Unconditional observed gaps; not identified real conditional residuals. One-point-per-bag analytical approximation.'}
    return out


def main():
    started=time.monotonic();torch.set_num_threads(3);G.M=16
    cache=B.W.RUNS/'caches/hvg2000/fold0';fit=B.W.RUNS/'fits_r3/pbulk/hvg2000/fold0/seed0'
    calibration=B.MODES/'current32/fold0/seed0.receipt.json';opened=set()
    sys.addaudithook(B.Q.read_guard(cache,opened))
    J.ROOT.mkdir(parents=True,exist_ok=True);J.SMOKE.mkdir(parents=True,exist_ok=True)
    with ExitStack() as stack:
        roots=(J.ROOT,B.ROOT,B.Q.ROOT,B.Q.B.ROOT,B.D.ROOT,B.H.ROOT,B.W.ROOT,B.W.RUNS/'logs_r5/architecture_pass')
        for root in roots:
            lock=stack.enter_context((root/'queue.lock').open('a'));fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        B.require(not (J.ROOT/'report.json').exists() and not (J.ROOT/'compact_moments.npz').exists()
                  and not any(p.is_file() for p in J.SMOKE.rglob('*')),'existing/partial diagnostic; inspect before resuming')
        prerequisite();expected=B.P.read(J.HERE/'source_manifest_prelaunch.json')
        B.require(J.sources()==expected and J.disk_bytes()<2_800_000_000,'source freeze/cap check failed')
        anchor=B.P.verify_cache(0,payloads=False)
        payloads={name:B.sha(cache/name) for name in ('train/counts.npz','train/metadata.npz','space.npz')}
        B.require(all(payloads[k]==anchor['payload_sha256'][k] for k in payloads),'TRAIN payload changed')
        fit_hashes=B.P.verify_fit(0,0);cfg=B.P.read(fit/'config.json');ctx=B.T.Context(cache,'cpu',tuple(cfg['flags']))
        B.require(list(map(str,ctx.ids))==anchor['train_donors'] and len(ctx.ids)==25 and ctx.n_genes==2000,'wrong TRAIN scope')
        projection,raw_projection,basis,scale=projections(ctx);groups=J.groups(ctx.space['W'])
        try:real,selections=real_moments(ctx,projection,raw_projection)
        except ScopeInfeasible as exc:
            B.require(J.sources()==expected,'source changed before scope stop')
            B.write(J.ROOT/'report.json',{'status':'TRAIN_half_scope_infeasible','reason':str(exc),'sources':expected,
                'cache_files_opened':sorted(opened),'heldout_payloads_opened':False,'model_replays_performed':0,
                'TRAIN_payload_sha256':payloads,'fit_hashes':fit_hashes,'default_changed':False})
            print(str(exc),flush=True);return
        B.G.generation_streams=lambda f,index:B.streams(f,index,0)
        reference,reference_info,_,reference_path=replay(fit,cache,calibration,groups,None,projection,raw_projection,'frozen_reference',False)
        baseline,baseline_info,zero_heads,baseline_path=replay(fit,cache,calibration,groups,0.,projection,raw_projection,'zero_control',True)
        fingerprints=B.array_fingerprints(reference)
        B.require(fingerprints==B.array_fingerprints(baseline)
                  and reference_info['mode_extra']==baseline_info['mode_extra']==B.P.read(calibration)['mode_extra'],
                  'real literal-zero full TRAIN replay differs from frozen arrays/calibration')
        B.require(reference['state_names'].tolist()==B.A.STATES and all(np.array_equal(reference[k],baseline[k])
                  for k in ('state_names','state_index','conditions')),'zero TRAIN state/condition allocation differs')
        archives={'frozen_reference':B.sha(reference_path),'zero_control':B.sha(baseline_path)}
        proof={'passed':True,'fold':0,'fitseed':0,'M':16,'K':256,'train_donors':list(map(str,ctx.ids)),
            'sources':expected,'calibration_receipt_sha256':B.sha(calibration),'array_fingerprints':fingerprints,
            'complete_delta':reference_info['mode_extra']['biascal']['delta'],'complete_s':reference_info['mode_extra']['zcal']['s'],
            'state_names':reference['state_names'].tolist(),'state_index':reference['state_index'].tolist(),
            'conditions_float32_sha256':hashlib.sha256(np.ascontiguousarray(reference['conditions']).tobytes()).hexdigest(),
            'pruned_archive_sha256':archives.copy(),'scope':'Full fixed TRAIN replay fidelity, not refit identity or superiority'}
        reference_path.unlink();baseline_path.unlink();B.write(J.ROOT/'zero_replay_fidelity.json',proof)
        print('verified exact real TRAIN zero-head replay control',flush=True)
        del reference
        B.require(J.disk_bytes()<2_800_000_000,'insufficient cap headroom')
        response,response_info,heads,response_path=replay(fit,cache,calibration,groups,.5,projection,raw_projection,'fixed_response',True)
        B.require(set(heads)==set(zero_heads)==set(B.A.STATES),'wrong six-state head set')
        B.require(response_info['mode_extra']==baseline_info['mode_extra'] and all(np.array_equal(response[k],baseline[k])
                  for k in ('state_names','state_index','conditions')),'fixed response altered frozen calibration/state/conditions')
        numerical={name:head.stats for name,head in heads.items()}
        bound_sum=sum(s['rate_error_bound_sum'] for s in numerical.values())
        bound_max=max(s['rate_error_bound_max'] for s in numerical.values())
        B.require(bound_sum<1e-8 and bound_max<1e-12,'global six-head Gamma underflow bound exceeded')
        archives['fixed_response']=B.sha(response_path);response_path.unlink()
        print('generated fixed.5 TRAIN count response; computing compact geometry',flush=True)
        moments={'baseline':model_moments(baseline),'response':model_moments(response)}
        decision=decide(moments['baseline'],moments['response'],real)
        compact={'train_donors':np.asarray(ctx.ids).astype(str),'genes':np.asarray(ctx.genes).astype(str),
                 'gene_groups':groups,'random_basis':basis,'random_scale':scale,'projection':projection,
                 'raw_projection':raw_projection,'real_draw_indices':selections}
        for key,value in real.items():compact['real_'+key]=value
        genes={}
        for label,arrays in (('baseline',baseline),('response',response)):
            for key,value in moments[label].items():compact[label+'_'+key]=np.asarray(value)
            genes[label],vectors=gene_changes(arrays,moments[label],real)
            for key,value in vectors.items():compact[label+'_'+key]=value
        analytic=analytic_points(ctx,fit,calibration,projection,raw_projection,groups);compact.update(analytic)
        alignment=alignment_tables(moments['baseline'],moments['response'],real,analytic)
        out=J.ROOT/'compact_moments.npz';B.AS.E.save_npz(out,**compact)
        permitted={str((cache/name).resolve()) for name in ('manifest.json','space.npz','train/counts.npz','train/metadata.npz')}
        B.require(opened==permitted and J.sources()==expected and J.disk_bytes()<3_000_000_000,'read/source/cap final check failed')
        report={'status':'completed_TRAIN_count_dependence_diagnostic_only','sources':expected,'fold':0,'fitseed':0,
            'train_donors':list(map(str,ctx.ids)),'cache_files_opened':sorted(opened),'heldout_payloads_opened':False,
            'TRAIN_payload_sha256':payloads,'cache_manifest_sha256':anchor['manifest_sha256'],'fit_hashes':fit_hashes,
            'calibration_receipt_sha256':B.sha(calibration),'baseline_replay_info':baseline_info,'response_replay_info':response_info,
            'pruned_archives_sha256':archives,'baseline_array_fingerprints':B.array_fingerprints(baseline),
            'response_array_fingerprints':B.array_fingerprints(response),'zero_replay_fidelity_sha256':B.sha(J.ROOT/'zero_replay_fidelity.json'),
            'compact_moments_sha256':B.sha(out),'group_counts':np.bincount(groups,minlength=100).tolist(),
            'coupling':.5,'gamma_numerical':numerical,'global_rate_error_bound_sum':bound_sum,'global_rate_error_bound_max':bound_max,
            'final_cast_lost_rate_mass':sum(s['final_cast_lost_rate_mass'] for s in numerical.values()),
            'conditional_raw_kernel_diagonal':0,'conditional_points':400,'analytic_method':'Full logCP Jacobian delta approximation; one first cell per bag',
            'decision':decision,'raw_and_normalized_alignment':alignment,'TRAIN_gene_errors':genes,
            'runtime':{'python':sys.version,'numpy':np.__version__,'torch':str(torch.__version__),
                       'particlegan':str(B.T.VERSION),'compute_threads':torch.get_num_threads(),
                       'cpu_affinity':sorted(__import__('os').sched_getaffinity(0))},
            'default_changed':False,'logical_task_bytes':J.disk_bytes(),
            'seconds_total':time.monotonic()-started,'limitations':['TRAIN sampling halves reuse donors and PCA, not independent validation',
                'Mathematical conditional NB marginals only; machine rates approximate them',
                'No identified real-cell conditional residual, biological module or patient feature vector',
                'No held score/default eligibility; no coupling/module/rank rescue']}
        B.write(J.ROOT/'report.json',report)
        print(B.H.json.dumps({'status':report['status'],'TRAIN_mechanism_passes':decision['passes_TRAIN_mechanism_only'],
                              'seconds':report['seconds_total']}),flush=True)


if __name__=='__main__':main()

