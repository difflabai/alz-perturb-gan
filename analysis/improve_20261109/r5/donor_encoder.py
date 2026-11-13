"""Experimental set encoder for a frozen count generator's donor latent.

Encoder training uses synthetic bags only. Real donors enter evaluation after
training. Encoder inputs and reconstruction targets use disjoint cell halves.
This is final-state inversion, not the snapshot-mixture distribution score.
No claim of identified biological axes is made.
"""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ.setdefault(key,'3')
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from torch import nn
import trainer as T
import generate as G
import mode_biascal as MB
import evaluate as E


class DonorEncoder(nn.Module):
    def __init__(self,n_genes,z_dim=8):
        super().__init__()
        self.cell=nn.Sequential(nn.Linear(n_genes,64),nn.SiLU(),nn.Linear(64,64),nn.SiLU())
        self.donor=nn.Sequential(nn.Linear(128+7,64),nn.SiLU(),nn.Linear(64,z_dim))

    def forward(self,x,c):
        h=self.cell(x.clamp(-8,8))
        pooled=torch.cat([h.mean(1),(h.var(1,unbiased=False)+1e-6).sqrt(),c],1)
        return self.donor(pooled)


def repeatability(a,b):
    """Split-cell ICC-like ratio, retaining per-axis centering over donors."""
    centered=(a+b)/2
    signal=np.var(centered,axis=0,ddof=1).mean()
    noise=np.mean((a-b)**2)/2
    return float((signal-noise/2)/max(signal+noise/2,1e-12))


@torch.no_grad()
def decode_profile(model,z,c,library,seed=77,k=1024):
    gen=torch.Generator(device=z.device).manual_seed(seed)
    lib=library[:,None].expand(-1,k)
    mu,_=model.parameters_nb(z,c,k,lib,gen)
    proportions=mu.sum(1)/lib.sum(1,keepdim=True)
    return torch.log1p(1e4*proportions).cpu().numpy()


def run(fold,seed,steps,device,out_root):
    T.check_out_root(Path(out_root))
    fit=T.RUNS/f'fits_r3/pbulk/hvg2000/fold{fold}/seed{seed}'
    cache=T.RUNS/f'caches/hvg2000/fold{fold}'
    out=Path(out_root)/f'fold{fold}/seed{seed}'
    if (out/'evaluation.json').exists():
        raise FileExistsError(out)
    cfg=json.loads((fit/'config.json').read_text())
    ctx=T.Context(cache,device,tuple(cfg['flags']))
    models=G.load_models(ctx,fit,cfg,['final'])
    model,prior=models['final']
    receipt_path=fit/'generated_biascal@stagehalf.receipt.json'
    receipt=json.loads(receipt_path.read_text())
    delta=torch.tensor(receipt['mode_extra']['biascal']['delta'],device=device,dtype=torch.float32)
    with torch.no_grad():MB.head(model).add_(delta)
    model.eval().requires_grad_(False)
    prior.eval().requires_grad_(False)
    torch.manual_seed(20261113+seed)
    encoder=DonorEncoder(ctx.n_genes).to(device)
    optimizer=torch.optim.Adam(encoder.parameters(),lr=.001)
    rng=np.random.default_rng(20261113)
    rows=rng.permutation(len(prior.z))
    train_rows=torch.tensor(rows[:384],device=device)
    val_rows=torch.tensor(rows[384:],device=device)
    zmean=prior.z[train_rows].mean(0).detach()
    zsd=prior.z[train_rows].std(0,unbiased=False).clamp_min(.05).detach()
    latent=torch.Generator(device=device).manual_seed(20261113)
    cell=torch.Generator(device=device).manual_seed(20261114)
    counts_gen=torch.Generator(device=device).manual_seed(20261115)

    @torch.no_grad()
    def synthetic(rows,batch=16,k=64):
        ids=rows[torch.randint(len(rows),(batch,),device=device,generator=latent)]
        z=prior.z[ids]
        donor=torch.randint(len(ctx.ids),(batch,),device=device,generator=latent)
        c=ctx.conditions[donor]
        libs=ctx.fitlibs[torch.randint(len(ctx.fitlibs),(batch,k),device=device,generator=latent)]
        rate=model.rates(z,c,k,libs,cell)
        counts=torch.poisson(rate,generator=counts_gen)
        return T.standardize_counts(counts,ctx.mean,ctx.sd),c,(z-zmean)/zsd

    history=[]
    for step in range(steps):
        encoder.train()
        x,c,z=synthetic(train_rows)
        prediction=encoder(x,c)
        loss=(prediction-z).square().mean()
        if not bool(torch.isfinite(loss)):raise FloatingPointError('encoder loss')
        optimizer.zero_grad();loss.backward()
        torch.nn.utils.clip_grad_norm_(encoder.parameters(),5.)
        optimizer.step()
        if step%100==0 or step+1==steps:
            record={'step':step+1,'synthetic_train_mse':float(loss.detach())}
            history.append(record);print(json.dumps(record),flush=True)
    encoder.eval()
    with torch.no_grad():
        validation=[];null=[]
        for _ in range(32):
            x,c,z=synthetic(val_rows)
            validation.append(float((encoder(x,c)-z).square().mean()))
            null.append(float(z.square().mean()))
        x,c,_=synthetic(val_rows,k=64)
        assert torch.allclose(encoder(x,c),encoder(x.flip(1),c),atol=2e-6), 'encoder not permutation invariant'
    # No early stopping or parameter update follows any real-data evaluation.
    real=E.load_real(cache)
    train_mean=np.mean([np.log1p(1e4*np.asarray(real.cells[i].sum(0)).ravel()/real.cells[i].sum())
                        for i in np.flatnonzero(~real.is_heldout)],axis=0)
    records=[]
    for i,donor in enumerate(real.donors):
        n=real.cells[i].shape[0]
        permutation=np.random.default_rng(E.MX.stable_seed('encoder-cell-halves',donor)).permutation(n)
        a,b=np.array_split(permutation,2)
        if len(a)<16 or len(b)<16:raise ValueError('too few cells for donor encoder evaluation')
        c=torch.tensor(ctx.condition_matrix(real.raw_c[i:i+1]),device=device)
        inputs=[];zs=[];predictions=[];profiles=[]
        library=[]
        for rows_half in (a,b):
            choice=np.random.default_rng(E.MX.stable_seed('encoder-input',donor)).choice(rows_half,min(64,len(rows_half)),replace=False)
            x=torch.tensor(real.cells[i][choice].toarray(),device=device)
            pb=x.sum(0).cpu().numpy().astype(np.float64)
            profiles.append(np.log1p(1e4*pb/max(pb.sum(),1)))
            lib=x.sum(-1).mean().reshape(1)
            inputs.append(x)
            with torch.no_grad():
                z=encoder(T.standardize_counts(x[None],ctx.mean,ctx.sd),c)*zsd+zmean
                zs.append(z.cpu().numpy()[0]);library.append(lib)
                predictions.append(decode_profile(model,z,c,lib)[0])
        target=np.asarray(real.cells[i][b].sum(0)).ravel().astype(np.float64)
        target=np.log1p(1e4*target/max(target.sum(),1))
        depth={}
        for k in (32,128):
            rows_half=a
            choice=np.random.default_rng(E.MX.stable_seed('encoder-depth',donor,k)).choice(rows_half,k,replace=len(rows_half)<k)
            x=torch.tensor(real.cells[i][choice].toarray(),device=device)
            with torch.no_grad():
                z=encoder(T.standardize_counts(x[None],ctx.mean,ctx.sd),c)*zsd+zmean
            depth[str(k)]=float(((z.cpu().numpy()[0]-zs[0])/zsd.cpu().numpy()).dot(
                                  (z.cpu().numpy()[0]-zs[0])/zsd.cpu().numpy())/8)
        rmse=lambda p:float(np.sqrt(np.mean((p-target)**2)))
        records.append({'donor':str(donor),'heldout':bool(real.is_heldout[i]),'input_cells':len(inputs[0]),
                        'target_cells':len(b),'latent_a':zs[0].tolist(),'latent_b':zs[1].tolist(),
                        'encoder_reconstruction_rmse':rmse(predictions[0]),
                        'input_pseudobulk_rmse':rmse(profiles[0]),'train_mean_rmse':rmse(train_mean),
                        'depth_standardized_latent_mse':depth,
                        'predicted_profile':predictions[0].tolist()})
    aggregate={}
    scale=zsd.cpu().numpy()
    for split,held in [('train',False),('heldout',True)]:
        group=[r for r in records if r['heldout']==held]
        aggregate[split]={key:float(np.mean([r[key] for r in group])) for key in (
            'encoder_reconstruction_rmse','input_pseudobulk_rmse','train_mean_rmse')}
        aggregate[split]['split_cell_repeatability']=repeatability(
            np.array([r['latent_a'] for r in group])/scale,np.array([r['latent_b'] for r in group])/scale)
        aggregate[split]['depth_standardized_latent_mse']={str(k):float(np.mean([
            r['depth_standardized_latent_mse'][str(k)] for r in group])) for k in (32,128)}
    result={'fold':fold,'seed':seed,'steps':steps,'synthetic_validation_mse':float(np.mean(validation)),
            'synthetic_null_mse':float(np.mean(null)),'synthetic_validation_particles':len(val_rows),
            'aggregate':aggregate,'records':records,'history':history,
            'fit_final_sha256':T.sha(fit/'final.pt'),'calibration_receipt_sha256':T.sha(receipt_path),
            'scope':'synthetic-only encoder training; disjoint real cell halves; final-state inversion; no clinical/axis claim'}
    result['passes_experimental_checks']=(result['synthetic_validation_mse']<.8*result['synthetic_null_mse']
        and aggregate['heldout']['split_cell_repeatability']>=.8
        and aggregate['heldout']['encoder_reconstruction_rmse']<aggregate['heldout']['train_mean_rmse'])
    out.mkdir(parents=True,exist_ok=True)
    torch.save({'encoder':encoder.cpu().state_dict(),'zmean':zmean.cpu(),'zsd':zsd.cpu(),
                'fold':fold,'seed':seed,'genes':ctx.genes.tolist(),'steps':steps},out/'encoder.pt')
    T.dump(out/'evaluation.json',result)
    print(json.dumps({'aggregate':aggregate,'synthetic_validation_mse':result['synthetic_validation_mse'],
                      'passes_experimental_checks':result['passes_experimental_checks']},indent=2),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--fold',type=int,choices=range(9),required=True)
    p.add_argument('--seed',type=int,required=True)
    p.add_argument('--steps',type=int,default=2000)
    p.add_argument('--device',choices=['cpu','cuda'],default='cpu')
    p.add_argument('--gpu-mem-fraction',type=float,default=.04)
    p.add_argument('--out-root',default=str(T.RUNS/'smoke_r5/architecture_pass/encoder'))
    a=p.parse_args()
    torch.set_num_threads(3)
    if not 0<a.gpu_mem_fraction<=.04:p.error('invalid shared GPU cap')
    if a.steps<1 or a.seed<0:p.error('steps/seed invalid')
    if a.device=='cuda':torch.cuda.set_per_process_memory_fraction(a.gpu_mem_fraction)
    run(a.fold,a.seed,a.steps,a.device,a.out_root)
