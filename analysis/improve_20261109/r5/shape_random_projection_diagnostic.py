"""Check count-shape in independent random gene directions, no calibration fit.

Regenerates from stored train-only calibration coefficients, whose rounded
total bias delta need not reproduce incremental bias additions bitwise.
This diagnostic does not replace the prospective PCA-space comparison.
"""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[key]='3'
os.environ['CUDA_VISIBLE_DEVICES']=''
import argparse
import json
from pathlib import Path
import numpy as np
import torch
import trainer as T
import generate as G
import evaluate as E
import mode_cellshape as CS
import mode_zcal as ZC
import mode_biascal as BC
from arm_d_randproj import fixed_directions
from architecture_score import observed_summaries
from interval_metrics import summarize


@torch.no_grad()
def run(fold,seed):
    cache=T.RUNS/f'caches/hvg2000/fold{fold}'
    fit=T.RUNS/f'fits_r3/pbulk/hvg2000/fold{fold}/seed{seed}'
    source=T.RUNS/'smoke_r5/architecture_pass/shape_confirmation'
    output=T.RUNS/f'results_r5/architecture_pass/shape_random_projection/fold{fold}/seed{seed}.json'
    if output.exists():raise FileExistsError(output)
    cfg=json.loads((fit/'config.json').read_text())
    ctx=T.Context(cache,'cpu',tuple(cfg['flags']))
    real=E.load_real(cache)
    # Independent directions; only scaling uses expression, from training cells.
    direction=fixed_directions(ctx.n_genes)
    counts=ctx.X.cpu().numpy().astype(np.float64)
    standard=(E.MX.log_cp10k(counts)-real.mean)/real.sd
    scale=np.maximum((standard@direction).std(0),1e-6)
    real.W,real.pc_sd=direction,scale
    real.pcs=[real.project(c) for c in real.cells]
    actual=observed_summaries(real,64)
    targets=G.targets(ctx)
    states=G.available_states(fit,cfg)
    plans,_,_=G.mode_plans(ctx,'stagehalf',targets,states,updates=cfg['updates'])
    names=[n for n in states if any(n in group for _,group in plans)]
    results={}
    for recipe in ('default','count_shape'):
        receipt_path=source/recipe/f'fold{fold}/seed{seed}.receipt.json'
        receipt=json.loads(receipt_path.read_text())
        if Path(receipt['fit']).resolve()!=fit.resolve() or receipt['generation_device']!='cpu':
            raise ValueError('diagnostic requires the matched CPU receipt from this fit')
        # Load under original evaluation projection so frozen arm-state checks
        # remain unchanged; swap the observation projector only after loading.
        models=G.load_models(ctx,fit,cfg,names)
        extra=receipt['mode_extra']
        for model,_ in models.values():
            BC.head(model).add_(torch.tensor(extra['biascal']['delta'],dtype=torch.float32))
            if recipe=='count_shape':
                CS.patch_generator(model,float(extra['cellshape']['alpha']))
                calibration=ZC.patch_generator(model,ctx.device)
                calibration['s'].copy_(torch.tensor(extra['zcal']['s'],dtype=torch.float32))
        original=ctx.W
        ctx.W=torch.tensor((direction/scale).astype(np.float32))
        try:arrays,stats=G.generate_arrays(ctx,models,plans,T=targets)
        finally:ctx.W=original
        gen=E.align_generated(arrays,real)
        summaries=E.MX.pc_bag_summary(gen['pcs'].astype(np.float32))
        intervals={split:summarize(summaries,actual,mask) for split,mask in (
            ('train',~real.is_heldout),('heldout',real.is_heldout))}
        energy=E.pc_space_repeated(gen,real,E.DEFAULT_PC_REPEATS,E.DEFAULT_CELL_REPEATS)
        results[recipe]={'intervals':intervals,'projected_distribution':energy,
                         'calibration_receipt_sha256':T.sha(receipt_path)}
        print('completed random projection diagnostic',fold,seed,recipe,flush=True)
        del models,arrays,gen
    value={'fold':fold,'seed':seed,'direction_seed':20261113,'directions':50,'M':32,
           'scaling':'training cells only','results':results,
           'scope':'diagnostic regeneration, frozen fits/calibration, independent random observation directions',
           'source_sha256':T.sha(Path(__file__))}
    T.dump(output,value)
    print(json.dumps({r:v['intervals']['heldout'] for r,v in results.items()},indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--fold',type=int,choices=[3],default=3)
    p.add_argument('--seed',type=int,choices=[0,1],required=True)
    a=p.parse_args();torch.set_num_threads(3)
    run(a.fold,a.seed)
