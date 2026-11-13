"""More predictive draws, freezing the M=32 train-only output calibration.

Supports existing pbulk and experimental fit classes. Outputs outside fit
folders. No optimizer update, new fitted calibration or score replacement.
"""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ.setdefault(key,'3')
import argparse
import json
from pathlib import Path
import numpy as np
import torch
import trainer as T
import generate as G
import mode_biascal as MB


@torch.no_grad()
def run(fit,cache,calibration_receipt,out,draws,device):
    fit,cache,calibration_receipt,out=map(Path,(fit,cache,calibration_receipt,out))
    if draws < 32 or draws % 32:
        raise ValueError('draws must be a positive multiple of32, >=32')
    G._check_out(fit,json.loads((fit/'config.json').read_text()),out)
    if out.exists():
        raise FileExistsError(out)
    config=json.loads((fit/'config.json').read_text())
    ctx=T.Context(cache,device,tuple(config['flags']))
    targets=G.targets(ctx)
    states=G.available_states(fit,config)
    G.M=draws
    plans,_,_=G.mode_plans(ctx,'stagehalf',targets,states,updates=config['updates'])
    names=[n for n in states if any(n in selected for _,selected in plans)]
    models=G.load_models(ctx,fit,config,names)
    receipt=json.loads(calibration_receipt.read_text())
    # A receipt from a different fit could silently mix incompatible calibration.
    if Path(receipt['fit']).resolve() != fit.resolve():
        raise ValueError('calibration receipt belongs to a different fit')
    info=receipt['mode_extra']['biascal']
    delta=torch.tensor(info['delta'],dtype=torch.float32,device=device)
    for model,_ in models.values():
        MB.head(model).add_(delta)
    arrays,stats=G.generate_arrays(ctx,models,plans,T=targets)
    arrays['mode']=np.array('biascal@stagehalf')
    arrays['conditions']=np.stack([c for c,_ in plans]).astype(np.float32)
    out.parent.mkdir(parents=True,exist_ok=True)
    temp=out.with_name('.'+out.stem+'.partial.npz')
    np.savez_compressed(temp,**arrays)
    temp.replace(out)
    T.dump(out.with_suffix('.receipt.json'),dict(stats,fit=str(fit.resolve()),
           generation_device=device,generated_SHA256=T.sha(out),
           calibration_receipt_sha256=T.sha(calibration_receipt),calibration_M=32,
           calibration='fixed train-only M32 delta; no refitting',
           note='M128 changes state/stage allocation and MC draws, not only percentile estimation'))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('fit','cache','calibration-receipt','out'):
        p.add_argument('--'+name,required=True)
    p.add_argument('--draws',type=int,default=128)
    p.add_argument('--device',choices=['cpu','cuda'],default='cpu')
    p.add_argument('--gpu-mem-fraction',type=float,default=.04)
    a=p.parse_args()
    torch.set_num_threads(3)
    if not 0<a.gpu_mem_fraction<=.04:
        p.error('memory fraction must be in(0,.04]')
    if a.device=='cuda':
        torch.cuda.set_per_process_memory_fraction(a.gpu_mem_fraction)
    run(a.fit,a.cache,a.calibration_receipt,a.out,a.draws,a.device)
