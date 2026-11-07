"""Fixed calibration ED measurement with isolated generation streams.

Two common-random-number replicates, four equally represented stages, fitting-only
library traces. No held-out cohort is loaded. Only one scalar can control the LR.
Reduced bag counts are allowed exclusively on explicit synthetic CPU fixtures.
"""
from contextlib import contextmanager
from pathlib import Path
import hashlib
import json
import math
import time
import numpy as np
import torch
from calibration import cell_energy, targets, sha
from evaluation import Reservoir

MANIFEST_SHA256='597ff7929037aabd1d7bc2bd8631aa569dff56ca188ce4a48fbf946ef08e908b'
DONORS=['H20.33.032','H21.33.001','H21.33.023','H21.33.025']

def array_hash(x):
    x=np.ascontiguousarray(x)
    return hashlib.sha256(str(x.dtype).encode()+repr(x.shape).encode()+x.tobytes()).hexdigest()

@contextmanager
def isolated_generation(ctx,G,P):
    modules=list(G.modules())+list(P.modules())
    modes=[m.training for m in modules]
    devices=[] if ctx.device.type=='cpu' else [ctx.device.index if ctx.device.index is not None else torch.cuda.current_device()]
    try:
        with torch.random.fork_rng(devices=devices), torch.no_grad():
            G.eval();P.eval()
            yield
    finally:
        for module,mode in zip(modules,modes):module.training=mode

class CalibrationScalar:
    split_role='calibration'
    metric='calibration_cell_energy_distance'
    def __init__(self,*,synthetic_target=None,bags=200):
        if type(bags) is not int or bags<1:raise ValueError('positive integer bags required')
        self.bags=bags;self.synthetic=synthetic_target is not None
        if self.synthetic:
            self.real=np.asarray(synthetic_target).copy();self.manifest_sha=None
        else:
            if bags!=200:raise ValueError('production calibration uses 200 donor bags per stage')
            from calibration import DATA
            if sha(DATA/'manifest.json')!=MANIFEST_SHA256:raise ValueError('calibration manifest differs')
            manifest,y,cells=targets()
            if sorted(y['donors'].tolist())!=sorted(DONORS):raise ValueError('calibration donor identities differ')
            self.real=cells['equal_stage'].copy();self.manifest_sha=MANIFEST_SHA256
            y.close();cells.close()
        if self.real.shape!=(1500,50) or not np.isfinite(self.real).all():raise ValueError('finite equal-stage calibration PCA50 reservoir required')

    def __call__(self,ctx,bundle,seed,*,completed_updates,out=None,end_monotonic=None):
        if end_monotonic is not None and (type(end_monotonic) not in (int,float) or not math.isfinite(end_monotonic)):raise ValueError('finite measurement deadline required')
        if ctx.fold!=0 or type(seed) is not int or seed not in (5,6,7):raise ValueError('unregistered fresh adaptive namespace')
        if bundle[-1]._phase!='ready' or bundle[-1].completed_steps!=completed_updates:raise ValueError('calibration requires a completed optimizer pair')
        if self.synthetic and (ctx.device.type!='cpu' or getattr(ctx,'synthetic',False) is not True):raise ValueError('synthetic targets are CPU-fixture only')
        G,P=bundle[1],bundle[3];pools=[];replicas=[];trace_pins=[]
        with isolated_generation(ctx,G,P):
            for rep in (0,1):
                stage_pools=[]
                for stage in range(4):
                    s=20261104+1000*seed+100*rep+stage
                    c=np.zeros(7,dtype=np.float32);c[3]=stage/3
                    schedule=ctx.law.schedule(c,s,m=self.bags,k=400)
                    trace_pins.append(dict(replicate=rep,stage=stage,seed=s,donors_SHA256=array_hash(schedule.donors),rows_SHA256=array_hash(schedule.rows),libraries_SHA256=array_hash(schedule.libraries)))
                    draw_seed=s+500000000
                    torch.set_rng_state(torch.Generator().manual_seed(draw_seed).get_state())
                    if ctx.device.type=='cuda':
                        with torch.cuda.device(ctx.device):torch.cuda.manual_seed(draw_seed)
                    lat=torch.Generator(device=ctx.device).manual_seed(draw_seed+1)
                    cell=torch.Generator(device=ctx.device).manual_seed(draw_seed+2)
                    count=torch.Generator(device=ctx.device).manual_seed(draw_seed+3)
                    conditions=torch.as_tensor(c,device=ctx.device)[None]
                    reservoir=Reservoir(seed=20261106+10*rep+stage,size=375)
                    for start in range(0,self.bags,5):
                        if end_monotonic is not None and time.monotonic()>=end_monotonic:raise TimeoutError('finite calibration allocation ended')
                        n=min(5,self.bags-start)
                        z,_=P.sample(n,generator=lat)
                        lib=torch.as_tensor(schedule.libraries[start:start+n],device=ctx.device)
                        rates=G.rates(z,conditions.expand(n,-1),400,lib,cell)
                        counts=torch.poisson(rates,generator=count)
                        if counts.shape!=(n,400,2000) or not bool(torch.isfinite(counts).all()) or not bool((counts>=0).all()) or not bool((counts==counts.floor()).all()):raise ValueError('native integer RNA generation contract failed')
                        pcs=ctx.project(counts).cpu().numpy()
                        if pcs.shape!=(n,400,50) or not np.isfinite(pcs).all():raise ValueError('projected cell shape/range differs')
                        reservoir.add(pcs)
                    if reservoir.cells.shape!=(375,50):raise ValueError('stage reservoir incomplete')
                    stage_pools.append(reservoir.cells.copy())
                pool=np.stack(stage_pools);pools.append(pool)
                value=cell_energy(pool.reshape(-1,50),self.real)
                if not np.isfinite(value) or value<0:raise ValueError('invalid calibration energy distance')
                replicas.append(value)
        all_pools=np.stack(pools)
        result=dict(metric=self.metric,split_role=self.split_role,value=float(np.mean(replicas)),replicate_values=replicas,generation_replicates=2,completed_updates=completed_updates,training_seed=seed,cells_per_replicate=4*self.bags*400,pool_SHA256=array_hash(all_pools),trace_SHA256=hashlib.sha256(json.dumps(trace_pins,sort_keys=True).encode()).hexdigest(),calibration_manifest_SHA256=self.manifest_sha,synthetic_CPU_fixture=self.synthetic)
        if out is not None:
            out=Path(out);out.mkdir(exist_ok=True,parents=True)
            path=out/f'update{completed_updates:05d}.npz'
            if path.exists():raise FileExistsError('calibration arrays must not be replaced')
            np.savez_compressed(path,generated_cells_by_stage=all_pools,real_equal_stage_cells=self.real)
            result['arrays_path']=str(path);result['arrays_SHA256']=sha(path)
        return result
