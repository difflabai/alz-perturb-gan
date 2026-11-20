"""Pure fitting-donor stage law; no model, GPU or external data access."""
from dataclasses import dataclass
import numpy as np

def stages(c):
    x=np.asarray(c)
    if x.shape[-1:]!=(7,) or not np.isfinite(x).all():raise ValueError('Expected finite seven-slot condition')
    v=x[...,3]*3;s=np.rint(v).astype(int)
    if np.any((s<0)|(s>3)) or not np.allclose(v,s,atol=1e-6,rtol=0):raise ValueError('ADNC must be one of four frozen ordinal codes')
    return s

def mask_adnc(c):
    x=np.asarray(c,dtype=np.float32);stages(x)
    masked=np.zeros_like(x);masked[...,3]=x[...,3]
    return masked

def validate_split(split,cached_donors):
    fit=set(split['fit_donors']);val=set(split['validation_donors']);test=set(split['test_donors']);cached=set(cached_donors)
    if fit&val or fit&test or val&test:raise ValueError('Donor partitions overlap')
    if cached&test or cached!=fit|val:raise ValueError('Cache must contain fitting/validation donors only')
    if len(fit)!=len(split['fit_donors']) or len(val)!=len(split['validation_donors']) or len(test)!=len(split['test_donors']):raise ValueError('Duplicate donor in partition')

@dataclass(frozen=True)
class Schedule:
    donors: np.ndarray
    rows: np.ndarray
    libraries: np.ndarray

class StageLaw:
    def __init__(self,full_conditions,rows_by_donor,libraries):
        self.full_conditions=np.asarray(full_conditions,dtype=np.float32).copy();self.stage=stages(self.full_conditions)
        self.by=[np.asarray(r,dtype=int).copy() for r in rows_by_donor];self.libraries=np.asarray(libraries,dtype=np.float32).copy()
        if len(self.by)!=len(self.full_conditions) or not self.by or any(len(r)==0 for r in self.by):raise ValueError('Each fitting donor needs cells and covariates')
        rows=np.concatenate(self.by)
        if not np.array_equal(np.sort(rows),np.arange(len(self.libraries))):raise ValueError('Fitting cell rows must partition the fitting-only library pool')
        if not np.isfinite(self.libraries).all() or np.any(self.libraries<0):raise ValueError('Invalid library')
        self.groups={s:np.flatnonzero(self.stage==s) for s in range(4)}
    def schedule(self,c,seed,*,m=200,k=400):
        stage=int(stages(c));pool=self.groups[stage]
        if not len(pool):raise ValueError('No fitting donors at requested stage')
        if m<1 or k<1:raise ValueError('Positive donor/cell sample sizes required')
        # Separate deterministic stream shared by G/resampling/scVI. Donor
        # selection is uniform regardless of how many cells each donor has.
        rng=np.random.default_rng(int(seed)+88000)
        donors=rng.choice(pool,size=m,replace=True)
        rows=np.stack([rng.choice(self.by[d],size=k,replace=True) for d in donors])
        return Schedule(donors,rows,self.libraries[rows])
