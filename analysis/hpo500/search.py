"""Deterministic six-scalar search with a paired control and constrained EI."""
import hashlib,json
from pathlib import Path
import numpy as np
from scipy.special import ndtr
from scipy.stats import qmc
HERE=Path(__file__).resolve().parent
def read(p):return json.loads(Path(p).read_bytes())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def protocol():return read(HERE/'protocol.json')
def coordinates(h):
    p=protocol();return np.log2([h[k]/p['baseline_hyperparameters'][k] for k in p['parameter_order']])
def parameters(x):
    p=protocol();return {k:float(p['baseline_hyperparameters'][k]*2**float(v)) for k,v in zip(p['parameter_order'],x)}
def kernel(a,b):
    r=np.sqrt(((a[:,None]-b[None,:])**2).sum(-1))/1.5
    v=np.sqrt(5)*r
    return (1+v+5*r*r/3)*np.exp(-v)
def posterior(x,y,pool):
    # Fixed kernel/regularization: this is a deterministic search heuristic,
    # not an uncertainty claim about four biological calibration donors.
    mean=y.mean();scale=max(float(y.std()),.02);z=(y-mean)/scale
    k=kernel(x,x)+np.eye(len(x))*1e-5;cross=kernel(pool,x)
    chol=np.linalg.cholesky(k)
    weights=np.linalg.solve(chol.T,np.linalg.solve(chol,z))
    v=np.linalg.solve(chol,cross.T)
    variance=np.maximum(1e-10,1-(v*v).sum(0))
    return mean+scale*(cross@weights),scale*np.sqrt(variance)
def feasible(row,control):
    if not row['finite']:return False
    m=row['metrics'];return m['cell_energy_distance']<=control['cell_energy_distance'] and .7<=m['spread_ratio']<=1.3
def suggest(index,records):
    assert 0<=index<25 and len(records)==index
    assert [r['candidate_id'] for r in records]==[f'c{i:03d}' for i in range(index)]
    if index==0:return np.zeros(6),dict(method='current_hyperparameter_control')
    if index<=12:
        x=np.zeros(6);x[(index-1)//2]=-1 if index%2 else 1
        return x,dict(method='paired_log2_coordinate_design',coordinate=(index-1)//2)
    control=records[0]['metrics'];assert records[0]['finite'] and control['energy_score']>0 and control['cell_energy_distance']>0
    x=np.stack([coordinates(r['hyperparameters']) for r in records])
    labels=[]
    for r in records:
        if r['finite']:
            m=r['metrics'];labels.append([m['energy_score']/control['energy_score'],m['cell_energy_distance']/control['cell_energy_distance'],np.log(max(1e-8,m['spread_ratio']))])
        else:labels.append([10.,10.,np.log(50.)])
    labels=np.asarray(labels)
    pool=4*qmc.Sobol(d=6,scramble=True,seed=20261106).random_base2(m=12)-2
    # An evaluated setting cannot be proposed twice, including a failed fit.
    keep=np.min(((pool[:,None]-x[None,:])**2).sum(-1),axis=1)>1e-6;pool=pool[keep]
    means=[];sds=[]
    for column in range(3):
        mu,sd=posterior(x,labels[:,column],pool);means.append(mu);sds.append(sd)
    candidates=[r['metrics']['energy_score']/control['energy_score'] for r in records if feasible(r,control)]
    best=min(candidates) if candidates else min(labels[:,0])
    mu,sd=means[0],sds[0];z=(best-mu)/sd
    ei=(best-mu)*ndtr(z)+sd*np.exp(-z*z/2)/np.sqrt(2*np.pi)
    p_ed=ndtr((1-means[1])/sds[1])
    p_spread=ndtr((np.log(1.3)-means[2])/sds[2])-ndtr((np.log(.7)-means[2])/sds[2])
    acquisition=np.maximum(ei,0)*p_ed*np.maximum(p_spread,0)
    fallback=not np.isfinite(acquisition).all() or float(acquisition.max())<1e-12
    merit=sds[0]+sds[1]+sds[2] if fallback else acquisition
    chosen=int(np.argmax(merit))
    return pool[chosen],dict(method='max_uncertainty_fallback' if fallback else 'constrained_expected_improvement',surrogate='fixed_Matern52_length1.5_noise1e-5',pool_size=len(pool),selected_pool_index=chosen,acquisition=float(merit[chosen]),failed_fit_imputation_only_for_search=[10.,10.,float(np.log(50.))],source_results=[r['attempt'] for r in records])
def make_proposal(index,records):
    x,reason=suggest(index,records)
    return dict(id=f'c{index:03d}',endpoint=500,native_version='0.9.6',hyperparameters=parameters(x),log2_coordinates=x.tolist(),proposal_method=reason,protocol_SHA256=sha(HERE/'protocol.json'),heldout_results_used=False)
def shortlist(records):
    assert len(records)==25 and records[0]['finite']
    control=records[0]['metrics'];finite=[r for r in records if r['finite'] and r['candidate_id']!='c000']
    valid=[r for r in finite if feasible(r,control)]
    def objective(r):return (r['metrics']['energy_score'],r['metrics']['cell_energy_distance'],r['candidate_id'])
    valid.sort(key=objective)
    if valid:chosen=valid[:2];exploratory=False
    else:
        finite.sort(key=lambda r:(max(r['metrics']['energy_score']/control['energy_score'],r['metrics']['cell_energy_distance']/control['cell_energy_distance']),*objective(r)))
        chosen=finite[:2];exploratory=True
    return dict(status='PROSPECTIVE_FROZEN500STEP_FINALISTS_BEFORE_FRESH_SEEDS',candidates=['c000']+[r['candidate_id'] for r in chosen],confirmation_seeds=[1,2,3,4],exploratory_without_feasible_seed0_candidate=exploratory,selection_complete_before_confirmation=True,protocol_SHA256=sha(HERE/'protocol.json'),search_result_SHA256={r['attempt']:sha(HERE/'results'/(r['attempt']+'.json')) for r in records})
