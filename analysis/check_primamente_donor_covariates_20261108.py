"""Exploratory donor-level covariate sensitivity; preserves original ranking."""
import argparse,json,sys,hashlib,datetime
from pathlib import Path
ROOT=Path('/ml2/alz');B=Path('/ml2/codex-storage/future-workspaces/2026-10-27/task-2/benchmark')
sys.path[:0]=[str(B/'current_base_20261029/runtime/lib/python3.12/site-packages'),str(B/'.venv312/lib/python3.12/site-packages')]
import numpy as np
PLAN=ROOT/'analysis/primamente_training_donor_covariate_sensitivity_plan_20261108.json'
STATE=ROOT/'artifacts/primamente-t005-gene-inference-recovery-v1-20261108'
OUTPUT=ROOT/'analysis/primamente_donor_covariate_sensitivity_20261108.json'
def read(p):return json.loads(Path(p).read_bytes())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def fit(X,Y):
 assert np.isfinite(X).all() and np.isfinite(Y).all() and len(X)>X.shape[1]
 assert np.linalg.matrix_rank(X)==X.shape[1],'Covariate design rank deficient; do not report adjusted effect'
 return np.linalg.lstsq(X,Y,rcond=None)[0][1]
def check():
 rng=np.random.default_rng(71);hi=np.repeat([0.,1.],20);age=hi*4+rng.normal(size=40);sex=rng.integers(0,2,40);apoe=rng.integers(0,3,40);pmi=rng.normal(size=40)
 X=np.column_stack([np.ones(40),hi,age,sex,apoe,pmi]);coef=np.array([1.,2.,.5,-.2,.7,.9]);y=X@coef
 np.testing.assert_allclose(fit(X,y[:,None]),[2.],atol=1e-12);assert abs(y[hi==1].mean()-y[hi==0].mean()-2)>.5
 refused=False
 try:fit(np.column_stack([X,X[:,2]]),y[:,None])
 except AssertionError:refused=True
 assert refused
 print(json.dumps(dict(known_stage_effect_recovered_despite_covariate_confounding=True,rank_deficient_design_refused=True)))
def main():
 p=argparse.ArgumentParser();p.add_argument('--check',action='store_true');a=p.parse_args()
 if a.check:check();return
 assert not OUTPUT.exists();plan=read(PLAN);rank=read(ROOT/'analysis/primamente_gene_inference_exploratory_ranking_20261108.json');assert sha(STATE/'completed.json')==rank['input_completion_SHA256']
 with np.load(STATE/'real_training_donor_summaries.npz',allow_pickle=False) as z:genes=z['genes'];donors=z['donors'];stages=z['stages'];sums=z['sums'];n=z['n']
 cov=[plan['donors'][d] for d in donors];age=np.array([float(x['Age at Death']) for x in cov]);male=np.array([{'Male':1.,'Female':0.}[x['Sex']] for x in cov]);apoe=np.array([x['APOE Genotype'].split('/').count('4') for x in cov],float)
 try:pmi=np.array([float(x['PMI']) for x in cov]);use_pmi=bool(np.isfinite(pmi[np.isin(stages,[0,3])]).all())
 except (TypeError,ValueError):use_pmi=False
 pergroup=[];allbeta=[];allloo=[]
 for d in rank['within_subtype_reference']:
  group=d['group'];keep=np.isin(stages,[0,3])&(n[:,group]>=20);ix=np.flatnonzero(keep);hi=(stages[ix]==3).astype(float);a=(age[ix]-age[ix].mean())/age[ix].std();columns=[np.ones(len(ix)),hi,a,male[ix],apoe[ix]];names=['intercept','High_vs_NotAD','standardized_age','Male','APOE4_dose']
  if use_pmi:columns.append((pmi[ix]-pmi[ix].mean())/pmi[ix].std());names.append('standardized_PMI')
  X=np.column_stack(columns);Y=sums[ix,group]/n[ix,group,None];beta=fit(X,Y);loo=[];excluded=[]
  for j in range(len(ix)):
   k=np.arange(len(ix))!=j
   try:loo.append(fit(X[k],Y[k]))
   except AssertionError:excluded.append(str(donors[ix[j]]))
  assert loo
  allbeta.append(beta);allloo.append(np.stack(loo));pergroup.append(dict(group=group,donors=len(ix),NotAD=int((hi==0).sum()),High=int((hi==1).sum()),design_rank=int(np.linalg.matrix_rank(X)),parameters=names,condition_number=float(np.linalg.cond(X)),maximum_hat_leverage=float(np.diag(X@np.linalg.pinv(X)).max()),leave_one_donor_fits=len(loo),rank_deficient_deletions_excluded=excluded))
 avg=np.stack(allbeta).mean(0);stability=np.concatenate([np.sign(v)==np.sign(avg) for v in allloo]).mean(0);original={x['gene']:x for x in rank['within_subtype_ranking']};rows=[]
 for j,gene in enumerate(genes):
  base=original[str(gene)];real=base['real_high_minus_NotAD_mean_log1p_panelCP10k'];model=base['model_high_minus_NotAD_mean_log1p_panelCP10k'];rows.append(dict(gene=str(gene),adjusted_High_vs_NotAD_coefficient=float(avg[j]),subtype_coefficients=[float(b[j]) for b in allbeta],groupwise_adjusted_donor_deletion_sign_agreement=float(stability[j]),unadjusted_reference_effect=real,model_effect=model,adjusted_sign_agrees_with_unadjusted=bool(np.sign(avg[j])==np.sign(real)),adjusted_sign_agrees_with_model=bool(model is not None and np.sign(avg[j])==np.sign(model)),original_prioritization_heuristic=base['prioritization_heuristic']))
 result=dict(status='DONOR_LEVEL_COVARIATE_SENSITIVITY_COMPLETE_NOT_CAUSAL',UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),plan_SHA256=sha(PLAN),source_SHA256=sha(__file__),original_ranking_unchanged=True,groups=pergroup,PMI_included=use_pmi,rows=rows,pvalues_computed=False,limitations=['Only3 NotAD donors; covariate adjustment is unstable and does not establish causality.','NoAPOE4 carriers amongNotAD reference donors; adjustment requires assumptions and is not genotype-stratified validation.','Model conditioning remains stage only and cannot separate these covariate effects.','RNA panel-normalized means are observational associations rather than perturbation responses.'])
 OUTPUT.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n');print(json.dumps({'status':result['status'],'groups':pergroup,'genes':len(rows)}))
if __name__=='__main__':main()
