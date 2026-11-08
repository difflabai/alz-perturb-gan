"""Exploratory association ranking after the complete frozen inference batch.

The ranking is not a causal model, a statistical discovery claim or a drug score.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT=Path('/ml2/alz');PROJECT=Path('/ml2/codex-storage/future-workspaces/2026-10-27/task-2/benchmark')
sys.path[:0]=[str(PROJECT/'current_base_20261029/runtime/lib/python3.12/site-packages'),str(PROJECT/'.venv312/lib/python3.12/site-packages')]
import numpy as np
STATE=ROOT/'artifacts/primamente-t005-gene-inference-recovery-v1-20261108'
OUTPUT=ROOT/'analysis/primamente_gene_inference_exploratory_ranking_20261108.json'
PANELS={
 'lipid_handling':['LPL','PLIN2','APOE','ABCA1','ABCG1','ACSL1','DGAT1','DGAT2','SOAT1','SCD','FABP5','NPC1','NPC2'],
 'lysosome_and_phagocytosis':['CTSB','CTSD','CTSL','LAMP1','LAMP2','CST7','TREM2','TYROBP','AXL','MERTK','GRN','GPNMB'],
 'inflammatory_markers':['CCL2','CCL3','CCL4','CCL5','CXCL8','IL1B','TNF','NLRP3','SPP1','LGALS3'],
 'homeostatic_markers':['P2RY12','TMEM119','CX3CR1','SALL1','GPR34','OLFML3']}
def read(p):return json.loads(Path(p).read_bytes())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def readiness():
    if not (STATE/'completed.json').exists():return False
    q=read(STATE/'completed.json')
    assert q['generated_cells']==12800000 and len(q['stage_replicates'])==160 and q['source_weights_unchanged'] and not q['evaluation_or_calibration_targets_read']
    assert not(STATE/'failure.json').exists()
    seen=set()
    for x in q['stage_replicates']:
        key=x['seed'],x['replicate'],x['stage'];assert key not in seen;seen.add(key);assert sha(x['path'])==x['SHA256']
    assert seen=={(s,r,t) for s in range(5) for r in range(8) for t in range(4)}
    return True
def difference(means,stages,keep):
    lo=keep&(stages==0);hi=keep&(stages==3)
    assert lo.sum() and hi.sum()
    return means[hi].mean(0)-means[lo].mean(0)
def donor_support(sums,n,stages,groups):
    effects=[];loo=[];details=[]
    for g in groups:
        keep=n[:,g]>=(1 if g==0 else 20)
        if min(int((keep&(stages==s)).sum()) for s in (0,3))<3:continue
        means=sums[:,g]/np.maximum(n[:,g,None],1)
        effect=difference(means,stages,keep);effects.append(effect)
        leave=[]
        for i in np.flatnonzero(keep&np.isin(stages,[0,3])):
            k=keep.copy();k[i]=False;leave.append(difference(means,stages,k))
        loo.append(np.stack(leave))
        details.append(dict(group=g,NotAD_donors=int((keep&(stages==0)).sum()),High_donors=int((keep&(stages==3)).sum()),min_cells_per_donor=1 if g==0 else 20))
    if not effects:return None,None,details
    # Equal fixed weight for each eligible subtype; equal real-donor weight within subtype/stage.
    effect=np.stack(effects).mean(0)
    # Group-wise leave-one-donor sign stability is conservative and transparent.
    consistency=np.concatenate([np.sign(v)==np.sign(effect) for v in loo]).mean(0)
    return effect,consistency,details
def measure(model,counts,real,loo,groups):
    # Require every fixed eligible group to have100 synthetic cells in each stage/replicate.
    means=model[:,:,:,groups,:].mean(3)
    eligible=(counts[:,:,:,groups]>=100).all(3)
    effects=means[:,:,3]-means[:,:,0];valid=eligible[:,:,3]&eligible[:,:,0]
    effects=np.where(valid[:,:,None],effects,np.nan)
    count=np.isfinite(effects).sum(1);seedmean=np.divide(np.nansum(effects,axis=1),count,out=np.full((5,2000),np.nan),where=count>0)
    all_models=np.isfinite(seedmean).all(0);avg=np.where(all_models,seedmean.mean(0),np.nan)
    seed_agreement=(np.sign(seedmean)==np.sign(real)).sum(0)
    rep_agreement=np.sum((np.sign(effects)==np.sign(real))&np.isfinite(effects),axis=(0,1))/40
    score=np.where(all_models,np.minimum(np.abs(avg),np.abs(real))*(seed_agreement/5)*loo,0.)
    score=np.where((seed_agreement>=4)&(rep_agreement>=.8)&(loo>=.8)&(np.sign(avg)==np.sign(real)),score,0.)
    return dict(real_effect=real,real_leave_one_donor_sign_agreement=loo,model_mean_effect=avg,model_seed_mean_effects=seedmean,model_seeds_agreeing_with_real=seed_agreement,model_replicates_agreeing_with_real=rep_agreement,heuristic_score=score,synthetic_groups_complete_in_all_models=all_models,valid_model_replicate_cases=int(valid.sum()))
def number(v):return float(v) if np.isfinite(v) else None
def rows(genes,stats):
    result=[]
    for j,gene in enumerate(genes):
        result.append(dict(gene=str(gene),real_high_minus_NotAD_mean_log1p_panelCP10k=number(stats['real_effect'][j]),model_high_minus_NotAD_mean_log1p_panelCP10k=number(stats['model_mean_effect'][j]),model_seed_effects=[number(v) for v in stats['model_seed_mean_effects'][:,j]],model_seeds_agreeing_with_real=int(stats['model_seeds_agreeing_with_real'][j]),model_replicate_sign_agreement=float(stats['model_replicates_agreeing_with_real'][j]),real_leave_one_donor_sign_agreement=float(stats['real_leave_one_donor_sign_agreement'][j]),prioritization_heuristic=float(stats['heuristic_score'][j]),synthetic_groups_complete_in_all_models=bool(stats['synthetic_groups_complete_in_all_models'][j])))
    return sorted(result,key=lambda x:(-x['prioritization_heuristic'],x['gene']))
def check():
    assert not readiness(),'Check is intended to verify premature-read refusal before completed inference'
    n=np.full((8,7),25);stages=np.array([0]*4+[3]*4);means=np.zeros((8,7,2));means[4:]=2.;sums=means*n[:,:,None]
    effect,stability,details=donor_support(sums,n,stages,[1,2]);np.testing.assert_equal(effect,[2,2]);np.testing.assert_equal(stability,[1,1]);assert len(details)==2
    n[:,2]=0;effect,_,details=donor_support(sums,n,stages,[1,2]);assert len(details)==1
    model=np.zeros((5,8,4,7,2000));model[:,:,3]=2;counts=np.full((5,8,4,7),1000)
    q=measure(model,counts,np.full(2000,2.),np.ones(2000),[1]);assert q['valid_model_replicate_cases']==40;np.testing.assert_equal(q['heuristic_score'],2.)
    counts[0,:,:,1]=0;q=measure(model,counts,np.full(2000,2.),np.ones(2000),[1]);assert not q['synthetic_groups_complete_in_all_models'].any();assert not q['heuristic_score'].any()
    counts[:]=1000;model[:,:,3]=.1;model[0,:,3]=-5
    q=measure(model,counts,np.full(2000,2.),np.ones(2000),[1]);assert (q['model_seeds_agreeing_with_real']==4).all() and not q['heuristic_score'].any()
    print(json.dumps(dict(premature_numerical_read_refused=True,equal_donor_subtype_contrast_checked=True,missing_real_subtype_excluded_by_fixed_rule=True,seed_and_replicate_agreement_checked=True,missing_synthetic_subtype_never_silently_dropped=True,mean_direction_reversal_from_one_large_outlier_receives_zero_priority=True)))
def main():
    p=argparse.ArgumentParser();p.add_argument('--run',action='store_true');p.add_argument('--check',action='store_true');a=p.parse_args()
    if a.check:check();return
    assert a.run and readiness(),'Require all five frozen models and all160 stage replicates before numerical analysis'
    assert not OUTPUT.exists()
    with np.load(STATE/'real_training_donor_summaries.npz',allow_pickle=False) as z:
        genes=z['genes'];rs=z['sums'];rn=z['n'];stages=z['stages'];donors=z['donors']
    assert len(genes)==2000
    pooled,ploo,pdetail=donor_support(rs,rn,stages,[0]);within,wloo,wdetail=donor_support(rs,rn,stages,list(range(1,7)))
    model=np.full((5,8,4,7,2000),np.nan);counts=np.zeros((5,8,4,7),np.int64)
    for s in range(5):
        for rep in range(8):
            for stage in range(4):
                with np.load(STATE/f'seed{s}_rep{rep}_stage{stage}.npz',allow_pickle=False) as z:
                    count=z['n'];counts[s,rep,stage]=count
                    model[s,rep,stage]=np.divide(z['sums'],count[:,None],out=np.full((7,2000),np.nan),where=count[:,None]>0)
    result=dict(status='COMPLETE_EXPLORATORY_STAGE_ASSOCIATION_RANKING_NOT_CAUSAL_TARGET_PREDICTION',generated_cells=12800000,independent_model_initializations=5,real_optimization_donors=len(donors),biological_replicates_added_by_simulation=0,contrast='High minus Not AD; difference in mean per-cell log1p10k-normalized2000HVG expression, not log fold-change',donor_unit='Equal real-donor weight per subtype/stage. Pooling synthetic cells is not a biological confidence interval.',minimum_real_donors_per_stage=3,minimum_real_cells_per_subtype_donor=20,minimum_synthetic_cells_per_stage_replicate_subtype=100,prioritization_heuristic='min(abs(real effect),abs(mean5seed model effect))*fraction5seed signs agreeing with real*real leave-one-donor sign agreement; zero unless >=4/5 seeds,>=80% of40replicates,>=80% real donor deletion checks and the5seed mean direction agree. Exploratory effect-size heuristic; no significance or causal claim.',pooled_reference=pdetail,within_subtype_reference=wdetail,predicted_subtype_mixture_mean=counts[:,:,:,1:].mean((0,1)).tolist(),predicted_subtypes_are_not_observed_truth=True,panels={name:dict(measured=[g for g in members if g in genes],not_measured=[g for g in members if g not in genes],formal_pathway_enrichment=False) for name,members in PANELS.items()},pooled_ranking=rows(genes,measure(model,counts,pooled,ploo,[0])) if pooled is not None else [],within_subtype_ranking=rows(genes,measure(model,counts,within,wloo,[d['group'] for d in wdetail])) if within is not None else [],input_completion_SHA256=sha(STATE/'completed.json'),source_SHA256=sha(__file__),limitations=['One33-donor optimization fold; age,sex,genotype and other covariates are not causally controlled.','Generated subtype assignments use a training-only classifier and may be inaccurate.','HVG filtering omits many actionable risk genes; absence is not evidence of absence.','Model predictions describe observational stage conditions; they do not simulate genes,drugs,doses or intervention response.','Candidate and intervention direction require external primary perturbation evidence and subsequent wet-lab testing.'],literature_validation_pending=True,therapeutic_target_chosen=False)
    OUTPUT.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n');print(json.dumps(dict(output=str(OUTPUT),pooled_genes=len(result['pooled_ranking']),within_subtype_genes=len(result['within_subtype_ranking']),therapeutic_target_chosen=False)))
if __name__=='__main__':main()
