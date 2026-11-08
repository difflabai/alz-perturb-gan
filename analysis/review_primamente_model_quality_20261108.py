"""Read-only diagnostics of the completed stage-association study and prior calibration."""
import importlib.util,json,datetime
from pathlib import Path
ROOT=Path('/ml2/alz')
spec=importlib.util.spec_from_file_location('inference',ROOT/'analysis/infer_t005_gene_hypotheses_cpu.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
np,torch=m.np,m.torch
from scipy.stats import spearmanr
STATE=ROOT/'artifacts/primamente-t005-gene-inference-recovery-v1-20261108'
RANKING=ROOT/'analysis/primamente_gene_inference_exploratory_ranking_20261108.json'
OUTPUT=ROOT/'analysis/primamente_model_capability_review_20261108.json'
def main():
 torch.set_num_threads(1);torch.set_grad_enabled(False);assert not torch.cuda.is_initialized() and not OUTPUT.exists()
 done=m.read(STATE/'completed.json');rank=m.read(RANKING);assert done['generated_cells']==12800000 and rank['input_completion_SHA256']==m.sha(STATE/'completed.json')
 plan=m.read(m.PLAN)
 for p,pin in plan['input_SHA256'].items():assert m.sha(p)==pin,p
 ctx=m.Context(Path(plan['fitting_cache']),'cpu')
 with np.load(ctx.cache/'train/metadata.npz',allow_pickle=False) as z:
  labels=z['type'];donors=z['donor'];stages=np.rint(z['raw_c'][:,3]*3).astype(int)
 confusion=np.zeros((6,6),np.int64);realconf=[]
 for start in range(0,len(ctx.X),2000):
  prob=ctx.space.clf(ctx.trainpcs[start:start+2000]).softmax(-1);pred=prob.argmax(-1).numpy();truth=labels[start:start+2000];np.add.at(confusion,(truth,pred),1);realconf.extend(prob.max(-1).values.tolist())
 with np.load(STATE/'real_training_donor_summaries.npz',allow_pickle=False) as z:
  rn=z['n'];realmix=rn[:,1:]/rn[:,0,None]
 true_stage=np.stack([realmix[stages==s].mean(0) for s in range(4)])
 mix=np.empty((5,8,4,6));conf=np.empty((5,8,4))
 for row in done['stage_replicates']:
  assert m.sha(row['path'])==row['SHA256']
  with np.load(row['path'],allow_pickle=False) as z:
   s,r,t=row['seed'],row['replicate'],row['stage'];mix[s,r,t]=z['n'][1:]/z['n'][0];conf[s,r,t]=z['classifier_mean_confidence']
 tv=.5*np.abs(mix.mean(1)-true_stage[None]).sum(-1)
 agreement={}
 for key in ['pooled_ranking','within_subtype_ranking']:
  rows=rank[key];x=np.array([v['real_high_minus_NotAD_mean_log1p_panelCP10k'] for v in rows],float);y=np.array([v['model_high_minus_NotAD_mean_log1p_panelCP10k'] for v in rows],float);valid=np.isfinite(x)&np.isfinite(y)
  agreement[key]=dict(genes=int(valid.sum()),descriptive_spearman_across_genes=float(spearmanr(x[valid],y[valid]).statistic),direction_agreement=float((np.sign(x[valid])==np.sign(y[valid])).mean()),model_to_real_effect_L2_ratio=float(np.linalg.norm(y[valid])/np.linalg.norm(x[valid])),genes_passing_fixed_heuristic=sum(v['prioritization_heuristic']>0 for v in rows),these_genes_are_correlated_not_independent_replicates=True)
 calibration=[]
 for item in plan['models']:
  q=m.read(item['original_result_path']);assert q['calibration_only']
  calibration.append(dict(seed=item['seed'],PG={k:q['metrics'][k] for k in ['cell_energy_distance','energy_score','coverage80','spread_ratio']},resampling={k:q['baseline_metrics'][k] for k in ['cell_energy_distance','energy_score','coverage80','spread_ratio']}))
 means={method:{metric:float(np.mean([v[method][metric] for v in calibration])) for metric in calibration[0][method]} for method in ['PG','resampling']}
 result=dict(status='FROZEN_MODEL_CAPABILITY_AND_LIMITATION_REVIEW_COMPLETE',UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),new_training_fits=0,new_optimizer_updates=0,CUDA_initialized=False,stage_order=['Not AD','Low','Intermediate','High'],calibration_per_seed=calibration,calibration_mean_over5seeds=means,calibration_biological_donors=4,calibration_is_selection_data_not_external_generalization=True,real_training_classifier_confusion=confusion.tolist(),real_training_classifier_accuracy=float(np.trace(confusion)/confusion.sum()),real_training_classifier_balanced_accuracy=float(np.mean(confusion.diagonal()/confusion.sum(1))),real_training_classifier_confidence_mean=float(np.mean(realconf)),classifier_training_accuracy_not_out_of_sample_validation=True,generated_classifier_confidence_mean_by_stage=conf.mean((0,1)).tolist(),generated_classifier_confidence_range=[float(conf.min()),float(conf.max())],real_equal_donor_stage_mixture=true_stage.tolist(),generated_seed_mean_mixtures=mix.mean(1).tolist(),mixture_total_variation_per_seed_stage=tv.tolist(),mixture_total_variation_mean_by_stage=tv.mean(0).tolist(),mixture_comparison_exposure_limitation='Synthetic stage conditions use common pooled training library-depth templates; real stage subsets have their actual stage-specific depths.',gene_contrast_descriptive_agreement=agreement,individual_gene_constraints='Adversarial training uses bag features from50PCs; accurate2000gene reconstruction is not established by a low projected distribution distance.',observational_conditioning='Only ADNC stage slot is supplied; no drug,gene,dose or intervention conditioning exists.',known_risk_genes_missing_from_measured_panel=plan['current_model_scope']['known_unmeasured_targets'],source_and_input_SHA256={str(Path(__file__)):m.sha(__file__),str(RANKING):m.sha(RANKING),str(STATE/'completed.json'):m.sha(STATE/'completed.json')},conclusion='Suitable for exploratory program/association prioritization only. Neither intervention efficacy, causal therapeutic direction, nor SOTA is established by this study.')
 m.write(OUTPUT,result);print(json.dumps({k:result[k] for k in ['status','calibration_mean_over5seeds','gene_contrast_descriptive_agreement','mixture_total_variation_mean_by_stage']}))
if __name__=='__main__':main()
