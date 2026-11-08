"""Export anonymous gene-level stage means from immutable inference summaries.

Only needed to regenerate the committed aggregate profiles from the original
inference artifacts. Public report generation itself never needs raw cells.
"""
import argparse,hashlib,json
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def run(state):
 completion=json.loads((state/'completed.json').read_text());assert completion['generated_cells']==12800000 and len(completion['stage_replicates'])==160
 with np.load(state/'real_training_donor_summaries.npz',allow_pickle=False) as z:
  genes=z['genes'];sums=z['sums'];n=z['n'];stage=z['stages']
 groups=[2,4];model=np.full((5,8,4,2000),np.nan);valid=np.zeros((5,8,4),bool)
 for row in completion['stage_replicates']:
  p=Path(row['path']);assert sha(p)==row['SHA256']
  with np.load(p,allow_pickle=False) as z:
   s,r,t=row['seed'],row['replicate'],row['stage'];valid[s,r,t]=bool((z['n'][groups]>=100).all())
   if valid[s,r,t]:model[s,r,t]=(z['sums'][groups]/z['n'][groups,None]).mean(0)
 means=np.full((5,4,2000),np.nan)
 for s in range(5):
  for t in range(4):
   if valid[s,:,t].all():means[s,t]=model[s,:,t].mean(0)
 real=np.full((4,2000),np.nan);counts=[]
 for t in range(4):
  values=[];detail=[]
  for g in groups:
   keep=(stage==t)&(n[:,g]>=20);detail.append(int(keep.sum()))
   if keep.sum()>=3:values.append((sums[keep,g]/n[keep,g,None]).mean(0))
  counts.append(detail)
  if len(values)==len(groups):real[t]=np.mean(values,axis=0)
 def finite(x):return round(float(x),8) if np.isfinite(x) else None
 out=dict(schema=1,stage_order=['Not AD','Low','Intermediate','High'],fixed_groups=groups,group_names=['Micro-PVM_2','Micro-PVM_2_3-SEAAD'],real_eligible_donors_by_stage_group=counts,model_valid_replicates_by_seed_stage=valid.sum(1).tolist(),groups_are_analysis_assignments_not_generator_inputs=True,model_groups_are_classifier_predictions=True,units='Mean per-cell log1p(panel CP10k), equal weights over the two eligible subtypes.',cross_sectional_not_longitudinal_or_intervention_response=True,input_completion_SHA256=sha(state/'completed.json'),real_summary_SHA256=sha(state/'real_training_donor_summaries.npz'),source_SHA256=sha(Path(__file__)),profiles={str(g):dict(real=[finite(v) for v in real[:,j]],model_by_seed=[[finite(v) for v in means[s,:,j]] for s in range(5)]) for j,g in enumerate(genes)})
 dest=ROOT/'docs/evidence/stage-profiles.json';dest.write_text(json.dumps(out,separators=(',',':'),allow_nan=False)+'\n');print(json.dumps({'genes':len(genes),'output':str(dest),'eligible_donors_by_stage_group':counts,'valid_model_replicates':valid.sum(1).tolist()}))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--state',type=Path,required=True);run(p.parse_args().state)
