"""Paired local exploration with explicitly previously measured native controls."""
from released097_selection import summarize,proposed_confirmation,require

def summarize_local(candidates,new_rows,control_rows):
 require(2<=len(candidates)<=6,'Frozen local study must bracket the prior weight at minimum')
 require(len(control_rows)==4 and {(r['candidate_id'],r['seed']) for r in control_rows}=={(c,s) for c in ('f000','f001') for s in (0,1)},'Both corrected baseline settings at both seeds required')
 require(len(new_rows)==2*len(candidates) and {(r['candidate_id'],r['seed']) for r in new_rows}=={(c['id'],s) for c in candidates for s in (0,1)},'Every new paired endpoint required')
 require(not ({c['id'] for c in candidates}&{'f000','f001'}),'Previously measured references cannot be relabeled as new trials')
 for r in control_rows:
  require(r['read_only_previously_measured_reference'] and r['new_model_fitted'] is False and r['new_GPU_charge']==0 and r['source_result_path'] and r['source_result_SHA256'],'Measured control provenance and zero new charge required')
 for r in new_rows:require(not r.get('read_only_previously_measured_reference',False) and r.get('new_model_fitted',True) is not False and r['charged_GPU_seconds']>0,'New fit cannot be substituted with a zero-charge reference')
 choices=[dict(candidate_id=c['id'],hyperparameters=c['hyperparameters']) for c in candidates]
 choices += [dict(candidate_id=cid,hyperparameters=next(r['hyperparameters'] for r in control_rows if r['candidate_id']==cid)) for cid in ('f000','f001')]
 summary=summarize(choices,[*new_rows,*control_rows],[0,1]);summary['reference_id']='f000';summary['proposed_confirmation_configurations']=proposed_confirmation(summary,'f000')
 for c in summary['candidates']:c['previously_measured_control']=c['candidate_id'] in ('f000','f001')
 summary.update(required_new_fits=2*len(candidates),previously_measured_control_fits=4,controls_refitted=False,control_seeds_previously_observed=True,independent_population_confirmation=False)
 return summary
