"""Combined exploratory selection;16 old endpoints are labeled observed controls."""
from released097_selection import summarize,proposed_confirmation,require
def summarize_local(candidates,new_rows,control_rows):
 require(1<=len(candidates)<=2,'Only the frozen one/two follow-up settings allowed')
 old_ids={'f000','f001',*[f't{i:03}' for i in range(6)]}
 require(len(control_rows)==16 and {(r['candidate_id'],r['seed']) for r in control_rows}=={(c,s) for c in old_ids for s in (0,1)},'All16 original observed endpoints required')
 require(len(new_rows)==2*len(candidates) and {(r['candidate_id'],r['seed']) for r in new_rows}=={(c['id'],s) for c in candidates for s in (0,1)},'Every new paired endpoint required')
 require(not ({c['id'] for c in candidates}&old_ids),'New and previous controls must be distinct')
 for r in control_rows:require(r['read_only_previously_measured_reference'] and r['new_model_fitted'] is False and r['new_GPU_charge']==0 and r['source_result_path'] and r['source_result_SHA256'],'Observed controls cannot be warm starts or new charges')
 for r in new_rows:require(not r.get('read_only_previously_measured_reference',False) and r.get('new_model_fitted',True) is not False and r['charged_GPU_seconds']>0,'Each new outcome must have its own real fit/cost')
 choices=[dict(candidate_id=c['id'],hyperparameters=c['hyperparameters']) for c in candidates]+[dict(candidate_id=cid,hyperparameters=next(r['hyperparameters'] for r in control_rows if r['candidate_id']==cid)) for cid in sorted(old_ids)]
 summary=summarize(choices,[*new_rows,*control_rows],[0,1]);summary['reference_id']='f000';summary['proposed_confirmation_configurations']=proposed_confirmation(summary,'f000')
 for c in summary['candidates']:c['previously_measured_control']=c['candidate_id'] in old_ids
 summary.update(required_new_fits=2*len(candidates),previously_measured_control_fits=16,controls_refitted=False,control_seeds_previously_observed=True,independent_population_confirmation=False)
 return summary
