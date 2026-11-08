"""Cross-cohort direction checks for the completed exploratory model ranking.

Published association evidence never determines a causal intervention direction.
"""
import argparse,csv,hashlib,json,math
from pathlib import Path
ROOT=Path('/ml2/alz');FOLDER=ROOT/'artifacts/primamente-external-evidence'
RANKING=ROOT/'analysis/primamente_gene_inference_exploratory_ranking_20261108.json'
MANIFEST=ROOT/'analysis/primamente_external_validation_preparation_20261108.json'
OUTPUT=ROOT/'analysis/primamente_model_gene_ROSMAP_crosscheck_20261108.json'
def read(p):return json.loads(Path(p).read_bytes())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def finite(x):
 try:v=float(x)
 except (TypeError,ValueError):return None
 return v if math.isfinite(v) else None
def sign(x):return (x>0)-(x<0)
def load_table(item):
 assert sha(item['download_path'])==item['SHA256']
 rows=list(csv.DictReader(Path(item['download_path']).open()));table={}
 for row in rows:
  assert row['gene'] not in table and row['coef']==item['contrast']
  table[row['gene']]=dict(logFC=finite(row['logFC']),published_global_adjusted_p=finite(row['p_adj.glb']),published_local_adjusted_p=finite(row['p_adj.loc']))
 assert len(table)==item['rows']
 return table
def evidence(gene,expected,table):
 if gene not in table:return dict(measured_in_external_table=False,association_support=False)
 q=dict(table[gene]);value=q['logFC'];p=q['published_global_adjusted_p'];agree=expected is not None and value is not None and sign(expected)!=0 and sign(expected)==sign(value)
 q.update(measured_in_external_table=True,direction_agrees=agree,association_support=bool(agree and p is not None and p<.05))
 return q
def crosscheck(rows,tables):
 out=[]
 for row in rows:
  x=dict(row);expected=x['real_high_minus_NotAD_mean_log1p_panelCP10k'];x['ROSMAP']={name:evidence(x['gene'],expected,table) for name,table in tables.items()}
  x['model_and_primary_cohort_association_supported']=bool(x['prioritization_heuristic']>0 and x['ROSMAP']['gpath']['association_support']);out.append(x)
 return sorted(out,key=lambda x:(not x['model_and_primary_cohort_association_supported'],-x['prioritization_heuristic'],x['gene']))
def check():
 t={'up':dict(logFC=1.,published_global_adjusted_p=.01),'down':dict(logFC=-1.,published_global_adjusted_p=.01),'weak':dict(logFC=1.,published_global_adjusted_p=.2),'invalid':dict(logFC=None,published_global_adjusted_p=None)}
 assert evidence('up',1,t)['association_support']
 for g in ['down','weak','invalid','absent']:assert not evidence(g,1,t)['association_support']
 assert not evidence('up',None,t)['association_support'];assert finite('NA') is None and finite('nan') is None
 assert not RANKING.exists(),'This preparation check must precede completed model ranking'
 tables={x['contrast']:load_table(x) for x in read(MANIFEST)['tables']};print(json.dumps(dict(direction_and_adjusted_evidence_checked=True,missing_and_invalid_never_support=True,all_three_actual_author_tables_hashes_and_schema_checked=True,table_genes={k:len(v) for k,v in tables.items()},model_ranking_not_read=True)))
def main():
 p=argparse.ArgumentParser();p.add_argument('--run',action='store_true');p.add_argument('--check',action='store_true');a=p.parse_args()
 if a.check:check();return
 assert a.run and RANKING.exists(),'Completed frozen inference and model ranking required before external join'
 q=read(RANKING);assert q['status']=='COMPLETE_EXPLORATORY_STAGE_ASSOCIATION_RANKING_NOT_CAUSAL_TARGET_PREDICTION'
 completion=ROOT/'artifacts/primamente-t005-gene-inference-20261108/completed.json';assert sha(completion)==q['input_completion_SHA256']
 assert not OUTPUT.exists();m=read(MANIFEST);tables={x['contrast']:load_table(x) for x in m['tables']}
 pooled=crosscheck(q['pooled_ranking'],tables);within=crosscheck(q['within_subtype_ranking'],tables)
 result=dict(status='MODEL_TRAINING_DONOR_AND_ROSMAP_ASSOCIATION_CROSSCHECK_COMPLETE_PERTURBATION_REVIEW_REMAINS',primary_external_trait='gpath',secondary_external_traits=['amyloid','tangles'],secondary_traits_not_independent_replications=True,published_association_support_rule='Primarygpath direction agrees with realHigh-minus-NotAD contrast and published global adjustedp<.05. Model heuristic must separately pass its fixed stability checks. This is not a newly computed significance test.',pooled_supported_genes=sum(x['model_and_primary_cohort_association_supported'] for x in pooled),within_subtype_supported_genes=sum(x['model_and_primary_cohort_association_supported'] for x in within),pooled_rows=pooled,within_subtype_rows=within,source_SHA256=sha(__file__),input_SHA256={str(RANKING):sha(RANKING),str(MANIFEST):sha(MANIFEST),str(completion):sha(completion)},limitations=['Expression associations do not determine therapeutic inhibition or activation.','SEA-AD andROSMAP have distinct study provenance; no subject-level linkage audit available.','The exactDecontX preprocessing that produced the author tables has not been reproduced.','The three external traits come from one cohort and are not independent replications.','Published adjustedpvalues describe external associations, not significance of our model-selected shortlist.'],gene_or_compound_selected=False,perturbation_and_functional_safety_review_required=True)
 OUTPUT.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n');print(json.dumps({k:v for k,v in result.items() if k not in ['pooled_rows','within_subtype_rows']}))
if __name__=='__main__':main()
