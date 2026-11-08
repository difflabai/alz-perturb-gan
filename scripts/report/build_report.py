"""Build a portable public evidence report from committed aggregate research results.

No GPU, raw cells, donor identifiers, model weights, or native runtime is needed.
"""
from pathlib import Path
import argparse,csv,hashlib,json,math,statistics,colorsys,tempfile
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
ROOT=Path(__file__).resolve().parents[2];DOCS=ROOT/'docs';EVIDENCE=DOCS/'evidence';ASSETS=DOCS/'assets';ANALYSIS=ROOT/'analysis'
FILES={
 'ranking':'primamente_gene_inference_exploratory_ranking_20261108.json',
 'cohort':'primamente_model_gene_ROSMAP_crosscheck_20261108.json',
 'sensitivity':'primamente_donor_covariate_sensitivity_20261108.json',
 'quality':'primamente_model_capability_review_20261108.json',
 'recommendation':'primamente_registration_recommendation_20261108.json',
 'plan':'primamente_inference_hypothesis_plan_20261108.json',
 'covariate_plan':'primamente_training_donor_covariate_sensitivity_plan_20261108.json',
 'markers':'primamente_published_state_marker_coverage_20261108.json',
 'audit':'primamente_selected_CLEC5A_numerical_audit_20261108.json'}
BRAND=json.loads((DOCS/'brand/tokens.json').read_bytes())
BG=BRAND['color']['brand']['navy']['hex'];TEAL=BRAND['color']['brand']['amber-bright']['hex'];INK=BRAND['color']['brand']['paper']['hex'];RUST=BRAND['color']['topo']['peak']['hex']
def hsl_hex(value):
 h,s,l=map(float,value.replace('hsl(','').replace(')','').replace('%','').split());rgb=colorsys.hls_to_rgb(h/360,l/100,s/100);return '#'+''.join(f'{round(v*255):02x}' for v in rgb)
MUTED=hsl_hex(BRAND['color']['semantic']['dark']['fg-muted']['$value']);PALE=hsl_hex(BRAND['color']['semantic']['dark']['border']['$value'])
def read(p):return json.loads(Path(p).read_bytes())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def rounded(v):return round(float(v),8) if v is not None and math.isfinite(v) else None
def save(fig,name):
 svg=ASSETS/(name+'.svg')
 fig.savefig(svg,bbox_inches='tight',facecolor=BG,metadata={'Date':None})
 svg.write_text('\n'.join(line.rstrip() for line in svg.read_text().splitlines())+'\n')
 fig.savefig(ASSETS/(name+'.png'),dpi=170,bbox_inches='tight',facecolor=BG,metadata={'Software':'Matplotlib'})
 plt.close(fig)
def style():
 from matplotlib import font_manager
 from fontTools.ttLib import TTFont
 from fontTools.varLib.instancer import instantiateVariableFont
 font_path=next((DOCS/'brand').glob('inter-*.woff2'))
 global FONT_CACHE
 FONT_CACHE=tempfile.TemporaryDirectory(prefix='alz-report-font-')
 for weight in [400,700]:
  font=TTFont(font_path);font.flavor=None
  if 'fvar' in font:font=instantiateVariableFont(font,{'wght':weight},inplace=True)
  target=Path(FONT_CACHE.name)/f'Inter-{weight}.ttf';font.save(target);font_manager.fontManager.addfont(str(target))
 plt.rcParams.update({'font.family':'Inter','figure.facecolor':BG,'axes.facecolor':BG,'font.size':10,'axes.labelcolor':INK,'text.color':INK,'xtick.color':MUTED,'ytick.color':MUTED,'axes.spines.top':False,'axes.spines.right':False,'axes.edgecolor':PALE,'axes.titleweight':'bold','svg.fonttype':'path','svg.hashsalt':'alz-evidence-report-20261108'})
def data_export():
 q={k:read(ANALYSIS/name) for k,name in FILES.items()};cov={x['gene']:x for x in q['sensitivity']['rows']};cohort={x['gene']:x for x in q['cohort']['within_subtype_rows']};pooled={x['gene']:x for x in q['ranking']['pooled_ranking']};rows=[]
 for rank,x in enumerate(q['ranking']['within_subtype_ranking'],1):
  gene=x['gene'];s=cov[gene];e=cohort[gene];p=pooled[gene];external=e['ROSMAP']['gpath'];checks=s['stage_stratified_donor_deletion_checks']
  robust=bool(e['model_and_primary_cohort_association_supported'] and s['adjusted_sign_agrees_with_unadjusted'] and all(c['adjusted_sign_agreement'] is not None and c['adjusted_sign_agreement']>=.8 and c['unadjusted_sign_agreement']>=.8 for c in checks))
  rows.append(dict(gene=gene,rank=rank,model_effect=rounded(x['model_high_minus_NotAD_mean_log1p_panelCP10k']),real_effect=rounded(x['real_high_minus_NotAD_mean_log1p_panelCP10k']),seed_effects=[rounded(v) for v in x['model_seed_effects']],adjusted_effect=rounded(s['adjusted_High_vs_NotAD_coefficient']),seeds_agree=x['model_seeds_agreeing_with_real'],mc_agreement=x['model_replicate_sign_agreement'],heuristic=rounded(x['prioritization_heuristic']),donor_deletion_agreement=x['real_leave_one_donor_sign_agreement'],stable=x['prioritization_heuristic']>0,rosmap_logFC=external.get('logFC'),rosmap_fdr=external.get('published_global_adjusted_p'),rosmap_support=e['model_and_primary_cohort_association_supported'],robust_sensitivity=robust,minimum_control_deletion_agreement=min(c['adjusted_sign_agreement'] for c in checks if c['deleted_donor_stage']==0),pooled_model_effect=rounded(p['model_high_minus_NotAD_mean_log1p_panelCP10k']),pooled_real_effect=rounded(p['real_high_minus_NotAD_mean_log1p_panelCP10k'])))
 quality=q['quality'];profiles=read(EVIDENCE/'stage-profiles.json');donors=q['covariate_plan']['donors'];covariates=[]
 for stage in ['Not AD','Low','Intermediate','High']:
  subset=[v for v in donors.values() if v['Overall AD neuropathological Change']==stage]
  covariates.append(dict(stage=stage,n=len(subset),male=sum(v['Sex']=='Male' for v in subset),apoe4_carriers=sum('4' in v['APOE Genotype'].split('/') for v in subset),mean_age=statistics.mean(float(v['Age at Death']) for v in subset)))
 data=dict(schema=1,snapshot_date='2026-11-08',training_dataset='SEA-AD middle temporal gyrus',challenge_atlas_used=False,challenge_context='Independent proof of concept; Prima Mente atlas planned for Spring 2027',research_snapshot_commit='89fa5ea',repository='https://github.com/difflabai/alz-perturb-gan',version='0.9.7',frozen_updates=20000,models=5,real_training_donors=33,training_cells=14762,genes_measured=2000,synthetic_cells=12800000,stages=profiles['stage_order'],subtypes=profiles['group_names'],covariates=covariates,hyperparameters=q['plan']['hyperparameters'],calibration=quality['calibration_per_seed'],calibration_means=quality['calibration_mean_over5seeds'],gene_agreement=quality['gene_contrast_descriptive_agreement'],classifier=dict(apparent_training_accuracy=quality['real_training_classifier_accuracy'],apparent_balanced_training_accuracy=quality['real_training_classifier_balanced_accuracy'],generated_mean_confidence_by_stage=quality['generated_classifier_confidence_mean_by_stage']),mixture=dict(real=quality['real_equal_donor_stage_mixture'],generated_by_seed=quality['generated_seed_mean_mixtures'],total_variation_mean=quality['mixture_total_variation_mean_by_stage']),screen=dict(measured=2000,stable=sum(x['stable'] for x in rows),cohort_supported=sum(x['rosmap_support'] for x in rows),sensitivity_robust=sum(x['robust_sensitivity'] for x in rows)),rows=rows,stage_profiles=profiles['profiles'],stage_profile_eligible_donors=profiles['real_eligible_donors_by_stage_group'],form_answer='I’d like to test whether reducing CLEC5A in human stem-cell-derived microglia improves amyloid-β clearance without increasing synapse loss. In an independent proof of concept using public SEA-AD data, our trained generative model and donor-level analyses prioritize CLEC5A as a pathology-associated candidate. Published mouse knockout results motivate testing whether the benefit translates to human cells, including across APOE3 and APOE4 backgrounds.')
 assert data['screen']==dict(measured=2000,stable=614,cohort_supported=107,sensitivity_robust=58)
 for row in rows:
  profile=profiles['profiles'][row['gene']];delta=np.mean([v[3]-v[0] for v in profile['model_by_seed']]);real=profile['real'][3]-profile['real'][0]
  assert abs(delta-row['model_effect'])<3e-8 and abs(real-row['real_effect'])<3e-8
 (EVIDENCE/'report-data.json').write_text(json.dumps(data,separators=(',',':'),ensure_ascii=False,allow_nan=False)+'\n')
 with (EVIDENCE/'genes.csv').open('w',newline='') as f:
  fields=[k for k in rows[0] if k!='seed_effects']+[f'model_seed_{s}' for s in range(5)];writer=csv.DictWriter(f,fieldnames=fields,lineterminator='\n');writer.writeheader()
  for row in rows:
   writer.writerow({k:v for k,v in row.items() if k!='seed_effects'}|{f'model_seed_{s}':row['seed_effects'][s] for s in range(5)})
 return data

def figures(data):
 style();rows=data['rows'];by={x['gene']:x for x in rows}
 fig,ax=plt.subplots(figsize=(5.8,3.8))
 p=data['stage_profiles']['CLEC5A'];a=np.array(p['model_by_seed'])
 for line in a:ax.plot(range(4),line,color=TEAL,alpha=.23,lw=1)
 ax.plot(range(4),a.mean(0),color=TEAL,lw=2.3,marker='o',markersize=4,label='Model mean')
 ax.plot(range(4),p['real'],color=INK,lw=2,marker='D',markersize=4,label='Observed donors')
 ax.set_ylim(bottom=0);ax.set_xticks(range(4),['Not AD','Low','Intermediate','High']);ax.set_ylabel('Mean log1p(panel CP10k)');ax.grid(axis='y',color=MUTED,alpha=.15);ax.legend(frameon=False,fontsize=9,loc='lower right');fig.tight_layout();save(fig,'clec5a-profile')
 fig,axes=plt.subplots(1,2,figsize=(10.4,3.4))
 for ax,(metric,title) in zip(axes,[('energy_score','Donor energy score'),('cell_energy_distance','Cell energy distance')]):
  for r in data['calibration']:
   y=[r['resampling'][metric],r['PG'][metric]];ax.plot([0,1],y,color=MUTED,lw=1,zorder=1);ax.scatter([0],[y[0]],color=INK,s=28,zorder=3);ax.scatter([1],[y[1]],color=TEAL,s=35,zorder=3)
  ax.set_xticks([0,1],['Matched resampling','Generative model']);ax.set_xlim(-.35,1.35);ax.set_title(title+' ↓',loc='left',pad=14);ax.set_ylabel('Score (lower is better)');ax.grid(axis='y',alpha=.14)
 fig.text(.5,-.015,'Each connected pair is one training seed. Four calibration donors; two generation replicates per seed. No confidence intervals.',ha='center',fontsize=9,color=MUTED);fig.tight_layout(w_pad=3);save(fig,'calibration')
 fig,ax=plt.subplots(figsize=(7.5,5.2));x=np.array([r['real_effect'] for r in rows]);y=np.array([r['model_effect'] for r in rows]);support=np.array([r['rosmap_support'] for r in rows]);ax.scatter(x[~support],y[~support],s=9,c=MUTED,alpha=.37,edgecolors='none',rasterized=True);ax.scatter(x[support],y[support],s=15,c=TEAL,alpha=.75,edgecolors='none');bound=max(abs(x).max(),abs(y).max())*1.04;ax.plot([-bound,bound],[-bound,bound],ls='--',c=MUTED,lw=.8);ax.axhline(0,color=PALE,lw=.7);ax.axvline(0,color=PALE,lw=.7)
 for gene,offset in {'CLEC5A':(12,21),'CD163':(27,-19),'CDKN1A':(-69,-22),'USP9Y':(-25,15),'GPNMB':(10,-28)}.items():
  r=by[gene];ax.scatter([r['real_effect']],[r['model_effect']],s=35,c=RUST if gene in ['CD163','USP9Y'] else TEAL,zorder=6);ax.annotate(gene,(r['real_effect'],r['model_effect']),xytext=offset,textcoords='offset points',fontsize=9,arrowprops=dict(arrowstyle='-',color=MUTED,lw=.6))
 ax.set_xlim(-bound,bound);ax.set_ylim(-bound,bound);ax.set_aspect('equal');ax.set_xlabel('Observed donor contrast');ax.set_ylabel('Mean model contrast across five seeds');ax.set_title('Which stage-associated gene changes are reproduced?',loc='left',pad=15);fig.text(.5,-.005,'High − Not AD, mean per-cell log1p(panel CP10k). Amber: model + primary ROSMAP support. Dashed: exact agreement.',ha='center',fontsize=8.6,color=MUTED);fig.tight_layout();save(fig,'gene-agreement')
 selected=['CLEC5A','PPARG','GPNMB','PLIN2','LPL','CD163','USP9Y'];fig,ax=plt.subplots(figsize=(10,4.9));positions=np.arange(len(selected))[::-1]
 for pos,gene in zip(positions,selected):
  r=by[gene];effects=r['seed_effects'];ax.plot([min(effects),max(effects)],[pos,pos],color=TEAL,lw=2,alpha=.5);ax.scatter(effects,[pos]*5,s=24,c=TEAL,alpha=.8);ax.scatter([r['real_effect']],[pos+.16],marker='D',s=35,c=INK,label='Observed, unadjusted' if gene==selected[0] else None);ax.scatter([r['adjusted_effect']],[pos-.16],marker='s',s=32,c=RUST,label='Observed, covariate-adjusted' if gene==selected[0] else None)
 ax.scatter([],[],s=24,c=TEAL,label='Five model seed means');ax.set_yticks(positions,selected);ax.axvline(0,c=MUTED,lw=.8);ax.set_xlabel('High − Not AD contrast / adjusted coefficient on the same expression scale');ax.set_title('A reproducible association can still depend on donor covariates',loc='left',pad=14);ax.legend(loc='upper center',bbox_to_anchor=(.5,-.19),frameon=False,ncol=3,fontsize=8.5);ax.grid(axis='x',alpha=.12);fig.tight_layout();save(fig,'candidate-effects')
 fig,axes=plt.subplots(2,3,figsize=(11,6.1),sharex=True)
 for ax,gene in zip(axes.ravel(),['CLEC5A','PLIN2','GPNMB','LPL','PPARG','CD163']):
  p=data['stage_profiles'][gene];a=np.array(p['model_by_seed']);
  for line in a:ax.plot(range(4),line,color=TEAL,alpha=.2,lw=.9)
  ax.plot(range(4),a.mean(0),color=TEAL,lw=2,marker='o',markersize=3,label='Mean model');ax.plot(range(4),p['real'],color=INK,lw=1.7,marker='D',markersize=3,label='Observed donors');ax.set_title(gene,loc='left',fontsize=11);ax.set_ylim(bottom=0);ax.set_xticks(range(4),['Not AD','Low','Interm.','High']);ax.grid(axis='y',alpha=.12)
 for ax in axes[:,0]:ax.set_ylabel('Mean log1p(panel CP10k)')
 axes[0,0].legend(frameon=False,fontsize=8);fig.text(.5,.005,'Faint lines: five model initializations. Different donors at each stage; these are cross-sectional profiles, not treatment responses.',ha='center',fontsize=9,color=MUTED);fig.tight_layout(rect=[0,.025,1,1],h_pad=2,w_pad=2);save(fig,'stage-profiles')

def manifest():
 sources={f'analysis/{name}':sha(ANALYSIS/name) for name in FILES.values()};sources['docs/evidence/stage-profiles.json']=sha(EVIDENCE/'stage-profiles.json')
 sources['scripts/report/report.template.html']=sha(ROOT/'scripts/report/report.template.html')
 sources['docs/brand/manifest.json']=sha(DOCS/'brand/manifest.json')
 paths=[EVIDENCE/'report-data.json',EVIDENCE/'genes.csv',DOCS/'index.html',DOCS/'report.css',DOCS/'report.js',*ASSETS.glob('*.svg'),*ASSETS.glob('*.png'),*sorted((DOCS/'brand').glob('*'))]
 output=dict(schema=1,study_snapshot='2026-11-08',source_research_commit='89fa5ea',aggregate_input_SHA256=sources,generated_output_SHA256={str(p.relative_to(ROOT)):sha(p) for p in sorted(paths)},builder_SHA256=sha(Path(__file__)),description='Public aggregate exports and figures; no raw cells or model weights. All gene contrasts are exploratory observational associations.')
 (EVIDENCE/'manifest.json').write_text(json.dumps(output,indent=2)+'\n')

def html_export(data):
 template=(ROOT/'scripts/report/report.template.html').read_text();embed=json.dumps(data,separators=(',',':'),ensure_ascii=False,allow_nan=False).replace('<','\\u003c');out=template.replace('__REPORT_JSON__',embed)
 assert '__REPORT_JSON__' not in out;(DOCS/'index.html').write_text(out)

def check():
 data=read(EVIDENCE/'report-data.json');m=read(EVIDENCE/'manifest.json');assert len(data['rows'])==len(set(x['gene'] for x in data['rows']))==2000
 assert sha(Path(__file__))==m['builder_SHA256'],'builder changed since export'
 for name,pin in (m['aggregate_input_SHA256']|m['generated_output_SHA256']).items():assert sha(ROOT/name)==pin,name
 for p in [EVIDENCE/'report-data.json',EVIDENCE/'genes.csv',EVIDENCE/'stage-profiles.json',DOCS/'index.html']:
  text=p.read_text()
  assert not any(v in text for v in ['/ml2/','/home/mikkel/','lease_token','H20.33.','H21.33.']),str(p)
 from html.parser import HTMLParser
 class Links(HTMLParser):
  def __init__(self):super().__init__();self.ids=set();self.links=[]
  def handle_starttag(self,tag,attrs):
   a=dict(attrs)
   if 'id' in a:assert a['id'] not in self.ids,a['id'];self.ids.add(a['id'])
   if tag in ['a','img','script','link']:
    for key in ['href','src']:
     if key in a:self.links.append(a[key])
 parser=Links();parser.feed((DOCS/'index.html').read_text())
 for link in parser.links:
  if link.startswith('#'):assert link[1:] in parser.ids,link
  elif not link.startswith(('https:','http:','mailto:','data:')):assert (DOCS/link).is_file(),link
 print(json.dumps(dict(status='REPORT_DATA_HASHES_GENE_COUNTS_STAGE_CONTRASTS_LINKS_AND_PUBLIC_EXPORT_CHECKS_PASS',genes=2000,screen=data['screen'],no_raw_cell_or_donor_identifier_exports=True)))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--check',action='store_true');args=p.parse_args()
 if args.check:check()
 else:
  EVIDENCE.mkdir(exist_ok=True,parents=True);ASSETS.mkdir(exist_ok=True);data=data_export();figures(data);html_export(data);manifest();check()
