"""Fixed donor-fold contract. No model, GPU or controller side effects."""
from pathlib import Path
import hashlib,json,math
B=Path('/ml2/codex-storage/future-workspaces/2026-10-27/task-2/benchmark')
HERE=Path(__file__).resolve().parent
FOLDS=HERE.parent/'matched-benchmark-20261103/folds.json'
FOLDS_SHA256='763d081e387c2d00c6412f07c03af43babca06430a49d5e3f3b5d081c327ade3'
STAGES=['Not AD','Low','Intermediate','High']
TYPES=['Micro-PVM_1','Micro-PVM_2','Micro-PVM_2_1-SEAAD','Micro-PVM_2_3-SEAAD','Micro-PVM_3-SEAAD','Micro-PVM_4-SEAAD']
SEEDS=list(range(5));METHODS=['pg','scvi'];PG_UPDATES=20000;SCVI_EPOCHS=400;JOB_CAP_SECONDS=21600;CORE_CAP_GPU_HOURS=180
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def dump(p,obj):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_name('.'+p.name+'.tmp');tmp.write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n');tmp.replace(p)
def load_folds(path=FOLDS):
 if sha(path)!=FOLDS_SHA256:raise ValueError('frozen fold provenance changed')
 q=json.loads(Path(path).read_text());seen=[];universe=set(q['stage_by_fitting_donor'])
 if len(universe)!=57:raise ValueError('exact original57 required')
 for f in q['folds']:
  a,b,c=[f[k] for k in ['optimization_donors','internal_calibration_donors','evaluation_donors']]
  if any(len(x)!=len(set(x)) for x in [a,b,c]) or set(a)&set(b) or set(a)&set(c) or set(b)&set(c) or set(a+b+c)!=universe:raise ValueError('donor split is not disjoint/exhaustive')
  seen+=c
 if len(seen)!=57 or len(set(seen))!=57:raise ValueError('evaluation donors must cover57 exactly once')
 return q
def fold(index):
 if type(index) is not int or index not in range(3):raise ValueError('fold0/1/2 required')
 return load_folds()['folds'][index]
def validate_train_donors(observed,expected):
 if set(map(str,observed))!=set(expected):raise ValueError('actual training donors differ from optimization-only manifest')
def stage_condition(stage):
 if type(stage) is not int or stage not in range(4):raise ValueError('valid ADNC stage required')
 return [0.,0.,0.,stage/3,0.,0.,0.]
def generation_seed(fold,training_seed,replicate,stage):
 if fold not in range(3) or training_seed not in SEEDS or replicate not in [0,1] or stage not in range(4):raise ValueError('unregistered sampling cell')
 return 20261104+100000*fold+1000*training_seed+100*replicate+stage
def task_matrix():
 return [{'allocation_id':f'ALZ_CORE_V1_f{f}_{m}_s{s}','fold':f,'method':m,'seed':s,'fixed_endpoint':PG_UPDATES if m=='pg' else SCVI_EPOCHS,'job_cap_seconds':JOB_CAP_SECONDS,'status':'HELD_FOR_SOURCE_REVIEW_AND_SCHEDULER'} for f in range(3) for m in METHODS for s in SEEDS]
def checked_cache(cache,role):
 cache=Path(cache);manifest=json.loads((cache/'manifest.json').read_text());f=fold(manifest['fold'])
 if manifest['folds_SHA256']!=FOLDS_SHA256 or manifest['status']!='CPU_FOLD_PREPARED_NO_GENERATIVE_MODEL_FIT':raise ValueError('unqualified fold cache')
 if role not in ['train','evaluation']:raise ValueError('train/evaluation role required')
 expected=f['optimization_donors'] if role=='train' else f['evaluation_donors']
 if manifest[role+'_donors']!=expected:raise ValueError('fold donor manifest differs')
 for file,pin in manifest['files_SHA256'].items():
  if sha(cache/file)!=pin:raise ValueError('fold cache input changed: '+file)
 return manifest,f
