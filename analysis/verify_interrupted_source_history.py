"""Prove the interrupted allocation used its committed exact historical sources."""
import datetime,hashlib,json,subprocess
from pathlib import Path
ROOT=Path('/ml2/alz');a='CAL_GPU0_V1_pure_confirm_0da2a09092c823377ca5_f0_pg_s1'
dep=json.loads((ROOT/'artifacts/allocations'/a/'deployment.json').read_text());path=Path(dep['authorization_path']);sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest();auth=json.loads(path.read_text());assert sha(path)==dep['authorization_SHA256']
commit=dep['candidate_commit'];assert subprocess.check_output(['git','rev-parse',commit],cwd=ROOT,text=True).strip()==commit
checked={}
for name,pin in auth['research_source_SHA256'].items():
 p=Path(name);relative=p.relative_to(ROOT)
 raw=subprocess.check_output(['git','show',commit+':'+str(relative)],cwd=ROOT)
 assert hashlib.sha256(raw).hexdigest()==pin,'Historical source differs: '+name
 checked[name]=pin
freeze=ROOT/'analysis/pure_ensemble_confirmation_freeze.json';f=json.loads(freeze.read_text())
assert all(checked[p]==pin and sha(p)==pin for p,pin in f['scientific_source_SHA256'].items())
result={'status':'VERIFIED_INTERRUPTED_ALLOCATION_COMMITTED_SOURCE_HISTORY','utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'attempt':a,'candidate_commit':commit,'authorization_path':str(path),'authorization_SHA256':sha(path),'checked_source_files':len(checked),'source_dictionary':checked,'frozen_scientific_sources_unchanged':True,'original_freeze_SHA256':sha(freeze),'GPU_allocation_created':False,'scope':'Read-only committed Git blob hashes versus original immutable authorization. No current-source substitution for the interrupted allocation, no model/optimizer changes or native admission.','verification_source_SHA256':{str(Path(__file__).resolve()):sha(Path(__file__).resolve())}}
dest=ROOT/'analysis/interrupted_seed1_source_history.json';assert not dest.exists();dest.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k!='source_dictionary'},indent=2))
