"""Standard-library-only retained pure PG scope; no model imports or fitting."""
import hashlib
import json
from pathlib import Path
ROOT=Path('/ml2/alz')
def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
 return h.hexdigest()
def read(path):return json.loads(Path(path).read_text())
def load_entry(label):
 m=read(ROOT/'pure_formulations.json')
 assert m['generation']=='direct_iter_pg' and m['real_bag_replacements']==0 and m['calibration_only'] and m['replicates']==[0,1]
 return m['entries'][label]
def validate_authorization(auth):
 manifest=ROOT/'pure_formulations.json';pins=auth['files_SHA256'];entry=load_entry(auth['pure_formulation'])
 assert auth['cap_seconds']==900 and auth['qualification_only'] is True and auth.get('hybrid_config') is None
 assert pins.get(str(manifest))==auth['pure_manifest_SHA256']==sha(manifest)
 for key in ['checkpoint_path','checkpoint_SHA256','variant_config']:assert auth[key]==entry[key],key
 assert auth['fixed_endpoint']==entry['completed_updates'] in (20000,40000)
 assert auth['cell']['seed']==entry['training_seed']==0
 checkpoint=Path(entry['checkpoint_path']);receipt=checkpoint.parent/'checkpoint_receipt.json';q=read(receipt)
 assert pins.get(str(receipt))==sha(receipt) and q['status']=='completed'
 assert q['completed_updates']==entry['completed_updates'] and q['path']==checkpoint.name
 assert q['SHA256']==entry['checkpoint_SHA256']==pins.get(str(checkpoint))==sha(checkpoint)
