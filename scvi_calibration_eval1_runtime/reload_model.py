"""Reload the exact final native model using scVI's public auto-device API."""
import json
from pathlib import Path
def load_native(cache,fit_dir,device):
 import scvi,torch
 from scvi_adapter import load_training
 from contracts import sha
 from state_fingerprint import fingerprint
 fit_dir=Path(fit_dir);receipt=json.loads((fit_dir/'fit_receipt.json').read_bytes())
 assert receipt['status']=='completed' and receipt['completed_epochs']==receipt['fixed_endpoint']==400
 assert receipt['scvi_tools']==scvi.__version__=='1.3.3'
 assert receipt['cache_manifest_SHA256']==sha(Path(cache)/'manifest.json')
 for name,pin in receipt['model_files_SHA256'].items():assert sha(fit_dir/name)==pin
 saved=torch.load(fit_dir/'model/model.pt',map_location='cpu',weights_only=False)
 state=saved['model_state_dict']
 # SCVI.save appends a separate Pyro store to the saved dictionary. Its native
 # loader pops that sidecar before module.load_state_dict; this VAE uses none.
 store=state.pop('pyro_param_store')
 assert set(store)=={'params','constraints'} and not store['params'] and not store['constraints']
 expected=fingerprint(state)
 a,_=load_training(cache)
 model=scvi.model.SCVI.load(fit_dir/'model',adata=a,accelerator='cpu' if device=='cpu' else 'gpu',device='auto')
 model.to_device(device);model.module.eval()
 assert fingerprint(model.module.state_dict())==expected
 assert all(p.device.type==('cpu' if device=='cpu' else 'cuda') for p in model.module.parameters())
 return model,receipt,expected
