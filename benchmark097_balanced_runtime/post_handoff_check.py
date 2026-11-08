"""All90 real source closures after handoff/activation; no reservation/import."""
import datetime,json,os,sys
from pathlib import Path
HERE=Path(__file__).resolve().parent;sys.path.insert(0,str(HERE))
import binding as b,preset_scope as s,prepare
from dual_coordination import GPUS

def main():
 assert os.environ.get('CUDA_VISIBLE_DEVICES')=='' and s.read(s.PLAN)==s.ENABLED
 assert s.read(s.HANDOFF)['status']=='ORIGINAL_ADMITTED_CELLS_CLOSED_AND_RESOURCE_HANDOFF_READY'
 count=0
 for seed in range(5):
  for fold in range(3):
   for method in ('pg','scvi','cfgen'):
    for gpu in GPUS:
     auth,_=prepare.build_authorization(fold=fold,method=method,seed=seed,gpu=gpu,check_only=True)
     s.validate_authorization(auth);b.source_closure(auth,b.NATIVE_LAYOUT);count+=1
 assert count==90 and 'torch' not in sys.modules
 out=s.AUX/'post_handoff_source_closure.json';assert not out.exists();out.write_text(json.dumps(dict(status='ACTUAL_POST_HANDOFF_ENABLED_PLAN_ALL90_SOURCE_CLOSURES_PASS',UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),native_cell_device_bindings=count,runtime_qualification_SHA256=b.sha(s.RUNTIME_PROOF),handoff_SHA256=b.sha(s.HANDOFF),GPU_allocations=0,model_imported=False,scope='Eligibility proof only; consumed cells remain excluded by canonical unseen and the immutable handoff prefix.'),indent=2)+'\n')
 print(json.dumps(dict(status='ACTUAL_POST_HANDOFF_ENABLED_PLAN_ALL90_SOURCE_CLOSURES_PASS',GPU_allocations=0)))
if __name__=='__main__':main()
