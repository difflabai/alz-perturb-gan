"""All12 real scalar cell/card source closures at the completed science boundary."""
import datetime,json,os,sys
from pathlib import Path
HERE=Path(__file__).resolve().parent;sys.path.insert(0,str(HERE))
import binding as b,preset_scope as s,prepare
from dual_coordination import GPUS

def main():
 assert os.environ.get('CUDA_VISIBLE_DEVICES')=='' and s.read(s.PLAN)==s.ENABLED;s.prior_study_ready();count=0
 for seed in (5,6,7):
  for arm in ('fixed','adaptive'):
   for gpu in GPUS:
    auth,_=prepare.build_authorization(arm=arm,seed=seed,gpu=gpu,check_only=True);s.validate_authorization(auth);b.source_closure(auth,b.NATIVE_LAYOUT);count+=1
 assert count==12 and 'torch' not in sys.modules
 out=s.STUDY/'post_handoff_source_closure_v3.json';assert not out.exists()
 out.write_text(json.dumps(dict(status='SCALARCAL097_ENABLED_PLAN_ALL12_SOURCE_CLOSURES_PASS',UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),native_cell_device_bindings=count,runtime_qualification_SHA256=b.sha(s.RUNTIME_PROOF),prior_scientific_completion_SHA256=s.prior_study_ready(),new_GPU_allocations=0,model_imported=False),indent=2)+'\n')
 print(json.dumps(dict(status='SCALARCAL097_ENABLED_PLAN_ALL12_SOURCE_CLOSURES_PASS',new_GPU_allocations=0)))
if __name__=='__main__':main()
