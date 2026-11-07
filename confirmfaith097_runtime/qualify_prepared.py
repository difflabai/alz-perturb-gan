"""Confined accounting and actual early-refusal checks only; no deployment."""
import argparse,datetime,json,os,re,subprocess,sys
from pathlib import Path
HERE=Path(__file__).resolve().parent;sys.path.insert(0,str(HERE))
import binding as b,preset_scope as s
def main():
 parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,default=s.STUDY/'prepared_accounting_CPU_qualification.json');args=parser.parse_args();assert args.output.resolve().parent==s.STUDY.resolve() and not args.output.exists()
 assert os.environ.get('CUDA_VISIBLE_DEVICES')=='' and not s.read(s.PLAN)['enabled']
 results=[];count=0
 for name in ('test_integration.py','test_prepared.py'):
  cmd=[sys.executable,'-I','-S','-B','-c','import sys,runpy;sys.path.insert(0,sys.argv[1]);p=sys.argv[2];sys.argv=[p];runpy.run_path(p,run_name="__main__")',str(HERE),str(HERE/name)]
  p=subprocess.run(cmd,capture_output=True,text=True,timeout=180);print(p.stderr);results.append(dict(file=name,returncode=p.returncode,stdout=p.stdout,stderr=p.stderr));assert p.returncode==0,name
  count+=int(re.search(r'Ran (\d+) tests?',p.stderr).group(1))
 assert count==16 and 'torch' not in sys.modules
 files=[*HERE.glob('*.py'),HERE/'policy.prepared.json',s.PROTOCOL,s.DESIGN,s.SCALAR_PROOF,s.BASELINE_PROOF]
 q=dict(status='PREPARED_RELEASED_CONFIRMATION_ACCOUNTING_AND_REAL_EARLY_REFUSAL_CPU_PASS',UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),actual_tests=count,tests=results,
   actual_corrected_selection_not_complete_admission_refused=True,actual_frozen_confirmation_scope_qualified=False,live_runtime_deployment_qualified=False,
   GPU_allocation_created=False,CUDA_initialized=False,source_SHA256={str(f):b.sha(f) for f in files})
 output=args.output;output.write_text(json.dumps(q,indent=2)+'\n')
 print(json.dumps({k:q[k] for k in ('status','actual_tests','live_runtime_deployment_qualified','GPU_allocation_created')}))
if __name__=='__main__':main()
