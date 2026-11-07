"""Confined controller fixtures and real premature-admission refusal only."""
import datetime,json,os,re,subprocess,sys
from pathlib import Path
HERE=Path(__file__).resolve().parent;sys.path.insert(0,str(HERE))
import binding as b,preset_scope as s
def main():
 assert os.environ.get('CUDA_VISIBLE_DEVICES')=='' and not s.read(s.PLAN)['enabled'];tests=[];count=0
 for name in ('test_integration.py','test_prepared.py'):
  args=[sys.executable,'-I','-S','-B','-c','import sys,runpy;sys.path.insert(0,sys.argv[1]);p=sys.argv[2];sys.argv=[p];runpy.run_path(p,run_name="__main__")',str(HERE),str(HERE/name)]
  p=subprocess.run(args,capture_output=True,text=True,timeout=180);print(p.stderr);tests.append(dict(file=name,returncode=p.returncode,stdout=p.stdout,stderr=p.stderr));assert p.returncode==0,name
  count+=int(re.search(r'Ran (\d+) tests?',p.stderr).group(1))
 assert count==16 and 'torch' not in sys.modules
 paths=[*HERE.glob('*.py'),HERE/'policy.prepared.json',s.PROTOCOL,s.RULE,s.SELECTOR]
 q=dict(status='PREPARED_FOLLOWUP_ACCOUNTING_AND_REAL_PREMATURE_ADMISSION_REFUSAL_CPU_PASS',UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),actual_tests=count,tests=tests,
   actual_selected_configuration_scope_qualified=False,actual_paired_study_incomplete_admission_refused=True,live_runtime_deployment_qualified=False,GPU_allocation_created=False,CUDA_initialized=False,source_SHA256={str(p):b.sha(p) for p in paths})
 path=s.STUDY/'prepared_accounting_CPU_qualification.json';assert not path.exists();path.write_text(json.dumps(q,indent=2)+'\n');print(json.dumps({k:q[k] for k in ('status','actual_tests','actual_selected_configuration_scope_qualified','GPU_allocation_created')}))
if __name__=='__main__':main()
