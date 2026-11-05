"""Run the actual isolated runtime/identity fixtures, without GPU admission."""
import datetime,json,os,re,subprocess,sys
from pathlib import Path
HERE=Path(__file__).resolve().parent;sys.path.insert(0,str(HERE))
import binding as b,preset_scope as scope
assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
def main():
 rows=[];count=0
 for name in ('test_integration.py','test_scope.py'):
  p=HERE/name;command=[sys.executable,'-I','-S','-B','-c',"import sys,runpy;sys.path.insert(0,sys.argv[1]);p=sys.argv[2];sys.argv=[p];runpy.run_path(p,run_name='__main__')",str(HERE),str(p)]
  q=subprocess.run(command,capture_output=True,text=True,timeout=90);print(q.stderr);assert q.returncode==0,name
  n=int(re.search(r'Ran (\d+) tests',q.stderr).group(1));count+=n;rows.append({'test_file':str(p),'returncode':q.returncode,'tests':n,'stdout':q.stdout,'stderr':q.stderr})
 paths=[*HERE.glob('*.py'),*scope.MODEL_DIR.glob('*.py'),scope.PROTOCOL,scope.MODEL_PROOF,scope.PARITY_PROOF,scope.ENGINE_PROOF,scope.WINDOW]
 assert 'torch' not in sys.modules
 result={'status':'MATCHED_SCVI_ISOLATED_RUNTIME_CPU_QUALIFIED','utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'tests_passed':count,'GPU_allocation_created':False,'model_imported':False,'test_processes':rows,'source_SHA256':{str(p):b.sha(p) for p in paths},'scope':'Actual confined synthetic owner/worker/budget/lease/cleanup tests and matched comparator endpoint/source/version/held-out-exclusion/refusal checks. Native real-data GPU performance and quality remain unmeasured.'}
 scope.RUNTIME_PROOF.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({'status':result['status'],'tests_passed':count}))
if __name__=='__main__':main()
