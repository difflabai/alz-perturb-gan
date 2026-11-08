"""Finite CPU-only completion of the frozen inference review; never trains models."""
import datetime,hashlib,json,os,subprocess,time
from pathlib import Path
ROOT=Path('/ml2/alz');INFERENCE=ROOT/'artifacts/primamente-t005-gene-inference-recovery-v1-20261108';STATE=ROOT/'artifacts/primamente-association-review-20261108'
PYTHON='/ml2/codex-storage/future-workspaces/2026-10-27/task-2/benchmark/current_base_20261029/runtime/bin/python'
SOURCES=['aggregate_primamente_gene_inference.py','compare_primamente_genes_with_rosmap.py','review_primamente_model_quality_20261108.py']
def read(p):return json.loads(Path(p).read_bytes())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,q):
 t=p.with_suffix('.partial');t.write_text(json.dumps(q,indent=2)+'\n');t.replace(p)
def alive(i):
 p=Path('/proc')/str(i['pid'])/'stat'
 if not p.exists():return False
 q=p.read_text().rsplit(')',1)[1].split()
 return int(q[19])==i['birth'] and q[0]!='Z' and Path('/proc/sys/kernel/random/boot_id').read_text().strip()==i['boot']
def main():
 assert os.environ.get('CUDA_VISIBLE_DEVICES')=='';STATE.mkdir();worker=read(INFERENCE/'identity.json');p=os.getpid()
 pins={str(ROOT/'analysis'/s):sha(ROOT/'analysis'/s) for s in SOURCES}
 write(STATE/'identity.json',dict(pid=p,birth=int(Path('/proc',str(p),'stat').read_text().rsplit(')',1)[1].split()[19]),boot=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),source_SHA256=sha(__file__),inference_owner=worker,analysis_sources_SHA256=pins,new_training_fits=0))
 deadline=time.monotonic()+7200
 try:
  while not (INFERENCE/'completed.json').exists():
   assert not (INFERENCE/'failure.json').exists(),'Inference failed; retain all partials without ranking'
   assert alive(worker),'Exact inference owner exited without a completion receipt'
   assert time.monotonic()<deadline,'Finite analysis wait allowance ended'
   write(STATE/'progress.json',dict(status='WAITING_FOR_COMPLETE_FIVE_MODEL_INFERENCE',inference_owner_live=True,new_training_fits=0));time.sleep(30)
  for p,pin in pins.items():assert sha(p)==pin,p
  for source in SOURCES:
   args=[PYTHON,'-I','-S','-B',str(ROOT/'analysis'/source)]+([] if source.startswith('review_') else ['--run'])
   p=subprocess.run(args,cwd=ROOT,env=os.environ.copy(),capture_output=True,text=True,timeout=600)
   (STATE/(source+'.stdout')).write_text(p.stdout);(STATE/(source+'.stderr')).write_text(p.stderr)
   assert p.returncode==0,f'Analysis failed: {source}; retain exact error'
   write(STATE/'progress.json',dict(status='COMPLETED_'+source,new_training_fits=0))
  paths=[ROOT/'analysis'/s for s in ['primamente_gene_inference_exploratory_ranking_20261108.json','primamente_model_gene_ROSMAP_crosscheck_20261108.json','primamente_model_capability_review_20261108.json']]
  write(STATE/'completed.json',dict(status='FROZEN_INFERENCE_ASSOCIATIONS_COHORT_CROSSCHECK_AND_QUALITY_REVIEW_COMPLETE',UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),output_SHA256={str(p):sha(p) for p in paths},new_training_fits=0,therapeutic_target_selected=False,primary_literature_functional_review_required=True))
 except BaseException as exc:write(STATE/'failure.json',dict(error=repr(exc),automatic_retry=False));raise
if __name__=='__main__':main()
