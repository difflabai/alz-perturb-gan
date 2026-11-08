"""Resource-only dispatch for the same45 fresh native models.

Previously completed native fits are immutable references. Remaining scientific
cells keep their original canonical keys and start once at update0. No result
quality or target array is read by the scheduling decisions.
"""
import argparse,datetime,fcntl,importlib.util,json,os,subprocess,sys,time
from pathlib import Path
ROOT=Path('/ml2/alz');RUNTIME=ROOT/'benchmark097_infra1_runtime';STUDY=ROOT/'analysis/benchmark097_infra1';STATE=ROOT/'artifacts/benchmark097-infra1-pipeline'
sys.path.insert(0,str(RUNTIME))
import binding as b,preset_scope as scope
from dual_coordination import GPUS,RESOURCE_PROTOCOL,identity,validate_resource_protocol
from runtime_dispatch import choose_next,settled_cost_rows,cost_table
from cell_metadata import completed_metadata
ENV=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
def read(p):return json.loads(Path(p).read_bytes())
def write(p,q):
 p=Path(p);tmp=p.with_suffix('.partial');tmp.write_text(json.dumps(q,indent=2,allow_nan=False)+'\n');tmp.replace(p)
def command(args,timeout=120):
 p=subprocess.run(list(map(str,args)),cwd=ROOT,env=ENV,capture_output=True,text=True,timeout=timeout)
 if p.returncode:raise RuntimeError(p.stderr[-3000:])
 return p.stdout
def commit(paths,message):command(['git','add','--',*map(str,paths)]);command(['git','commit','--only','-m',message,'--',*map(str,paths)])
def report(kind,event,summary,details):
 from babel_reporting import report as send
 return send(kind,event,summary,details,log_root=STATE)

def matrix():return [scope.cell(*case) for case in scope.CASES]
def public_handle(h):return {k:h[k] for k in ('cell','gpu','identity','attempt') if k in h}
def run_queue(cells,*,initial_rows,start,ready,finish,observe,sleep,now,training_expiry,select):
 pending=list(cells);active={};rows=list(initial_rows);errors=[];unsettled=[]
 seen={b.allocation_key(r['cell']) for r in rows}
 assert len(seen)==len(rows) and not seen.intersection(b.allocation_key(v) for v in pending)
 while pending or active:
  if not errors and pending and now()>=training_expiry-150:errors.append(dict(error='Owner training admission window ended',pending_cells=list(pending)))
  if not errors:
   for gpu in reversed(GPUS):
    if gpu in active:continue
    value=select(pending,active,gpu)
    if value is None:continue
    assert value in pending and b.allocation_key(value) not in seen
    try:
     h=start(value,gpu,len(rows))
     if h is not None:
      assert h['gpu']==gpu and h['cell']==value
      active[gpu]=h;pending.remove(value);seen.add(b.allocation_key(value))
    except BaseException as exc:errors.append(dict(cell=value,error=repr(exc),stage='admission'));break
  for gpu,h in list(active.items()):
   try:
    if not ready(h):continue
    q=finish(h);rows.append(q);del active[gpu]
    if not q['complete']:errors.append(dict(cell=h['cell'],error='Incomplete native model cell retained',stage='endpoint'))
   except BaseException as exc:
    errors.append(dict(**public_handle(h),error=repr(exc),stage='observation_or_audit'));unsettled.append(public_handle(h));del active[gpu]
  observe(rows,pending,active,errors,unsettled)
  if errors and not active:break
  if now()>training_expiry+7200:
   errors.append(dict(error='Finite CPU observation allowance ended; exact handles retained'));unsettled.extend(public_handle(h) for h in active.values());break
  if pending or active:sleep(2 if active else 15)
 return dict(rows=rows,errors=errors,unsettled=unsettled,pending=pending)

def select_native(pending,active,gpu):
 costs,evidence=cost_table(settled_cost_rows())
 timed={g:dict(cell=h['cell'],elapsed_seconds=max(0.,time.time()-h['ledger_started_unix'])) for g,h in active.items()}
 chosen,allocation=choose_next(pending,timed,gpu,costs)
 write(STATE/'latest_cost_dispatch.json',dict(UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),requested_gpu=gpu,selected_cell=chosen,method_card_cost_seconds=costs,cost_source_attempts=evidence,allocation=allocation,quality_values_used=False,target_arrays_loaded=False))
 return chosen

def start_cell(value,gpu,completed):
 attempt=b.attempt_key(value);folder=ROOT/'artifacts/allocations'/attempt;result=STUDY/'results'/(attempt+'.json')
 assert not folder.exists() and not result.exists(),'One-use scientific key already prepared/consumed; inspect instead of retry'
 assert not command(['git','status','--porcelain']),'Commit completed research outputs before new admission'
 exe=b.runtime_path(b.NATIVE_LAYOUT,value);prepare=[exe,'-I','-S','-B',RUNTIME/'prepare.py','--fold',value['fold'],'--method',value['method'],'--seed',value['seed'],'--gpu',gpu]
 check=subprocess.run(list(map(str,[*prepare,'--check-only'])),cwd=ROOT,env=ENV,capture_output=True,text=True,timeout=120)
 if check.returncode:
  if any(text in check.stderr for text in ('Exact parallel GPU growth forecasts do not fit','Exact parallel host growth forecasts do not fit','ALZ early stop: actual GPU free below3.5GiB','ALZ early stop: host available below12GiB')):return None
  raise RuntimeError(check.stderr[-3000:])
 command(prepare);dep=read(folder/'deployment.json');auth=read(dep['authorization_path']);assert auth['gpu_uuid']==gpu and not Path(auth['ledger_path']).exists() and not Path(auth['output_dir']).exists()
 with (folder/'owner.stdout').open('xb') as out,(folder/'owner.stderr').open('xb') as err:
  owner=subprocess.Popen([str(exe),'-I','-S','-B',str(RUNTIME/'launch_core_gpu0.py'),'--authorization',dep['authorization_path']],cwd=ROOT,env=ENV,stdin=subprocess.DEVNULL,stdout=out,stderr=err,start_new_session=True)
 ident=identity(owner.pid);assert ident is not None;write(folder/'launch_receipt.json',dict(**ident,cell=value,gpu_uuid=gpu))
 return dict(cell=value,gpu=gpu,identity=ident,attempt=attempt,owner=owner,auth=auth,exe=exe,ledger_started_unix=time.time(),settlement=Path(auth['ledger_path']).with_name(attempt+'-settlement.json'))
def ready_cell(h):
 if h['owner'].poll() is not None and not Path(h['auth']['ledger_path']).exists():raise RuntimeError('Owner exited without a ledger; original attempt retained')
 if time.time()>h['auth']['expires_unix']+660:raise TimeoutError('Cell observation allowance ended; inspect exact handles')
 ledger=Path(h['auth']['ledger_path'])
 if ledger.exists():h['ledger_started_unix']=read(ledger)['jobs'][0]['started_unix']
 return h['settlement'].exists() and h['owner'].poll() is not None

def finish_cell(h):
 h['owner'].wait(timeout=30)
 (STATE/(h['attempt']+'-finish.stdout')).write_text(command([h['exe'],'-I','-S','-B',RUNTIME/'finish.py','--attempt',h['attempt']],timeout=600))
 result=STUDY/'results'/(h['attempt']+'.json')
 # The unchanged scientific auditor wrote this result only after native state,
 # count, saved-score and clean settlement checks. Cost scheduling uses metadata.
 row=completed_metadata(h['cell'],require_full=False);commit([result],f'Retain fresh balanced-resource {h["attempt"]}: full native endpoint, counts, provenance and cost')
 return row

def observe_queue(rows,pending,active,errors,unsettled):
 q=dict(status='BALANCED_RESOURCE_SAME45_RUNNING' if active else 'WAITING_FOR_SHARED_CAPACITY',completed_cells=sum(bool(r['complete']) for r in rows),resolved_cells=len(rows),required_cells=45,pending_cells=len(pending),active=[public_handle(h) for h in active.values()],errors=errors,unsettled=unsettled,scientific_protocol_unchanged=True,full_goal_complete=False,SOTA_claim=False)
 if errors:q['status']='NO_NEW_ADMISSIONS_DRAINING_OR_RETAINING_EXACT_HANDLES'
 write(STATE/'progress.json',q)
 if rows:write(STATE/'partial_results.json',dict(rows=rows,required_cells=45,full_goal_complete=False))

