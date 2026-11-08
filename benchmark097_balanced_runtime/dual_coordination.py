"""Ninety allowed cell/device bindings; each authorization fixes one card."""
import contextlib,fcntl,hashlib,json,os
from pathlib import Path
ROOT=Path('/ml2/alz');GPUS=('GPU-72c1b506-891d-b8bc-b353-e020585e1c47','GPU-cb4ce47d-d968-bffd-5646-e830a9fa1c69')
RESOURCE_PROTOCOL=ROOT/'analysis/benchmark097_balanced/resource_protocol.json'
def valid_cell(cell):
 assert cell['experiment']=='external097' and type(cell['fold']) is type(cell['seed']) is int and cell['fold'] in range(3) and cell['seed'] in range(5) and cell['method'] in ('pg','scvi','cfgen')
 return cell
def eligible_gpus(cell):valid_cell(cell);return GPUS
def validate_resource_protocol():
 p=json.loads(RESOURCE_PROTOCOL.read_text())
 assert p['status']=='PREPARED_BALANCED_RESOURCE_CHOICE_FOR_SAME_FROZEN45'
 assert p['registered_scientific_cells']==45 and p['registered_cell_device_bindings']==90 and tuple(p['eligible_GPUs'])==GPUS
 assert p['at_most_one_ALZ_cell_per_physical_card'] and p['peer_ALZ_host_reserve_bytes']==6*2**30 and p['all45_fresh_models_required'] and p['scientific_protocol_unchanged'] and p['original57_and_prespecified53_preserved']
 assert hashlib.sha256(Path(p['scientific_protocol_path']).read_bytes()).hexdigest()==p['scientific_protocol_SHA256']
 return p
def identity(pid):
 try:
  p=Path('/proc',str(pid));v=(p/'stat').read_text().rsplit(')',1)[1].split()
  if v[0]=='Z' or p.stat().st_uid!=os.getuid():return None
  return dict(pid=pid,birth=int(v[19]),boot=Path('/proc/sys/kernel/random/boot_id').read_text().strip())
 except (FileNotFoundError,ProcessLookupError,PermissionError):return None
def validate_active_peers(state,auth):
 if any(state.get(k) for k in ('holds','project_holds','execution_holds')):raise ValueError('Project hold remains active')
 rows=state.get('open_reservations',[])
 if len(rows)>1:raise ValueError('Both ALZ lanes already occupied')
 bindings=state.get('execution_controller',{}).get('calibration_parallel_backends',{})
 for r in rows:
  if r['status']!='reserved' or identity(r['owner']['pid'])!=r['owner']:raise ValueError('Prior owned allocation needs reconciliation')
  if r['allocation_id']==auth['allocation_id'] or r['gpu_uuids']==[auth['gpu_uuid']]:raise ValueError('Cell or assigned ALZ lane already occupied')
  meta=bindings.get(r['allocation_id'],{})
  if not meta or meta.get('gpu_uuid') not in GPUS or r['gpu_uuids']!=[meta['gpu_uuid']]:raise ValueError('Unscoped prior allocation blocks admission')
  valid_cell(meta['cell'])
  if meta.get('benchmark_protocol_SHA256')!=auth['benchmark_protocol_SHA256'] or meta.get('resource_protocol_SHA256')!=hashlib.sha256(RESOURCE_PROTOCOL.read_bytes()).hexdigest():raise ValueError('Peer scientific/resource protocol differs')
  if meta.get('path')!=str(Path(__file__).parent/'project_runner_gpu0.py'):raise ValueError('Peer backend differs')
  if r['device_count']!=1 or r['declared_wall_seconds']!=r['job_cap_seconds'] or r['job_cap_seconds']!=auth['cap_seconds']:raise ValueError('Peer deadline differs')
  if not meta.get('ledger_path') or Path(meta['ledger_path'])!=Path(auth['project_root'])/r['ledger_path']:raise ValueError('Peer ledger differs')
  if any(meta.get('source_SHA256',{}).get(k)!=v for k,v in auth['files_SHA256'].items() if k.endswith('.py') and Path(k).parent==Path(__file__).parent):raise ValueError('Peer runtime source differs')
 return True
@contextlib.contextmanager
def execution_locks(science,auth):
 science=Path(science);gpu=auth['gpu_uuid'];assert gpu in GPUS
 key=auth['allocation_id'];assert key and all(c.isalnum() or c in '_-' for c in key)
 with contextlib.ExitStack() as stack:
  for path,kind in [(science/'core_phase.lock',fcntl.LOCK_SH),(science/f'core_phase.dual-{GPUS.index(gpu)}.lock',fcntl.LOCK_EX),(science/f'core_phase.dual-{key}.lock',fcntl.LOCK_EX)]:
   f=stack.enter_context(path.open('a+'));fcntl.flock(f,kind|fcntl.LOCK_NB)
  yield
