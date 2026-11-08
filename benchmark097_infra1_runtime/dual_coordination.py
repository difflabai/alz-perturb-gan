"""One fresh replacement after the original lanes close; physical locks retained."""
import contextlib,fcntl,hashlib,json,os
from pathlib import Path
ROOT=Path('/ml2/alz');GPUS=('GPU-72c1b506-891d-b8bc-b353-e020585e1c47','GPU-cb4ce47d-d968-bffd-5646-e830a9fa1c69')
RESOURCE_PROTOCOL=ROOT/'analysis/benchmark097_infra1/protocol.json'
def valid_cell(cell):
 assert cell==dict(experiment='external097infra1',fold=2,method='cfgen',seed=1)
 assert type(cell['fold']) is type(cell['seed']) is int
 return cell
def eligible_gpus(cell):valid_cell(cell);return GPUS
def validate_resource_protocol():
 p=json.loads(RESOURCE_PROTOCOL.read_text());assert p['new_attempt_limit']==1 and p['all_original_attempts_close_before_activation'] and p['cap_seconds']==21600
 assert hashlib.sha256(Path(p['original_scientific_protocol_path']).read_bytes()).hexdigest()==p['original_scientific_protocol_SHA256']
 return p
def identity(pid):
 try:
  p=Path('/proc',str(pid));v=(p/'stat').read_text().rsplit(')',1)[1].split()
  if v[0]=='Z' or p.stat().st_uid!=os.getuid():return None
  return dict(pid=pid,birth=int(v[19]),boot=Path('/proc/sys/kernel/random/boot_id').read_text().strip())
 except (FileNotFoundError,ProcessLookupError,PermissionError):return None
def validate_active_peers(state,auth):
 if any(state.get(k) for k in ('holds','project_holds','execution_holds')):raise ValueError('Project hold remains active')
 if state.get('open_reservations'):raise ValueError('Original or replacement ALZ allocation still active; no duplicate admission')
 return True
@contextlib.contextmanager
def execution_locks(science,auth):
 science=Path(science);gpu=auth['gpu_uuid'];assert gpu in GPUS
 key=auth['allocation_id'];assert key and all(c.isalnum() or c in '_-' for c in key)
 with contextlib.ExitStack() as stack:
  for path,kind in [(science/'core_phase.lock',fcntl.LOCK_SH),(science/f'core_phase.dual-{GPUS.index(gpu)}.lock',fcntl.LOCK_EX),(science/f'core_phase.dual-{key}.lock',fcntl.LOCK_EX)]:
   f=stack.enter_context(path.open('a+'));fcntl.flock(f,kind|fcntl.LOCK_NB)
  yield
