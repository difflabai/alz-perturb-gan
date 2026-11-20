#!/usr/bin/env python3
"""Execute the frozen native EMPF host under the current application selection."""
import os
os.environ.update(OMP_NUM_THREADS='3', OPENBLAS_NUM_THREADS='3', MKL_NUM_THREADS='3',
                  CUBLAS_WORKSPACE_CONFIG=':4096:8', PYTHONDONTWRITEBYTECODE='1')
import argparse
import datetime
import enum
import fcntl
import hashlib
import importlib
import importlib.util
import json
from pathlib import Path
import shutil
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT/'src') not in sys.path:
    sys.path.insert(0, str(ROOT/'src'))
from alz_perturb_gan import durable_state as ds  # noqa: E402
SELECTION = ROOT / 'analysis/default_empf_20261117'
SOURCE_MANIFEST = SELECTION / 'runtime_source_manifest.json'
CACHE_ANCHORS = ROOT / 'analysis/empirical_challenge_20261114/cache_and_fit_anchors.json'
# Derives the native EMPF host with checkout-relative import paths (vendored model sources);
# analysis/empf_comparison_20261117/host.py itself stays unchanged with its concluded study.
HOST = SELECTION / 'vendored_host.py'
# Only for a host module without its own locking (vendored_host.LOCKS_ITSELF): serialize its activate().
HOST_LOCK = ROOT / 'artifacts/native-releases/particlegan-0.10.0/alz-host-empf-native-vendored.lock'
# Machine-wide CPU slot locks, shared by every checkout of this repository on the host.
CPU_LOCK_DIR = Path(os.environ.get('MICROGLIA_CPU_LOCK_DIR') or f'/tmp/microglia-cpu-slots-{os.getuid()}')
THREADS = 3
_CPU_SLOT = None  # open slot-lock handle; held for the life of the process


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def check_sources():
    manifest = json.loads(SOURCE_MANIFEST.read_text())
    for name, digest in manifest['sources'].items():
        path = Path(name) if Path(name).is_absolute() else ROOT / name
        if not path.exists() or sha(path) != digest:
            raise ValueError(f'pinned EMPF runtime source missing or changed: {name}')
    old = manifest['archived_previous_entrypoint']
    if sha(ROOT / old['archive']) != old['sha256']:
        raise ValueError('previous public entrypoint archive changed')
    return manifest


def check_unchanged_sources(started, digest):
    if sha(SOURCE_MANIFEST) != digest or check_sources() != started or sha(SOURCE_MANIFEST) != digest:
        raise ValueError('runtime source manifest changed during execution')


def check_cache(cache, fold, held=False):
    anchor = json.loads(CACHE_ANCHORS.read_text())['caches'].get(str(fold))
    if anchor is None:
        raise ValueError(f'no frozen development cache anchor for fold {fold}')
    if sha(cache/'manifest.json') != anchor['manifest_sha256']:
        raise ValueError('development cache manifest differs from the frozen fold')
    names = anchor['payload_sha256'] if held else ('space.npz', 'train/counts.npz', 'train/metadata.npz')
    payloads = {name: anchor['payload_sha256'][name] for name in names}
    for name, digest in payloads.items():
        if sha(cache/name) != digest:
            raise ValueError(f'development cache payload changed: {name}')
    return {'manifest_sha256': anchor['manifest_sha256'], 'payload_sha256': payloads}


def check_checkpoint(q, saved):
    expected = {key: saved[key] for key in ('arm', 'flags', 'panel', 'fold', 'seed',
                'cache_manifest_SHA256', 'recipe', 'particlegan_version', 'native_recipe')}
    expected.update(status='completed', completed_updates=20000, fixed_endpoint=20000,
                    released_hyperparameters=saved['hyperparameters'],
                    penalty_options={'input_domain': 'continuous'})
    for key, value in expected.items():
        if json.loads(json.dumps(q.get(key))) != value:
            raise ValueError(f'final checkpoint header differs from the selected fit: {key}')
    try:
        policy = q['policy']
        record = policy['optimizers'][1]['regularizer']['record']
        packet = record['empf']
        valid = (json.loads(json.dumps(policy['recipe'])) == saved['recipe'] and
                 policy['completed_steps'] == record['calls'] == record['observed_steps'] == 20000 and
                 packet['schema'] == 2 and packet['initialized'] is True and
                 packet['proto_initialized'] is False and packet['proto_sum'] is None and
                 packet['proto_count'] is None)
    except (KeyError, IndexError, TypeError) as exc:
        raise ValueError('final checkpoint lacks the completed native EMPF state') from exc
    if not valid:
        raise ValueError('final checkpoint does not verify the native recipe and every-update regularization')


def application():
    spec = importlib.util.spec_from_file_location('microglia_application', ROOT/'scripts/microglia.py')
    module = importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def cpu_busy(cpus, seconds=.2):
    """Per-CPU busy fraction over a short /proc/stat sample; {} when unavailable."""
    def sample():
        out = {}
        for line in Path('/proc/stat').read_text().splitlines():
            name, *values = line.split()
            if name.startswith('cpu') and name[3:].isdigit():
                values = [int(v) for v in values]
                out[int(name[3:])] = (sum(values), values[3] + (values[4] if len(values) > 4 else 0))
        return out
    try:
        a = sample(); time.sleep(seconds); b = sample()
    except (OSError, ValueError, IndexError):
        return {}
    return {c: 1 - (b[c][1]-a[c][1]) / max(b[c][0]-a[c][0], 1) for c in cpus if c in a and c in b}


def choose_cpus(allowed, threads=THREADS, env=None, lock_dir=None, load=cpu_busy):
    """Pick the CPUs for this process on a shared host.

    An inherited mask of at most `threads` CPUs is respected. MICROGLIA_CPU_SLOT=k
    selects the k-th block of `threads` allowed CPUs. Otherwise the process claims
    the least-loaded free block under a machine-wide flock; the claim is released
    when the process exits. If every block is claimed, or locking is unavailable,
    the least-loaded CPUs are used without a claim.
    """
    global _CPU_SLOT
    env = os.environ if env is None else env
    lock_dir = CPU_LOCK_DIR if lock_dir is None else Path(lock_dir)
    allowed = sorted(allowed)
    if len(allowed) <= threads:
        return allowed, {'source': 'inherited_mask'}
    blocks = [allowed[i:i+threads] for i in range(0, len(allowed)-threads+1, threads)]
    slot = env.get('MICROGLIA_CPU_SLOT')
    if slot not in (None, ''):
        k = int(slot)
        if not 0 <= k < len(blocks):
            raise ValueError(f'MICROGLIA_CPU_SLOT must be in 0..{len(blocks)-1} '
                             f'({len(blocks)} blocks of {threads} allowed CPUs), got {k}')
        return blocks[k], {'source': 'MICROGLIA_CPU_SLOT', 'slot': k}
    try:
        lock_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
        with open(lock_dir/'select.lock', 'a+') as select:
            fcntl.flock(select, fcntl.LOCK_EX)
            busy = load(allowed)
            order = sorted(range(len(blocks)),
                           key=lambda i: (round(sum(busy.get(c, 0.) for c in blocks[i]), 1), i))
            for i in order:
                handle = open(lock_dir/('cpus-' + '-'.join(map(str, blocks[i])) + '.lock'), 'a+')
                try:
                    fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError:
                    handle.close()
                    continue
                if _CPU_SLOT is not None:
                    _CPU_SLOT.close()
                _CPU_SLOT = handle
                return blocks[i], {'source': 'slot_lock', 'lock_dir': str(lock_dir)}
    except OSError:
        pass
    busy = load(allowed)
    return sorted(sorted(allowed, key=lambda c: (busy.get(c, 0.), c))[:threads]), {'source': 'least_loaded'}


def pin_cpus():
    cpus, info = choose_cpus(os.sched_getaffinity(0))
    os.sched_setaffinity(0, cpus)
    info['cpus'] = sorted(os.sched_getaffinity(0))
    return info


def activate_host(host=None, lock=None):
    """Activate the host so concurrent starts never see half-written files.

    vendored_host locks itself (shared to load, exclusive to derive, per versioned directory), so
    processes load in parallel. A host module without LOCKS_ITSELF runs under one exclusive lock.
    """
    host = HOST if host is None else Path(host)
    lock = HOST_LOCK if lock is None else Path(lock)
    if str(host.parent) not in sys.path:
        sys.path.insert(0, str(host.parent))
    module = importlib.import_module(host.stem)
    if getattr(module, 'LOCKS_ITSELF', False):
        return module.activate('native')
    lock.parent.mkdir(parents=True, exist_ok=True)
    with open(lock, 'a+') as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        return module.activate('native')


def activate(device, memory):
    if device == 'cpu':
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
    elif not os.environ.get('CUDA_VISIBLE_DEVICES'):
        raise ValueError('CUDA_VISIBLE_DEVICES must explicitly select a shared GPU')
    cpus = pin_cpus()
    print(f'microglia runtime: CPU affinity {cpus["cpus"]} ({cpus["source"]})', flush=True)
    T,G,provenance = activate_host()
    importlib.import_module('mode_cellshape')
    T.torch.set_num_threads(THREADS)
    T.torch.use_deterministic_algorithms(True)
    T.torch.backends.cuda.matmul.allow_tf32 = False
    T.torch.backends.cudnn.allow_tf32 = False
    T.torch.backends.cudnn.benchmark = False
    T.torch.backends.cudnn.deterministic = True
    if device == 'cuda':T.torch.cuda.set_per_process_memory_fraction(memory)
    else:assert not T.torch.cuda.is_initialized()
    return T,G,provenance,cpus


def check_weights(fit, receipt):
    expected={'final.pt':receipt['final_SHA256'],
              **{f'snap_{step}.pt':digest for step,digest in receipt['snapshots_SHA256'].items()}}
    if set(expected) != {'final.pt', *(f'snap_{step}.pt' for step in (5000,10000,15000,16000,17000,18000,19000))}:
        raise ValueError('fit does not retain the full declared snapshot set')
    for name,digest in expected.items():
        if sha(fit/name) != digest:raise ValueError(f'checkpoint hash mismatch: {name}')
    return expected


def write(path,value):
    """Atomic JSON write through a temp name unique to this writer (durable_state)."""
    ds.atomic_write_bytes(path,(json.dumps(value,indent=2,allow_nan=False)+'\n').encode())


def utc_now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat(timespec='seconds')


def receipt_path(out):
    return out.with_name(out.stem+'.receipt.json')


def pending_path(out):
    return out.with_name('.'+out.stem+'.pending.npz')


def pending_sidecars(pending):
    """The pending NPZ plus the files the generator writes beside it.

    The generator dumps its own receipt next to its `out` and stages np.savez through a
    hidden `.partial.npz`; both would otherwise survive next to the final output.
    """
    receipt=receipt_path(pending)
    return (pending,receipt,receipt.with_name('.'+receipt.name+'.tmp'),
            receipt.with_name(f'.{receipt.name}.{os.getpid()}.tmp'),  # the derived trainer's dump()
            pending.with_name('.'+pending.stem+'.partial.npz'))


def resolve_profile(M,cfg,profile,mode):
    """Resolve the read-out profile; with no --profile, infer it from --mode (the default wins ties)."""
    inferred=profile is None and mode is not None
    if profile is None and mode is None:profile='default'
    if inferred:
        names=['default',*cfg.get('experimental_profiles',{}),*cfg.get('reference_profiles',{})]
        matches=[n for n in names if M.resolve_profile(cfg,n)['generation']['mode']==mode]
        if not matches:raise ValueError('readout is outside the declared application profiles')
        profile=matches[0]
    try:resolved=M.resolve_profile(cfg,profile)
    except KeyError:raise ValueError(f'unknown application profile: {profile}') from None
    if mode is None:mode=resolved['generation']['mode']
    if mode!=resolved['generation']['mode']:
        raise ValueError(f'--mode {mode} differs from profile {profile} ({resolved["generation"]["mode"]})')
    return profile,resolved,mode,inferred


# --- fit lifecycle ---------------------------------------------------------------------------------
# fit_state.json in the fit directory is the explicit state of one fit; it is written last, after the
# files it describes. Every transition is journaled in fit_state.json.journal.jsonl. An exclusive lock on
# <fit>/fit.lock is held across training, receipt and audit, so two trainers on one fit cannot run.

EX_TEMPFAIL=75   # another process holds the fit lock
ALREADY_DONE=3   # train on an AUDITED fit whose files verify: nothing to do
EX_DAMAGED=4     # the recorded state is ahead of the files (DAMAGED); rerun with --repair
FIT_STATE='fit_state.json'
FIT_LOCK='fit'   # FileLock(<fit>/fit) locks <fit>/fit.lock
AUDIT_PASSED='passed_actual_full_EMPF_no_skips'
TRAINER_RECEIPT_STATUSES=('training_completed','completed')  # 'completed': written by the native generator


class Fit(enum.Enum):
    ABSENT='absent'
    CONFIGURED='configured'
    TRAINING='training'
    FAILED_RESUMABLE='failed_resumable'
    TRAINED='trained'            # final.pt, no fit_receipt.json
    RECEIPTED='receipted'        # final.pt + fit_receipt.json
    AUDITED='audited'            # no_skip_audit.json passed after check_fit
    STRANDED='stranded'          # epoch mismatch; left only by an explicit --adopt-epoch
    DAMAGED='damaged'            # the files no longer support the recorded state; left only by --repair


_EPOCH_CHECKED=(Fit.ABSENT,Fit.CONFIGURED,Fit.TRAINING,Fit.FAILED_RESUMABLE,Fit.TRAINED,Fit.RECEIPTED)
# DAMAGED -repair_to_<state>-> the state the files support (infer_fit without check_fit: at most RECEIPTED).
_REPAIR_TARGETS=(Fit.ABSENT,Fit.CONFIGURED,Fit.FAILED_RESUMABLE,Fit.TRAINED,Fit.RECEIPTED)
FIT_TRANSITIONS={
    (Fit.ABSENT,'configure'):Fit.CONFIGURED,
    (Fit.CONFIGURED,'start'):Fit.TRAINING,
    (Fit.TRAINING,'checkpoint'):Fit.TRAINING,
    (Fit.TRAINING,'error'):Fit.FAILED_RESUMABLE,
    (Fit.FAILED_RESUMABLE,'resume'):Fit.TRAINING,
    (Fit.TRAINING,'final_written'):Fit.TRAINED,
    (Fit.FAILED_RESUMABLE,'final_written'):Fit.TRAINED,  # reconcile: an error was recorded after final.pt landed
    (Fit.TRAINED,'receipt_written'):Fit.RECEIPTED,
    (Fit.TRAINED,'reconstruct_receipt'):Fit.RECEIPTED,
    (Fit.RECEIPTED,'audit_passed'):Fit.AUDITED,
    (Fit.RECEIPTED,'audit_failed'):Fit.RECEIPTED,
    **{(state,'epoch_mismatch'):Fit.STRANDED for state in _EPOCH_CHECKED},
    # pre-#19 fits without train_start.json: resume or audit only after an explicit --adopt-epoch
    (Fit.FAILED_RESUMABLE,'adopt_epoch'):Fit.FAILED_RESUMABLE,
    (Fit.TRAINED,'adopt_epoch'):Fit.TRAINED,
    (Fit.RECEIPTED,'adopt_epoch'):Fit.RECEIPTED,
    (Fit.STRANDED,'adopt_epoch'):Fit.FAILED_RESUMABLE,
    (Fit.STRANDED,'adopt_epoch_trained'):Fit.TRAINED,
    # A recorded state the files no longer support (a deleted final.pt, receipt or audit) is journaled as
    # DAMAGED; only an explicit --repair moves it back to the state the files support.
    **{(state,'damaged'):Fit.DAMAGED for state in (Fit.ABSENT,Fit.CONFIGURED,Fit.TRAINED,Fit.RECEIPTED,Fit.AUDITED)},
    **{(Fit.DAMAGED,'repair_to_'+state.value):state for state in _REPAIR_TARGETS},
}
# No terminal state: AUDITED leaves only through 'damaged', which train records when its files fail to verify.
FIT_MACHINE=ds.Machine('microglia_fit',Fit,Fit.ABSENT,(),FIT_TRANSITIONS,
                       adopt_entries=(Fit.CONFIGURED,Fit.FAILED_RESUMABLE,Fit.TRAINED,Fit.RECEIPTED,Fit.AUDITED))


def _read_json(path):
    try:return json.loads(Path(path).read_text())
    except FileNotFoundError:return None


def start_epoch_of(fit):
    """The source-manifest digest recorded in train_start.json, or None."""
    started=_read_json(Path(fit)/'train_start.json')
    return None if started is None else started.get('application_source_manifest_sha256')


def audited_digest(fit):
    audit=_read_json(Path(fit)/'no_skip_audit.json')
    return None if audit is None else audit.get('application_source_manifest_sha256')


def infer_fit(fit,check_fit=None):
    """Read-only legacy reader: (Fit, evidence) from the files of a fit directory. Never writes.

    check_fit() is called only for a receipted fit whose no_skip_audit.json passed; without it such a
    fit reads as RECEIPTED.
    """
    fit=Path(fit)
    names=('config.json','train_start.json','recovery.pt','failure.json','final.pt','checkpoint_receipt.json',
           'fit_receipt.json','no_skip_audit.json')
    evidence={'files':[n for n in names if (fit/n).exists()],'start_epoch':start_epoch_of(fit)}
    if (fit/'final.pt').exists():
        receipt=_read_json(fit/'fit_receipt.json')
        if receipt is None:
            return Fit.TRAINED,evidence
        evidence['receipt_status']=receipt.get('status')
        if receipt.get('status') not in TRAINER_RECEIPT_STATUSES:
            evidence['reason']=f'fit_receipt.json status {receipt.get("status")!r} is not a training receipt'
            return Fit.TRAINED,evidence
        audit=_read_json(fit/'no_skip_audit.json')
        if audit is None or audit.get('status')!=AUDIT_PASSED:
            return Fit.RECEIPTED,evidence
        evidence['audit_source_manifest_sha256']=audit.get('application_source_manifest_sha256')
        if check_fit is None:
            evidence['reason']='no_skip_audit.json present; check_fit not run'
            return Fit.RECEIPTED,evidence
        try:check_fit()
        except (ValueError,KeyError,OSError) as exc:
            evidence['reason']=f'no_skip_audit.json present but check_fit failed: {type(exc).__name__}: {exc}'
            return Fit.RECEIPTED,evidence
        return Fit.AUDITED,evidence
    if (fit/'recovery.pt').exists():
        record=_read_json(fit/'checkpoint_receipt.json') or {}
        matches=record.get('path')=='recovery.pt' and record.get('SHA256')==sha(fit/'recovery.pt')
        evidence['step']=record.get('completed_updates') if matches else None
        return Fit.FAILED_RESUMABLE,evidence
    if (fit/'config.json').exists():
        return Fit.CONFIGURED,evidence
    return Fit.ABSENT,evidence


def fit_state_file(fit):
    return ds.StateFile(Path(fit)/FIT_STATE,FIT_MACHINE)


def adopt_fit(sf,fit,check_fit=None):
    """Return the fit's state, adopting a legacy directory read-only when it has no fit_state.json yet."""
    return sf.adopt(lambda _dir:infer_fit(fit,check_fit),data={'start_epoch':start_epoch_of(fit)})


def _noted(event,**info):
    def update(data):
        data['last']={'event':event,'pid':os.getpid(),'at_utc':utc_now(),**info}
        if 'step' in info and info['step'] is not None:data['step']=info['step']
        return data
    return update


def audit_on_disk(fit,check_fit,digest):
    """None when no_skip_audit.json on disk completes the audit of this fit under `digest`, else why not.

    audit_fit writes no_skip_audit.json only after check_checkpoint, check_weights and a staged check_fit
    passed, and writes fit_state.json after it; this re-runs the fit-local checks on the files as they are.
    """
    fit=Path(fit)
    audit=_read_json(fit/'no_skip_audit.json')
    if audit is None or audit.get('status')!=AUDIT_PASSED:
        return 'no passing no_skip_audit.json'
    if audit.get('application_source_manifest_sha256')!=digest:
        return f'no_skip_audit.json records source manifest {audit.get("application_source_manifest_sha256")}, not {digest}'
    try:
        check_fit()
        check_weights(fit,json.loads((fit/'fit_receipt.json').read_text()))
    except (ValueError,KeyError,OSError) as exc:
        return f'{type(exc).__name__}: {exc}'
    return None


def reconcile_fit(sf,fit,check_fit=None,digest=None):
    """Catch fit_state.json up with files written after its last write (it is always written last).

    Runs under the fit lock, so a TRAINING state means its owner died without recording an outcome. With
    check_fit and the live digest, a RECEIPTED fit whose no_skip_audit.json was written under this fit's
    own epoch and still verifies (audit_fit died between the audit and the state write) becomes AUDITED.
    """
    for _ in range(5):
        snap=sf.read()
        inferred,evidence=infer_fit(fit)
        event,info,update=None,{},None
        if snap.state is Fit.ABSENT and (Path(fit)/'config.json').exists():
            event='configure'
        elif snap.state is Fit.TRAINING:
            if inferred in (Fit.TRAINED,Fit.RECEIPTED):
                event='final_written'
            else:
                event,info='error',{'error':f'owner pid {snap.owner.get("pid")} exited without recording an outcome',
                                    'step':evidence.get('step')}
        elif snap.state is Fit.FAILED_RESUMABLE and inferred in (Fit.TRAINED,Fit.RECEIPTED):
            event='final_written'
        elif snap.state is Fit.TRAINED and inferred is Fit.RECEIPTED:
            event='receipt_written'
        elif snap.state is Fit.RECEIPTED and check_fit is not None and digest is not None:
            start=snap.data.get('start_epoch') or start_epoch_of(fit)
            if (start==digest or snap.data.get('adopted_epoch')==digest) and audit_on_disk(fit,check_fit,digest) is None:
                event='audit_passed'
                update=lambda d:{**_noted('audit_passed',reconciled=True)(d),'audited_epoch':digest,
                                 'audit_sha256':sha(Path(fit)/'no_skip_audit.json')}
        if event is None:
            return snap
        sf.transition(event,update=update or _noted(event,reconciled=True,**info),expect_version=snap.version)
    return sf.read()


def damage_of(sf,fit,check_fit):
    """Why the recorded state is ahead of the fit's files, or None. Read-only."""
    snap=sf.read()
    fit=Path(fit)
    if snap.state is Fit.AUDITED:
        try:
            check_fit()
            check_weights(fit,json.loads((fit/'fit_receipt.json').read_text()))
        except (ValueError,KeyError,OSError) as exc:
            return f'AUDITED fit no longer verifies: {type(exc).__name__}: {exc}'
    elif snap.state in (Fit.TRAINED,Fit.RECEIPTED):
        need=('final.pt',) if snap.state is Fit.TRAINED else ('final.pt','fit_receipt.json')
        missing=[n for n in need if not (fit/n).exists()]
        if missing:
            return f'{snap.state.name} fit is missing {", ".join(missing)}'
    elif snap.state in (Fit.ABSENT,Fit.CONFIGURED) and (fit/'final.pt').exists():
        # final.pt is only written while TRAINING, so no reconcile edge explains it from here
        return f'{snap.state.name} fit already has final.pt'
    return None


def repair_fit(sf,fit):
    """DAMAGED -repair_to_<state>-> the state the files support (journaled with the evidence)."""
    snap=sf.read()
    inferred,evidence=infer_fit(fit)
    sf.transition('repair_to_'+inferred.value,expect_version=snap.version,
                  update=lambda d:{**_noted('repair',to=inferred.name,evidence=evidence)(d),
                                   'repairs':[*d.get('repairs',[]),{'from_reason':d.get('damaged_reason'),
                                              'to':inferred.name,'pid':os.getpid(),'at_utc':utc_now()}]})
    print(f'microglia runtime: repaired DAMAGED fit {fit} to {inferred.name} (journaled)',flush=True)
    return sf.read()


def readopt_fit(sf,fit,check_fit):
    """Archive fit_state.json and its journal beside the fit and adopt the directory again from its files."""
    stamp=datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    archived={}
    for path in (Path(sf.path),Path(sf.journal_path)):
        if path.exists():
            target=path.with_name(f'{path.name}.superseded.{stamp}')
            path.rename(target)
            archived[path.name]=target.name
    ds._fsync_dir(str(fit))
    snap=sf.adopt(lambda _dir:infer_fit(fit,check_fit),
                  data={'start_epoch':start_epoch_of(fit),'readopted':{'archived':archived,'pid':os.getpid(),
                                                                      'at_utc':utc_now()}})
    print(f'microglia runtime: re-adopted {fit} as {snap.state.name}; archived {archived or "nothing"}',flush=True)
    return snap


def remove_superseded_recovery(fit):
    """Under the fit lock: drop a recovery.pt left beside a final.pt that a receipt verifies."""
    fit=Path(fit)
    if not ((fit/'final.pt').exists() and (fit/'recovery.pt').exists()):
        return False
    final_sha=sha(fit/'final.pt')
    record=_read_json(fit/'checkpoint_receipt.json') or {}
    receipt=_read_json(fit/'fit_receipt.json') or {}
    if (record.get('path')=='final.pt' and record.get('SHA256')==final_sha) or receipt.get('final_SHA256')==final_sha:
        (fit/'recovery.pt').unlink()
        print(f'microglia runtime: removed recovery.pt superseded by the verified final.pt in {fit}',flush=True)
        return True
    return False


def fit_events(sf,fit,digest,start_record):
    """The derived trainer's FIT_EVENTS hook: map trainer events onto fit_state.json transitions."""
    def on(event,out,info):
        if Path(out).resolve()!=Path(fit).resolve():
            raise ValueError(f'trainer reported {out}, not the locked fit {fit}')
        snap=sf.read()
        if event=='configure':
            if snap.state is Fit.ABSENT:sf.transition('configure',update=_noted(event))
        elif event=='begin':
            if snap.state is Fit.CONFIGURED:
                pinned=start_epoch_of(fit)
                if pinned not in (None,digest):
                    raise ValueError(f'fit started under source manifest {pinned}; refusing to start it under {digest}')
                if pinned is None:write(Path(fit)/'train_start.json',start_record())
                sf.transition('start',guard=lambda s:start_epoch_of(fit)==digest,
                              update=lambda d:{**_noted(event)(d),'start_epoch':digest})
            elif snap.state is Fit.FAILED_RESUMABLE:
                sf.transition('resume',guard=lambda s:s.data.get('start_epoch')==digest,update=_noted(event,**info))
            else:
                raise ds.IllegalTransition(f'trainer began a run while the fit is {snap.state.name}')
        elif event=='error':
            if snap.state is Fit.TRAINING:sf.transition('error',update=_noted(event,**info))
        else:  # checkpoint, final_written, receipt_written, reconstruct_receipt
            sf.transition(event,update=_noted(event,**info))
    return on


def _strand(sf,snap,reason):
    sf.transition('epoch_mismatch',update=lambda d:{**_noted('epoch_mismatch')(d),'stranded_reason':reason,
                                                    'stranded_from':snap.state.name},expect_version=snap.version)
    raise ValueError(reason)


def _adopt_epoch(digest,previous):
    def update(data):
        data=_noted('adopt_epoch',digest=digest)(data)
        data.setdefault('epoch_adoptions',[]).append({'digest':digest,'previous_start_epoch':previous,
                                                      'pid':os.getpid(),'at_utc':utc_now()})
        data.update(start_epoch=digest,adopted_epoch=digest)
        return data
    return update


def check_epoch(sf,fit,digest,adopt=False):
    """Refuse to resume, audit or relabel a fit under a source manifest other than its own (its epoch).

    A mismatch moves the fit to STRANDED, which only an explicit --adopt-epoch leaves; the adoption is
    journaled. A pre-#19 fit with recovery.pt but no train_start.json has an unknown epoch: resume is
    refused unless --adopt-epoch is given.
    """
    snap=sf.read()
    if snap.state is Fit.STRANDED:
        if not adopt:
            raise ValueError(f'fit {fit} is STRANDED: {snap.data.get("stranded_reason")}; '
                             'it stays stranded until an explicit --adopt-epoch')
        event='adopt_epoch_trained' if (Path(fit)/'final.pt').exists() else 'adopt_epoch'
        sf.transition(event,update=_adopt_epoch(digest,snap.data.get('start_epoch')),expect_version=snap.version)
        print(f'microglia runtime: adopted epoch {digest} for stranded fit {fit} (journaled)',flush=True)
        return reconcile_fit(sf,fit)
    if snap.state not in _EPOCH_CHECKED:
        return snap
    adopted=snap.data.get('adopted_epoch')==digest
    audited=audited_digest(fit)
    if audited is not None and audited!=digest and not adopted:
        _strand(sf,snap,f'{Path(fit)/"no_skip_audit.json"} records source manifest {audited}; '
                        f'refusing to relabel it with {digest}')
    start=snap.data.get('start_epoch') or start_epoch_of(fit)
    if start is not None and start!=digest:
        _strand(sf,snap,f'fit started under source manifest {start}; refusing to resume or audit it under {digest}')
    unknown=start is None and not adopted and (
        (snap.state is Fit.FAILED_RESUMABLE and (Path(fit)/'recovery.pt').exists()) or
        snap.state in (Fit.TRAINED,Fit.RECEIPTED))
    if unknown:
        what='resume' if snap.state is Fit.FAILED_RESUMABLE else 'audit'
        if not adopt:
            raise ValueError(f'{fit} is {snap.state.name} with no train_start.json (it predates start pinning), so '
                             f'the sources it trained under are unknown and {what} is refused; rerun with '
                             f'--adopt-epoch to {what} it under {digest} and journal that adoption')
        snap=sf.transition('adopt_epoch',update=_adopt_epoch(digest,None),expect_version=snap.version)
        print(f'microglia runtime: adopted epoch {digest} for pre-pinning fit {fit} (journaled)',flush=True)
    return snap


def describe_fit(sf,fit,digest):
    snap=sf.read()
    start=snap.data.get('start_epoch') or start_epoch_of(fit)
    return {'fit':str(fit),'state':snap.state.name,'version':snap.version,'start_epoch':start,
            'live_epoch':digest,'audited_epoch':audited_digest(fit),
            'allowed_events':FIT_MACHINE.allowed_events(snap.state),
            'adopted':snap.data.get('adopted'),'last':snap.data.get('last'),
            'stranded_reason':snap.data.get('stranded_reason'),'damaged_reason':snap.data.get('damaged_reason')}


def audit_fit(sf,fit,T,M,cfg,fold,seed,cache,manifest,manifest_digest,cache_anchor):
    """RECEIPTED -audit_passed-> AUDITED: check_fit on a staged copy passes, no_skip_audit.json is written,
    check_fit passes on the fit itself, then the state is written. Any failure is RECEIPTED -audit_failed->."""
    snap=sf.read()
    if snap.state is not Fit.RECEIPTED:
        raise ValueError(f'fit {fit} is {snap.state.name}, not RECEIPTED; it cannot be audited')
    path=fit/'no_skip_audit.json'
    try:
        q=T.torch.load(fit/'final.pt',map_location='cpu',weights_only=False)
        saved=json.loads((fit/'config.json').read_text())
        check_checkpoint(q,saved)
        record=q['policy']['optimizers'][1]['regularizer']['record']
        if not record['calls']==record['observed_steps']==q['completed_updates']==20000:
            raise ValueError('native final checkpoint does not verify regularization every update')
        check_cache(cache,fold)
        check_unchanged_sources(manifest,manifest_digest)
        started=start_epoch_of(fit)
        audit=dict(status=AUDIT_PASSED,penalty_calls=record['calls'],
            observed_steps=record['observed_steps'],completed_updates=q['completed_updates'],lazy_k=1,
            final_checkpoint_sha256=sha(fit/'final.pt'),application_source_manifest_sha256=manifest_digest,
            training_start_source_manifest_sha256=started,cache_anchor=cache_anchor)
        del q
        if started is None:
            audit['training_start_note']='fit predates train_start.json; the sources it started under are not recorded'
        if snap.data.get('epoch_adoptions'):
            audit['epoch_adoptions']=snap.data['epoch_adoptions']
        receipt=json.loads((fit/'fit_receipt.json').read_text())
        with tempfile.TemporaryDirectory(prefix='microglia-audit-') as stage:
            stage=Path(stage)
            for name in ('config.json','fit_receipt.json'):
                shutil.copyfile(fit/name,stage/name)
            write(stage/'no_skip_audit.json',audit)
            M.check_fit(stage,cfg,fold,seed)
        check_weights(fit,receipt)
        existing=_read_json(path)
        if existing!=json.loads(json.dumps(audit)):
            if existing is not None:  # never lose an earlier audit's bytes
                old=path.read_bytes()
                archive=fit/f'no_skip_audit.superseded.{hashlib.sha256(old).hexdigest()[:12]}.json'
                if not archive.exists():ds.atomic_write_bytes(archive,old)
            write(path,audit)
        M.check_fit(fit,cfg,fold,seed)
    except (ValueError,KeyError,OSError) as exc:
        sf.transition('audit_failed',update=_noted('audit_failed',reason=f'{type(exc).__name__}: {exc}'))
        raise
    sf.transition('audit_passed',update=lambda d:{**_noted('audit_passed')(d),'audited_epoch':manifest_digest,
                                                   'audit_sha256':sha(path)})
    return audit


def train(args,fit,M,cfg,fold,cache,cache_anchor,manifest,manifest_digest):
    """Train, receipt and audit one fit under its exclusive lock; return the exit code.

    EX_TEMPFAIL (75) when another process holds the fit, ALREADY_DONE (3) for an AUDITED fit whose files
    verify, EX_DAMAGED (4) when the recorded state is ahead of the files and --repair was not given.
    """
    if args.dry_run and not (fit/FIT_STATE).exists() and not fit.is_dir():
        inferred,evidence=infer_fit(fit)  # read-only: a fit that does not exist is not created by a dry run
        print(json.dumps({'fit':str(fit),'state':None,'would_adopt_as':inferred.name,'evidence':evidence,
                          'live_epoch':manifest_digest},indent=2),flush=True)
        return 0
    fit.mkdir(parents=True,exist_ok=True)
    lock=ds.FileLock(fit/FIT_LOCK,blocking=False,purpose=f'microglia train fold{fold} seed{args.seed}')
    try:lock.acquire()
    except ds.Busy as busy:
        owner=busy.owner or {}
        print(f'microglia runtime: fit {fit} is busy (owner pid {owner.get("pid")}, {owner.get("status")}); '
              f'exit {EX_TEMPFAIL}',file=sys.stderr,flush=True)
        return EX_TEMPFAIL
    try:
        checked=lambda:M.check_fit(fit,cfg,fold,args.seed)
        try:
            sf=ds.StateFile(fit/FIT_STATE,FIT_MACHINE,recover=not args.readopt)
            if args.readopt:
                readopt_fit(sf,fit,checked)
            adopt_fit(sf,fit,checked)
            if args.dry_run:
                print(json.dumps(describe_fit(sf,fit,manifest_digest),indent=2),flush=True)
                return 0
            snap=reconcile_fit(sf,fit,checked,manifest_digest)
        except ds.CorruptState as exc:
            raise ValueError(f'{exc}; fit_state.json or its journal is damaged. Inspect {fit}, then rerun with '
                             f'--readopt to archive both and adopt the directory again from its files') from exc
        ds.sweep_temps(fit,min_age_s=0)  # leftovers of dead writers; nobody else can write without the fit lock
        remove_superseded_recovery(fit)
        damage=damage_of(sf,fit,checked)
        if damage is not None:
            prior=snap.state.name
            snap=sf.transition('damaged',expect_version=snap.version,
                               update=lambda d:{**_noted('damaged',reason=damage)(d),'damaged_reason':damage,
                                                'damaged_from':prior})
            print(f'microglia runtime: fit {fit} is DAMAGED: {damage}',file=sys.stderr,flush=True)
        if snap.state is Fit.DAMAGED:
            if not args.repair:
                print(f'microglia runtime: fit {fit} is DAMAGED ({snap.data.get("damaged_reason")}); inspect it and '
                      f'rerun with --repair to continue from what its files support (exit {EX_DAMAGED})',
                      file=sys.stderr,flush=True)
                return EX_DAMAGED
            repair_fit(sf,fit)
            snap=reconcile_fit(sf,fit,checked,manifest_digest)
        print(f'microglia runtime: fit {fit} is {snap.state.name} (state v{snap.version})',flush=True)
        if snap.state is Fit.AUDITED:
            print(f'microglia runtime: fit is complete and audited; nothing to train (exit {ALREADY_DONE})',flush=True)
            return ALREADY_DONE
        snap=check_epoch(sf,fit,manifest_digest,args.adopt_epoch)
        T,G,provenance,cpus=activate(args.device,args.gpu_mem_fraction)
        expected=json.loads(M.RECIPE.read_text())['native']['recipe']
        if json.loads(json.dumps(T.recipe().to_dict()))!=expected:
            raise ValueError('live trainer differs from selected native EMPF recipe')
        if snap.state is not Fit.RECEIPTED:
            start_record=lambda:dict(schema=1,application_source_manifest_sha256=manifest_digest,started_utc=utc_now(),
                                     fold=fold,seed=args.seed,cache_anchor=cache_anchor,cpu_affinity=cpus,
                                     provenance=provenance)
            native_args=T.parser().parse_args(['--cache',str(cache),'--arm','pbulk','--seed',str(args.seed),
                '--updates','20000','--out-root',str(args.out_root),'--device',args.device,'--threads','3',
                '--gpu-mem-fraction',str(args.gpu_mem_fraction)])
            original=G.generate_fit;G.generate_fit=lambda *a,**kw:(None,{'status':'deferred_to_explicit_application_readout'})
            T.FIT_EVENTS=fit_events(sf,fit,manifest_digest,start_record)
            try:trained=T.run(native_args)
            finally:G.generate_fit=original;T.FIT_EVENTS=None
            if Path(trained).resolve()!=fit:
                raise ValueError(f'trainer wrote {trained}, not the expected fit {fit}')
            snap=reconcile_fit(sf,fit)
            remove_superseded_recovery(fit)
        if snap.state is not Fit.RECEIPTED:
            raise ValueError(f'fit {fit} is {snap.state.name} after training, not RECEIPTED '
                             f'({snap.data.get("adopted",{}).get("evidence",{}).get("reason") or snap.data.get("last")})')
        check_epoch(sf,fit,manifest_digest)
        audit_fit(sf,fit,T,M,cfg,fold,args.seed,cache,manifest,manifest_digest,cache_anchor)
        check_unchanged_sources(manifest,manifest_digest)
        print(f'completed native EMPF training: {fit} (AUDITED)',flush=True)
        return 0
    finally:
        lock.release()


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('command',choices=('train','generate'))
    p.add_argument('--cache',type=Path,required=True)
    p.add_argument('--device',choices=('cpu','cuda'),default='cpu')
    p.add_argument('--threads',type=int,default=3)
    p.add_argument('--gpu-mem-fraction',type=float,default=.04)
    p.add_argument('--arm',default='pbulk');p.add_argument('--seed',type=int,default=0)
    p.add_argument('--updates',type=int,default=20000);p.add_argument('--out-root',type=Path)
    p.add_argument('--fit',type=Path);p.add_argument('--out',type=Path);p.add_argument('--mode')
    p.add_argument('--profile',help='application profile; inferred from --mode when omitted')
    p.add_argument('--force',action='store_true',help='replace an existing --out and its receipt after a successful run')
    p.add_argument('--adopt-epoch',action='store_true',
                   help='train: explicitly adopt the current source manifest as the epoch of a STRANDED fit, or of a '
                        'pre-pinning fit with recovery.pt and no train_start.json (journaled in fit_state.json)')
    p.add_argument('--dry-run',action='store_true',
                   help='train: report the fit state (an existing fit directory is adopted on first touch; a missing '
                        'one is not created); train nothing')
    p.add_argument('--repair',action='store_true',
                   help='train: move a DAMAGED fit (its files no longer support the recorded state) back to the state '
                        'its files support, journaled, and continue from there')
    p.add_argument('--readopt',action='store_true',
                   help='train: archive a damaged fit_state.json and its journal beside the fit and adopt the '
                        'directory again from its files')
    args=p.parse_args(argv)
    if args.threads!=3 or not 0<args.gpu_mem_fraction<=.04 or args.arm!='pbulk' or args.updates!=20000 or args.seed<0:
        p.error('the default requires pbulk/20000,three threads and at most.04 GPU fraction')
    if args.command=='generate' and (args.adopt_epoch or args.dry_run or args.repair or args.readopt):
        p.error('--adopt-epoch, --dry-run, --repair and --readopt are for train')
    manifest_digest=sha(SOURCE_MANIFEST);manifest=check_sources()
    check_unchanged_sources(manifest,manifest_digest)
    M=application();cfg=M.load_config()
    cache=args.cache.resolve()
    fold=json.loads((cache/'manifest.json').read_text())['inner_fold']
    cache_anchor=check_cache(cache,fold,held=args.command=='generate')
    if args.command=='generate':
        if args.fit is None or args.out is None:p.error('generate requires --fit and --out')
        fit=args.fit.resolve();out=args.out.resolve()
        if fit==out.parent or fit in out.parents:p.error('generation output must be outside the fit')
        if out.suffix!='.npz':p.error('--out must end in .npz')
        profile,resolved,mode,inferred=resolve_profile(M,cfg,args.profile,args.mode)
        existing=[str(x) for x in (out,receipt_path(out)) if x.exists()]
        if existing and not args.force:
            raise FileExistsError(f'generation output exists: {", ".join(existing)} (use --force to replace)')
        # Read-only (recover=False: no lock file, no journal line): a fit with an explicit state must be AUDITED
        # (legacy fits without one rely on check_fit).
        fit_state=ds.StateFile(fit/FIT_STATE,FIT_MACHINE,recover=False).read() if (fit/FIT_STATE).exists() else None
        if fit_state is not None and fit_state.state is not Fit.AUDITED:
            raise ValueError(f'{fit/FIT_STATE} records {fit_state.state.name}; only an AUDITED fit generates '
                             f'(microglia_runtime train reconciles, audits or reports it)')
        M.check_fit(fit,cfg,fold,args.seed)
        saved=json.loads((fit/'config.json').read_text())
        if sha(cache/'manifest.json')!=saved['cache_manifest_SHA256']:
            raise ValueError('generation cache differs from trained fit')
        weights=check_weights(fit,json.loads((fit/'fit_receipt.json').read_text()))
        if fit==(ROOT/cfg['runtime']['selected_fit']).resolve():
            selected=json.loads((SELECTION/'selected_model.json').read_text())
            if weights!=selected['weights_sha256']:
                raise ValueError('selected model weights differ from the completed native EMPF pilot')
            for name,digest in selected['metadata_sha256'].items():
                if sha(fit/name)!=digest:raise ValueError(f'selected model metadata changed: {name}')
        print(f'microglia runtime: generating from fit {fit} (profile {profile}, mode {mode}) -> {out}',flush=True)
    elif args.out_root is None:p.error('train requires --out-root')
    else:
        fit=(args.out_root/args.arm/cfg['training']['panel']/f'fold{fold}'/f'seed{args.seed}').resolve()
        return train(args,fit,M,cfg,fold,cache,cache_anchor,manifest,manifest_digest)
    T,G,provenance,cpus=activate(args.device,args.gpu_mem_fraction)
    expected=json.loads(M.RECIPE.read_text())['native']['recipe']
    if json.loads(json.dumps(T.recipe().to_dict()))!=expected:
        raise ValueError('live trainer differs from selected native EMPF recipe')
    q=T.torch.load(fit/'final.pt',map_location='cpu',weights_only=False)
    check_checkpoint(q,saved)
    del q
    # Same five named streams as the fixed EMPF pilot, repeat0.
    def streams(fold,index):
        children=T.np.random.SeedSequence([20261114,6,fold,index,0]).spawn(5)
        return {key:int(child.generate_state(1,dtype=T.np.uint64)[0])%(2**63-16)
                for key,child in zip(G.GEN_STREAMS,children)}
    G.generation_streams=streams
    pending=pending_path(out);pending.unlink(missing_ok=True)
    try:
        _,receipt=G.generate_fit(fit,cache,mode=mode,out=pending,device=args.device)
        check_cache(cache,fold,held=True)
        check_unchanged_sources(manifest,manifest_digest)
        if sha(pending)!=receipt['generated_SHA256']:
            raise ValueError('generated output changed before it was receipted')
        receipt.update(generation_streams=[20261114,6,fold,'development donor index',0],
            cache_anchor=cache_anchor,output=str(out),cpu_affinity=cpus,
            application_default=dict(name=resolved['name'],status=resolved['status'],profile=profile,
                profile_inferred_from_mode=inferred,mode=mode,
                calibration=resolved['generation']['calibration'],evidence=resolved['evidence'],
                population_qualified=False,
                config_sha256=sha(M.CONFIG),source_manifest_sha256=manifest_digest,
                native_recipe='empf',particlegan_version='0.10.0',provenance=provenance))
        if not args.force and (out.exists() or receipt_path(out).exists()):
            raise FileExistsError(f'generation output appeared during the run: {out}')
        # The receipt is written first; the NPZ only appears once it is receipted. Both
        # steps are atomic replacements, so with --force the previous output and receipt
        # stay in place until each is overwritten.
        write(receipt_path(out),receipt)
        pending.replace(out)
    finally:
        for leftover in pending_sidecars(pending):
            leftover.unlink(missing_ok=True)
    check_unchanged_sources(manifest,manifest_digest)
    return 0


if __name__=='__main__':
    try:raise SystemExit(main())
    except (ValueError,OSError,RuntimeError,KeyError,ds.DurableStateError) as exc:raise SystemExit(f'microglia runtime: {type(exc).__name__}: {exc}')
