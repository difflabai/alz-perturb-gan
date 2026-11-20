#!/usr/bin/env python3
"""Execute the frozen native EMPF host under the current application selection."""
import os
os.environ.update(OMP_NUM_THREADS='3', OPENBLAS_NUM_THREADS='3', MKL_NUM_THREADS='3',
                  CUBLAS_WORKSPACE_CONFIG=':4096:8', PYTHONDONTWRITEBYTECODE='1')
import argparse
import atexit
import contextlib
import datetime
import errno
import enum
import fcntl
import hashlib
import importlib
import importlib.util
import json
from pathlib import Path
import re
import secrets
import shutil
import socket
import subprocess
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
    # In-process consumers (e.g. a scorer that imports this module) read the pinned sources too: they hold
    # the shared source-epoch lease from here until exit, so a refreeze cannot land under them.
    hold_source_epoch()
    manifest = json.loads(SOURCE_MANIFEST.read_text())
    for name, digest in manifest['sources'].items():
        path = Path(name) if Path(name).is_absolute() else ROOT / name
        if not path.exists() or sha(path) != digest:
            raise ValueError(f'pinned EMPF runtime source missing or changed: {name}')
    old = manifest['archived_previous_entrypoint']
    if sha(ROOT / old['archive']) != old['sha256']:
        raise ValueError('previous public entrypoint archive changed')
    return manifest


class ManifestChanged(ValueError):
    """The runtime source manifest (or a pinned source) changed while this process was using it."""


class Stranded(ValueError):
    """A fit is STRANDED: its source epoch is not the live one (exit EX_STRANDED)."""


def check_unchanged_sources(started, digest):
    try:
        same = sha(SOURCE_MANIFEST) == digest and check_sources() == started and sha(SOURCE_MANIFEST) == digest
    except ValueError as exc:
        raise ManifestChanged(f'runtime source manifest changed during execution: {exc}') from exc
    if not same:
        raise ManifestChanged('runtime source manifest changed during execution')


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
    hold_source_epoch()  # library callers (score.py) activate without main(): take the lease here
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
    """The fixed staging name used by #19 (legacy; recognized and swept, never written)."""
    return out.with_name('.'+out.stem+'.pending.npz')


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

EX_USAGE=2       # argparse usage errors
EX_TEMPFAIL=75   # another process holds the fit lock, the output lock or (too long) the source-epoch lease
ALREADY_DONE=3   # train on an AUDITED fit whose files verify, or generate onto a COMMITTED output without --force
EX_DAMAGED=4     # the recorded state is ahead of the files (DAMAGED); rerun with --repair
EX_STRANDED=4    # the fit is STRANDED (another source epoch); --audit-under-epoch from its epoch, or --adopt-epoch
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
    STRANDED='stranded'          # epoch mismatch; left by --adopt-epoch, or --audit-under-epoch from its own epoch
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
    # guard: the live manifest digest of the running checkout is the fit's own (stranded) epoch
    (Fit.STRANDED,'audit_under_epoch'):Fit.AUDITED,
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


def stranded_epoch(snap,fit):
    """The source epoch a STRANDED fit belongs to (the one its audit must run under)."""
    return snap.data.get('stranded_epoch') or snap.data.get('start_epoch') or start_epoch_of(fit)


def _strand(sf,snap,reason,epoch=None,hint=None):
    sf.transition('epoch_mismatch',update=lambda d:{**_noted('epoch_mismatch')(d),'stranded_reason':reason,
                                                    'stranded_from':snap.state.name,
                                                    'stranded_epoch':epoch or d.get('start_epoch')},
                  expect_version=snap.version)
    raise Stranded(reason if hint is None else f'{reason}\n{hint()}')


def _adopt_epoch(digest,previous):
    def update(data):
        data=_noted('adopt_epoch',digest=digest)(data)
        data.setdefault('epoch_adoptions',[]).append({'digest':digest,'previous_start_epoch':previous,
                                                      'pid':os.getpid(),'at_utc':utc_now()})
        data.update(start_epoch=digest,adopted_epoch=digest)
        return data
    return update


def check_epoch(sf,fit,digest,adopt=False,hint=None):
    """Refuse to resume, audit or relabel a fit under a source manifest other than its own (its epoch).

    A mismatch moves the fit to STRANDED, which only an explicit --adopt-epoch leaves; the adoption is
    journaled. A pre-#19 fit with recovery.pt but no train_start.json has an unknown epoch: resume is
    refused unless --adopt-epoch is given.
    """
    snap=sf.read()
    if snap.state is Fit.STRANDED:
        if not adopt:
            message=(f'fit {fit} is STRANDED: {snap.data.get("stranded_reason")}; it stays stranded until '
                     '--audit-under-epoch from a checkout of its own epoch, or an explicit --adopt-epoch')
            raise Stranded(message if hint is None else f'{message}\n{hint(stranded_epoch(snap,fit))}')
        event='adopt_epoch_trained' if (Path(fit)/'final.pt').exists() else 'adopt_epoch'
        sf.transition(event,update=_adopt_epoch(digest,snap.data.get('start_epoch')),expect_version=snap.version)
        print(f'microglia runtime: adopted epoch {digest} for stranded fit {fit} (journaled)',flush=True)
        return reconcile_fit(sf,fit)
    if snap.state not in _EPOCH_CHECKED:
        return snap
    adopted=snap.data.get('adopted_epoch')==digest
    audited=audited_digest(fit)
    start=snap.data.get('start_epoch') or start_epoch_of(fit)
    if audited is not None and audited!=digest and not adopted:
        _strand(sf,snap,f'{Path(fit)/"no_skip_audit.json"} records source manifest {audited}; '
                        f'refusing to relabel it with {digest}',start or audited,hint and (lambda:hint(start or audited)))
    if start is not None and start!=digest:
        _strand(sf,snap,f'fit started under source manifest {start}; refusing to resume or audit it under {digest}',
                start,hint and (lambda:hint(start)))
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


def _write_audit(snap,fit,T,M,cfg,fold,seed,cache,manifest,manifest_digest,cache_anchor):
    """check_fit on a staged copy passes, no_skip_audit.json is written, check_fit passes on the fit itself."""
    path=fit/'no_skip_audit.json'
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
    if snap.state is Fit.STRANDED:
        audit['audited_under_epoch']={'stranded_reason':snap.data.get('stranded_reason'),'epoch':manifest_digest}
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
    return audit,path


def audit_fit(sf,fit,T,M,cfg,fold,seed,cache,manifest,manifest_digest,cache_anchor,hint=None):
    """RECEIPTED -audit_passed-> AUDITED: check_fit on a staged copy passes, no_skip_audit.json is written,
    check_fit passes on the fit itself, then the state is written. A manifest change during execution strands
    the fit under its own epoch (RECEIPTED -epoch_mismatch-> STRANDED, reason 'manifest changed during
    execution'), so it can be audited later from that epoch; any other failure is RECEIPTED -audit_failed->."""
    snap=sf.read()
    if snap.state is not Fit.RECEIPTED:
        raise ValueError(f'fit {fit} is {snap.state.name}, not RECEIPTED; it cannot be audited')
    try:
        audit,path=_write_audit(snap,fit,T,M,cfg,fold,seed,cache,manifest,manifest_digest,cache_anchor)
    except ManifestChanged as exc:
        _strand(sf,sf.read(),f'manifest changed during execution ({exc})',manifest_digest,
                hint and (lambda:hint(manifest_digest)))
    except (ValueError,KeyError,OSError) as exc:
        sf.transition('audit_failed',update=_noted('audit_failed',reason=f'{type(exc).__name__}: {exc}'))
        raise
    sf.transition('audit_passed',update=lambda d:{**_noted('audit_passed')(d),'audited_epoch':manifest_digest,
                                                   'audit_sha256':sha(path)})
    return audit


def audit_under_epoch(sf,fit,T,M,cfg,fold,seed,cache,manifest,manifest_digest,cache_anchor):
    """STRANDED -audit_under_epoch-> AUDITED, guarded: the live manifest digest is the fit's own epoch.

    Runs the same audit as audit_fit; it is meant for a checkout of the commit whose manifest has that digest
    (microglia_runtime prints the git worktree and audit commands when it strands a fit). A failed audit
    leaves the fit STRANDED.
    """
    snap=sf.read()
    if snap.state is not Fit.STRANDED:
        raise ValueError(f'fit {fit} is {snap.state.name}, not STRANDED; --audit-under-epoch is only for stranded fits')
    epoch=stranded_epoch(snap,fit)
    if epoch!=manifest_digest:
        raise Stranded(f'fit {fit} belongs to source epoch {epoch}, but this checkout runs {manifest_digest}; '
                       '--audit-under-epoch needs a checkout of that epoch')
    inferred,evidence=infer_fit(fit)
    if inferred not in (Fit.RECEIPTED,):
        raise ValueError(f'fit {fit} is stranded with files of a {inferred.name} fit; --audit-under-epoch needs '
                         'final.pt and a training fit_receipt.json (use --adopt-epoch to continue training it)')
    audit,path=_write_audit(snap,fit,T,M,cfg,fold,seed,cache,manifest,manifest_digest,cache_anchor)
    sf.transition('audit_under_epoch',expect_version=snap.version,
                  guard=lambda s:stranded_epoch(s,fit)==manifest_digest==sha(SOURCE_MANIFEST),
                  update=lambda d:{**_noted('audit_under_epoch',epoch=manifest_digest)(d),
                                   'audited_epoch':manifest_digest,'audit_sha256':sha(path)})
    print(f'microglia runtime: audited stranded fit {fit} under its own epoch {manifest_digest} (AUDITED)',flush=True)
    return audit


def train(args,fit,M,cfg,fold,cache,cache_anchor,manifest,manifest_digest):
    """Train, receipt and audit one fit under its exclusive lock; return the exit code.

    EX_TEMPFAIL (75) when another process holds the fit, ALREADY_DONE (3) for an AUDITED fit whose files
    verify, EX_DAMAGED (4) when the recorded state is ahead of the files and --repair was not given. A
    STRANDED fit raises Stranded (exit EX_STRANDED, 4) with the worktree and audit commands of its epoch.
    """
    hint=lambda epoch:epoch_recovery_hint(fit,epoch,args)
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
        if args.audit_under_epoch:
            if snap.state is not Fit.STRANDED:
                raise ValueError(f'fit {fit} is {snap.state.name}, not STRANDED; --audit-under-epoch is only for '
                                 'stranded fits')
            epoch=stranded_epoch(snap,fit)
            if epoch!=manifest_digest:  # refuse before any host activation
                raise Stranded(f'fit {fit} belongs to source epoch {epoch}, but this checkout runs {manifest_digest}; '
                               f'--audit-under-epoch needs a checkout of that epoch\n{hint(epoch)}')
            T,G,provenance,cpus=activate(args.device,args.gpu_mem_fraction)
            audit_under_epoch(sf,fit,T,M,cfg,fold,args.seed,cache,manifest,manifest_digest,cache_anchor)
            check_unchanged_sources(manifest,manifest_digest)
            return 0
        snap=check_epoch(sf,fit,manifest_digest,args.adopt_epoch,hint)
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
        check_epoch(sf,fit,manifest_digest,hint=hint)
        audit_fit(sf,fit,T,M,cfg,fold,args.seed,cache,manifest,manifest_digest,cache_anchor,hint)
        check_unchanged_sources(manifest,manifest_digest)
        print(f'completed native EMPF training: {fit} (AUDITED)',flush=True)
        return 0
    finally:
        lock.release()


# --- generation output promotion ------------------------------------------------------------------
# <out_stem>.state.json beside <out>.npz and <out_stem>.receipt.json is the explicit state of one
# generation output; FileLock(<out>) (<out>.lock) is held by the one runtime process that may stage,
# promote or recover it. Pending files carry the run id, so two runs never share a staging name.

class Out(enum.Enum):
    NONE='none'                        # no output, no receipt
    STAGING='staging'                  # a run is generating into its own pending names
    STAGED='staged'                    # pending NPZ and pending receipt written and post-checked (shas recorded)
    PROMOTING='promoting'              # journaled before any replace; pending files are kept until COMMITTED
    COMMITTED='committed'              # out.npz and receipt match each other and the recorded shas
    INCONSISTENT='inconsistent'        # receipt and NPZ do not describe each other (e.g. a pre-state G1 pair)
    ORPHAN_RECEIPT='orphan_receipt'    # receipt without NPZ
    UNRECEIPTED_NPZ='unreceipted_npz'  # NPZ without receipt (pre-#19 outputs)


_OUT_STABLE=(Out.NONE,Out.COMMITTED,Out.INCONSISTENT,Out.ORPHAN_RECEIPT,Out.UNRECEIPTED_NPZ)
OUT_TRANSITIONS={
    # stage: guard in stage_output (NONE always; any other stable state only with --force)
    **{(state,'stage'):Out.STAGING for state in _OUT_STABLE},
    (Out.STAGING,'post_checks_passed'):Out.STAGED,
    # a failed or abandoned run returns to the state it started from; its pending files are unlinked first
    **{(Out.STAGING,'abort_to_'+state.value):state for state in _OUT_STABLE},
    **{(Out.STAGED,'abort_to_'+state.value):state for state in _OUT_STABLE},
    (Out.STAGED,'begin_promote'):Out.PROMOTING,
    (Out.PROMOTING,'commit'):Out.COMMITTED,
    (Out.PROMOTING,'lost'):Out.INCONSISTENT,  # pending files gone and the finals do not match (outside interference)
    # files changed outside the runtime: the recorded stable state follows what is on disk, except that a
    # consistent pair this machine did not commit is observed as INCONSISTENT (only --force replaces it);
    # COMMITTED and INCONSISTENT self-loops record new shas
    **{(a,'observed_'+b.value):b for a in _OUT_STABLE for b in _OUT_STABLE
       if a is not b or a in (Out.COMMITTED,Out.INCONSISTENT)},
}
OUT_MACHINE=ds.Machine('microglia_output',Out,Out.NONE,(),OUT_TRANSITIONS,
                       adopt_entries=(Out.COMMITTED,Out.INCONSISTENT,Out.ORPHAN_RECEIPT,Out.UNRECEIPTED_NPZ))
OUTPUT_FAULTS=('out_after_pending_npz','out_after_pending_receipt','out_after_begin_promote',
               'out_after_npz_replace','out_after_receipt_replace','out_before_commit')
LEGACY_SIDECAR_MIN_AGE_S=6*3600  # #19 fixed-name pending files: a pre-lock runtime may still be writing younger ones
_RUN_ID=re.compile(r'r\d{8}T\d{6}-\d+-[0-9a-f]{8}')


def output_state_path(out):
    out=Path(out)
    return out.with_name(out.stem+'.state.json')


def new_run_id():
    stamp=datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S')
    return f'r{stamp}-{os.getpid()}-{secrets.token_hex(4)}'


def run_pending(out,run_id):
    """(pending NPZ, pending receipt) of one run: .<stem>.<run_id>.pending.npz/.receipt.json."""
    out=Path(out)
    npz=out.with_name(f'.{out.stem}.{run_id}.pending.npz')
    return npz,receipt_path(npz)


def _pending_names(out):
    """{'run': [paths of run-specific staging files], 'legacy': [#19 fixed-name staging files]} beside out."""
    out=Path(out)
    stem=re.escape(out.stem)
    run=re.compile(rf'\.?\.{stem}\.(?P<run>{_RUN_ID.pattern})\.pending[.]')
    legacy=re.compile(rf'\.?\.{stem}\.pending[.]')
    found={'run':[],'legacy':[]}
    if not out.parent.is_dir():
        return found
    for path in sorted(out.parent.iterdir()):
        if run.match(path.name):found['run'].append(path)
        elif legacy.match(path.name):found['legacy'].append(path)
    return found


def _read_receipt(path):
    """The receipt as a dict; None when absent; {} when unreadable or not a JSON object (a list, string...)."""
    try:doc=json.loads(Path(path).read_text())
    except FileNotFoundError:return None
    except (OSError,ValueError):return {}
    return doc if isinstance(doc,dict) else {}


def infer_output(out):
    """Read-only legacy reader: (Out, evidence) from out.npz and its receipt. Never writes."""
    out=Path(out);receipt=receipt_path(out)
    have_npz,have_receipt=out.exists(),receipt.exists()
    evidence={'npz':have_npz,'receipt':have_receipt}
    if have_npz:evidence['npz_sha']=sha(out)
    if have_receipt:evidence['receipt_sha']=sha(receipt)
    if have_npz and have_receipt:
        recorded=(_read_receipt(receipt) or {}).get('generated_SHA256')
        evidence['receipt_generated_SHA256']=recorded
        return (Out.COMMITTED if recorded==evidence['npz_sha'] else Out.INCONSISTENT),evidence
    if have_receipt:return Out.ORPHAN_RECEIPT,evidence
    if have_npz:return Out.UNRECEIPTED_NPZ,evidence
    return Out.NONE,evidence


def observe_files(out,recorded=None,data=None):
    """(Out, evidence): the stable state a recorded state `recorded` must follow, from the files. Read-only.

    Like infer_output, except that a consistent pair this machine did not commit (recorded is not COMMITTED,
    or COMMITTED with other shas: a receipt edited in place, a pair copied in) is INCONSISTENT, so a routine
    run can never turn bytes the runtime never produced into COMMITTED. recorded=None is legacy adoption.
    """
    observed,evidence=infer_output(out)
    if observed is Out.COMMITTED and recorded is not None:
        data=data if isinstance(data,dict) else {}
        if not (recorded is Out.COMMITTED and (data.get('npz_sha'),data.get('receipt_sha'))==
                (evidence.get('npz_sha'),evidence.get('receipt_sha'))):
            return Out.INCONSISTENT,{**evidence,'foreign_pair':True}
    return observed,evidence


def _journal_ahead(state_path,version):
    """True when <state>.journal.jsonl records a seq beyond `version` (the runtime refuses that file)."""
    return ds.replay_journal(str(state_path)+ds.JOURNAL_SUFFIX)[1]>version


def is_committed(out):
    """score.py-compatible check that out.npz is a complete, receipted generation output. Read-only.

    With <stem>.state.json it must be a valid microglia_output state file (schema, machine, version, journal
    not ahead) recording COMMITTED, and both files must still have the recorded shas; a legacy output
    without a state file (and without an orphaned journal) is valid only when its receipt's
    generated_SHA256 == sha(out.npz). Any parse or type error reads as False, never as an exception.
    """
    out=Path(out);receipt=receipt_path(out);state=output_state_path(out)
    try:
        if state.exists():
            snap=ds.StateFile(state,OUT_MACHINE,recover=False).read()
            if snap is None or snap.state is not Out.COMMITTED or _journal_ahead(state,snap.version):
                return False
            if not (out.exists() and receipt.exists()):
                return False
            if sha(out)!=snap.data.get('npz_sha') or sha(receipt)!=snap.data.get('receipt_sha'):
                return False
        elif Path(str(state)+ds.JOURNAL_SUFFIX).exists():
            return False  # a journal without its state file is damaged, not legacy (generate --readopt)
        return infer_output(out)[0] is Out.COMMITTED
    except (OSError,ValueError,TypeError,AttributeError,KeyError,ds.DurableStateError):
        return False


def output_state_file(out):
    return ds.StateFile(output_state_path(out),OUT_MACHINE)


def _drop(paths):
    for path in paths:
        Path(path).unlink(missing_ok=True)


def _run_sidecars(out,run_id):
    return [p for p in _pending_names(out)['run'] if _RUN_ID.search(p.name).group(0)==run_id]


def _committed_shas(evidence):
    return {'npz_sha':evidence.get('npz_sha'),'receipt_sha':evidence.get('receipt_sha')}


def _observe(sf,out,snap):
    """Bring a stable recorded state level with the files on disk (journaled as observed_<state>)."""
    observed,evidence=observe_files(out,snap.state,snap.data)
    data=snap.data
    if observed is snap.state and (observed not in (Out.COMMITTED,Out.INCONSISTENT) or (
            data.get('npz_sha'),data.get('receipt_sha'))==(evidence.get('npz_sha'),evidence.get('receipt_sha'))):
        return snap
    foreign=' (a consistent pair this runtime did not commit; --force replaces it)' if evidence.get('foreign_pair') else ''
    print(f'microglia runtime: output {out} recorded {snap.state.name} but its files are {observed.name}{foreign}; '
          f'journaled as observed_{observed.value}',file=sys.stderr,flush=True)
    return sf.transition('observed_'+observed.value,expect_version=snap.version,
                         update=lambda d:{**d,**_committed_shas(evidence),'observed':evidence,'pending':None})


def _promote(sf,out,snap):
    """PROMOTING -> COMMITTED: replace the NPZ, then the receipt, from the pending files named in the state.

    Idempotent, so it is also the roll-forward of a run that died while PROMOTING. Pending files are
    moved, never deleted, until the commit is written; a pending file that no longer has its recorded
    sha, or finals that match neither the pending files nor the recorded shas, are journaled as 'lost'.
    """
    data=snap.data;pending=data.get('pending');run_id=data.get('run_id')
    # Only this run's own staging names in this directory: never a path from elsewhere (absolute, '..').
    expected=(tuple(p.name for p in run_pending(out,run_id))
              if isinstance(run_id,str) and _RUN_ID.fullmatch(run_id) else None)
    if not isinstance(pending,dict) or expected is None or (pending.get('npz'),pending.get('receipt'))!=expected \
            or not all(isinstance(data.get(k),str) for k in ('npz_sha','receipt_sha')):
        return sf.transition('lost',expect_version=snap.version,
                             update=_noted('lost',reason='the PROMOTING record does not name this run\'s own staging '
                                                         f'files and shas (pending {pending!r}, run {run_id!r})'))
    npz,receipt=out.parent/expected[0],out.parent/expected[1]
    final_receipt=receipt_path(out)
    for source,target,key,fault in ((npz,out,'npz_sha','out_after_npz_replace'),
                                    (receipt,final_receipt,'receipt_sha','out_after_receipt_replace')):
        if source.exists():
            if sha(source)!=data[key]:
                return sf.transition('lost',expect_version=snap.version,
                                     update=_noted('lost',reason=f'{source.name} does not have its recorded {key}'))
            os.replace(source,target)
            ds._fsync_dir(str(out.parent))
            ds._fault(fault)
    if not (out.exists() and final_receipt.exists() and sha(out)==data['npz_sha']
            and sha(final_receipt)==data['receipt_sha']):
        return sf.transition('lost',expect_version=snap.version,
                             update=_noted('lost',reason='pending files are gone and the outputs do not match them'))
    ds._fault('out_before_commit')
    snap=sf.transition('commit',expect_version=snap.version,
                       update=lambda d:{**_noted('commit')(d),'pending':None,'previous':None,
                                        'committed_run':d.get('run_id'),'committed_utc':utc_now()})
    _drop(_run_sidecars(out,data['run_id']))
    return snap


def _abort(sf,out,snap,reason):
    """STAGING/STAGED -> the state the run started from; its pending files are unlinked first.

    A previous record that is missing, unknown or not a stable state cannot be trusted: the run then
    returns to the state its files support (a consistent pair as INCONSISTENT, since its origin is unknown).
    """
    run_id=snap.data.get('run_id')
    if isinstance(run_id,str) and _RUN_ID.fullmatch(run_id):
        _drop(_run_sidecars(out,run_id))
    previous=snap.data.get('previous')
    previous=previous if isinstance(previous,dict) else {}
    target=Out.__members__.get(previous.get('state')) if isinstance(previous.get('state'),str) else None
    if target in _OUT_STABLE:
        shas={k:previous.get(k) for k in ('npz_sha','receipt_sha')}
    else:
        target,evidence=observe_files(out,Out.NONE)
        shas=_committed_shas(evidence)
        reason+=f'; previous state {previous.get("state")!r} is not a stable state, so the files decide'
    return sf.transition('abort_to_'+target.value,expect_version=snap.version,
                         update=lambda d:{**_noted('abort',reason=reason)(d),'pending':None,**shas})


def sweep_output(out):
    """Under the output lock: remove staging files no live run owns. Returns what was removed and kept."""
    names=_pending_names(out)
    removed,kept=[],[]
    for path in names['run']:  # a run-specific file is written only under this lock: its owner is gone
        path.unlink(missing_ok=True);removed.append(path.name)
    now=time.time()
    for path in names['legacy']:  # #19 fixed names; a pre-lock runtime may still be writing a young one
        try:age=now-path.stat().st_mtime
        except FileNotFoundError:continue
        if age>=LEGACY_SIDECAR_MIN_AGE_S:
            path.unlink(missing_ok=True);removed.append(path.name)
        else:kept.append(path.name)
    if removed:
        print(f'microglia runtime: removed staging files of dead runs beside {out}: {removed}',flush=True)
    if kept:
        print(f'microglia runtime: kept young legacy (#19) staging files beside {out}: {kept}',file=sys.stderr,flush=True)
    temps=sweep_output_temps(out)
    if temps:
        print(f'microglia runtime: removed temp files of dead writers beside {out}: {temps}',flush=True)
    return removed,kept


def sweep_output_temps(out):
    """Remove durable_state temps (.<target>.<rand>.<pid>.tmp) of this output whose writer is dead.

    Targets: <stem>.state.json, its and <out>'s lock-owner records, and this output's run staging files.
    A temp whose writer pid is alive (this process included) is kept.
    """
    out=Path(out)
    state=output_state_path(out).name
    targets={state,state+ds.OWNER_SUFFIX,out.name+ds.OWNER_SUFFIX}
    run=re.compile(rf'\.{re.escape(out.stem)}\.{_RUN_ID.pattern}\.pending[.]')
    removed=[]
    if not out.parent.is_dir():
        return removed
    for entry in os.scandir(out.parent):
        own=ds._own_temp(entry.name)
        if own is None or not (own[0] in targets or run.match(own[0])):
            continue
        try:mtime=entry.stat(follow_symlinks=False).st_mtime
        except FileNotFoundError:continue
        if ds._writer_alive(own[1],mtime):
            continue
        Path(entry.path).unlink(missing_ok=True);removed.append(entry.name)
    return sorted(removed)


class OutputDamaged(ValueError):
    """<stem>.state.json (or its journal) cannot be used; `generate --readopt` archives both and adopts again."""


def readopt_output(out):
    """Under the output lock: archive <stem>.state.json and its journal beside the output, so the next open
    adopts the output again from its files (legacy rules: a receipt naming the NPZ's sha is COMMITTED)."""
    out=Path(out)
    stamp=datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    state=output_state_path(out)
    archived={}
    for path in (state,Path(str(state)+ds.JOURNAL_SUFFIX)):
        if path.exists():
            target=path.with_name(f'{path.name}.superseded.{stamp}')
            os.replace(path,target)
            archived[path.name]=target.name
    if out.parent.is_dir():
        ds._fsync_dir(str(out.parent))
    print(f'microglia runtime: --readopt archived {archived or "nothing"} beside {out}',flush=True)
    return archived


def open_output(out,readopt=False):
    """Under the output lock: adopt (read-only for the files), roll forward or back, observe, sweep.

    An unusable state file (bad schema/machine/version/state, a journal ahead of it or without it) raises
    OutputDamaged; readopt=True archives it first and adopts the output again from its files.
    """
    if readopt:
        readopt_output(out)
    try:
        return _open_output(out)
    except (ds.CorruptState,ds.InvalidMachine) as exc:
        raise OutputDamaged(f'{exc}; inspect {output_state_path(out).name} and its journal, then rerun generate with '
                            '--readopt to archive them and adopt the output again from its files') from None


def _open_output(out):
    sf=output_state_file(out)
    inferred,evidence=infer_output(out)  # under the output lock, so the files cannot change in between
    snap=sf.adopt(lambda _dir:(inferred,evidence),data={'pending':None,**_committed_shas(evidence)})
    if snap.state is Out.PROMOTING:
        print(f'microglia runtime: output {out} was PROMOTING when its run stopped; rolling forward',flush=True)
        snap=_promote(sf,out,snap)
    elif snap.state in (Out.STAGING,Out.STAGED):
        print(f'microglia runtime: output {out} was {snap.state.name} when its run stopped; rolling back',flush=True)
        snap=_abort(sf,out,snap,f'owner pid {snap.owner.get("pid")} stopped while {snap.state.name}')
    snap=_observe(sf,out,snap)
    sweep_output(out)
    return sf,snap


def stage_output(sf,out,snap,force,run_id):
    """Stable state -stage-> STAGING; guard: NONE, or --force with any other stable state."""
    if snap.state is not Out.NONE and not force:
        raise ds.GuardFailed(f'output {out} is {snap.state.name}; --force is required to replace it')
    previous={'state':snap.state.name,'npz_sha':snap.data.get('npz_sha'),'receipt_sha':snap.data.get('receipt_sha')}
    npz,receipt=run_pending(out,run_id)
    return sf.transition('stage',expect_version=snap.version,
                         guard=lambda s:s.state is Out.NONE or force,
                         update=lambda d:{**_noted('stage',run_id=run_id)(d),'run_id':run_id,'previous':previous,
                                          'pending':{'npz':npz.name,'receipt':receipt.name}})


def finish_output(sf,out,run_id,receipt):
    """STAGING -post_checks_passed-> STAGED -begin_promote-> PROMOTING -> COMMITTED.

    The pending receipt replaces whatever the generator dumped beside the pending NPZ.
    """
    npz,pending_receipt=run_pending(out,run_id)
    write(pending_receipt,receipt)
    ds._fault('out_after_pending_receipt')
    shas={'npz_sha':sha(npz),'receipt_sha':sha(pending_receipt)}
    if shas['npz_sha']!=receipt['generated_SHA256']:
        raise ValueError('generated output changed before it was receipted')
    snap=sf.read()
    snap=sf.transition('post_checks_passed',expect_version=snap.version,
                       guard=lambda s:s.data.get('run_id')==run_id,update=lambda d:{**d,**shas})
    snap=sf.transition('begin_promote',expect_version=snap.version,update=_noted('begin_promote',run_id=run_id))
    ds._fault('out_after_begin_promote')
    return _promote(sf,out,snap)


def output_failed(sf,out,run_id,exc):
    """Run failure: STAGING/STAGED roll back now; PROMOTING is left for the next open to roll forward."""
    snap=sf.read()
    if snap is None or snap.data.get('run_id')!=run_id:
        return snap
    if snap.state in (Out.STAGING,Out.STAGED):
        return _abort(sf,out,snap,f'{type(exc).__name__}: {exc}')
    if snap.state is Out.PROMOTING:
        print(f'microglia runtime: output {out} stays PROMOTING after {type(exc).__name__}; the next run rolls it '
              'forward from the retained pending files',file=sys.stderr,flush=True)
    return snap


# --- source-epoch lease ---------------------------------------------------------------------------
# Every runtime process holds a shared FileLock on <epoch dir>/lease for its lifetime and records itself in
# holders/. scripts/refreeze_runtime_manifest.py takes the exclusive lock, so a refreeze never lands while
# a runtime process is using the pinned sources; it refuses and lists the live holders (or --wait).
# <epoch dir>/gate is the intent gate: a runtime passes it (shared, briefly) before taking the lease, and a
# refreeze holds it exclusively while it waits for the lease, so new runtimes queue behind a pending
# refreeze instead of starving it (flock gives a waiting exclusive locker no preference).
# In-process consumers that import this module (score.py) take the lease through check_sources()/activate()
# (held until exit), or explicitly with `with source_epoch_lease(cmd):`.

EPOCH_DIR_ENV='MICROGLIA_EPOCH_DIR'
EPOCH_WAIT_ENV='MICROGLIA_EPOCH_WAIT_S'
_READ_ONLY_ERRNOS=(errno.EACCES,errno.EPERM,errno.EROFS)


def epoch_dir(root=None):
    """The source-epoch lease directory: $MICROGLIA_EPOCH_DIR, else <checkout>/runs/.source-epoch.

    scripts/refreeze_runtime_manifest.epoch_dir resolves it the same way (tests/test_microglia_epoch_lease.py
    checks that both agree), so a refreeze always locks the directory the runtimes hold.
    """
    return Path(os.environ.get(EPOCH_DIR_ENV) or Path(ROOT if root is None else root)/'runs/.source-epoch')


class EpochLease:
    """The shared source-epoch lease of one runtime process (created lazily; absence means no holders).

    optional=True (read-only commands such as train --dry-run): when the epoch directory cannot be written
    and has no lease file yet, warn and continue without a lease. A directory that cannot be written but has
    a lease file is locked read-only (shared flock) without a holder record.
    """

    def __init__(self,cmd,directory=None,timeout=None,optional=False):
        self.directory=Path(directory) if directory is not None else epoch_dir()
        self.cmd=cmd
        self.timeout=float(os.environ.get(EPOCH_WAIT_ENV,'60')) if timeout is None else timeout
        self.optional=optional
        self.lock=None
        self.record=None
        self.degraded=None
        self.info={}

    @property
    def held(self):
        return self.lock is not None or self.degraded=='no lease'

    def acquire(self):
        writable=True
        try:
            (self.directory/'holders').mkdir(parents=True,exist_ok=True)
            if not os.access(self.directory/'holders',os.W_OK|os.X_OK):
                raise PermissionError(errno.EACCES,os.strerror(errno.EACCES),str(self.directory/'holders'))
        except OSError as exc:
            if exc.errno not in _READ_ONLY_ERRNOS:raise
            writable=False
            if not (self.directory/('lease'+ds.LOCK_SUFFIX)).exists():
                if self.optional:
                    self.degraded='no lease'
                    print(f'microglia runtime: cannot create the source-epoch lease in {self.directory} '
                          f'({exc.strerror}); continuing without it (read-only command)',file=sys.stderr,flush=True)
                    return self
                raise PermissionError(exc.errno,f'cannot create the source-epoch lease in {self.directory} '
                                      f'({exc.strerror}); train and generate need write access to it (or set '
                                      f'{EPOCH_DIR_ENV}); --dry-run works without it') from None
            self.degraded='read-only'
            print(f'microglia runtime: {self.directory} is not writable; holding its source-epoch lease read-only '
                  'without a holder record',file=sys.stderr,flush=True)
        deadline=time.monotonic()+self.timeout
        gate=ds.FileLock(self.directory/'gate',exclusive=False,timeout=self.timeout)
        try:
            gate.acquire()
        except PermissionError:
            gate=None  # no gate file and no write access: the gate only orders waiters, the lease still excludes
        try:
            lock=ds.FileLock(self.directory/'lease',exclusive=False,timeout=max(0.,deadline-time.monotonic()))
            lock.acquire()
            self.lock=lock
        finally:
            if gate is not None:gate.release()
        if writable:
            prune_holders(self.directory)
            pid=os.getpid()
            self.record=self.directory/'holders'/f'{pid}-{ds.pid_starttime(pid)}-{secrets.token_hex(4)}.json'
        self.update(cmd=self.cmd)
        return self

    def update(self,**info):
        self.info.update(info)
        if self.record is None:
            return
        pid=os.getpid()
        ds.atomic_write_json(self.record,{'pid':pid,'pid_starttime':ds.pid_starttime(pid),'host':socket.gethostname(),
                                          'acquired_utc':self.info.setdefault('acquired_utc',utc_now()),**self.info})

    def release(self):
        if self.record is not None:
            self.record.unlink(missing_ok=True);self.record=None
        if self.lock is not None:
            self.lock.release();self.lock=None
        self.degraded=None


_PROCESS_LEASE=None  # the lease this process holds (main(), or taken lazily by a library consumer)


def hold_source_epoch(cmd=None):
    """Hold this process's shared source-epoch lease (taken now if not held yet; released at exit).

    check_sources() and activate() call it, so a library consumer that imports this module holds the lease
    from its first source check or activation until it exits. Raises ds.Busy while a refreeze holds it.
    """
    global _PROCESS_LEASE
    lease=_PROCESS_LEASE
    if lease is not None and lease.held and lease.directory==epoch_dir():
        return lease
    lease=EpochLease(cmd or ['library',*sys.argv]).acquire()
    atexit.register(lease.release)
    _PROCESS_LEASE=lease
    return lease


@contextlib.contextmanager
def source_epoch_lease(cmd=None):
    """`with source_epoch_lease(cmd):` holds the shared source-epoch lease for the block (reentrant: an
    already held process lease is reused and kept)."""
    global _PROCESS_LEASE
    if _PROCESS_LEASE is not None and _PROCESS_LEASE.held and _PROCESS_LEASE.directory==epoch_dir():
        yield _PROCESS_LEASE
        return
    lease=EpochLease(cmd or ['library',*sys.argv]).acquire()
    _PROCESS_LEASE=lease
    try:
        yield lease
    finally:
        lease.release()
        if _PROCESS_LEASE is lease:_PROCESS_LEASE=None


def prune_holders(directory=None):
    """Remove holder records whose process is gone (pid and start time no longer match). Returns them."""
    directory=Path(directory) if directory is not None else epoch_dir()
    pruned=[]
    for path in sorted((directory/'holders').glob('*.json')):
        info=_read_json_quiet(path)
        if isinstance(info,dict) and info.get('pid') is not None and \
                ds.pid_starttime(info['pid'])!=info.get('pid_starttime'):
            path.unlink(missing_ok=True);pruned.append(path.name)
    return pruned


def live_holders(directory=None):
    """The holder records whose process is still alive (pid and start time match)."""
    directory=Path(directory) if directory is not None else epoch_dir()
    live=[]
    for path in sorted((directory/'holders').glob('*.json')):
        info=_read_json_quiet(path)
        if isinstance(info,dict) and info.get('pid') is not None and \
                ds.pid_starttime(info['pid'])==info.get('pid_starttime'):
            live.append({**info,'record':path.name})
    return live


def _read_json_quiet(path):
    try:return json.loads(Path(path).read_text())
    except (OSError,ValueError):return None


def manifest_archives(directory=None):
    """Archived manifests under previous/, both name forms (<UTC>.json and <UTC>-<sha12>.json)."""
    directory=Path(directory) if directory is not None else SOURCE_MANIFEST.parent/'previous'
    return sorted(directory.glob('runtime_source_manifest.*.json'))


def find_manifest_archive(digest,directory=None):
    for path in manifest_archives(directory):
        if sha(path)==digest:return path
    return None


def find_epoch_commit(digest,manifest=None):
    """(repository, commit): the newest commit whose source manifest has sha256 `digest`; (repo, None) if none.

    Scans `git log -- <manifest>` of the repository that contains the manifest (all its branches).
    """
    manifest=Path(manifest) if manifest is not None else SOURCE_MANIFEST
    try:
        top=subprocess.run(['git','-C',str(manifest.parent),'rev-parse','--show-toplevel'],capture_output=True,
                           text=True,check=True).stdout.strip()
        rel=os.path.relpath(manifest.resolve(),Path(top).resolve())
        log=subprocess.run(['git','-C',top,'log','--all','--format=%H','--',rel],capture_output=True,text=True,
                           check=True)
    except (OSError,subprocess.CalledProcessError):
        return None,None
    for commit in log.stdout.split():
        blob=subprocess.run(['git','-C',top,'show',f'{commit}:{rel}'],capture_output=True)
        if blob.returncode==0 and hashlib.sha256(blob.stdout).hexdigest()==digest:
            return Path(top),commit
    return Path(top),None


def epoch_recovery_hint(fit,epoch,args):
    """How to audit a STRANDED fit under its own epoch: the exact worktree and audit commands."""
    repo,commit=find_epoch_commit(epoch)
    archive=find_manifest_archive(epoch)
    lines=[f'fit {fit} belongs to source epoch {epoch}'+(f' (archived as {archive.name})' if archive else '')]
    if commit is None:
        lines.append('no commit of runtime_source_manifest.json has that digest; inspect the fit and rerun with '
                     '--adopt-epoch to train or audit it under the live sources (journaled)')
        return '\n'.join(lines)
    tmp=f'/tmp/microglia-epoch-{epoch[:12]}'
    cmd=[sys.executable,'-B','scripts/microglia_runtime.py','train','--cache',str(Path(args.cache).resolve()),
         '--out-root',str(Path(args.out_root).resolve()),'--seed',str(args.seed),'--device',args.device]
    if args.device=='cuda':cmd+=['--gpu-mem-fraction',str(args.gpu_mem_fraction)]
    lines+=[f'audit it under that epoch from a checkout of commit {commit}:',
            f'  git -C {repo} worktree add {tmp} {commit}',
            f'  (give {tmp} its own artifacts/: see analysis/default_empf_20261117/README.md)',
            f'  cd {tmp} && '+' '.join(cmd+['--audit-under-epoch'])]
    return '\n'.join(lines)


EXIT_CODES='''exit codes:
  0   done (trained/audited, or generated and COMMITTED)
  1   error (reason printed)
  2   usage error
  3   already done: train on an AUDITED fit that verifies; generate onto a COMMITTED output without --force
      (including an interrupted PROMOTING run that this invocation rolled forward)
  4   the fit is DAMAGED (--repair) or STRANDED (--audit-under-epoch from its epoch, or --adopt-epoch);
      generate: the output's state file is damaged (generate --readopt)
  75  busy: another process holds the fit or the output, or a refreeze holds the source-epoch lease'''


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__,epilog=EXIT_CODES,formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('command',choices=('train','generate'))
    p.add_argument('--cache',type=Path,required=True)
    p.add_argument('--device',choices=('cpu','cuda'),default='cpu')
    p.add_argument('--threads',type=int,default=3)
    p.add_argument('--gpu-mem-fraction',type=float,default=.04)
    p.add_argument('--arm',default='pbulk');p.add_argument('--seed',type=int,default=0)
    p.add_argument('--updates',type=int,default=20000);p.add_argument('--out-root',type=Path)
    p.add_argument('--fit',type=Path);p.add_argument('--out',type=Path);p.add_argument('--mode')
    p.add_argument('--profile',help='application profile; inferred from --mode when omitted')
    p.add_argument('--force',action='store_true',
                   help='generate: replace an existing --out (any recorded output state) after a successful run')
    p.add_argument('--adopt-epoch',action='store_true',
                   help='train: explicitly adopt the current source manifest as the epoch of a STRANDED fit, or of a '
                        'pre-pinning fit with recovery.pt and no train_start.json (journaled in fit_state.json)')
    p.add_argument('--audit-under-epoch',action='store_true',
                   help='train: audit a STRANDED fit from a checkout whose source manifest is the fit\'s own epoch '
                        '(STRANDED -audit_under_epoch-> AUDITED); a stranded run prints the exact commands')
    p.add_argument('--dry-run',action='store_true',
                   help='train: report the fit state (an existing fit directory is adopted on first touch; a missing '
                        'one is not created); train nothing')
    p.add_argument('--repair',action='store_true',
                   help='train: move a DAMAGED fit (its files no longer support the recorded state) back to the state '
                        'its files support, journaled, and continue from there')
    p.add_argument('--readopt',action='store_true',
                   help='train: archive a damaged fit_state.json and its journal beside the fit and adopt the '
                        'directory again from its files; generate: the same for <out stem>.state.json (the '
                        'output is adopted by the legacy rule: a receipt naming the NPZ sha is COMMITTED)')
    args=p.parse_args(argv)
    if args.threads!=3 or not 0<args.gpu_mem_fraction<=.04 or args.arm!='pbulk' or args.updates!=20000 or args.seed<0:
        p.error('the default requires pbulk/20000,three threads and at most.04 GPU fraction')
    if args.command=='generate' and (args.adopt_epoch or args.dry_run or args.repair or args.audit_under_epoch):
        p.error('--adopt-epoch, --audit-under-epoch, --dry-run and --repair are for train')
    if args.audit_under_epoch and (args.adopt_epoch or args.repair or args.readopt):
        p.error('--audit-under-epoch excludes --adopt-epoch, --repair and --readopt')
    if args.command=='generate' and (args.fit is None or args.out is None):
        p.error('generate requires --fit and --out')
    if args.command=='train' and args.out_root is None:
        p.error('train requires --out-root')
    # The shared source-epoch lease is taken before the manifest is read and held until exit, so a refreeze
    # (exclusive) can never land between the digest read here and the last source check.
    global _PROCESS_LEASE
    lease=EpochLease(['microglia_runtime.py',*(sys.argv[1:] if argv is None else argv)],
                     optional=args.command=='train' and args.dry_run)
    try:
        lease.acquire()
    except ds.Busy:
        print(f'microglia runtime: the source-epoch lease {lease.directory/"lease"} is held exclusively (a refreeze '
              f'is running or pending); exit {EX_TEMPFAIL}',file=sys.stderr,flush=True)
        return EX_TEMPFAIL
    outer,_PROCESS_LEASE=_PROCESS_LEASE,lease
    try:
        return _main(p,args,lease)
    finally:
        lease.release()
        _PROCESS_LEASE=outer


def _main(p,args,lease):
    manifest_digest=sha(SOURCE_MANIFEST);manifest=check_sources()
    check_unchanged_sources(manifest,manifest_digest)
    lease.update(digest=manifest_digest)
    M=application();cfg=M.load_config()
    cache=args.cache.resolve()
    fold=json.loads((cache/'manifest.json').read_text())['inner_fold']
    cache_anchor=check_cache(cache,fold,held=args.command=='generate')
    if args.command=='train':
        fit=(args.out_root/args.arm/cfg['training']['panel']/f'fold{fold}'/f'seed{args.seed}').resolve()
        lease.update(fit=str(fit))
        return train(args,fit,M,cfg,fold,cache,cache_anchor,manifest,manifest_digest)
    fit=args.fit.resolve();out=args.out.resolve()
    lease.update(fit=str(fit),out=str(out))
    if fit==out.parent or fit in out.parents:p.error('generation output must be outside the fit')
    if out.suffix!='.npz':p.error('--out must end in .npz')
    profile,resolved,mode,inferred=resolve_profile(M,cfg,args.profile,args.mode)
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
    out.parent.mkdir(parents=True,exist_ok=True)
    lock=ds.FileLock(out,blocking=False,purpose=f'microglia generate {out.name}')
    try:
        lock.acquire()
    except ds.Busy as busy:
        owner=busy.owner or {}
        print(f'microglia runtime: output {out} is busy (owner pid {owner.get("pid")}, {owner.get("status")}); '
              f'exit {EX_TEMPFAIL}',file=sys.stderr,flush=True)
        return EX_TEMPFAIL
    try:
        return generate(args,M,fit,saved,out,cache,fold,cache_anchor,manifest,manifest_digest,
                        profile,resolved,mode,inferred)
    finally:
        lock.release()


def generate(args,M,fit,saved,out,cache,fold,cache_anchor,manifest,manifest_digest,profile,resolved,mode,inferred):
    """Generate under the output lock: open (adopt/recover), stage, post-check, promote; return the exit code."""
    sf,snap=open_output(out,readopt=args.readopt)
    if snap.state is Out.COMMITTED and not args.force:
        print(f'microglia runtime: output {out} is already COMMITTED (npz {snap.data.get("npz_sha")}); nothing to '
              f'generate without --force (exit {ALREADY_DONE})',flush=True)
        return ALREADY_DONE
    if snap.state is not Out.NONE and not args.force:
        existing=[str(x) for x in (out,receipt_path(out)) if x.exists()]
        raise FileExistsError(f'generation output exists ({snap.state.name}): {", ".join(existing)} '
                              '(use --force to replace)')
    print(f'microglia runtime: generating from fit {fit} (profile {profile}, mode {mode}) -> {out}',flush=True)
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
    run_id=new_run_id()
    pending,_=run_pending(out,run_id)
    stage_output(sf,out,sf.read(),args.force,run_id)
    try:
        _,receipt=G.generate_fit(fit,cache,mode=mode,out=pending,device=args.device)
        ds._fault('out_after_pending_npz')
        check_cache(cache,fold,held=True)
        check_unchanged_sources(manifest,manifest_digest)
        if sha(pending)!=receipt['generated_SHA256']:
            raise ValueError('generated output changed before it was receipted')
        receipt.update(generation_streams=[20261114,6,fold,'development donor index',0],
            cache_anchor=cache_anchor,output=str(out),cpu_affinity=cpus,generation_run=run_id,
            application_default=dict(name=resolved['name'],status=resolved['status'],profile=profile,
                profile_inferred_from_mode=inferred,mode=mode,
                calibration=resolved['generation']['calibration'],evidence=resolved['evidence'],
                population_qualified=False,
                config_sha256=sha(M.CONFIG),source_manifest_sha256=manifest_digest,
                native_recipe='empf',particlegan_version='0.10.0',provenance=provenance))
        # STAGED (both pending shas recorded) -> PROMOTING (journaled) -> NPZ replaced, then receipt -> COMMITTED.
        finish_output(sf,out,run_id,receipt)
    except BaseException as exc:
        output_failed(sf,out,run_id,exc)
        raise
    check_unchanged_sources(manifest,manifest_digest)
    return 0


def cli(argv=None):
    """main() with the documented exit codes (see EXIT_CODES): errors print their reason and exit 1."""
    try:
        return main(argv)
    except Stranded as exc:
        print(f'microglia runtime: Stranded: {exc}',file=sys.stderr,flush=True)
        return EX_STRANDED
    except OutputDamaged as exc:
        print(f'microglia runtime: OutputDamaged: {exc}',file=sys.stderr,flush=True)
        return EX_STRANDED
    except (ValueError,OSError,RuntimeError,KeyError,ds.DurableStateError) as exc:
        print(f'microglia runtime: {type(exc).__name__}: {exc}',file=sys.stderr,flush=True)
        return 1


if __name__=='__main__':
    raise SystemExit(cli())
