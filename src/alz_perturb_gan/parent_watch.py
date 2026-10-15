"""Arm-side parent watch (v4.4): an arm exits, releasing the lock FDs it inherited, if its supervisor dies.

Two independent layers:
1. arm_preexec_factory(): installed in the arm group leader (`timeout`) between fork and exec. It sets
   PR_SET_PDEATHSIG(SIGTERM), then re-checks the parent pid, so a supervisor that died before prctl took effect
   is also caught. GNU timeout forwards a received SIGTERM to its whole process group, then sends SIGKILL after -k.
   The same hook unblocks the signals the supervisor blocks around spawning, so the arm starts with a clean mask.
2. start(): a daemon thread in the arm's inner process polls the supervisor's RECORDED identity
   (pid + /proc start time, from ALZ_SUPERVISOR_IDENT="pid:starttime:pgid:sid"). When that identity is gone, it sends
   SIGTERM, then SIGKILL, to the arm's OWN process group only. It does this only if the process is in its own arm
   session (sid == pgid, as start_new_session gives) and neither that group nor that session is the supervisor's;
   otherwise it exits by itself and signals nothing.
"""
from __future__ import annotations
import ctypes, os, signal, sys, threading, time

PR_SET_PDEATHSIG = 1
ENV = 'ALZ_SUPERVISOR_IDENT'
_libc = None


def _load_libc():
    global _libc
    if _libc is None: _libc = ctypes.CDLL(None, use_errno=True)
    return _libc


def starttime(pid):
    """/proc start time of a live (non-zombie) pid, else None."""
    try:
        f = open(f'/proc/{int(pid)}/stat').read().rsplit(')', 1)[1].split()
        return None if f[0] in ('Z', 'X') else int(f[19])
    except (OSError, IndexError, ValueError):
        return None


def identity_env(pid=None):
    pid = os.getpid() if pid is None else pid
    return f'{pid}:{starttime(pid)}:{os.getpgid(pid)}:{os.getsid(pid)}'


def arm_preexec_factory(parent_pid, *, pdeathsig=True, unblock=()):
    """Return a preexec_fn for the arm leader. Everything is resolved before fork; the child only makes syscalls."""
    libc = _load_libc(); sig = ctypes.c_ulong(int(signal.SIGTERM)); zero = ctypes.c_ulong(0)
    opt = ctypes.c_int(PR_SET_PDEATHSIG); unblock = frozenset(unblock)

    def preexec():
        if pdeathsig:
            if libc.prctl(opt, sig, zero, zero, zero) != 0: os._exit(96)
            if os.getppid() != parent_pid: os._exit(97)      # supervisor already gone before prctl took effect
        if unblock: signal.pthread_sigmask(signal.SIG_UNBLOCK, unblock)
    return preexec


def _terminate_own_group(reason, sup_pid, sup_pgid, sup_sid, grace):
    pg, sid = os.getpgrp(), os.getsid(0)
    if sid != pg or pg in (sup_pgid, sup_pid, sup_sid) or sid == sup_sid:
        print(f'parent_watch: {reason}; not in an own arm session (sid={os.getsid(0)} pgid={pg}); exiting alone',
              file=sys.stderr, flush=True)
        os._exit(98)
    print(f'parent_watch: {reason}; terminating arm group {pg}', file=sys.stderr, flush=True)
    os.killpg(pg, signal.SIGTERM)          # reaches this process too
    time.sleep(grace)
    os.killpg(pg, signal.SIGKILL)


def start(env=None, *, interval=1.0, grace=5.0):
    """Start the parent-identity poll if ALZ_SUPERVISOR_IDENT is set; returns the thread (or None)."""
    env = os.environ if env is None else env
    v = env.get(ENV)
    if not v: return None
    sup_pid, sup_start, sup_pgid, sup_sid = (int(x) for x in v.split(':'))
    if starttime(sup_pid) != sup_start:
        _terminate_own_group(f'supervisor {sup_pid} gone at arm start', sup_pid, sup_pgid, sup_sid, grace)

    def loop():
        while True:
            time.sleep(interval)
            if starttime(sup_pid) != sup_start:
                _terminate_own_group(f'supervisor {sup_pid} (start {sup_start}) gone', sup_pid, sup_pgid, sup_sid, grace); return
    t = threading.Thread(target=loop, name='parent_watch', daemon=True); t.start()
    return t
