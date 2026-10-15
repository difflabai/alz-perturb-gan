"""Arm-side parent watch (v4.5): if its supervisor dies, an arm exits and releases the lock FDs it inherited.

Two independent layers:
1. arm_preexec_factory() runs in the arm group leader (`timeout`) between fork and exec. It sets
   PR_SET_PDEATHSIG(SIGTERM) and then re-checks the parent pid. GNU timeout forwards a received SIGTERM to its
   whole group, but it exits as soon as its own child exits, so its -k SIGKILL may never fire for a descendant
   that ignores SIGTERM. Layer 2 covers that case.
   The same hook unblocks the signals the supervisor blocks around spawning.
2. start() runs in the arm's inner process. It spawns a SEPARATE watcher process
   (`python -m alz_perturb_gan.parent_watch`). The watcher:
   - has its own session and process group, so it is never inside the group it signals;
   - inherits no file descriptors except stdio, so it never holds a lock FD;
   - is tied to recorded identities (pid + /proc start time) of the supervisor, the arm group leader and the
     inner process.
   It polls the supervisor identity. When that identity is gone, it re-verifies that the arm group is still the
   one it recorded, sends SIGTERM to it, waits, sends SIGKILL if members remain, and then verifies that the group
   is empty. It exits by itself when the arm group is empty, and ignores SIGTERM/SIGHUP/SIGINT.
   It never signals a group that is the supervisor's (or in the supervisor's session), and never a group whose
   identity it cannot verify. In those cases it terminates only the recorded inner process, if that identity still
   matches.
Watcher exit codes: 0 group empty, 98 refused (group not verifiably ours), 99 members survived SIGKILL.
"""
from __future__ import annotations
import argparse, ctypes, os, signal, subprocess, sys, time

PR_SET_PDEATHSIG = 1
ENV = 'ALZ_SUPERVISOR_IDENT'
MODULE = 'alz_perturb_gan.parent_watch'
_libc = None


def _load_libc():
    global _libc
    if _libc is None: _libc = ctypes.CDLL(None, use_errno=True)
    return _libc


def _stat(pid):
    try:
        f = open(f'/proc/{int(pid)}/stat').read().rsplit(')', 1)[1].split()
        return {'state': f[0], 'pgid': int(f[2]), 'sid': int(f[3]), 'starttime': int(f[19])}
    except (OSError, IndexError, ValueError):
        return None


def starttime(pid):
    """/proc start time of a live (non-zombie) pid, else None."""
    s = _stat(pid)
    return None if s is None or s['state'] in ('Z', 'X') else s['starttime']


def identity_env(pid=None):
    pid = os.getpid() if pid is None else pid
    return f'{pid}:{starttime(pid)}:{os.getpgid(pid)}:{os.getsid(pid)}'


def group_members(pgid):
    out = []
    for d in os.listdir('/proc'):
        if d.isdigit():
            s = _stat(int(d))
            if s and s['pgid'] == pgid and s['state'] not in ('Z', 'X'): out.append((int(d), s['starttime']))
    return out


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


def _log(msg):
    try: print(f'parent_watch[{os.getpid()}]: {msg}', file=sys.stderr, flush=True)
    except OSError: pass


def _wait_empty(pgid, timeout):
    end = time.monotonic() + timeout
    while True:
        m = group_members(pgid)
        if not m or time.monotonic() >= end: return m
        time.sleep(0.05)


def _term_inner_only(inner):
    """Signal only the recorded inner process, re-verifying its identity before each signal."""
    for sig in (signal.SIGTERM, signal.SIGKILL):
        if starttime(inner[0]) != inner[1]: return
        try: os.kill(inner[0], sig)
        except ProcessLookupError: return
        end = time.monotonic() + 2
        while time.monotonic() < end and starttime(inner[0]) == inner[1]: time.sleep(0.05)


def cleanup_group(arm_pgid, leader, inner, sup, *, grace=5.0, known=()):
    """TERM, wait, KILL, verify-empty for the arm group, but only if it is verifiably the recorded arm group.

    leader/inner: (pid, starttime) recorded at arm start; sup: (pid, starttime, pgid, sid); known: member identities
    observed by watch() while the group stayed continuously non-empty (so the pgid cannot have been reused)."""
    members = group_members(arm_pgid)
    if not members:
        _log(f'arm group {arm_pgid} already empty'); return 0
    me = _stat(os.getpid())
    sids = {(_stat(pid) or {}).get('sid') for pid, _ in members}
    if arm_pgid in (sup[0], sup[2], sup[3]) or sup[3] in sids or arm_pgid in (me['pgid'], me['sid']):
        _log(f'refusing to signal group {arm_pgid}: it is not the arm\'s own session; terminating the recorded inner process only')
        _term_inner_only(inner); return 98
    holder = starttime(arm_pgid)
    if holder is not None and holder != leader[1]:
        _log(f'refusing to signal group {arm_pgid}: leader identity changed (start {holder} != {leader[1]})'); return 98
    ids = {tuple(leader), tuple(inner), *map(tuple, known)}
    if not any((pid, st) in ids for pid, st in members):
        _log(f'refusing to signal group {arm_pgid}: no member matches a recorded identity'); return 98
    # From here on the group has members, so its pgid cannot have been reused; it is the group we recorded.
    os.killpg(arm_pgid, signal.SIGTERM); _log(f'sent SIGTERM to arm group {arm_pgid} ({len(members)} members)')
    left = _wait_empty(arm_pgid, grace)
    if left:
        try: os.killpg(arm_pgid, signal.SIGKILL)
        except ProcessLookupError: left = []
        else: _log(f'sent SIGKILL to arm group {arm_pgid} ({len(left)} members ignored SIGTERM)')
        left = _wait_empty(arm_pgid, grace)
    if left:
        _log(f'arm group {arm_pgid} NOT EMPTY after SIGKILL: {left}'); return 99
    _log(f'arm group {arm_pgid} verified empty'); return 0


def watch(sup, arm_pgid, leader, inner, *, interval, grace):
    known = {tuple(leader), tuple(inner)}
    while True:
        m = group_members(arm_pgid)
        if not m: return 0                                            # arm finished normally
        if starttime(sup[0]) != sup[1]:
            _log(f'supervisor {sup[0]} (start {sup[1]}) gone; cleaning up arm group {arm_pgid}')
            return cleanup_group(arm_pgid, leader, inner, sup, grace=grace, known=known)
        # Continuity: a member we already know is still here, so the group never emptied and its pgid was not
        # reused; the other current members (e.g. descendants the arm started) belong to the same group.
        if any(x in known for x in m): known.update(m)
        time.sleep(interval)


def start(env=None, *, interval=1.0, grace=5.0):
    """Spawn the separate watcher if ALZ_SUPERVISOR_IDENT is set; returns its Popen (or None)."""
    env = os.environ if env is None else env
    v = env.get(ENV)
    if not v: return None
    pg = os.getpgrp(); me = os.getpid()
    cmd = [sys.executable, '-m', MODULE, '--supervisor', v, '--arm-pgid', str(pg), '--arm-leader', f'{pg}:{starttime(pg)}',
           '--arm-inner', f'{me}:{starttime(me)}', '--interval', str(interval), '--grace', str(grace)]
    w = subprocess.Popen(cmd, start_new_session=True, close_fds=True, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                         env=dict(env))
    _log(f'watcher pid={w.pid} for arm group {pg} (own session; no inherited FDs)')
    return w


def _pair(s):
    a, b = s.split(':'); return int(a), (None if b == 'None' else int(b))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--supervisor', required=True); ap.add_argument('--arm-pgid', type=int, required=True)
    ap.add_argument('--arm-leader', required=True); ap.add_argument('--arm-inner', required=True)
    ap.add_argument('--interval', type=float, default=1.0); ap.add_argument('--grace', type=float, default=5.0)
    a = ap.parse_args(argv)
    # The watcher is never in the arm group; it also ignores TERM/HUP/INT so that a stop signal sent to everything
    # (e.g. systemctl stop) cannot remove it before it has verified the arm group is empty. It exits by itself.
    for s in (signal.SIGTERM, signal.SIGHUP, signal.SIGINT): signal.signal(s, signal.SIG_IGN)
    sup = tuple(int(x) for x in a.supervisor.split(':'))
    return watch(sup, a.arm_pgid, _pair(a.arm_leader), _pair(a.arm_inner), interval=a.interval, grace=a.grace)


if __name__ == '__main__':
    sys.exit(main())
