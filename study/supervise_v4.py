#!/usr/bin/env python3
"""Supervisor v4.6: no aggregate time ceiling (Mikkel, 2026-10-14); per-arm timeouts, cancellation, cleanup.

Time: there is NO fixed cohort budget. An aggregate deadline exists only if the operator passes
--total-timeout and/or --lease-valid-until (real runs require a lease; the lease is an ownership safeguard,
not a compute budget). Each arm has a configurable --arm-timeout (default 600 s) enforced by the supervisor
and an independent `timeout -k` wrapper. Teardown has its own --teardown-timeout budget, so cleanup and the
receipt still happen when the aggregate deadline has already passed.
Identity: supervisor, each arm's group leader (the `timeout` wrapper) and the inner harness process are
recorded as (pid, /proc start time, uid). Cleanup signals a group only after re-verifying the leader identity
(pid == pgid with the recorded start time) or, if the leader has exited, that no different process now owns
that pid (the kernel cannot reuse a pid while its process group has members). Groups we did not create are
never signalled.
Locks: existing files only (no O_CREAT), dev+inode verified, LOCK_EX|LOCK_NB, never LOCK_UN; released at exit.
Post-run probe (--post-run-probe): separate process with its own bounded --probe-timeout; verifies the
receipt sha256; per lock: open without O_CREAT, fstat dev/inode, flock NB then close (no LOCK_UN); a busy
lock is labelled ours / other_owner / unknown_owner from /proc/locks against the RECORDED identities;
nvidia-smi bounded; owned identities gone; llama-server identities unchanged; post_run_receipt.json created
exclusively. Exit codes: 0 ok, 2 usage, 3 stopped, 5 incomplete, 6 probe attention, 7 descendants survived,
8 probe: run incomplete or unknown (never clean).
--dry-run: CPU only, test locks in --dry-dir (default /ml2/altgan/.cache/supervisor-dryrun), fake arms.
v4.4 hangup/parent death: SIGHUP is handled exactly like SIGTERM (identity-based teardown, receipt still written).
TERM/INT/HUP/ALRM are blocked from cohort creation until the try block and around each spawn, so a signal can
neither skip the receipt nor leave a spawned arm unregistered. Each arm leader gets PR_SET_PDEATHSIG(SIGTERM)
with a parent re-check, and the arm's inner process polls the supervisor's recorded identity (ALZ_SUPERVISOR_IDENT).
If the supervisor dies, the arm group exits, which releases the lock FDs it inherited.
v4.5: the arm-side watcher is a separate process in its own session (never inside the group it signals): TERM,
wait, KILL, verify empty. Completeness is labelled everywhere: a receipt is "complete" only for rc 0; any
stop, interruption, external SIGKILL (manual or kernel OOM kill; cause recorded as unknown unless the system
oom_kill counter moved) or survivors is "incomplete". The post-run probe reports residue separately and its
verdict is "clean" ONLY for a complete run with clean residue (exit 0); "incomplete" exits 8; "attention" 6.
Without a supervisor receipt (supervisor SIGKILLed / OOM-killed), --post-run-probe --cohort DIR probes from the
lease receipt and can only answer "incomplete" (lease known) or "unknown" (no lease).
v4.6: --stop-requested REASON (passed by the chain wrapper when IT received SIGTERM, e.g. systemctl stop) makes the
probe itself label the run incomplete (exit 8, or 6 for residue), even if the supervisor finished with rc 0.
"""
from __future__ import annotations
import argparse, datetime as dt, fcntl, hashlib, json, math, os, signal, subprocess, sys, time
from pathlib import Path
from zoneinfo import ZoneInfo

REPO = Path(os.environ.get("ALZ_HARNESS_REPO", "/ml2/altgan/gene-feature-harness-20261014")).resolve(strict=True)
sys.path.insert(0, str(REPO / "src"))
from alz_perturb_gan.paths import scoped, create_new  # noqa: E402
from alz_perturb_gan import parent_watch  # noqa: E402

DEN = ZoneInfo("America/Denver")
STEPS = 2000
TRAINER = "bcap"
PARTICLEGAN_COMMIT = "6a91a49910568720b2fdfd040ee5891c7d2bde46"
MIN_START_MIB, MIN_RUN_MIB = 9 * 1024, 4 * 1024
POLL_S, MIN_ARM_S, KILL_RESERVE, KILL_GRACE, PROBE_MAX = 5, 120, 30, 10, 5
GPU_UUID = "GPU-cb4ce47d-d968-bffd-5646-e830a9fa1c69"
REAL_LOCKS = ["/tmp/ember-pg-20261012-slots/gpu1-slot0.lock", "/tmp/ember-pg-20261012-slots/gpu1-slot1.lock"]
ROOT = Path("/ml2/altgan/studies/gene-feature-20261014")
PY, CACHE = os.environ.get("ALZ_PYTHON", "/ml2/altgan/.venv/bin/python"), Path("/ml2/altgan/.cache")
DEFAULT_DRY_DIR = CACHE / "supervisor-dryrun"
F = Path("/ml2/altgan/features")
FEATURES = {
    "feature_manifest": (F / "manifest.json", "d6d501f7b5ca7af2920502537582d0d6401359b5ae2cb6f12271891c8895d844"),
    "string_matrix": (F / "string_spectral.npy", "6a428429b6556f30bc690d252cb6d7910665cb52755d76a9d0795cfcf15c90f5"),
    "string_row_mapping": (F / "addons/string_row_mapping.json", "044b2756e940d31519c85d81073a3dc17ac7f3115a8fc4e04b6a539f4b994a74"),
    "string_license_receipt": (F / "addons/string_license_receipt.json", "1a7666849ddabcfedf7547b8eac844514eb8038be12a591743221b6e3711da86"),
}
ALLOC_CAP_GIB = 4
ARMS = ["B0", "G", "H", "GH", "shuffle", "G_frozen"]
BLOCK = frozenset({signal.SIGTERM, signal.SIGINT, signal.SIGHUP, signal.SIGALRM})
DRY_ARM = ("import os, sys, time\n"
           "if sys.argv[5] == '1':\n"
           "    from alz_perturb_gan import parent_watch; parent_watch.start(interval=0.2, grace=1.0)\n"
           "if sys.argv[6] == '1':\n"
           "    import subprocess\n"
           "    fds = tuple(int(x) for x in os.environ.get('ALZ_LEASE_FDS', '').split(',') if x)\n"
           "    c = subprocess.Popen([sys.executable, '-c', 'import signal, sys, time; signal.signal(signal.SIGTERM, signal.SIG_IGN); '\n"
           "                          'print(\"ignoring SIGTERM\", flush=True); time.sleep(600)'], pass_fds=fds, stdout=subprocess.PIPE, text=True)\n"
           "    print('term-ignoring child', c.pid, c.stdout.readline().strip(), flush=True)\n"
           "time.sleep(float(sys.argv[3]))\n"
           "print('fake', sys.argv[1], 'seed', sys.argv[2], 'fds=' + os.environ.get('ALZ_LEASE_FDS', ''), flush=True)\n"
           "sys.exit(int(sys.argv[4]))\n")


class Stop(Exception): pass


def sha(p):
    h = hashlib.sha256()
    with open(scoped(p), "rb") as f:
        for c in iter(lambda: f.read(8 << 20), b""): h.update(c)
    return h.hexdigest()


def jbytes(o): return (json.dumps(o, indent=2, sort_keys=True) + "\n").encode()


def proc_info(pid):
    try:
        st = open(f"/proc/{pid}/stat").read()
        uid = next(l.split()[1] for l in open(f"/proc/{pid}/status") if l.startswith("Uid:"))
        f = st.rsplit(")", 1)[1].split()
        return {"pid": pid, "comm": st[st.index("(") + 1:st.rindex(")")], "uid": int(uid), "starttime": int(f[19]),
                "pgid": int(f[2]), "state": f[0]}
    except (OSError, ValueError, StopIteration, IndexError):
        return {"pid": pid, "comm": None, "uid": None, "starttime": None, "pgid": None, "state": None}


def ident(i): return {k: i.get(k) for k in ("pid", "starttime", "uid", "comm")}


def alive_same(rec):
    """True iff the process recorded as (pid, starttime) is still alive (not a zombie) with that identity."""
    if not rec or rec.get("pid") is None: return False
    now = proc_info(rec["pid"])
    return now["starttime"] is not None and now["starttime"] == rec["starttime"] and now["state"] != "Z"


def group_members(pgid):
    out = []
    for d in os.listdir("/proc"):
        if d.isdigit():
            i = proc_info(int(d))
            if i["pgid"] == pgid and i["state"] not in (None, "Z"): out.append(i)
    return out


def group_ownership(pgid, leader, known=()):
    """('ours'|'empty'|'reused'|'unverified', members).

    'ours' ONLY if at least one live member of the group is a process whose recorded identity (pid AND
    /proc start time) we captured ourselves: the group leader or the inner harness process. A group whose
    pid==pgid process has a different start time is 'reused'. Members exist but none is a recorded
    identity -> 'unverified' (never signalled)."""
    members = group_members(pgid)
    if not members: return "empty", []
    ids = [r for r in (leader, *known) if r and r.get("pid") is not None and r.get("starttime") is not None]
    holder = proc_info(pgid)
    if holder["starttime"] is not None and holder["state"] != "Z" and holder["starttime"] != leader.get("starttime"):
        return "reused", members
    verified = [m for m in members if any(m["pid"] == r["pid"] and m["starttime"] == r["starttime"] for r in ids)]
    return ("ours" if verified else "unverified"), members


def children_of(ppid):
    out = []
    for d in os.listdir("/proc"):
        if not d.isdigit(): continue
        try:
            f = open(f"/proc/{d}/stat").read().rsplit(")", 1)[1].split()
            if int(f[1]) == ppid: out.append(int(d))
        except (OSError, IndexError, ValueError): pass
    return sorted(out)


def inner_process(wrapper_pid, until):
    while True:
        kids = children_of(wrapper_pid)
        if kids:
            info = proc_info(kids[0])
            if info["starttime"] is not None: return dict(ident(info), ppid=wrapper_pid, all_children=kids)
        if time.monotonic() >= until: return {"pid": None, "ppid": wrapper_pid, "reason": "no child seen before poll limit"}
        time.sleep(0.02)


def llama_servers():
    return sorted((ident(i) for i in (proc_info(int(d)) for d in os.listdir("/proc") if d.isdigit())
                   if i["comm"] and i["comm"].startswith("llama-server")), key=lambda x: x["pid"])


def nvidia_smi(args, timeout):
    if timeout <= 0: return None, "no time left in probe budget"
    try:
        r = subprocess.run(["/usr/bin/nvidia-smi", *args, "--format=csv,noheader,nounits"], capture_output=True, text=True,
                           timeout=max(0.5, timeout), check=True)
        return [[x.strip() for x in l.split(",")] for l in r.stdout.strip().splitlines() if l.strip()], None
    except Exception as e:  # noqa: BLE001
        return None, repr(e)


def lock_key(st): return f"{os.major(st.st_dev):02x}:{os.minor(st.st_dev):02x}:{st.st_ino}"


def proc_lock_holders(st):
    key, out = lock_key(st), []
    try:
        for line in open("/proc/locks"):
            f = line.split()
            if len(f) >= 6 and f[5] == key and not f[1].startswith("->"):
                out.append({"type": f[1], "mode": f[3], "pid": int(f[4])})
    except OSError as e:
        return None, repr(e)
    return out, None


def backstop_schedule(slot, scale):
    """The independent `timeout -k K D` schedule used by arm_cmd, in real seconds after spawn."""
    grace = KILL_GRACE * scale
    hard = max(1, int(slot - grace)); k = max(1, int(grace / 2)); d = max(1, hard - int(grace / 2))
    return {"term_after_s": d, "kill_after_s": d + k}


BACKSTOP_TOLERANCE_S = 0.5   # process start/exit latency around the backstop's own clock


def classify_exit(returncode, *, elapsed, schedule):
    """rc 124 (TERM by timeout) / 137 (KILL by timeout -k) count as a timeout only at or after the scheduled backstop.
    v4.5: death by a signal we did not send (negative rc; or 137 before the backstop) is "killed_<sig>"; for SIGKILL
    that covers both a manual kill and a kernel OOM kill (the cause is recorded separately, never assumed)."""
    if returncode == 0: return "ok"
    if returncode == 124 and elapsed >= schedule["term_after_s"] - BACKSTOP_TOLERANCE_S: return "timeout"
    if returncode == 137 and elapsed >= schedule["kill_after_s"] - BACKSTOP_TOLERANCE_S: return "timeout"
    if returncode == 137: return "killed_sigkill"
    if returncode is not None and returncode < 0:
        try: return "killed_" + signal.Signals(-returncode).name.lower()
        except ValueError: return f"killed_sig{-returncode}"
    return f"error_rc{returncode}"


def oom_kill_count():
    """System-wide kernel OOM-kill counter (/proc/vmstat oom_kill), or None if unavailable."""
    try:
        for line in open("/proc/vmstat"):
            if line.startswith("oom_kill "): return int(line.split()[1])
    except (OSError, ValueError): pass
    return None


def watchers_for(pgids):
    """Live arm watchers (python -m alz_perturb_gan.parent_watch --arm-pgid N) of ours for the given arm groups."""
    out, pgids = [], set(pgids)
    for d in os.listdir("/proc"):
        if not d.isdigit(): continue
        try: argv = open(f"/proc/{d}/cmdline", "rb").read().split(b"\0")
        except OSError: continue
        if b"alz_perturb_gan.parent_watch" not in argv or b"--arm-pgid" not in argv: continue
        try: pg = int(argv[argv.index(b"--arm-pgid") + 1])
        except (IndexError, ValueError): continue
        i = proc_info(int(d))
        if pg in pgids and i["uid"] == os.getuid() and i["state"] not in (None, "Z"): out.append(dict(ident(i), arm_pgid=pg))
    return out


class Clock:
    def __init__(self, scale): self.scale, self.t0 = scale, time.monotonic()
    def wall(self): return dt.datetime.now(DEN)
    def s(self, x): return x * self.scale


class Supervisor:
    def __init__(self, a):
        self.a, self.dry = a, a.dry_run
        self.fds, self.groups, self.signals_sent, self.refused = [], {}, [], []
        self.cleaning, self.current = False, None
        self.clock = Clock(a.time_scale if self.dry else 1.0)
        self.dry_dir = Path(a.dry_dir) if a.dry_dir else DEFAULT_DRY_DIR

    def on_signal(self, signum, _f):
        if self.cleaning: return
        raise Stop({signal.SIGTERM: "sigterm", signal.SIGINT: "sigint", signal.SIGHUP: "sighup", signal.SIGALRM: "outer_watchdog"}[signum])

    def left(self): return math.inf if self.deadline is None else self.deadline - time.monotonic()

    def free_mib(self):
        if self.dry: return 20000
        rows, err = nvidia_smi(["--query-gpu=uuid,memory.free"], min(self.clock.s(PROBE_MAX), self.left()))
        if rows is None: raise Stop(f"nvidia_smi_failed:{err}")
        for u, f in rows:
            if u == GPU_UUID: return int(f)
        raise Stop("gpu_not_visible")

    def kill_group(self, pgid, until):
        """Identity-checked TERM then KILL of a group we created; every wait bounded by `until` (monotonic)."""
        g = self.groups.get(pgid)
        if g is None: self.refused.append({"pgid": pgid, "reason": "not_created_by_us"}); return False
        grace = max(0.2, min(self.clock.s(KILL_GRACE) / 2, (until - time.monotonic()) / 2))
        known = [g.get("inner")]
        for sig in (signal.SIGTERM, signal.SIGKILL):
            state, members = group_ownership(pgid, g["leader"], known)
            if state == "empty": return True
            if sig == signal.SIGKILL and state == "unverified" and any(
                    (m["pid"], m["starttime"]) in snap for m in members):
                # v4.5: a member seen before our SIGTERM is still in the group, so it never emptied and the pgid was
                # not reused: the remaining members (e.g. a descendant that ignores SIGTERM) are this arm's.
                state = "ours"; known = known + [ident(m) for m in members if (m["pid"], m["starttime"]) in snap]
            if sig == signal.SIGTERM: snap = {(m["pid"], m["starttime"]) for m in members}
            if state in ("reused", "unverified"):
                self.refused.append({"pgid": pgid, "signal_withheld": signal.Signals(sig).name,
                                     "reason": "identity_mismatch" if state == "reused" else "unverified_identity",
                                     "members": [ident(m) for m in members]}); return False
            self.signals_sent.append({"pgid": pgid, "signal": signal.Signals(sig).name, "leader": g["leader"],
                                      "members": [ident(m) for m in members], "at": self.clock.wall().isoformat()})
            try: os.killpg(pgid, sig)
            except ProcessLookupError: return True
            end = time.monotonic() + grace
            while time.monotonic() < end:
                if group_ownership(pgid, g["leader"], known)[0] == "empty": return True
                time.sleep(0.05)
        return group_ownership(pgid, g["leader"], known)[0] == "empty"

    def static_checks(self):
        out = {"script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}  # own file, read as executed
        if self.dry and self.a.dry_skip_static_checks:
            out.update(harness_commit="DRY-SKIPPED", bundle="DRY-SKIPPED", **{k: "DRY-SKIPPED" for k in FEATURES}); self.bundle = None
            return out
        head = subprocess.run(["git", "-C", str(REPO), "rev-parse", "HEAD"], capture_output=True, text=True, timeout=10).stdout.strip()
        dirty = subprocess.run(["git", "-C", str(REPO), "status", "--porcelain", "--untracked-files=no"], capture_output=True, text=True, timeout=10).stdout.strip()
        if head != self.a.harness_commit or dirty: raise Stop(f"harness_commit_mismatch:{head}:{'dirty' if dirty else 'clean'}")
        out["harness_commit"] = head
        self.bundle = scoped(self.a.bundle)
        if sha(self.bundle) != self.a.bundle_sha256: raise Stop("hash_mismatch:bundle")
        out["bundle"] = self.a.bundle_sha256
        for k, (p, want) in FEATURES.items():
            if sha(p) != want: raise Stop(f"hash_mismatch:{k}")
            out[k] = want
        return out

    def lock_paths(self):
        if not self.dry: return REAL_LOCKS
        ps = [self.dry_dir / "gpu1-slot0.test.lock", self.dry_dir / "gpu1-slot1.test.lock"]
        for p in ps: p.touch()   # test setup creates the test locks; acquisition never creates
        return [str(scoped(p)) for p in ps]

    def acquire(self, paths):
        info = []
        for p in paths:
            try: fd = os.open(p, os.O_RDWR | os.O_NOFOLLOW)
            except FileNotFoundError: raise Stop(f"lock_missing:{p}")
            st, st2 = os.fstat(fd), os.stat(p, follow_symlinks=False)
            if (st.st_ino, st.st_dev) != (st2.st_ino, st2.st_dev): raise Stop(f"lock_inode_changed:{p}")
            try: fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError: raise Stop(f"lock_busy:{p}")   # non-preemption: never wait for or break another holder
            os.set_inheritable(fd, True); self.fds.append(fd)
            info.append({"path": p, "inode": st.st_ino, "device": st.st_dev, "proc_locks_key": lock_key(st), "fd": fd})
        for i in info:
            st = os.stat(i["path"], follow_symlinks=False)
            if (st.st_ino, st.st_dev) != (i["inode"], i["device"]): raise Stop(f"lock_inode_changed:{i['path']}")
        return info

    def write_grant(self, run, cohort, seed, statics, lease_path, lease_sha, cap_sha):
        valid_until = self.lease_until.astimezone(dt.timezone.utc).isoformat() if self.lease_until else None
        if not self.dry and not valid_until: raise Stop("missing_lease_valid_until")
        grant = {"approved": True, "authorization_reference": self.a.authorization_reference or "DRY-RUN",
                 "phases": ["train", "evaluate"], "arms": ARMS, "seeds": [seed], "protocol_sha256": self.a.protocol_sha256,
                 "bundle_sha256": statics["bundle"], **{f"{k}_sha256": statics[k] for k in FEATURES},
                 "harness_commit": statics["harness_commit"], "supervisor_sha256": statics["script_sha256"],
                 "cuda_allocator_cap_gib": ALLOC_CAP_GIB, "max_steps_per_arm": STEPS, "max_wall_seconds_per_arm": self.a.arm_timeout,
                 "particle_trainer": TRAINER, "particlegan_commit": PARTICLEGAN_COMMIT,
                 "aggregate_time_cap": None, "total_timeout_s": self.a.total_timeout,
                 "gpu_uuid": GPU_UUID, "lease_receipt": str(lease_path), "lease_receipt_sha256": lease_sha,
                 "capacity_receipt_sha256": cap_sha, "capacity_valid_until": valid_until if valid_until else "DRY-RUN: no lease",
                 "ownership_reference": self.a.ownership_reference or "DRY-RUN", "cohort": str(cohort)}
        p = create_new(run, "grant.json", jbytes(grant))
        return p, sha(p)

    def arm_cmd(self, arm, seed, run, slot, grant, gsha):
        sch = backstop_schedule(slot, self.clock.scale)
        wd = ["timeout", "-k", str(sch["kill_after_s"] - sch["term_after_s"]), str(sch["term_after_s"])]
        if self.dry:
            rc = 7 if arm == self.a.dry_fail_arm else 0
            poll = "0" if self.a.dry_no_parent_poll else "1"
            ign = "1" if self.a.dry_arm_term_ignoring_child else "0"
            return wd + [PY, "-c", DRY_ARM, arm, str(seed), str(self.a.dry_arm_seconds), str(rc), poll, ign]
        mw = int(slot - self.clock.s(KILL_RESERVE))
        cmd = [PY, "-u", "-m", "alz_perturb_gan.gene_harness", "train", "--bundle", str(self.bundle), "--bundle-sha256", self.a.bundle_sha256,
               "--output", str(run / arm), "--trainer", TRAINER, "--arm", arm, "--seed", str(seed), "--steps", str(STEPS), "--max-wall-seconds", str(mw),
               "--device", "cuda:0", "--cuda-allocator-cap-gib", str(ALLOC_CAP_GIB), "--grant", str(grant), "--grant-sha256", gsha,
               "--feature-manifest", str(FEATURES["feature_manifest"][0]), "--feature-manifest-sha256", FEATURES["feature_manifest"][1],
               "--tau-source", "string_spectral",
               "--string-row-mapping", str(FEATURES["string_row_mapping"][0]), "--string-row-mapping-sha256", FEATURES["string_row_mapping"][1],
               "--string-license-receipt", str(FEATURES["string_license_receipt"][0]), "--string-license-receipt-sha256", FEATURES["string_license_receipt"][1]]
        if arm == "G_frozen":
            ck = run / "G" / "checkpoint.pt"
            cmd += ["--frozen-source", str(ck), "--frozen-source-sha256", sha(ck), "--frozen-generator-arm", "G"]
        return wd + cmd

    def run_arm(self, arm, seed, run, entry, grant, gsha, env):
        t_start = time.monotonic()
        slot = min(self.clock.s(self.a.arm_timeout), self.left())
        entry["slot_s"] = round(slot / self.clock.scale, 1)
        if self.deadline is not None and slot < self.clock.s(min(MIN_ARM_S, self.a.arm_timeout)):
            entry["status"] = "skipped_operator_deadline"; return False
        free = self.free_mib(); entry["free_mib_at_spawn"] = free
        if free < MIN_RUN_MIB: entry["status"] = "not_started_low_memory"; raise Stop("capacity")
        cmd = self.arm_cmd(arm, seed, run, slot, grant, gsha)
        entry.update(command=cmd, started=self.clock.wall().isoformat(), oom_kill_at_spawn=oom_kill_count())
        dfd = os.open(scoped(run), os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
        try: lfd = os.open(f"{arm}.log", os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o644, dir_fd=dfd)
        finally: os.close(dfd)
        self.current = (entry, None)
        pre = parent_watch.arm_preexec_factory(os.getpid(), pdeathsig=not (self.dry and self.a.dry_no_pdeathsig), unblock=BLOCK)
        old_mask = signal.pthread_sigmask(signal.SIG_BLOCK, BLOCK)   # spawn + registration cannot be split by a signal
        try:
            p = subprocess.Popen(cmd, cwd=str(REPO), env=env, stdout=lfd, stderr=subprocess.STDOUT, start_new_session=True,
                                 pass_fds=tuple(self.fds), preexec_fn=pre)
            os.close(lfd)
            pgid = p.pid
            leader = ident(proc_info(p.pid))
            self.groups[pgid] = {"arm": arm, "seed": seed, "leader": leader, "popen": p}
        finally:
            signal.pthread_sigmask(signal.SIG_SETMASK, old_mask)
        entry["pgid"], entry["leader"] = pgid, leader
        entry["inner"] = inner_process(p.pid, time.monotonic() + min(2.0, max(0.1, slot / 2)))  # real seconds, not scaled
        self.groups[pgid]["inner"] = entry["inner"] if entry["inner"].get("pid") else None
        self.current = (entry, pgid)
        kill_at = t_start + slot - self.clock.s(KILL_GRACE)
        reason, minfree = None, None
        while p.poll() is None:
            now = time.monotonic()
            if now >= kill_at: reason = "timeout"; break
            free = self.free_mib(); minfree = free if minfree is None else min(minfree, free)
            if free < MIN_RUN_MIB: reason = "low_memory_killed"; break
            try: p.wait(timeout=max(0.02, min(self.clock.s(POLL_S), kill_at - time.monotonic())))
            except subprocess.TimeoutExpired: pass
        entry["min_free_mib"] = minfree
        entry["backstop"] = sch = backstop_schedule(slot, self.clock.scale)
        if reason is None:
            reason = classify_exit(p.returncode, elapsed=time.monotonic() - t_start, schedule=sch)
            if reason == "timeout": entry["killed_by"] = "timeout_wrapper"   # the independent backstop fired first
        entry["status"] = reason
        if reason.startswith("killed_"):
            oom1 = oom_kill_count(); entry["oom_kill_counter"] = {"at_spawn": entry.get("oom_kill_at_spawn"), "at_exit": oom1}
            moved = None if oom1 is None or entry.get("oom_kill_at_spawn") is None else oom1 > entry["oom_kill_at_spawn"]
            entry["kill_cause"] = ("unknown: external signal" if reason != "killed_sigkill" else
                                   "unknown: external SIGKILL or kernel OOM kill" + ("" if moved is None else
                                   "; system oom_kill counter " + ("increased (OOM kill likely)" if moved else "unchanged")))
        entry["group_killed_clean"] = self.kill_group(pgid, t_start + slot)
        try: p.wait(timeout=0.5)
        except subprocess.TimeoutExpired: pass
        self.current = None
        end = time.monotonic()
        entry.update(returncode=p.poll(), ended=self.clock.wall().isoformat(), elapsed_s=round((end - t_start) / self.clock.scale, 1))
        if end > t_start + slot or not entry["group_killed_clean"]:
            entry["status"] = "incomplete_overrun"; entry["overrun_s"] = round((end - t_start - slot) / self.clock.scale, 1)
        try:
            s = json.loads((run / arm / "evaluation" / "metrics" / "summary.json").read_text())
            cal = s["calibration"]["calibrator"]
            entry.update(calibrator_sha256=cal.get("sha256"), calibration_split_sha256=cal.get("calibration_split_sha256"),
                         calibration_status=cal.get("status"), sealed=s.get("sealed", {}).get("exploratory"))
        except (OSError, KeyError, ValueError): entry["calibrator_sha256"] = None
        if reason == "low_memory_killed": raise Stop("capacity")
        return entry["status"] == "ok"

    def main(self):
        # v4.5: restore the caller's signal dispositions afterwards, so an in-process call (tests) does not leave
        # SIGTERM/SIGHUP/SIGINT ignored for the caller and for every process it starts later.
        saved = {s: signal.getsignal(s) for s in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP, signal.SIGALRM)}
        try: return self._main()
        finally:
            for s, h in saved.items():
                if h is not None: signal.signal(s, h)

    def _main(self):
        a, rc = self.a, 5
        for s in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP, signal.SIGALRM): signal.signal(s, self.on_signal)
        start_wall, start_mono = self.clock.wall(), time.monotonic()
        self.lease_until = None
        limit = math.inf
        if a.total_timeout is not None:
            if a.total_timeout <= 0: print("ABORT: --total-timeout must be positive"); return 2
            limit = a.total_timeout
        if a.lease_valid_until:
            lv = dt.datetime.fromisoformat(a.lease_valid_until)
            if lv.tzinfo is None: print("ABORT: --lease-valid-until needs a timezone"); return 2
            self.lease_until = lv
            limit = min(limit, (lv - start_wall).total_seconds() / self.clock.scale)
        if limit <= 0: print("ABORT: lease already expired"); return 2
        self.deadline = None if math.isinf(limit) else start_mono + self.clock.s(limit)
        # Held until the try block: a signal from here on is delivered inside it, so the receipt is always written.
        signal.pthread_sigmask(signal.SIG_BLOCK, BLOCK)
        if self.deadline is not None:
            # backstop only: arm slots are already cut to the deadline; fire after the in-slot kill grace has elapsed
            signal.setitimer(signal.ITIMER_REAL, max(0.1, self.left() + self.clock.s(KILL_GRACE)))
        tag = ("dry-v4-" if self.dry else "v4-") + dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        if self.dry: self.dry_dir.mkdir(parents=True, exist_ok=True)
        base = scoped(self.dry_dir if self.dry else ROOT / "runs")
        cohort = base / tag; os.mkdir(cohort, 0o755)
        seeds = [int(x) for x in a.seeds.split(",")]
        R = {"schema": "supervisor_receipt_v4_5", "dry_run": self.dry, "gpu_uuid": GPU_UUID, "seeds": seeds,
             "supervisor": ident(proc_info(os.getpid())), "start": start_wall.isoformat(), "particle_trainer": TRAINER, "particlegan_commit": PARTICLEGAN_COMMIT,
                 "aggregate_time_cap": None,
             "total_timeout_s": a.total_timeout, "lease_valid_until": a.lease_valid_until, "arm_timeout_s": a.arm_timeout,
             "teardown_timeout_s": a.teardown_timeout, "operator_deadline_epoch": None if self.deadline is None else time.time() + self.left(),
             "time_scale": self.clock.scale, "arms": [], "runs": {}, "stop_reason": None,
             "signals_handled": ["SIGTERM", "SIGINT", "SIGHUP", "SIGALRM"],
             "parent_watch": {"pdeathsig": not (self.dry and a.dry_no_pdeathsig), "parent_poll": not (self.dry and a.dry_no_parent_poll)}}
        if self.dry: R["real_lock_files_exist"] = {p: os.path.exists(p) for p in REAL_LOCKS}
        try:
            signal.pthread_sigmask(signal.SIG_UNBLOCK, BLOCK)
            try:   # lets the wrapper probe the cohort even if this process is later SIGKILLed (no receipt possible then)
                print(json.dumps({"event": "start", "cohort": str(cohort), "supervisor": R["supervisor"]}), flush=True)
            except OSError: pass
            R["static_checks"] = statics = self.static_checks()
            R["llama_server_before"] = llama_servers()
            R["locks"] = locks = self.acquire(self.lock_paths())
            cap = {"schema": "gpu1_capacity_receipt_v1", "gpu_uuid": GPU_UUID, "at": self.clock.wall().isoformat(),
                   "free_mib": self.free_mib(), "dry_run": self.dry}
            lease = {"schema": "gpu1_lease_receipt_v2", "locks": locks, "holder": R["supervisor"], "acquired_at": self.clock.wall().isoformat(),
                     "valid_until": a.lease_valid_until, "ownership_reference": a.ownership_reference or "DRY-RUN", "dry_run": self.dry}
            cap_sha = sha(create_new(cohort, "capacity_receipt.json", jbytes(cap)))
            lease_path = create_new(cohort, "lease_receipt.json", jbytes(lease)); lease_sha = sha(lease_path)
            R["capacity_at_start"] = cap
            if cap["free_mib"] < MIN_START_MIB: raise Stop(f"start_capacity_{cap['free_mib']}MiB_lt_9GiB")
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=("" if self.dry else GPU_UUID), PYTHONPATH=str(REPO / "src"),
                       ALZ_LEASE_FDS=",".join(map(str, self.fds)), TMPDIR=str(CACHE), XDG_CACHE_HOME=str(CACHE),
                       MPLCONFIGDIR=str(CACHE / "matplotlib"), NUMBA_CACHE_DIR=str(CACHE / "numba"), PYTHONDONTWRITEBYTECODE="1")
            env.pop(parent_watch.ENV, None)
            if not (self.dry and a.dry_no_parent_poll): env[parent_watch.ENV] = parent_watch.identity_env()
            out_of_time = False
            for seed in seeds:
                run = cohort / f"seed{seed}"; os.mkdir(run, 0o755)
                grant, gsha = self.write_grant(run, cohort, seed, statics, lease_path, lease_sha, cap_sha)
                R["runs"][str(seed)] = {"path": str(run), "grant_sha256": gsha}
                for arm in ARMS:
                    e = {"seed": seed, "arm": arm, "status": None}; R["arms"].append(e)
                    if out_of_time: e["status"] = "skipped_operator_deadline"; continue
                    if arm == "G_frozen" and next(x for x in R["arms"] if x["seed"] == seed and x["arm"] == "G")["status"] != "ok":
                        e["status"] = "skipped_no_G_checkpoint"; continue
                    if not self.run_arm(arm, seed, run, e, grant, gsha, env):
                        if e["status"] == "skipped_operator_deadline": out_of_time = True; continue
                        raise Stop(e["status"])
            rc = 0 if all(x["status"] == "ok" for x in R["arms"]) else 5
        except Stop as s:
            R["stop_reason"] = str(s.args[0]); rc = 3
            if self.current and self.current[0]["status"] is None: self.current[0]["status"] = f"interrupted_{s.args[0]}"
        finally:
            self.cleaning = True
            signal.setitimer(signal.ITIMER_REAL, 0)
            for s in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP): signal.signal(s, signal.SIG_IGN)
            signal.pthread_sigmask(signal.SIG_UNBLOCK, BLOCK)
            # Teardown has its own budget: it runs even if the operator deadline has already passed.
            t_end = time.monotonic() + self.clock.s(a.teardown_timeout)
            for pgid in list(self.groups): self.kill_group(pgid, t_end)
            for g in self.groups.values():
                try: g["popen"].wait(timeout=max(0.01, min(0.5, t_end - time.monotonic())))
                except subprocess.TimeoutExpired: pass
            def survey():
                w = watchers_for(self.groups)
                v = {str(pg): {"leader_alive": alive_same(g["leader"]), "inner_alive": alive_same(g.get("inner")),
                               "group": group_ownership(pg, g["leader"], [g.get("inner")])[0],
                               "watchers": [x for x in w if x["arm_pgid"] == pg]} for pg, g in self.groups.items()}
                return {k: x for k, x in v.items() if x["leader_alive"] or x["inner_alive"] or x["group"] == "ours" or x["watchers"]}
            before = survey()
            # Final wait (holding the lock FDs) for owned descendants, bounded by the teardown budget,
            # BEFORE the receipt is written so the receipt states the true end state.
            w0 = time.monotonic(); after = before
            while after and time.monotonic() < t_end:
                time.sleep(0.2); after = survey()
            gone = not after
            R["final_wait"] = {"survivors_before": before, "survivors_after": after, "waited_s": round(time.monotonic() - w0, 3),
                               "ended_by": "all_gone" if gone else "teardown_budget"}
            R["cleanup"] = {"owned_groups": [{"pgid": pg, "leader": g["leader"], "inner": g.get("inner")} for pg, g in self.groups.items()],
                            "survivors": after, "all_children_gone": gone, "refused_signals": self.refused,
                            "teardown_within_budget": time.monotonic() <= t_end}
            R["signals_sent"] = self.signals_sent
            R["llama_server_after"] = llama_servers()
            R["llama_server_unchanged"] = R.get("llama_server_before") == R["llama_server_after"]
            R["locks_held_until_exit"] = bool(self.fds)
            R["end"] = self.clock.wall().isoformat()
            R["elapsed_s"] = round((time.monotonic() - start_mono) / self.clock.scale, 1)
            R["past_operator_deadline"] = self.deadline is not None and time.monotonic() > self.deadline
            if not gone: rc = 7 if rc == 0 else rc
            R["rc"] = rc
            R["completeness"] = "complete" if rc == 0 else "incomplete"
            R["incomplete_reason"] = (None if rc == 0 else R["stop_reason"] or ("descendants_survived" if not gone else "not_all_arms_ok"))
            p = create_new(cohort, "supervisor_receipt.json", jbytes(R))
            try:   # the receipt exists either way; a vanished stdout (hangup) must not turn into a traceback
                print(json.dumps({"receipt": str(p), "sha256": sha(p), "rc": rc, "completeness": R["completeness"], "stop_reason": R["stop_reason"],
                                  "arms": [(x["seed"], x["arm"], x["status"]) for x in R["arms"]], "cleanup_ok": gone}), flush=True)
            except OSError:
                pass
        return rc


def _probe_locks(locks, is_ours, left):
    out = []
    for L in locks:
        e = {"path": L["path"]}; out.append(e)
        if left() <= 0: e["state"] = "not_probed_timeout"; continue
        try: fd = os.open(scoped(L["path"]), os.O_RDONLY | os.O_NOFOLLOW)
        except FileNotFoundError: e["state"] = "missing"; continue
        try:
            st = os.fstat(fd); e.update(inode=st.st_ino, device=st.st_dev, identity_matches=(st.st_ino, st.st_dev) == (L["inode"], L["device"]))
            if not e["identity_matches"]: e["state"] = "identity_changed"; continue
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB); e["state"] = "free"   # released by close; no LOCK_UN
            except BlockingIOError:
                holders, err = proc_lock_holders(st); e["holders"] = holders; e["state"] = "busy"
                if not holders: e.update(owner="unknown_owner", error=err)
                else:
                    pids = sorted({h["pid"] for h in holders}); e["holder_identities"] = [ident(proc_info(x)) for x in pids]
                    e["owner"] = ("ours" if all(is_ours(x) for x in pids) else
                                  "other_owner" if all(proc_info(x)["starttime"] is not None for x in pids) else "unknown_owner")
        finally: os.close(fd)
    return out


def _probe_gpu(P, left):
    rows, err = nvidia_smi(["--query-gpu=uuid,memory.free,memory.used"], min(PROBE_MAX, left()))
    P["nvidia_smi_gpu"] = {"rows": rows, "error": err}
    apps, err2 = nvidia_smi(["--query-compute-apps=gpu_uuid,pid,used_memory"], min(PROBE_MAX, left()))
    P["nvidia_smi_apps"] = {"rows": apps, "error": err2}
    P["free_mib_per_gpu"] = {r[0]: int(r[1]) for r in rows} if rows else None
    # HOLD 1: any failed / timed-out / empty GPU query, or our GPU missing from the answer, means NOT clean.
    P["gpu_query_ok"] = (err is None and err2 is None and rows is not None and apps is not None
                         and any(r and r[0] == GPU_UUID for r in rows))
    return apps


def _finish_probe(P, directory, stop_requested=None, defer=False):
    """Write post_run_receipt.json exclusively; verdict 'clean' ONLY for a complete run with clean residue.
    v4.6: a stop requested of the chain (wrapper got SIGTERM) means the run is never complete."""
    if defer:
        # An observation is deliberately not a terminal verdict. The wrapper
        # closes cancellation acceptance before publishing the final receipt.
        P["publication"] = "provisional"
        p = create_new(directory, "post_run_probe_observation.json", jbytes(P))
        print(json.dumps({"probe_observation": str(p), "sha256": sha(p)}), flush=True)
        return 0
    P["publication"] = "final"
    P["stop_requested"] = stop_requested
    if stop_requested:
        why = f"stop requested of the chain ({stop_requested})"
        if P["run_status"] == "complete": P["run_status"], P["run_status_reason"] = "incomplete", why
        else: P["run_status_reason"] = f"{P.get('run_status_reason')}; {why}"
    P["verdict"] = ("attention" if P["residue"] == "attention" else
                    "clean" if P["run_status"] == "complete" and P["residue"] == "clean" else
                    P["run_status"] if P["run_status"] != "complete" else "unknown")
    p = create_new(directory, "post_run_receipt.json", jbytes(P))
    print(json.dumps({"post_run_receipt": str(p), "sha256": sha(p), "verdict": P["verdict"], "run_status": P["run_status"],
                      "residue": P["residue"], "locks": [(l["path"], l.get("state"), l.get("owner")) for l in P["locks"]]}), flush=True)
    return {"clean": 0, "attention": 6}.get(P["verdict"], 8)


def post_run_probe(receipt_path, want_sha, probe_timeout, stop_requested=None, defer=False):
    """Separate process after the supervisor exited; bounded by its own probe_timeout, independent of the run.
    Exit 0 clean (complete run, clean residue), 6 attention (residue), 8 incomplete run with clean residue."""
    t_end = time.monotonic() + probe_timeout
    def left(): return t_end - time.monotonic()
    receipt_path = scoped(receipt_path)
    if sha(receipt_path) != want_sha: print("ABORT: supervisor receipt sha256 mismatch"); return 2
    R = json.loads(Path(receipt_path).read_text())
    sup = R["supervisor"]
    groups = R["cleanup"]["owned_groups"]
    recorded = [sup] + [g["leader"] for g in groups] + [g["inner"] for g in groups if g.get("inner") and g["inner"].get("pid")]
    P = {"schema": "post_run_receipt_v3", "mode": "with_supervisor_receipt", "supervisor_receipt": str(receipt_path),
         "supervisor_receipt_sha256": want_sha, "prober": ident(proc_info(os.getpid())), "at": dt.datetime.now(DEN).isoformat(),
         "probe_timeout_s": probe_timeout, "supervisor_alive": alive_same(sup)}
    def is_ours(pid):
        i = proc_info(pid)
        if any(r["pid"] == pid and r["starttime"] == i["starttime"] for r in recorded): return True
        for g in groups:
            if i["pgid"] == g["pgid"] and group_ownership(g["pgid"], g["leader"], [g.get("inner")])[0] == "ours": return True
        return False
    P["locks"] = _probe_locks(R.get("locks", []), is_ours, left)
    apps = _probe_gpu(P, left)
    P["recorded_identities_alive"] = [r for r in recorded if alive_same(r)]
    P["owned_groups_live"] = [g["pgid"] for g in groups if group_ownership(g["pgid"], g["leader"], [g.get("inner")])[0] == "ours"]
    P["unverified_groups_live"] = [g["pgid"] for g in groups if group_ownership(g["pgid"], g["leader"], [g.get("inner")])[0] == "unverified"]
    P["watchers_alive"] = watchers_for([g["pgid"] for g in groups])
    P["owned_pids_on_gpu"] = [r for r in (apps or []) if r[1].isdigit() and is_ours(int(r[1]))]
    now_llama = llama_servers()
    P["llama_server_before"], P["llama_server_now"] = R.get("llama_server_before"), now_llama
    P["llama_server_unchanged"] = R.get("llama_server_before") == now_llama
    P["within_probe_budget"] = left() > 0
    P["residue"] = ("clean" if P["gpu_query_ok"] and not P["recorded_identities_alive"] and not P["owned_groups_live"]
                    and not P["unverified_groups_live"] and not P["owned_pids_on_gpu"] and not P["watchers_alive"]
                    and P["llama_server_unchanged"] and P["within_probe_budget"]
                    and all(l.get("state") == "free" or l.get("owner") == "other_owner" for l in P["locks"]) else "attention")
    complete = (R.get("rc") == 0 and R.get("stop_reason") is None and bool(R.get("arms"))
                and all(x.get("status") == "ok" for x in R["arms"]) and R.get("completeness", "complete") == "complete")
    P["run_status"] = "complete" if complete else "incomplete"
    P["run_status_reason"] = None if complete else (R.get("incomplete_reason") or R.get("stop_reason") or f"rc={R.get('rc')}")
    return _finish_probe(P, Path(receipt_path).parent, stop_requested, defer)


def post_run_probe_no_receipt(cohort, probe_timeout, stop_requested=None, defer=False):
    """Probe a cohort whose supervisor left NO receipt (SIGKILL, kernel OOM kill, crash). Never 'clean':
    the run is 'incomplete' (lease receipt present) or 'unknown' (no lease receipt). Exit 8, or 6 for residue."""
    t_end = time.monotonic() + probe_timeout
    def left(): return t_end - time.monotonic()
    cohort = scoped(cohort)
    if not cohort.is_dir(): print("ABORT: --cohort is not a directory"); return 2
    if (cohort / "supervisor_receipt.json").exists():
        print("ABORT: cohort has a supervisor receipt; probe it with --supervisor-receipt/--sha256"); return 2
    lease_path = cohort / "lease_receipt.json"
    lease = json.loads(lease_path.read_text()) if lease_path.is_file() and not lease_path.is_symlink() else None
    sup = (lease or {}).get("holder")
    P = {"schema": "post_run_receipt_v3", "mode": "no_supervisor_receipt", "cohort": str(cohort), "supervisor_receipt": None,
         "lease_receipt": str(lease_path) if lease else None, "lease_receipt_sha256": sha(lease_path) if lease else None,
         "prober": ident(proc_info(os.getpid())), "at": dt.datetime.now(DEN).isoformat(), "probe_timeout_s": probe_timeout,
         "supervisor_alive": alive_same(sup) if sup else None,
         "run_status": "incomplete" if lease else "unknown",
         "run_status_reason": "supervisor exited without a receipt (external SIGKILL, kernel OOM kill or crash; cause not determined)",
         "arm_identities": "unknown: not recorded without a supervisor receipt"}
    def is_ours(pid):
        i = proc_info(pid)
        return bool(sup) and pid == sup["pid"] and i["starttime"] == sup["starttime"]
    P["locks"] = _probe_locks((lease or {}).get("locks", []), is_ours, left)
    _probe_gpu(P, left)
    P["llama_server_now"] = llama_servers()
    P["within_probe_budget"] = left() > 0
    P["residue"] = ("unknown" if not lease else "clean" if P["gpu_query_ok"] and not P["supervisor_alive"] and P["within_probe_budget"]
                    and all(l.get("state") == "free" for l in P["locks"]) else "attention")
    return _finish_probe(P, cohort, stop_requested, defer)


def parse(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="42,43,44")
    ap.add_argument("--arm-timeout", type=float, default=600, help="per-arm timeout in seconds (configurable)")
    ap.add_argument("--total-timeout", type=float, default=None, help="optional operator deadline; no aggregate cap by default")
    ap.add_argument("--teardown-timeout", type=float, default=60)
    ap.add_argument("--harness-commit"); ap.add_argument("--bundle"); ap.add_argument("--bundle-sha256"); ap.add_argument("--protocol-sha256")
    ap.add_argument("--authorization-reference"); ap.add_argument("--ownership-reference")
    ap.add_argument("--lease-valid-until", help="ISO-8601 with timezone; required for real runs (ownership lease)")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--dry-dir"); ap.add_argument("--time-scale", type=float, default=0.05)
    ap.add_argument("--dry-arm-seconds", type=float, default=2.0); ap.add_argument("--dry-fail-arm")
    ap.add_argument("--dry-skip-static-checks", action="store_true")
    ap.add_argument("--dry-no-pdeathsig", action="store_true", help="test only: disable layer 1 of the parent watch")
    ap.add_argument("--dry-no-parent-poll", action="store_true", help="test only: disable layer 2 of the parent watch")
    ap.add_argument("--dry-arm-term-ignoring-child", action="store_true", help="test only: each fake arm starts a child that ignores SIGTERM")
    ap.add_argument("--defer-probe-finalization", action="store_true")
    ap.add_argument("--finalize-probe", action="store_true")
    ap.add_argument("--probe-observation")
    ap.add_argument("--post-run-probe", action="store_true"); ap.add_argument("--supervisor-receipt"); ap.add_argument("--sha256")
    ap.add_argument("--cohort", help="probe a cohort that has NO supervisor receipt (supervisor killed); result is never clean")
    ap.add_argument("--probe-timeout", type=float, default=30)
    ap.add_argument("--stop-requested", choices=["wrapper_sigterm"], help="probe: the chain wrapper received a stop; never clean")
    return ap.parse_args(argv)


def main(argv=None):
    a = parse(argv)
    if a.finalize_probe:
        if a.post_run_probe or a.defer_probe_finalization or not (a.probe_observation and a.sha256):
            print("ABORT: finalization needs only --probe-observation/--sha256 and optional --stop-requested"); return 2
        observation = scoped(a.probe_observation)
        if sha(observation) != a.sha256: raise ValueError("probe observation hash mismatch")
        P = json.loads(observation.read_text())
        if P.get("publication") != "provisional" or "verdict" in P: raise ValueError("not a provisional observation")
        P["observation_sha256"] = a.sha256
        return _finish_probe(P, observation.parent, a.stop_requested)
    if a.probe_observation or (a.defer_probe_finalization and not a.post_run_probe):
        print("ABORT: probe-only options"); return 2
    if a.post_run_probe:
        if a.cohort and not (a.supervisor_receipt or a.sha256): return post_run_probe_no_receipt(a.cohort, a.probe_timeout, a.stop_requested, a.defer_probe_finalization)
        if a.cohort or not (a.supervisor_receipt and a.sha256):
            print("ABORT: --post-run-probe needs --supervisor-receipt and --sha256, or only --cohort"); return 2
        return post_run_probe(a.supervisor_receipt, a.sha256, a.probe_timeout, a.stop_requested, a.defer_probe_finalization)
    if a.stop_requested: print("ABORT: --stop-requested is a probe option"); return 2
    if not a.dry_run:
        bad = [k for k in ("dry_dir", "dry_fail_arm", "dry_skip_static_checks", "dry_no_pdeathsig", "dry_no_parent_poll",
                           "dry_arm_term_ignoring_child") if getattr(a, k)]
        if bad: print(f"ABORT: dry-only options in a real run: {bad}"); return 2
        for k in ("authorization_reference", "ownership_reference", "lease_valid_until"):
            if not getattr(a, k): print(f"ABORT: --{k.replace('_', '-')} is required"); return 2
    if not (a.dry_run and a.dry_skip_static_checks):
        for k in ("harness_commit", "bundle", "bundle_sha256", "protocol_sha256"):
            if not getattr(a, k): print(f"ABORT: --{k.replace('_', '-')} is required"); return 2
    if a.arm_timeout <= 0 or a.teardown_timeout <= 0: print("ABORT: timeouts must be positive"); return 2
    return Supervisor(a).main()


if __name__ == "__main__":
    sys.exit(main())
