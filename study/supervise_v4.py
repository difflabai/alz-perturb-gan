#!/usr/bin/env python3
"""Supervisor v4 (plan v4 sections 5-6): one global monotonic deadline across --seeds, plus a post-run probe.

Global budget: deadline = start + --total-cap (default 3600 s, the approved cap; 4500 s is NOT approved),
never beyond --lease-valid-until. A 60 s final reserve is kept for cleanup/receipts. Each arm slot is
min(600, deadline - now - 60); an arm whose slot is < 120 s is recorded as skipped_global_budget and no
further arm starts. If cleanup of an arm overruns its slot the arm is recorded incomplete_overrun.
Identity: supervisor and every arm leader are recorded as (pid, /proc start time, uid).
Locks: existing files only (no O_CREAT), dev+inode verified, LOCK_EX|LOCK_NB, never LOCK_UN; released at exit.
Post-run probe (--post-run-probe): separate process after the supervisor exits; refuses after the
supervisor's recorded deadline; verifies the supervisor receipt sha256; per lock: open without O_CREAT,
fstat dev/inode == receipt, flock NB then close (no LOCK_UN); a busy lock is labelled ours/other_owner/
unknown_owner from /proc/locks; nvidia-smi bounded; owned pgids empty; llama-server identities unchanged;
writes post_run_receipt.json exclusively (openat O_CREAT|O_EXCL|O_NOFOLLOW).
--dry-run: CPU only, test locks under /ml2/altgan/.cache/supervisor-dryrun, fake arms, times x0.05.
"""
from __future__ import annotations
import argparse, datetime as dt, fcntl, hashlib, json, os, signal, subprocess, sys, time
from pathlib import Path
from zoneinfo import ZoneInfo

REPO = Path("/ml2/altgan/gene-feature-harness-20261014")
sys.path.insert(0, str(REPO / "src"))
from alz_perturb_gan.paths import scoped, create_new  # noqa: E402

DEN = ZoneInfo("America/Denver")
APPROVED_TOTAL_CAP, PER_ARM_CAP, STEPS = 3600, 600, 2000
MIN_START_MIB, MIN_RUN_MIB = 9 * 1024, 4 * 1024
POLL_S, MIN_ARM_S, FINAL_RESERVE, KILL_RESERVE, KILL_GRACE, PROBE_MAX = 5, 120, 60, 30, 10, 5
GPU_UUID = "GPU-cb4ce47d-d968-bffd-5646-e830a9fa1c69"
REAL_LOCKS = ["/tmp/ember-pg-20261012-slots/gpu1-slot0.lock", "/tmp/ember-pg-20261012-slots/gpu1-slot1.lock"]
ROOT = Path("/ml2/altgan/studies/gene-feature-20261014")
PY, CACHE = "/ml2/altgan/.venv/bin/python", Path("/ml2/altgan/.cache")
DRY_DIR = CACHE / "supervisor-dryrun"
F = Path("/ml2/altgan/features")
FEATURES = {
    "feature_manifest": (F / "manifest.json", "d6d501f7b5ca7af2920502537582d0d6401359b5ae2cb6f12271891c8895d844"),
    "string_matrix": (F / "string_spectral.npy", "6a428429b6556f30bc690d252cb6d7910665cb52755d76a9d0795cfcf15c90f5"),
    "string_row_mapping": (F / "addons/string_row_mapping.json", "044b2756e940d31519c85d81073a3dc17ac7f3115a8fc4e04b6a539f4b994a74"),
    "string_license_receipt": (F / "addons/string_license_receipt.json", "1a7666849ddabcfedf7547b8eac844514eb8038be12a591743221b6e3711da86"),
}
ALLOC_CAP_GIB = 4
ARMS = ["B0", "G", "H", "GH", "shuffle", "G_frozen"]


class Stop(Exception): pass


def sha(p):
    h = hashlib.sha256()
    with open(scoped(p) if not str(p).startswith("/proc") else p, "rb") as f:
        for c in iter(lambda: f.read(8 << 20), b""): h.update(c)
    return h.hexdigest()


def jbytes(o): return (json.dumps(o, indent=2, sort_keys=True) + "\n").encode()


def proc_info(pid):
    try:
        st = open(f"/proc/{pid}/stat").read()
        uid = next(l.split()[1] for l in open(f"/proc/{pid}/status") if l.startswith("Uid:"))
        return {"pid": pid, "comm": st[st.index("(") + 1:st.rindex(")")], "uid": int(uid), "starttime": int(st.rsplit(")", 1)[1].split()[19])}
    except (OSError, ValueError, StopIteration, IndexError):
        return {"pid": pid, "comm": None, "uid": None, "starttime": None}


def pids_in_group(pgid):
    found = set()
    for d in os.listdir("/proc"):
        if not d.isdigit(): continue
        try:
            f = open(f"/proc/{d}/stat").read().rsplit(")", 1)[1].split()
            if int(f[2]) == pgid and f[0] != "Z": found.add(int(d))
        except (OSError, IndexError, ValueError): pass
    return found


def llama_servers():
    out = []
    for d in os.listdir("/proc"):
        if d.isdigit():
            i = proc_info(int(d))
            if i["comm"] and i["comm"].startswith("llama-server"): out.append(i)
    return sorted(out, key=lambda x: x["pid"])


def nvidia_smi(args, timeout):
    """Bounded, read-only query. Returns (rows, error)."""
    try:
        r = subprocess.run(["/usr/bin/nvidia-smi", *args, "--format=csv,noheader,nounits"], capture_output=True, text=True,
                           timeout=max(0.5, timeout), check=True)
        return [[x.strip() for x in l.split(",")] for l in r.stdout.strip().splitlines() if l.strip()], None
    except Exception as e:  # noqa: BLE001 - recorded, never fatal for the probe
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


class Clock:
    def __init__(self, scale): self.scale, self.t0 = scale, time.monotonic()
    def wall(self): return dt.datetime.now(DEN)
    def s(self, real_seconds): return real_seconds * self.scale


class Supervisor:
    def __init__(self, a):
        self.a, self.dry = a, a.dry_run
        self.fds, self.active, self.history, self.cleaning, self.signals_sent = [], {}, [], False, []
        self.clock = Clock(0.05 if self.dry else 1.0)
        self.current = None

    def on_signal(self, signum, _f):
        if self.cleaning: return
        raise Stop({signal.SIGTERM: "sigterm", signal.SIGINT: "sigint", signal.SIGALRM: "outer_watchdog"}[signum])

    def left(self): return self.deadline - time.monotonic()

    def free_mib(self):
        if self.dry: return 20000
        rows, err = nvidia_smi(["--query-gpu=uuid,memory.free"], min(self.clock.s(PROBE_MAX), self.left()))
        if rows is None: raise Stop(f"nvidia_smi_failed:{err}")
        for u, f in rows:
            if u == GPU_UUID: return int(f)
        raise Stop("gpu_not_visible")

    def kill_group(self, pgid, deadline):
        if not pids_in_group(pgid): return True
        grace = max(0.2, min(self.clock.s(KILL_GRACE) / 2, (deadline - time.monotonic()) / 2))
        for sig in (signal.SIGTERM, signal.SIGKILL):
            self.signals_sent.append({"pgid": pgid, "signal": signal.Signals(sig).name, "owned": pgid in self.history,
                                      "at": self.clock.wall().isoformat()})
            if pgid not in self.history: return False          # never signal a group we did not create
            try: os.killpg(pgid, sig)
            except ProcessLookupError: return True
            end = time.monotonic() + grace
            while time.monotonic() < end:
                if not pids_in_group(pgid): return True
                time.sleep(0.1)
        return not pids_in_group(pgid)

    def static_checks(self):
        out = {"script_sha256": sha(__file__)}
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
        if not self.dry:
            for k in ("authorization_reference", "ownership_reference", "lease_valid_until"):
                if not getattr(self.a, k): raise Stop(f"missing_--{k.replace('_', '-')}")
        return out

    def lock_paths(self):
        if not self.dry: return REAL_LOCKS
        DRY_DIR.mkdir(parents=True, exist_ok=True)
        ps = [DRY_DIR / "gpu1-slot0.test.lock", DRY_DIR / "gpu1-slot1.test.lock"]
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
            except BlockingIOError: raise Stop(f"lock_busy:{p}")
            os.set_inheritable(fd, True); self.fds.append(fd)
            info.append({"path": p, "inode": st.st_ino, "device": st.st_dev, "proc_locks_key": lock_key(st), "fd": fd})
        for i in info:
            st = os.stat(i["path"], follow_symlinks=False)
            if (st.st_ino, st.st_dev) != (i["inode"], i["device"]): raise Stop(f"lock_inode_changed:{i['path']}")
        return info

    def write_grant(self, run, cohort, seed, statics, lease_sha, cap_sha):
        grant = {"approved": True, "authorization_reference": self.a.authorization_reference or "DRY-RUN",
                 "phases": ["train", "evaluate"], "arms": ARMS, "seeds": [seed], "protocol_sha256": self.a.protocol_sha256,
                 "bundle_sha256": statics["bundle"], **{f"{k}_sha256": statics[k] for k in FEATURES},
                 "harness_commit": statics["harness_commit"], "supervisor_sha256": statics["script_sha256"],
                 "cuda_allocator_cap_gib": ALLOC_CAP_GIB, "max_steps_per_arm": STEPS, "max_wall_seconds_per_arm": PER_ARM_CAP,
                 "max_total_wall_seconds": self.a.total_cap, "global_deadline_utc": self.deadline_wall.astimezone(dt.timezone.utc).isoformat(),
                 "gpu_uuid": GPU_UUID, "lease_receipt_sha256": lease_sha, "capacity_receipt_sha256": cap_sha,
                 "ownership_reference": self.a.ownership_reference or "DRY-RUN", "cohort": str(cohort)}
        p = create_new(run, "grant.json", jbytes(grant))
        return p, sha(p)

    def arm_cmd(self, arm, seed, run, slot, grant, gsha):
        hard = max(1, int(slot - self.clock.s(KILL_GRACE)))
        wd = ["timeout", "-k", str(max(1, int(self.clock.s(KILL_GRACE) / 2))), str(max(1, hard - int(self.clock.s(KILL_GRACE) / 2)))]
        if self.dry:
            return wd + ["bash", "-c", f"sleep 2 & wait; echo fake {arm} seed {seed} fds=$ALZ_LEASE_FDS"]
        mw = int(slot - self.clock.s(KILL_RESERVE))
        cmd = [PY, "-u", "-m", "alz_perturb_gan.gene_harness", "train", "--bundle", str(self.bundle), "--bundle-sha256", self.a.bundle_sha256,
               "--output", str(run / arm), "--arm", arm, "--seed", str(seed), "--steps", str(STEPS), "--max-wall-seconds", str(mw),
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
        slot = min(self.clock.s(PER_ARM_CAP), self.deadline - t_start - self.clock.s(FINAL_RESERVE))
        entry["slot_s"] = round(slot / self.clock.scale, 1)
        entry["global_left_s"] = round((self.deadline - t_start) / self.clock.scale, 1)
        if slot < self.clock.s(MIN_ARM_S): entry["status"] = "skipped_global_budget"; return False
        free = self.free_mib(); entry["free_mib_at_spawn"] = free
        if free < MIN_RUN_MIB: entry["status"] = "not_started_low_memory"; raise Stop("capacity")
        cmd = self.arm_cmd(arm, seed, run, slot, grant, gsha)
        entry.update(command=cmd, started=self.clock.wall().isoformat())
        logfd = os.open(scoped(run), os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
        try: lfd = os.open(f"{arm}.log", os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o644, dir_fd=logfd)
        finally: os.close(logfd)
        p = subprocess.Popen(cmd, cwd=str(REPO), env=env, stdout=lfd, stderr=subprocess.STDOUT, start_new_session=True, pass_fds=tuple(self.fds))
        os.close(lfd)
        pgid = p.pid
        self.active[pgid] = {"arm": arm, "leader": proc_info(p.pid)}; self.history.append(pgid)
        entry["pgid"], entry["leader"] = pgid, self.active[pgid]["leader"]
        kill_at = t_start + slot - self.clock.s(KILL_GRACE)
        reason, minfree = None, None
        self.current = (entry, pgid)
        while p.poll() is None:
            now = time.monotonic()
            if now >= kill_at: reason = "timeout"; break
            free = self.free_mib(); minfree = free if minfree is None else min(minfree, free)
            if free < MIN_RUN_MIB: reason = "low_memory_killed"; break
            try: p.wait(timeout=max(0.05, min(self.clock.s(POLL_S), kill_at - time.monotonic())))
            except subprocess.TimeoutExpired: pass
        entry["min_free_mib"] = minfree
        entry["status"] = reason or ("ok" if p.returncode == 0 else f"error_rc{p.returncode}")
        entry["group_killed_clean"] = self.kill_group(pgid, min(t_start + slot, self.deadline))
        if entry["group_killed_clean"]: self.active.pop(pgid, None)
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
        a, rc = self.a, 5
        if a.total_cap > APPROVED_TOTAL_CAP and not self.dry:
            print(f"ABORT: --total-cap {a.total_cap} exceeds the approved {APPROVED_TOTAL_CAP} s"); return 2
        for s in (signal.SIGTERM, signal.SIGINT, signal.SIGALRM): signal.signal(s, self.on_signal)
        start_wall, start_mono = self.clock.wall(), time.monotonic()
        budget_real = a.total_cap
        if a.lease_valid_until:
            lv = dt.datetime.fromisoformat(a.lease_valid_until)
            if lv.tzinfo is None: print("ABORT: --lease-valid-until needs a timezone"); return 2
            budget_real = min(budget_real, (lv - start_wall).total_seconds() / self.clock.scale)
        self.deadline = start_mono + self.clock.s(budget_real)              # ONE monotonic deadline for every seed
        self.deadline_wall = start_wall + dt.timedelta(seconds=self.clock.s(budget_real))
        signal.setitimer(signal.ITIMER_REAL, max(0.1, self.left() - self.clock.s(FINAL_RESERVE) / 2))
        tag = ("dry-v4-" if self.dry else "v4-") + dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        if self.dry: DRY_DIR.mkdir(parents=True, exist_ok=True)
        base = scoped(DRY_DIR if self.dry else ROOT / "runs")
        cohort = base / tag; os.mkdir(cohort, 0o755)
        seeds = [int(x) for x in a.seeds.split(",")]
        R = {"schema": "supervisor_receipt_v4", "dry_run": self.dry, "gpu_uuid": GPU_UUID, "seeds": seeds,
             "supervisor": proc_info(os.getpid()), "start": start_wall.isoformat(), "total_cap_s": a.total_cap,
             "approved_total_cap_s": APPROVED_TOTAL_CAP, "final_reserve_s": FINAL_RESERVE, "per_arm_cap_s": PER_ARM_CAP,
             "global_deadline": self.deadline_wall.isoformat(), "global_deadline_epoch": time.time() + self.left(),
             "time_scale": self.clock.scale, "arms": [], "runs": {}, "stop_reason": None}
        if self.dry: R["real_lock_files_exist"] = {p: os.path.exists(p) for p in REAL_LOCKS}
        try:
            R["static_checks"] = statics = self.static_checks()
            R["llama_server_before"] = llama_servers()
            R["locks"] = locks = self.acquire(self.lock_paths())
            cap = {"schema": "gpu1_capacity_receipt_v1", "gpu_uuid": GPU_UUID, "at": self.clock.wall().isoformat(),
                   "free_mib": self.free_mib(), "dry_run": self.dry}
            lease = {"schema": "gpu1_lease_receipt_v2", "locks": locks, "holder": R["supervisor"],
                     "acquired_at": self.clock.wall().isoformat(), "ownership_reference": a.ownership_reference or "DRY-RUN", "dry_run": self.dry}
            cap_sha = sha(create_new(cohort, "capacity_receipt.json", jbytes(cap)))
            lease_sha = sha(create_new(cohort, "lease_receipt.json", jbytes(lease)))
            R["capacity_at_start"] = cap
            if cap["free_mib"] < MIN_START_MIB: raise Stop(f"start_capacity_{cap['free_mib']}MiB_lt_9GiB")
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=("" if self.dry else GPU_UUID), PYTHONPATH=str(REPO / "src"),
                       ALZ_LEASE_FDS=",".join(map(str, self.fds)), TMPDIR=str(CACHE), XDG_CACHE_HOME=str(CACHE),
                       MPLCONFIGDIR=str(CACHE / "matplotlib"), NUMBA_CACHE_DIR=str(CACHE / "numba"), PYTHONDONTWRITEBYTECODE="1")
            budget_out = False
            for seed in seeds:
                run = cohort / f"seed{seed}"; os.mkdir(run, 0o755)
                grant, gsha = self.write_grant(run, cohort, seed, statics, lease_sha, cap_sha)
                R["runs"][str(seed)] = {"path": str(run), "grant_sha256": gsha}
                for arm in ARMS:
                    e = {"seed": seed, "arm": arm, "status": None}; R["arms"].append(e)
                    if budget_out: e["status"] = "skipped_global_budget"; continue
                    if arm == "G_frozen" and next(x for x in R["arms"] if x["seed"] == seed and x["arm"] == "G")["status"] != "ok":
                        e["status"] = "skipped_no_G_checkpoint"; continue
                    if not self.run_arm(arm, seed, run, e, grant, gsha, env):
                        if e["status"] == "skipped_global_budget": budget_out = True; continue
                        R["stop_reason"] = e["status"]; raise Stop(e["status"])
            rc = 0 if all(x["status"] == "ok" for x in R["arms"]) else 5
        except Stop as s:
            R["stop_reason"] = str(s.args[0]); rc = 3
            if self.current and self.current[0]["status"] is None: self.current[0]["status"] = f"interrupted_{s.args[0]}"
        finally:
            self.cleaning = True
            signal.setitimer(signal.ITIMER_REAL, 0)
            for s in (signal.SIGTERM, signal.SIGINT): signal.signal(s, signal.SIG_IGN)
            for g in list(self.active):
                if self.kill_group(g, self.deadline): self.active.pop(g)
            remaining = {g: sorted(pids_in_group(g)) for g in self.active}
            R["cleanup"] = {"owned_groups": self.history, "remaining_after_cleanup": remaining, "all_children_gone": not any(remaining.values())}
            R["signals_sent"] = self.signals_sent
            R["all_signals_to_owned_groups"] = all(x["owned"] for x in self.signals_sent)
            R["llama_server_after"] = llama_servers()
            R["llama_server_unchanged"] = R.get("llama_server_before") == R["llama_server_after"]
            R["locks_held_until_exit"] = bool(self.fds)
            R["end"] = self.clock.wall().isoformat()
            R["elapsed_s"] = round((time.monotonic() - start_mono) / self.clock.scale, 1)
            R["within_deadline"] = time.monotonic() <= self.deadline
            p = create_new(cohort, "supervisor_receipt.json", jbytes(R))
            print(json.dumps({"receipt": str(p), "sha256": sha(p), "rc": rc, "stop_reason": R["stop_reason"],
                              "arms": [(x["seed"], x["arm"], x["status"]) for x in R["arms"]], "within_deadline": R["within_deadline"]}))
            while any(pids_in_group(g) for g in self.active): time.sleep(1)   # hold lock FDs while owned descendants live
        return rc


def post_run_probe(receipt_path, want_sha):
    """Separate process, after the supervisor exited. Shares the supervisor's global deadline."""
    receipt_path = scoped(receipt_path)
    if sha(receipt_path) != want_sha: print("ABORT: supervisor receipt sha256 mismatch"); return 2
    R = json.loads(Path(receipt_path).read_text())
    deadline = R["global_deadline_epoch"]
    def left(): return deadline - time.time()
    if left() <= 0: print("ABORT: past the supervisor's global deadline"); return 4
    sup = R["supervisor"]; owned = set(R["cleanup"]["owned_groups"]) | {sup["pid"]}
    P = {"schema": "post_run_receipt_v1", "supervisor_receipt": str(receipt_path), "supervisor_receipt_sha256": want_sha,
         "prober": proc_info(os.getpid()), "at": dt.datetime.now(DEN).isoformat(), "deadline_epoch": deadline,
         "supervisor_alive": proc_info(sup["pid"]).get("starttime") == sup["starttime"], "locks": []}
    for L in R.get("locks", []):
        e = {"path": L["path"]}; P["locks"].append(e)
        try: fd = os.open(scoped(L["path"]), os.O_RDONLY | os.O_NOFOLLOW)      # never O_CREAT
        except FileNotFoundError: e["state"] = "missing"; continue
        try:
            st = os.fstat(fd); e.update(inode=st.st_ino, device=st.st_dev, identity_matches=(st.st_ino, st.st_dev) == (L["inode"], L["device"]))
            if not e["identity_matches"]: e["state"] = "identity_changed"; continue
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB); e["state"] = "free"   # released by close below; no LOCK_UN
            except BlockingIOError:
                holders, err = proc_lock_holders(st); e["holders"] = holders
                if holders is None: e.update(state="busy", owner="unknown_owner", error=err)
                else:
                    pids = {h["pid"] for h in holders}
                    def is_ours(pid):
                        try: return pid == sup["pid"] and proc_info(pid)["starttime"] == sup["starttime"] or os.getpgid(pid) in owned
                        except ProcessLookupError: return False
                    e["state"] = "busy"
                    e["owner"] = ("ours" if pids and all(is_ours(x) for x in pids) else
                                  "other_owner" if pids and all(proc_info(x)["comm"] for x in pids) else "unknown_owner")
                    e["holder_identities"] = [proc_info(x) for x in sorted(pids)]
        finally: os.close(fd)
    rows, err = nvidia_smi(["--query-gpu=uuid,memory.free,memory.used"], min(PROBE_MAX, left()))
    P["nvidia_smi_gpu"] = {"rows": rows, "error": err}
    apps, err2 = nvidia_smi(["--query-compute-apps=gpu_uuid,pid,used_memory"], min(PROBE_MAX, left()))
    P["nvidia_smi_apps"] = {"rows": apps, "error": err2}
    P["free_mib_per_gpu"] = {r[0]: int(r[1]) for r in rows} if rows else None
    P["owned_pgids_live"] = {str(g): sorted(pids_in_group(g)) for g in R["cleanup"]["owned_groups"]}
    P["owned_pgids_empty"] = not any(P["owned_pgids_live"].values())
    def gpg(pid):
        try: return os.getpgid(int(pid))
        except (ProcessLookupError, ValueError): return None
    P["owned_pids_on_gpu"] = [r for r in (apps or []) if gpg(r[1]) in owned]
    now_llama = llama_servers()
    P["llama_server_before"], P["llama_server_now"] = R.get("llama_server_before"), now_llama
    P["llama_server_unchanged"] = R.get("llama_server_before") == now_llama
    P["within_deadline"] = left() > 0
    P["verdict"] = ("clean" if P["owned_pgids_empty"] and not P["owned_pids_on_gpu"] and P["llama_server_unchanged"]
                    and all(l.get("state") == "free" or l.get("owner") == "other_owner" for l in P["locks"]) and P["within_deadline"]
                    else "attention")
    p = create_new(Path(receipt_path).parent, "post_run_receipt.json", jbytes(P))
    print(json.dumps({"post_run_receipt": str(p), "sha256": sha(p), "verdict": P["verdict"],
                      "locks": [(l["path"], l.get("state"), l.get("owner")) for l in P["locks"]]}))
    return 0 if P["verdict"] == "clean" else 6


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="42,43,44")
    ap.add_argument("--total-cap", type=float, default=APPROVED_TOTAL_CAP, help="seconds; default and maximum 3600 (4500 is not approved)")
    ap.add_argument("--harness-commit"); ap.add_argument("--bundle"); ap.add_argument("--bundle-sha256"); ap.add_argument("--protocol-sha256")
    ap.add_argument("--authorization-reference"); ap.add_argument("--ownership-reference")
    ap.add_argument("--lease-valid-until", help="ISO-8601 with timezone; the global deadline never exceeds it")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--post-run-probe", action="store_true"); ap.add_argument("--supervisor-receipt"); ap.add_argument("--sha256")
    a = ap.parse_args()
    if a.post_run_probe:
        if not (a.supervisor_receipt and a.sha256): sys.exit("ABORT: --post-run-probe needs --supervisor-receipt and --sha256")
        sys.exit(post_run_probe(a.supervisor_receipt, a.sha256))
    for k in ("harness_commit", "bundle", "bundle_sha256", "protocol_sha256"):
        if not getattr(a, k): sys.exit(f"ABORT: --{k.replace('_', '-')} is required")
    sys.exit(Supervisor(a).main())
