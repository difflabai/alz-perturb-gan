"""Source-epoch lease, collision-free manifest archives, STRANDED audit_under_epoch and review records.

(d) a runtime process holds the shared lease; refreeze refuses and lists it; after the holder is killed
    the refreeze succeeds; two refreezes in one second get distinct archive names (old names still
    readable); a fit stranded by a manifest change during execution (the train_trap repro) is audited
    under its own epoch from a git worktree at the matching commit, and refused at another digest
(e) check_review_records flags the known stale digests of scripts/microglia*.py and never writes
CPU only; no data, GPU or torch.
"""
import datetime
import importlib.util
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import textwrap

import pytest

ROOT = Path(__file__).resolve().parents[1]
TESTS = Path(__file__).resolve().parent
if str(TESTS) not in sys.path:
    sys.path.insert(0, str(TESTS))

import test_microglia_output_state as H  # noqa: E402  (shared stub host)

R, M, ds, Fit = H.R, H.M, H.ds, H.R.Fit


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


F = _load('refreeze_runtime_manifest_epoch', ROOT/'scripts/refreeze_runtime_manifest.py')
C = _load('check_review_records_epoch', ROOT/'scripts/check_review_records.py')
NOW = datetime.datetime(2026, 11, 20, 12, tzinfo=datetime.timezone.utc)

HOLDER = textwrap.dedent('''
    import importlib.util, sys, time
    spec = importlib.util.spec_from_file_location('rt', sys.argv[1])
    R = importlib.util.module_from_spec(spec); spec.loader.exec_module(R)
    lease = R.EpochLease(['stub fit holder'], directory=sys.argv[2]).acquire()
    lease.update(digest='d'*64, fit='/stub/fit')
    print('held', flush=True)
    time.sleep(600)
''')


def make_tree(tmp_path):
    root = tmp_path/'repo'
    (root/'pkg').mkdir(parents=True)
    (root/'prev').mkdir()
    (root/'pkg/a.py').write_text('a = 1\n')
    (root/'prev/old.py').write_text('old entrypoint\n')
    (root/'m').mkdir()
    manifest = root/'m/manifest.json'
    manifest.write_text(json.dumps({'schema': 1, 'sources': {'pkg/a.py': F.sha(root/'pkg/a.py')},
        'archived_previous_entrypoint': {'archive': 'prev/old.py', 'sha256': F.sha(root/'prev/old.py')}}, indent=2))
    return root, manifest


def start_holder(epoch):
    proc = subprocess.Popen([sys.executable, '-B', '-c', HOLDER, str(ROOT/'scripts/microglia_runtime.py'), str(epoch)],
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    assert proc.stdout.readline().strip() == 'held', proc.stderr.read()
    return proc


# --------------------------------------------------------------------------------------------------
# (d) lease and refreeze

def test_refreeze_refuses_while_a_runtime_process_holds_the_lease(tmp_path):
    root, manifest = make_tree(tmp_path)
    epoch = tmp_path/'epoch'
    (root/'pkg/a.py').write_text('a = 2\n')
    before = manifest.read_bytes()
    holder = start_holder(epoch)
    try:
        with pytest.raises(F.EpochBusy) as busy:
            F.refreeze(manifest, root, 'during a fit', now=NOW, epoch_dir=epoch)
        assert [h['pid'] for h in busy.value.holders] == [holder.pid]
        assert busy.value.holders[0]['fit'] == '/stub/fit' and busy.value.holders[0]['digest'] == 'd'*64
        assert f'pid {holder.pid}' in str(busy.value) and '--wait' in str(busy.value)
        assert manifest.read_bytes() == before and not (root/'m/previous').exists()
        # The CLI form exits 75 (busy) and lists the holder.
        assert [h['pid'] for h in R.live_holders(epoch)] == [holder.pid]
    finally:
        holder.send_signal(signal.SIGKILL)
        holder.wait()
    # The kernel lock died with the holder; its record is stale and is swept by the exclusive holder.
    assert R.live_holders(epoch) == []
    result = F.refreeze(manifest, root, 'after the fit', now=NOW, epoch_dir=epoch)
    assert result['changed'] == ['pkg/a.py'] and list((epoch/'holders').iterdir()) == []


def test_refreeze_cli_exits_75_with_live_holders(tmp_path):
    epoch = tmp_path/'epoch'
    holder = start_holder(epoch)
    try:
        out = subprocess.run([sys.executable, '-B', str(ROOT/'scripts/refreeze_runtime_manifest.py'),
                              '--reason', 'x', '--epoch-dir', str(epoch), '--manifest', str(tmp_path/'missing.json')],
                             capture_output=True, text=True, timeout=60)
        assert out.returncode == F.EX_BUSY and f'pid {holder.pid}' in out.stderr, out.stderr
    finally:
        holder.send_signal(signal.SIGKILL)
        holder.wait()


def test_runtime_holds_the_lease_for_its_lifetime_and_releases_it(tmp_path, monkeypatch):
    work = H.prepare(tmp_path/'w')
    stub = H.install(monkeypatch.setattr, work, H.Stub())
    epoch = Path(os.environ[R.EPOCH_DIR_ENV])
    assert R.main(H.gen_argv(work)) == 0
    (seen,) = stub.observed_holders
    assert [h['pid'] for h in seen] == [os.getpid()]
    assert seen[0]['digest'] == R.sha(work/'manifest.json') and seen[0]['fit'] == str((work/'fit').resolve())
    assert seen[0]['cmd'][:2] == ['microglia_runtime.py', 'generate']
    assert R.live_holders(epoch) == [] and list((epoch/'holders').iterdir()) == []
    lock = F.exclusive_epoch(epoch)  # nothing holds it any more
    lock.release()


def test_runtime_exits_75_while_a_refreeze_holds_the_epoch(tmp_path, monkeypatch):
    work = H.prepare(tmp_path/'w')
    stub = H.install(monkeypatch.setattr, work, H.Stub())
    monkeypatch.setenv(R.EPOCH_WAIT_ENV, '0.2')
    lock = F.exclusive_epoch(Path(os.environ[R.EPOCH_DIR_ENV]))
    try:
        assert R.main(H.gen_argv(work)) == R.EX_TEMPFAIL
    finally:
        lock.release()
    assert stub.activated == 0 and not H.out_of(work).exists()
    assert R.main(H.gen_argv(work)) == 0


def test_two_refreezes_in_one_second_get_distinct_archives(tmp_path):
    root, manifest = make_tree(tmp_path)
    epoch = tmp_path/'epoch'
    first = manifest.read_bytes()
    (root/'pkg/a.py').write_text('a = 2\n')
    one = F.refreeze(manifest, root, 'first', now=NOW, epoch_dir=epoch)
    second = manifest.read_bytes()
    (root/'pkg/a.py').write_text('a = 3\n')
    two = F.refreeze(manifest, root, 'second', now=NOW, epoch_dir=epoch)
    assert one['archive'] != two['archive']
    assert Path(one['archive']).read_bytes() == first and Path(two['archive']).read_bytes() == second
    assert Path(one['archive']).name == f'runtime_source_manifest.20261120T120000Z-{F.sha(Path(one["archive"]))[:12]}.json'
    # An archive with the old <UTC>.json name stays readable, and digest lookups scan both forms.
    legacy = root/'m/previous/runtime_source_manifest.20261119T000000Z.json'
    legacy.write_bytes(b'{"schema": 1, "sources": {}}\n')
    previous = root/'m/previous'
    for path in (legacy, Path(one['archive']), Path(two['archive'])):
        assert F.find_archive(previous, F.sha(path)) == path
        assert R.find_manifest_archive(F.sha(path), previous) == path
    assert len(R.manifest_archives(previous)) == 3


def test_repo_archives_keep_both_name_forms_readable():
    previous = ROOT/'analysis/default_empf_20261117/previous'
    names = [p.name for p in R.manifest_archives(previous)]
    assert any(len(n.split('.')[1]) == len('20261120T040256Z') for n in names)  # old <UTC>.json form
    manifest = json.loads(R.SOURCE_MANIFEST.read_text())
    archived = manifest['previous_runtime_manifest']
    assert R.find_manifest_archive(archived['sha256'], previous) == ROOT/archived['archive']


# --------------------------------------------------------------------------------------------------
# (d) STRANDED -audit_under_epoch-> AUDITED (the train_trap repro, recoverable without hand edits)

def git(repo, *args):
    return subprocess.run(['git', '-C', str(repo), *args], check=True, capture_output=True, text=True).stdout.strip()


@pytest.fixture
def epoch_repo(tmp_path):
    repo = tmp_path/'checkout'
    repo.mkdir()
    git(repo, 'init', '-q', '-b', 'main')
    git(repo, 'config', 'user.email', 'test@example.invalid')
    git(repo, 'config', 'user.name', 'test')
    manifest = repo/'runtime_source_manifest.json'
    manifest.write_text('{"sources": {"epoch": "A"}}\n')
    git(repo, 'add', '-A')
    git(repo, 'commit', '-q', '-m', 'epoch A')
    return repo, manifest, git(repo, 'rev-parse', 'HEAD')


def train(work, root, *extra):
    return R.cli(['train', '--cache', str(work/'cache'), '--out-root', str(root), *extra])


def test_stranded_fit_is_audited_under_its_own_epoch(tmp_path, monkeypatch, epoch_repo, capsys):
    repo, manifest, commit_a = epoch_repo
    digest_a = R.sha(manifest)
    work = H.prepare(tmp_path/'w')

    def refreeze_and_merge():  # the train_trap: a re-pin lands while the 20k-update fit runs
        manifest.write_text('{"sources": {"epoch": "B"}}\n')
        git(repo, 'commit', '-q', '-am', 'epoch B')
    stub = H.install(monkeypatch.setattr, work, H.Stub(during_training=refreeze_and_merge), manifest=manifest)
    root = tmp_path/'fits'
    fit = root/'pbulk/hvg2000/fold0/seed0'
    assert train(work, root) == R.EX_STRANDED
    err = capsys.readouterr().err
    snap = R.fit_state_file(fit).read()
    assert snap.state is Fit.STRANDED and 'manifest changed during execution' in snap.data['stranded_reason']
    assert snap.data['stranded_epoch'] == digest_a and snap.data['stranded_from'] == 'RECEIPTED'
    assert f'worktree add /tmp/microglia-epoch-{digest_a[:12]} {commit_a}' in err
    assert '--audit-under-epoch' in err and R.find_epoch_commit(digest_a, manifest) == (repo.resolve(), commit_a)
    assert not (fit/'no_skip_audit.json').exists()

    # Under the new epoch B: still stranded (exit 4, with the commands), and audit_under_epoch is refused.
    stub.during_training = None
    assert train(work, root) == R.EX_STRANDED
    assert commit_a in capsys.readouterr().err
    assert train(work, root, '--audit-under-epoch') == R.EX_STRANDED
    assert 'needs a checkout of that epoch' in capsys.readouterr().err
    assert R.fit_state_file(fit).read().state is Fit.STRANDED
    activated = stub.activated

    # From a git worktree at the commit whose manifest digest is the fit's epoch: AUDITED.
    wt = tmp_path/'epoch-a'
    git(repo, 'worktree', 'add', '-q', str(wt), commit_a)
    monkeypatch.setattr(R, 'SOURCE_MANIFEST', wt/'runtime_source_manifest.json')
    assert R.sha(R.SOURCE_MANIFEST) == digest_a
    assert train(work, root, '--audit-under-epoch') == 0
    snap = R.fit_state_file(fit).read()
    assert snap.state is Fit.AUDITED and snap.data['audited_epoch'] == digest_a
    assert [e['event'] for e in R.fit_state_file(fit).journal()][-2:] == ['epoch_mismatch', 'audit_under_epoch']
    audit = json.loads((fit/'no_skip_audit.json').read_text())
    assert audit['application_source_manifest_sha256'] == digest_a
    assert audit['audited_under_epoch']['epoch'] == digest_a
    assert stub.activated == activated + 1

    # Back in the main checkout (epoch B) the audited fit is simply done.
    monkeypatch.setattr(R, 'SOURCE_MANIFEST', manifest)
    assert train(work, root) == R.ALREADY_DONE


def test_audit_under_epoch_is_only_for_stranded_fits(tmp_path, monkeypatch):
    work = H.prepare(tmp_path/'w')
    H.install(monkeypatch.setattr, work, H.Stub())
    root = tmp_path/'fits'
    with pytest.raises(ValueError, match='not STRANDED'):
        R.main(['train', '--cache', str(work/'cache'), '--out-root', str(root), '--audit-under-epoch'])
    with pytest.raises(SystemExit) as usage:
        R.main(['train', '--cache', str(work/'cache'), '--out-root', str(root), '--audit-under-epoch', '--adopt-epoch'])
    assert usage.value.code == 2


def test_epoch_without_a_commit_points_to_adopt_epoch(tmp_path):
    args = type('A', (), {'cache': tmp_path, 'out_root': tmp_path, 'seed': 0, 'device': 'cpu'})()
    hint = R.epoch_recovery_hint(tmp_path/'fit', 'e'*64, args)
    assert 'no commit' in hint and '--adopt-epoch' in hint


# --------------------------------------------------------------------------------------------------
# (e) review records

def test_review_records_flag_the_stale_application_sources():
    records = [ROOT/name for name in C.RECORDS]
    before = {p: p.read_bytes() for p in records}
    rows = C.check(ROOT)
    stale = {r['path'] for r in rows if r['status'] == 'STALE'}
    assert {'scripts/microglia.py', 'scripts/microglia_runtime.py',
            'analysis/default_empf_20261117/runtime_source_manifest.json'} <= stale
    by_path = {r['path']: r for r in rows}
    assert by_path['configs/microglia_default.json']['status'] == 'OK'
    assert by_path['analysis/default_empf_20261117/selected_model.json']['status'] == 'OK'
    assert {p: p.read_bytes() for p in records} == before  # never mutated
    summary = C.summary(rows)
    assert not summary['ok'] and 'scripts/microglia.py' in summary['stale']
    assert C.main([]) == 0 and C.main(['--strict']) == 1


def test_review_records_report_only_what_changed(tmp_path):
    root = tmp_path/'r'
    for name in (*C.RECORDS, 'scripts/microglia.py', 'scripts/microglia_runtime.py', 'configs/microglia_default.json',
                 'tests/test_microglia_runtime.py', C.MANIFEST, 'analysis/default_empf_20261117/selection.json',
                 'analysis/default_empf_20261117/selected_model.json',
                 'analysis/default_empf_20261117/application_generation.receipt.json'):
        (root/name).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT/name, root/name)
    review = json.loads((root/C.RECORDS[0]).read_text())
    for name, digest in review['reviewed_application_source_sha256'].items():  # restore the reviewed bytes' digests
        if C._sha(root/name) != digest:
            (root/name).write_text(f'stand-in for {name}\n')
            review['reviewed_application_source_sha256'][name] = C._sha(root/name)
    review['runtime_source_manifest_sha256'] = C._sha(root/C.MANIFEST)
    (root/C.RECORDS[0]).write_text(json.dumps(review))
    rows = [r for r in C.check(root, records=C.RECORDS[:1]) if r['status'] != 'ABSENT']
    assert rows and [r['path'] for r in rows if r['status'] == 'STALE'] == []
    (root/'scripts/microglia.py').write_text('edited after the review\n')
    stale = {r['path'] for r in C.check(root, records=C.RECORDS[:1]) if r['status'] == 'STALE'}
    assert 'scripts/microglia.py' in stale


def test_dry_run_preflight_reports_review_records(capsys):
    assert M.main(['generate', '--dry-run']) == 0
    plan = json.loads(capsys.readouterr().out)
    review = plan['preflight']['review_records']
    assert review['ok'] is False and 'scripts/microglia.py' in review['stale']
    assert plan['preflight']['output_state'] in ('NONE', 'COMMITTED', 'INCONSISTENT', 'ORPHAN_RECEIPT',
                                                 'UNRECEIPTED_NPZ', 'STAGING', 'STAGED', 'PROMOTING')
