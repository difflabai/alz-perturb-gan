"""Cancellation after probe launch must reach the final receipt, CPU only."""
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys

import pytest
from test_v46_kill_recheck_and_stop import CHAIN, SUP, _env, _wait


@pytest.mark.parametrize('window', ['probe_started', 'observation_written'])
@pytest.mark.parametrize('stop', [False, True])
def test_late_wrapper_term_finalizes_consistently(tmp_path, window, stop):
    root = tmp_path / 'scope'; study = root / 'study'; cache = root / 'cache'; dry = root / 'dry'
    ready, release = root / 'ready', root / 'release'
    for d in (study, cache, dry): d.mkdir(parents=True)
    (study / 'study_bundle_v2.pt').write_bytes(b'x')
    text = CHAIN.read_text(); name = re.search(r'^SUPERVISOR_NAME=(\S+)$', text, re.M).group(1)
    (study / name).write_text(
        'import os, sys, time, subprocess\nfrom pathlib import Path\n'
        f'REAL, READY, RELEASE, DRY, WINDOW = {str(SUP)!r}, {str(ready)!r}, {str(release)!r}, {str(dry)!r}, {window!r}\n'
        'def barrier():\n'
        '    Path(READY).touch()\n'
        '    end=time.monotonic()+30\n'
        '    while not Path(RELEASE).exists():\n'
        '        if time.monotonic()>end: raise RuntimeError("barrier timed out")\n'
        '        time.sleep(.01)\n'
        'if "--finalize-probe" in sys.argv: os.execv(sys.executable, [sys.executable, REAL]+sys.argv[1:])\n'
        'if "--post-run-probe" in sys.argv:\n'
        '    if WINDOW=="probe_started": barrier()\n'
        '    rc=subprocess.call([sys.executable, REAL]+sys.argv[1:])\n'
        '    if WINDOW=="observation_written": barrier()\n'
        '    sys.exit(rc)\n'
        'os.execv(sys.executable, [sys.executable, REAL, "--dry-run", "--dry-skip-static-checks", "--dry-dir", DRY,\n'
        '                         "--seeds", "42", "--time-scale", "0.01", "--dry-arm-seconds", "0.05"])\n')
    for k, v in (('SCOPE', root), ('S', study), ('PY', sys.executable), ('CACHE', cache)):
        text = re.sub(rf'^{k}=/ml2/altgan\S*$', f'{k}={v}', text, count=1, flags=re.M)
    sh = tmp_path / 'chain.sh'; sh.write_text(text.replace('@PINNED_COMMIT@', 'a'*40))
    w = subprocess.Popen(['bash', str(sh), '--dry-run'], stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                         text=True, start_new_session=True, env=_env(tmp_path))
    try:
        assert _wait(ready.exists, 30)
        if stop: os.kill(w.pid, signal.SIGTERM)  # only the wrapper, never a real training process
        release.touch()
        out, err = w.communicate(timeout=60)
    finally:
        if w.poll() is None: w.kill(); w.wait()
    assert 'supervisor rc=0' in out, (out, err)
    receipt = json.loads(next(dry.glob('dry-v4-*/post_run_receipt.json')).read_text())
    observation = json.loads(next(dry.glob('dry-v4-*/post_run_probe_observation.json')).read_text())
    assert 'verdict' not in observation and observation['publication'] == 'provisional'
    assert receipt['publication'] == 'final' and receipt['residue'] == 'clean'
    assert receipt['verdict'] == ('incomplete' if stop else 'clean')
    assert receipt['run_status'] == ('incomplete' if stop else 'complete')
    assert receipt['stop_requested'] == ('wrapper_sigterm' if stop else None)
    assert w.returncode == (8 if stop else 0), (out, err)
    assert ('CHAIN_DONE' in out) == (not stop)
    assert ('CHAIN_INCOMPLETE' in out) == stop
