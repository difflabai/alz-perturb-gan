"""Lint: every absolute path literal in harness code is under the scope root or exactly allowlisted;
v4 shell scripts use strict mode and `--` on ls/cd/rm, and never reference home directories."""
import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
ALLOWED_PREFIXES = ('/ml2/altgan', '/proc/locks', '/proc/vmstat', '/proc/<', '/proc/{', '/usr/bin/nvidia-smi', '/dev/null',
                    '/tmp/ember-pg-20261012-slots/gpu1-slot0.lock', '/tmp/ember-pg-20261012-slots/gpu1-slot1.lock')
LIT = re.compile(r"""['"](/[A-Za-z~][^'"\s]*)""")
SOURCES = sorted((REPO / 'src').rglob('*.py')) + sorted((REPO / 'study').glob('*v[24]*.py'))
SHELLS = sorted((REPO / 'study').glob('*v4*.sh'))


def test_sources_found():
    assert any(p.name == 'paths.py' for p in SOURCES)


def test_python_path_literals_are_scoped():
    bad = [f'{p.relative_to(REPO)}: {m}' for p in SOURCES for m in LIT.findall(p.read_text())
           if not (m == '/proc' or m.startswith(ALLOWED_PREFIXES))]
    assert not bad, bad


def test_no_home_references():
    pat = re.compile(r'(/home/|(?<![\w.])~/|\$HOME|expanduser)')
    bad = [str(p.relative_to(REPO)) for p in SOURCES + SHELLS if p.name != 'test_path_hygiene.py' and pat.search(p.read_text())]
    assert not bad, bad


def test_shell_guards():
    for p in SHELLS:
        text = p.read_text()
        assert 'set -euo pipefail' in text, p
        assert 'shopt -s nullglob' in text or '*' not in text, p
        for n, line in enumerate(text.splitlines(), 1):
            code = line.split('#', 1)[0]
            for cmd in re.findall(r'(?:^|[;&|(]\s*|\bthen\s+|\bdo\s+)(ls|cd|rm)\b([^;&|]*)', code):
                assert ' -- ' in cmd[1] + ' ', f'{p.name}:{n}: {cmd[0]} without --'
            assert '/tmp' not in code or 'ember-pg-20261012-slots' in code, f'{p.name}:{n}: /tmp'
