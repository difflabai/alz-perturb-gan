"""Gate freeze for the EMPF confirmation analysis (analysis/coverage_empf_confirm_20261120).

PROTOCOL.md: "No read-out, threshold, fold or metric may be added after the first
new score exists". score.py and analyze.py are byte-identical copies of the files
the live confirmation queue hash-checks. gate_freeze.json records, from those
committed copies, the analysis pins and the GUARDS thresholds as of the first
score. A later `score.pin_analysis()` after an analyze.py or test_confirm.py edit
rewrites source_manifest.json silently; with this test it also fails CI and shows
up as a git diff of gate_freeze.json.

CPU only; reads only committed files (no runs/ data).
"""

from __future__ import annotations

import ast
import hashlib
import json
import shutil
from datetime import datetime
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
HERE = ROOT/'analysis/coverage_empf_confirm_20261120'
PREFIX = 'analysis/coverage_empf_confirm_20261120/'
ANALYSIS = ('analyze.py', 'test_confirm.py')
GUARD_FILES = ('score.py', 'analyze.py')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def extract_guards(path):
    """(value, source text) of the module-level GUARDS assignment, by ast; the value is
    evaluated with no names and no builtins (it is a dict display, possibly with **{...})."""
    text = Path(path).read_text()
    tree = ast.parse(text)
    found = [node for node in tree.body if isinstance(node, ast.Assign)
             and any(isinstance(t, ast.Name) and t.id == 'GUARDS' for t in node.targets)]
    if len(found) != 1:
        raise AssertionError(f'{path}: expected exactly one module-level GUARDS, found {len(found)}')
    node = found[0].value
    names = {n.id for n in ast.walk(node) if isinstance(n, ast.Name)}
    bound = {t.id for c in ast.walk(node) if isinstance(c, ast.comprehension) for t in ast.walk(c.target)
             if isinstance(t, ast.Name)}
    if names - bound:
        raise AssertionError(f'{path}: GUARDS refers to names {sorted(names - bound)}')
    value = eval(compile(ast.Expression(node), str(path), 'eval'), {'__builtins__': {}}, {})  # noqa: S307
    return {k: float(v) for k, v in value.items()}, ast.get_source_segment(text, found[0])


MUTATORS = {'update', 'setdefault', 'pop', 'popitem', 'clear', '__setitem__', '__delitem__', '__ior__'}


def guard_mutations(path):
    """Every place in ``path`` that could change GUARDS after its one display assignment:
    a second assignment or annotated assignment, an augmented assignment, a subscript or
    attribute store or delete, a mutating method call, `global GUARDS`, or a walrus. Thresholds
    written that way would bypass extract_guards(), which reads only the display."""
    tree = ast.parse(Path(path).read_text())
    found, assigns = [], 0

    def is_guards(node):
        return isinstance(node, ast.Name) and node.id == 'GUARDS'

    def stored(target):
        for t in ast.walk(target):
            if is_guards(t) and isinstance(t.ctx, (ast.Store, ast.Del)):
                yield 'GUARDS rebound'
            elif isinstance(t, (ast.Subscript, ast.Attribute)) and is_guards(t.value):
                yield 'GUARDS item or attribute stored or deleted'

    for node in ast.walk(tree):
        line = getattr(node, 'lineno', '?')
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if is_guards(target) and node in tree.body:
                    assigns += 1
                    continue
                found += [f'line {line}: {w}' for w in stored(target)]
        elif isinstance(node, (ast.AugAssign, ast.AnnAssign, ast.Delete, ast.For, ast.With, ast.NamedExpr)):
            targets = (node.targets if isinstance(node, ast.Delete) else
                       [i.optional_vars for i in node.items if i.optional_vars] if isinstance(node, ast.With) else
                       [node.target])
            for target in targets:
                found += [f'line {line}: {w}' for w in stored(target)]
        elif isinstance(node, ast.Global) and 'GUARDS' in node.names:
            found.append(f'line {line}: global GUARDS')
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and is_guards(node.func.value) \
                and node.func.attr in MUTATORS:
            found.append(f'line {line}: GUARDS.{node.func.attr}()')
    if assigns != 1:
        found.append(f'{assigns} module-level GUARDS assignments (expected 1)')
    return found


def check_freeze(directory):
    """Raise AssertionError when the committed sources disagree with the freeze or the pins."""
    directory = Path(directory)
    manifest = json.loads((directory/'source_manifest.json').read_text())
    freeze = json.loads((directory/'gate_freeze.json').read_text())
    problems = []
    pinned_score = manifest['execution_sources'].get(PREFIX + 'score.py')
    score_sha = sha(directory/'score.py')
    if score_sha != pinned_score:
        problems.append('score.py differs from its execution_sources pin')
    # score.py is frozen as a whole (thresholds outside GUARDS count too). A deliberate later
    # version (e.g. the post-drain two-step write) is listed in score_py_post_freeze with a reason.
    allowed = {freeze['score_py_sha256'], *(e['sha256'] for e in freeze.get('score_py_post_freeze', [])
                                             if e.get('reason'))}
    if score_sha not in allowed:
        problems.append('score.py differs from gate_freeze.json score_py_sha256 (edited after the first score)')
    analysis = manifest.get('analysis_sources', {})
    for name in ANALYSIS:
        digest = sha(directory/name)
        if analysis.get(PREFIX + name) != digest:
            problems.append(f'{name} differs from source_manifest.json analysis_sources')
        if freeze['analysis_sources'].get(PREFIX + name) != digest:
            problems.append(f'{name} differs from gate_freeze.json (edited after the first score)')
    if analysis != freeze['analysis_sources']:
        problems.append('source_manifest.json analysis_sources were re-pinned after the freeze')
    for name in GUARD_FILES:
        for where in guard_mutations(directory/name):
            problems.append(f'{name} changes GUARDS outside its frozen display: {where}')
        try:
            value, _ = extract_guards(directory/name)
        except AssertionError as exc:
            problems.append(f'{name} GUARDS cannot be read as one frozen display: {exc}')
            continue
        if value != freeze['guards'][name]:
            problems.append(f'{name} GUARDS differ from gate_freeze.json: {value} != {freeze["guards"][name]}')
    if freeze['guards']['score.py'] != freeze['guards']['analyze.py']:
        problems.append('score.py and analyze.py froze different GUARDS')
    if problems:
        raise AssertionError('; '.join(problems))


def build_freeze(directory, first_score_cutoff):
    """The gate_freeze.json document for the files in ``directory`` (used once, at freeze time)."""
    directory = Path(directory)
    manifest = json.loads((directory/'source_manifest.json').read_text())
    guards, sources = {}, {}
    for name in GUARD_FILES:
        guards[name], sources[name] = extract_guards(directory/name)
    return dict(
        schema='confirm_gate_freeze/1',
        purpose='PROTOCOL.md: no read-out, threshold, fold or metric may be added after the first new score '
                'exists. These values are frozen as of that score; tests/test_confirm_gate_freeze.py checks them.',
        analysis_sources={PREFIX + n: manifest['analysis_sources'][PREFIX + n] for n in ANALYSIS},
        score_py_sha256=manifest['execution_sources'][PREFIX + 'score.py'],
        guards=guards, guards_source=sources,
        first_score_cutoff=first_score_cutoff)


# --------------------------------------------------------------------------- the committed freeze
def test_committed_sources_match_the_manifest_pins():
    manifest = json.loads((HERE/'source_manifest.json').read_text())
    assert sha(HERE/'score.py') == manifest['execution_sources'][PREFIX + 'score.py']
    assert sha(HERE/'PROTOCOL.md') == manifest['execution_sources'][PREFIX + 'PROTOCOL.md']
    for name in ANALYSIS:
        assert sha(HERE/name) == manifest['analysis_sources'][PREFIX + name], name


def test_committed_freeze_holds():
    check_freeze(HERE)


def test_freeze_records_the_pilot_thresholds_and_cutoff():
    freeze = json.loads((HERE/'gate_freeze.json').read_text())
    expected = {'pb_rmse': .005, 'meanlog_rmse_gene': .02, 'nonzero_rmse_gene': .02, 'logvar_rmse_gene': .02,
                'pc_energy_rep': .02, 'cell_energy_rep': .02}
    assert freeze['guards'] == {'score.py': expected, 'analyze.py': expected}
    cutoff = freeze['first_score_cutoff']
    when = datetime.fromisoformat(cutoff['utc'])
    # after the protocol was written (03:30 UTC) and on the day of the confirmation
    assert datetime.fromisoformat('2026-11-20T03:30:00+00:00') < when < datetime.fromisoformat(
        '2026-11-21T00:00:00+00:00')
    assert cutoff['source']


# --------------------------------------------------------------------------- the freeze catches edits
@pytest.fixture
def copy(tmp_path):
    for name in ('score.py', 'analyze.py', 'test_confirm.py', 'source_manifest.json', 'gate_freeze.json'):
        shutil.copy2(HERE/name, tmp_path/name)
    check_freeze(tmp_path)
    return tmp_path


def repin(directory):
    """What score.pin_analysis() does: refresh analysis_sources to the current files."""
    path = Path(directory)/'source_manifest.json'
    manifest = json.loads(path.read_text())
    manifest['analysis_sources'] = {PREFIX + n: sha(Path(directory)/n) for n in ANALYSIS}
    path.write_text(json.dumps(manifest, indent=2) + '\n')


@pytest.mark.parametrize('name', GUARD_FILES)
def test_modified_guards_fail(copy, name):
    path = copy/name
    text = path.read_text()
    assert "'pb_rmse': .005" in text
    path.write_text(text.replace("'pb_rmse': .005", "'pb_rmse': .006", 1))
    with pytest.raises(AssertionError, match='GUARDS differ'):
        check_freeze(copy)
    repin(copy)  # a post-result pin_analysis() does not hide it
    with pytest.raises(AssertionError, match='GUARDS differ'):
        check_freeze(copy)


def test_post_result_pin_analysis_fails(copy):
    path = copy/'test_confirm.py'
    path.write_text(path.read_text() + '\n# a post-result edit\n')
    repin(copy)
    with pytest.raises(AssertionError, match='edited after the first score|re-pinned after the freeze'):
        check_freeze(copy)


def test_guards_extraction_refuses_names(tmp_path):
    (tmp_path/'m.py').write_text("LIMIT = .02\nGUARDS = {'pb_rmse': LIMIT}\n")
    with pytest.raises(AssertionError, match='refers to names'):
        extract_guards(tmp_path/'m.py')
    (tmp_path/'n.py').write_text("GUARDS = {'pb_rmse': .005, **{k: .02 for k in ('a', 'b')}}\n")
    assert extract_guards(tmp_path/'n.py')[0] == {'pb_rmse': .005, 'a': .02, 'b': .02}


# --------------------------------------------------------------------------- bypasses (adversarial review)
def repin_score(directory):
    """What a manifest refreeze does to execution_sources after a score.py edit."""
    path = Path(directory)/'source_manifest.json'
    manifest = json.loads(path.read_text())
    manifest['execution_sources'][PREFIX + 'score.py'] = sha(Path(directory)/'score.py')
    path.write_text(json.dumps(manifest, indent=2) + '\n')


def _after_guards(text, line):
    """Insert ``line`` right after the (multi-line) GUARDS display."""
    tree = ast.parse(text)
    node = next(n for n in tree.body if isinstance(n, ast.Assign) and any(
        isinstance(t, ast.Name) and t.id == 'GUARDS' for t in n.targets))
    lines = text.splitlines(keepends=True)
    return ''.join(lines[:node.end_lineno]) + line + '\n' + ''.join(lines[node.end_lineno:])


@pytest.mark.parametrize('line', [
    "GUARDS['pb_rmse'] = 0.5  # loosened after results",
    "GUARDS.update(pb_rmse=0.5)",
    "GUARDS |= {'pb_rmse': 0.5}",
    "GUARDS = {**GUARDS, 'pb_rmse': 0.5}",
    "del GUARDS['pc_energy_rep']",
    "def _loosen():\n    global GUARDS\n    GUARDS = {}",
])
@pytest.mark.parametrize('name', GUARD_FILES)
def test_guards_overridden_after_the_display_fail(copy, name, line):
    """t8 repro: GUARDS display untouched, value overridden later in the module, then the
    execution or analysis pins refreshed; check_freeze used to pass."""
    path = copy/name
    path.write_text(_after_guards(path.read_text(), line))
    repin_score(copy)
    repin(copy)
    with pytest.raises(AssertionError, match='outside its frozen display'):
        check_freeze(copy)


def test_score_py_edit_outside_guards_fails_even_after_a_refreeze(copy):
    path = copy/'score.py'
    path.write_text(path.read_text() + '\nMIN_CELLS = 1  # a threshold outside GUARDS\n')
    repin_score(copy)
    with pytest.raises(AssertionError, match='score_py_sha256'):
        check_freeze(copy)
    # a deliberate post-freeze version is accepted only when listed with a reason
    freeze_path = copy/'gate_freeze.json'
    freeze = json.loads(freeze_path.read_text())
    freeze['score_py_post_freeze'] = [dict(sha256=sha(path), reason='')]
    freeze_path.write_text(json.dumps(freeze))
    with pytest.raises(AssertionError, match='score_py_sha256'):
        check_freeze(copy)
    freeze['score_py_post_freeze'] = [dict(sha256=sha(path), reason='post-drain two-step write (test)')]
    freeze_path.write_text(json.dumps(freeze))
    check_freeze(copy)


def test_committed_sources_have_no_guard_mutations():
    for name in GUARD_FILES:
        assert guard_mutations(HERE/name) == [], name
