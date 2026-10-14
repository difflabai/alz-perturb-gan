"""Scoped-path containment (plan v4 §7): canonicalize, check by path components, exact allowlist.

Raw prefix checks are not used. O_NOFOLLOW only protects the final component, so directory
components are canonicalized first and new files are created relative to a directory handle.
"""
from __future__ import annotations
import glob as _glob
import os
from pathlib import Path

DEFAULT_ROOT = '/ml2/altgan'


def _root():
    return Path(os.environ.get('ALZ_SCOPE_ROOT', DEFAULT_ROOT))
ALLOW_EXACT = frozenset({'/tmp/ember-pg-20261012-slots/gpu1-slot0.lock', '/tmp/ember-pg-20261012-slots/gpu1-slot1.lock',
                         '/proc/locks', '/usr/bin/nvidia-smi'})


class ScopeError(PermissionError):
    pass


def _allowed_proc(p: Path) -> bool:
    parts = p.parts  # /proc/<pid>/{stat,status}
    return len(parts) == 4 and parts[1] == 'proc' and parts[2].isdigit() and parts[3] in ('stat', 'status')


def _contained(r: Path, root: Path) -> bool:
    return r == root or root in r.parents


def scoped(p, *, root=None, must_exist=True):
    """Return the canonical path if it lies under root (component-wise) or is exactly allowlisted.

    must_exist=False is for new output files: the *existing parent directory* is canonicalized and
    contained, and the final name is validated as a single component.
    """
    if p is None or str(p).strip() == '':
        raise ScopeError('empty path')
    s = str(p)
    if s.startswith('~'):
        raise ScopeError('home-relative paths are refused')
    if not os.path.isabs(s):
        raise ScopeError(f'relative path refused: {s!r}')
    root_arg = root
    root = Path(root or _root()).resolve(strict=True)
    if s in ALLOW_EXACT or _allowed_proc(Path(s)):
        return Path(s)
    lexical = Path(os.path.normpath(s))  # refuse obvious escapes before touching the filesystem
    if not (_contained(lexical, root) or _contained(lexical, Path(os.path.normpath(str(root_arg or _root()))))):
        raise ScopeError(f'outside {root}: {s!r}')
    if must_exist:
        r = Path(s).resolve(strict=True)
    else:
        parent, name = os.path.split(s.rstrip('/') if s != '/' else s)
        check_name(name)
        r = Path(parent).resolve(strict=True) / name
        if r.is_symlink():
            raise ScopeError(f'final component is a symlink: {s!r}')
    if not _contained(r, root):
        raise ScopeError(f'outside {root}: {s!r} -> {r}')
    return r


def check_name(name):
    if not name or name in ('.', '..') or '/' in name or '\0' in name:
        raise ScopeError(f'invalid file name: {name!r}')
    return name


def scoped_glob(pattern, *, root=None, require_nonempty=True):
    """Glob whose pattern and every match are contained; an empty result is refused by default."""
    if not pattern or not os.path.isabs(pattern):
        raise ScopeError('absolute, non-empty glob pattern required')
    scoped(os.path.dirname(pattern.split('*')[0]) or '/', root=root)
    out = [scoped(m, root=root) for m in sorted(_glob.glob(pattern))]
    if require_nonempty and not out:
        raise ScopeError(f'glob matched nothing: {pattern!r}')
    return out


def create_new(directory, name, data: bytes, *, root=None, mode=0o444):
    """Create a new file exactly once, anchored on a directory handle (openat-style).

    The directory is canonicalized and contained first, then opened O_DIRECTORY|O_NOFOLLOW; the
    file is created with O_CREAT|O_EXCL|O_NOFOLLOW relative to that handle.
    """
    d = scoped(directory, root=root)
    check_name(name)
    dfd = os.open(d, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        fd = os.open(name, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, mode, dir_fd=dfd)
        try:
            view = memoryview(data)
            while view:
                view = view[os.write(fd, view):]
            os.fsync(fd)
        finally:
            os.close(fd)
    finally:
        os.close(dfd)
    return d / name


def scoped_new_dir(p, *, root=None):
    """Create a NEW directory (and missing parents) strictly under root; refuses an existing final dir.

    The nearest existing ancestor is canonicalized and contained first; each new component is a validated
    single name created with mkdir (never following a symlink, since it did not exist), then re-verified.
    """
    s = str(p) if p is not None else ''
    scoped_lex = Path(os.path.normpath(s)) if s and os.path.isabs(s) else None
    if scoped_lex is None: scoped(s, root=root)  # raises the right ScopeError for empty, tilde or relative input
    anc, todo = scoped_lex, []
    while not os.path.lexists(anc):
        todo.append(anc.name); anc = anc.parent
    base = scoped(str(anc), root=root)
    if not todo: raise FileExistsError(f'directory already exists: {s!r}')
    for name in reversed(todo):
        check_name(name); os.mkdir(base / name, 0o755); base = scoped(str(base / name), root=root)
    return base


def check_new_target(p, *, root=None):
    """Validate, WITHOUT creating anything, that `p` is a not-yet-existing path under root.

    Used before any work is done so that nothing (training included) runs for an out-of-scope output.
    """
    s = str(p) if p is not None else ''
    if not s or not os.path.isabs(s): scoped(s, root=root)
    lex = Path(os.path.normpath(s)); anc, todo = lex, []
    while not os.path.lexists(anc):
        todo.append(anc.name); anc = anc.parent
    base = scoped(str(anc), root=root)
    if not todo: raise FileExistsError(f'output already exists: {s!r}')
    for name in todo: check_name(name)
    return base.joinpath(*reversed(todo))
