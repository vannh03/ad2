import zlib
try:
    from functools import cached_property
except ImportError:
    class cached_property(object):
        def __init__(self, func):
            self.func = func
            self.attrname = func.__name__
            self.__doc__ = func.__doc__

        def __get__(self, instance, owner=None):
            if instance is None:
                return self
            val = self.func(instance)
            instance.__dict__[self.attrname] = val
            return val
from pathlib import Path

class GitRepo:
    """Represent a local Git repository and expose branch, commit, and remote metadata.

    This class discovers the repository root by searching for a .git entry from the given path upward, resolves the
    actual .git directory (including worktrees), and reads Git metadata directly from on-disk files. It does not invoke
    the git binary and therefore works in restricted environments. All metadata properties are resolved lazily and
    cached; construct a new instance to refresh state.

    Attributes:
        root (Path | None): Repository root directory containing the .git entry; None if not in a repository.
        gitdir (Path | None): Resolved .git directory path; handles worktrees; None if unresolved.
        refdir (Path | None): Directory containing shared refs, objects, and config; None if unresolved.
        head (str | None): Raw contents of HEAD; a SHA for detached HEAD or "ref: <refname>" for branch heads.
        is_repo (bool): Whether the provided path resides inside a Git repository.
        branch (str | None): Current branch name when HEAD points to a branch; None for detached HEAD or non-repo.
        commit (str | None): Current commit SHA for HEAD; None if not determinable.
        message (str | None): Current commit subject from a loose object; None if not determinable.
        origin (str | None): URL of the "origin" remote as read from gitdir/config; None if unset or unavailable.

    Examples:
        Initialize from the current working directory and read metadata
        >>> from pathlib import Path
        >>> repo = GitRepo(Path.cwd())
        >>> is_repo = repo.is_repo
        >>> branch, commit, origin = repo.branch, repo.commit, repo.origin

    Notes:
        - Resolves metadata by reading files: HEAD, packed-refs, config, and objects; no subprocess calls are used.
        - Caches properties on first access using cached_property; recreate the object to reflect repository changes.
    """

    def __init__(self, path=None):
        """Initialize a Git repository context by discovering the repository root from a starting path.

        Args:
            path (Path, optional): File or directory path used as the starting point to locate the repository root.
        """
        self.root = self._find_root(path or Path(__file__).resolve())
        self.gitdir = self._gitdir(self.root) if self.root else None
        self.refdir = self._refdir(self.gitdir)

    @staticmethod
    def _find_root(p):
        """Return repo root or None."""
        return next((d for d in [p, *list(p.parents)] if (d / '.git').exists()), None)

    @staticmethod
    def _gitdir(root):
        """Resolve actual .git directory (handles worktrees)."""
        g = root / '.git'
        if g.is_dir():
            return g
        if g.is_file():
            t = g.read_text(errors='ignore').strip()
            if t.startswith('gitdir:'):
                return (root / t.split(':', 1)[1].strip()).resolve()
        return None

    @staticmethod
    def _refdir(gitdir):
        """Resolve directory containing refs, objects, and config."""
        p = gitdir / 'commondir' if gitdir else None
        s = GitRepo._read(p)
        if s:
            d = Path(s)
            return (gitdir / d).resolve() if not d.is_absolute() else d
        return gitdir

    @staticmethod
    def _read(p):
        """Read and strip file if exists."""
        return p.read_text(errors='ignore').strip() if p and p.exists() else None

    @cached_property
    def head(self):
        """HEAD file contents."""
        return self._read(self.gitdir / 'HEAD' if self.gitdir else None)

    def _ref_commit(self, ref):
        """Commit for ref (handles packed-refs)."""
        rf = self.refdir / ref
        s = self._read(rf)
        if s:
            return s
        pf = self.refdir / 'packed-refs'
        b = pf.read_bytes().splitlines() if pf.exists() else []
        tgt = ref.encode()
        for line in b:
            if line[:1] in (b'#', b'^') or b' ' not in line:
                continue
            (sha, name) = line.split(b' ', 1)
            if name.strip() == tgt:
                return sha.decode()
        return None

    def _commit_subject(self, commit):
        """Commit subject from loose object or None."""
        obj = self.refdir / 'objects' / commit[:2] / commit[2:]
        if not obj.exists():
            return None
        data = zlib.decompress(obj.read_bytes())
        if b'\x00' not in data:
            return None
        (kind, body) = data.split(b'\x00', 1)
        if not kind.startswith(b'commit ') or b'\n\n' not in body:
            return None
        subject = body.split(b'\n\n', 1)[1].splitlines()[0].decode(errors='replace').strip()
        return subject or None

    @property
    def is_repo(self):
        """True if inside a git repo."""
        return self.gitdir is not None

    @cached_property
    def branch(self):
        """Current branch or None."""
        if not self.is_repo or not self.head or (not self.head.startswith('ref: ')):
            return None
        ref = self.head[5:].strip()
        return ref[11:] if ref.startswith('refs/heads/') else ref

    @cached_property
    def commit(self):
        """Current commit SHA or None."""
        if not self.is_repo or not self.head:
            return None
        return self._ref_commit(self.head[5:].strip()) if self.head.startswith('ref: ') else self.head

    @cached_property
    def message(self):
        """Current commit subject or None."""
        if not self.is_repo or not self.commit:
            return None
        return self._commit_subject(self.commit)

    @cached_property
    def origin(self):
        """Origin URL or None."""
        if not self.is_repo:
            return None
        cfg = self.refdir / 'config'
        (remote, url) = (None, None)
        for s in (self._read(cfg) or '').splitlines():
            t = s.strip()
            if t.startswith('[') and t.endswith(']'):
                remote = t.lower()
            elif t.lower().startswith('url =') and remote == '[remote "origin"]':
                url = t.split('=', 1)[1].strip()
                break
        return url
if __name__ == '__main__':
    import time
    g = GitRepo()
    if g.is_repo:
        t0 = time.perf_counter()
        print(f'repo={g.root}\nbranch={g.branch}\ncommit={g.commit}\nmessage={g.message}\norigin={g.origin}')
        dt = (time.perf_counter() - t0) * 1000
        print(f'\n⏱️ Profiling: total {dt:.3f} ms')