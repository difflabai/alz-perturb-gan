"""Exclusive ALZ role protection, distinct from both shared physical GPU locks."""
import fcntl
import os
from pathlib import Path
import stat

NATIVE_ROLE_LOCK = Path('/mnt/nvme/sliders/locks/gpu-1-alz-training.lock')


class RoleLease:
    def __init__(self, path):
        self.path, self.fd = Path(path), None

    def __enter__(self):
        self.fd = os.open(self.path, os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW | os.O_CLOEXEC, 0o600)
        try:
            fcntl.flock(self.fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            self.check()
            return self
        except BaseException:
            os.close(self.fd)
            self.fd = None
            raise

    def check(self):
        if self.fd is None:
            raise RuntimeError('ALZ role lease is not held')
        opened, current = os.fstat(self.fd), self.path.lstat()
        if not stat.S_ISREG(current.st_mode) or opened.st_uid != os.getuid() or (opened.st_dev, opened.st_ino) != (current.st_dev, current.st_ino):
            raise RuntimeError('ALZ role lock identity changed')
        lines = Path(f'/proc/self/fdinfo/{self.fd}').read_text().splitlines()
        if not any(line.split()[2:5] == ['FLOCK', 'ADVISORY', 'WRITE'] for line in lines if line.startswith('lock:')):
            raise RuntimeError('ALZ exclusive role lease lost')

    def __exit__(self, *unused):
        if self.fd is not None:
            os.close(self.fd)
            self.fd = None
