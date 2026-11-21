"""Resume the public NeuroKO downloads and verify the Zenodo version checksums."""
from __future__ import annotations

import argparse
import fcntl
import hashlib
from pathlib import Path
import subprocess

RECORD = 23260389
FILES = {
    "iNeurons": (6_314_792_054, "cd4d1fbb316a66f6d4daebfd770acd8e"),
    "iAstrocytes": (7_465_978_066, "10abe464ca16bcaab07d644adcaae3ff"),
}
DEFAULT_DATA = Path(__file__).resolve().parents[1] / "data" / "neuroko"


def verify(path, size, md5):
    if path.stat().st_size != size:
        raise ValueError(f"{path} has {path.stat().st_size} bytes, expected {size}")
    with path.open("rb") as handle:
        actual = hashlib.file_digest(handle, "md5").hexdigest()
    if actual != md5:
        raise ValueError(f"Zenodo checksum mismatch for {path}: {actual}")


def fetch(directory, screen):
    size, md5 = FILES[screen]
    path = directory / f"{screen}.h5ad"
    partial = path.with_suffix(".h5ad.part")
    with path.with_suffix(".h5ad.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if path.exists():
            verify(path, size, md5)
            return path
        if not partial.exists() or partial.stat().st_size < size:
            url = f"https://zenodo.org/api/records/{RECORD}/files/{path.name}/content"
            subprocess.run([
                "curl", "--fail", "--location", "--retry", "4", "--continue-at", "-",
                "--output", str(partial), url,
            ], check=True)
        try:
            verify(partial, size, md5)
        except ValueError:
            if partial.stat().st_size >= size:
                partial.replace(partial.with_suffix(".corrupt"))
            raise
        partial.replace(path)
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--screen", choices=tuple(FILES), action="append")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    for screen in args.screen or FILES:
        print(f"Verified {fetch(args.out, screen)}", flush=True)


if __name__ == "__main__":
    main()
