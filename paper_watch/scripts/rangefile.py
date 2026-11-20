"""Read-only seekable file over HTTP Range requests (proxy-friendly).

Written on 2026-11-18, when plain GETs to ftp.ncbi.nlm.nih.gov returned
403 through that run's proxy while Range requests succeeded; requests
honours the proxy env where aiohttp/fsspec did not connect. On
2026-11-20 from pop-os, HEAD returns 200 and a range read returns 206,
so a local download plus scripts/extract_hypoad_microglia.py is the
working route. Block-cached, LRU.

Every block request sends If-Range with the validator recorded at open: a
strong ETag (the served one, or the ``etag`` the caller expects, for
example the one in a parallel_download receipt), else Last-Modified. A
weak ETag (W/...) is never sent as If-Range, because RFC 9110 compares
If-Range strongly and a compliant server would answer every block with
200; NCBI sends no ETag at all, only Last-Modified. A 200 instead of 206,
or a 206 with a different ETag or Last-Modified, means the object changed
under the reader; that raises RemoteChanged and is never retried, so
blocks of two versions are never mixed.
"""
from __future__ import annotations
import requests
from collections import OrderedDict

class RemoteChanged(IOError):
    """The remote object is not the version this reader was opened against."""


class RangeFile:
    def __init__(self, url: str, block: int = 4 * 1024 * 1024, max_blocks: int = 48,
                 etag: str | None = None):
        self.url = url
        self.block = block
        self.max_blocks = max_blocks
        self.cache: OrderedDict[int, bytes] = OrderedDict()
        self.pos = 0
        r = requests.get(url, headers={"Range": "bytes=0-0"}, timeout=60)
        cr = r.headers.get("Content-Range", "")
        total = cr.rsplit("/", 1)[-1] if "/" in cr else ""
        if r.status_code != 206 or not total.isdigit():
            raise IOError(
                f"{url}: range probe got HTTP {r.status_code} with Content-Range "
                f"{cr!r}; the server must answer a Range request with 206 and a "
                "known total size")
        self.size = int(total)
        served = r.headers.get("ETag")
        if etag is not None and served != etag:
            raise RemoteChanged(f"{url}: ETag is {served!r}, expected {etag!r}")
        self.etag = etag if etag is not None else served
        self.last_modified = r.headers.get("Last-Modified")
        if self.etag and not self.etag.startswith("W/"):
            self.if_range = self.etag
        else:
            self.if_range = self.last_modified
        self._sess = requests.Session()

    def _block(self, i: int) -> bytes:
        if i in self.cache:
            self.cache.move_to_end(i)
            return self.cache[i]
        lo = i * self.block
        hi = min(self.size, lo + self.block) - 1
        import time
        last = None
        for attempt in range(6):
            try:
                headers = {"Range": f"bytes={lo}-{hi}"}
                if self.if_range:
                    headers["If-Range"] = self.if_range
                r = self._sess.get(self.url, headers=headers, timeout=120)
                r.raise_for_status()
                served = r.headers.get("ETag")
                served_lm = r.headers.get("Last-Modified")
                if r.status_code != 206 or (self.etag and served and served != self.etag) or (
                        self.last_modified and served_lm and served_lm != self.last_modified):
                    raise RemoteChanged(
                        f"{self.url}: block {i} got HTTP {r.status_code} with ETag {served!r}, "
                        f"Last-Modified {served_lm!r} (opened at {self.etag!r}, "
                        f"{self.last_modified!r})")
                data = r.content
                if len(data) != hi - lo + 1:
                    raise IOError(f"short block {len(data)} != {hi - lo + 1}")
                break
            except RemoteChanged:
                raise
            except Exception as e:  # retry transient proxy drops
                last = e
                time.sleep(2 * (attempt + 1))
        else:
            raise last
        self.cache[i] = data
        while len(self.cache) > self.max_blocks:
            self.cache.popitem(last=False)
        return data

    def read(self, n: int = -1) -> bytes:
        if n is None or n < 0:
            n = self.size - self.pos
        out = bytearray()
        while n > 0 and self.pos < self.size:
            i = self.pos // self.block
            off = self.pos % self.block
            b = self._block(i)
            take = min(n, len(b) - off)
            out += b[off:off + take]
            self.pos += take
            n -= take
        return bytes(out)

    def readinto(self, buf) -> int:
        data = self.read(len(buf))
        buf[: len(data)] = data
        return len(data)

    def seek(self, offset: int, whence: int = 0) -> int:
        if whence == 0:
            self.pos = offset
        elif whence == 1:
            self.pos += offset
        else:
            self.pos = self.size + offset
        return self.pos

    def tell(self) -> int:
        return self.pos

    def seekable(self) -> bool:
        return True

    def readable(self) -> bool:
        return True

    def flush(self) -> None:
        pass

    def close(self) -> None:
        self._sess.close()
