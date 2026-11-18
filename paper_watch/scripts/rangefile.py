"""Read-only seekable file over HTTP Range requests (proxy-friendly).

Plain GETs to ftp.ncbi.nlm.nih.gov 403 through this environment's
proxy, but Range requests succeed; requests honours the proxy env
where aiohttp/fsspec do not connect. Block-cached, LRU.
"""
from __future__ import annotations
import requests
from collections import OrderedDict

class RangeFile:
    def __init__(self, url: str, block: int = 4 * 1024 * 1024, max_blocks: int = 48):
        self.url = url
        self.block = block
        self.max_blocks = max_blocks
        self.cache: OrderedDict[int, bytes] = OrderedDict()
        self.pos = 0
        r = requests.get(url, headers={"Range": "bytes=0-0"}, timeout=60)
        cr = r.headers.get("Content-Range", "")
        self.size = int(cr.rsplit("/", 1)[1])
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
                r = self._sess.get(self.url, headers={"Range": f"bytes={lo}-{hi}"}, timeout=120)
                r.raise_for_status()
                data = r.content
                if len(data) != hi - lo + 1:
                    raise IOError(f"short block {len(data)} != {hi - lo + 1}")
                break
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
