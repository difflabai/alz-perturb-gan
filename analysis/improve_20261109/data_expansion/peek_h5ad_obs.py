"""Read ONLY the obs column names (and optionally small categorical obs columns) of a remote
h5ad over HTTPS range requests, without downloading the file.

Hard byte budget (default 40 MB) so the "no download > 50 MB" rule is respected: the reader
raises once the budget is exhausted. Nothing is written to disk except the JSON summary printed
to stdout. Usage:
  python3 -I peek_h5ad_obs.py URL [col1,col2,...] [budget_MB]
For each requested column that is an AnnData categorical, prints category counts per donor
summary (categories + value counts over cells, and per-donor first value when 'donor_id' given).
"""
import io
import json
import sys
import urllib.request

import h5py
import numpy as np


class RangeFile(io.RawIOBase):
    def __init__(self, url, budget, block=1 << 16):
        self.url, self.budget, self.block, self.pos, self.used = url, budget, block, 0, 0
        req = urllib.request.Request(url, method="HEAD")
        with urllib.request.urlopen(req, timeout=60) as r:
            self.size = int(r.headers["Content-Length"])
        self.cache = {}

    def readable(self):
        return True

    def seekable(self):
        return True

    def seek(self, off, whence=0):
        self.pos = off if whence == 0 else (self.pos + off if whence == 1 else self.size + off)
        return self.pos

    def tell(self):
        return self.pos

    def _blk(self, i):
        if i not in self.cache:
            a = i * self.block
            b = min(self.size, a + self.block) - 1
            if self.used + (b - a + 1) > self.budget:
                raise RuntimeError(f"byte budget exhausted ({self.used} bytes read)")
            req = urllib.request.Request(self.url, headers={"Range": f"bytes={a}-{b}"})
            with urllib.request.urlopen(req, timeout=120) as r:
                data = r.read()
            self.used += len(data)
            self.cache[i] = data
        return self.cache[i]

    def readinto(self, buf):
        n = min(len(buf), self.size - self.pos)
        if n <= 0:
            return 0
        out, p = bytearray(), self.pos
        while len(out) < n:
            i, o = divmod(p, self.block)
            chunk = self._blk(i)[o:o + n - len(out)]
            out += chunk
            p += len(chunk)
        buf[:n] = out
        self.pos += n
        return n


def col(obs, name):
    g = obs[name]
    if isinstance(g, h5py.Group) and "categories" in g:
        cats = [c.decode() if isinstance(c, bytes) else str(c) for c in g["categories"][:]]
        codes = g["codes"][:]
        return np.array([cats[c] if c >= 0 else "NA" for c in codes], dtype=object)
    v = g[:]
    return np.array([x.decode() if isinstance(x, bytes) else x for x in v], dtype=object)


def main():
    url = sys.argv[1]
    cols = sys.argv[2].split(",") if len(sys.argv) > 2 and sys.argv[2] else []
    budget = int(float(sys.argv[3]) * 1e6) if len(sys.argv) > 3 else 40_000_000
    f = RangeFile(url, budget)
    res = {"url": url, "file_bytes": f.size}
    with h5py.File(io.BufferedReader(f, buffer_size=1 << 16), "r") as h:
        obs = h["obs"]
        res["n_obs"] = int(h["obs"][obs.attrs.get("_index", "_index")].shape[0]) if obs.attrs.get("_index", "_index") in obs else None
        res["obs_columns"] = sorted(obs.keys())
        res["layers"] = sorted(h["layers"].keys()) if "layers" in h else []
        res["raw"] = "raw" in h
        donors = col(obs, "donor_id") if "donor_id" in obs and cols else None
        res["columns"] = {}
        for c in cols:
            if c not in obs:
                res["columns"][c] = "absent"
                continue
            v = col(obs, c)
            u, n = np.unique(v.astype(str), return_counts=True)
            entry = {"cell_counts": dict(zip(u.tolist(), n.tolist()))}
            if donors is not None:
                dv = {}
                for d, x in zip(donors, v.astype(str)):
                    dv.setdefault(d, x)
                du, dn = np.unique(list(dv.values()), return_counts=True)
                entry["donor_counts"] = dict(zip(du.tolist(), dn.tolist()))
                entry["n_donors"] = len(dv)
            res["columns"][c] = entry
    res["bytes_read"] = f.used
    print(json.dumps(res, indent=1, default=str))


if __name__ == "__main__":
    main()
