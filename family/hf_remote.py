"""Read single BF16 tensors (or row ranges of them) from a Hugging Face repo with HTTP range requests.

Only the safetensors header and the bytes of the requested tensor are downloaded; results are cached as float32
.npy files in paths.CACHE, and the repo's index is fetched only when a tensor is not cached yet.
"""
import json, os, struct, sys, urllib.request
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paths import CACHE


def get(url, rng=None):
    req = urllib.request.Request(url, headers={'Range': f'bytes={rng[0]}-{rng[1] - 1}'} if rng else {})
    for a in range(5):
        try:
            with urllib.request.urlopen(req, timeout=300) as r: return r.read()
        except Exception:
            if a == 4: raise


class Remote:
    def __init__(self, repo):
        self.base = f'https://huggingface.co/{repo}/resolve/main/'
        self._where = None; self.h = {}
        self.tag = repo.replace('/', '_')

    @property
    def where(self):
        if self._where is None:
            self._where = json.loads(get(self.base + 'model.safetensors.index.json'))['weight_map']
        return self._where

    def get(self, name, r0=0, r1=None):
        path = os.path.join(CACHE, f'{self.tag}__{name}__{r0}_{r1}.npy')
        if os.path.exists(path): return np.load(path)
        fn = self.where[name]
        if fn not in self.h:
            n = struct.unpack('<Q', get(self.base + fn, (0, 8)))[0]; self.h[fn] = (json.loads(get(self.base + fn, (8, 8 + n))), 8 + n)
        hh, b = self.h[fn]; t = hh[name]; s, e = t['data_offsets']; shape = list(t['shape'])
        if r1 is not None or r0:
            rb = int(np.prod(shape[1:])) * 2; r1 = shape[0] if r1 is None else r1
            s, e = s + r0 * rb, s + r1 * rb; shape[0] = r1 - r0
        u = np.frombuffer(get(self.base + fn, (b + s, b + e)), dtype=np.uint16)
        x = (u.astype(np.uint32) << 16).view(np.float32).reshape(shape)
        os.makedirs(CACHE, exist_ok=True); np.save(path, x); return x
