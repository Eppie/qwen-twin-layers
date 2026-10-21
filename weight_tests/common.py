"""Shared helpers for the weight-only twin-layer tests in this directory (see the README).

Model files: the three Q8_0 GGUFs under their Hugging Face file names, in the directory given by the
environment variable GGUF_DIR (default: the current directory). GGUF_32B, GGUF_27B and GGUF_MOE override the
path of a single model. Output (logs, CSVs, intermediate .npz arrays) goes to TWIN_OUT (default:
results/weight_tests in the repository).

GGUF access: the tensor index comes from gguf-py's GGUFReader (pip install gguf); tensor bytes are read with
os.pread on a descriptor opened with F_NOCACHE where available (no page-cache churn on a shared machine) and
Q8_0 is dequantised here in chunks. Arrays come back in storage order: shape = reversed GGUF ne, i.e. a 2-D
tensor with ne = [n_in, n_out] is returned as (n_out, n_in), one row per storage row.

Layout facts used throughout (checked against the GGUF shapes):
  ffn_gate / ffn_up  ne [n_embd, n_ff]  -> (n_ff, n_embd): row j  = neuron j's input vector
  ffn_down           ne [n_ff, n_embd]  -> (n_embd, n_ff): column j = neuron j's output vector
  ffn_{gate,up}_exps ne [d, 512, 256]   -> (256, 512, d):  [e, n, :] = neuron n of expert e
  ffn_down_exps      ne [512, d, 256]   -> (256, d, 512):  [e, :, n] = neuron n of expert e (a column)
  ffn_gate_inp       ne [d, 256] F32    -> (256, d):       row e = router row of expert e
"""
import fcntl
import os
import zlib

import numpy as np
from gguf import GGUFReader, GGMLQuantizationType

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.environ.get('TWIN_OUT') or os.path.join(os.path.dirname(HERE), 'results', 'weight_tests')
LOGS = RES
os.makedirs(RES, exist_ok=True)
GGUF_DIR = os.environ.get('GGUF_DIR', '.')


def _model_path(env, filename):
    return os.environ.get(env) or os.path.join(GGUF_DIR, filename)


MODELS = {
    '32b': dict(
        name='Qwen3-32B',
        path=_model_path('GGUF_32B', 'Qwen3-32B-Q8_0.gguf'),
        n_layers=64, blocks=[list(range(8, 32)), list(range(32, 56))], hybrid=False,
        rope_dims=128, head_dim=128),
    '27b': dict(
        name='Qwen3.8-27B',
        path=_model_path('GGUF_27B', 'Qwen3.8-27B-Q8_0.gguf'),
        n_layers=64,  # decoder layers 0..63; blk.64 is the MTP block and is excluded everywhere
        blocks=[list(range(8, 24)), list(range(24, 40)), list(range(40, 56))], hybrid=True,
        rope_dims=64, head_dim=256),
    'moe': dict(
        name='Qwen3.6-35B-A3B',
        path=_model_path('GGUF_MOE', 'Qwen3.6-35B-A3B-Q8_0.gguf'),
        n_layers=40, blocks=[list(range(8, 20)), list(range(20, 32))], hybrid=True,
        rope_dims=64, head_dim=256),
}


def family(cfg, l):
    """All layers that share a twin family with l (including l). Untwinned layers: {l}."""
    for blk in cfg['blocks']:
        if l in blk:
            p = blk.index(l)
            return {b[p] for b in cfg['blocks']}
    return {l}


def is_twin(cfg, a, b):
    return a != b and b in family(cfg, a)


def twin_pairs(cfg):
    out = []
    bl = cfg['blocks']
    for x in range(len(bl)):
        for y in range(x + 1, len(bl)):
            out += [(a, b) for a, b in zip(bl[x], bl[y])]
    return sorted(out)


def layer_type(cfg, l):
    if not cfg['hybrid']:
        return 'full'
    return 'full' if (l + 1) % 4 == 0 else 'linear'


def relation(cfg, a, b):
    """'twin' / 'nontwin' for a decoder-layer pair."""
    return 'twin' if is_twin(cfg, a, b) else 'nontwin'


def matched_controls(cfg, a, b, layers=None):
    """Architecture- and depth-matched non-twin controls for pair (a, b): (a, b-4) and (a, b+4), kept only
    if both layers exist (decoder layers only), have the same type as the twin pair, and are not twins."""
    layers = set(range(cfg['n_layers'])) if layers is None else set(layers)
    out = []
    for bb in (b - 4, b + 4):
        if bb in layers and bb != a and layer_type(cfg, bb) == layer_type(cfg, a) and not is_twin(cfg, a, bb):
            out.append((a, bb))
    return out


class GGUF:
    def __init__(self, key):
        self.cfg = MODELS[key]
        self.key = key
        r = GGUFReader(self.cfg['path'])
        self.info = {}
        for t in r.tensors:
            self.info[t.name] = (t.tensor_type, [int(x) for x in t.shape], int(t.data_offset), int(t.n_bytes))
        del r
        self.fd = os.open(self.cfg['path'], os.O_RDONLY)
        try:
            fcntl.fcntl(self.fd, fcntl.F_NOCACHE, 1)
        except (AttributeError, OSError):
            pass

    def has(self, name):
        return name in self.info

    def shape(self, name):
        return tuple(reversed(self.info[name][1]))

    def _pread(self, off, n):
        out = bytearray(n)
        mv = memoryview(out)
        got = 0
        while got < n:
            k = min(n - got, 1 << 26)
            b = os.pread(self.fd, k, off + got)
            if not b:
                raise IOError('short read')
            mv[got:got + len(b)] = b
            got += len(b)
        return out

    def _row_geom(self, name):
        tt, ne, off, nbytes = self.info[name]
        ne0 = ne[0]
        n_rows = int(np.prod(ne[1:])) if len(ne) > 1 else 1
        if tt == GGMLQuantizationType.Q8_0:
            rb = ne0 // 32 * 34
        elif tt == GGMLQuantizationType.F32:
            rb = ne0 * 4
        elif tt == GGMLQuantizationType.F16:
            rb = ne0 * 2
        else:
            raise NotImplementedError(tt)
        assert rb * n_rows == nbytes, (name, rb, n_rows, nbytes)
        return tt, ne0, n_rows, rb, off

    @staticmethod
    def _decode(tt, buf, n_rows, ne0):
        if tt == GGMLQuantizationType.Q8_0:
            a = np.frombuffer(buf, dtype=np.uint8).reshape(-1, 34)
            d = a[:, :2].copy().view(np.float16).astype(np.float32)
            q = a[:, 2:].view(np.int8).astype(np.float32)
            q *= d
            return q.reshape(n_rows, ne0)
        if tt == GGMLQuantizationType.F32:
            return np.frombuffer(buf, dtype=np.float32).reshape(n_rows, ne0).copy()
        if tt == GGMLQuantizationType.F16:
            return np.frombuffer(buf, dtype=np.float16).reshape(n_rows, ne0).astype(np.float32)
        raise NotImplementedError(tt)

    def read(self, name, rows=None, chunk_elems=1 << 25):
        """Dequantised float32 tensor in storage order (shape = reversed ne). rows: optional sorted row indices
        into the flattened (n_rows, ne0) view; then returns (len(rows), ne0)."""
        tt, ne0, n_rows, rb, off = self._row_geom(name)
        if rows is None:
            out = np.empty((n_rows, ne0), dtype=np.float32)
            step = max(1, chunk_elems // ne0)
            for r0 in range(0, n_rows, step):
                r1 = min(n_rows, r0 + step)
                out[r0:r1] = self._decode(tt, self._pread(off + r0 * rb, (r1 - r0) * rb), r1 - r0, ne0)
            return out.reshape(self.shape(name))
        rows = np.asarray(rows)
        out = np.empty((len(rows), ne0), dtype=np.float32)
        # group contiguous runs
        i = 0
        while i < len(rows):
            j = i
            while j + 1 < len(rows) and rows[j + 1] == rows[j] + 1:
                j += 1
            r0, r1 = int(rows[i]), int(rows[j]) + 1
            out[i:j + 1] = self._decode(tt, self._pread(off + r0 * rb, (r1 - r0) * rb), r1 - r0, ne0)
            i = j + 1
        return out

    def n_rows(self, name):
        return self._row_geom(name)[2]


# ----------------------------------------------------------------------------------------------- stats
def rownorm(X, axis=-1):
    return np.sqrt(np.einsum('...i,...i->...', X, X, dtype=np.float64) if axis == -1 else (X * X).sum(axis))


def rowcos(X, Y):
    """Cosine between matching rows of X and Y (float64 accumulation)."""
    num = np.einsum('ij,ij->i', X, Y, dtype=np.float64)
    den = np.sqrt(np.einsum('ij,ij->i', X, X, dtype=np.float64) * np.einsum('ij,ij->i', Y, Y, dtype=np.float64))
    return num / np.maximum(den, 1e-30)


def colcos(X, Y):
    """Cosine between matching columns of X and Y."""
    num = np.einsum('ij,ij->j', X, Y, dtype=np.float64)
    den = np.sqrt(np.einsum('ij,ij->j', X, X, dtype=np.float64) * np.einsum('ij,ij->j', Y, Y, dtype=np.float64))
    return num / np.maximum(den, 1e-30)


def flatcos(X, Y):
    x = X.ravel()
    y = Y.ravel()
    return float(np.dot(x.astype(np.float64), y.astype(np.float64)) /
                 np.sqrt(np.dot(x.astype(np.float64), x.astype(np.float64)) * np.dot(y.astype(np.float64), y.astype(np.float64))))


def pearson(x, y):
    x = np.asarray(x, np.float64) - np.mean(x)
    y = np.asarray(y, np.float64) - np.mean(y)
    d = np.sqrt((x * x).sum() * (y * y).sum())
    return float((x * y).sum() / d) if d > 0 else float('nan')


def boot_ci(values, stat=np.median, B=1000, seed=0, alpha=0.05):
    """Percentile bootstrap CI over units (rows of `values` if 2-D, entries if 1-D)."""
    v = np.asarray(values)
    rng = np.random.default_rng(seed)
    n = len(v)
    s = np.empty(B)
    for b in range(B):
        s[b] = stat(v[rng.integers(0, n, n)])
    return float(np.quantile(s, alpha / 2)), float(np.quantile(s, 1 - alpha / 2))


def seed_of(*parts):
    return zlib.crc32('|'.join(map(str, parts)).encode())


def derangement(n, rng):
    """Random permutation with no fixed points."""
    while True:
        p = rng.permutation(n)
        if not np.any(p == np.arange(n)):
            return p


class Log:
    def __init__(self, path):
        self.f = open(path, 'w')

    def __call__(self, *a):
        s = ' '.join(str(x) for x in a)
        print(s, flush=True)
        self.f.write(s + '\n')
        self.f.flush()


def write_csv(path, rows, cols=None):
    import csv
    if not rows:
        return
    cols = cols or list(rows[0].keys())
    with open(path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction='ignore')
        w.writeheader()
        for r in rows:
            w.writerow({k: (f'{v:.6g}' if isinstance(v, (float, np.floating)) else v) for k, v in r.items()})
