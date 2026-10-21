#!/usr/bin/env python3
"""Four-point (quartet) analysis of where Qwen3-14B attaches among Qwen3-32B's depth and width copies.

    python3 loop_hypothesis/phylogeny.py HF_14B_DIR GGUF_32B OUT_DIR [FAMILIES [LAYERS]]

For block layers p = 8..31 and families (FFN gate rows, up rows, down columns for neurons j < 8192; q_proj rows
and o_proj columns for query heads 0-2 of every KV group), five taxa: T (14B unit j), Lj, Uj (32B unit j at p,
p+24), Lc, Uc (its width copy at p, p+24: FFN neuron 17,408 + j; for heads the duplicated local head). Cosines on
the concatenation of all units; d = -log cos. For each of the five quartets, the three four-point sums; the
smallest gives the split, half the gap to the next its internal edge. See phylogeny_predictions.md (written
before running). FAMILIES (default gate,up,down,q,o) and LAYERS (default 8..31) are comma-separated.

HF_14B_DIR is the BF16 Qwen3-14B checkpoint (safetensors), GGUF_32B the Qwen3-32B Q8_0 GGUF. Tensors are read with
one sequential pread per needed row range (or per tensor, for column subsets) at offsets taken from the
safetensors header and the GGUF tensor table; Q8_0 blocks are dequantised with vectorised numpy, only for the
rows / column blocks used. The 5x5 cosine matrix is accumulated in float64 over row chunks. Writes
OUT_DIR/phylogeny.txt and OUT_DIR/phylogeny.json (input of tree_fit.py).

This is the faster of two implementations; the first one read memory-mapped tensors and dequantised them with
gguf.quants, and its output (committed in results/loop_hypothesis/) is identical to this script's.
"""
import itertools, json, os, struct, sys
import numpy as np
from gguf import GGUFReader, GGMLQuantizationType

NJ, FF14, HD = 8192, 17408, 128
Q8 = GGMLQuantizationType.Q8_0


def pread(path, offset, nbytes, cache={}):
    if path not in cache: cache[path] = os.open(path, os.O_RDONLY)
    buf = bytearray(nbytes); view = memoryview(buf); got = 0
    while got < nbytes:
        got += os.preadv(cache[path], [view[got:]], offset + got)
    return np.frombuffer(buf, dtype=np.uint8)


class SafeTensors:
    def __init__(self, root):
        self.where = json.load(open(os.path.join(root, 'model.safetensors.index.json')))['weight_map']
        self.root, self.headers = root, {}

    def meta(self, name):
        fn = self.where[name]; path = os.path.join(self.root, fn)
        if fn not in self.headers:
            with open(path, 'rb') as f:
                n = struct.unpack('<Q', f.read(8))[0]; self.headers[fn] = (json.loads(f.read(n)), 8 + n)
        h, base = self.headers[fn]; t = h[name]
        assert t['dtype'] == 'BF16'
        return path, base + t['data_offsets'][0], t['shape']

    def rows(self, name, r0=None, r1=None):
        path, off, shape = self.meta(name)
        r0 = 0 if r0 is None else r0; r1 = shape[0] if r1 is None else r1
        rb = int(np.prod(shape[1:])) * 2
        u = pread(path, off + r0 * rb, (r1 - r0) * rb).view(np.uint16).reshape([r1 - r0] + shape[1:])
        return (u.astype(np.uint32) << 16).view(np.float32)


class GGUF:
    def __init__(self, path):
        self.path = path
        self.t = {t.name: t for t in GGUFReader(path).tensors}

    def rows(self, name, r0, r1, c_blocks=None):
        """Rows r0..r1 of a Q8_0 matrix (row length ne0); c_blocks = list of (b0, b1) column-block ranges."""
        t = self.t[name]; assert t.tensor_type == Q8
        ne0 = int(t.shape[0]); nb = ne0 // 32; rb = nb * 34
        raw = pread(self.path, int(t.data_offset) + r0 * rb, (r1 - r0) * rb).reshape(r1 - r0, nb, 34)
        parts = []
        for b0, b1 in (c_blocks or [(0, nb)]):
            blk = raw[:, b0:b1]
            d = np.ascontiguousarray(blk[:, :, :2]).view(np.float16).astype(np.float32)
            q = np.ascontiguousarray(blk[:, :, 2:]).view(np.int8).astype(np.float32)
            parts.append((d * q).reshape(r1 - r0, (b1 - b0) * 32))
        return parts if c_blocks else parts[0]


HEADS14 = [g * 5 + h for g in range(8) for h in range(3)]     # 14B: 5 query heads per KV group; heads 0-2 were duplicated
ORIG32 = [g * 8 + 3 + h for g in range(8) for h in range(3)]  # 32B: 8 per group; local heads 3-5 hold 14B heads 0-2
COPY32 = [g * 8 + h for g in range(8) for h in range(3)]      # and local heads 0-2 their copies


def head_rows(W, heads):
    return np.concatenate([W[h * HD:(h + 1) * HD] for h in heads])


def head_cols(W, heads):
    return np.concatenate([W[:, h * HD:(h + 1) * HD] for h in heads], axis=1)


def taxa(h14, g32, p, fam):
    if fam in ('gate', 'up'):
        hf, gg = {'gate': ('mlp.gate_proj', 'ffn_gate'), 'up': ('mlp.up_proj', 'ffn_up')}[fam]
        T = h14.rows(f'model.layers.{p}.{hf}.weight', 0, NJ)
        L, U = f'blk.{p}.{gg}.weight', f'blk.{p + 24}.{gg}.weight'
        return (T, g32.rows(L, 0, NJ), g32.rows(U, 0, NJ), g32.rows(L, FF14, FF14 + NJ), g32.rows(U, FF14, FF14 + NJ))
    if fam == 'down':  # columns: j < 8192 and 17408 + j, i.e. column blocks 0..255 and 544..799
        T = h14.rows(f'model.layers.{p}.mlp.down_proj.weight')[:, :NJ]
        cb = [(0, NJ // 32), (FF14 // 32, (FF14 + NJ) // 32)]
        Lj, Lc = g32.rows(f'blk.{p}.ffn_down.weight', 0, 5120, cb)
        Uj, Uc = g32.rows(f'blk.{p + 24}.ffn_down.weight', 0, 5120, cb)
        return T, Lj, Uj, Lc, Uc
    if fam == 'q':
        T = head_rows(h14.rows(f'model.layers.{p}.self_attn.q_proj.weight'), HEADS14)
        L = g32.rows(f'blk.{p}.attn_q.weight', 0, 8192); U = g32.rows(f'blk.{p + 24}.attn_q.weight', 0, 8192)
        return T, head_rows(L, ORIG32), head_rows(U, ORIG32), head_rows(L, COPY32), head_rows(U, COPY32)
    if fam == 'o':
        T = head_cols(h14.rows(f'model.layers.{p}.self_attn.o_proj.weight'), HEADS14)
        L = g32.rows(f'blk.{p}.attn_output.weight', 0, 5120); U = g32.rows(f'blk.{p + 24}.attn_output.weight', 0, 5120)
        return T, head_cols(L, ORIG32), head_cols(U, ORIG32), head_cols(L, COPY32), head_cols(U, COPY32)


def cosines(X, chunk=256):
    G = np.zeros((5, 5))
    for r in range(0, X[0].shape[0], chunk):
        M = np.stack([x[r:r + chunk].ravel() for x in X]).astype(np.float64)
        G += M @ M.T
    n = np.sqrt(np.diag(G))
    return G / np.outer(n, n)


NAMES = ['T', 'Lj', 'Uj', 'Lc', 'Uc']


def quartets(C):
    d = -np.log(np.clip(C, 1e-9, None))
    res = []
    for a, b, c, e in itertools.combinations(range(5), 4):
        splits = [((a, b), (c, e)), ((a, c), (b, e)), ((a, e), (b, c))]
        sums = [d[x[0]][x[1]] + d[y[0]][y[1]] for x, y in splits]
        o = np.argsort(sums); win = splits[o[0]]
        res.append(dict(split='{' + ','.join(NAMES[i] for i in win[0]) + '}|{' + ','.join(NAMES[i] for i in win[1]) + '}',
                        edge=float((sums[o[1]] - sums[o[0]]) / 2), sums=[float(s) for s in sums]))
    return res


def main():
    HF14, G32, OUT = sys.argv[1], sys.argv[2], sys.argv[3]
    h14, g32 = SafeTensors(HF14), GGUF(G32)
    fams = sys.argv[4].split(',') if len(sys.argv) > 4 else ['gate', 'up', 'down', 'q', 'o']
    layers = [int(x) for x in sys.argv[5].split(',')] if len(sys.argv) > 5 else list(range(8, 32))
    rows_out = []
    log = open(os.path.join(OUT, 'phylogeny.txt'), 'w')
    def out(s=''):
        print(s, flush=True); log.write(s + '\n'); log.flush()
    out('cosines among T, Lj, Uj, Lc, Uc; then the winning split of each quartet with its internal edge (in -log cos units)')
    for fam in fams:
        for p in layers:
            C = cosines(taxa(h14, g32, p, fam))
            Q = quartets(C)
            rows_out.append(dict(fam=fam, p=p, cos=C.tolist(), quartets=Q))
            out(f'{fam:4s} p={p:2d}  cos T-Lj {C[0,1]:.3f} T-Uj {C[0,2]:.3f} T-Lc {C[0,3]:.3f} T-Uc {C[0,4]:.3f} '
                f'Lj-Uj {C[1,2]:.3f} Lc-Uc {C[3,4]:.3f} Lj-Lc {C[1,3]:.3f} Uj-Uc {C[2,4]:.3f} Lj-Uc {C[1,4]:.3f} Uj-Lc {C[2,3]:.3f} | '
                + '  '.join(f'{r["split"]} {r["edge"]:.3f}' for r in Q))
        json.dump(rows_out, open(os.path.join(OUT, 'phylogeny.json'), 'w'))
    log.close()


if __name__ == '__main__':
    main()
