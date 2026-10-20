#!/usr/bin/env python3
"""Same-index similarity between layers of a GGUF, for every per-layer tensor type (dense, MoE experts, router).

    python3 twin_layers.py MODEL.gguf OFFSET FIRST LAST      # layer i vs i+OFFSET, i = FIRST..LAST (Qwen3-32B: 24 8 31)
    python3 twin_layers.py MODEL.gguf scan                   # no offset assumed: offset profile, best partner per layer

Matrices: signed cosine of the flattened tensors on a fixed sample of rows (the same rows in every layer); for
expert tensors a row is one (expert, neuron), so a match needs both indices aligned. Norm gains: Pearson r of
the gain vectors, raw and after removing each channel's mean across layers. Gains start at 1 under any random
initialisation, so random init gives twins no shared gain pattern.
OFFSET mode compares the twin pairs with the off-by-one pairs (i, i+OFFSET±1), adjacent layers and all other
pairs; ± is the standard deviation across layer pairs. Needs numpy and gguf (pip install gguf numpy).
"""
import sys, zlib
import numpy as np
from gguf import GGUFReader
from gguf.quants import dequantize

path = sys.argv[1]
scan = sys.argv[2] == 'scan'
if not scan:
    offset, lo, hi = int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
T = {t.name: t for t in GGUFReader(path).tensors}
L = max(int(n.split('.')[1]) for n in T if n.startswith('blk.')) + 1
kinds = sorted({n.split('.', 2)[2][:-len('.weight')] for n in T if n.startswith('blk.') and n.endswith('.weight')})


def layers_of(kind):
    return [l for l in range(L) if f'blk.{l}.{kind}.weight' in T]


def fmt(v):
    return f'{np.mean(v):+.3f} (max {np.max(v):+.3f})' if v else 'n/a (layer type absent)'


def summary(kind, S, layers):
    idx = {l: k for k, l in enumerate(layers)}
    twins = [(idx[i], idx[i + offset]) for i in layers if i + offset in idx and lo <= i <= hi]
    if not twins: return
    tw = np.array([S[a, b] for a, b in twins])
    mask = ~np.eye(len(layers), dtype=bool)
    for a, b in twins: mask[a, b] = mask[b, a] = False
    other = S[mask]
    near = [S[idx[i], idx[i + d]] for i in layers for d in (offset - 1, offset + 1) if i + d in idx]
    adj = [S[idx[i], idx[i + 1]] for i in layers if i + 1 in idx]
    print(f'{kind:20s} twins(+{offset}) {tw.mean():+.3f} ± {tw.std():.3f} (min {tw.min():+.3f} max {tw.max():+.3f}) | '
          f'off-by-one {fmt(near)} | adjacent {fmt(adj)} | '
          f'all other pairs {other.mean():+.3f} ± {other.std():.3f} (max {other.max():+.3f})')


def profile(kind, S, layers):
    # mean similarity of (i, i+d) for every offset d; twin offsets stand out
    by_d = {}
    for a, la in enumerate(layers):
        for b in range(a + 1, len(layers)):
            by_d.setdefault(layers[b] - la, []).append(S[a, b])
    means = {d: np.mean(v) for d, v in by_d.items()}
    allv = np.concatenate([v for v in by_d.values()])
    top = sorted(means, key=means.get, reverse=True)[:4]
    print(f'{kind:20s} all pairs {allv.mean():+.3f} ± {allv.std():.3f} | top offsets ' +
          '  '.join(f'+{d}: {means[d]:+.3f} (n={len(by_d[d])})' for d in top))


report = profile if scan else summary
pooled = []
for kind in kinds:
    layers = layers_of(kind)
    t0 = T[f'blk.{layers[0]}.{kind}.weight']
    if len(layers) < 2 or len(t0.shape) < 2: continue
    if len({tuple(T[f'blk.{l}.{kind}.weight'].shape) for l in layers}) > 1: continue
    data = lambda l: T[f'blk.{l}.{kind}.weight'].data.reshape(-1, T[f'blk.{l}.{kind}.weight'].data.shape[-1])
    n_rows = data(layers[0]).shape[0]
    row_len = int(t0.shape[0])
    rng = np.random.default_rng(zlib.crc32(kind.encode()))
    rows = np.sort(rng.choice(n_rows, min(n_rows, 64 if row_len > 6000 else 256), replace=False))
    V = np.stack([dequantize(data(l)[rows], T[f'blk.{l}.{kind}.weight'].tensor_type).astype(np.float32).ravel()
                  for l in layers])
    V /= np.linalg.norm(V, axis=1, keepdims=True) + 1e-12
    if len(layers) == L: pooled.append(V @ V.T)
    report(kind, V @ V.T, layers)

if pooled:
    S = np.mean(pooled, axis=0)
    report('all matrices', S, list(range(L)))
    if scan:
        np.fill_diagonal(S, -np.inf)
        print('best partner per layer (all matrices present in every layer):')
        print('  ' + '  '.join(f'{l}→{int(S[l].argmax())} ({S[l].max():+.2f})' for l in range(L)))

for kind in kinds:
    layers = layers_of(kind)
    t0 = T[f'blk.{layers[0]}.{kind}.weight']
    if 'norm' not in kind or len(layers) < 2 or len(t0.shape) != 1: continue
    G = np.stack([np.array(T[f'blk.{l}.{kind}.weight'].data, dtype=np.float64) for l in layers])
    report(kind + ' raw', np.corrcoef(G), layers)
    report(kind + ' centred', np.corrcoef(G - G.mean(0, keepdims=True)), layers)
