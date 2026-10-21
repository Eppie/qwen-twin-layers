#!/usr/bin/env python3
"""Original flattened-cosine analysis (twin_layers.py) repeated on fresh, disjoint row samples.

For every matrix tensor type: the original row sample (reproduced exactly: seed crc32(kind), 64 rows if a row has
more than 6000 entries, else 256) plus K fresh samples of the same size, disjoint from it and from each other.
Per sample: signed cosine of the flattened sampled rows for every pair of decoder layers that hold the tensor.
Baselines exclude ALL known twin relations (matters for Qwen3.8-27B) and the MTP block (27B blk.64).

    python3 -I s0_fresh_rows.py {32b|27b|moe} [K=4]
Writes to the output directory (see common.py): s0_<model>_pairs.csv (one row per sample x layer pair) and s0_<model>.log.
"""
import os
import sys
import zlib

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np

from common import GGUF, LOGS, Log, RES, is_twin, layer_type, matched_controls, seed_of, write_csv

key = sys.argv[1]
K = int(sys.argv[2]) if len(sys.argv) > 2 else 4
g = GGUF(key)
cfg = g.cfg
log = Log(os.path.join(LOGS, f's0_{key}.log'))
NL = cfg['n_layers']


def group_of(a, b):
    if not is_twin(cfg, a, b):
        return 'nontwin'
    if key != '27b':
        return 'twin'
    if b - a == 32:
        return 'twin_AC'
    return 'twin_AB' if a < 24 else 'twin_BC'


kinds = sorted({n.split('.', 2)[2][:-len('.weight')] for n in g.info
                if n.startswith('blk.') and n.endswith('.weight') and int(n.split('.')[1]) < NL})
rows_out = []
pooled = {}
log(f'{cfg["name"]}: fresh-row replication, K={K} fresh disjoint samples per tensor type')
log('cos = signed cosine of flattened sampled rows; "twin" = mean over twin pairs (SD across pairs); '
    'nontwin = all other decoder-layer pairs holding the tensor, all twin relations excluded; '
    'ctl±4 = pairs (i, partner±4), same layer type, not twins')
for kind in kinds:
    layers = [l for l in range(NL) if g.has(f'blk.{l}.{kind}.weight')]
    name0 = f'blk.{layers[0]}.{kind}.weight'
    shp = g.info[name0][1]
    if len(layers) < 2 or len(shp) < 2:
        continue
    if len({tuple(g.info[f'blk.{l}.{kind}.weight'][1]) for l in layers}) > 1:
        continue
    row_len = shp[0]
    n_rows = g.n_rows(name0)
    m = min(n_rows, 64 if row_len > 6000 else 256)
    rng = np.random.default_rng(zlib.crc32(kind.encode()))
    orig = np.sort(rng.choice(n_rows, m, replace=False))
    samples = [('orig', orig)]
    if n_rows > m:
        rest = np.setdiff1d(np.arange(n_rows), orig)
        rest = np.random.default_rng(seed_of('fresh', key, kind)).permutation(rest)
        for k in range(min(K, len(rest) // m)):
            samples.append((f'fresh{k + 1}', np.sort(rest[k * m:(k + 1) * m])))
    # disjointness check
    allr = np.concatenate([s for _, s in samples])
    assert len(np.unique(allr)) == len(allr)
    pairs = [(a, b) for i, a in enumerate(layers) for b in layers[i + 1:]]
    ctl = {(a, bb) for a, b in pairs if is_twin(cfg, a, b) for (a, bb) in matched_controls(cfg, a, b, layers)}
    line = []
    for sname, rows in samples:
        V = np.stack([g.read(f'blk.{l}.{kind}.weight', rows).ravel() for l in layers]).astype(np.float64)
        V /= np.linalg.norm(V, axis=1, keepdims=True) + 1e-30
        S = V @ V.T
        idx = {l: i for i, l in enumerate(layers)}
        if len(layers) == NL:
            pooled.setdefault(sname, {})[kind] = S
        for a, b in pairs:
            rows_out.append(dict(model=cfg['name'], kind=kind, sample=sname, n_rows_sampled=len(rows), row_len=row_len,
                                 a=a, b=b, offset=b - a, type=layer_type(cfg, a) if layer_type(cfg, a) == layer_type(cfg, b) else 'mixed',
                                 group=group_of(a, b), ctl4=int((a, b) in ctl), cos=S[idx[a], idx[b]]))
        tw = {}
        for a, b in pairs:
            gr = group_of(a, b)
            if gr != 'nontwin':
                tw.setdefault(gr, []).append(S[idx[a], idx[b]])
        nt = [S[idx[a], idx[b]] for a, b in pairs if not is_twin(cfg, a, b)]
        c4 = [S[idx[a], idx[b]] for a, b in ctl]
        line.append((sname, tw, nt, c4))
    log(f'\n{kind}  (row_len {row_len}, {n_rows} rows, sample size {m}, layers {len(layers)})')
    for sname, tw, nt, c4 in line:
        tws = '  '.join(f'{gr} {np.mean(v):+.3f} ± {np.std(v):.3f} (n={len(v)})' for gr, v in sorted(tw.items()))
        log(f'  {sname:7s} {tws} | nontwin {np.mean(nt):+.4f} ± {np.std(nt):.4f} (max {np.max(nt):+.3f}, n={len(nt)})'
            + (f' | ctl±4 {np.mean(c4):+.4f} ± {np.std(c4):.4f} (n={len(c4)})' if c4 else ''))
    if len(line) > 1:
        for gr in sorted(line[0][1]):
            fm = [np.mean(t[gr]) for _, t, _, _ in line[1:]]
            log(f'  fresh-sample spread of the {gr} mean: {np.min(fm):+.3f} .. {np.max(fm):+.3f} '
                f'(SD across the {len(fm)} fresh samples {np.std(fm, ddof=1) if len(fm) > 1 else 0:.4f}; orig {np.mean(line[0][1][gr]):+.3f})')

if pooled:
    log('\nall matrices (mean over tensor types present in every decoder layer)')
    names = [n for n in pooled if n == 'orig' or int(n[5:]) <= 3]
    common = sorted(set.intersection(*[set(pooled[n]) for n in names]))
    log(f'  (pooled over the {len(common)} tensor types present in every decoder layer that have >= 3 fresh samples: {common})')
    for sname in names:
        S = np.mean([pooled[sname][k] for k in common], axis=0)
        tw = {}
        for a in range(NL):
            for b in range(a + 1, NL):
                gr = group_of(a, b)
                if gr != 'nontwin':
                    tw.setdefault(gr, []).append(S[a, b])
        nt = [S[a, b] for a in range(NL) for b in range(a + 1, NL) if not is_twin(cfg, a, b)]
        tws = '  '.join(f'{gr} {np.mean(v):+.3f} ± {np.std(v):.3f}' for gr, v in sorted(tw.items()))
        log(f'  {sname:7s} {tws} | nontwin {np.mean(nt):+.4f} ± {np.std(nt):.4f} (max {np.max(nt):+.3f})')
        # best partner per layer
        S2 = S.copy()
        np.fill_diagonal(S2, -np.inf)
        best = S2.argmax(1)
        ok = [l for l in range(NL) if any(is_twin(cfg, l, x) for x in range(NL))]
        hit = sum(is_twin(cfg, l, int(best[l])) for l in ok)
        log(f'          best partner is a twin for {hit}/{len(ok)} twinned layers')

write_csv(os.path.join(RES, f's0_{key}_pairs.csv'), rows_out)
log(f'\nwrote {len(rows_out)} rows to results/s0_{key}_pairs.csv')
