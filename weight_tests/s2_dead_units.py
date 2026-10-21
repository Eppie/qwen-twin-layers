#!/usr/bin/env python3
"""All-zero FFN neurons ("dead units") and whether twin layers share them (found while running s2_moe_experts.py).

A unit is dead if its gate row, up row and down column dequantise to exactly zero (Q8_0: every 32-value block
has |w| < ~7.6e-6, below the smallest f16 block scale times 127). For each layer: number of dead units; for
every pair of layers: size of the intersection of their dead sets, compared between twin and non-twin pairs.
Also a softer criterion (unit norm < 1% of the layer's median unit norm, computed per projection and required
in all three) to see whether near-dead units follow the same pattern.

Why it matters: with gate = up = 0 a SwiGLU unit outputs 0 and every gradient into its gate, up and down
weights is 0, so an exactly dead unit stays dead under any gradient optimiser (and weight decay keeps 0 at 0).
Random initialisation produces no dead units. Dead sets that coincide unit-for-unit between twins were
therefore most likely present before the twins' parameters were separated (copy or untying), not produced
independently afterwards. (Native BF16 weights would show whether these units are exactly 0 or merely tiny;
that belongs to the native-weight replication.)

    python3 -I s2_dead_units.py {moe|32b|27b}
Writes to the output directory (see common.py): s2_dead_units_<model>.csv (per layer pair) and s2_dead_units_<model>.log.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np

from common import GGUF, LOGS, Log, RES, is_twin, write_csv

key = sys.argv[1]
g = GGUF(key)
cfg = g.cfg
NL = cfg['n_layers']
log = Log(os.path.join(LOGS, f's2_dead_units_{key}.log'))

dead, soft, nunits = {}, {}, None
for l in range(NL):
    if key == 'moe':
        ng = np.sqrt((g.read(f'blk.{l}.ffn_gate_exps.weight').astype(np.float64) ** 2).sum(2)).ravel()
        nu = np.sqrt((g.read(f'blk.{l}.ffn_up_exps.weight').astype(np.float64) ** 2).sum(2)).ravel()
        nd = np.sqrt((g.read(f'blk.{l}.ffn_down_exps.weight').astype(np.float64) ** 2).sum(1)).ravel()
    else:
        ng = np.sqrt((g.read(f'blk.{l}.ffn_gate.weight').astype(np.float64) ** 2).sum(1))
        nu = np.sqrt((g.read(f'blk.{l}.ffn_up.weight').astype(np.float64) ** 2).sum(1))
        nd = np.sqrt((g.read(f'blk.{l}.ffn_down.weight').astype(np.float64) ** 2).sum(0))
    nunits = len(ng)
    z = (ng == 0) & (nu == 0) & (nd == 0)
    partial = int(((ng == 0) | (nu == 0) | (nd == 0)).sum() - z.sum())
    sf = (ng < 0.01 * np.median(ng)) & (nu < 0.01 * np.median(nu)) & (nd < 0.01 * np.median(nd))
    dead[l], soft[l] = set(np.where(z)[0].tolist()), set(np.where(sf)[0].tolist())
    log(f'layer {l:2d}: dead units {len(dead[l]):6d}  (units zero in only some projections: {partial});  '
        f'near-dead (<1% of median norm in all three) {len(soft[l]):6d}' +
        (f'  experts touched {len({u // 512 for u in dead[l]})}' if key == 'moe' and dead[l] else ''))

rows = []
for a in range(NL):
    for b in range(a + 1, NL):
        for crit, S in (('dead', dead), ('near_dead', soft)):
            A, B = S[a], S[b]
            if not A and not B:
                continue
            inter = len(A & B)
            exp = len(A) * len(B) / nunits
            rows.append(dict(model=cfg['name'], criterion=crit, a=a, b=b, twin=int(is_twin(cfg, a, b)), n_a=len(A), n_b=len(B),
                             intersection=inter, expected_if_independent=exp,
                             jaccard=inter / len(A | B) if A | B else np.nan))
write_csv(os.path.join(RES, f's2_dead_units_{key}.csv'), rows)
for crit in ('dead', 'near_dead'):
    tw = [r for r in rows if r['criterion'] == crit and r['twin']]
    nt = [r for r in rows if r['criterion'] == crit and not r['twin'] and r['n_a'] and r['n_b']]
    log(f'\n{crit}: twin pairs (with any such units)')
    for r in tw:
        log(f'  ({r["a"]},{r["b"]}) sizes {r["n_a"]}/{r["n_b"]}, shared {r["intersection"]} '
            f'(expected if placed independently {r["expected_if_independent"]:.3g}), Jaccard {r["jaccard"]:.3f}')
    if nt:
        j = np.array([r['jaccard'] for r in nt])
        ex = sum(r['intersection'] for r in nt)
        exp = sum(r['expected_if_independent'] for r in nt)
        top = sorted(nt, key=lambda r: -r['jaccard'])[:5]
        log(f'  non-twin pairs with units in both layers: {len(nt)}; mean Jaccard {j.mean():.4f}, max {j.max():.3f}; '
            f'total shared {ex} vs {exp:.1f} expected if independent; highest: ' +
            ', '.join(f'({r["a"]},{r["b"]}) {r["intersection"]}/{min(r["n_a"], r["n_b"])}' for r in top))
log('\ndone')
