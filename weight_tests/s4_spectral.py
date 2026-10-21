#!/usr/bin/env python3
"""Is the twin overlap concentrated in a few leading singular directions?

For each FFN matrix M (neuron-major: gate, up, down^T; N neurons x D residual dims) of both layers, remove the
top-k singular components of THAT layer's own matrix (M_k = M - M V_k V_k^T, V_k = top-k right singular vectors
from the D x D Gram matrix) and report: retained energy ||M_k||^2/||M||^2 per layer, the flattened twin cosine of
the residuals cos(A_k, B_k), and the cosine of the removed parts cos(A - A_k, B - B_k). Inner products are
computed from small projections (A V_A, B V_A, A V_B, B V_B), exactly. A matched non-twin control pair is run
the same way.

    python3 -I s4_spectral.py
Writes to the output directory (see common.py): s4_spectral.csv and s4_spectral.log.
"""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np

from common import GGUF, LOGS, Log, RES, write_csv

log = Log(os.path.join(LOGS, 's4_spectral.log'))
KS = [0, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512]
KMAX = max(KS)
JOBS = [('32b', ('ffn_gate', 'ffn_up', 'ffn_down'), [(16, 40, 'twin'), (16, 44, 'ctl'), (24, 48, 'twin')]),
        ('27b', ('ffn_gate', 'ffn_up', 'ffn_down'), [(16, 32, 'twin'), (32, 48, 'twin'), (16, 48, 'twin'), (16, 36, 'ctl')]),
        ('moe', ('ffn_gate_shexp', 'ffn_up_shexp', 'ffn_down_shexp'), [(12, 24, 'twin'), (12, 28, 'ctl')])]

rows = []
for key, names, pairs in JOBS:
    g = GGUF(key)
    cache = {}

    def load(l, name):
        M = g.read(f'blk.{l}.{name}.weight')
        return np.ascontiguousarray(M.T) if 'down' in name else M

    def spec(l, name, M):
        """top-KMAX right singular vectors and all squared singular values of M (cached per layer/matrix)"""
        if (l, name) not in cache:
            if M.shape[1] <= M.shape[0]:
                w, V = np.linalg.eigh((M.T @ M).astype(np.float64))
                w, V = w[::-1], V[:, ::-1]
            else:  # fewer neurons than dims (MoE shared expert: 512 x 2048): SVD directly
                _, sv, Vt = np.linalg.svd(M.astype(np.float64), full_matrices=False)
                w, V = sv ** 2, Vt.T
            k = min(KMAX, V.shape[1])
            cache[(l, name)] = (np.maximum(w, 0), np.ascontiguousarray(V[:, :k]).astype(np.float32))
        return cache[(l, name)]

    for a, b, rel in pairs:
        t0 = time.time()
        for name in names:
            A, B = load(a, name), load(b, name)
            wA, VA = spec(a, name, A)
            wB, VB = spec(b, name, B)
            AA = float(np.einsum('ij,ij->', A, A, dtype=np.float64))
            BB = float(np.einsum('ij,ij->', B, B, dtype=np.float64))
            AB = float(np.einsum('ij,ij->', A, B, dtype=np.float64))
            PA_A, PB_A = (A @ VA).astype(np.float64), (B @ VA).astype(np.float64)   # projections on A's directions
            PA_B, PB_B = (A @ VB).astype(np.float64), (B @ VB).astype(np.float64)   # projections on B's directions
            M1 = PA_A.T @ PB_B                     # <A v_i, B u_j>
            M2 = VA.T.astype(np.float64) @ VB      # <v_i, u_j>
            del A, B
            out = []
            for k in KS:
                if k > VA.shape[1] or k >= min(g.shape(f'blk.{a}.{name}.weight')):
                    continue  # k at or above the matrix rank leaves nothing
                topA_B = float((PA_A[:, :k] * PB_A[:, :k]).sum())     # <A P_A, B>
                A_topB = float((PA_B[:, :k] * PB_B[:, :k]).sum())     # <A, B P_B>
                top_top = float((M1[:k, :k] * M2[:k, :k]).sum())       # <A P_A, B P_B>
                eA, eB = float(wA[:k].sum()), float(wB[:k].sum())
                resid = AB - topA_B - A_topB + top_top
                rA, rB = AA - eA, BB - eB
                r = dict(model=g.cfg['name'], a=a, b=b, rel=rel, matrix=name, k=k,
                         retained_A=rA / AA, retained_B=rB / BB, cos_resid=resid / np.sqrt(rA * rB),
                         cos_top=(top_top / np.sqrt(eA * eB)) if k > 0 else np.nan)
                rows.append(r)
                out.append(r)
            log(f'{g.cfg["name"]} ({a},{b}) {rel} {name}: ' + '; '.join(
                f'k={r["k"]}: kept {r["retained_A"]:.2f}/{r["retained_B"]:.2f} cos {r["cos_resid"]:+.3f}'
                + (f' (top {r["cos_top"]:+.3f})' if r['k'] else '') for r in out))
        log(f'   ({time.time() - t0:.0f}s)')
    del cache
write_csv(os.path.join(RES, 's4_spectral.csv'), rows)
log('done')
