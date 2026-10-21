#!/usr/bin/env python3
"""How far is each Qwen3-14B layer from the point where Qwen3-32B's two width copies of a unit split?

    python3 loop_hypothesis/drift_profile.py HF_14B_DIR GGUF_32B OUT_DIR

For every 14B layer q (0..39) and every 32B layer m that holds a copy of it (m = q for q < 8; q and q + 24 for
8 <= q <= 31; q + 24 for q >= 32), a three-leaf star over the width-copied units (FFN neurons j < 8192: gate rows,
up rows, down columns; query heads 0-2 of each KV group: q rows, o columns): T = 14B unit j, Mj = 32B unit j in
layer m, Mc = its width copy in layer m. With a = cos(T, Mj), b = cos(T, Mc), c = cos(Mj, Mc):
s_T = sqrt(ab/c) and dT = -log s_T, the 14B's distance from the j/c split point (likewise dMj, dMc). Writes
OUT_DIR/drift_profile.txt. Reuses the readers of phylogeny.py.
"""
import os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from phylogeny import SafeTensors, GGUF, head_rows, head_cols, HEADS14, ORIG32, COPY32, NJ, FF14


def units(h14, g32, q, fam, m):
    """14B layer q and 32B layer m: (T_j, M_j, M_c) for width-copied units."""
    if fam in ('gate', 'up'):
        hf, gg = {'gate': ('mlp.gate_proj', 'ffn_gate'), 'up': ('mlp.up_proj', 'ffn_up')}[fam]
        return (h14.rows(f'model.layers.{q}.{hf}.weight', 0, NJ), g32.rows(f'blk.{m}.{gg}.weight', 0, NJ),
                g32.rows(f'blk.{m}.{gg}.weight', FF14, FF14 + NJ))
    if fam == 'down':
        cb = [(0, NJ // 32), (FF14 // 32, (FF14 + NJ) // 32)]
        Mj, Mc = g32.rows(f'blk.{m}.ffn_down.weight', 0, 5120, cb)
        return h14.rows(f'model.layers.{q}.mlp.down_proj.weight')[:, :NJ], Mj, Mc
    if fam == 'q':
        M = g32.rows(f'blk.{m}.attn_q.weight', 0, 8192)
        return head_rows(h14.rows(f'model.layers.{q}.self_attn.q_proj.weight'), HEADS14), head_rows(M, ORIG32), head_rows(M, COPY32)
    if fam == 'o':
        M = g32.rows(f'blk.{m}.attn_output.weight', 0, 5120)
        return head_cols(h14.rows(f'model.layers.{q}.self_attn.o_proj.weight'), HEADS14), head_cols(M, ORIG32), head_cols(M, COPY32)


def cos3(X, chunk=256):
    G = np.zeros((3, 3))
    for r in range(0, X[0].shape[0], chunk):
        M = np.stack([x[r:r + chunk].ravel() for x in X]).astype(np.float64); G += M @ M.T
    n = np.sqrt(np.diag(G)); return G / np.outer(n, n)


def main():
    HF14, G32, OUT = sys.argv[1], sys.argv[2], sys.argv[3]
    h14, g32 = SafeTensors(HF14), GGUF(G32)
    log = open(os.path.join(OUT, 'drift_profile.txt'), 'w')
    def out(s=''):
        print(s, flush=True); log.write(s + '\n'); log.flush()
    out('3-leaf star (T = 14B unit j, Mj = 32B unit j, Mc = its width copy) per 14B layer q and 32B copy m;')
    out('dT = -log s_T = the 14B\'s distance from the j/c split point; dMj, dMc likewise. s_T = sqrt(c(T,Mj) c(T,Mc) / c(Mj,Mc)).')
    res = {}
    for fam in ['gate', 'up', 'down', 'q', 'o']:
        for q in range(40):
            copies = [('L', q)] if q < 8 else ([('L', q), ('U', q + 24)] if q < 32 else [('U', q + 24)])
            for tag, m in copies:
                C = cos3(units(h14, g32, q, fam, m)); a, b, c = C[0, 1], C[0, 2], C[1, 2]
                sT, sJ, sC = np.sqrt(a * b / c), np.sqrt(a * c / b), np.sqrt(b * c / a)
                res.setdefault(fam, []).append((q, tag, -np.log(sT), -np.log(sJ), -np.log(sC)))
        out(f'== {fam}:  q: dT via the copy closest to T (L or U), [dT via L, dT via U]')
        for q in range(40):
            rr = [r for r in res[fam] if r[0] == q]
            best = min(rr, key=lambda r: r[2])
            out(f'  q={q:2d}  dT {best[2]:.3f} ({best[1]})   ' + '  '.join(f'{r[1]}: dT {r[2]:.3f} dMj {r[3]:.3f} dMc {r[4]:.3f}' for r in rr))
    log.close()


if __name__ == '__main__':
    main()
