#!/usr/bin/env python3
"""Where does the state at which Qwen3-32B's twin copies split sit, relative to the released Qwen3-14B?

    python3 loop_hypothesis/star_decomposition.py Qwen3-32B-Q8_0.gguf results/native/qwen3_14b_parent_bf16.npz \
        [results/loop_hypothesis/star_decomposition.csv]

Three-leaf star: T = Qwen3-14B layer p, L = Qwen3-32B layer p (lower copy), U = Qwen3-32B layer p + 24 (upper
copy), p = 8..31. If all three descend from one state S by independent drifts, cos(X, Y) = s_X * s_Y with
s_X = cos(X, S), so with a = cos(T, L), b = cos(T, U), c = cos(L, U):
    s_T = sqrt(ab/c), s_L = sqrt(ac/b), s_U = sqrt(bc/a).

a and b are read from the 40 x 64 cosine matrices in NPZ (14B layer vs 32B layer, same index; written by the
BF16 version of qwen3_14b_parent.py and committed as results/native/qwen3_14b_parent_bf16.npz): full k_proj and
v_proj, and gate_proj / up_proj rows of FFN neurons 0-511. c is computed here from the Q8_0 GGUF over the same
rows (full attn_k / attn_v; the first 512 storage rows, i.e. neurons 0-511, of ffn_gate / ffn_up), dequantised to
float64. Prints the selected layers shown in the text output and the median over p = 8..31; the optional CSV
gets every layer.
"""
import sys
import numpy as np
from gguf import GGUFReader
from gguf.quants import dequantize

GGUF_32B, NPZ = sys.argv[1], sys.argv[2]
CSV = sys.argv[3] if len(sys.argv) > 3 else None
M = np.load(NPZ)
T = {t.name: t for t in GGUFReader(GGUF_32B).tensors}


def get(name, rows=None):
    t = T[name]; d = t.data if rows is None else t.data[:rows]
    return dequantize(d, t.tensor_type).astype(np.float64).ravel()


def cos(x, y):
    return float(x @ y / np.sqrt((x @ x) * (y @ y)))


fams = {'k_proj': ('attn_k', None), 'v_proj': ('attn_v', None), 'gate_proj': ('ffn_gate', 512), 'up_proj': ('ffn_up', 512)}
print('Star model for three descendants of one state S with independent drifts: cos(X,Y) = s_X * s_Y.')
print('T = released Qwen3-14B layer p, L = 32B layer p (lower copy), U = 32B layer p+24 (upper copy).')
print('s_T = sqrt(ab/c), s_L = sqrt(ac/b), s_U = sqrt(bc/a), with a = cos(T,L), b = cos(T,U), c = cos(L,U).\n')
rows = []
for f, (g, nr) in fams.items():
    print(f'{f}:   p    a(T,L)  b(T,U)  c(L,U)  |  s_T    s_L    s_U')
    for p in range(8, 32):
        a, b = M[f][p, p], M[f][p, p + 24]
        c = cos(get(f'blk.{p}.{g}.weight', nr), get(f'blk.{p + 24}.{g}.weight', nr))
        sT, sL, sU = np.sqrt(a * b / c), np.sqrt(a * c / b), np.sqrt(b * c / a)
        rows.append((f, p, a, b, c, sT, sL, sU))
        if p in (8, 9, 10, 12, 16, 20, 24, 28, 31):
            print(f'        {p:2d}   {a:+.3f}  {b:+.3f}  {c:+.3f}  |  {sT:.3f}  {sL:.3f}  {sU:.3f}')
    r = [x for x in rows if x[0] == f]
    print(f'   median over p = 8..31: s_T {np.median([x[5] for x in r]):.3f}  s_L {np.median([x[6] for x in r]):.3f}  s_U {np.median([x[7] for x in r]):.3f};'
          f'  s_T range {min(x[5] for x in r):.3f}-{max(x[5] for x in r):.3f}\n')
if CSV:
    with open(CSV, 'w') as fh:
        fh.write('family,p,cos_T_L,cos_T_U,cos_L_U,s_T,s_L,s_U\n')
        for x in rows:
            fh.write(f'{x[0]},{x[1]},' + ','.join(f'{v:.6f}' for v in x[2:]) + '\n')
