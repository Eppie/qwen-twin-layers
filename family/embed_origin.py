#!/usr/bin/env python3
"""Where do Qwen3-4B's tied embeddings come from: Qwen3-8B's input embeddings or its output head?

    python3 family/embed_origin.py

Q is the 8B -> 4B residual map fitted on k/v weights of all ten paired layers (as in weight_map.py). For token t,
compare the 4B embedding row with Q applied to the 8B input-embedding row and to the 8B lm_head row (cosine per
token, median over 30,000 tokens); control: 8B rows of shuffled tokens. Qwen3-4B ties its input and output
embeddings; Qwen3-8B does not. Fetches the 8B's lm_head rows 0-29,999 (about 0.25 GB).
"""
import os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pair_4_8 as P

def rows(n, layers):
    return np.concatenate([P.MOD[n].get(f'model.layers.{l}.self_attn.{f}.weight').astype(np.float64)
                           for l in layers for f in ('k_proj', 'v_proj')])
Xb, Xs = rows('8B', P.LAY['8B']), rows('4B', P.LAY['4B'])
Qt = np.linalg.solve(Xs.T @ Xs + 1e-2 * np.mean(Xs * Xs) * len(Xs) * np.eye(Xs.shape[1]), Xs.T @ Xb)  # W8 ~ W4 Qt
# reading weights transform as W8 = W4 Qt, so residual vectors as x4 = Qt x8
E4 = P.emb('4B').astype(np.float64)
E8 = P.emb('8B').astype(np.float64)
H8 = P.M8.get('lm_head.weight', 0, 30000).astype(np.float64)
rng = np.random.default_rng(0); perm = rng.permutation(30000)
def per_token_cos(A, B):
    return np.sum(A * B, 1) / np.linalg.norm(A, axis=1) / np.linalg.norm(B, axis=1)
for name, S in [('8B input embedding', E8), ('8B lm_head (output embedding)', H8)]:
    pred = S @ Qt.T
    c = per_token_cos(E4, pred); cn = per_token_cos(E4, pred[perm])
    print(f'4B embedding vs Q x ({name}): median per-token cosine {np.median(c):+.3f} (shuffled tokens {np.median(cn):+.3f}); '
          f'tokens above 0.2: {np.mean(c > 0.2):.0%}')
print(f'(8B input embedding vs 8B lm_head directly, same tokens: median cosine {np.median(per_token_cos(E8, H8)):+.3f})')
