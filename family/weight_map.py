#!/usr/bin/env python3
"""One linear map between two models' residual streams, fitted on attention weights of half the layers and tested
on the other half.

    python3 family/weight_map.py

Rows of k_proj and v_proj are head channels (8 KV heads x 128) in every dense Qwen3, aligned across related
models (q/k-norm evidence). If the smaller model's residual stream is a linear function of the larger one's,
x_small ~ Q^T x_large, then its read weights satisfy W_large ~ W_small Q^T for one Q in every layer. Q is fitted by
ridge regression on rows from the even-indexed paired layers and evaluated (cosine of predicted vs actual
weights, per layer and matrix) on the odd-indexed ones. Controls: unrelated model pairs.
"""
import os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pair_4_8 as P  # reuses readers, layer lists and caches

def rows(n, layers):
    return np.concatenate([P.MOD[n].get(f'model.layers.{l}.self_attn.{f}.weight').astype(np.float64)
                           for l in layers for f in ('k_proj', 'v_proj')])

print('== 4. one linear map fitted on attention weights (even layer pairs), tested on held-out layers (odd pairs)')
for big, small in [('8B', '4B'), ('1.7B', '0.6B'), ('14B', '4B'), ('8B', '1.7B'), ('4B', '1.7B'), ('14B', '8B')]:
    LB, LS = P.LAY[big], P.LAY[small]
    tr, te = list(range(0, 10, 2)), list(range(1, 10, 2))
    Xb, Xs = rows(big, [LB[i] for i in tr]), rows(small, [LS[i] for i in tr])      # [rows, d_big], [rows, d_small]
    # W_big ~ W_small @ Qt with Qt [d_small, d_big]
    lam = 1e-2 * np.mean(Xs * Xs) * len(Xs)
    Qt = np.linalg.solve(Xs.T @ Xs + lam * np.eye(Xs.shape[1]), Xs.T @ Xb)
    cs = []
    for i in te:
        for f in ('k_proj', 'v_proj'):
            Ws = P.MOD[small].get(f'model.layers.{LS[i]}.self_attn.{f}.weight').astype(np.float64)
            Wb = P.MOD[big].get(f'model.layers.{LB[i]}.self_attn.{f}.weight').astype(np.float64)
            pred = Ws @ Qt
            cs.append(float((pred * Wb).sum() / np.linalg.norm(pred) / np.linalg.norm(Wb)))
    trfit = float(((Xs @ Qt) * Xb).sum() / np.linalg.norm(Xs @ Qt) / np.linalg.norm(Xb))
    print(f'  {small:4s} -> {big:4s}: fit on train layers cos {trfit:.3f}; held-out layers median cos {np.median(cs):.3f} '
          f'(min {np.min(cs):.3f}, max {np.max(cs):.3f})', flush=True)
