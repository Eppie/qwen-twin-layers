#!/usr/bin/env python3
"""Qwen3-4B vs Qwen3-8B: a mixed (rotated / projected) residual basis? And is the 8B unsorted?

    python3 family/pair_4_8.py

8B tensors fetched with HTTP range requests and cached (k_proj, v_proj for 10 layers; input_layernorm gains; the
first 30,000 embedding rows, reused from analyze_small.py embed if that has run). Local models from QWEN_HF_DIR.
1. Sortedness of the 8B's residual dims (embedding column norm, |gain|, k/v input-column norm), same measures for
   the 4B and other local models for comparison.
2. Rotation-invariant comparison: Gram matrices of k_proj and v_proj on the head side (W W^T, 1024 x 1024; head
   channels are 8 KV heads x 128 in every dense Qwen3), layer p of one model vs layers of another. Unchanged by an
   orthogonal map of the residual stream; approximately unchanged by a projection that keeps most of the energy.
3. One linear map for everything: fit P (8B residual -> 4B residual) by ridge regression of 4B embedding rows on
   8B embedding rows (tokens 0-19,999), check on held-out tokens, then test whether the same P maps 8B k/v weights
   onto 4B k/v weights (W4 ~ W8 P^T) layer by layer. Controls: the same procedure for unrelated model pairs.
weight_map.py and embed_origin.py import the readers and layer lists defined here (importing does not run 1-3).
"""
import os, sys
import numpy as np
from scipy.stats import spearmanr
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from analyze_small import Local
from hf_remote import Remote
from paths import CACHE, local

LAYERS8 = [0, 4, 8, 12, 16, 20, 24, 28, 32, 35]
M8 = Remote('Qwen/Qwen3-8B')
M4, M17, M14, M06 = (Local(local(f'Qwen3-{n}')) for n in ['4B', '1.7B', '14B', '0.6B'])
LAY = {'8B': LAYERS8, '4B': LAYERS8, '1.7B': [0, 3, 6, 9, 12, 15, 18, 21, 24, 27], '14B': [0, 4, 9, 13, 18, 22, 26, 31, 35, 39],
       '0.6B': [0, 3, 6, 9, 12, 15, 18, 21, 24, 27]}
MOD = {'8B': M8, '4B': M4, '1.7B': M17, '14B': M14, '0.6B': M06}


def emb(n, _cache={}):
    """First 30,000 embedding rows of model n (the slice cached by analyze_small.py embed, if present)."""
    if n not in _cache:
        p = os.path.join(CACHE, f'{n}_embed30000.npy')
        _cache[n] = np.load(p) if os.path.exists(p) else MOD[n].get('model.embed_tokens.weight', 0, 30000)
    return _cache[n]


def gram(n, l, f):
    W = MOD[n].get(f'model.layers.{l}.self_attn.{f}.weight').astype(np.float64)
    G = W @ W.T; return (G / np.linalg.norm(G)).ravel()


def fit_map(src, dst, lam=1e-1):
    X, Y = emb(src)[:20000].astype(np.float64), emb(dst)[:20000].astype(np.float64)
    Xm, Ym = X.mean(0), Y.mean(0); X, Y = X - Xm, Y - Ym
    P = np.linalg.solve(X.T @ X + lam * len(X) * np.eye(X.shape[1]) * np.mean(X * X), X.T @ Y)  # dst = src @ P
    Xt, Yt = emb(src)[20000:] - Xm, emb(dst)[20000:] - Ym
    r2 = 1 - ((Xt @ P - Yt) ** 2).sum(0) / (Yt ** 2).sum(0)
    return P, np.median(r2)


def main():
    print('== 1. sortedness of residual dims (Spearman rho of index vs property)')
    for n in ['8B', '4B', '1.7B', '0.6B', '14B']:
        m, E = MOD[n], emb(n); idx = np.arange(E.shape[1])
        L = {'8B': 36, '4B': 36, '1.7B': 28, '0.6B': 28, '14B': 40}[n]
        g = np.mean([np.abs(m.get(f'model.layers.{l}.input_layernorm.weight')) for l in range(L)], 0)
        kv = np.sum([np.linalg.norm(m.get(f'model.layers.{l}.self_attn.{f}.weight'), axis=0) for l in LAY[n] for f in ('k_proj', 'v_proj')], 0)
        print(f'  {n:5s} embed col norm {spearmanr(idx, np.linalg.norm(E, axis=0))[0]:+.3f}   |gain| {spearmanr(idx, g)[0]:+.3f}   '
              f'k/v col norm {spearmanr(idx, kv)[0]:+.3f}', flush=True)

    print('\n== 2. head-side Gram matrices of k_proj and v_proj: cosine between layers of two models (rotation invariant)')
    GR = {n: {f: np.stack([gram(n, l, f) for l in LAY[n]]) for f in ('k_proj', 'v_proj')} for n in MOD}
    for a, b in [('4B', '8B'), ('0.6B', '1.7B'), ('4B', '1.7B'), ('4B', '14B'), ('8B', '14B'), ('1.7B', '8B')]:
        out = []
        for f in ('k_proj', 'v_proj'):
            S = GR[a][f] @ GR[b][f].T
            diag = np.diag(S) if len(LAY[a]) == len(LAY[b]) else S.max(1)
            off = S[~np.eye(len(S), dtype=bool)] if S.shape[0] == S.shape[1] else S.ravel()
            out.append(f'{f}: matched layers {np.median(diag):.3f} (min {diag.min():.3f}), other layer pairs {np.median(off):.3f}')
        print(f'  {a:4s} vs {b:4s}: ' + ' | '.join(out))

    print('\n== 3. one linear map: ridge-fit P from embeddings, then predict k/v weights')
    for src, dst in [('8B', '4B'), ('1.7B', '0.6B'), ('14B', '4B'), ('8B', '1.7B'), ('14B', '8B')]:
        P, r2 = fit_map(src, dst)
        cs = []
        for la, lb in zip(LAY[dst], LAY[src]):
            for f in ('k_proj', 'v_proj'):
                Wd = MOD[dst].get(f'model.layers.{la}.self_attn.{f}.weight').astype(np.float64)   # [1024, d_dst]
                Ws = MOD[src].get(f'model.layers.{lb}.self_attn.{f}.weight').astype(np.float64)   # [1024, d_src]
                # x_dst ~ x_src @ P  =>  a weight reading x_dst, W_dst x_dst, matches W_src x_src if W_src ~ W_dst P^T
                pred = Wd @ P.T
                cs.append((f, float((pred * Ws).sum() / np.linalg.norm(pred) / np.linalg.norm(Ws))))
        k = np.median([c for f, c in cs if f == 'k_proj']); v = np.median([c for f, c in cs if f == 'v_proj'])
        print(f'  {src:4s} -> {dst:4s}: embedding map held-out R^2 (median over dims) {r2:.3f}; k_proj predicted via the same map, '
              f'cosine {k:+.3f}; v_proj {v:+.3f}  (layers paired by relative depth)', flush=True)


if __name__ == '__main__':
    main()
