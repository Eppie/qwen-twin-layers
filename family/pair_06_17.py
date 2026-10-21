#!/usr/bin/env python3
"""Qwen3-0.6B vs Qwen3-1.7B: does one residual-dimension map align the weights, and which way did width change?

    python3 family/pair_06_17.py          (local BF16 Qwen3-0.6B and Qwen3-1.7B under QWEN_HF_DIR)

1. Map each 0.6B residual dim d to the 1.7B dim pi(d) (and sign) with the largest |corr| of embedding columns
   over the first 30,000 tokens. Map built from tokens 0-14,999 and checked on tokens 15,000-29,999.
2. Check the map on independent quantities: per-dim norm gains (input_layernorm, post_attention_layernorm,
   final norm), and same-row cosines of k/v/q projections (head channels are identical in both models: 8 KV x 128,
   16 Q x 128) with columns remapped by pi; o_proj with rows remapped. Controls: pi shuffled; other layers.
3. FFN neurons per layer: 0.6B gate/up rows vs all 1.7B rows (columns remapped), best match, one-to-one-ness.
4. Direction signatures: is pi monotone (old order kept, as when dims are added)? Is the 0.6B dim order sorted
   by an importance measure of the matched 1.7B dims (as when dims are ranked and the top ones kept)?
"""
import os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from analyze_small import Local
from paths import local

A, B = Local(local('Qwen3-0.6B')), Local(local('Qwen3-1.7B'))
L, NT = 28, 30000
rng = np.random.default_rng(0)


def z(X):
    X = X - X.mean(0); return X / X.std(0)


EA, EB = A.get('model.embed_tokens.weight', 0, NT), B.get('model.embed_tokens.weight', 0, NT)
C1 = z(EA[:15000]).T @ z(EB[:15000]) / 15000
C2 = z(EA[15000:]).T @ z(EB[15000:]) / 15000
pi = np.abs(C1).argmax(1); sg = np.sign(C1[np.arange(1024), pi])
held = C2[np.arange(1024), pi] * sg
print('== 1. residual-dim map from embedding columns')
print(f'map built on tokens 0-14999: median |corr| {np.median(np.abs(C1).max(1)):.3f}; on held-out tokens 15000-29999 '
      f'the same pairs give median {np.median(held):.3f} (all positive after sign: {np.mean(held > 0):.0%}); '
      f'distinct targets {len(set(pi))}/1024; held-out best-match agreement {np.mean(np.abs(C2).argmax(1) == pi):.0%}')

def gain(m, name):
    return m.get(name).astype(np.float64)

print('\n== 2. independent checks of the map')
for nm in ['input_layernorm', 'post_attention_layernorm']:
    r = [np.corrcoef(gain(A, f'model.layers.{l}.{nm}.weight'), gain(B, f'model.layers.{l}.{nm}.weight')[pi])[0, 1] for l in range(L)]
    sh = [np.corrcoef(gain(A, f'model.layers.{l}.{nm}.weight'), gain(B, f'model.layers.{l}.{nm}.weight')[rng.permutation(pi)])[0, 1] for l in range(L)]
    print(f'{nm:26s} per-dim gain r under pi: median {np.median(r):.3f} (min {np.min(r):.3f}); pi shuffled: median {np.median(sh):.3f}')
r = np.corrcoef(gain(A, 'model.norm.weight'), gain(B, 'model.norm.weight')[pi])[0, 1]
print(f'{"final norm":26s} r under pi {r:.3f}')

def remap_cos(f, l, la=None, col=True, perm=None):
    a = A.get(f'model.layers.{l}.{f}.weight').astype(np.float64)
    b = B.get(f'model.layers.{la if la is not None else l}.{f}.weight').astype(np.float64)
    p = pi if perm is None else perm
    b = b[:, p] * sg if col else b[p] * sg[:, None]
    return float((a * b).sum() / np.sqrt((a * a).sum() * (b * b).sum()))

print('same-row cosine with residual dims remapped (layer p vs p | layer p vs p+5 | pi shuffled):')
for f, col in [('self_attn.k_proj', True), ('self_attn.v_proj', True), ('self_attn.q_proj', True), ('self_attn.o_proj', False),
               ('mlp.gate_proj', None), ('mlp.down_proj', None)]:
    if col is None: continue
    same = [remap_cos(f, l, col=col) for l in range(L)]
    other = [remap_cos(f, l, la=(l + 5) % L, col=col) for l in range(L)]
    shuf = [remap_cos(f, l, col=col, perm=rng.permutation(pi)) for l in range(L)]
    print(f'  {f:18s} median {np.median(same):.3f} (range {np.min(same):.3f}-{np.max(same):.3f}) | {np.median(other):+.4f} | {np.median(shuf):+.4f}')

print('\n== 3. FFN neurons (gate and up rows, columns remapped): best 1.7B match for each 0.6B neuron')
for l in [0, 7, 14, 21, 27]:
    S = 0
    for f in ['mlp.gate_proj', 'mlp.up_proj']:
        a = A.get(f'model.layers.{l}.{f}.weight'); b = B.get(f'model.layers.{l}.{f}.weight')[:, pi] * sg
        a = a / np.linalg.norm(a, axis=1, keepdims=True); b = b / np.linalg.norm(b, axis=1, keepdims=True)
        S = S + a @ b.T / 2
    best, arg = S.max(1), S.argmax(1)
    second = np.sort(S, 1)[:, -2]
    da = A.get(f'model.layers.{l}.mlp.down_proj.weight'); db = B.get(f'model.layers.{l}.mlp.down_proj.weight')[pi] * sg[:, None]
    dcos = np.array([da[:, j] @ db[:, arg[j]] / np.linalg.norm(da[:, j]) / np.linalg.norm(db[:, arg[j]]) for j in range(da.shape[1])])
    rho = np.corrcoef(np.arange(len(arg)), arg)[0, 1]
    print(f'  layer {l:2d}: median best {np.median(best):.3f} (second best {np.median(second):.3f}); down column at that match {np.median(dcos):.3f}; '
          f'distinct targets {len(set(arg))}/{len(arg)}; index order correlation (0.6B j vs matched 1.7B index) {rho:+.3f}')

print('\n== 4. direction signatures for the residual map')
from scipy.stats import spearmanr
print(f'pi monotone? Spearman(d, pi(d)) = {spearmanr(np.arange(1024), pi)[0]:+.3f}')
imp = {'embedding column norm': np.linalg.norm(EB, axis=0),
       'mean |input_layernorm gain| over layers': np.mean([np.abs(gain(B, f"model.layers.{l}.input_layernorm.weight")) for l in range(L)], 0),
       'k/q/v input column norm (all layers)': np.sum([np.linalg.norm(B.get(f"model.layers.{l}.self_attn.{f}.weight"), axis=0) for l in range(L) for f in ("q_proj", "k_proj", "v_proj")], 0)}
for k, v in imp.items():
    rank = np.argsort(np.argsort(-v))  # 0 = most important 1.7B dim
    matched = rank[pi]
    print(f'{k:42s}: Spearman(0.6B index d, importance rank of pi(d)) {spearmanr(np.arange(1024), matched)[0]:+.3f}; '
          f'matched dims median rank {np.median(matched):.0f} vs unmatched {np.median(np.delete(rank, list(set(pi)))):.0f} (of 2048)')
