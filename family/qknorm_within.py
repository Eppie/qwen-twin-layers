#!/usr/bin/env python3
"""Twin-layer scan within each Qwen3 model using only the q/k-norm gains.

    python3 family/qknorm_within.py          (after qknorm_screen.py, which caches the gains in paths.CACHE)

For each model: centre each 128-d gain vector on the model's mean profile, then Pearson r between every pair of
layers (mean of the q and k correlations). Printed per model: the mean and SD over all pairs, the four layer
offsets with the highest mean r (offsets with at least 4 pairs; n = number of pairs, and how many exceed 0.5),
and for the top offset the pairs above 0.5 (at most 30 listed). For the two MoE models, every layer's partners
with r > 0.6.
"""
import os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paths import CACHE


def corr(name):
    d = np.load(os.path.join(CACHE, f'qknorm_{name}.npz')); L = len(d['q'])
    R = 0
    for k in 'qk':
        a = d[k] - d[k].mean(0); a = (a - a.mean(1, keepdims=True)) / a.std(1, keepdims=True)
        R = R + (a @ a.T / a.shape[1]) / 2
    return R, L


print('within-model twin scan on q/k-norm gains (centred on the model mean, mean of q and k Pearson r)')
for name in ['0.6B', '1.7B', '4B', '8B', '14B', '32B', '30B-A3B', '235B-A22B']:
    R, L = corr(name)
    off = {o: np.array([R[i, i + o] for i in range(L - o)]) for o in range(1, L)}
    means = sorted(((v.mean(), o) for o, v in off.items() if len(v) >= 4), reverse=True)[:4]
    allp = np.concatenate([v for v in off.values()])
    print(f'{name:10s} {L:3d} layers  all pairs {allp.mean():+.3f} ± {allp.std():.3f} | top offsets ' +
          '  '.join(f'+{o}: {m:+.2f} (n={len(off[o])}, pairs > 0.5: {(off[o] > 0.5).sum()})' for m, o in means))
    for m, o in means[:1]:
        hi = [(i, i + o, R[i, i + o]) for i in range(L - o) if R[i, i + o] > 0.5]
        if hi: print(f'{"":15s} +{o} pairs above 0.5: ' + ' '.join(f'{a}-{b}:{r:.2f}' for a, b, r in hi[:30]))

for name in ['30B-A3B', '235B-A22B']:
    R, L = corr(name)
    np.fill_diagonal(R, -1)
    print(f'\n{name}: partners with r > 0.6 for each layer')
    for i in range(L):
        p = [j for j in range(L) if R[i, j] > 0.6]
        print(f'  {i:2d}: ' + ' '.join(f'{j}({R[i,j]:.2f})' for j in p))
