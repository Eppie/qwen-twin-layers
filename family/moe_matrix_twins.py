#!/usr/bin/env python3
"""Matrix-level check of the twin blocks found in Qwen3 MoE q/k norms, without downloading the models.

    python3 family/moe_matrix_twins.py Qwen/Qwen3-30B-A3B [Qwen/Qwen3-235B-A22B ...]

Fetches k_proj, v_proj and the router (mlp.gate) of every layer with HTTP range requests and computes the
same-index cosine between every pair of layers. For the router, rows are experts, so a twin match means
expert e of one layer resembles expert e of the other; also reported after removing the mean expert row
(which does not change routing). The fetched tensors are cached in paths.CACHE (REPO_kvrouter.npz; about 0.4 GB
for Qwen3-30B-A3B and 1.7 GB for Qwen3-235B-A22B as float32), and the similarity matrices saved as REPO_twin_sims.npz.
The "fetched ..." lines are printed only when the tensors are fetched, not when they come from the cache.
"""
import json, os, struct, sys, urllib.request
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paths import CACHE


def get(url, rng=None):
    req = urllib.request.Request(url, headers={'Range': f'bytes={rng[0]}-{rng[1] - 1}'} if rng else {})
    for attempt in range(5):
        try:
            with urllib.request.urlopen(req, timeout=300) as r: return r.read()
        except Exception:
            if attempt == 4: raise


def fetch(repo):
    path = os.path.join(CACHE, repo.replace('/', '_') + '_kvrouter.npz')
    if os.path.exists(path): return dict(np.load(path))
    base = f'https://huggingface.co/{repo}/resolve/main/'
    where = json.loads(get(base + 'model.safetensors.index.json'))['weight_map']
    L = json.loads(get(base + 'config.json'))['num_hidden_layers']
    heads, out = {}, {}
    for key, suffix in [('k', 'self_attn.k_proj.weight'), ('v', 'self_attn.v_proj.weight'), ('router', 'mlp.gate.weight')]:
        mats = []
        for l in range(L):
            name = f'model.layers.{l}.{suffix}'; fn = where[name]
            if fn not in heads:
                n = struct.unpack('<Q', get(base + fn, (0, 8)))[0]; heads[fn] = (json.loads(get(base + fn, (8, 8 + n))), 8 + n)
            h, b = heads[fn]; t = h[name]; s, e = t['data_offsets']
            u = np.frombuffer(get(base + fn, (b + s, b + e)), dtype=np.uint16)
            mats.append((u.astype(np.uint32) << 16).view(np.float32).reshape(t['shape']))
        out[key] = np.stack(mats)
        print(f'  fetched {key}: {out[key].shape}', flush=True)
    os.makedirs(CACHE, exist_ok=True); np.savez(path, **out)
    return out


def sim(X):
    V = X.reshape(len(X), -1).astype(np.float64)
    V /= np.linalg.norm(V, axis=1, keepdims=True)
    return V @ V.T


for repo in sys.argv[1:]:
    print(f'== {repo}', flush=True)
    d = fetch(repo); L = len(d['k'])
    mats = {'k_proj': sim(d['k']), 'v_proj': sim(d['v']), 'router': sim(d['router']),
            'router (expert-centred)': sim(d['router'] - d['router'].mean(1, keepdims=True))}
    for name, S in mats.items():
        np.fill_diagonal(S, np.nan)
        print(f'  {name}: all off-diagonal pairs {np.nanmean(S):+.3f} ± {np.nanstd(S):.3f}')
    S = (mats['k_proj'] + mats['v_proj'] + mats['router (expert-centred)']) / 3
    print('  partners with mean(k, v, centred router) cosine > 0.1:')
    for i in range(L):
        p = [j for j in range(L) if j != i and S[i, j] > 0.1]
        print(f'   {i:2d}: ' + ' '.join(f'{j}(k {mats["k_proj"][i,j]:.2f} v {mats["v_proj"][i,j]:.2f} r {mats["router (expert-centred)"][i,j]:.2f})' for j in p))
    np.savez(os.path.join(CACHE, repo.replace('/', '_') + '_twin_sims.npz'), **{k.split()[0] + ('_c' if 'centred' in k else ''): v for k, v in mats.items()})
    S0 = np.nan_to_num(S)
    tw = [(i, j) for i in range(L) for j in range(i + 1, L) if S0[i, j] > 0.05]
    mask = ~np.eye(L, dtype=bool)
    for i, j in tw: mask[i, j] = mask[j, i] = False
    print(f'  twin pairs = pairs with mean(k, v, centred router) cosine > 0.05: {len(tw)}; mean (min) over them vs all other pairs:')
    for name, X in mats.items():
        X = np.nan_to_num(X); t = [X[i, j] for i, j in tw]
        print(f'    {name:24s} twins {np.mean(t):.3f} (min {np.min(t):.3f}) | other pairs {X[mask].mean():+.4f} ± {X[mask].std():.4f} (max {X[mask].max():.3f})')
