#!/usr/bin/env python3
"""Neuron-level check that routed experts kept their index across the twin blocks of the original Qwen3 MoEs.

    python3 family/moe_experts.py

A few whole experts (gate, up, down) are fetched with HTTP range requests at twin layers and at a non-twin
control layer. For each pair of (layer, expert): cosine of the flattened expert (rows = neurons in order), plus
per-neuron cosines for the same expert: same neuron index vs shuffled neuron index. Fetches about 0.6 GB (cached in
paths.CACHE).
"""
import itertools, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from hf_remote import Remote

SETS = {'Qwen/Qwen3-30B-A3B': dict(layers=[6, 18, 30, 7], experts=[0, 37, 85, 120]),
        'Qwen/Qwen3-235B-A22B': dict(layers=[13, 33, 47, 67, 14], experts=[5, 77])}
rng = np.random.default_rng(0)
for repo, s in SETS.items():
    R = Remote(repo)
    X = {}
    for l in s['layers']:
        for e in s['experts']:
            g = R.get(f'model.layers.{l}.mlp.experts.{e}.gate_proj.weight'); u = R.get(f'model.layers.{l}.mlp.experts.{e}.up_proj.weight')
            d = R.get(f'model.layers.{l}.mlp.experts.{e}.down_proj.weight').T     # neurons as rows
            X[(l, e)] = np.concatenate([g, u, d], axis=1).astype(np.float64)       # [n_ff, 3 * hidden]
    print(f'== {repo}: layers {s["layers"]} (last is the non-twin control), experts {s["experts"]}', flush=True)
    keys = list(X)
    def flat(a, b):
        A, B = X[a].ravel(), X[b].ravel(); return A @ B / np.linalg.norm(A) / np.linalg.norm(B)
    twins = [l for l in s['layers'][:-1]]; ctrl = s['layers'][-1]
    for e in s['experts']:
        same = [flat((a, e), (b, e)) for a, b in itertools.combinations(twins, 2)]
        cross = [flat((a, e), (b, f)) for a, b in itertools.combinations(twins, 2) for f in s['experts'] if f != e]
        vs_ctrl = [flat((a, e), (ctrl, e)) for a in twins]
        a, b = twins[0], twins[1]
        A, B = X[(a, e)], X[(b, e)]
        An, Bn = A / np.linalg.norm(A, axis=1, keepdims=True), B / np.linalg.norm(B, axis=1, keepdims=True)
        per = np.sum(An * Bn, 1); shuf = np.sum(An * Bn[rng.permutation(len(Bn))], 1)
        print(f'  expert {e:3d}: same expert across twin layers {np.min(same):.3f}-{np.max(same):.3f} | other experts {np.max(np.abs(cross)):.3f} max | '
              f'same expert vs control layer {np.max(np.abs(vs_ctrl)):.3f} max | neuron-level (layers {a},{b}): same index median {np.median(per):.3f}, '
              f'{np.mean(per > 0.1):.0%} above 0.1; shuffled {np.median(shuf):+.3f}', flush=True)
