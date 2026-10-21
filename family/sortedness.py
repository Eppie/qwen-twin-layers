#!/usr/bin/env python3
"""Are a model's residual dimensions or FFN neurons sorted by some property of their own?

    python3 family/sortedness.py          (local BF16 0.6B, 1.7B, 4B, 14B under QWEN_HF_DIR; Qwen3-32B from GGUF_32B)

Random initialisation is symmetric under permutations, so a model trained from scratch has no reason for its
channel order to track any per-channel property. A model produced by ranking channels by importance and keeping
the top ones in rank order (common in structured pruning) inherits that order. Reported: Spearman rho between
index and (residual dims) embedding column norm, mean |input_layernorm gain|, summed q/k/v input-column norm;
(FFN neurons, per layer, median over layers) gate-row norm, up-row norm, down-column norm.
"""
import os, sys
import numpy as np
from scipy.stats import spearmanr
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from analyze_small import Local
from paths import GGUF_32B, local


def gguf_model(path):
    from gguf import GGUFReader
    from gguf.quants import dequantize
    T = {t.name: t for t in GGUFReader(path).tensors}
    names = {'model.embed_tokens.weight': 'token_embd.weight'}
    for l in range(64):
        for hf, g in [('input_layernorm', 'attn_norm'), ('self_attn.q_proj', 'attn_q'), ('self_attn.k_proj', 'attn_k'),
                      ('self_attn.v_proj', 'attn_v'), ('mlp.gate_proj', 'ffn_gate'), ('mlp.up_proj', 'ffn_up'), ('mlp.down_proj', 'ffn_down')]:
            names[f'model.layers.{l}.{hf}.weight'] = f'blk.{l}.{g}.weight'
    class G:
        def get(self, name, r0=0, r1=None):
            t = T[names[name]]
            d = t.data if r1 is None else t.data[r0:r1]
            return np.asarray(dequantize(d, t.tensor_type), dtype=np.float32)
    return G(), 64


def report(name, m, L):
    E = m.get('model.embed_tokens.weight', 0, 30000)
    idx = np.arange(E.shape[1])
    gains = np.mean([np.abs(m.get(f'model.layers.{l}.input_layernorm.weight')) for l in range(L)], 0)
    kqv = np.sum([np.linalg.norm(m.get(f'model.layers.{l}.self_attn.{f}.weight'), axis=0)
                  for l in range(0, L, max(1, L // 8)) for f in ('q_proj', 'k_proj', 'v_proj')], 0)
    r = [spearmanr(idx, v)[0] for v in (np.linalg.norm(E, axis=0), gains, kqv)]
    ff = []
    for l in range(0, L, max(1, L // 6)):
        g = m.get(f'model.layers.{l}.mlp.gate_proj.weight'); u = m.get(f'model.layers.{l}.mlp.up_proj.weight')
        d = m.get(f'model.layers.{l}.mlp.down_proj.weight')
        j = np.arange(g.shape[0])
        ff.append([spearmanr(j, np.linalg.norm(g, axis=1))[0], spearmanr(j, np.linalg.norm(u, axis=1))[0],
                   spearmanr(j, np.linalg.norm(d, axis=0))[0]])
    ff = np.array(ff)
    print(f'{name:6s} residual dims: rho(index, embed col norm) {r[0]:+.3f}, (index, |gain|) {r[1]:+.3f}, (index, qkv col norm) {r[2]:+.3f} | '
          f'FFN neurons (median over sampled layers): gate {np.median(ff[:, 0]):+.3f}, up {np.median(ff[:, 1]):+.3f}, down {np.median(ff[:, 2]):+.3f}', flush=True)


for name in ['0.6B', '1.7B', '4B', '14B']:
    report(name, Local(local(f'Qwen3-{name}')), {'0.6B': 28, '1.7B': 28, '4B': 36, '14B': 40}[name])
m, L = gguf_model(GGUF_32B); report('32B', m, L)
