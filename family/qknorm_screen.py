#!/usr/bin/env python3
"""Cross-model ancestry screen with q/k-norm gains (128-d in every Qwen3 model, so directly comparable).

    python3 family/qknorm_screen.py

Local models are read from QWEN_HF_DIR and GGUF_32B (see paths.py); 8B, 30B-A3B and 235B-A22B are fetched from
Hugging Face with HTTP range requests (only the q_norm / k_norm vectors). The vectors are cached in paths.CACHE as
qknorm_NAME.npz, which qknorm_within.py reads.

Each model's vectors are centred on that model's mean profile (removing the shape shared by all its layers);
then for every pair of models, Pearson r between every layer of one and every layer of the other. Inherited
layers show r near 1 (Qwen3-14B vs 32B: 0.97 / 0.98 raw); unrelated layers sit near 0.
"""
import itertools, json, os, struct, sys, urllib.request
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paths import CACHE, GGUF_32B, local

MODELS = [('0.6B', local('Qwen3-0.6B')), ('1.7B', local('Qwen3-1.7B')), ('4B', local('Qwen3-4B')), ('8B', 'hf:Qwen/Qwen3-8B'),
          ('14B', local('Qwen3-14B')), ('32B', GGUF_32B), ('30B-A3B', 'hf:Qwen/Qwen3-30B-A3B'),
          ('235B-A22B', 'hf:Qwen/Qwen3-235B-A22B')]


def bf16(raw):
    return (np.frombuffer(raw, dtype=np.uint16).astype(np.uint32) << 16).view(np.float32)


def get(url, rng=None):
    req = urllib.request.Request(url, headers={'Range': f'bytes={rng[0]}-{rng[1] - 1}'} if rng else {})
    with urllib.request.urlopen(req, timeout=120) as r: return r.read()


def norms(src):
    """dict kind -> [layers, 128] for kind in q, k."""
    if src.endswith('.gguf'):
        from gguf import GGUFReader
        T = {t.name: t for t in GGUFReader(src).tensors}
        L = 1 + max(int(n.split('.')[1]) for n in T if n.startswith('blk.'))
        return {k: np.stack([np.array(T[f'blk.{l}.attn_{k}_norm.weight'].data, dtype=np.float32) for l in range(L)]) for k in 'qk'}
    if src.startswith('hf:'):
        base = f'https://huggingface.co/{src[3:]}/resolve/main/'
        where = json.loads(get(base + 'model.safetensors.index.json'))['weight_map']
        cfg = json.loads(get(base + 'config.json')); L = cfg['num_hidden_layers']
        heads = {}
        out = {}
        for k in 'qk':
            vs = []
            for l in range(L):
                name = f'model.layers.{l}.self_attn.{k}_norm.weight'; fn = where[name]
                if fn not in heads:
                    n = struct.unpack('<Q', get(base + fn, (0, 8)))[0]; heads[fn] = (json.loads(get(base + fn, (8, 8 + n))), 8 + n)
                h, b = heads[fn]; t = h[name]; s, e = t['data_offsets']
                vs.append(bf16(get(base + fn, (b + s, b + e))))
            out[k] = np.stack(vs)
        return out
    idx = os.path.join(src, 'model.safetensors.index.json')
    where = json.load(open(idx))['weight_map'] if os.path.exists(idx) else None
    cfg = json.load(open(os.path.join(src, 'config.json'))); L = cfg['num_hidden_layers']
    out = {}
    for k in 'qk':
        vs = []
        for l in range(L):
            name = f'model.layers.{l}.self_attn.{k}_norm.weight'
            fn = where[name] if where else 'model.safetensors'
            with open(os.path.join(src, fn), 'rb') as f:
                n = struct.unpack('<Q', f.read(8))[0]; h = json.loads(f.read(n)); t = h[name]
                f.seek(8 + n + t['data_offsets'][0]); vs.append(bf16(f.read(t['data_offsets'][1] - t['data_offsets'][0])))
        out[k] = np.stack(vs)
    return out


os.makedirs(CACHE, exist_ok=True)
data = {}
for name, src in MODELS:
    path = os.path.join(CACHE, f'qknorm_{name}.npz')
    if not os.path.exists(path):
        try:
            np.savez(path, **norms(src))
        except Exception as e:
            print(f'{name}: skipped ({e})'); continue
    data[name] = dict(np.load(path))

print('per model pair: for each layer of the shallower model, its best-matching layer in the other by centred q/k-norm r')
print('(mean of the q and k correlations); null = median of the best |r| over all layers when one model is shuffled across channels')
rng = np.random.default_rng(0)
for (na, A), (nb, B) in itertools.combinations(data.items(), 2):
    if len(A['q']) > len(B['q']): (na, A), (nb, B) = (nb, B), (na, A)
    R = 0
    for k in 'qk':
        a = A[k] - A[k].mean(0); b = B[k] - B[k].mean(0)
        a = (a - a.mean(1, keepdims=True)) / a.std(1, keepdims=True); b = (b - b.mean(1, keepdims=True)) / b.std(1, keepdims=True)
        R = R + (a @ b.T / a.shape[1]) / 2
    best, arg = R.max(1), R.argmax(1)
    perm = rng.permutation(128)
    Rn = 0
    for k in 'qk':
        a = A[k] - A[k].mean(0); b = (B[k] - B[k].mean(0))[:, perm]
        a = (a - a.mean(1, keepdims=True)) / a.std(1, keepdims=True); b = (b - b.mean(1, keepdims=True)) / b.std(1, keepdims=True)
        Rn = Rn + (a @ b.T / a.shape[1]) / 2
    print(f'\n{na} ({len(A["q"])} layers) vs {nb} ({len(B["q"])} layers): median best r {np.median(best):.2f} (null {np.median(Rn.max(1)):.2f})')
    print('   ' + ' '.join(f'{l}->{arg[l]}:{best[l]:.2f}' for l in range(len(best))))
