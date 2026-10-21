#!/usr/bin/env python3
"""Do Qwen3-14B and Qwen3-32B share residual-stream coordinates in their activations, layer by layer?

    python3 family/acts_14_32.py HF_14B_DIR DUMP_PREFIX OUT_PREFIX

DUMP_PREFIX.bin / .idx: the Qwen3-32B residual stream after every layer (0-63) for a set of texts.
  .idx: a header line "layers 0 1 ... 63", then one line per text: "i T tok_1 ... tok_T" (text index, token count,
        token ids);
  .bin: per text, per layer in header order, T x 5120 float16 values (row-major, token by token).
The published numbers come from a dump made with llama.cpp on the Q8_0 GGUF (a private tool capturing the l_out-L
tensors), not included here; dump_residuals_hf.py writes the same format from the BF16 checkpoint with
transformers. The 14B runs on exactly the token ids in the .idx (forward hooks on every decoder layer's output,
i.e. the residual stream after the layer), on Apple-silicon MPS by default (TORCH_DEVICE to change). Memory: the
BF16 14B (about 30 GB) plus about 20 GB of activations for the 10,179 tokens used (both models, all layers).
For each 14B layer p and 32B layer m: Pearson r of every residual dim d with the same dim d in the other model,
across all tokens (median over dims), and for selected pairs the fraction of 14B dims whose best-correlated 32B
dim is the same index. Control: dims shuffled. Saves the 40 x 64 matrix of medians as OUT_PREFIX_samedim_r.npy.
"""
import os, sys
import numpy as np, torch
from transformers import AutoModelForCausalLM

HF, PRE, OUT = sys.argv[1], sys.argv[2], sys.argv[3]
DEVICE = os.environ.get('TORCH_DEVICE', 'mps')
lines = open(PRE + '.idx').read().strip().split('\n')
layers32 = [int(x) for x in lines[0].split()[1:]]
docs = [list(map(int, l.split()[2:])) for l in lines[1:]]
T = sum(map(len, docs))
raw = np.memmap(PRE + '.bin', dtype=np.float16, mode='r')
H = raw.size // (T * len(layers32)); assert H * T * len(layers32) == raw.size and H == 5120, H
A32 = np.zeros((len(layers32), T, H), dtype=np.float16)
off, t0 = 0, 0
for d in docs:
    n = len(d)
    for k in range(len(layers32)):
        A32[k, t0:t0 + n] = raw[off:off + n * H].reshape(n, H); off += n * H
    t0 += n
print(f'32B: {len(docs)} texts, {T} tokens, {len(layers32)} layers', flush=True)

model = AutoModelForCausalLM.from_pretrained(HF, dtype=torch.bfloat16, device_map={'': DEVICE}); model.eval()
L14 = len(model.model.layers)
A14 = np.zeros((L14, T, H), dtype=np.float16)
cur = {}
hooks = [model.model.layers[l].register_forward_hook(lambda m, i, o, l=l: cur.__setitem__(l, (o[0] if isinstance(o, tuple) else o)[0].float().cpu().numpy()))
         for l in range(L14)]
t0 = 0
with torch.no_grad():
    for d in docs:
        model.model(input_ids=torch.tensor([d], device=DEVICE), use_cache=False)
        for l in range(L14): A14[l, t0:t0 + len(d)] = cur[l]
        t0 += len(d)
for h in hooks: h.remove()
del model
if DEVICE == 'mps': torch.mps.empty_cache()
print(f'14B: {L14} layers captured', flush=True)

def zs(X):
    X = X.astype(np.float32); X -= X.mean(0); X /= (X.std(0) + 1e-6); return X

Z14 = [zs(A14[l]) for l in range(L14)]
R = np.zeros((L14, len(layers32)))
for k in range(len(layers32)):
    Z = zs(A32[k])
    for l in range(L14):
        R[l, k] = np.median((Z14[l] * Z).mean(0))
np.save(OUT + '_samedim_r.npy', R)
rng = np.random.default_rng(0); perm = rng.permutation(H)
print('\nmedian over dims of same-dim Pearson r; for each 14B layer: its best 32B layer, and the predicted ones')
for l in range(L14):
    pred = [l] if l < 8 else ([l, l + 24] if l < 32 else [l + 24])
    best = int(R[l].argmax())
    null = np.median((Z14[l] * zs(A32[layers32.index(pred[-1])])[:, perm]).mean(0))
    print(f'  14B layer {l:2d}: best 32B layer {best:2d} (r {R[l, best]:.3f}); predicted ' +
          ', '.join(f'{m}: {R[l, m]:.3f}' for m in pred) + f'; dims shuffled {null:+.3f}; median over all 32B layers {np.median(R[l]):.3f}')

print('\nbest-match identity: for 14B layer p vs 32B layer m, fraction of 14B dims whose most-correlated 32B dim is the same index')
for l, m in [(4, 4), (12, 12), (12, 36), (20, 20), (20, 44), (36, 60), (20, 30)]:
    C = (Z14[l].T @ zs(A32[m])) / T
    print(f'  14B {l:2d} vs 32B {m:2d}: same-index best match {np.mean(np.abs(C).argmax(1) == np.arange(H)):.0%}; '
          f'median same-dim r {np.median(np.diag(C)):.3f}')
