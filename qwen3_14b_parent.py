#!/usr/bin/env python3
"""Does the released Qwen3-14B line up index by index with Qwen3-32B where a width + depth expansion predicts?

    python3 qwen3_14b_parent.py Qwen3-32B-Q8_0.gguf [--out DIR] [--cache DIR] [--compare REF.npz]

Qwen3-32B is read from its Q8_0 GGUF (same row layout as the Hugging Face safetensors: no transpose and no
Q/K row permutation). Selected Qwen3-14B tensors are fetched from Hugging Face with HTTP range requests on the
safetensors shards (about 1.4 GB in total; no full download):
  all 40 layers: k_proj and v_proj (8 KV heads x 128 rows, the same shape as in the 32B), q_norm, k_norm, and
                 the first 512 rows of gate_proj and up_proj;
  layers 0, 20, 39: the full q_proj (40 heads x 128 rows).

Part 1 compares every 14B layer with every 32B layer, same index (signed cosine of the flattened tensors in
float64; Pearson r for the q/k-norm gains). A 14B with 40 layers stacked as layers 0-31 followed by layers 8-39
predicts that 14B layer p matches 32B layer p (p <= 31) and 32B layer p + 24 (p >= 8), and nothing else. The
32B FFN has 25600 = 17408 + 8192 neurons; 'up_proj_copy' compares 14B up rows 0..511 with 32B up rows
17408..17919 (the width copies of neurons 0..511 found within the 32B).
Part 2 maps query heads: each 14B head (128 x 5120 block of q_proj) against every 32B head, cosine. 14B has 5
query heads per KV group, 32B has 8; the predicted map is 14B head (g, i) -> 32B heads (g, 3 + i) and, for
i < 3, also (g, i).

--compare REF.npz prints the largest differences from a reference run (e.g. the native-BF16 run in
results/native/), which bounds the effect of Q8_0 quantization on these numbers.
The cache holds the fetched 14B tensors as .npy files; without --cache a temporary directory is used and
deleted at the end. Needs numpy and gguf (pip install gguf numpy).
"""
import argparse, json, os, shutil, struct, sys, tempfile, time, urllib.request
import numpy as np
from gguf import GGUFReader
from gguf.quants import dequantize

ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
ap.add_argument('gguf', help='Qwen3-32B-Q8_0.gguf (Qwen/Qwen3-32B-GGUF)')
ap.add_argument('--out', default=os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results', 'qwen3_14b'))
ap.add_argument('--cache', help='directory for fetched Qwen3-14B tensors (kept); default: temporary, deleted')
ap.add_argument('--repo', default='Qwen/Qwen3-14B')
ap.add_argument('--revision', default='40c069824f4251a91eefaf281ebe4c544efd3e18')
ap.add_argument('--compare', help='reference .npz from an earlier run (e.g. native BF16 32B)')
args = ap.parse_args()
os.makedirs(args.out, exist_ok=True)
CACHE = args.cache or tempfile.mkdtemp(prefix='qwen3_14b_')
os.makedirs(CACHE, exist_ok=True)
BASE = f'https://huggingface.co/{args.repo}/resolve/{args.revision}/'
N_ROWS, COPY_START = 512, 17408
HEAD_LAYERS = [(0, [0]), (20, [20, 44]), (39, [63])]   # (14B layer, 32B layers) for the head map


def get(url, rng=None, tries=5):
    for t in range(tries):
        try:
            req = urllib.request.Request(url, headers={'Range': f'bytes={rng[0]}-{rng[1] - 1}'} if rng else {})
            with urllib.request.urlopen(req, timeout=120) as r:
                data = r.read()
            if rng and len(data) != rng[1] - rng[0]:
                raise IOError(f'short read {len(data)} of {rng[1] - rng[0]}')
            return data
        except Exception as e:
            if t == tries - 1:
                raise
            print(f'  retry {url} ({e})', file=sys.stderr)
            time.sleep(2 * (t + 1))


class Remote:
    """Qwen3-14B safetensors on Hugging Face, read tensor by tensor with HTTP range requests."""
    def __init__(self):
        self.where = json.loads(get(BASE + 'model.safetensors.index.json'))['weight_map']
        self.headers = {}

    def header(self, fn):
        if fn not in self.headers:
            n = struct.unpack('<Q', get(BASE + fn, (0, 8)))[0]
            self.headers[fn] = (json.loads(get(BASE + fn, (8, 8 + n))), 8 + n)
        return self.headers[fn]

    def tensor(self, name, rows=None):
        path = os.path.join(CACHE, name + (f'.r{rows}' if rows else '') + '.npy')
        if os.path.exists(path):
            return np.load(path)
        fn = self.where[name]
        h, base = self.header(fn)
        t = h[name]
        assert t['dtype'] == 'BF16', t['dtype']
        b, e = t['data_offsets']
        shape = list(t['shape'])
        if rows:
            e = b + rows * int(np.prod(shape[1:])) * 2
            shape[0] = rows
        u = np.frombuffer(get(BASE + fn, (base + b, base + e)), dtype=np.uint16).reshape(shape)
        x = (u.astype(np.uint32) << 16).view(np.float32)
        np.save(path, x)
        return x


T = {t.name: t for t in GGUFReader(args.gguf).tensors}


def g32(name, start=0, rows=None):
    """Dequantised float32 GGUF tensor (storage rows), optionally rows start..start+rows."""
    t = T[name]
    d = t.data
    if d.ndim > 1:
        d = d.reshape(-1, d.shape[-1])
        if rows:
            d = d[start:start + rows]
    return np.asarray(dequantize(np.asarray(d), t.tensor_type), dtype=np.float32)


def cos(a, b):
    a, b = a.astype(np.float64).ravel(), b.astype(np.float64).ravel()
    return float(a @ b / np.sqrt((a @ a) * (b @ b)))


log = open(os.path.join(args.out, 'qwen3_14b_parent.txt'), 'w')
def out(s=''):
    print(s, flush=True)
    log.write(s + '\n')
    log.flush()


r14 = Remote()
cfg = json.loads(get(BASE + 'config.json'))
out(f'{args.repo} @ {args.revision} config: layers {cfg["num_hidden_layers"]}, hidden {cfg["hidden_size"]}, '
    f'ffn {cfg["intermediate_size"]}, heads {cfg["num_attention_heads"]}/{cfg["num_key_value_heads"]}')
out(f'Qwen3-32B side: {os.path.basename(args.gguf)} (Q8_0, dequantised)')

# ------------------------------------------------------------------------------------- part 1: layers
P = cfg['num_hidden_layers']
L32 = 1 + max(int(n.split('.')[1]) for n in T if n.startswith('blk.'))
FAM = {'k_proj': ('self_attn.k_proj', 'attn_k'), 'v_proj': ('self_attn.v_proj', 'attn_v')}
X32 = {}  # 32B tensors used for every 14B layer, read once
for m in range(L32):
    for f, (_, gn) in FAM.items():
        X32[(m, f)] = g32(f'blk.{m}.{gn}.weight')
    X32[(m, 'gate')] = g32(f'blk.{m}.ffn_gate.weight', 0, N_ROWS)
    X32[(m, 'up')] = g32(f'blk.{m}.ffn_up.weight', 0, N_ROWS)
    X32[(m, 'upc')] = g32(f'blk.{m}.ffn_up.weight', COPY_START, N_ROWS)
    X32[(m, 'q')] = g32(f'blk.{m}.attn_q_norm.weight')
    X32[(m, 'k')] = g32(f'blk.{m}.attn_k_norm.weight')
M = {f: np.zeros((P, L32)) for f in list(FAM) + ['gate_proj', 'up_proj', 'up_proj_copy', 'q_norm', 'k_norm']}
for p in range(P):
    a14 = {f: r14.tensor(f'model.layers.{p}.{hn}.weight') for f, (hn, _) in FAM.items()}
    g14 = r14.tensor(f'model.layers.{p}.mlp.gate_proj.weight', N_ROWS)
    u14 = r14.tensor(f'model.layers.{p}.mlp.up_proj.weight', N_ROWS)
    q14 = r14.tensor(f'model.layers.{p}.self_attn.q_norm.weight')
    k14 = r14.tensor(f'model.layers.{p}.self_attn.k_norm.weight')
    for m in range(L32):
        for f in FAM:
            M[f][p, m] = cos(a14[f], X32[(m, f)])
        M['gate_proj'][p, m] = cos(g14, X32[(m, 'gate')])
        M['up_proj'][p, m] = cos(u14, X32[(m, 'up')])
        M['up_proj_copy'][p, m] = cos(u14, X32[(m, 'upc')])
        M['q_norm'][p, m] = np.corrcoef(q14, X32[(m, 'q')])[0, 1]
        M['k_norm'][p, m] = np.corrcoef(k14, X32[(m, 'k')])[0, 1]
    out(f'14B layer {p:2d}: best 32B match by k_proj ' + ', '.join(
        f'{m}({M["k_proj"][p, m]:+.3f})' for m in np.argsort(-M['k_proj'][p])[:3]) +
        f' | v {np.max(M["v_proj"][p]):+.3f} gate {np.max(M["gate_proj"][p]):+.3f} up {np.max(M["up_proj"][p]):+.3f}'
        f' up({COPY_START}+j) {np.max(M["up_proj_copy"][p]):+.3f}')
del X32
np.savez(os.path.join(args.out, 'qwen3_14b_parent.npz'), **M)

expect = {p: [p] if p < 8 else ([p, p + 24] if p < 32 else [p + 24]) for p in range(P)}
out('\nmean over 14B layers of the cosine at the expected 32B layers vs all other 32B layers')
for f, X in M.items():
    hit = [X[p, m] for p in range(P) for m in expect[p]]
    miss = [X[p, m] for p in range(P) for m in range(L32) if m not in expect[p]]
    out(f'  {f:13s} expected {np.mean(hit):+.4f} (min {np.min(hit):+.4f})   others {np.mean(miss):+.4f} ± {np.std(miss):.4f} '
        f'(max {np.max(miss):+.4f})')
best_ok = sum(set(np.argsort(-M['k_proj'][p])[:len(expect[p])]) == set(expect[p]) for p in range(P))
out(f'14B layers whose best k_proj matches (top 1 for p < 8 or p > 31, top 2 for 8 <= p <= 31) are exactly the '
    f'expected 32B layers: {best_ok}/{P}')

out('\n14B layers 8-31 (two copies in the 32B): lower copy (32B layer p) vs upper copy (32B layer p + 24)')
for f, X in M.items():
    lo = np.array([X[p, p] for p in range(8, 32)])
    up = np.array([X[p, p + 24] for p in range(8, 32)])
    out(f'  {f:13s} lower {lo.mean():+.3f}  upper {up.mean():+.3f}  upper closer in {int(np.sum(up > lo))}/24; '
        f'lower closer at p = {[p for p in range(8, 32) if X[p, p] > X[p, p + 24]]}')

if args.compare:
    ref = np.load(args.compare)
    out(f'\ndifference from reference {os.path.basename(args.compare)} (this run minus reference), over all '
        f'{P} x {L32} layer pairs and at the expected pairs')
    for f in M:
        if f not in ref.files:
            continue
        d = M[f] - ref[f]
        de = np.array([d[p, m] for p in range(P) for m in expect[p]])
        out(f'  {f:13s} max |diff| {np.abs(d).max():.4f} (mean {np.abs(d).mean():.5f}); expected pairs: max |diff| '
            f'{np.abs(de).max():.4f}, mean diff {de.mean():+.5f}')

# ------------------------------------------------------------------------------------- part 2: query heads
H14, KV = cfg['num_attention_heads'], cfg['num_key_value_heads']
HD = cfg.get('head_dim', 128)
D = cfg['hidden_size']
per14 = H14 // KV
out(f'\nquery-head map: each 14B head (layer p) -> best-matching 32B head by q_proj rows ({HD} x {D}), cosine')
for p, ms in HEAD_LAYERS:
    q14 = r14.tensor(f'model.layers.{p}.self_attn.q_proj.weight').reshape(H14, HD * D)
    for m in ms:
        q32 = g32(f'blk.{m}.attn_q.weight')
        H32 = q32.shape[0] // HD
        per32 = H32 // KV
        a = q14.astype(np.float64)
        b = q32.reshape(H32, HD * D).astype(np.float64)
        C = (a / np.linalg.norm(a, axis=1, keepdims=True)) @ (b / np.linalg.norm(b, axis=1, keepdims=True)).T
        best = C.argmax(1)
        groups = [f'{h // per14}:{h % per14}->{best[h] // per32}:{best[h] % per32}' for h in range(H14)]
        second = [int(np.argsort(-C[h])[1]) for h in range(H14)]
        out(f'14B layer {p} vs 32B layer {m}: best cos median {np.median(C.max(1)):.3f}; same KV group '
            f'{np.mean(best // per32 == np.arange(H14) // per14):.2f}; off-best median {np.median(np.sort(C, 1)[:, -3]):.3f}')
        out('   (group:local) ' + ' '.join(groups[:15]) + ' ...')
        out('   second-best for heads 0-4 of group 0: ' + ', '.join(
            f'{s // per32}:{s % per32} ({C[h, s]:.2f})' for h, s in zip(range(5), second[:5])))
        # predicted map: 14B (g, i) -> 32B (g, per32 - per14 + i), and (g, i) for i < per32 - per14
        shift = per32 - per14
        pred = [{(h // per14) * per32 + shift + h % per14} | ({(h // per14) * per32 + h % per14} if h % per14 < shift else set())
                for h in range(H14)]
        is_pred = np.zeros_like(C, dtype=bool)
        for h, s in enumerate(pred):
            is_pred[h, list(s)] = True
        hit = np.mean([best[h] in pred[h] for h in range(H14)])
        dup = [h for h in range(H14) if h % per14 < shift]
        both = np.mean([set(np.argsort(-C[h])[:2]) == pred[h] for h in dup])
        cp, co = C[is_pred], np.where(is_pred, -np.inf, C).max(1)
        out(f'   predicted map: best match is a predicted head for {hit:.2f} of {H14} heads; for the {len(dup)} heads with '
            f'two predicted copies, the top two are exactly those copies for {both:.2f}; cos at predicted heads median '
            f'{np.median(cp):.3f} (min {cp.min():.3f}), best non-predicted head median {np.median(co):.3f} (max {co.max():.3f})')
log.close()
if not args.cache:
    shutil.rmtree(CACHE, ignore_errors=True)
