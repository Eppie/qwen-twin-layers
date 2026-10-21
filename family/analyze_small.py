#!/usr/bin/env python3
"""Twin layers, width copies and cross-model residual-basis relations in small Qwen3 checkpoints (BF16 safetensors).

    python3 family/analyze_small.py twins MODEL_DIR          # same-index similarity of every layer pair
    python3 family/analyze_small.py width MODEL_DIR          # within-layer duplicate FFN neurons / attention heads
    python3 family/analyze_small.py embed SPEC SPEC ...      # residual dims matched through embedding columns

MODEL_DIR is a local BF16 checkpoint directory (its name is printed, e.g. Qwen3-4B). SPEC for `embed`: name=path,
where path is a local HF dir, a GGUF file, or hf:Repo/Name (embedding rows fetched with HTTP range requests). All
Qwen3 models share one tokenizer, so each residual dimension has a profile over the same token rows (the first
N_TOK rows). The embedding slices are cached in paths.CACHE as NAME_embed30000.npy (pair_4_8.py reuses them).
"""
import itertools, json, os, struct, sys, urllib.request
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paths import CACHE

N_TOK = 30000


def bf16(u):
    return (np.asarray(u, dtype=np.uint16).astype(np.uint32) << 16).view(np.float32)


class Local:
    def __init__(self, root):
        self.root, self.cache = root, {}
        idx = os.path.join(root, 'model.safetensors.index.json')
        if os.path.exists(idx):
            self.where = json.load(open(idx))['weight_map']
        else:  # single-file checkpoint
            with open(os.path.join(root, 'model.safetensors'), 'rb') as f:
                n = struct.unpack('<Q', f.read(8))[0]; h = json.loads(f.read(n))
            self.where = {k: 'model.safetensors' for k in h if k != '__metadata__'}

    def info(self, name):
        fn = self.where[name]
        if fn not in self.cache:
            p = os.path.join(self.root, fn)
            with open(p, 'rb') as f:
                n = struct.unpack('<Q', f.read(8))[0]; h = json.loads(f.read(n))
            self.cache[fn] = (h, p, 8 + n)
        h, p, base = self.cache[fn]
        return h[name], p, base

    def get(self, name, r0=0, r1=None):
        t, p, base = self.info(name)
        assert t['dtype'] == 'BF16', t['dtype']
        shape = t['shape']; r1 = shape[0] if r1 is None else r1
        rb = int(np.prod(shape[1:])) * 2
        with open(p, 'rb') as f:
            f.seek(base + t['data_offsets'][0] + r0 * rb)
            u = np.frombuffer(f.read((r1 - r0) * rb), dtype=np.uint16)
        return bf16(u).reshape([r1 - r0] + shape[1:])


# ---------------------------------------------------------------- twins
def cmd_twins(root):
    m = Local(root); cfg = json.load(open(os.path.join(root, 'config.json')))
    L = cfg['num_hidden_layers']
    fams = ['self_attn.q_proj', 'self_attn.k_proj', 'self_attn.v_proj', 'self_attn.o_proj',
            'mlp.gate_proj', 'mlp.up_proj', 'mlp.down_proj']
    norms = ['self_attn.q_norm', 'self_attn.k_norm', 'input_layernorm', 'post_attention_layernorm']
    print(f'{os.path.basename(root)}: {L} layers, hidden {cfg["hidden_size"]}, ffn {cfg["intermediate_size"]}, '
          f'heads {cfg["num_attention_heads"]}/{cfg["num_key_value_heads"]}')
    pooled = np.zeros((L, L))
    for f in fams:
        rows = m.info(f'model.layers.0.{f}.weight')[0]['shape'][0]
        G = np.zeros((L, L))
        for r in range(0, rows, 1024):
            X = np.stack([m.get(f'model.layers.{l}.{f}.weight', r, min(rows, r + 1024)).ravel() for l in range(L)]).astype(np.float64)
            G += X @ X.T
        n = np.sqrt(np.diag(G)); S = G / np.outer(n, n); pooled += S / len(fams)
        report(f.split('.')[-1], S, L)
    report('all matrices', pooled, L)
    np.fill_diagonal(pooled, -np.inf)
    print('best partner per layer (pooled): ' + ' '.join(f'{l}->{int(pooled[l].argmax())}({pooled[l].max():+.3f})' for l in range(L)))
    for f in norms:
        G = np.stack([m.get(f'model.layers.{l}.{f}.weight').astype(np.float64) for l in range(L)])
        report(f.split('.')[-1] + ' centred r', np.corrcoef(G - G.mean(0)), L)


def report(name, S, L):
    off = {}
    for i in range(L):
        for j in range(i + 1, L): off.setdefault(j - i, []).append(S[i, j])
    allv = np.concatenate(list(off.values()))
    means = {d: np.mean(v) for d, v in off.items()}
    top = sorted(means, key=means.get, reverse=True)[:3]
    iu = np.triu_indices(L, 1); vals = S[iu]; k = np.argsort(-vals)[:3]
    print(f'  {name:26s} all pairs {allv.mean():+.4f} ± {allv.std():.4f} | top offsets ' +
          ' '.join(f'+{d}:{means[d]:+.4f}' for d in top) + ' | top pairs ' +
          ' '.join(f'({iu[0][x]},{iu[1][x]}):{vals[x]:+.3f}' for x in k))


# ---------------------------------------------------------------- width copies
def cmd_width(root):
    m = Local(root); cfg = json.load(open(os.path.join(root, 'config.json')))
    L, F, H = cfg['num_hidden_layers'], cfg['intermediate_size'], cfg['num_attention_heads']
    KV, hd = cfg['num_key_value_heads'], cfg.get('head_dim', 128)
    print(f'{os.path.basename(root)}: FFN {F} neurons, {H} query heads in {KV} KV groups')
    for l in sorted(set([0, 1, L // 4, L // 2, 3 * L // 4, L - 2, L - 1])):
        g = m.get(f'model.layers.{l}.mlp.gate_proj.weight'); u = m.get(f'model.layers.{l}.mlp.up_proj.weight')
        d = m.get(f'model.layers.{l}.mlp.down_proj.weight').T
        S = 0
        for W in (g, u, d):
            W = W / np.linalg.norm(W, axis=1, keepdims=True); S = S + (W @ W.T) / 3
        np.fill_diagonal(S, -np.inf)
        best, val = S.argmax(1), S.max(1)
        thr = 0.1
        strong = val > thr
        offs = (best - np.arange(F))[strong]
        common = np.unique(np.abs(offs), return_counts=True)
        top = sorted(zip(common[1], common[0]), reverse=True)[:3]
        mutual = np.mean(best[best[strong]] == np.arange(F)[strong]) if strong.any() else 0
        print(f'  layer {l:2d} FFN: best within-layer match (mean of gate/up/down cosines): median {np.median(val):.3f}, '
              f'>{thr} for {strong.mean():.1%} of neurons (mutual {mutual:.0%}); commonest |offset| ' +
              ', '.join(f'{o} x{c}' for c, o in top))
        q = m.get(f'model.layers.{l}.self_attn.q_proj.weight').reshape(H, hd, -1).reshape(H, -1)
        o = m.get(f'model.layers.{l}.self_attn.o_proj.weight').reshape(-1, H, hd).transpose(1, 0, 2).reshape(H, -1)
        Sh = 0
        for W in (q, o):
            W = W / np.linalg.norm(W, axis=1, keepdims=True); Sh = Sh + (W @ W.T) / 2
        np.fill_diagonal(Sh, -np.inf)
        bh = Sh.argmax(1)
        pairs = sorted({tuple(sorted((h, int(bh[h])))) for h in range(H) if bh[bh[h]] == h and Sh[h, bh[h]] > 0.1})
        print(f'           heads: max off-diagonal {np.max(Sh):.3f}, median best {np.median(Sh.max(1)):.3f}; '
              f'mutual pairs above 0.1: {pairs[:12]}{" ..." if len(pairs) > 12 else ""}')


# ---------------------------------------------------------------- cross-model embeddings
def remote_rows(repo, name, r0, r1):
    base = f'https://huggingface.co/{repo}/resolve/main/'
    def get(url, rng=None):
        req = urllib.request.Request(url, headers={'Range': f'bytes={rng[0]}-{rng[1] - 1}'} if rng else {})
        with urllib.request.urlopen(req, timeout=300) as r: return r.read()
    try:
        where = json.loads(get(base + 'model.safetensors.index.json'))['weight_map']; fn = where[name]
    except Exception:
        fn = 'model.safetensors'
    n = struct.unpack('<Q', get(base + fn, (0, 8)))[0]
    h = json.loads(get(base + fn, (8, 8 + n)))[name]
    rb = int(np.prod(h['shape'][1:])) * 2; b = 8 + n + h['data_offsets'][0]
    return bf16(np.frombuffer(get(base + fn, (b + r0 * rb, b + r1 * rb)), dtype=np.uint16)).reshape(r1 - r0, -1)


def embedding(spec):
    name, path = spec.split('=', 1)
    cache = os.path.join(CACHE, name + f'_embed{N_TOK}.npy')
    if os.path.exists(cache): return name, np.load(cache)
    if path.startswith('hf:'):
        E = remote_rows(path[3:], 'model.embed_tokens.weight', 0, N_TOK)
    elif path.endswith('.gguf'):
        from gguf import GGUFReader
        from gguf.quants import dequantize
        t = {x.name: x for x in GGUFReader(path).tensors}['token_embd.weight']
        E = dequantize(t.data[:N_TOK], t.tensor_type).astype(np.float32)
    else:
        E = Local(path).get('model.embed_tokens.weight', 0, N_TOK)
    os.makedirs(os.path.dirname(cache), exist_ok=True); np.save(cache, E)
    return name, E


def cmd_embed(specs):
    embs = [embedding(s) for s in specs]
    print(f'residual dimensions matched through embedding columns over the first {N_TOK} tokens')
    print('for each dim of the narrower model: best |corr| with any dim of the wider one; null = same with tokens shuffled')
    rng = np.random.default_rng(0)
    for (na, A), (nb, B) in itertools.combinations(embs, 2):
        if A.shape[1] > B.shape[1]: (na, A), (nb, B) = (nb, B), (na, A)
        Za = (A - A.mean(0)) / A.std(0); Zb = (B - B.mean(0)) / B.std(0)
        C = Za.T @ Zb / len(Za)
        best = np.abs(C).max(1); arg = np.abs(C).argmax(1)
        Zs = Zb[rng.permutation(len(Zb))]
        null = np.abs(Za.T @ Zs / len(Za)).max(1)
        ident = np.mean(arg == np.arange(len(arg)))
        print(f'  {na:>8s} ({A.shape[1]}) vs {nb:>8s} ({B.shape[1]}): median best |corr| {np.median(best):.3f} '
              f'(null {np.median(null):.3f}); dims > 0.5: {np.mean(best > 0.5):.0%}; '
              f'best match is the same index for {ident:.0%}; distinct targets {len(set(arg))}/{len(arg)}')


if __name__ == '__main__':
    cmd = sys.argv[1]
    if cmd == 'twins': cmd_twins(sys.argv[2])
    elif cmd == 'width': cmd_width(sys.argv[2])
    elif cmd == 'embed': cmd_embed(sys.argv[2:])
