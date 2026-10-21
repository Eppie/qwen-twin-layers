#!/usr/bin/env python3
"""Same-index twin similarity from the native BF16 safetensors of Qwen3-32B, and a check against the GGUF.

    python3 native_replication.py HF_DIR GGUF OUT_DIR

HF_DIR: hf download Qwen/Qwen3-32B --revision 9216db5781bf21249d130ec9da846c4624c16137 --local-dir HF_DIR
GGUF: Qwen3-32B-Q8_0.gguf from Qwen/Qwen3-32B-GGUF. CPU only; needs numpy and gguf (pip install gguf numpy).

Independent of the GGUF path: its own safetensors parser (header JSON + raw BF16, read with numpy), full
tensors, dot products and squared norms accumulated in float64 over row chunks. Then the same tensors are
dequantized from the GGUF and compared coordinate by coordinate; e = ||u - u~|| for the unit-normalised
native (u) and GGUF (u~) tensors bounds how far quantization can move a cosine: |<u_i,u_j> - <u~_i,u~_j>|
<= e_i + e_j.
"""
import json, os, struct, sys, time
import numpy as np

HF, GGUF_PATH, OUT = sys.argv[1], sys.argv[2], sys.argv[3]
os.makedirs(OUT, exist_ok=True)

TWINS = [(8, 32), (12, 36), (16, 40), (20, 44), (24, 48), (28, 52), (31, 55)]
CONTROLS = [(7, 31), (32, 56), (8, 31), (8, 33), (16, 39), (16, 41), (8, 36), (20, 40)]
FAMILIES = {  # HF name -> GGUF name
    'self_attn.q_proj': 'attn_q', 'self_attn.k_proj': 'attn_k', 'self_attn.v_proj': 'attn_v',
    'self_attn.o_proj': 'attn_output', 'mlp.gate_proj': 'ffn_gate', 'mlp.up_proj': 'ffn_up',
    'mlp.down_proj': 'ffn_down'}
NORMS = {'input_layernorm': 'attn_norm', 'post_attention_layernorm': 'ffn_norm',
         'self_attn.q_norm': 'attn_q_norm', 'self_attn.k_norm': 'attn_k_norm'}
CHUNK = 2048


class SafeTensors:
    """Minimal reader: one memory map per shard, tensors returned as numpy views of the raw bytes."""
    def __init__(self, root):
        idx = json.load(open(os.path.join(root, 'model.safetensors.index.json')))
        self.where = idx['weight_map']
        self.root, self.files = root, {}

    def _shard(self, fn):
        if fn not in self.files:
            path = os.path.join(self.root, fn)
            with open(path, 'rb') as f:
                n = struct.unpack('<Q', f.read(8))[0]
                header = json.loads(f.read(n))
            raw = np.memmap(path, dtype=np.uint8, mode='r')
            self.files[fn] = (header, raw, 8 + n)
        return self.files[fn]

    def info(self, name):
        header, _, _ = self._shard(self.where[name])
        return header[name]

    def get(self, name):
        header, raw, base = self._shard(self.where[name])
        h = header[name]
        assert h['dtype'] == 'BF16', h['dtype']
        b, e = h['data_offsets']
        u16 = raw[base + b: base + e].view(np.uint16).reshape(h['shape'])
        return u16

    def names(self):
        return list(self.where)


def bf16_to_f64(u16):
    return (u16.astype(np.uint32) << 16).view(np.float32).astype(np.float64)


def full_cosine(a16, b16):
    dot = na = nb = 0.0
    for r in range(0, a16.shape[0], CHUNK):
        a, b = bf16_to_f64(a16[r:r + CHUNK]), bf16_to_f64(b16[r:r + CHUNK])
        dot += float(np.sum(a * b)); na += float(np.sum(a * a)); nb += float(np.sum(b * b))
    return dot / np.sqrt(na * nb), dot, na, nb


st = SafeTensors(HF)
log = open(os.path.join(OUT, 'native_replication.txt'), 'w')
def out(s=''):
    print(s); log.write(s + '\n'); log.flush()

# --- inventory and provenance
layer_params = sorted({n.split('.', 3)[3] for n in st.names() if n.startswith('model.layers.')})
other = sorted(n for n in st.names() if not n.startswith('model.layers.'))
cfg = json.load(open(os.path.join(HF, 'config.json')))
out(f'checkpoint: {os.path.basename(os.path.normpath(HF))}; architectures {cfg.get("architectures")}; layers {cfg.get("num_hidden_layers")}; '
    f'rms_norm_eps {cfg.get("rms_norm_eps")}; tie_word_embeddings {cfg.get("tie_word_embeddings")}')
out(f'per-layer parameters ({len(layer_params)}): ' + ', '.join(layer_params))
out(f'non-layer parameters: ' + ', '.join(other))
missing = [p for p in layer_params if p.rsplit('.', 1)[0] not in FAMILIES and p.rsplit('.', 1)[0] not in NORMS]
out(f'per-layer parameters not covered by this analysis: {missing or "none"}')
for p in FAMILIES:
    h = st.info(f'model.layers.0.{p}.weight'); out(f'  {p}: shape {h["shape"]} {h["dtype"]}')

# --- full-tensor cosines
out('\n## Full-tensor cosine (float64), native BF16')
rows = []
pairs = [('twin', a, b) for a, b in TWINS] + [('control', a, b) for a, b in CONTROLS]
hdr = f'{"kind":8s} {"pair":>8s} ' + ' '.join(f'{v:>11s}' for v in FAMILIES.values()) + f' {"mean":>8s}'
out(hdr)
t0 = time.time()
for kind, a, b in pairs:
    cs = []
    for p, g in FAMILIES.items():
        c, dot, na, nb = full_cosine(st.get(f'model.layers.{a}.{p}.weight'), st.get(f'model.layers.{b}.{p}.weight'))
        cs.append(c); rows.append(dict(kind=kind, a=a, b=b, family=g, cos=c, dot=dot, norm2_a=na, norm2_b=nb))
    out(f'{kind:8s} {a:>3d},{b:<4d} ' + ' '.join(f'{c:11.4f}' for c in cs) + f' {np.mean(cs):8.4f}')
out(f'({time.time() - t0:.0f}s)')
json.dump(rows, open(os.path.join(OUT, 'full_cosines.json'), 'w'), indent=1)

# --- norm gains, native
out('\n## Norm gains (native), Pearson r, raw')
for p, g in NORMS.items():
    vals = []
    for kind, a, b in pairs:
        x = bf16_to_f64(st.get(f'model.layers.{a}.{p}.weight')); y = bf16_to_f64(st.get(f'model.layers.{b}.{p}.weight'))
        vals.append(f'{a},{b}:{np.corrcoef(x, y)[0, 1]:+.3f}')
    out(f'{g:12s} ' + '  '.join(vals))

# --- GGUF vs native, coordinate level
from gguf import GGUFReader
from gguf.quants import dequantize
T = {t.name: t for t in GGUFReader(GGUF_PATH).tensors}

def gguf_full(name):
    t = T[name]
    return np.concatenate([dequantize(t.data[r:r + CHUNK], t.tensor_type).astype(np.float64)
                           for r in range(0, t.data.shape[0], CHUNK)])

out('\n## GGUF (Q8_0) vs native, same tensors')
out('layout: GGUF row r of blk.L.<t> compared with native row r of model.layers.L.<p> (no transpose / permutation)')
layers = sorted({l for _, a, b in pairs for l in (a, b)})
unit = {}
for p, g in FAMILIES.items():
    worst_cos, worst_e, worst_rel = 1.0, 0.0, 0.0
    for l in layers:
        nat = st.get(f'model.layers.{l}.{p}.weight')
        q = gguf_full(f'blk.{l}.{g}.weight')
        assert q.shape == nat.shape, (p, q.shape, nat.shape)
        dot = nn = qq = d2 = 0.0
        for r in range(0, nat.shape[0], CHUNK):
            x = bf16_to_f64(nat[r:r + CHUNK]); y = q[r:r + CHUNK]
            dot += np.sum(x * y); nn += np.sum(x * x); qq += np.sum(y * y); d2 += np.sum((x - y) ** 2)
        cos = dot / np.sqrt(nn * qq)
        e = np.sqrt(max(0.0, 2 - 2 * cos))  # ||u - u~|| for unit vectors
        unit[(l, g)] = e
        worst_cos, worst_e, worst_rel = min(worst_cos, cos), max(worst_e, e), max(worst_rel, np.sqrt(d2 / nn))
    out(f'{g:12s} over {len(layers)} layers: min cos(native, gguf) {worst_cos:.6f}; max e {worst_e:.4f}; '
        f'max relative error {worst_rel:.4f}')
for p, g in NORMS.items():
    dmax = max(np.max(np.abs(bf16_to_f64(st.get(f'model.layers.{l}.{p}.weight')) -
                             np.asarray(T[f'blk.{l}.{g}.weight'].data, dtype=np.float64))) for l in layers)
    out(f'{g:12s} max |native - gguf| over {len(layers)} layers: {dmax:.2e}')

out('\n## Bound on the quantization effect for each measured cosine: e_a + e_b (max over pairs, per family)')
for g in FAMILIES.values():
    out(f'{g:12s} max e_a + e_b = {max(unit[(a, g)] + unit[(b, g)] for _, a, b in pairs):.4f}')
log.close()
