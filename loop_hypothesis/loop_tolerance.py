#!/usr/bin/env python3
"""Does the released Qwen3-14B tolerate having a block of its layers applied twice (as a looped model would)?

    python3 loop_hypothesis/loop_tolerance.py HF_14B_DIR OUT_DIR WEB_TXT MIXED_TXT CODE_TXT

HF_14B_DIR: the BF16 Qwen3-14B checkpoint (revision 40c069824f4251a91eefaf281ebe4c544efd3e18 was used).
WEB_TXT, MIXED_TXT, CODE_TXT: text files of documents separated by the ASCII record separator 0x1E (English web,
mixed multilingual/web/code/math/chat, source code). Documents are taken from fixed offsets (1300, 3300, 120) on,
past those used by the other scripts; the first 8 per domain with at least 512 tokens are cut to 512 tokens.
Runs the BF16 model on Apple-silicon MPS (set TORCH_DEVICE to change); needs torch and transformers and ~30 GB.

Layer modules are reused, not copied, so a looped configuration costs no extra memory. Mean next-token NLL on
held-out text for the normal model and for every 24-layer window [s, s+23] applied twice (s = 0, 2, ..., 16)
or three times (s = 0, 8, 16). Uncertainty: bootstrap over sequences of the NLL difference to the normal model.
See predictions.md (written before running). Writes OUT_DIR/loop_tolerance.txt and loop_tolerance.json.
"""
import json, os, sys, time
import numpy as np, torch
from transformers import AutoModelForCausalLM, AutoTokenizer

HF, OUT, WEB, MIXED, CODE = sys.argv[1:6]
DEVICE = os.environ.get('TORCH_DEVICE', 'mps')
SEQ, dev = 512, torch.device(DEVICE)
log = open(os.path.join(OUT, 'loop_tolerance.txt'), 'w')
def out(s=''):
    print(s, flush=True); log.write(s + '\n'); log.flush()

tok = AutoTokenizer.from_pretrained(HF)
def docs(path, start):
    return [x for x in open(path, encoding='utf-8', errors='replace').read().split('\x1e') if x.strip()][start:]
seqs = []
for dom, path, start, n in [('web', WEB, 1300, 8), ('multilingual', MIXED, 3300, 8), ('code', CODE, 120, 8)]:
    got = 0
    for d in docs(path, start):
        ids = tok(d, add_special_tokens=False)['input_ids'][:SEQ]
        if len(ids) == SEQ: seqs.append((dom, ids)); got += 1
        if got == n: break
out(f'{len(seqs)} sequences of {SEQ} tokens: ' + ', '.join(f'{d} {sum(x == d for x, _ in seqs)}' for d in ['web', 'multilingual', 'code']))

model = AutoModelForCausalLM.from_pretrained(HF, dtype=torch.bfloat16, device_map={'': DEVICE})
model.eval()
layers = list(model.model.layers)
layer_types = list(model.config.layer_types)
N = len(layers)

def set_order(order):
    model.model.layers = torch.nn.ModuleList([layers[i] for i in order])
    model.config.num_hidden_layers = len(order)
    model.config.layer_types = [layer_types[i] for i in order]

def nll_per_seq():
    res = []
    with torch.no_grad():
        for _, ids in seqs:
            x = torch.tensor([ids], device=dev)
            logits = model(input_ids=x, use_cache=False).logits[0, :-1].float()
            res.append(torch.nn.functional.cross_entropy(logits, x[0, 1:]).item())
    return np.array(res)

configs = [('normal', list(range(N)))]
for s in range(0, 17, 2):
    blk = list(range(s, s + 24))
    configs.append((f'x2 [{s}-{s + 23}]', list(range(s + 24)) + blk + list(range(s + 24, N))))
for s in (0, 8, 16):
    blk = list(range(s, s + 24))
    configs.append((f'x3 [{s}-{s + 23}]', list(range(s + 24)) + blk + blk + list(range(s + 24, N))))

rng = np.random.default_rng(0)
results, base = [], None
for name, order in configs:
    t0 = time.time()
    set_order(order)
    v = nll_per_seq()
    if base is None: base = v
    d = v - base
    boot = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(2000)]
    dom = {k: float(np.mean([d[i] for i, (dd, _) in enumerate(seqs) if dd == k])) for k in ['web', 'multilingual', 'code']}
    results.append(dict(config=name, layers=len(order), nll=float(v.mean()), delta=float(d.mean()),
                        ci=[float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))], by_domain=dom))
    out(f'{name:14s} {len(order):2d} layers  NLL {v.mean():.3f}  ΔNLL {d.mean():+.3f} [{np.percentile(boot, 2.5):+.3f}, '
        f'{np.percentile(boot, 97.5):+.3f}]  web {dom["web"]:+.3f} multilingual {dom["multilingual"]:+.3f} code {dom["code"]:+.3f}'
        f'  ({time.time() - t0:.0f}s)')
    json.dump(results, open(os.path.join(OUT, 'loop_tolerance.json'), 'w'), indent=1)
set_order(list(range(N)))
log.close()
