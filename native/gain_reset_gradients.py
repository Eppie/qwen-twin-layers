#!/usr/bin/env python3
"""q/k-norm gain gradients at a reset point, Qwen3-32B.

    python3 gain_reset_gradients.py HF_DIR OUT_DIR WEB_TXT MIXED_TXT CODE_TXT

HF_DIR: hf download Qwen/Qwen3-32B --revision 9216db5781bf21249d130ec9da846c4624c16137 --local-dir HF_DIR
WEB_TXT, MIXED_TXT, CODE_TXT: text files of documents separated by the ASCII record separator 0x1E, as for
functional_2x2.py; sequences are taken from later offsets (1450, 3600, 150), disjoint from that script's.
Runs forward and backward passes of the BF16 model on Apple-silicon MPS (device below); needs torch and
transformers. 512-token sequences keep it within 128 GB of unified memory without swapping.

For each layer pair (twins and depth-matched non-twin controls) and each arm, the q_norm and k_norm gains of
both layers are set to: 'ones'; 'mean' (a constant equal to the original vector's mean); or 'orig' (left as
trained, for reference). Everything else is frozen. The gradient of next-token loss with respect to those four
128-d vectors is summed over held-out sequences (not used elsewhere). With a plain SGD step from a constant,
the correlation of the two updated gain vectors equals the correlation of their gradients, so a high twin
gradient correlation means a shared gain pattern appears immediately from the current matrices alone.
Also reported: correlation of -gradient with each layer's original (mean-removed) gain profile and with the
partner's, and the same after removing the per-frequency trend shared by all layers (Qwen3 rotates channel c
with c + 64, so the trend is taken as the across-layer mean profile).
"""
import json, os, sys, time
import numpy as np, torch
from transformers import AutoModelForCausalLM, AutoTokenizer

HF, OUT, WEB, MIXED, CODE = sys.argv[1:6]
os.makedirs(OUT, exist_ok=True)
PAIRS = [('twin', 8, 32), ('twin', 16, 40), ('twin', 24, 48), ('twin', 31, 55),
         ('control', 8, 36), ('control', 16, 44), ('control', 24, 52), ('control', 7, 31)]
ARMS = ['ones', 'mean', 'orig']
SEQ = 512   # 1024 tokens made the stored activations push a 128 GB machine into swap
dev = torch.device('mps')
log = open(os.path.join(OUT, 'gain_reset_gradients.txt'), 'w')
def out(s=''):
    print(s, flush=True); log.write(s + '\n'); log.flush()

def docs(path, start):
    return [x for x in open(path, encoding='utf-8', errors='replace').read().split('\x1e') if x.strip()][start:]
tok = AutoTokenizer.from_pretrained(HF)
seqs = []
for path, start, n in [(WEB, 1450, 4), (MIXED, 3600, 2), (CODE, 150, 2)]:
    got = 0
    for d in docs(path, start):
        ids = tok(d, add_special_tokens=False)['input_ids'][:SEQ]
        if len(ids) >= SEQ: seqs.append(ids); got += 1
        if got == n: break
out(f'{len(seqs)} sequences, {sum(map(len, seqs))} tokens')

t0 = time.time()
model = AutoModelForCausalLM.from_pretrained(HF, dtype=torch.bfloat16, device_map={'': 'mps'})
model.eval()
for p in model.parameters(): p.requires_grad_(False)
out(f'loaded in {time.time() - t0:.0f}s')
L = model.model.layers
orig = {(l, w): getattr(L[l].self_attn, w).weight.detach().float().cpu().numpy().copy()
        for l in range(len(L)) for w in ('q_norm', 'k_norm')}
trend = {w: np.mean([orig[(l, w)] for l in range(len(L))], 0) for w in ('q_norm', 'k_norm')}

def corr(x, y):
    return float(np.corrcoef(x, y)[0, 1])

results = []
for kind, A, B in PAIRS:
    for arm in ARMS:
        t0 = time.time()
        params = {}
        for l in (A, B):
            for w in ('q_norm', 'k_norm'):
                p = getattr(L[l].self_attn, w).weight
                if arm == 'ones': p.data.fill_(1.0)
                elif arm == 'mean': p.data.fill_(float(orig[(l, w)].mean()))
                p.requires_grad_(True); params[(l, w)] = p
        grads = {k: np.zeros(128) for k in params}
        loss_sum = 0.0
        for ids in seqs:
            x = torch.tensor([ids], device=dev)
            loss = model(input_ids=x, labels=x, use_cache=False).loss
            loss.backward()
            loss_sum += loss.item()
            for k, p in params.items():
                grads[k] += p.grad.float().cpu().numpy(); p.grad = None
        for (l, w), p in params.items():  # restore
            p.requires_grad_(False); p.data.copy_(torch.tensor(orig[(l, w)], dtype=p.dtype, device=dev))
        r = dict(kind=kind, A=A, B=B, arm=arm, loss=loss_sum / len(seqs))
        for w in ('q_norm', 'k_norm'):
            gA, gB = grads[(A, w)], grads[(B, w)]
            oA, oB = orig[(A, w)] - orig[(A, w)].mean(), orig[(B, w)] - orig[(B, w)].mean()
            dA, dB = orig[(A, w)] - trend[w], orig[(B, w)] - trend[w]
            r[w] = dict(grad_corr=corr(gA, gB),
                        own_A=corr(-gA, oA), own_B=corr(-gB, oB), partner_A=corr(-gA, oB), partner_B=corr(-gB, oA),
                        own_A_detrended=corr(-gA, dA), own_B_detrended=corr(-gB, dB),
                        partner_A_detrended=corr(-gA, dB), partner_B_detrended=corr(-gB, dA),
                        grad_corr_minus_trendfit=corr(gA - np.polyval(np.polyfit(trend[w], gA, 1), trend[w]),
                                                      gB - np.polyval(np.polyfit(trend[w], gB, 1), trend[w])),
                        grads=[gA.tolist(), gB.tolist()])
        results.append(r)
        out(f'{kind:7s} {A:2d},{B:2d} {arm:4s} loss {r["loss"]:.3f} | ' + ' | '.join(
            f'{w}: grad r {r[w]["grad_corr"]:+.2f} (trend-removed {r[w]["grad_corr_minus_trendfit"]:+.2f}); '
            f'-g vs own {r[w]["own_A"]:+.2f}/{r[w]["own_B"]:+.2f}, vs partner {r[w]["partner_A"]:+.2f}/{r[w]["partner_B"]:+.2f}; '
            f'detrended own {r[w]["own_A_detrended"]:+.2f}/{r[w]["own_B_detrended"]:+.2f} partner '
            f'{r[w]["partner_A_detrended"]:+.2f}/{r[w]["partner_B_detrended"]:+.2f}'
            for w in ('q_norm', 'k_norm')) + f'  ({time.time() - t0:.0f}s)')
        json.dump(results, open(os.path.join(OUT, 'gain_reset_gradients.json'), 'w'))
log.close()
