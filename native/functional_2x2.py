#!/usr/bin/env python3
"""Qwen3-32B: twin FFNs on native and on identical inputs.

    python3 functional_2x2.py HF_DIR OUT_DIR WEB_TXT MIXED_TXT CODE_TXT

HF_DIR: hf download Qwen/Qwen3-32B --revision 9216db5781bf21249d130ec9da846c4624c16137 --local-dir HF_DIR
WEB_TXT, MIXED_TXT, CODE_TXT: text files of documents separated by the ASCII record separator 0x1E (English web,
mixed multilingual/web/code/math/chat, source code); documents are taken from fixed offsets (1200, 3000, 100) on.
Runs the BF16 model on Apple-silicon MPS (device below); needs torch and transformers and ~70 GB of memory.

Phase 1 runs the native BF16 model once over held-out text (English web, multilingual web, code) and keeps
X_l, the residual stream entering each studied layer's FFN (input of post_attention_layernorm). Phase 2
drops everything but the studied FFNs and, for each layer pair (A, B), evaluates
    F_A(X_A), F_B(X_A), F_A(X_B), F_B(X_B)
in two versions: 'full' (each FFN applies its own norm to X) and 'matched' (both FFNs get the identical
normalised input, A's norm of X_A or B's norm of X_B), isolating matrix differences.
Per pair: for 512 pre-selected random neurons, fixed-index activation correlation, a shuffled-index control,
the retrieval rank of the same index among all neurons of the partner layer, top-1% context overlap; for the
whole sublayer, cosine and normalised error of the FFN outputs (the residual updates). Uncertainty: bootstrap
over documents.
"""
import json, os, random, sys, time
import numpy as np, torch
from transformers import AutoModelForCausalLM, AutoTokenizer

HF, OUT, WEB, MIXED, CODE = sys.argv[1:6]
os.makedirs(OUT, exist_ok=True)
TWINS = [(8, 32), (16, 40), (24, 48), (31, 55)]
CONTROLS = [(8, 36), (16, 44), (24, 52), (7, 31)]   # same depth range, not twins; (7, 31) has the twin offset
PAIRS = [('twin', a, b) for a, b in TWINS] + [('control', a, b) for a, b in CONTROLS]
LAYERS = sorted({l for _, a, b in PAIRS for l in (a, b)})
N_NEURONS, PER_DOMAIN, SEQ = 512, 10000, 1024
dev = torch.device('mps')
log = open(os.path.join(OUT, 'functional_2x2.txt'), 'w')
def out(s=''):
    print(s, flush=True); log.write(s + '\n'); log.flush()

# ---------------------------------------------------------------- data (held out: not used in earlier runs)
def docs(path, start):
    d = [x for x in open(path, encoding='utf-8', errors='replace').read().split('\x1e') if x.strip()]
    return d[start:]
tok = AutoTokenizer.from_pretrained(HF)
seqs = []  # (domain, doc id, token ids)
for domain, ds in [('web', docs(WEB, 1200)), ('multilingual', docs(MIXED, 3000)),
                   ('code', docs(CODE, 100))]:
    n = 0
    for k, d in enumerate(ds):
        ids = tok(d, add_special_tokens=False)['input_ids'][:SEQ]
        if len(ids) < 64: continue
        seqs.append((domain, f'{domain}:{k}', ids)); n += len(ids)
        if n >= PER_DOMAIN: break
out(f'{len(seqs)} documents, {sum(len(s) for _, _, s in seqs)} tokens: ' +
    ', '.join(f'{d} {sum(len(s) for dd, _, s in seqs if dd == d)}' for d in ['web', 'multilingual', 'code']))

# ---------------------------------------------------------------- phase 1: capture X_l
t0 = time.time()
model = AutoModelForCausalLM.from_pretrained(HF, dtype=torch.bfloat16, device_map={'': 'mps'})
model.eval()
out(f'loaded in {time.time() - t0:.0f}s')
X = {l: [] for l in LAYERS}
hooks = [model.model.layers[l].post_attention_layernorm.register_forward_pre_hook(
    lambda mod, args, l=l: X[l].append(args[0][0].detach().to('cpu', torch.bfloat16))) for l in LAYERS]
t0 = time.time()
with torch.no_grad():
    for _, _, ids in seqs:
        model.model(input_ids=torch.tensor([ids], device=dev), use_cache=False)  # trunk only, no logits
for h in hooks: h.remove()
out(f'captured {len(LAYERS)} layers in {time.time() - t0:.0f}s')
ffn = {l: (model.model.layers[l].post_attention_layernorm, model.model.layers[l].mlp) for l in LAYERS}
for l in range(len(model.model.layers)):
    if l not in ffn: model.model.layers[l] = torch.nn.Identity()
model.model.embed_tokens = torch.nn.Identity(); model.lm_head = torch.nn.Identity()
torch.mps.empty_cache()
doc_of_tok = np.concatenate([[i] * len(s) for i, (_, _, s) in enumerate(seqs)])
dom_of_doc = [d for d, _, _ in seqs]

# ---------------------------------------------------------------- phase 2
rng = np.random.default_rng(0)
NEUR = np.sort(rng.choice(25600, N_NEURONS, replace=False))      # chosen before any activation is seen
SHUF = rng.permutation(N_NEURONS)
torch.manual_seed(0)

def acts(l, xn, idx=None):
    """SwiGLU activations of layer l on normalised input xn [T, H] (bf16), fp32 out."""
    m = ffn[l][1]
    g = m.gate_proj.weight if idx is None else m.gate_proj.weight[idx]
    u = m.up_proj.weight if idx is None else m.up_proj.weight[idx]
    return (torch.nn.functional.silu(xn @ g.T) * (xn @ u.T)).float()

def ffn_out(l, xn):
    return ffn[l][1](xn).float()

def norm(l, x):
    return ffn[l][0](x)

idx_t, shuf_t = torch.tensor(NEUR, device=dev), torch.tensor(SHUF, device=dev)
results = []
for kind, A, B in PAIRS:
    t0 = time.time()
    conds = {  # name -> (input for A's neurons, input for B's neurons): normalised tensors per document
        'full: X_A': lambda d: (norm(A, X[A][d].to(dev)), norm(B, X[A][d].to(dev))),
        'full: X_B': lambda d: (norm(A, X[B][d].to(dev)), norm(B, X[B][d].to(dev))),
        'matched: n_A(X_A)': lambda d: (norm(A, X[A][d].to(dev)),) * 2,
        'matched: n_B(X_B)': lambda d: (norm(B, X[B][d].to(dev)),) * 2,
        'native': lambda d: (norm(A, X[A][d].to(dev)), norm(B, X[B][d].to(dev))),
    }
    for cname, f in conds.items():
        S = dict(n=0, a=0, aa=0, b=0, bb=0, ab=0)                    # A's 512 neurons vs all B neurons
        perdoc = []                                                    # per-doc sums for the fixed-index pairs
        aA, aB = [], []                                                # per-token activations of the 512 / 512
        cos_sum, err_sum, ntok = np.zeros(len(seqs)), np.zeros(len(seqs)), np.zeros(len(seqs))
        with torch.no_grad():
            for d in range(len(seqs)):
                xa, xb = f(d)
                a = acts(A, xa, idx_t)                                 # [T, 512]
                bfull = acts(B, xb)                                    # [T, 25600]
                b = bfull[:, idx_t]
                S['n'] += a.shape[0]; S['a'] += a.sum(0); S['aa'] += (a * a).sum(0)
                S['b'] += bfull.sum(0); S['bb'] += (bfull * bfull).sum(0); S['ab'] += a.T @ bfull
                perdoc.append(torch.stack([a.sum(0), b.sum(0), (a * a).sum(0), (b * b).sum(0), (a * b).sum(0),
                                           (a * b[:, shuf_t]).sum(0), (b[:, shuf_t] ** 2).sum(0), b[:, shuf_t].sum(0)]).cpu().numpy())
                aA.append(a.cpu()); aB.append(b.cpu())
                fa, fb = ffn_out(A, xa), ffn_out(B, xb)
                cos_sum[d] = torch.nn.functional.cosine_similarity(fa, fb, dim=1).sum().item()
                err_sum[d] = ((fb - fa).norm(dim=1) / fa.norm(dim=1).clamp_min(1e-6)).sum().item()
                ntok[d] = fa.shape[0]
        n = S['n']
        ma, mb = S['a'] / n, S['b'] / n
        cov = S['ab'] / n - ma[:, None] * mb[None, :]
        sa = (S['aa'] / n - ma ** 2).clamp_min(1e-12).sqrt(); sb = (S['bb'] / n - mb ** 2).clamp_min(1e-12).sqrt()
        corr = np.nan_to_num((cov / (sa[:, None] * sb[None, :])).cpu().numpy(), nan=-1.0)  # [512, 25600]
        same = corr[np.arange(N_NEURONS), NEUR]
        shuf = corr[np.arange(N_NEURONS), NEUR[SHUF]]
        rank = (corr > same[:, None]).sum(1)                          # 0 = same index is the best match
        aA, aB = torch.cat(aA).numpy(), torch.cat(aB).numpy()
        k = max(1, int(0.01 * aA.shape[0]))
        def jacc(x, y):
            return len(set(x) & set(y)) / len(set(x) | set(y))
        top_pos = [jacc(np.argsort(-aA[:, i])[:k], np.argsort(-aB[:, i])[:k]) for i in range(N_NEURONS)]
        top_neg = [jacc(np.argsort(aA[:, i])[:k], np.argsort(aB[:, i])[:k]) for i in range(N_NEURONS)]
        top_shuf = [jacc(np.argsort(-aA[:, i])[:k], np.argsort(-aB[:, SHUF[i]])[:k]) for i in range(N_NEURONS)]
        # bootstrap over documents: median fixed-index r and mean FFN-output cosine
        P = np.stack(perdoc)                                           # [docs, 8, 512]
        boot_r, boot_c = [], []
        for _ in range(200):
            s = rng.integers(0, len(seqs), len(seqs))
            q = P[s].sum(0); nn = ntok[s].sum()
            m1, m2 = q[0] / nn, q[1] / nn
            r = (q[4] / nn - m1 * m2) / np.sqrt(np.clip(q[2] / nn - m1 ** 2, 1e-12, None) * np.clip(q[3] / nn - m2 ** 2, 1e-12, None))
            boot_r.append(np.nanmedian(r)); boot_c.append(cos_sum[s].sum() / nn)
        res = dict(kind=kind, A=A, B=B, cond=cname, median_r=float(np.nanmedian(same)),
                   median_r_ci=[float(np.percentile(boot_r, 2.5)), float(np.percentile(boot_r, 97.5))],
                   median_r_shuffled=float(np.nanmedian(shuf)), frac_rank0=float(np.mean(rank == 0)),
                   median_rank=float(np.median(rank)), top1pct_jaccard_pos=float(np.mean(top_pos)),
                   top1pct_jaccard_neg=float(np.mean(top_neg)), top1pct_jaccard_shuffled=float(np.mean(top_shuf)),
                   ffn_out_cos=float(cos_sum.sum() / ntok.sum()),
                   ffn_out_cos_ci=[float(np.percentile(boot_c, 2.5)), float(np.percentile(boot_c, 97.5))],
                   ffn_out_rel_err=float(err_sum.sum() / ntok.sum()),
                   by_domain={dm: float(cos_sum[[i for i in range(len(seqs)) if dom_of_doc[i] == dm]].sum() /
                                        ntok[[i for i in range(len(seqs)) if dom_of_doc[i] == dm]].sum())
                              for dm in ['web', 'multilingual', 'code']})
        results.append(res)
        out(f'{kind:7s} {A:2d},{B:2d} {cname:18s} neuron r {res["median_r"]:+.3f} '
            f'[{res["median_r_ci"][0]:+.3f},{res["median_r_ci"][1]:+.3f}] (shuffled {res["median_r_shuffled"]:+.3f}) '
            f'rank0 {res["frac_rank0"]:.2f} med rank {res["median_rank"]:.0f} | top1% J +{res["top1pct_jaccard_pos"]:.2f} '
            f'-{res["top1pct_jaccard_neg"]:.2f} (shuf {res["top1pct_jaccard_shuffled"]:.2f}) | FFN out cos '
            f'{res["ffn_out_cos"]:+.3f} [{res["ffn_out_cos_ci"][0]:+.3f},{res["ffn_out_cos_ci"][1]:+.3f}] rel err {res["ffn_out_rel_err"]:.2f}')
    out(f'  ({time.time() - t0:.0f}s)')
    json.dump(results, open(os.path.join(OUT, 'functional_2x2.json'), 'w'), indent=1)
log.close()
