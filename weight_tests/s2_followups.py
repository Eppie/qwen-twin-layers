#!/usr/bin/env python3
"""Follow-ups to s2 that came out of the per-neuron results (all weights only).

(a) Neuron-index structure in the dense FFNs. Same-index log-norm correlations between NON-twin layers are
    positive in Qwen3-32B (and slightly in Qwen3.8-27B). This part locates the cause: per-layer log norms of
    gate rows, up rows and down columns; an unconstrained single-step search over the neuron index (pooled
    over layers 9-30); the step size per layer; layer x layer same-index correlations before and after
    removing a 1025-neuron moving average; twin per-neuron cosines on each side of the step.
(b) Do the neurons above the step duplicate neurons below it (Net2Net-style widening)? For 2048 neurons on
    each side of the step: the best within-layer cosine to any neuron below the step (gate, up, down).
(c) Attention heads in Qwen3-32B: is there a matching step between heads < 40 and >= 40 (per-head q-row norm)?
(d) Cross-validation misses (s2): when the gate-half match is not the same index, how close was the same
    index, and are j and its match near-duplicates inside each layer?

    python3 -I s2_followups.py
Needs s2_32b_<a>_<b>.npz and s2_27b_24_40.npz from s2_neurons.py runs on 32b and 27b.
Writes to the output directory (see common.py): s2_followups_lognorm_<model>.npz, s2_followups_steps.csv,
s2_followups.log.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np

from common import GGUF, LOGS, Log, RES, is_twin, rowcos, seed_of, write_csv

log = Log(os.path.join(LOGS, 's2_followups.log'))
FAMS = (('ffn_gate', 1), ('ffn_up', 1), ('ffn_down', 0))


def detrend(Z, k=1025):
    pad = np.pad(Z, ((0, 0), (k // 2, k // 2)), mode='reflect')
    cs = np.concatenate([np.zeros((Z.shape[0], 1)), np.cumsum(pad, 1)], 1)
    return Z - (cs[:, k:] - cs[:, :-k]) / k


step_rows = []
for key in ('32b', '27b'):
    g = GGUF(key)
    cfg = g.cfg
    NL = cfg['n_layers']
    LN = {}
    def lognorms(l, fam, ax):
        X = g.read(f'blk.{l}.{fam}.weight')
        return np.log(np.sqrt(np.einsum('ij,ij->' + ('i' if ax == 1 else 'j'), X, X, dtype=np.float64)))
    for fam, ax in FAMS:
        LN[fam] = np.stack([lognorms(l, fam, ax) for l in range(NL)])
    np.savez_compressed(os.path.join(RES, f's2_followups_lognorm_{key}.npz'), **LN)
    log(f'\n=== {cfg["name"]} (a) neuron-index structure of FFN norms')
    N = LN['ffn_up'].shape[1]
    iu = np.triu_indices(NL, 1)
    for fam, _ in FAMS:
        Z = LN[fam] - LN[fam].mean(1, keepdims=True)
        for lab, ZZ in (('raw', Z), ('detrended', detrend(Z))):
            C = np.corrcoef(ZZ)
            tw = np.array([C[a, b] for a, b in zip(*iu) if is_twin(cfg, a, b)])
            nt = np.array([C[a, b] for a, b in zip(*iu) if not is_twin(cfg, a, b)])
            log(f'  {fam:8s} {lab:9s}: same-index log-norm r, twins {tw.mean():+.3f} (min {tw.min():+.3f}); '
                f'non-twins {nt.mean():+.3f} ± {nt.std():.3f} (SD over pairs), |r| > 0.1 in {np.mean(np.abs(nt) > 0.1):.3f}')
        pooled = Z[9:31].mean(0)
        cs = np.cumsum(pooled)
        ks = np.arange(1000, N - 1000)
        left, right = cs[ks - 1] / ks, (cs[-1] - cs[ks - 1]) / (N - ks)
        score = (left - right) ** 2 * ks * (N - ks) / N
        kbest = int(ks[score.argmax()])
        log(f'  {fam:8s} best single step (layers 9-30 pooled, searched over 1000..{N - 1000}): index {kbest}, '
            f'mean centred log norm {left[score.argmax()]:+.3f} below vs {right[score.argmax()]:+.3f} above')
        if key == '32b':
            d = Z[:, 17408:].mean(1) - Z[:, :17408].mean(1)
            for l in range(NL):
                step_rows.append(dict(model=cfg['name'], family=fam, layer=l, step_17408=d[l]))
            log(f'  {fam:8s} per-layer step at 17408 (log norm above minus below): layers 0-31 mean {d[:32].mean():+.3f} '
                f'(range {d[:32].min():+.2f}..{d[:32].max():+.2f}); layers 32-63 mean {d[32:].mean():+.3f} '
                f'(range {d[32:].min():+.2f}..{d[32:].max():+.2f})')
    if key == '32b':
        log('  twin per-neuron cosine (from s2 npz), median below / above 17408, and fraction above the matched-control p99:')
        for a, b in [(8, 32), (16, 40), (24, 48), (31, 55)]:
            z = np.load(os.path.join(RES, f's2_32b_{a}_{b}.npz'))
            parts = []
            for c in ('gate', 'up', 'down'):
                v = z[f'cos_{c}']
                ctl = np.concatenate([z[k] for k in z.files if k.startswith('ctl') and k.endswith('_' + c)])
                p99 = np.quantile(ctl, .99)
                parts.append(f'{c} {np.median(v[:17408]):+.3f}/{np.median(v[17408:]):+.3f} '
                             f'({np.mean(v[:17408] > p99):.3f}/{np.mean(v[17408:] > p99):.3f})')
            log(f'    ({a},{b}) ' + '; '.join(parts))

        # (b) duplicates across the step?
        log('\n(b) within-layer best cosine to a neuron below 17408, for 2048 neurons above vs 2048 below (excluding self)')
        rng = np.random.default_rng(seed_of('dup'))
        qa = np.sort(rng.choice(np.arange(17408, N), 2048, replace=False))
        qb = np.sort(rng.choice(17408, 2048, replace=False))
        for l in (9, 16, 40):
            out = []
            for fam, ax in FAMS:
                X = g.read(f'blk.{l}.{fam}.weight')
                X = np.ascontiguousarray(X.T) if ax == 0 else X
                Xn = X / np.linalg.norm(X, axis=1, keepdims=True)
                base = Xn[:17408]
                Sa = Xn[qa] @ base.T
                Sb = Xn[qb] @ base.T
                Sb[np.arange(len(qb)), qb] = -1
                out.append(f'{fam}: above {np.median(Sa.max(1)):.3f} (99% {np.quantile(Sa.max(1), .99):.3f}), '
                           f'below {np.median(Sb.max(1)):.3f} (99% {np.quantile(Sb.max(1), .99):.3f})')
                del X, Xn, base, Sa, Sb
            log(f'  layer {l}: ' + '; '.join(out))

        # (c) attention heads
        d = []
        for l in range(NL):
            Q = g.read(f'blk.{l}.attn_q.weight')
            qn = np.log(np.sqrt(np.einsum('hij,hij->h', Q.reshape(64, 128, -1), Q.reshape(64, 128, -1), dtype=np.float64)))
            d.append(qn[40:].mean() - qn[:40].mean())
        d = np.array(d)
        rng = np.random.default_rng(1)
        null = []
        for l in (9, 30):
            Q = g.read(f'blk.{l}.attn_q.weight')
            qn = np.log(np.sqrt(np.einsum('hij,hij->h', Q.reshape(64, 128, -1), Q.reshape(64, 128, -1), dtype=np.float64)))
            null += [qn[p[40:]].mean() - qn[p[:40]].mean() for p in (rng.permutation(64) for _ in range(1000))]
        log(f'\n(c) attn_q per-head log-norm, heads >= 40 minus < 40: mean over 64 layers {d.mean():+.3f}, SD {d.std():.3f}; '
            f'random 24/40 split SD {np.std(null):.3f}; layers with |diff| > 0.15: {[(l, round(float(x), 2)) for l, x in enumerate(d) if abs(x) > 0.15]}')

    # (d) cross-validation misses
    pair = (16, 40) if key == '32b' else (24, 40)
    a, b = pair
    z = np.load(os.path.join(RES, f's2_{key}_{a}_{b}.npz'))
    q, m, H = z['Qidx'], z['xval_m'], z['H']
    miss = m != q
    A = [g.read(f'blk.{a}.ffn_gate.weight'), g.read(f'blk.{a}.ffn_up.weight'), np.ascontiguousarray(g.read(f'blk.{a}.ffn_down.weight').T)]
    B = [g.read(f'blk.{b}.ffn_gate.weight'), g.read(f'blk.{b}.ffn_up.weight'), np.ascontiguousarray(g.read(f'blk.{b}.ffn_down.weight').T)]
    Ga, Gb = A[0][q][:, H], B[0][:, H]
    S = (Ga / np.linalg.norm(Ga, axis=1, keepdims=True)) @ (Gb / np.linalg.norm(Gb, axis=1, keepdims=True)).T
    own_rank = 1 + (S > S[np.arange(len(q)), q][:, None]).sum(1)
    del S, Ga, Gb
    log(f'\n(d) {cfg["name"]} ({a},{b}) gate-half matching: misses {miss.mean():.3f} of {len(q)} queries; '
        f'rank of the same index among misses: median {np.median(own_rank[miss]):.0f}, in top 5 {np.mean(own_rank[miss] <= 5):.3f}')
    for c, lab in enumerate(('gate', 'up', 'down')):
        wa = rowcos(A[c][q[miss]], A[c][m[miss]])
        wb = rowcos(B[c][q[miss]], B[c][m[miss]])
        ra = rowcos(A[c][q], A[c][np.random.default_rng(5).permutation(A[c].shape[0])[:len(q)]])
        log(f'  {lab}: within-layer cos(j, match) among misses: layer {a} median {np.median(wa):+.3f}, layer {b} median {np.median(wb):+.3f} '
            f'(random pairs {np.median(ra):+.3f})')
    if key == '32b':
        log(f'  misses among queries < 17408: {miss[q < 17408].mean():.3f}; >= 17408: {miss[q >= 17408].mean():.3f}')
    del A, B

write_csv(os.path.join(RES, 's2_followups_steps.csv'), step_rows)
log('\ndone')
