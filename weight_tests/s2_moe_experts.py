#!/usr/bin/env python3
"""Per-unit correspondence of the Qwen3.6-35B-A3B routed experts: every expert, matched gate rows / up rows / down columns.

Unit = (expert e, neuron n): gate_exps[e, n, :], up_exps[e, n, :], down_exps[e, :, n] (a column of the
storage rows e*2048 .. e*2048+2047). All 256 x 512 units are evaluated; a stratified sample (16 neurons from
every expert, fixed seed) is reported alongside and gives the same picture.

Four conditions for each unit (layer a unit vs layer b unit):
  SS  same expert, same neuron           (claimed correspondence)
  SN  same expert, shuffled neuron        (neuron labels preserved?)   n -> pi_e(n), a derangement within e
  ES  shuffled expert, same neuron index  (expert labels preserved?)   e -> sigma(e), a derangement of experts
  NN  shuffled expert, shuffled neuron    (broad negative control)
Matched non-twin controls (a, b-4), (a, b+4) give an architecture- and depth-matched SS null.
Also: per-expert coverage (how many experts carry the match), retrieval rank of the same neuron within the
same expert (among 512) and globally (among all 131072 units, 1024 queries), and expert-level retrieval
(flattened per-expert matrices, rank of the same expert among 256).

    python3 -I s2_moe_experts.py
Writes to the output directory (see common.py): s2_moe_experts_summary.csv, s2_moe_experts_perexpert.csv, s2_moe_experts.log.
"""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np

from common import GGUF, LOGS, Log, RES, boot_ci, derangement, matched_controls, seed_of, write_csv

g = GGUF('moe')
cfg = g.cfg
log = Log(os.path.join(LOGS, 's2_moe_experts.log'))
PAIRS = [(8, 20), (12, 24), (16, 28), (19, 31)]
FAM = {'gate': 'ffn_gate_exps', 'up': 'ffn_up_exps', 'down': 'ffn_down_exps'}
E, NN_, Dm = 256, 512, 2048
rng0 = np.random.default_rng(seed_of('moe_experts'))
S16 = np.stack([np.sort(rng0.choice(NN_, 16, replace=False)) for _ in range(E)])  # 16 neurons per expert
sigma = derangement(E, rng0)
pi = np.stack([derangement(NN_, rng0) for _ in range(E)])
gq = np.sort(rng0.choice(E * NN_, 1024, replace=False))  # global retrieval queries


def load(l, fam):
    X = g.read(f'blk.{l}.{FAM[fam]}.weight')  # gate/up: (E, 512, D); down: (E, D, 512)
    if fam == 'down':
        X = np.ascontiguousarray(X.transpose(0, 2, 1))
    return X  # (E, 512, D), unit-major


def ucos(X, Y):
    """per-unit cosine; NaN where either unit is all-zero (the MoE has exactly-zero neurons, see s2_dead_units.py)"""
    num = np.einsum('end,end->en', X, Y, dtype=np.float64)
    den = np.sqrt(np.einsum('end,end->en', X, X, dtype=np.float64) * np.einsum('end,end->en', Y, Y, dtype=np.float64))
    with np.errstate(invalid='ignore', divide='ignore'):
        return np.where(den > 0, num / np.where(den > 0, den, 1), np.nan)


def safe(n):
    """norms for normalisation: zero-norm units get inf (cosine 0 with everything); they are excluded from stats"""
    return np.where(n > 0, n, np.inf).astype(np.float32)


def norms(X):
    return np.sqrt(np.einsum('end,end->en', X, X, dtype=np.float64))


log(f'{cfg["name"]} routed experts: {E} experts x {NN_} neurons, D={Dm}; all units + 16/expert sample')
summary, perexp = [], []
for a, b in PAIRS:
    t0 = time.time()
    partners = [b] + [bb for _, bb in matched_controls(cfg, a, b)]
    out = {bb: {} for bb in partners}
    for fam in FAM:
        A = load(a, fam)
        nA = norms(A)
        for bb in partners:
            B = load(bb, fam)
            r = out[bb]
            r[f'SS_{fam}'] = ucos(A, B)
            r[f'nA_{fam}'], r[f'nB_{fam}'] = nA, norms(B)
            r[f'dot_{fam}'] = np.einsum('end,end->en', A, B, dtype=np.float64)
            if bb == b:
                Bp = np.take_along_axis(B, pi[:, :, None], axis=1)        # same expert, shuffled neuron
                r[f'SN_{fam}'] = ucos(A, Bp)
                r[f'NN_{fam}'] = ucos(A, Bp[sigma])                       # shuffled expert and neuron
                del Bp
                r[f'ES_{fam}'] = ucos(A, B[sigma])                        # shuffled expert, same neuron
                # within-expert neuron retrieval (rank among 512)
                rk = np.empty((E, NN_), dtype=np.int32)
                for e in range(E):
                    Sa = A[e] / safe(nA[e])[:, None]
                    Sb = B[e] / safe(r[f'nB_{fam}'][e])[:, None]
                    S = Sa @ Sb.T
                    own = np.diag(S)
                    rk[e] = 1 + (S > own[:, None]).sum(1)
                rk = rk.astype(float)
                rk[(nA == 0) | (r[f'nB_{fam}'] == 0)] = np.nan
                r[f'rank_within_{fam}'] = rk
                # global retrieval among all units
                Af = A.reshape(E * NN_, Dm)
                Bf = B.reshape(E * NN_, Dm)
                q = Af[gq] / safe(nA.ravel()[gq])[:, None]
                S = (q @ Bf.T) / safe(r[f'nB_{fam}'].ravel())[None, :]
                own = S[np.arange(len(gq)), gq]
                rg = (1 + (S > own[:, None]).sum(1)).astype(float)
                rg[(nA.ravel()[gq] == 0) | (r[f'nB_{fam}'].ravel()[gq] == 0)] = np.nan
                r[f'rank_global_{fam}'] = rg
                del S
                # expert-level retrieval: flattened per-expert matrices
                Ae = A.reshape(E, -1)
                Be = B.reshape(E, -1)
                C = (Ae @ Be.T).astype(np.float64)
                C /= np.sqrt(np.einsum('ij,ij->i', Ae, Ae, dtype=np.float64))[:, None]
                C /= np.sqrt(np.einsum('ij,ij->i', Be, Be, dtype=np.float64))[None, :]
                r[f'expC_{fam}'] = C
            del B
        del A
    tw = out[b]
    frac = lambda v, t: float(np.mean(v[~np.isnan(v)] > t))
    fracrow = lambda v, t: np.array([frac(x, t) for x in v])
    ctls = [out[bb] for bb in partners[1:]]
    log(f'\n=== pair ({a},{b}); matched controls {[(a, x) for x in partners[1:]]}  ({time.time() - t0:.0f}s)')
    for fam in list(FAM) + ['mean3']:
        if fam == 'mean3':
            get = lambda r, c: np.mean([r[f'{c}_{f}'] for f in FAM], axis=0)
        else:
            get = lambda r, c, fam=fam: r[f'{c}_{fam}']
        ss = get(tw, 'SS')
        nul = get(tw, 'NN')
        ctl = np.concatenate([get(x, 'SS').ravel() for x in ctls])
        p99_null, p99_ctl = np.nanquantile(nul, .99), np.nanquantile(ctl, .99)
        row = dict(model=cfg['name'], a=a, b=b, family=fam)
        msg = []
        for c in ('SS', 'SN', 'ES', 'NN'):
            v = get(tw, c)
            v16 = np.take_along_axis(v, S16, axis=1)
            row.update({f'{c}_median': np.nanmedian(v), f'{c}_mean': np.nanmean(v), f'{c}_q25': np.nanquantile(v, .25),
                        f'{c}_q75': np.nanquantile(v, .75), f'{c}_frac_gt_NNp99': frac(v, p99_null),
                        f'{c}_16_median': np.nanmedian(v16), f'{c}_16_frac_gt_NNp99': frac(v16, p99_null)})
            msg.append(f'{c} median {np.nanmedian(v):+.3f} (IQR {np.nanquantile(v, .25):+.3f}..{np.nanquantile(v, .75):+.3f}; '
                       f'16/expert {np.nanmedian(v16):+.3f}) >NNp99 {frac(v, p99_null):.3f}')
        ss16 = np.take_along_axis(ss, S16, axis=1)
        ci = boot_ci(ss16.ravel()[~np.isnan(ss16.ravel())], np.median, B=300, seed=3)
        row.update(dict(ctl_SS_median=np.nanmedian(ctl), ctl_SS_p99=p99_ctl, SS_frac_gt_ctl_p99=frac(ss, p99_ctl),
                        SS_16_median_lo=ci[0], SS_16_median_hi=ci[1]))
        # per-expert coverage: mean SS over the expert's 512 neurons vs the ES/NN per-expert means
        pe = np.nanmean(ss, 1)
        pe_null = np.nanmean(get(tw, 'ES'), 1)
        pe_ctl = np.concatenate([np.nanmean(get(x, 'SS'), 1) for x in ctls])
        thr = max(np.nanmax(pe_null), np.nanmax(pe_ctl))
        frac_e = fracrow(ss, p99_null)
        row.update(dict(perexpert_mean_min=np.nanmin(pe), perexpert_mean_p05=np.nanquantile(pe, .05), perexpert_mean_median=np.nanmedian(pe),
                        perexpert_null_max=thr, n_experts_above_null_max=int((pe > thr).sum()),
                        perexpert_frac_gt_NNp99_min=frac_e.min(), perexpert_frac_gt_NNp99_median=np.nanmedian(frac_e)))
        log(f'  {fam:5s} ' + ' | '.join(msg) + f'  [{int(np.isnan(ss).sum())} units excluded: all-zero in a or b]')
        log(f'        SS 16/expert median 95% CI [{ci[0]:+.3f},{ci[1]:+.3f}]; matched non-twin SS median {np.nanmedian(ctl):+.4f}, '
            f'p99 {p99_ctl:+.4f}; SS > ctl p99 {frac(ss, p99_ctl):.3f}')
        log(f'        per-expert mean SS: min {np.nanmin(pe):+.3f}, 5% {np.nanquantile(pe, .05):+.3f}, median {np.nanmedian(pe):+.3f}, '
            f'max {np.nanmax(pe):+.3f}; null per-expert max {thr:+.4f}; experts above it {int((pe > thr).sum())}/{E}; '
            f'per-expert frac of neurons > NN p99: min {frac_e.min():.3f}, median {np.nanmedian(frac_e):.3f}')
        if fam != 'mean3':
            nA, nB, dot = tw[f'nA_{fam}'].ravel(), tw[f'nB_{fam}'].ravel(), tw[f'dot_{fam}'].ravel()
            s = np.sqrt(nA * nB)
            ok = s > 0
            top = ok & (s >= np.nanquantile(s[ok], .99))
            terc = np.where(ok, np.digitize(s, np.nanquantile(s[ok], [1 / 3, 2 / 3])), -1)
            ssr = ss.ravel()
            flat = dot.sum() / np.sqrt((nA ** 2).sum() * (nB ** 2).sum())
            flat_wo = dot[~top].sum() / np.sqrt((nA[~top] ** 2).sum() * (nB[~top] ** 2).sum())
            row['n_zero_units_a_or_b'] = int((~ok).sum())
            rw = tw[f'rank_within_{fam}']
            rg = tw[f'rank_global_{fam}']
            C = tw[f'expC_{fam}']
            erank = 1 + (C > np.diag(C)[:, None]).sum(1)
            offd = C[~np.eye(E, dtype=bool)]
            row.update(dict(flat_cos=flat, flat_cos_wo_top1pct=flat_wo, top1pct_share_of_dot=dot[top].sum() / dot.sum(),
                            **{f'terc{t}_SS_median': np.nanmedian(ssr[terc == t]) for t in range(3)},
                            **{f'terc{t}_frac_gt_NNp99': frac(ssr[terc == t], p99_null) for t in range(3)},
                            rank_within_frac1=np.nanmean(np.where(np.isnan(rw), np.nan, rw == 1)), rank_within_median=np.nanmedian(rw),
                            rank_global_frac1=np.nanmean(np.where(np.isnan(rg), np.nan, rg == 1)), rank_global_median=np.nanmedian(rg),
                            expert_rank_frac1=np.mean(erank == 1), expert_diag_cos_median=np.nanmedian(np.diag(C)),
                            expert_offdiag_cos_mean=offd.mean(), expert_offdiag_cos_max=offd.max()))
            log(f'        flat cos (full tensor) {flat:+.3f}, w/o top-1% norm units {flat_wo:+.3f}, top 1% share of inner product '
                f'{dot[top].sum() / dot.sum():.3f}; tercile SS median {np.nanmedian(ssr[terc == 0]):+.3f} / {np.nanmedian(ssr[terc == 1]):+.3f} / '
                f'{np.nanmedian(ssr[terc == 2]):+.3f}')
            log(f'        retrieval: same neuron rank 1 within its expert (of 512) {np.nanmean(np.where(np.isnan(rw), np.nan, rw == 1)):.3f} (median rank {np.nanmedian(rw):.0f}); '
                f'globally (of 131072) rank 1 {np.nanmean(np.where(np.isnan(rg), np.nan, rg == 1)):.3f} (median {np.nanmedian(rg):.0f}); '
                f'expert-level (flattened expert, of 256) rank 1 {np.mean(erank == 1):.3f}, same-expert cos median {np.nanmedian(np.diag(C)):+.3f} '
                f'vs other experts mean {offd.mean():+.4f} (max {offd.max():+.3f})')
            for e in range(E):
                perexp.append(dict(a=a, b=b, family=fam, expert=e, SS_mean=np.nanmean(ss[e]), SS_median=np.nanmedian(ss[e]),
                                   frac_gt_NNp99=frac_e[e], expert_flat_cos=C[e, e], expert_rank=erank[e],
                                   within_rank1=np.nanmean(np.where(np.isnan(rw[e]), np.nan, rw[e] == 1))))
        summary.append(row)
    write_csv(os.path.join(RES, 's2_moe_experts_summary.csv'), summary, cols=list(dict.fromkeys(k for r in summary for k in r)))
    write_csv(os.path.join(RES, 's2_moe_experts_perexpert.csv'), perexp)
log('\ndone')
