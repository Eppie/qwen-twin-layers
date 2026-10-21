#!/usr/bin/env python3
"""Complete-neuron correspondence and joint FFN fingerprints.

Neuron j of a dense FFN (or of the MoE shared expert) = (gate row j, up row j, down COLUMN j).
For each twin pair (a, b) and its matched non-twin controls (a, b-4), (a, b+4) (same layer type, not twins):

  (i)    per-neuron same-index cosines for gate / up / down; shuffled-index null; flattened full-tensor cosine;
         share of the flattened inner product carried by the top-1% norm neurons; results without them and by
         norm tercile; retrieval rank of the same-index neuron among ALL neurons of the partner layer
         (Q query neurons, both directions, per component and for the sum of the three cosines);
         cross-component validation: best match chosen on a fixed random half H of gate coordinates only
         (argmax cosine over all partner neurons), then frozen and evaluated on held-out gate coordinates,
         the up row and the down column.
  (ii)   per-neuron fingerprints cos(g,u), cos(g,d), cos(u,d), log norms and norm ratios; their twin correlation
         across neurons, against two correlation-preserving controls (R replicates each):
           (a) one random signed permutation of residual coordinates per projection family, shared by both
               twins (preserves each family's twin cosines, norms, within-family geometry and spectrum;
               breaks the geometry between families);
           (b) one random neuron permutation per projection family, shared by both twins (preserves each
               family's per-neuron twin correspondence; decouples gate/up/down of the same neuron).
         Retention coupling: correlation across neurons between the per-neuron twin cosines of two families.

    python3 -I s2_neurons.py {32b|27b|moe} [Q=2048] [R=8]
Writes to the output directory (see common.py): s2_<model>_summary.csv, s2_<model>_fingerprints.csv, s2_<model>_<a>_<b>.npz
(per-neuron arrays) and s2_<model>.log.
"""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np

from common import (GGUF, LOGS, Log, RES, boot_ci, derangement, layer_type, matched_controls, pearson,
                    rowcos, seed_of, write_csv)

key = sys.argv[1]
Q = int(sys.argv[2]) if len(sys.argv) > 2 else 2048
R = int(sys.argv[3]) if len(sys.argv) > 3 else 8
g = GGUF(key)
cfg = g.cfg
log = Log(os.path.join(LOGS, f's2_{key}.log'))

if key == 'moe':
    NAMES = ('ffn_gate_shexp', 'ffn_up_shexp', 'ffn_down_shexp')
    PAIRS = [(8, 20), (12, 24), (16, 28), (19, 31)]
else:
    NAMES = ('ffn_gate', 'ffn_up', 'ffn_down')
    PAIRS = ([(8, 32), (16, 40), (24, 48), (31, 55)] if key == '32b' else
             [(8, 24), (16, 32), (23, 39), (24, 40), (32, 48), (39, 55), (8, 40), (16, 48), (23, 55)])
COMPS = ('gate', 'up', 'down')


def load(l):
    G = g.read(f'blk.{l}.{NAMES[0]}.weight')          # (N, D)
    U = g.read(f'blk.{l}.{NAMES[1]}.weight')          # (N, D)
    Dn = np.ascontiguousarray(g.read(f'blk.{l}.{NAMES[2]}.weight').T)  # storage (D, N) -> neuron-major (N, D)
    return [G, U, Dn]


def norms(X):
    return np.sqrt(np.einsum('ij,ij->i', X, X, dtype=np.float64))


N, D = g.shape(f'blk.8.{NAMES[0]}.weight')
assert g.shape(f'blk.8.{NAMES[2]}.weight') == (D, N)
rng0 = np.random.default_rng(seed_of('s2', key))
Qidx = np.sort(rng0.choice(N, min(Q, N), replace=False))
H = np.sort(rng0.choice(D, D // 2, replace=False))      # gate coordinates used for matching
Hc = np.setdiff1d(np.arange(D), H)                      # held-out gate coordinates
log(f'{cfg["name"]}: tensors {NAMES}, N={N} neurons, D={D}; {len(Qidx)} query neurons; R={R} control replicates')


def ranks(A, B, q):
    """rank (1 = best) of the same-index neuron for queries q, per component and joint (sum of cosines)."""
    out = {}
    Sj = None
    for c, (Xa, Xb) in enumerate(zip(A, B)):
        na = norms(Xa[q])
        nb = norms(Xb)
        S = (Xa[q] @ Xb.T) / (na[:, None] * nb[None, :]).astype(np.float32)
        own = S[np.arange(len(q)), q]
        out[COMPS[c]] = 1 + (S > own[:, None]).sum(1)
        Sj = S if Sj is None else Sj + S
        del S
    own = Sj[np.arange(len(q)), q]
    out['joint'] = 1 + (Sj > own[:, None]).sum(1)
    return out


def xval(A, B, q):
    """Match on gate coordinates H only, freeze, evaluate on held-out gate coords, up, down."""
    Ga = A[0][q][:, H]
    Gb = B[0][:, H]
    S = (Ga @ Gb.T) / (norms(Ga)[:, None] * norms(Gb)[None, :]).astype(np.float32)
    m = S.argmax(1)
    del S
    ev = {}
    for name, Xa, Xb in (('gate_heldout', A[0][:, Hc], B[0][:, Hc]), ('up', A[1], B[1]), ('down', A[2], B[2])):
        ev[name + '_matched'] = rowcos(Xa[q], Xb[m])
        ev[name + '_same'] = rowcos(Xa[q], Xb[q])
    return m, ev


def fingerprints(G, U, Dn):
    nG, nU, nD = np.log(norms(G)), np.log(norms(U)), np.log(norms(Dn))
    return dict(cos_gu=rowcos(G, U), cos_gd=rowcos(G, Dn), cos_ud=rowcos(U, Dn),
                lognorm_g=nG, lognorm_u=nU, lognorm_d=nD, logratio_g_u=nG - nU, logratio_d_u=nD - nU,
                lognorm_mean=(nG + nU + nD) / 3)


FEATS = ['cos_gu', 'cos_gd', 'cos_ud', 'lognorm_g', 'lognorm_u', 'lognorm_d', 'logratio_g_u', 'logratio_d_u',
         'lognorm_mean']

summary_rows, fp_rows = [], []
for (a, b) in PAIRS:
    t0 = time.time()
    A = load(a)
    partners = [b] + [bb for (_, bb) in matched_controls(cfg, a, b)]
    res = {}
    for bb in partners:
        B = load(bb)
        r = dict(partner=bb, rel='twin' if bb == b else 'ctl')
        rng = np.random.default_rng(seed_of('shuffle', key, a, bb))
        perm = derangement(N, rng)
        for c in range(3):
            r['cos_' + COMPS[c]] = rowcos(A[c], B[c])
            r['shuf_' + COMPS[c]] = rowcos(A[c], B[c][perm])
            r['nA_' + COMPS[c]] = norms(A[c])
            r['nB_' + COMPS[c]] = norms(B[c])
            r['dot_' + COMPS[c]] = np.einsum('ij,ij->i', A[c], B[c], dtype=np.float64)
        r['rank_AB'] = ranks(A, B, Qidx)
        r['rank_BA'] = ranks(B, A, Qidx) if bb == b else None
        r['xval_m'], r['xval'] = xval(A, B, Qidx)
        # ---- (ii) fingerprints (actual) ----
        fa, fb = fingerprints(*A), fingerprints(*B)
        r['fp_actual'] = {f: pearson(fa[f], fb[f]) for f in FEATS}
        r['coupling'] = {f'{x}~{y}': pearson(r['cos_' + x], r['cos_' + y])
                         for x, y in (('gate', 'up'), ('gate', 'down'), ('up', 'down'))}
        if bb == b:
            ctl_a = {f: [] for f in FEATS[:3]}
            ctl_b = {f: [] for f in FEATS}
            ctl_b_coup = {k: [] for k in r['coupling']}
            for rep in range(R):
                rr = np.random.default_rng(seed_of('fpctl', key, a, b, rep))
                # (a) shared signed residual permutations for up and down (relative to gate)
                pu, pd = rr.permutation(D), rr.permutation(D)
                su = rr.choice([-1.0, 1.0], D).astype(np.float32)
                sd = rr.choice([-1.0, 1.0], D).astype(np.float32)
                fpa = {}
                for tag, L in (('A', A), ('B', B)):
                    Up = L[1][:, pu] * su
                    Dp = L[2][:, pd] * sd
                    fpa[tag] = dict(cos_gu=rowcos(L[0], Up), cos_gd=rowcos(L[0], Dp), cos_ud=rowcos(Up, Dp))
                    del Up, Dp
                for f in ctl_a:
                    ctl_a[f].append(pearson(fpa['A'][f], fpa['B'][f]))
                # (b) shared neuron permutations for up and down (relative to gate)
                su_n, sd_n = rr.permutation(N), rr.permutation(N)
                fpb = {tag: fingerprints(L[0], L[1][su_n], L[2][sd_n]) for tag, L in (('A', A), ('B', B))}
                for f in FEATS:
                    ctl_b[f].append(pearson(fpb['A'][f], fpb['B'][f]))
                cg, cu, cd = r['cos_gate'], r['cos_up'][su_n], r['cos_down'][sd_n]
                for k, (x, y) in zip(ctl_b_coup, ((cg, cu), (cg, cd), (cu, cd))):
                    ctl_b_coup[k].append(pearson(x, y))
            r['fp_ctl_a'], r['fp_ctl_b'], r['coup_ctl_b'] = ctl_a, ctl_b, ctl_b_coup
        res[bb] = r
        del B
    del A

    # ---------------------------------------------------------------- summaries for this twin pair
    tw = res[b]
    ctls = [res[bb] for bb in partners[1:]]
    log(f'\n=== pair ({a},{b}) [{layer_type(cfg, a)}], controls {[(a, c["partner"]) for c in ctls]}  ({time.time() - t0:.0f}s)')
    np.savez_compressed(os.path.join(RES, f's2_{key}_{a}_{b}.npz'), Qidx=Qidx, H=H,
                        **{f'{k}_{c}': tw[f'{k}_{c}'] for k in ('cos', 'shuf', 'nA', 'nB', 'dot') for c in COMPS},
                        **{f'rankAB_{k}': v for k, v in tw['rank_AB'].items()},
                        **{f'rankBA_{k}': v for k, v in tw['rank_BA'].items()},
                        xval_m=tw['xval_m'], **{f'xval_{k}': v for k, v in tw['xval'].items()},
                        **{f'ctl{bb}_cos_{c}': res[bb]['cos_' + c] for bb in partners[1:] for c in COMPS})
    # joint neuron scale: geometric mean of the six norms (3 components x 2 layers)
    def scale(r):
        return np.exp(np.mean([np.log(r[f'n{x}_{c}']) for x in 'AB' for c in COMPS], axis=0))
    s = scale(tw)
    top1 = s >= np.quantile(s, 0.99)
    terc = np.digitize(s, np.quantile(s, [1 / 3, 2 / 3]))
    for c in COMPS:
        cos = tw['cos_' + c]
        pool = np.concatenate([x['cos_' + c] for x in ctls]) if ctls else tw['shuf_' + c]
        p99, p999 = np.quantile(pool, 0.99), np.quantile(pool, 0.999)
        shuf = tw['shuf_' + c]
        dot, nA, nB = tw['dot_' + c], tw['nA_' + c], tw['nB_' + c]
        flat = dot.sum() / np.sqrt((nA ** 2).sum() * (nB ** 2).sum())
        flat_wo = dot[~top1].sum() / np.sqrt((nA[~top1] ** 2).sum() * (nB[~top1] ** 2).sum())
        share_dot = dot[top1].sum() / dot.sum()
        share_mass = (nA * nB)[top1].sum() / (nA * nB).sum()
        sc = np.sqrt(nA * nB)  # component-specific scale
        top1c = sc >= np.quantile(sc, 0.99)
        flat_wo_c = dot[~top1c].sum() / np.sqrt((nA[~top1c] ** 2).sum() * (nB[~top1c] ** 2).sum())
        # neurons above control p99: their share of the flattened inner product
        above = cos > p99
        share_above = dot[above].sum() / dot.sum()
        med_ci = boot_ci(cos, np.median, B=500, seed=1)
        frac_ci = boot_ci((cos > p99).astype(float), np.mean, B=500, seed=2)
        rk = tw['rank_AB'][c]
        rk2 = tw['rank_BA'][c]
        row = dict(model=cfg['name'], a=a, b=b, comp=c, n_neurons=N,
                   median=np.median(cos), median_lo=med_ci[0], median_hi=med_ci[1],
                   q25=np.quantile(cos, .25), q75=np.quantile(cos, .75), mean=cos.mean(),
                   q05=np.quantile(cos, .05), q95=np.quantile(cos, .95),
                   frac_pos=np.mean(cos > 0), ctl_p99=p99, frac_gt_ctl_p99=np.mean(cos > p99),
                   frac_gt_ctl_p99_lo=frac_ci[0], frac_gt_ctl_p99_hi=frac_ci[1],
                   ctl_p999=p999, frac_gt_ctl_p999=np.mean(cos > p999),
                   shuf_median=np.median(shuf), shuf_p99=np.quantile(shuf, .99), frac_gt_shuf_p99=np.mean(cos > np.quantile(shuf, .99)),
                   ctl_median=np.median(pool), ctl_mean=pool.mean(), ctl_sd=pool.std(),
                   flat_cos=flat, flat_cos_wo_top1pct_joint=flat_wo, flat_cos_wo_top1pct_comp=flat_wo_c,
                   top1pct_share_of_dot=share_dot, top1pct_share_of_normmass=share_mass,
                   median_wo_top1pct=np.median(cos[~top1]),
                   frac_gt_ctl_p99_wo_top1pct=np.mean(cos[~top1] > p99),
                   above_p99_share_of_dot=share_above,
                   **{f'terc{t}_median': np.median(cos[terc == t]) for t in range(3)},
                   **{f'terc{t}_frac_gt_ctl_p99': np.mean(cos[terc == t] > p99) for t in range(3)},
                   rankAB_median=np.median(rk), rankAB_frac1=np.mean(rk == 1), rankAB_frac_top10=np.mean(rk <= 10),
                   rankAB_frac_top1pct=np.mean(rk <= N / 100),
                   rankBA_median=np.median(rk2), rankBA_frac1=np.mean(rk2 == 1),
                   ctl_rank_median=np.median(np.concatenate([x['rank_AB'][c] for x in ctls])) if ctls else np.nan,
                   ctl_rank_frac1=np.mean(np.concatenate([x['rank_AB'][c] for x in ctls]) == 1) if ctls else np.nan)
        summary_rows.append(row)
        log(f'  {c:5s} same-index cos median {row["median"]:+.3f} [{med_ci[0]:+.3f},{med_ci[1]:+.3f}] '
            f'IQR {row["q25"]:+.3f}..{row["q75"]:+.3f} (5%..95% {row["q05"]:+.3f}..{row["q95"]:+.3f}) | '
            f'ctl median {np.median(pool):+.4f}, p99 {p99:+.4f}; shuffled p99 {row["shuf_p99"]:+.4f} | '
            f'frac > ctl p99 {row["frac_gt_ctl_p99"]:.3f} [{frac_ci[0]:.3f},{frac_ci[1]:.3f}], > p99.9 {row["frac_gt_ctl_p999"]:.3f}, > 0 {row["frac_pos"]:.3f}')
        log(f'        flat cos {flat:+.3f}; w/o top-1% joint-norm neurons {flat_wo:+.3f} (w/o top-1% {c}-norm {flat_wo_c:+.3f}); '
            f'top 1% carry {share_dot:.3f} of the inner product vs {share_mass:.3f} of norm mass; '
            f'neurons > ctl p99 carry {share_above:.3f} of it')
        log(f'        tercile (low/mid/high joint norm) median {row["terc0_median"]:+.3f} / {row["terc1_median"]:+.3f} / {row["terc2_median"]:+.3f}; '
            f'frac > ctl p99 {row["terc0_frac_gt_ctl_p99"]:.3f} / {row["terc1_frac_gt_ctl_p99"]:.3f} / {row["terc2_frac_gt_ctl_p99"]:.3f}')
        log(f'        retrieval among all {N}: A->B median rank {row["rankAB_median"]:.0f}, rank 1 {row["rankAB_frac1"]:.3f}, '
            f'top 10 {row["rankAB_frac_top10"]:.3f}, top 1% {row["rankAB_frac_top1pct"]:.3f}; B->A rank 1 {row["rankBA_frac1"]:.3f} | '
            f'controls: median rank {row["ctl_rank_median"]:.0f}, rank 1 {row["ctl_rank_frac1"]:.4f}')
    rkj = tw['rank_AB']['joint']
    log(f'  joint (sum of 3 cosines) retrieval: rank 1 {np.mean(rkj == 1):.3f} (B->A {np.mean(tw["rank_BA"]["joint"] == 1):.3f}), '
        f'median rank {np.median(rkj):.0f}; controls rank 1 '
        + ', '.join(f'{np.mean(x["rank_AB"]["joint"] == 1):.4f}' for x in ctls))
    summary_rows.append(dict(model=cfg['name'], a=a, b=b, comp='joint', n_neurons=N,
                             rankAB_median=np.median(rkj), rankAB_frac1=np.mean(rkj == 1),
                             rankAB_frac_top10=np.mean(rkj <= 10), rankAB_frac_top1pct=np.mean(rkj <= N / 100),
                             rankBA_median=np.median(tw['rank_BA']['joint']), rankBA_frac1=np.mean(tw['rank_BA']['joint'] == 1),
                             ctl_rank_median=np.median(np.concatenate([x['rank_AB']['joint'] for x in ctls])) if ctls else np.nan,
                             ctl_rank_frac1=np.mean(np.concatenate([x['rank_AB']['joint'] for x in ctls]) == 1) if ctls else np.nan))
    # cross-component validation
    for r in [tw] + ctls:
        m, ev = r['xval_m'], r['xval']
        hit = m == Qidx
        rng = np.random.default_rng(seed_of('xvrand', key, a))
        msg = (f'  xval {"twin" if r is tw else "ctl "} ({a},{r["partner"]}): gate-half match = same index for {hit.mean():.3f} of queries; '
               'held-out cos of the frozen match (all / match==same / match!=same) vs same-index: ')
        parts = []
        xrow = dict(model=cfg['name'], a=a, b=b, partner=r['partner'], comp='xval', rel=r['rel'], xval_hit=hit.mean())
        for comp in ('gate_heldout', 'up', 'down'):
            mm, ss = ev[comp + '_matched'], ev[comp + '_same']
            parts.append(f'{comp} {np.mean(mm):+.3f} / {np.mean(mm[hit]) if hit.any() else np.nan:+.3f} / '
                         f'{np.mean(mm[~hit]) if (~hit).any() else np.nan:+.3f} vs {np.mean(ss):+.3f}')
            xrow.update({f'{comp}_matched_mean': np.mean(mm), f'{comp}_matched_hit_mean': np.mean(mm[hit]) if hit.any() else np.nan,
                         f'{comp}_matched_miss_mean': np.mean(mm[~hit]) if (~hit).any() else np.nan,
                         f'{comp}_same_mean': np.mean(ss)})
        summary_rows.append(xrow)
        log(msg + '; '.join(parts))
    # (ii)
    log('  fingerprint twin correlation across neurons: actual | control (a) shared signed residual perms mean ± SD | '
        'control (b) shared neuron perms mean ± SD | non-twin ctl pairs')
    for f in FEATS:
        ca = tw['fp_ctl_a'].get(f)
        cb = tw['fp_ctl_b'][f]
        nt = [x['fp_actual'][f] for x in ctls]
        fp_rows.append(dict(model=cfg['name'], a=a, b=b, feature=f, actual=tw['fp_actual'][f],
                            ctl_a_mean=np.mean(ca) if ca else np.nan, ctl_a_sd=np.std(ca, ddof=1) if ca else np.nan,
                            ctl_b_mean=np.mean(cb), ctl_b_sd=np.std(cb, ddof=1),
                            nontwin_ctl_pairs=';'.join(f'{v:.4f}' for v in nt)))
        log(f'    {f:13s} {tw["fp_actual"][f]:+.3f} | ' +
            (f'{np.mean(ca):+.3f} ± {np.std(ca, ddof=1):.3f}' if ca else '   (n/a: norms unchanged)') +
            f' | {np.mean(cb):+.3f} ± {np.std(cb, ddof=1):.3f} | ' + ', '.join(f'{v:+.3f}' for v in nt))
    for k, v in tw['coupling'].items():
        cb = tw['coup_ctl_b'][k]
        nt = [x['coupling'][k] for x in ctls]
        fp_rows.append(dict(model=cfg['name'], a=a, b=b, feature='retention_coupling_' + k, actual=v,
                            ctl_b_mean=np.mean(cb), ctl_b_sd=np.std(cb, ddof=1),
                            nontwin_ctl_pairs=';'.join(f'{x:.4f}' for x in nt)))
        log(f'    retention coupling corr(twin cos {k}) {v:+.3f} | control (b) {np.mean(cb):+.3f} ± {np.std(cb, ddof=1):.3f} | '
            'non-twin pairs ' + ', '.join(f'{x:+.3f}' for x in nt))
    write_csv(os.path.join(RES, f's2_{key}_summary.csv'), summary_rows,
              cols=sorted({k for r in summary_rows for k in r}, key=lambda k: (k not in ('model', 'a', 'b', 'comp', 'partner', 'rel'), k)))
    write_csv(os.path.join(RES, f's2_{key}_fingerprints.csv'), fp_rows,
              cols=['model', 'a', 'b', 'feature', 'actual', 'ctl_a_mean', 'ctl_a_sd', 'ctl_b_mean', 'ctl_b_sd', 'nontwin_ctl_pairs'])
log('\ndone')
