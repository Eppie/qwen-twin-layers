#!/usr/bin/env python3
"""Raw second moments of Qwen3.8-27B layer triplets (A, B, C) = (i, i+16, i+32), i = 8..23.

For every tensor family present in the triplet and for G = 8 disjoint random groups of storage rows (the same
row assignment for every triplet of a family), saves n, sums, sums of squares and the three cross sums, so that
means, raw inner products, variances and covariances can be recomputed exactly (results/s6_27b_raw.csv).
1-D families (q/k norms, ssm_norm, ssm_a, ssm_dt.bias) are first centred on the channel mean of the same-type
layers outside the triplet's family (their raw values share a large common profile); matrices are used raw.

Per-element covariance C_XY = sum(xy)/n - mean(x) mean(y). Restricted models (independent zero-mean drift):
  SIM   simultaneous triple copy, A = S + e_A, B = S + e_B, C = S + e_C      -> C_AB = C_AC = C_BC
  SUCC_BC successive copy, B and C share an extra update before separating  -> C_BC > C_AB = C_AC
  SUCC_AB the other order (A and B separate last)                          -> C_AB > C_AC = C_BC
  SCALED  simultaneous copy with layer-specific shrinkage s_X (C_XY = s_X s_Y V): fits any positive triplet;
          it is only constrained by the implied shared fraction v_X = C_XY C_XZ / (C_YZ V_X) <= 1.
Equalities are judged with the spread across the 8 row groups (t = mean difference / SE over groups, |t| < 2
counts as equal; > 2 as greater). Row groups are random, so they test replication across held-out rows, not
the models' structural assumptions.

    python3 -I s6_triplets.py
Writes to the output directory (see common.py): s6_27b_raw.csv, s6_27b_triplets.csv, s6_27b.log.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np

from common import GGUF, LOGS, Log, RES, family, layer_type, seed_of, write_csv

g = GGUF('27b')
cfg = g.cfg
NL = cfg['n_layers']
G = 8
log = Log(os.path.join(LOGS, 's6_27b.log'))

MATS = ['attn_q', 'attn_k', 'attn_v', 'attn_output', 'attn_qkv', 'attn_gate', 'ssm_out', 'ssm_alpha', 'ssm_beta',
        'ssm_conv1d', 'ffn_gate', 'ffn_up', 'ffn_down']
VECS = ['attn_q_norm', 'attn_k_norm', 'ssm_norm', 'ssm_a', 'ssm_dt.bias']


def tname(l, f):
    n = f'blk.{l}.{f}.weight'
    return n if g.has(n) else f'blk.{l}.{f}'


raw, res = [], []
for i in range(8, 24):
    trip = (i, i + 16, i + 32)
    for fam in MATS + VECS:
        if not g.has(tname(i, fam)):
            continue
        X = [g.read(tname(l, fam)).astype(np.float32) for l in trip]
        X = [x.reshape(-1, x.shape[-1]) if x.ndim > 1 else x.reshape(-1, 1) for x in X]
        if fam in VECS:
            ref_layers = [l for l in range(NL) if g.has(tname(l, fam)) and l not in family(cfg, i)
                          and layer_type(cfg, l) == layer_type(cfg, i)]
            ref = np.mean([g.read(tname(l, fam)) for l in ref_layers], axis=0).reshape(-1, 1)
            X = [x - ref for x in X]
        nrow = X[0].shape[0]
        grp = np.random.default_rng(seed_of('s6', fam)).integers(0, G, nrow) if nrow >= 2 * G else np.arange(nrow) % G
        stats = np.zeros((G, 10))
        for k in range(G):
            sel = grp == k
            a, b, c = (x[sel].astype(np.float64).ravel() for x in X)
            stats[k] = [a.size, a.sum(), b.sum(), c.sum(), a @ a, b @ b, c @ c, a @ b, a @ c, b @ c]
            raw.append(dict(i=i, family=fam, group=k, n=a.size, sA=a.sum(), sB=b.sum(), sC=c.sum(), sAA=a @ a, sBB=b @ b,
                            sCC=c @ c, sAB=a @ b, sAC=a @ c, sBC=b @ c))

        def moments(s):
            n = s[0]
            mA, mB, mC = s[1] / n, s[2] / n, s[3] / n
            VA, VB, VC = s[4] / n - mA ** 2, s[5] / n - mB ** 2, s[6] / n - mC ** 2
            CAB, CAC, CBC = s[7] / n - mA * mB, s[8] / n - mA * mC, s[9] / n - mB * mC
            return dict(mA=mA, mB=mB, mC=mC, VA=VA, VB=VB, VC=VC, CAB=CAB, CAC=CAC, CBC=CBC,
                        rawAB=s[7] / n, rawAC=s[8] / n, rawBC=s[9] / n)
        full = moments(stats.sum(0))
        per = [moments(s) for s in stats]

        def t(key1, key2):
            d = np.array([p[key1] - p[key2] for p in per])
            se = d.std(ddof=1) / np.sqrt(G)
            return float(d.mean() / se) if se > 0 else np.nan
        t_bc_ab, t_ab_ac, t_bc_ac = t('CBC', 'CAB'), t('CAB', 'CAC'), t('CBC', 'CAC')
        eq = lambda x: abs(x) < 2
        sim = eq(t_bc_ab) and eq(t_ab_ac) and eq(t_bc_ac)
        succ_bc = t_bc_ab > 2 and t_bc_ac > 2 and eq(t_ab_ac)
        succ_ab = t_ab_ac > 2 and -t_bc_ab > 2 and eq(t_bc_ac)
        C = full
        vA = C['CAB'] * C['CAC'] / C['CBC'] / C['VA'] if C['CBC'] > 0 else np.nan
        vB = C['CAB'] * C['CBC'] / C['CAC'] / C['VB'] if C['CAC'] > 0 else np.nan
        vC = C['CAC'] * C['CBC'] / C['CAB'] / C['VC'] if C['CAB'] > 0 else np.nan
        groups_order = ''.join('S' if p['CBC'] > max(p['CAB'], p['CAC']) else ('A' if p['CAB'] > p['CAC'] else 'c') for p in per)
        row = dict(i=i, A=trip[0], B=trip[1], C=trip[2], family=fam, type=layer_type(cfg, i), n=int(stats[:, 0].sum()),
                   **full,
                   cosAB=C['CAB'] / np.sqrt(C['VA'] * C['VB']), cosAC=C['CAC'] / np.sqrt(C['VA'] * C['VC']),
                   cosBC=C['CBC'] / np.sqrt(C['VB'] * C['VC']),
                   ratio_BC_AB=C['CBC'] / C['CAB'], ratio_AC_AB=C['CAC'] / C['CAB'],
                   t_BC_minus_AB=t_bc_ab, t_AB_minus_AC=t_ab_ac, t_BC_minus_AC=t_bc_ac,
                   fits_SIM=int(sim), fits_SUCC_BC=int(succ_bc), fits_SUCC_AB=int(succ_ab),
                   vA=vA, vB=vB, vC=vC, scaled_valid=int(all(0 < v <= 1 for v in (vA, vB, vC))),
                   sB_over_sA=C['CBC'] / C['CAC'], sC_over_sA=C['CBC'] / C['CAB'],
                   group_pattern=groups_order)
        res.append(row)
    log(f'triplet {trip} done')

write_csv(os.path.join(RES, 's6_27b_raw.csv'), raw)
write_csv(os.path.join(RES, 's6_27b_triplets.csv'), res, cols=list(res[0].keys()))

log('\nPer family (over triplets): median per-element covariances (x1e6 for matrices), ratios to C_AB, '
    'and how many triplets fit each restricted model (t over 8 row groups, |t|<2 = equal)')
log(f'{"family":13s} {"n":>3s} {"C_AB":>9s} {"C_AC":>9s} {"C_BC":>9s} {"BC/AB":>6s} {"AC/AB":>6s} {"VA":>9s} {"VB":>9s} {"VC":>9s} '
    f'{"SIM":>4s} {"S_BC":>4s} {"S_AB":>4s} {"none":>4s} {"scaled ok":>9s} {"vA":>5s} {"vB":>5s} {"vC":>5s}')
for fam in MATS + VECS:
    rr = [r for r in res if r['family'] == fam]
    if not rr:
        continue
    sc = 1e6 if fam in MATS else 1.0
    md = lambda k: np.median([r[k] for r in rr])
    none = sum(1 for r in rr if not (r['fits_SIM'] or r['fits_SUCC_BC'] or r['fits_SUCC_AB']))
    log(f'{fam:13s} {len(rr):3d} {md("CAB") * sc:9.3f} {md("CAC") * sc:9.3f} {md("CBC") * sc:9.3f} {md("ratio_BC_AB"):6.2f} '
        f'{md("ratio_AC_AB"):6.2f} {md("VA") * sc:9.2f} {md("VB") * sc:9.2f} {md("VC") * sc:9.2f} '
        f'{sum(r["fits_SIM"] for r in rr):4d} {sum(r["fits_SUCC_BC"] for r in rr):4d} {sum(r["fits_SUCC_AB"] for r in rr):4d} {none:4d} '
        f'{sum(r["scaled_valid"] for r in rr):5d}/{len(rr):<3d} {md("vA"):5.2f} {md("vB"):5.2f} {md("vC"):5.2f}')
tot = len(res)
log(f'\nall (triplet, family) units: {tot}; fit SIM {sum(r["fits_SIM"] for r in res)}, SUCC_BC {sum(r["fits_SUCC_BC"] for r in res)}, '
    f'SUCC_AB {sum(r["fits_SUCC_AB"] for r in res)}, none of the three {sum(1 for r in res if not (r["fits_SIM"] or r["fits_SUCC_BC"] or r["fits_SUCC_AB"]))}; '
    f'SCALED valid (all v in (0,1]) {sum(r["scaled_valid"] for r in res)}')
for key, lab in (('t_AB_minus_AC', 'C_AB - C_AC'), ('t_BC_minus_AB', 'C_BC - C_AB'), ('t_BC_minus_AC', 'C_BC - C_AC')):
    v = np.array([r[key] for r in res])
    log(f'  {lab}: t > 2 in {np.sum(v > 2)}, |t| < 2 in {np.sum(np.abs(v) < 2)}, t < -2 in {np.sum(v < -2)} units')
log('\nmeans (matrices): |mean| / sqrt(V) median ' + f'{np.median([abs(r["mA"]) / np.sqrt(r["VA"]) for r in res if r["family"] in MATS]):.4f}')
log('\ndone')
