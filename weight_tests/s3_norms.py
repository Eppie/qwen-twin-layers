#!/usr/bin/env python3
"""Norm-gain correlations with stricter references and controls.

For every norm family (and the other per-layer 1-D parameters of the hybrid models: ssm_a, ssm_dt.bias,
ffn_gate_inp_shexp) and every pair category:
  twin      all twin relations (27B: A-B, B-C, A-C reported separately)
  ctl4      depth/architecture-matched non-twin controls (a, b+-4) and (a+-4, b), same layer type, not twins
  adjacent  nearest same-type layer
  nontwin   all non-twin pairs (all twin relations excluded; MTP block excluded)
metrics (Pearson r over channels unless noted):
  raw            gains as stored
  c_all          minus the channel mean over ALL layers of the family (the original method)
  c_excl         minus the channel mean over the family's layers EXCLUDING fam(a) u fam(b) (the tested twin
                 family and, for controls, the families of both layers); in the hybrid models only layers of
                 the pair's own type (full / linear attention) enter the reference
  local_excl     each layer minus the mean of its 4 nearest same-type layers by depth, again excluding
                 fam(a) u fam(b) (removes slow depth trends, e.g. of residual-stream gains)
  c_excl_spear   Spearman rank correlation of the c_excl vectors (robust to outlier channels)
q/k norms (applied per head before RoPE; NEOX pairing: channel k and k + R/2 rotate together at frequency
theta_k = base^(-2k/R), R = rotary dims; Qwen3-32B R = 128 = head_dim, qwen35 R = 64 of 256), decomposed
linearly after c_excl centring:
  freq_smooth    degree-4 polynomial in k fitted to the pair means m_k = (g_k + g_{k+R/2})/2
  freq_rough     m_k minus that polynomial (frequency-specific detail that is not smooth in k)
  pair_diff      d_k = (g_k - g_{k+R/2})/2: the split within a rotation pair. A fixed 90-degree rotation of a
                 pair applied to q and k leaves attention scores unchanged and swaps g_k and g_{k+R/2}, so the
                 sign of d_k is an arbitrary label, not a property of the frequency.
  nonrot         (qwen35 only) the 192 channels without RoPE; permuting them consistently in q, k and both
                 gains leaves the layer's function unchanged, so they are exchangeable labels like neurons.
  smooth_ma / rough_ma: same split with a 9-tap moving average instead of the polynomial (robustness).
Also: the identity behind the "-1/(n-1)" baseline, checked numerically per family.

    python3 -I s3_norms.py {32b|27b|moe}
Writes to the output directory (see common.py): s3_<model>_pairs.csv, s3_<model>_summary.csv, s3_<model>.log.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np

from common import GGUF, LOGS, Log, RES, family, is_twin, layer_type, pearson, write_csv

key = sys.argv[1]
g = GGUF(key)
cfg = g.cfg
NL = cfg['n_layers']
log = Log(os.path.join(LOGS, f's3_{key}.log'))


def spearman(x, y):
    rx = np.argsort(np.argsort(x))
    ry = np.argsort(np.argsort(y))
    return pearson(rx, ry)


def group_of(a, b):
    if not is_twin(cfg, a, b):
        return None
    if key != '27b':
        return 'twin'
    if b - a == 32:
        return 'twin_AC'
    return 'twin_AB' if a < 24 else 'twin_BC'


# ------------------------------------------------------------------ q/k decomposition operators
R = cfg['rope_dims']
half = R // 2
kk = np.arange(half)
Vpoly = np.vander(kk / (half - 1), 5, increasing=True)          # degree-4 polynomial basis in k
Ppoly = Vpoly @ np.linalg.pinv(Vpoly)


def ma(x, w=9):
    pad = w // 2
    xp = np.concatenate([x[pad:0:-1], x, x[-2:-pad - 2:-1]])     # reflect padding
    return np.convolve(xp, np.ones(w) / w, mode='valid')


def qk_parts(v):
    m = (v[:half] + v[half:R]) / 2
    d = (v[:half] - v[half:R]) / 2
    sm = Ppoly @ m
    out = dict(freq_smooth=sm, freq_rough=m - sm, pair_diff=d, smooth_ma=ma(m), rough_ma=m - ma(m))
    if len(v) > R:
        out['nonrot'] = v[R:]
    return out


fams = sorted({n.split('.', 2)[2] for n in g.info if n.startswith('blk.') and int(n.split('.')[1]) < NL
               and len(g.info[n][1]) == 1 and ('norm' in n or n.endswith(('ssm_a', 'ssm_dt.bias', 'ffn_gate_inp_shexp.weight')))})
pair_rows, summ_rows = [], []
log(f'{cfg["name"]}: norm families {fams}')
log('stored values (decoder layers): '
    + '; '.join(f'{f} mean {np.mean([g.read(f"blk.{l}.{f}").mean() for l in range(NL) if g.has(f"blk.{l}.{f}")]):+.3f}' for f in fams))
for fam in fams:
    layers = [l for l in range(NL) if g.has(f'blk.{l}.{fam}')]
    G = {l: g.read(f'blk.{l}.{fam}').astype(np.float64) for l in layers}
    n = len(layers)
    Gm = np.stack([G[l] for l in layers])
    mean_all = Gm.mean(0)
    is_qk = fam.startswith(('attn_q_norm', 'attn_k_norm'))
    # ---- the -1/(n-1) identity
    Y = Gm - mean_all
    Y = Y - Y.mean(1, keepdims=True)                              # Pearson also centres over channels
    nrm = np.linalg.norm(Y, axis=1)
    Cr = (Y @ Y.T) / np.outer(nrm, nrm)
    iu = np.triu_indices(n, 1)
    w_mean = (Y @ Y.T)[iu].sum() / np.outer(nrm, nrm)[iu].sum()
    log(f'\n##### {fam}: {n} layers, dim {Gm.shape[1]}')
    log(f'  -1/(n-1) check (all-layer centring): mean r over all {len(iu[0])} pairs {Cr[iu].mean():+.4f}; '
        f'norm-weighted mean {w_mean:+.4f}; -1/(n-1) = {-1 / (n - 1):+.4f}; centred-vector norms CV {nrm.std() / nrm.mean():.2f}')

    pairs = {}
    for i, a in enumerate(layers):
        for b in layers[i + 1:]:
            gr = group_of(a, b)
            if gr:
                pairs[(a, b)] = gr
    tw_pairs = list(pairs)
    for a, b in tw_pairs:
        for x, y in ((a, b - 4), (a, b + 4), (a - 4, b), (a + 4, b)):
            if x in G and y in G and x != y and not is_twin(cfg, x, y):
                pairs.setdefault((min(x, y), max(x, y)), 'ctl4')
    for i in range(n - 1):
        a, b = layers[i], layers[i + 1]
        if not is_twin(cfg, a, b):
            pairs.setdefault((a, b), 'adjacent')
    for i, a in enumerate(layers):
        for b in layers[i + 1:]:
            pairs.setdefault((a, b), 'nontwin_other')

    for (a, b), cat in sorted(pairs.items()):
        excl = family(cfg, a) | family(cfg, b)
        # reference pool: layers of the same type as the pair (hybrid models), minus the tested families
        same = layer_type(cfg, a) == layer_type(cfg, b)
        ref_layers = [l for l in layers if l not in excl and (not same or layer_type(cfg, l) == layer_type(cfg, a))]
        ref = np.mean([G[l] for l in ref_layers], axis=0)

        def local(x):
            near = sorted(ref_layers, key=lambda l: (abs(l - x), l))[:4]
            return G[x] - np.mean([G[l] for l in near], axis=0)
        ca, cb = G[a] - ref, G[b] - ref
        row = dict(model=cfg['name'], family=fam, a=a, b=b, offset=b - a, category=cat,
                   raw=pearson(G[a], G[b]), c_all=pearson(G[a] - mean_all, G[b] - mean_all),
                   c_excl=pearson(ca, cb), local_excl=pearson(local(a), local(b)), c_excl_spear=spearman(ca, cb),
                   n_ref=len(ref_layers))
        if is_qk:
            pa, pb = qk_parts(ca), qk_parts(cb)
            for p in pa:
                row[p] = pearson(pa[p], pb[p])
            pr, qr = qk_parts(G[a]), qk_parts(G[b])
            for p in ('pair_diff', 'nonrot'):
                if p in pr:
                    row[p + '_raw'] = pearson(pr[p], qr[p])
            # variance shares of the centred vector (layer a) in each orthogonal part
            tot = (ca - ca.mean()) @ (ca - ca.mean())
            row['share_freq_smooth'] = 2 * ((pa['freq_smooth'] - ca.mean()) ** 2).sum() / tot
            row['share_freq_rough'] = 2 * (pa['freq_rough'] ** 2).sum() / tot
            row['share_pair_diff'] = 2 * (pa['pair_diff'] ** 2).sum() / tot
            if 'nonrot' in pa:
                row['share_nonrot'] = ((pa['nonrot'] - ca.mean()) ** 2).sum() / tot
        pair_rows.append(row)

    metrics = ['raw', 'c_all', 'c_excl', 'local_excl', 'c_excl_spear']
    if is_qk:
        metrics += ['freq_smooth', 'freq_rough', 'pair_diff', 'smooth_ma', 'rough_ma', 'pair_diff_raw'] + (['nonrot', 'nonrot_raw'] if Gm.shape[1] > R else [])
    rows = [r for r in pair_rows if r['family'] == fam]
    cats = sorted({r['category'] for r in rows}, key=lambda c: (not c.startswith('twin'), c))
    log('  ' + f'{"category":14s} n   ' + '  '.join(f'{m:>15s}' for m in metrics))
    for cat in cats:
        rr = [r for r in rows if r['category'] == cat]
        vals = {m: np.array([r[m] for r in rr]) for m in metrics}
        log(f'  {cat:14s} {len(rr):<3d} ' + '  '.join(f'{np.nanmean(vals[m]):+.3f}±{np.nanstd(vals[m]):.3f}' for m in metrics))
        summ_rows.append(dict(model=cfg['name'], family=fam, category=cat, n_pairs=len(rr),
                              **{f'{m}_mean': np.nanmean(vals[m]) for m in metrics},
                              **{f'{m}_sd': np.nanstd(vals[m]) for m in metrics},
                              **{f'{m}_min': np.nanmin(vals[m]) for m in metrics}))
    if is_qk:
        tw = [r for r in rows if r['category'].startswith('twin')]
        log('  variance share of the c_excl-centred gain (twin layers a): ' + ', '.join(
            f'{p} {np.mean([r[p] for r in tw]):.2f}' for p in ('share_freq_smooth', 'share_freq_rough', 'share_pair_diff', 'share_nonrot') if p in tw[0]))
    # how many twins beat every matched control on c_excl / local_excl
    for m in ('c_excl', 'local_excl'):
        tw = [r for r in rows if r['category'].startswith('twin')]
        ctl = [r for r in rows if r['category'] == 'ctl4']
        if tw and ctl:
            log(f'  {m}: twin pairs above the max over ctl4 pairs: {sum(r[m] > max(c[m] for c in ctl) for r in tw)}/{len(tw)}; '
                f'twin min {min(r[m] for r in tw):+.3f}, ctl4 max {max(c[m] for c in ctl):+.3f}')

write_csv(os.path.join(RES, f's3_{key}_pairs.csv'), pair_rows, cols=list(dict.fromkeys(k for r in pair_rows for k in r)))
write_csv(os.path.join(RES, f's3_{key}_summary.csv'), summ_rows, cols=list(dict.fromkeys(k for r in summ_rows for k in r)))
log('\ndone')
