#!/usr/bin/env python3
"""Router similarity in Qwen3.6-35B-A3B before and after centring rows across experts (weights only).

R (256 x 2048, ffn_gate_inp) gives expert logits R x. R_c = (I - 11^T/E) R subtracts the mean row; that adds
the same scalar to every expert's logit for any input, so softmax probabilities and top-k routing are unchanged.
Measured for every pair of layers:
  flat_raw / flat_c   cosine of the flattened raw / centred routers (flat_raw is the original analysis)
  row_raw / row_c     median over experts of cos(R_a[e], R_b[e]) (raw / centred rows)
  row_c_shuf          same with experts shuffled (derangement), centred
  meanrow_cos         cosine of the two mean rows (the part removed by centring)
  eff_c               flat_c after folding in each layer's post_attention_norm gain (the router reads the
                      normed residual x_hat * gamma, so the effective router is R diag(gamma))
  rank1_c             fraction of experts whose own index is the best match (centred rows) among 256
Per layer: share of the router's squared norm in the mean row (what centring removes).

    python3 -I s5_router.py
Writes to the output directory (see common.py): s5_router_pairs.csv and s5_router.log.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np

from common import GGUF, LOGS, Log, RES, derangement, is_twin, layer_type, rowcos, seed_of, write_csv

g = GGUF('moe')
cfg = g.cfg
NL = cfg['n_layers']
log = Log(os.path.join(LOGS, 's5_router.log'))
R = {l: g.read(f'blk.{l}.ffn_gate_inp.weight').astype(np.float64) for l in range(NL)}
gam = {l: g.read(f'blk.{l}.post_attention_norm.weight').astype(np.float64) for l in range(NL)}
E = R[0].shape[0]
Rc = {l: R[l] - R[l].mean(0, keepdims=True) for l in R}
Ec = {l: Rc[l] * gam[l][None, :] for l in R}
sigma = derangement(E, np.random.default_rng(seed_of('router')))


def fc(X, Y):
    return float((X * Y).sum() / np.sqrt((X * X).sum() * (Y * Y).sum()))


share = {l: E * (R[l].mean(0) ** 2).sum() / (R[l] ** 2).sum() for l in R}
log(f'{cfg["name"]}: router {R[0].shape}; mean-row share of squared norm per layer: '
    f'median {np.median(list(share.values())):.3f}, range {min(share.values()):.3f}..{max(share.values()):.3f}')
log('  twin layers 8..31: ' + ' '.join(f'{l}:{share[l]:.2f}' for l in range(8, 32)))

rows = []
for a in range(NL):
    for b in range(a + 1, NL):
        cat = 'twin' if is_twin(cfg, a, b) else 'other'  # non-twin categories assigned below
        rc = rowcos(Rc[a], Rc[b])
        S = (Rc[a] / np.linalg.norm(Rc[a], axis=1, keepdims=True)) @ (Rc[b] / np.linalg.norm(Rc[b], axis=1, keepdims=True)).T
        rows.append(dict(a=a, b=b, offset=b - a, category=cat, same_type=int(layer_type(cfg, a) == layer_type(cfg, b)),
                         flat_raw=fc(R[a], R[b]), flat_c=fc(Rc[a], Rc[b]), eff_c=fc(Ec[a], Ec[b]),
                         row_raw=float(np.median(rowcos(R[a], R[b]))), row_c=float(np.median(rc)),
                         row_c_mean=float(rc.mean()), row_c_min=float(rc.min()),
                         row_c_shuf=float(np.median(rowcos(Rc[a], Rc[b][sigma]))),
                         meanrow_cos=fc(R[a].mean(0), R[b].mean(0)),
                         rank1_c=float(np.mean(S.argmax(1) == np.arange(E)))))
# ctl4: (a, b-4) and (a, b+4) for each twin pair (a, b), same layer type, not twins
tw = {(r['a'], r['b']) for r in rows if r['category'] == 'twin'}
ctl = {(a, bb) for a, b in tw for bb in (b - 4, b + 4) if 0 <= bb < NL and not is_twin(cfg, a, bb)}
for r in rows:
    if r['category'] != 'twin':
        r['category'] = 'ctl4' if (r['a'], r['b']) in ctl else ('adjacent' if r['offset'] == 1 else 'nontwin_other')
write_csv(os.path.join(RES, 's5_router_pairs.csv'), rows)

mets = ['flat_raw', 'flat_c', 'eff_c', 'row_raw', 'row_c', 'row_c_min', 'row_c_shuf', 'meanrow_cos', 'rank1_c']
log('\nmean ± SD across pairs (SD across layer pairs, not a CI); nontwin_all = every non-twin pair')
log(f'{"category":14s} {"n":>4s} ' + ' '.join(f'{m:>15s}' for m in mets))
for cat in ('twin', 'ctl4', 'adjacent', 'nontwin_other', 'nontwin_all'):
    rr = [r for r in rows if (r['category'] == cat if cat != 'nontwin_all' else r['category'] != 'twin')]
    log(f'{cat:14s} {len(rr):4d} ' + ' '.join(f'{np.mean([r[m] for r in rr]):+.3f}±{np.std([r[m] for r in rr]):.3f}' for m in mets))
log('\ntwin pairs:')
for r in rows:
    if r['category'] == 'twin':
        log(f'  ({r["a"]},{r["b"]}) flat raw {r["flat_raw"]:+.3f} -> centred {r["flat_c"]:+.3f} (gain-folded {r["eff_c"]:+.3f}); '
            f'per-expert centred median {r["row_c"]:+.3f} (min {r["row_c_min"]:+.3f}), shuffled {r["row_c_shuf"]:+.3f}; '
            f'own expert best match {r["rank1_c"]:.3f}; mean-row cos {r["meanrow_cos"]:+.3f}')
nt = [r for r in rows if r['category'] != 'twin']
for m in ('flat_raw', 'flat_c'):
    v = np.array([r[m] for r in nt])
    t = np.array([r[m] for r in rows if r['category'] == 'twin'])
    log(f'{m}: twin min {t.min():+.3f}; non-twin max {v.max():+.3f} at '
        f'{[(r["a"], r["b"]) for r in nt if r[m] == v.max()][0]}; non-twin pairs above twin min: {(v >= t.min()).sum()}')
log('\ndone')
