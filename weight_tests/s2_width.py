#!/usr/bin/env python3
"""Width structure of the dense FFNs (follow-up to s2_followups.py part a/b).

1. Step search with a null: the pooled (layers 9-30) best single-step score over the neuron index, compared
   with the same search after randomly permuting neuron order (same permutation for all layers; 200 draws).
   Run for Qwen3-32B (observed step 17408) and Qwen3.8-27B (observed step 12288).
2. Partner test for the neurons above the step: for each of 2048 such neurons, its best within-layer match
   among the neurons below the step, separately on gate rows, up rows and down columns. If neurons above the
   step were made by duplicating neurons below it (Net2Net-style widening) and then trained, the best partner
   should be the SAME base neuron for gate, up and down (a connected copy). Control: 2048 neurons below the step,
   matched against the other neurons below it. Also: is the partner the same in the two twin layers?
3. Every layer: cos(neuron k, neuron step + k) for gate rows, up rows and down columns, k < N - step (the copy map
   found in 2), against a shifted control cos(k, step + k'), k' = the next sampled k.
4. Attention heads, every full-attention layer: within each GQA group, pairs of query heads that are mutual best
   matches on both their q-projection block and their output-projection block. A duplication of heads inside
   each group with a fixed index map shows up as the same local pairs in every group and layer.

    python3 -I s2_width.py
Needs s2_followups_lognorm_<model>.npz from s2_followups.py.
Writes to the output directory (see common.py): s2_width.log, s2_width_partners_<model>.csv and
s2_width_copymap_<model>.csv.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np

from common import GGUF, LOGS, Log, RES, rowcos, seed_of, write_csv

log = Log(os.path.join(LOGS, 's2_width.log'))
CFG = {'32b': dict(step=17408, layers=[(9, 33), (16, 40), (24, 48)]),
       '27b': dict(step=12288, layers=[(16, 32), (24, 40)])}


def best_step_score(pooled):
    N = len(pooled)
    cs = np.cumsum(pooled)
    ks = np.arange(1000, N - 1000)
    left, right = cs[ks - 1] / ks, (cs[-1] - cs[ks - 1]) / (N - ks)
    sc = (left - right) ** 2 * ks * (N - ks) / N
    return int(ks[sc.argmax()]), float(sc.max()), float(left[sc.argmax()] - right[sc.argmax()])


for key, c in CFG.items():
    g = GGUF(key)
    LN = np.load(os.path.join(RES, f's2_followups_lognorm_{key}.npz'))
    log(f'=== {g.cfg["name"]}')
    for fam in ('ffn_gate', 'ffn_up', 'ffn_down'):
        Z = LN[fam] - LN[fam].mean(1, keepdims=True)
        k, sc, diff = best_step_score(Z[9:31].mean(0))
        rng = np.random.default_rng(seed_of('stepnull', key, fam))
        null = [best_step_score(Z[9:31][:, p].mean(0))[1] for p in (rng.permutation(Z.shape[1]) for _ in range(200))]
        log(f'  {fam}: best step {k} (above minus below {-diff:+.3f} log), score {sc:.2f}; permuted-order null: '
            f'max over 200 draws {max(null):.2f}, 99th pct {np.quantile(null, .99):.2f} -> score / null max = {sc / max(null):.1f}')
    step = c['step']
    rows = []
    rng = np.random.default_rng(seed_of('partner', key))
    N = g.shape('blk.0.ffn_gate.weight')[0]
    qa = np.sort(rng.choice(np.arange(step, N), 2048, replace=False))
    qb = np.sort(rng.choice(step, 2048, replace=False))
    for la, lb in c['layers']:
        part = {}
        for l in (la, lb):
            for fam in ('ffn_gate', 'ffn_up', 'ffn_down'):
                X = g.read(f'blk.{l}.{fam}.weight')
                X = np.ascontiguousarray(X.T) if fam == 'ffn_down' else X
                Xn = X / np.linalg.norm(X, axis=1, keepdims=True)
                base = Xn[:step]
                for tag, q in (('above', qa), ('below', qb)):
                    S = Xn[q] @ base.T
                    if tag == 'below':
                        S[np.arange(len(q)), q] = -np.inf
                    part[(l, fam, tag)] = (S.argmax(1), S.max(1))
                    del S
                del X, Xn, base
        for tag in ('above', 'below'):
            for l in (la, lb):
                pg, pu, pd = (part[(l, f, tag)][0] for f in ('ffn_gate', 'ffn_up', 'ffn_down'))
                agree3 = np.mean((pg == pu) & (pu == pd))
                agree_gu = np.mean(pg == pu)
                agree_gd = np.mean(pg == pd)
                log(f'  layer {l} neurons {tag} the step: best partner below the step is the same base neuron for '
                    f'gate&up {agree_gu:.3f}, gate&down {agree_gd:.3f}, all three {agree3:.3f} '
                    f'(chance ~{1 / step:.1e}); median best cos gate {np.median(part[(l, "ffn_gate", tag)][1]):.3f}')
                for i in range(len(pg)):
                    rows.append(dict(layer=l, side=tag, neuron=int((qa if tag == 'above' else qb)[i]), partner_gate=int(pg[i]),
                                     partner_up=int(pu[i]), partner_down=int(pd[i]),
                                     cos_gate=float(part[(l, 'ffn_gate', tag)][1][i]), cos_up=float(part[(l, 'ffn_up', tag)][1][i]),
                                     cos_down=float(part[(l, 'ffn_down', tag)][1][i])))
            pa = part[(la, 'ffn_gate', tag)][0]
            pb = part[(lb, 'ffn_gate', tag)][0]
            ua = part[(la, 'ffn_up', tag)][0]
            ub = part[(lb, 'ffn_up', tag)][0]
            log(f'  twins ({la},{lb}) neurons {tag}: same gate partner in both layers {np.mean(pa == pb):.3f}; same up partner {np.mean(ua == ub):.3f}')
    write_csv(os.path.join(RES, f's2_width_partners_{key}.csv'), rows)

    # 3. copy map in every layer
    ncopy = N - step
    kk = np.sort(rng.choice(ncopy, min(4096, ncopy), replace=False))
    crow = []
    for l in range(g.cfg['n_layers']):
        rec = dict(model=g.cfg['name'], layer=l)
        for fam in ('ffn_gate', 'ffn_up', 'ffn_down'):
            X = g.read(f'blk.{l}.{fam}.weight')
            X = np.ascontiguousarray(X.T) if fam == 'ffn_down' else X
            cc = rowcos(X[kk], X[step + kk])
            sh = rowcos(X[kk], X[step + np.roll(kk, 1)])
            rec.update({f'{fam}_median': float(np.median(cc)), f'{fam}_frac_gt_0.1': float(np.mean(cc > 0.1)),
                        f'{fam}_shifted_median': float(np.median(sh))})
            del X
        crow.append(rec)
    write_csv(os.path.join(RES, f's2_width_copymap_{key}.csv'), crow)
    for fam in ('ffn_gate', 'ffn_up', 'ffn_down'):
        med = np.array([r[f'{fam}_median'] for r in crow])
        fr = np.array([r[f'{fam}_frac_gt_0.1'] for r in crow])
        shm = np.array([r[f'{fam}_shifted_median'] for r in crow])
        log(f'  3. {fam}: cos(k, {step}+k) median per layer {med.min():+.3f}..{med.max():+.3f} (mean {med.mean():+.3f}); '
            f'fraction > 0.1 per layer {fr.min():.2f}..{fr.max():.2f}; shifted control medians {shm.min():+.3f}..{shm.max():+.3f}')

    # 4. head pairs inside GQA groups
    nh, nkv, hd = {'32b': (64, 8, 128), '27b': (24, 4, 256)}[key]
    per = nh // nkv
    import collections
    cnt, slots = collections.Counter(), 0
    for l in range(g.cfg['n_layers']):
        if not g.has(f'blk.{l}.attn_q.weight'):
            continue
        Q = g.read(f'blk.{l}.attn_q.weight').reshape(nh, -1)
        O = g.read(f'blk.{l}.attn_output.weight')
        O = np.ascontiguousarray(O.reshape(O.shape[0], nh, hd).transpose(1, 0, 2)).reshape(nh, -1)
        for grp in range(nkv):
            hs = np.arange(grp * per, grp * per + per)
            bs = []
            for X in (Q, O):
                Xn = X[hs] / np.linalg.norm(X[hs], axis=1, keepdims=True)
                S = Xn @ Xn.T
                np.fill_diagonal(S, -1)
                bs.append(S.argmax(1))
            for i in range(per):
                j = int(bs[0][i])
                if bs[0][j] == i and bs[1][i] == j and bs[1][j] == i and i < j:
                    cnt[(i, j)] += 1
            slots += 1
    log(f'  4. {per} query heads per KV group; mutual q-and-o best pairs (local index) over {slots} (layer, group) slots '
        f'(any of {per * (per - 1) // 2} pairs possible): ' + ', '.join(f'{p}: {c}' for p, c in cnt.most_common(6)))
log('done')
