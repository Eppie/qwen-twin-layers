#!/usr/bin/env python3
"""Least-squares fit of all 15 unrooted 5-leaf trees to d = -log cos among T, Lj, Uj, Lc, Uc (phylogeny.json).

    python3 loop_hypothesis/tree_fit.py results/loop_hypothesis/phylogeny.json > results/loop_hypothesis/tree_fit.txt

For each family and layer: the best topology, its internal edges, and the residual relative to the next best.
Edge lengths are constrained >= 0 (NNLS). Also writes the input with the fitted trees added as *_trees.json.
"""
import itertools, json, sys
import numpy as np
from scipy.optimize import nnls

N = ['T', 'Lj', 'Uj', 'Lc', 'Uc']
pairs = list(itertools.combinations(range(5), 2))

def topologies():
    # unrooted binary trees on 5 leaves: ((a,b),c,(d,e)) with internal edges ab|cde and de|abc
    seen = set()
    for c in range(5):
        rest = [x for x in range(5) if x != c]
        for ab in itertools.combinations(rest, 2):
            de = tuple(x for x in rest if x not in ab)
            key = frozenset([frozenset(ab), frozenset(de)])
            if key not in seen:
                seen.add(key); yield ab, c, de

def design(ab, c, de):
    # edges: 5 pendant edges, internal e1 (ab | rest), internal e2 (de | rest)
    A = np.zeros((10, 7))
    for k, (i, j) in enumerate(pairs):
        A[k, i] = A[k, j] = 1
        side = lambda x: 0 if x in ab else (2 if x in de else 1)
        si, sj = side(i), side(j)
        if si != sj:
            if 0 in (si, sj): A[k, 5] = 1
            if 2 in (si, sj): A[k, 6] = 1
    return A

R = json.load(open(sys.argv[1]))
summary = {}
for r in R:
    C = np.array(r['cos']); d = np.array([-np.log(max(C[i][j], 1e-9)) for i, j in pairs])
    fits = []
    for ab, c, de in topologies():
        A = design(ab, c, de); x, res = nnls(A, d)
        fits.append((res, ab, c, de, x))
    fits.sort(key=lambda f: f[0])
    best, second = fits[0], fits[1]
    res, ab, c, de, x = best
    name = f'(({N[ab[0]]},{N[ab[1]]}),{N[c]},({N[de[0]]},{N[de[1]]}))'
    r['tree'] = dict(topology=name, internal=[float(x[5]), float(x[6])], resid=float(res), next_resid=float(second[0]),
                     next=f'(({N[second[1][0]]},{N[second[1][1]]}),{N[second[2]]},({N[second[3][0]]},{N[second[3][1]]}))',
                     pendant={N[i]: float(x[i]) for i in range(5)})
    summary.setdefault(r['fam'], []).append(r)
for fam, rs in summary.items():
    print(f'== {fam}')
    for r in rs:
        t = r['tree']
        print(f'  p={r["p"]:2d}  best {t["topology"]:26s} internal {t["internal"][0]:.3f} {t["internal"][1]:.3f}  '
              f'resid {t["resid"]:.3f}  next {t["next"]:26s} resid {t["next_resid"]:.3f}  '
              f'pendant T {t["pendant"]["T"]:.2f} Lj {t["pendant"]["Lj"]:.2f} Uj {t["pendant"]["Uj"]:.2f} Lc {t["pendant"]["Lc"]:.2f} Uc {t["pendant"]["Uc"]:.2f}')
json.dump(R, open(sys.argv[1].replace('.json', '_trees.json'), 'w'))
