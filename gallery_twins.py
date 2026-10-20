#!/usr/bin/env python3
"""Neurons in Goodfire's neuron gallery that share an index exactly one twin offset apart.

    python3 gallery_twins.py [GALLERY.json]     # default: download the gallery

Dense (Qwen3-32B, k='d'): same N at layers i and i+24, i = 8..31.
MoE (Qwen3.6-35B-A3B, k='m'): same expert E and neuron N at layers i and i+12, i = 8..19.
'expected' is the number of such pairs if gallery neurons were picked independently of index:
sum over i of count(layer i) * count(layer i+offset) / (neurons per layer). Standard library only.
"""
import json, sys, urllib.request
from collections import Counter

URL = 'https://already-clean.goodfire.com/data/neuron-gallery.json'
MODELS = [  # key, name, offset, first, last, neurons per layer, index fields
    ('d', 'Qwen3-32B', 24, 8, 31, 25600, ('N',)),
    ('m', 'Qwen3.6-35B-A3B', 12, 8, 19, 256 * 512, ('E', 'N')),
]

if len(sys.argv) > 1:
    with open(sys.argv[1]) as f: gallery = json.load(f)
else:
    with urllib.request.urlopen(URL) as r: gallery = json.load(r)

for key, name, offset, lo, hi, per_layer, fields in MODELS:
    entries = [e for e in gallery if e['k'] == key]
    by_pos = {(e['L'],) + tuple(e[f] for f in fields): e for e in entries}
    count = Counter(e['L'] for e in entries)
    expected = sum(count[i] * count[i + offset] for i in range(lo, hi + 1)) / per_layer
    pairs = [(e, by_pos[(e['L'] + offset,) + p[1:]]) for p, e in sorted(by_pos.items())
             if lo <= p[0] <= hi and (p[0] + offset,) + p[1:] in by_pos]
    print(f'{name}: {len(entries)} gallery neurons, {len(pairs)} twin pairs at +{offset} '
          f'(expected by chance {expected:.2f})')
    for a, b in pairs:
        print(f'  {a["id"]:16s} {a["t"]}\n  {b["id"]:16s} {b["t"]}\n')
