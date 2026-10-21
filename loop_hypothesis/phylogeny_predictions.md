# Where does Qwen3-14B attach in the 32B's family tree? Pre-registered before running `phylogeny.py`

Taxa for an original neuron (or query head) j that was width-copied, at block layer p (8..31):
T = released Qwen3-14B, layer p, unit j; Lj / Uj = 32B unit j at layer p / p+24 (lower / upper depth copy);
Lc / Uc = 32B width copy of j (FFN 17408+j; for heads the duplicate local head) at layer p / p+24.
Distance d = −log cos (additive along a tree under independent drifts). For each quartet, the split with the
smallest four-point sum wins; its internal edge = (second smallest − smallest) / 2.

Hypotheses and predicted splits (internal edges clearly > 0):
- (a) 14B continued from the pre-growth model: T never groups with a single depth copy. Allowed splits:
  {Lj,Uj}|{Lc,Uc} family splits ("width first"), {Lj,Lc}|{Uj,Uc} ("depth first"), or none (simultaneous).
  T sits at the root side of these.
- (b) 14B pruned back out of the grown 32B (most likely from the upper copy, unit j): split {T, Uj} | {Lj, Lc,
  Uc}-side with a clear internal edge (the 32B's training before pruning), present across the block including
  its bottom layers (p = 8–10), not only where the upper copy shares the 14B's context (p ≥ 11).

Known confounds, stated in advance:
- Copies in the same layer (Lj, Lc) receive nearly identical gradients and co-adapt; this produces
  {Lj,Lc}|{Uj,Uc} splits regardless of growth order, so a "depth first" result is weak evidence of order.
- If the width copy was created as j plus noise, c lineages carry extra length; this alone adds pendant length,
  not a split, unless width growth came first (then it sits on the {Lc,Uc} internal edge).
- Convergent drift: the upper copy shares the 14B's context away from the seam. A {T,Uj} grouping seen only at
  p ≥ 11 and absent or reversed at p = 8–10 would point to convergence, not inheritance.
