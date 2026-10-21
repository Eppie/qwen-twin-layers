# Loop hypothesis: pre-registered before running `loop_tolerance.py`

Hypothesis (LessWrong comment): Qwen trained a looped transformer (a weight-shared block applied repeatedly),
then fixed the loop count at 2 and untied the weights, giving Qwen3-32B's twin layers 8–31 / 32–55.

Test: run the released Qwen3-14B (the shape and, by the weight correspondence, the lineage of the 32B's parent)
with a 24-layer window [s, s+23] applied twice: layers 0..s+23, s..s+23, s+24..39 (64 layers, like the 32B).
Window starts s = 0, 2, …, 16; s = 8 is the hypothesised loop. Also 3 passes for s = 0, 8, 16. Measure mean
next-token NLL on held-out text (web, multilingual, code) relative to the normal 40-layer model.

Predictions:
- If block 8–31 of the shared ancestor was trained looped and the 14B retains that (its weights are within
  cosine ~0.9 of the split state), looping it twice should be nearly harmless and distinctly better than the
  neighbouring windows: ΔNLL(s=8) < 0.15 nats/token AND < 0.5 × the mean of ΔNLL(s=6) and ΔNLL(s=10).
- If there was no looped training (stacking a normal model), ΔNLL(s=8) should follow the smooth trend of the
  neighbouring windows (no dip at exactly s=8 beyond that criterion).
- A moderate dip at s=8 that misses the criterion is ambiguous: the block may have been chosen for stacking
  because duplicating it is tolerated, which needs no looped training.
Caveat: the released 14B was trained after the split; failure to tolerate looping does not exclude a looped
phase whose compatibility was later lost, it only removes this line of support.
