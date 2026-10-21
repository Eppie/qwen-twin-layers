# Same-index twin layers and model lineages in Qwen checkpoints

Five Qwen checkpoints contain blocks of layers whose weights line up index by index: for a layer i in the
block and its partner i + d, the weights attached to neuron, head or expert j in layer i resemble those of
the same j in layer i + d, and nothing comparable holds between other pairs of layers. We call such layers
same-index twins. The same kind of correspondence links released models to each other: Qwen3-32B lines up
index by index with Qwen3-14B along the map a width + depth expansion of the 14B would produce, and across
the Qwen3 family the released weights fall into five lineages.

| Model | Layers | Twin blocks | Offsets | Layers without a twin |
|---|---|---|---|---|
| Qwen3-32B (dense) | 64 | 8–31 ↔ 32–55 | 24 | 0–7, 56–63 |
| Qwen3-30B-A3B (MoE, 128 experts) | 48 | 6–17 ↔ 18–29 ↔ 30–41 | 12, 24 | 0–5, 42–47 |
| Qwen3-235B-A22B (MoE, 128 experts) | 94 | a 20-layer block four times: 10–29, 30–46 (its first 17 layers), 47–63 (its last 17), 64–83 | 14, 20, 34, 54 | 0–9, 84–93 |
| Qwen3.8-27B (dense FFN, hybrid attention) | 64 + 1 MTP block | 8–23 ↔ 24–39 ↔ 40–55 | 16, 32 | 0–7, 56–63, MTP block 64 |
| Qwen3.6-35B-A3B (MoE, 256 experts, hybrid attention) | 40 | 8–19 ↔ 20–31 | 12 | 0–7, 32–39 |

"Hybrid attention" means full attention in every fourth layer and linear attention (DeltaNet-style;
`attn_qkv`, `attn_gate`, `ssm_*` tensors) in the other three. In the two hybrid models the offsets are
multiples of 4, so twins are always the same kind of layer. Qwen3-0.6B, 1.7B and 4B have no twin layers in
any matrix; Qwen3-8B and Qwen3-14B show none in their q/k-norm gains (their matrices were not scanned).

**Nothing here is described in any Qwen publication.** Everything below is read off the released weights
(plus, for a few tests, model activations and losses). Correspondences between weights are measured against explicit
controls and are established; statements about how and in which order models were derived from one another
are inferences, labelled as such. In particular, which model of a small pair was derived from which rests on
circumstantial evidence (channel sortedness, [below](#direction-channel-sortedness-and-the-pruning-reading)).

Contents: [Qwen3-32B and Qwen3-14B](#qwen3-32b-lines-up-with-the-released-qwen3-14b) ·
[twin layers within each model](#twin-layers-within-each-model) · [follow-up tests](#follow-up-tests) ·
[loop hypothesis](#was-the-32bs-twin-block-trained-as-a-loop) ·
[14B in the 32B's family tree](#qwen3-14b-in-the-32bs-family-tree) ·
[the Qwen3 family map](#the-qwen3-family-map) ·
[direction of the small-model relations](#direction-channel-sortedness-and-the-pruning-reading) ·
[14B/32B activations](#qwen3-14b-and-qwen3-32b-activations-share-one-residual-basis) ·
[what the weights show, and what they do not](#what-the-weights-show-and-what-they-do-not) ·
[Goodfire's neuron gallery](#twin-neurons-in-goodfires-neuron-gallery) · [reproducing](#reproducing) ·
[corrections](#corrections-to-earlier-versions)

**Conventions.** Matrix similarities are signed cosines; gain similarities are Pearson r. Unless marked
otherwise, **± is the standard deviation across layer pairs** (for example across the 24 twin pairs). It
describes how much the pairs differ from one another; it is not a standard error or a confidence interval.
Where a confidence interval is given it is a bootstrap interval and is labelled as such. Ranges such as
"0.10–0.27" run over the layer pairs, tensor families or models named in the sentence.

## Qwen3-32B lines up with the released Qwen3-14B

Qwen3-32B's internal copies (twin layers 8–31 ↔ 32–55, and the width copies described
[below](#width-copies): FFN neuron 17,408 + j copies neuron j, and three of the five query heads in each KV
group are duplicated) imply a parent with 40 layers, hidden size 5,120, FFN width 17,408, and 40 query / 8 KV
heads. That is exactly the published configuration of Qwen3-14B. `qwen3_14b_parent.py` fetches selected
Qwen3-14B tensors from Hugging Face with HTTP range requests and compares them index by index with every
Qwen3-32B layer (read from the Q8_0 GGUF). A 14B stacked as layers 0–31 followed by layers 8–39 predicts that
**14B layer p matches 32B layer p (p ≤ 31) and 32B layer p + 24 (p ≥ 8), and nothing else.**

| 14B tensor vs 32B tensor, same index | At the predicted 32B layers: mean (min), 64 pairs | All other 32B layers: mean ± SD (max), 2,496 pairs |
|---|---|---|
| k_proj | 0.523 (0.156) | 0.0000 ± 0.0007 (0.003) |
| v_proj | 0.665 (0.328) | 0.0000 ± 0.0005 (0.002) |
| gate_proj, neurons 0–511 | 0.469 (0.161) | 0.003 ± 0.003 (0.032) |
| up_proj, neurons 0–511 | 0.444 (0.126) | 0.0000 ± 0.0007 (0.003) |
| 32B up_proj neurons 17,408–17,919 vs 14B neurons 0–511 | 0.325 (0.112) | 0.0000 ± 0.0006 (0.003) |
| q_norm gain (r, uncentred) | 0.970 (0.816) | 0.270 ± 0.178 (0.824) |
| k_norm gain (r, uncentred) | 0.983 (0.911) | 0.305 ± 0.161 (0.775) |

- **Depth.** For all 40 14B layers, the best-matching 32B layers by k_proj are exactly the predicted one
  (p < 8 or p > 31) or two (8 ≤ p ≤ 31). Cosines at the predicted layers range from 0.11 to 0.93; at every
  other layer they are at most 0.003 (gate_proj 0.032).
- **Width.** 32B FFN neuron 17,408 + j matches 14B neuron j (tested for j < 512) at the predicted layers and
  nowhere else. Query heads (q_proj, 128 × 5,120 per head; 14B layers 0, 20, 39 against 32B layers 0, 20, 44,
  63): every 14B head's best match lies in the same KV group, and the predicted map — 14B head i of a group ↔
  32B local heads 3 + i and, for i < 3, also local head i — gives the best match for 40 of 40 heads in three
  of the four comparisons (where the duplicated heads' two copies are also the top two in every case) and 36
  of 40 in the fourth, the lower copy of layer 20, where all cosines are low (median best 0.13).
- **Which copy is closer.** For p = 8–31 the upper copy (32B layer p + 24) is on average closer to the 14B
  than the lower copy (32B layer p): k_proj 0.70 vs 0.29, v_proj 0.81 vs 0.49, gate_proj and up_proj 0.55
  vs 0.30. The upper copy is closer in 21–22 of the 24 layers; the exceptions are at the bottom of the
  block, p = 8–9 (and p = 10 for gate/up). Similarity is lowest next to the seam between the two copies
  (32B layers 31 and 32): the lower copy declines from 0.40–0.58 at p = 8 to 0.13–0.33 at p = 31, and the
  upper copy is lowest just above the seam (0.22–0.40 at p = 8) and above 0.55 from p = 16 on. This fits a
  stacking in which each copy kept the parent's neighbours on one side: the lower copy still reads the
  output of parent layers 0–7, and the upper copy still feeds parent layers 32–39 (`results/qwen3_14b/`
  has the per-layer values).
- **Quantization does not matter here.** The same comparison run earlier on the native BF16 Qwen3-32B
  weights (`results/native/qwen3_14b_parent_bf16.*`) gives the same numbers: the largest difference over all
  40 × 64 layer pairs and all seven quantities is 1.8 × 10⁻⁵, and the head-map output is identical.

**What this does not show.** It does not identify the exact checkpoint that was expanded. The copy
structure lives inside Qwen3-32B, which requires a 14B-shaped ancestor; Qwen3-32B was trained after the
expansion (its two copies of each layer differ), and the 14B may have been trained further as well, so the
endpoints cannot say whether the ancestor was the released Qwen3-14B, an earlier checkpoint of it, or a
sibling trained from the same starting point. A tree analysis of the 14B and the 32B's copies
([below](#qwen3-14b-in-the-32bs-family-tree)) favours the 14B being the pre-growth model trained on, over it
being pruned back out of the grown 32B. Nor do the weights say whether the width or the depth expansion came
first, or whether both happened at once.

## Twin layers within each model

### Method

`twin_layers.py` reads a GGUF file and, for every per-layer tensor type, compares each pair of layers.

- **Matrices.** Signed cosine similarity between the flattened tensors of two layers. The script uses a
  fixed random sample of storage rows (the same rows in every layer: 64 rows if a row has more than 6000
  entries, otherwise 256, or all rows if there are fewer) and dequantises them from Q8_0. A storage row is a
  neuron for `ffn_gate`/`ffn_up` and a query/key/value channel for `attn_q`/`attn_k`/`attn_v`, but a
  residual-stream dimension for `ffn_down` and `attn_output`; there the sampled row runs across all neurons
  or heads, so a match still requires the neuron or head indices to line up. For 3-D expert tensors a row
  is one (expert, neuron), so a match needs both indices to agree. The router (`ffn_gate_inp`) is treated
  as a matrix of per-expert rows.
- **Norm gains.** Pearson correlation of the gain vectors, raw and "centred" (after subtracting each
  channel's mean across layers, which removes the shape shared by all layers).
- **Fixed-offset mode** (`MODEL OFFSET FIRST LAST`) compares the twin pairs (i, i + OFFSET) for
  FIRST ≤ i ≤ LAST with all other pairs of layers that contain the tensor type. It also prints off-by-one
  pairs (i, i + OFFSET ± 1) and adjacent layers.
- **Scan mode** (`MODEL scan`) assumes no offset: for each tensor type it prints the mean similarity at
  every layer offset (the four highest are shown) and, for the mean over the tensor types present in every
  layer, each layer's most similar partner.

The results below come from the Q8_0 GGUFs listed under [Reproducing](#reproducing). Numbers are copied
from the files in `results/`. "All matrices" is the mean over the tensor types present in every layer.
Full-attention tensors exist only in every fourth layer, so in the hybrid models they have fewer twin
pairs than the other tensors.

### Qwen3-32B: layers 8–31 vs 32–55 (offset 24, 24 pairs)

| Tensor | Twin pairs | All other pairs |
|---|---|---|
| attn_q | 0.135 ± 0.035 | 0.000 ± 0.001 |
| attn_k | 0.254 ± 0.049 | 0.000 ± 0.001 |
| attn_v | 0.447 ± 0.066 | 0.000 ± 0.001 |
| attn_output | 0.248 ± 0.068 | −0.000 ± 0.001 |
| ffn_gate | 0.224 ± 0.029 | 0.004 ± 0.003 |
| ffn_up | 0.206 ± 0.034 | 0.000 ± 0.001 |
| ffn_down | 0.229 ± 0.049 | 0.000 ± 0.001 |
| all matrices | 0.249 ± 0.041 | 0.001 ± 0.001 |
| attn_q_norm (centred r) | 0.877 ± 0.095 | −0.025 ± 0.140 |
| attn_k_norm (centred r) | 0.928 ± 0.050 | −0.022 ± 0.148 |

Uncentred, the twin q/k-norm gains correlate at 0.935 / 0.961 and other pairs at about 0.3. The scan finds
+24 as the top offset for every matrix type, and every layer from 8 to 55 has its twin as its most similar
layer (pooled cosine 0.15–0.31); for layers 0–7 and 56–63 the most similar layer is at 0.00–0.01. The
residual-stream gains (`attn_norm`, `ffn_norm`) behave differently: they follow depth more than twins (see
[norm gains](#norm-gains)).

### Qwen3.8-27B: three blocks 8–23, 24–39, 40–55

Twin pairs, mean ± SD across pairs (full attention: 4 pairs; linear attention: 12 pairs; FFN: 16 pairs).

| Tensor | 8–23 ↔ 24–39 (+16) | 24–39 ↔ 40–55 (+16) | 8–23 ↔ 40–55 (+32) |
|---|---|---|---|
| attn_q | 0.121 ± 0.007 | 0.169 ± 0.050 | 0.065 ± 0.014 |
| attn_k | 0.247 ± 0.026 | 0.368 ± 0.077 | 0.189 ± 0.020 |
| attn_v | 0.324 ± 0.036 | 0.321 ± 0.121 | 0.219 ± 0.069 |
| attn_output | 0.169 ± 0.012 | 0.155 ± 0.075 | 0.081 ± 0.029 |
| attn_qkv (linear) | 0.135 ± 0.018 | 0.166 ± 0.039 | 0.086 ± 0.013 |
| attn_gate (linear) | 0.127 ± 0.011 | 0.137 ± 0.029 | 0.076 ± 0.016 |
| ssm_out (linear) | 0.173 ± 0.013 | 0.172 ± 0.044 | 0.108 ± 0.025 |
| ffn_gate | 0.138 ± 0.013 | 0.170 ± 0.020 | 0.094 ± 0.008 |
| ffn_up | 0.124 ± 0.012 | 0.137 ± 0.034 | 0.073 ± 0.013 |
| ffn_down | 0.149 ± 0.008 | 0.149 ± 0.041 | 0.087 ± 0.024 |
| all matrices (FFN) | 0.137 ± 0.010 | 0.152 ± 0.031 | 0.085 ± 0.013 |
| attn_q_norm (centred r) | 0.619 ± 0.082 | 0.840 ± 0.060 | 0.418 ± 0.192 |
| attn_k_norm (centred r) | 0.649 ± 0.052 | 0.872 ± 0.037 | 0.540 ± 0.050 |
| ssm_norm (centred r) | 0.647 ± 0.120 | 0.744 ± 0.181 | 0.502 ± 0.152 |

The "all other pairs" baseline of each fixed-offset run of `twin_layers.py` still contains the other two
twin relations, which widens it (for the matrices in the table, means 0.001–0.020 and SDs up to 0.080; see
`results/`). With all three twin relations excluded (`weight_tests/s0_fresh_rows.py`), the non-twin
baseline is 0.000–0.003 with SD ≤ 0.003 for these matrices, and in the scan no offset other than +16 and +32
has a mean above 0.007 for any matrix in the table. By the pooled FFN cosine and the norm gains, the closest
pair of blocks is 24–39 ↔ 40–55 and the most distant is 8–23 ↔ 40–55; what that implies about the history
is discussed under [triple covariances](#qwen38-27b-triple-covariances).

### Qwen3.6-35B-A3B: layers 8–19 vs 20–31 (offset 12; full attention: 3 pairs; linear attention: 9 pairs; MoE: 12 pairs)

| Tensor | Twin pairs | All other pairs |
|---|---|---|
| attn_q | 0.352 ± 0.039 | 0.017 ± 0.009 |
| attn_k | 0.524 ± 0.044 | 0.000 ± 0.002 |
| attn_v | 0.494 ± 0.089 | 0.000 ± 0.002 |
| attn_output | 0.327 ± 0.071 | −0.000 ± 0.001 |
| attn_qkv (linear) | 0.317 ± 0.027 | 0.000 ± 0.002 |
| attn_gate (linear) | 0.344 ± 0.020 | 0.004 ± 0.004 |
| ssm_out (linear) | 0.359 ± 0.039 | 0.000 ± 0.001 |
| ffn_gate_shexp (shared expert) | 0.356 ± 0.033 | 0.002 ± 0.005 |
| ffn_up_shexp | 0.323 ± 0.037 | 0.000 ± 0.002 |
| ffn_down_shexp | 0.324 ± 0.044 | −0.000 ± 0.003 |
| ffn_gate_exps (routed experts) | 0.149 ± 0.013 | 0.002 ± 0.002 |
| ffn_up_exps | 0.102 ± 0.010 | −0.000 ± 0.001 |
| ffn_down_exps | 0.131 ± 0.026 | −0.000 ± 0.003 |
| ffn_gate_inp (router), expert-centred (`weight_tests/s5_router.py`) | 0.742 ± 0.031 | 0.000 ± 0.003 |
| ffn_gate_inp (router), raw | 0.791 ± 0.028 | 0.286 ± 0.125 |
| all matrices (MoE + raw router) | 0.311 ± 0.020 | 0.042 ± 0.018 |
| attn_q_norm (centred r) | 0.807 ± 0.060 | −0.168 ± 0.233 |
| attn_k_norm (centred r) | 0.823 ± 0.034 | −0.171 ± 0.236 |
| ssm_norm (centred r) | 0.809 ± 0.105 | −0.051 ± 0.090 |

A routed-expert match requires the expert index and the neuron index to agree, so the experts kept their
positions. The router should be read centred: subtracting the mean router row across experts adds the same
number to every expert's logit for any input, so it changes neither the softmax nor the top-k choice, and
that mean row holds 36–66% of each layer's router norm². Raw router rows correlate at 0.29 even between
unrelated layers because of it; centred, twins correlate at 0.742 and non-twins at 0.000 ± 0.003 (max
0.017), and for 96–99.6% of experts the most similar centred row in the twin layer is the same expert
(`weight_tests/s5_router.py`). The raw router also lifts the non-twin mean of the "all matrices" row;
pooled over `ffn_gate_exps`, `ffn_up_exps`, `ffn_down_exps` and `ffn_down_shexp` alone, twins are at 0.177
and non-twins at 0.0005 (`results/weight_tests/s0_moe.log`).

The centred q/k-norm baseline is negative here because centring over only 10 full-attention layers makes
the centred vectors sum to zero across those 10 layers: when the three twin pairs are strongly positive,
the remaining pairs are pushed negative. The [norm-gain tests](#norm-gains) use a reference that excludes
the tested pair's layers and compare with matched controls instead.

The small linear-attention tensors (`ssm_conv1d`, `ssm_alpha`, `ssm_beta`) show the same pattern in both
hybrid models; see `results/`.

## Follow-up tests

A set of follow-up tests was run to check the alignment from independent angles: native weights instead of
the GGUF, fresh row samples, every neuron and expert instead of a flattened sample, functional behaviour,
and stricter controls for the norm gains. Matched controls are the pairs (a, b − 4) and (a, b + 4) for a
twin pair (a, b): same layer type, similar depth, not twins. Non-twin baselines exclude every known twin
relation, and the Qwen3.8-27B MTP block is excluded throughout. Scripts are in `weight_tests/` (GGUF,
weights only) and `native/` (native BF16 checkpoint); outputs are in `results/weight_tests/` and
`results/native/`.

### Native-weight replication and the quantization bound

`native/native_replication.py` reads the native BF16 safetensors of Qwen3-32B (revision `9216db57…`) with
its own parser (no GGUF code) and computes full-tensor cosines (all rows) in float64 for seven twin pairs and
eight controls: the twin offset outside the block, (7, 31) and (32, 56); off by one, (8, 31), (8, 33),
(16, 39) and (16, 41); and other offsets, (8, 36) and (20, 40).

| Pairs | Mean over the 7 matrix families | Range over families |
|---|---|---|
| twins (8,32) (12,36) (16,40) (20,44) (24,48) (28,52) (31,55) | 0.21, 0.27, 0.29, 0.24, 0.28, 0.22, 0.14 | 0.07–0.55; all 49 values positive |
| controls (7,31) (32,56) (8,31) (8,33) (16,39) (16,41) (8,36) (20,40) | 0.0003–0.0009 | ffn_gate 0.0025–0.0070; every other family below 0.0011 in absolute value |

The GGUF has the same layout as the safetensors (no transpose and no Q/K row permutation): the dequantised
Q8_0 tensors have cosine ≥ 0.99998 with the native ones in all 19 layers checked, and the norm gains are
bit-identical. For unit-normalised native (u) and dequantised (ũ) tensors, |⟨u_a, u_b⟩ − ⟨ũ_a, ũ_b⟩| ≤
‖u_a − ũ_a‖ + ‖u_b − ũ_b‖, which is at most 0.0114 for every pair measured, against twin values of
0.07–0.55. Every layer has 11 parameter tensors (seven matrices, four gains, no biases), and all were
checked.

### Fresh row samples

`weight_tests/s0_fresh_rows.py` repeats the `twin_layers.py` measurement on up to four further row samples
per tensor type, disjoint from the original sample and from each other. Per matrix, the largest difference
between a fresh sample's twin mean and the original one is 0.001–0.010 in Qwen3-32B, 0.001–0.014 in
Qwen3.6-35B-A3B and 0.002–0.019 in Qwen3.8-27B (0.035 for the tiny `ssm_conv1d`, 4 values per row); the
largest shifts are in tensors sampled with only 64 long rows (`attn_output`, `ssm_out`; for example
Qwen3.8-27B `ssm_out` 8–23 ↔ 24–39 is 0.173 on the original rows and 0.156–0.162 on fresh rows). Pooled:
Qwen3-32B 0.249–0.251; Qwen3.8-27B FFN 8–23 ↔ 24–39 0.132–0.136, 24–39 ↔ 40–55 0.148–0.151, 8–23 ↔ 40–55
0.081–0.083; Qwen3.6-35B-A3B 0.177–0.180 (`ffn_gate_exps`, `ffn_up_exps`, `ffn_down_exps`,
`ffn_down_shexp`). Every twinned layer's most similar layer is its twin in every sample.

Measured in standard deviations of the non-twin pairs (all twin relations excluded), the twin means of the
attention, linear-attention projection, FFN and expert matrices lie 21–450 SDs above the non-twin means;
the small `ssm_alpha`, `ssm_beta` and `ssm_conv1d` tensors 3–16 SDs; the raw router 4 SDs. These SDs
describe the spread of the non-twin pairs, not sampling error.

### Per-neuron and per-expert correspondence

The flattened cosine could in principle be carried by a few large neurons. `weight_tests/s2_neurons.py`
treats a dense FFN neuron j as its gate row j, up row j and down **column** j, and compares every neuron
with the same-index neuron of the twin layer.

| | Qwen3-32B | Qwen3.8-27B 8–23 ↔ 24–39, 24–39 ↔ 40–55 | Qwen3.8-27B 8–23 ↔ 40–55 | Qwen3.6-35B-A3B shared expert |
|---|---|---|---|---|
| median same-index cosine (gate, up, down; per pair) | 0.10–0.27 | 0.06–0.18 | 0.03–0.11 | 0.23–0.42 |
| neurons above the 99th percentile of the matched controls | 90–99% | 72–98% | 46–88% | 83–97% |
| same index is the best match among all partner neurons (sum of the three cosines) | 63–73% | 58–74% | 48–67% | 88–93% |

In the matched controls the same index is the best match for 0–0.2% of neurons, and the median same-index
cosine is between −0.003 and 0.006.

- Not concentrated in large neurons: the top 1% of neurons by norm carry 1–3% of the flattened inner
  product, and removing them changes the flattened cosine by at most 0.004.
- Cross-validation: a neuron's best match among all partner neurons, chosen on a fixed random half of the
  gate coordinates only and then frozen, is the same index for 49–69% of neurons (Qwen3-32B), 30–70%
  (Qwen3.8-27B) and 64–85% (shared expert); in the matched controls it never is, and the frozen match gives
  held-out up/down cosines of 0.000 ± 0.004. When the frozen match is another neuron, it is mostly a
  near-duplicate of the same-index neuron within the layer, and the same index still has median rank 2–3
  (Qwen3-32B (16, 40), Qwen3.8-27B (24, 40); in Qwen3-32B these are mostly width copies, see below).
- Joint structure: each neuron's internal angles cos(g, u), cos(g, d), cos(u, d) correlate between twins
  at 0.39–0.92, against 0.00–0.20 for a control that applies one shared signed permutation of residual
  coordinates to each projection family (keeping each family's twin cosines, norms and spectrum) and
  0.02–0.48 for a control with one shared neuron permutation per family. How strongly a neuron is retained
  is shared across its gate, up and down vectors (r = 0.44–0.96; control 0.00 ± 0.03).
- Spectrum (`weight_tests/s4_spectral.py`): the overlap is spread through the bulk of the singular
  spectrum. Removing each matrix's top 512 singular directions keeps 60–70% of the twin cosine in the dense
  models (Qwen3-32B (16, 40) gate 0.242 → 0.142); the removed top-64 parts are more similar than the rest
  (0.43 for that gate matrix).

The correspondence is broad but uneven. It is weakest for low-norm neurons, at the block ends (Qwen3-32B
(31, 55), Qwen3.8-27B (39, 55)), for the Qwen3.8-27B 8–23 ↔ 40–55 relation, and for the Qwen3-32B neurons at
index 17,408 and above. Those are width copies (next section); in Qwen3-32B (16, 40), 80% of them miss in
the cross-validation, against 9% of the other neurons.

**Routed experts** (`weight_tests/s2_moe_experts.py`, all 256 experts × 512 neurons, gate rows, up rows and
down columns). Same expert, same neuron: median cosine 0.08–0.16 per pair and projection; 65–84% of units
above the 99th percentile of a shuffled-expert-and-neuron null (82–87% for the mean of the three
projections). Same expert with shuffled neuron, shuffled expert with the same neuron index, and both
shuffled: medians ≤ 0.009. The same neuron is the best match within its expert for 44–71% of units and among
all 131,072 units for 32–63%; at the whole-expert level the same expert is the best match for 99.6–100% of
experts. There is a low tail: per pair and projection, 8–29% of experts have fewer than half of their
neurons above the null's 99th percentile. These weak experts differ between pairs, keep matched router rows,
and have more uniform neuron norms.

### Width copies

Within every Qwen3-32B layer, FFN neuron 17,408 + k is a trained-apart copy of neuron k, for k = 0…8,191
(`weight_tests/s2_followups.py`, `weight_tests/s2_width.py`). Neuron norms step down at index 17,408 (a
single-step search over the neuron index finds exactly 17,408; the step score is 1,400–1,800 times the
largest of 200 permuted-order nulls). For 91–98% of sampled neurons above the step, the best within-layer
match on gate, up and down at once is neuron k (chance about 6 × 10⁻⁵), and the copied-from index is the
same in both twins for 90–98%. Per layer, the median cos(neuron k, neuron 17,408 + k) for up rows and down
columns is 0.11–0.48, except layers 30–31 (0.04–0.05); a shifted control gives 0.000. (Gate rows give
0.06–0.99; in layers 1–4 the gate rows share a common direction and even the shifted control reaches
0.10–0.15.) Within each of the 8 KV groups, query heads (0, 3), (1, 4) and (2, 5) are mutual best matches on
both the q and output projections in 368–395 of the 512 (layer, group) slots; no other pair reaches 36.
Removing one copy of each duplicated neuron and head and one copy of each twin layer gives 40 layers, FFN
width 17,408 and 40 query heads in 8 KV groups: Qwen3-14B's configuration, compared directly
[above](#qwen3-32b-lines-up-with-the-released-qwen3-14b).

Qwen3.8-27B has a weaker copy map of the same kind: neuron 12,288 + k copies neuron k for k < 5,120
(per-layer median cosine 0.05–0.25; the same base neuron is the best match on gate, up and down for 13–59%
of sampled neurons above the step, against at most 7% below it), which implies a parent FFN width of 12,288.
Its query heads pair only partly ((1, 3) and (0, 2) in 26 and 23 of 64 slots). No released checkpoint was
compared with it. Qwen3.6-35B-A3B shows neither kind of width copy.

### Dead units in the MoE

Some routed-expert units of Qwen3.6-35B-A3B are exactly zero in gate, up and down together
(`weight_tests/s2_dead_units.py`): 41% of units in layer 0, and 0–212 per layer in the twinned layers. Twin
layers have identical dead sets: in 9 of the 10 twin pairs that have any, the sets match exactly (up to 212
units; expected overlap if placed independently ≤ 0.34 units), and (13, 25) shares 34 of its 38 and 34
units. Non-twin layers overlap at chance (2,049 shared units in total against 2,070 expected). With gate and
up both zero, a SwiGLU unit outputs zero and receives zero gradient in all three of its weight vectors, so
it stays dead under any gradient optimiser (and weight decay keeps zero at zero); random initialisation
produces no dead units. Dead sets that coincide unit for unit were therefore most likely present before the
twins' parameters were separated. The dense models have no dead units.

### Norm gains

`weight_tests/s3_norms.py` recomputes the gain correlations with a stricter reference: each channel is
centred on its mean over the layers of the same type, **excluding** the twin family (or families) of the
pair being tested, and twins are compared with matched controls.

- **q/k norms.** Twins vs matched controls: Qwen3-32B 0.88 / 0.93 vs 0.05 / 0.03 (all 24 twin pairs above
  every control); Qwen3.8-27B 0.58–0.91 vs 0.11–0.18 (10–12 of 12 above every control); Qwen3.6-35B-A3B
  0.87 / 0.88 vs 0.21 / 0.21 (3 of 3).
- **Not the RoPE frequency profile.** The q/k gains act on each head before rotary position embedding, so
  their channels are tied to rotation frequencies rather than being free labels. Each centred gain vector
  was split into a smooth trend over frequency, frequency-specific detail, the split within each rotation
  pair (channels k and k + R/2 rotate together), and, in the hybrid models, the channels without RoPE.
  50–75% of the variance is the within-pair split, which is an arbitrary label: a fixed 90° rotation of a
  pair applied to both q and k swaps the two gains without changing attention. Twins agree on it at
  r = 0.79–0.96 and on the non-RoPE channels at 0.60–0.86; matched controls at 0.01–0.23. The smooth trend
  (6–21% of the variance) is not twin-specific in Qwen3-32B, where adjacent layers share it about as much as
  twins do.
- **Residual-stream gains track depth, not twins.** For `attn_norm`, `ffn_norm` and `post_attention_norm`,
  adjacent layers correlate about as strongly as twins or more (Qwen3-32B `attn_norm` 0.95 vs 0.71,
  `ffn_norm` 0.87 vs 0.39), and twins beat every matched control in at most 2 of 24 pairs (Qwen3-32B), 0 of
  48 (Qwen3.8-27B) and 1 of 12 (Qwen3.6-35B-A3B). After subtracting each layer's local depth trend a twin
  component does appear (Qwen3-32B: 0.55 / 0.49 vs controls 0.01 / 0.00), but these gains are not evidence
  for the twin relation on their own.
- Per-head linear-attention parameters (`ssm_a`, `ssm_dt.bias`) correlate at 0.94–0.99 between twins. They
  may be initialised randomly per head, so they bear on shared ancestry, not on a trained origin.
- On centring: subtracting each channel's mean across n layers makes the centred vectors sum to zero across
  the layers. That forces the **norm-weighted** mean of the pairwise correlations to be at most −1/(n − 1),
  with equality only when all centred vectors have the same norm; the plain mean of the correlations is not
  constrained (Qwen3-32B `attn_norm`: plain mean +0.357, norm-weighted mean −0.040, −1/(n − 1) = −0.016).
- Storage: in the qwen35-architecture GGUFs (Qwen3.8-27B, Qwen3.6-35B-A3B) the norms are stored as the
  effective gain 1 + w (values near 1); `ssm_norm` and the Qwen3-32B norms are plain weights. Pearson r is
  unaffected by the shift.

### Qwen3-32B twin FFNs on identical and on native inputs

`native/functional_2x2.py` runs the native BF16 Qwen3-32B over 31,016 held-out tokens (English web, mixed
multilingual web, code; 66 documents), stores the residual stream entering each studied FFN, and evaluates
both FFNs of a pair on the same inputs (both read X_A or both read X_B, each through its own norm or both
through the same norm) and on their native inputs (each reads its own layer's input). 512 neurons per pair
were chosen at random before any activation was looked at. Neuron r is the median over those neurons of the
correlation across tokens between same-index activations; rank is the rank of the same-index neuron among
all 25,600 neurons of the partner layer by activation correlation (0 = best; chance median 12,800).

| Pair | Neuron r, same input | Neuron r, native | Same index best, same / native | Median rank, same / native |
|---|---|---|---|---|
| twin (8, 32) | 0.13 | 0.08 | 45–47% / 31% | 1 / 10 |
| twin (16, 40) | 0.14–0.18 | 0.07 | 56–62% / 42% | 0 / 1 |
| twin (24, 48) | 0.12–0.16 | 0.08 | 55–58% / 39% | 0 / 1 |
| twin (31, 55) | 0.02–0.04 | 0.01 | 12–19% / 12% | 450–2,800 / 3,972 |
| controls (8, 36) (16, 44) (24, 52) (7, 31) | 0.000 | 0.000 | 0% / 0% | 11,900–14,400 |

Ranges in a cell run over the four same-input conditions. Shuffled-index controls give r = 0.000 ± 0.001
in every condition; bootstrap 95% intervals over documents for the twin r values are 0.01–0.03 wide (see
`results/native/functional_2x2.txt`). On identical input, the same index is the best match among 25,600 for
about half of the sampled neurons in three of the four twin pairs and for none in the controls, so the
weight correspondence is also a functional one. It is partial (r 0.12–0.18), and it roughly halves on
native inputs, so part of the difference between twins comes from what they read at their two depths. The
top pair (31, 55) has diverged most, as in the weights. The overlap of each neuron's top-1% contexts
(Jaccard) is 0.03–0.06 for the first three twin pairs and 0.02–0.04 for (31, 55), against 0.01–0.02 with
shuffled indices. The cosine between whole-FFN outputs is not a clean measure here: the control (7, 31)
reaches 0.37 when both read layer 7's normalised input, presumably through shared large output directions.

### Gain-reset gradients (inconclusive)

`native/gain_reset_gradients.py` asks whether the current matrices would regenerate the twins' shared q/k
gain pattern. For four twin pairs and four depth-matched controls, both layers' q_norm and k_norm gains are
reset, everything else is frozen, and the gradient of next-token loss with respect to the four gain vectors
is summed over 8 held-out sequences of 512 tokens. With a plain SGD step from a constant, the correlation of
the updated gain vectors equals the correlation of their gradients. Arms: all-ones; a constant equal to each
vector's original mean (which keeps the attention-logit scale); and the trained gains as reference.

| Arm | Twins: gradient r, q / k (q per pair) | Controls: gradient r, q / k (q per pair) | corr(−gradient, own original pattern), twins / controls |
|---|---|---|---|
| ones | 0.42 / 0.41 (0.35, 0.40, 0.06, 0.86) | 0.08 / 0.06 (0.01, −0.09, 0.28, 0.13) | 0.13 / 0.11 |
| mean | −0.02 / −0.03 | −0.01 / 0.01 | 0.18 / 0.17 |
| trained | 0.01 / −0.24 (single pairs −0.64 to 0.59) | 0.03 / 0.01 | −0.08 / −0.07 |

From all-ones gains the twins' gradients are more correlated than the controls', so the aligned matrices can
produce correlated gain updates; from a scale-preserving constant they are not. In no arm does the first
step point toward the original gain profile more for twins than for controls. Single-pair estimates are
noisy (at the trained point they range from −0.64 to 0.59 on 4,096 tokens). The result is inconclusive,
leaning against "the current matrices alone regenerate the twin gain pattern immediately". The full
reset-and-relearn training was not run.

### Qwen3.8-27B triple covariances

`weight_tests/s6_triplets.py` computes raw per-element covariances C_AB, C_AC, C_BC for every triplet
(i, i + 16, i + 32), i = 8–23, and tensor family, with 8 disjoint row groups giving a replication spread. Two
restricted histories with independent, zero-mean drift make sharp predictions: a simultaneous triple copy
predicts C_AB = C_AC = C_BC; a successive copy in which B and C share extra history predicts
C_BC > C_AB = C_AC.

- Matrices: C_AC is 0.53–0.86 of C_AB and C_BC is 0.93–1.51 of C_AB (medians over triplets) in every
  family except `ssm_conv1d` (1.11 and 1.66). Neither restricted model fits: of the 136 matrix
  (triplet, family) units, none fits the simultaneous pattern and 1 fits the successive B–C pattern
  ("equal" meaning |t| < 2 across the 8 row groups). A simultaneous copy with copy-specific shrinkage
  (C_XY = s_X s_Y V) fits every matrix triplet, with implied shared-variance fractions of 0.22–0.56 for B
  and 0.05–0.32 for A and C (medians per family, `ssm_conv1d` higher).
- Gains (q/k norms, `ssm_norm`) show the successive pattern instead: C_AB ≈ C_AC and C_BC ≈ 2 × C_AB
  (medians); 10 of their 20 units fit the successive B–C pattern and 2 the simultaneous one.
- Unequal shrinkage, or correlated updates between adjacent blocks, can produce either pattern. Neither
  simple history fits the matrices without copy-specific shrinkage, the gains favour extra B–C history, and
  endpoint covariances do not identify the history.

## Was the 32B's twin block trained as a loop?

A suggestion made in a LessWrong comment: Qwen trained a looped transformer (one weight-shared block applied repeatedly),
fixed the loop count at 2 and then untied the weights, which would also leave two untied copies of a block.
That history and plain stacking differ only in whether the shared block was trained tied before the split.
Qwen3-32B's weights are all post-split, so both tests use the released Qwen3-14B, whose weights correspond to
the 32B's index by index. Scripts and pre-registration are in `loop_hypothesis/`, outputs in
`results/loop_hypothesis/`.

**Where the split state sits** (`star_decomposition.py`). Three-leaf star: T = Qwen3-14B layer p, L = 32B
layer p (lower copy), U = 32B layer p + 24 (upper copy), p = 8–31. If all three descend from one state S by
independent drifts, cos(X, Y) = s_X s_Y, so with a = cos(T, L), b = cos(T, U), c = cos(L, U):
s_T = √(ab/c), s_L = √(ac/b), s_U = √(bc/a). Medians over p = 8–31:

| Tensor | s_T (14B vs split state) | s_L (lower copy) | s_U (upper copy) |
|---|---|---|---|
| k_proj | 0.89 (range 0.74–0.93) | 0.32 | 0.90 |
| v_proj | 0.95 (0.80–0.97) | 0.53 | 0.93 |
| gate_proj, neurons 0–511 | 0.84 (0.73–0.86) | 0.34 | 0.73 |
| up_proj, neurons 0–511 | 0.85 (0.75–0.89) | 0.33 | 0.71 |

The split state is close to the released 14B in every layer and tensor, least close at the bottom of the
block (p = 8–12: s_T 0.73–0.88) and closer from p = 16 up (0.82–0.97). a and b come from the 14B-vs-32B cosines
in `results/native/qwen3_14b_parent_bf16.npz`, c from the GGUF over the same rows. If the 14B lineage was the
starting point, any tied phase moved the block much less than the untied training after the split did. This
does not exclude a looped model from which both the 14B and the 32B were later derived.

**Does the 14B tolerate running a block twice?** (`loop_tolerance.py`; prediction in
`loop_hypothesis/predictions.md`, written before running.) Layer modules are reused, not copied: layers 0 to
s + 23, then s to s + 23 again, then s + 24 to 39 (64 layers, like the 32B). 24 held-out sequences of 512
tokens (8 English web, 8 multilingual, 8 code); baseline NLL 1.744 nats/token. ΔNLL with bootstrap 95%
interval over sequences:

| Window applied twice | ΔNLL | | Window applied three times | ΔNLL |
|---|---|---|---|---|
| 0–23 | +2.87 [2.36, 3.40] | | 0–23 | +4.57 [3.98, 5.11] |
| 2–25 | +3.86 [3.16, 4.57] | | 8–31 | +0.43 [0.34, 0.52] |
| 4–27 | +3.28 [2.91, 3.62] | | 16–39 | +1.87 [1.59, 2.13] |
| 6–29 | +0.14 [0.10, 0.18] | | | |
| **8–31** (the 32B's twin block) | **+0.15 [0.11, 0.19]** | | | |
| 10–33 | +0.18 [0.15, 0.22] | | | |
| 12–35 | +0.16 [0.13, 0.20] | | | |
| 14–37 | +0.14 [0.11, 0.17] | | | |
| 16–39 | +0.79 [0.64, 0.94] | | | |

Pre-registered loop signature: ΔNLL(8–31) < 0.15 **and** < half the mean of the neighbouring windows 6–29 and
10–33. The first part is met (0.147); the second is not (the neighbours give 0.137 and 0.183, half their mean
is 0.080). There is no dip at 8–31: every window starting at layers 6–14 costs about the same (+0.14 to
+0.18), windows that include layers 0–5 are catastrophic, and the window that includes the last layers costs
+0.79.

**Result: no support for the loop hypothesis.** The 14B tolerates duplicating any middle window about equally,
so the low cost of repeating 8–31 is a property of its middle layers, not a mark of looped training on that
block; 8–31 is a wide window that avoids the fragile first six and last few layers, which is what a stacking
recipe would pick. The width copies (duplicated FFN neurons and query heads) show that an explicit copying
step was used anyway, and copying layers is the same operation. Not excluded: a short tied phase after
copying, or a looped common ancestor of the 14B and the 32B whose loop tolerance the 14B later lost (the
caveat stated with the prediction). The data give no positive support for either.

## Qwen3-14B in the 32B's family tree

Two histories fit the index alignment between Qwen3-14B and Qwen3-32B (distillation into a fresh network does
not, because it would not preserve neuron indices):

- (a) the 14B is the pre-growth (14B-shaped) model, trained on; the 32B was grown from it;
- (b) the 32B was grown and trained, then the 14B was pruned back out of it (dropping duplicate layers,
  neurons and heads) and trained on.

Predictions and known confounds were written down before running (`loop_hypothesis/phylogeny_predictions.md`).

**Five-taxon trees** (`phylogeny.py`, `tree_fit.py`; `results/loop_hypothesis/phylogeny.txt`, `tree_fit.txt`).
For each width-copied unit j (FFN neurons j < 8,192: gate rows, up rows, down columns; query heads 0–2 of each
KV group: q rows, o columns) and block layer p = 8–31, five taxa: T (14B unit j), Lj and Uj (32B unit j at
layers p and p + 24), Lc and Uc (its width copy at p and p + 24). Distance d = −log cos over all such units;
all 15 unrooted five-leaf topologies are fitted by non-negative least squares.

- At the bottom of the block the best tree in all five tensor families is ((Lj,Lc),T,(Uj,Uc)): the 14B sits
  at the centre, between the lower-copy pair and the upper-copy pair. This holds for p = 8–10 in every
  family and up to p = 11 (up), 12 (gate, o) or 13 (q).
- Above that, up to p = 31, the best tree is ((T,Uj),Uc,(Lj,Lc)): the 14B groups with the same-index unit of
  the upper copy, with a shared branch of 0.02–0.26 (mostly 0.1–0.2); the runner-up swaps Uj for Uc at a
  slightly worse fit (one exception, o at p = 19, where that runner-up fits marginally better). The fits
  are only roughly tree-like: residuals (0.04–0.66) are comparable to the internal edges.
- Lj and Lc pair up in every best tree. As stated in advance, same-layer copies receive nearly identical
  gradients and co-adapt, so this says little about whether width or depth grew first.

Against the pre-registered predictions: (b) predicted a clear {T, Uj} grouping across the whole block
including p = 8–10; it is absent at p = 8–10 in every family. (a) predicted that T never groups with a single
depth copy; it does from p = 11–14 up. The pre-registration named this pattern, a {T, Uj} grouping only above
the bottom of the block, as the signature of convergence: away from the seam the upper copy has the same
neighbours as the 14B (32B layers 56–63 are copies of 14B layers 32–39), so the two can drift in correlated
ways.

**The 14B's own drift across all 40 layers** (`drift_profile.py`; not pre-registered). Per 14B layer, a
three-leaf star of the 14B unit j, the 32B unit j and its width copy (using the 32B copy closer to the 14B);
dT = −log s_T is the 14B's distance from the point where the 32B's two width copies split. (b) suggests a
spike at the 14B's internal seam (layers 8–11, re-adapting after pruning) that layers 0–7 do not share.

| 14B layers | gate | up | down | q | o |
|---|---|---|---|---|---|
| 0–7 | 0.18–0.57 | 0.23–0.65 | 0.18–0.56 | 0.36–0.75 | 0.44–0.56 |
| 8–11 | 0.36–0.51 | 0.33–0.44 | 0.23–0.34 | 0.67–0.79 | 0.28–0.63 |
| 12–15 | 0.05–0.28 | −0.01–0.23 | −0.03–0.15 | 0.20–0.54 | 0.00–0.28 |
| 16–39 | ≤ 0.01 | ≤ −0.04 | ≤ −0.06 | ≤ 0.15 | ≤ −0.01 |

Layers 8–11 lie within the range of layers 0–7 in every family, but that range is widened by layer 0; against
layers 1–7 alone, layers 8–9 are higher for gate (0.36 and 0.51 against 0.18–0.29) and at the top of the range
for up, down and q. We read this as no clear seam-specific spike, not as a clean null. The main pattern is a
depth trend: the 14B is farther from the split point in its lower third and sits at (negative values: beyond,
i.e. the star is not additive there) the split point from about layer 16 up, including layers 32–39, which
were never duplicated in depth.

**Reading (inference).** Favoured: (a), the 14B continued from the pre-growth model. At the bottom of the
block the 14B attaches at the centre of the tree, where the pre-growth parent should sit; the {T, Uj} grouping
appears only where the pre-registered convergence confound applies; and in the upper layers the 14B sits at
the split point of the 32B's width copies. (b) is distinguishable only if the 32B trained substantially before
the 14B was cut out of it, which would leave a long branch shared by the 14B and one 32B copy; none is found
beyond the short, not robust one in the upper layers. A variant in which the 14B was cut out immediately after
growth is practically the same as (a). Not determined: whether width or depth grew first, and whether the
14B's lower layers moved after the branch point or the 32B's lower layers moved in a correlated way.

## The Qwen3 family map

Scripts in `family/`, outputs in `results/family/`. Local BF16 checkpoints: Qwen3-0.6B, 1.7B, 4B, 14B;
Qwen3-32B from the Q8_0 GGUF; Qwen3-8B, Qwen3-30B-A3B and Qwen3-235B-A22B tensors fetched with HTTP range
requests (revisions under [Reproducing](#the-qwen3-family-family)).

### Twin blocks in the other Qwen3 models

- **Qwen3-0.6B, 1.7B, 4B** (`analyze_small.py twins|width`, every matrix, every pair of layers): no twin
  layers. The largest same-index cosine between any two layers is at most 0.006 for q, k, v, o, up and down
  projections; gate_proj reaches 0.009–0.028, between nearby layers (Qwen3-32B shows the same for gate rows at
  non-twin pairs). No fixed-offset neuron copies; a few query heads within a KV group resemble each other
  (best-match cosine up to 0.45).
- **Qwen3-8B, Qwen3-14B**: no twin structure in the q/k-norm gains (`qknorm_within.py`; the four best offsets
  have mean r of at most 0.09 and no pair above 0.5). In Qwen3-32B the same scan finds offset +24 with all 24 pairs 8–31 ↔ 32–55
  above 0.5, so the gains alone recover a known twin block.
- **Qwen3-30B-A3B** (48 layers): layers 6–17, 18–29 and 30–41 are three copies of one 12-layer block; layers
  0–5 and 42–47 are unique. The parent this implies has 24 layers (6 + 12 + 6). This confirms the observation
  by Maciej Kilian, who saw the same expert index highly active 12 and 24 layers apart in Qwen3-30B-A3B's
  router statistics and asked whether Qwen3 used depth-wise upcycling
  (https://x.com/kilian_maciej/status/1982612297874026731).
- **Qwen3-235B-A22B** (94 layers): layers 0–9 and 84–93 are unique; a 20-layer block (parent layers 10–29 of
  a 40-layer parent) appears as 10–29, 30–46 (parent 10–26), 47–63 (parent 13–29) and 64–83 (parent 10–29),
  giving 102 twin pairs.

Matrix-level check (`moe_matrix_twins.py`: k_proj, v_proj and router of every layer; twin pairs are the pairs
whose mean of the three cosines exceeds 0.05, which picks out exactly 36 and 102 pairs, i.e. the block structure
above):

| | k_proj, twins / other pairs | v_proj | router, expert-centred | router, raw |
|---|---|---|---|---|
| Qwen3-30B-A3B (36 twin pairs) | 0.475 / 0.0001 ± 0.0021 | 0.675 / 0.0000 ± 0.0010 | 0.807 / −0.0001 ± 0.0101 | 0.839 / 0.37 ± 0.10 |
| Qwen3-235B-A22B (102 twin pairs) | 0.508 / −0.0000 ± 0.0016 | 0.722 / −0.0000 ± 0.0009 | 0.823 / 0.0007 ± 0.0076 | 0.871 / 0.33 ± 0.13 |

Twin values are means (minima 0.26–0.71); the largest non-twin value is 0.018 for k_proj and v_proj and 0.040
for the centred router. Router rows are experts, so the router match is expert by expert. In Qwen3-30B-A3B the
closest pair of copies is 18–29 ↔ 30–41 and the most distant 6–17 ↔ 30–41 (for example layer 18 with 30:
k 0.53, v 0.81, router 0.86; layer 6 with 18: 0.39, 0.67, 0.79; layer 6 with 30: 0.28, 0.60, 0.71), the same
ordering as in Qwen3.8-27B.

Expert-level check (`moe_experts.py`: whole experts, gate, up and down, at twin layers and one adjacent
non-twin control layer):

- Qwen3-30B-A3B, layers 6, 18, 30 (control 7), experts 0, 37, 85, 120: the same expert across twin layers
  has cosine 0.22–0.37 for three experts and 0.07–0.13 for expert 120; different experts at most 0.008; the
  same expert against the control layer at most 0.005. Per neuron (layers 6 and 18), the same neuron index
  has median cosine 0.27–0.29 (0.09 for expert 120) against 0.000–0.007 with neurons shuffled.
- Qwen3-235B-A22B, layers 13, 33, 47, 67 (control 14), experts 5 and 77: same expert 0.295–0.415, other
  experts at most 0.003, control at most 0.001; same neuron index median 0.295–0.318 (shuffled 0.002).

So in both MoEs the experts, and the neurons inside them, kept their indices across the copies, as in
Qwen3.6-35B-A3B.

### Relations between models

All Qwen3 models use 128-dimensional q/k-norm gains, so these can be compared across models of any width
(`qknorm_screen.py`): each model's gain vectors are centred on that model's mean profile and every layer of
one model is correlated with every layer of another (mean of the q and k correlations).

| Pair | Median best r | Layer map |
|---|---|---|
| 0.6B ↔ 1.7B | 0.98 | p ↔ p for all 28 layers (min 0.91) |
| 4B ↔ 8B | 0.99 | p ↔ p for all 36 layers (min 0.96) |
| 14B ↔ 32B | 0.99 | p ↔ p (p ≤ 9) or p + 24 (p ≥ 10); both copies match in 8–31 (min 0.98) |
| every other pair, including both MoEs against every dense model and against each other | 0.18–0.29 | none |

The null (one model's channels shuffled) is 0.12–0.16. Residual dimensions can also be matched through
embedding columns, since all Qwen3 models share a tokenizer (`analyze_small.py embed`, first 30,000 tokens;
null 0.02):

- 14B ↔ 32B: median best |corr| 0.52, and the best match is the same index for 100% of dimensions: one
  residual basis.
- 0.6B ↔ 1.7B: 0.26, mapping onto 1,008 distinct 1.7B dimensions out of 1,024, never at the same index: a
  reshuffled subset.
- 4B ↔ 8B: 0.07, no dimension-wise correspondence; see the next section for why.
- All pairs from different groups: 0.06–0.09.

**Lineages.** The released Qwen3 weights fall into five groups with no detectable relation between groups:
{0.6B, 1.7B}, {4B, 8B}, {14B, 32B}, {30B-A3B}, {235B-A22B}. Within each group the weights correspond layer by
layer; between groups nothing measured here exceeds the unrelated-pair level. "No detectable relation" is a
statement about these measurements: models trained from different initialisations, or related through a
transformation these tests cannot see, would look the same.

### The 0.6B ↔ 1.7B map

`pair_06_17.py` maps each 0.6B residual dimension d to the 1.7B dimension π(d) (with sign) whose embedding
column correlates best, fitted on tokens 0–14,999. On held-out tokens 15,000–29,999 the same partner is the
best match for 99% of dimensions. The map then aligns quantities it was not fitted to:

- per-dimension norm gains: r = 0.84 (input_layernorm, median over layers), 0.90 (post_attention_layernorm),
  0.91 (final norm); with π shuffled, −0.01;
- attention weights with residual columns remapped (head channels are 8 KV × 128 and 16 Q × 128 in both
  models): same-row cosine at the same layer, median over layers, k_proj 0.17, q_proj 0.14, v_proj 0.04,
  o_proj 0.03; against another layer or with π shuffled, 0.000;
- FFN neurons (gate and up rows remapped): each 0.6B neuron's best 1.7B match has median cosine 0.10–0.20
  (second best 0.08–0.09), the down column agrees at that match (0.10–0.20), and the matches are nearly
  one to one (2,667–3,003 distinct partners for 3,072 neurons).

π is not monotone (Spearman ρ between d and π(d): −0.01): the dimensions were reordered, not merely
subsetted.

### The 4B ↔ 8B relation

The 4B and 8B have no dimension-wise residual correspondence, but they are related by one linear map of the
residual stream (`pair_4_8.py`, `weight_map.py`, `embed_origin.py`):

- Head-side Gram matrices W Wᵀ of k_proj and v_proj (unchanged by any orthogonal map of the residual stream):
  matched layers 0.81 (k) and 0.84 (v) against 0.43 and 0.66 for other layer pairs. Unrelated pairs (4B vs
  1.7B or 14B, 8B vs 14B, 1.7B vs 8B) show no matched-layer excess (for example 4B vs 14B: 0.45 vs 0.43,
  0.65 vs 0.66). 0.6B vs 1.7B: 0.60 vs 0.31 (k), 0.51 vs 0.47 (v).
- One linear map Q (4B residual = Q · 8B residual) fitted by ridge regression on the k/v weights of five
  paired layers predicts the k/v weights of the other five at median cosine 0.31 (0.13–0.46). The same
  procedure gives 0.000 (|values| ≤ 0.001) for all four unrelated pairs. For the related 0.6B → 1.7B pair it
  gives only 0.03 (0.015–0.16); that pair's relation is shown more directly by the permutation map above.
- A map fitted on embeddings instead (4B embeddings regressed on 8B input embeddings) does not generalise
  (held-out R² 0.007, k/v cosine 0.03–0.04), while for 0.6B/1.7B it does (R² 0.55). The next point explains
  the difference.
- Through Q (fitted on all ten paired layers), the 4B's tied embedding table matches the 8B's output head
  (lm_head): median per-token cosine 0.42, 100% (rounded) of the first 30,000 tokens above 0.2, shuffled tokens
  0.008. It does not match the 8B's input embeddings (0.075, no token above 0.2). Qwen3-4B ties its input and
  output embeddings; Qwen3-8B does not, and its two tables are unrelated (median cosine 0.004).

## Direction: channel sortedness and the pruning reading

Weight correspondences are symmetric: they show that two models are related, not which came from which.
`sortedness.py` and part 1 of `pair_4_8.py` test a property that can carry direction. Random initialisation
is symmetric under permutations of residual dimensions and FFN neurons, so a model trained from scratch has
no reason for its channel order to track any per-channel property. Structured pruning that ranks channels by
an importance score and keeps the top ones in rank order leaves the kept channels sorted.

Spearman ρ between channel index and per-channel properties (residual dimensions: embedding column norm,
mean |input_layernorm gain|, input-column norm of the attention projections; FFN neurons: gate-row, up-row and
down-column norm, median over sampled layers):

| Model | Residual dimensions | FFN neurons |
|---|---|---|
| Qwen3-0.6B | +0.20, −0.37, +0.76 (q/k/v; k/v only: +0.40) | +0.40, +0.49, +0.51 |
| Qwen3-1.7B | −0.01, −0.04, +0.04 | −0.01, −0.00, −0.01 |
| Qwen3-4B | +0.24, −0.15, +0.31 (q/k/v; k/v only: −0.19) | −0.02, +0.18, +0.38 |
| Qwen3-8B | −0.01, −0.00, +0.01 (k/v) | not measured |
| Qwen3-14B | +0.00, +0.00, +0.01 | +0.01, +0.00, +0.01 |
| Qwen3-32B | +0.02, +0.00, +0.01 | −0.09, −0.22, −0.10 |

The smaller member of each small lineage is sorted and the larger is not. In addition, the 0.6B's dimension
order tracks properties of the matched 1.7B dimensions (`pair_06_17.py`, part 4): Spearman ρ between the 0.6B
index d and the 1.7B importance rank of π(d) is −0.88 for the q/k/v input-column norm and +0.62 for the gain,
and the 1.7B dimensions that have a 0.6B partner differ systematically from those that do not (median
importance rank 1,518 vs 624 of 2,048 by q/k/v column norm, 758 vs 1,096 by gain). The Qwen3-32B's FFN
values come from norm steps at the width-copy boundaries (neurons 8,192 and 17,408): within each of the three
segments |ρ| ≤ 0.03 (a side check on ffn_up in layers 0, 10, …, 60, not in the committed outputs).

**Reading (inference).** Qwen3-0.6B was cut from a 1.7B-like model by importance-ranked width pruning (same
depth and heads, a reordered subset of residual dimensions and FFN neurons), and Qwen3-4B from an 8B-like model,
through a linear mixing of the residual stream rather than a subset, with its tied embedding table taken from
the 8B's output head. Within the {14B, 32B} lineage the direction is the opposite, growth: the 32B's copies
require a 14B-shaped ancestor. This direction claim for the small models is circumstantial. Sortedness is what
ranked pruning leaves behind, but another process that orders channels (or a deliberate reordering) would leave
it too, and nothing here identifies the checkpoints involved: as with the 14B and the 32B, the larger released
model may itself have trained on after the smaller one was derived from an earlier state of it.

## Qwen3-14B and Qwen3-32B activations share one residual basis

`acts_14_32.py` compares the residual streams of the two models on the same 36 held-out texts (24 English web,
12 mixed; 10,179 tokens). The 32B's residual stream after each of its 64 layers was dumped with llama.cpp from
the Q8_0 GGUF; the 14B (BF16, transformers) runs on exactly the same token ids. For every 14B layer p and 32B
layer m: Pearson r of residual dimension d with the same dimension d of the other model, across tokens,
median over dimensions.

- The best-matching 32B layer for every one of the 40 14B layers is a predicted one: p for p ≤ 11, p + 24 for
  12 ≤ p ≤ 31 (the upper copy), p + 24 for p ≥ 32. At those layers r = 0.41–0.74; with dimensions shuffled,
  |r| ≤ 0.004; the median over all 32B layers is 0.08–0.31.
- For 14B layers 8–31 the switch from the lower copy (best for p ≤ 11) to the upper copy (best from p = 12)
  matches the weights, where the upper copy is the closer one from p ≈ 10–11.
- For each 14B dimension, the most correlated 32B dimension is the same index for 93–100% of dimensions at the
  predicted layer pairs tested. This also holds at a non-predicted pair (14B layer 20 vs 32B layer 30: 86%),
  because the residual stream keeps its coordinates through depth; the shared basis itself is therefore not
  layer-specific, while the r profile across layers is.

So the index alignment of the weights carries over to the activations: the two models compute in the same
residual coordinates, layer for layer along the expansion map.

## What the weights show, and what they do not

**Established by the weights: twins share an origin, and so do the members of each lineage.** The order of
neurons, heads and experts within a layer is arbitrary: permuting them consistently across the layer's
matrices gives an equivalent network. Layers initialised independently therefore have no reason to line up
index by index, and non-twin pairs sit where unrelated vectors do (Qwen3-32B native weights, eight control
pairs: below 0.0011 in absolute value for six of the seven matrix families, 0.0025–0.0070 for `ffn_gate`).
The twin pairs are far outside that range for every attention, linear-attention projection, FFN and expert
matrix: in fresh row samples as well as the original one, in the native BF16 weights, neuron by neuron and
expert by expert, and, for the Qwen3-32B FFNs, in activations on identical inputs. The same holds in the
MoE models Qwen3-30B-A3B and Qwen3-235B-A22B for k_proj, v_proj, the router and whole experts. The same
argument applies between models: Qwen3-14B and Qwen3-32B line up index by index exactly where a width and
depth expansion of a 14B-shaped network predicts, at cosines 0.11–0.93, and nowhere else (at most 0.003;
`gate_proj` 0.032), and their activations share one residual basis. Qwen3-0.6B and Qwen3-1.7B are related by
a permutation (with signs) of residual dimensions and neurons, and Qwen3-4B and Qwen3-8B by a linear map of the
residual stream, both checked on quantities the maps were not fitted to. Across lineages nothing measured
here rises above the unrelated-pair level.

The correspondence is broad but not uniform: it is weaker at the block ends, for the Qwen3.8-27B
8–23 ↔ 40–55 relation, for the Qwen3-32B width-copy neurons, and for a minority of routed experts.

**Inferred: the shared origin was most likely a trained network.** Several lines of evidence point the same
way; none is conclusive alone.

- Qwen3-32B lines up with a released, trained model along the expansion map, and its layers next to the
  seam, whose neighbours a stacking changes, are the ones furthest from the 14B.
- In Qwen3.6-35B-A3B, twin layers have identical sets of exactly-zero expert units. Such units cannot come
  from random initialisation and cannot change under gradient training, so they existed before the twins
  separated.
- Centred twin q/k-norm gains correlate at 0.58–0.93 against 0.03–0.21 for matched controls, and the
  agreement sits in the arbitrary within-pair split and non-RoPE channels rather than in the frequency
  profile. Gains start at 1 under standard initialisation, so a shared random start gives no shared gain
  pattern. But gains learned in parallel on already-aligned matrices could also correlate, and the
  gain-reset gradient test neither confirmed nor excluded that.

This fits depth up-scaling as in SOLAR (arXiv:2312.15166): a trained model's layers are stacked with an
overlap and the result is trained further. The twin positions match that recipe:

- Qwen3-32B = layers 0–31 followed by layers 8–39 of a 40-layer parent, widened (FFN 17,408 → 25,600 by
  copying neurons 0–8,191; 40 → 64 query heads by duplicating three heads per KV group). The released
  Qwen3-14B has that shape and lines up with it; the [family-tree analysis](#qwen3-14b-in-the-32bs-family-tree)
  favours the 14B being that parent trained on, over it being pruned back out of the 32B.
- Qwen3-30B-A3B = layers 0–17 of a 24-layer parent, then its layers 6–17 twice more, then layers 18–23
  (parent layer k, 6 ≤ k ≤ 17, appears at k, k + 12 and k + 24).
- Qwen3-235B-A22B = a 40-layer parent whose layers 10–29 appear four times, two of the copies partial (see
  the table at the top).
- Qwen3.6-35B-A3B = layers 0–19 followed by layers 8–27 of a 28-layer parent (parent layer k, 8 ≤ k ≤ 19,
  appears at positions k and k + 12).
- Qwen3.8-27B: a 16-layer block appears three times, with partial width copies (parent FFN width 12,288).
  The covariances do not decide between one triple copy and two successive copies (see
  [triple covariances](#qwen38-27b-triple-covariances)).

For the four models other than Qwen3-32B the block structure is known but the parent is not: no released
checkpoint was found that corresponds to it (for the two original Qwen3 MoEs, no other released Qwen3 model
matches them in the q/k-norm screen). The loop hypothesis (a weight-shared block trained looped, then untied)
was tested for Qwen3-32B and found [no support](#was-the-32bs-twin-block-trained-as-a-loop).

Other histories are not excluded by the weights. A block whose weights were tied during training and later
untied produces the same endpoint pattern as a copy. Identical random initialisation of the blocks (for
example one seed per pipeline stage) followed by training is harder to reconcile with the identical
dead-unit sets, and for Qwen3-32B it would also require the 14B and the 32B to have started from the same
weights laid out along the expansion map; the weights alone still cannot rule it out.

**Inferred, on circumstantial evidence: the small dense models were pruned from larger ones.** The smaller
member of each small lineage has channels sorted by per-channel properties and the larger does not, which is
what importance-ranked pruning leaves; on that basis Qwen3-0.6B came from a 1.7B-like model and Qwen3-4B from
an 8B-like model (through a linear mix of the residual stream, with its tied embeddings taken from the 8B's
output head). See [direction](#direction-channel-sortedness-and-the-pruning-reading) for what this does not
rule out.

**Not recoverable from the weights.**

- Which twin is the original. A stacking copies one parent layer into both positions, and both copies have
  since been trained. Neither twin is privileged.
- Whether the parameters were copied after training or trained while tied and then untied: a copied model
  and an untied model that reach the same parameters and optimiser state have identical subsequent
  trajectories.
- Which checkpoint was expanded into Qwen3-32B (the released Qwen3-14B, an earlier checkpoint of it, or a
  sibling), and whether width or depth grew first; likewise which checkpoints the small models were cut from.
- The parent checkpoints of Qwen3-30B-A3B, Qwen3-235B-A22B, Qwen3.6-35B-A3B and Qwen3.8-27B.
- The copying history of Qwen3.8-27B (see above).

**No public account.** We found no Qwen publication that describes any of this. The Qwen3 technical report
(arXiv:2505.09388) lists Qwen3-32B as a 64-layer model pretrained on 36T tokens; we found no description
there of how it or the other models were initialised. A related public
observation: Maciej Kilian asked on X whether Qwen3 uses depth-wise upcycling, after seeing the same expert
index highly active 12 and 24 layers apart in Qwen3-30B-A3B's router statistics
(https://x.com/kilian_maciej/status/1982612297874026731). The weights confirm it: Qwen3-30B-A3B repeats a
12-layer block three times (from a 24-layer parent), with experts and their neurons in place.

## Twin neurons in Goodfire's neuron gallery

Goodfire's neuron gallery (https://already-clean.goodfire.com/data/neuron-gallery.json) has 1,000 neuron
write-ups, 500 for Qwen3-32B and 500 for Qwen3.6-35B-A3B. `gallery_twins.py` lists the gallery neurons that
share an index exactly one twin offset apart (dense: same neuron at layers i and i + 24, 8 ≤ i ≤ 31; MoE:
same expert and neuron at layers i and i + 12, 8 ≤ i ≤ 19). Output (`results/gallery_twins.txt`):

| Qwen3-32B pair | Titles |
|---|---|
| L12_N12088 / L36_N12088 | "Forty-two, when it’s the answer to everything: a Hitchhiker’s Guide neuron" / "A Hitchhiker's Guide neuron that lights up before the Answer" |
| L26_N17433 / L50_N17433 | "Little did they know: a neuron for narrated knowledge gaps" / "Little did they know: a neuron for narrated past ignorance" |
| L30_N8699 / L54_N8699 | "The pending verdict when a rival makes something look lesser" / "A neuron that sees “put them to shame” coming" |
| L31_N16641 / L55_N16641 | "Costumes as a figure of speech" / "A neuron for 'X is really a Y in disguise'" |

If gallery neurons had been chosen independently of their index, the expected number of such pairs would
be 0.05 (sum over i of the number of gallery neurons at layer i times the number at layer i + 24, divided
by 25,600 neurons per layer); there are 4, and in each pair the two titles describe related behaviour. For
Qwen3.6-35B-A3B there are none (0.01 expected). How the gallery's neurons were selected is not known to us,
so the chance baseline is indicative only; the pre-registered functional test
[above](#qwen3-32b-twin-ffns-on-identical-and-on-native-inputs) does not depend on it.

## Reproducing

Requirements: Python 3 with `numpy` and `gguf` (the GGUF reader from llama.cpp, published on PyPI) for the
weight scripts; `scipy` for `loop_hypothesis/tree_fit.py` and three scripts in `family/`; `torch` and
`transformers` for the scripts that run models (`native/functional_2x2.py`, `native/gain_reset_gradients.py`,
`loop_hypothesis/loop_tolerance.py`, `family/acts_14_32.py`, `family/dump_residuals_hf.py`).
`gallery_twins.py` uses only the standard library.

```
pip install gguf numpy scipy
```

Models (SHA-256 of the GGUF files used):

| Model | Hugging Face repo | File or revision | Size | SHA-256 |
|---|---|---|---|---|
| Qwen3-32B | Qwen/Qwen3-32B-GGUF | Qwen3-32B-Q8_0.gguf | 34.8 GB | 2c50eb8aad05047dbf24fa014eb621adf552e14176cabe0c5db4ef38c91e2169 |
| Qwen3.8-27B | lmstudio-community/Qwen3.8-27B-GGUF | Qwen3.8-27B-Q8_0.gguf | 29.0 GB | f0fc43cfd802bd87bc7b95f2120553a6407848e3229126b623cc397427857e8a |
| Qwen3.6-35B-A3B | lmstudio-community/Qwen3.6-35B-A3B-GGUF | Qwen3.6-35B-A3B-Q8_0.gguf | 36.9 GB | e56476632cf65ab1a32c135aead9eba4278f3bd3eb0499aa74304566d6e27d5b |
| Qwen3-14B (selected tensors fetched by `qwen3_14b_parent.py`) | Qwen/Qwen3-14B | revision 40c069824f4251a91eefaf281ebe4c544efd3e18 | about 1.4 GB fetched | |
| Qwen3-32B BF16 (`native/` scripts only) | Qwen/Qwen3-32B | revision 9216db5781bf21249d130ec9da846c4624c16137 | 61 GiB | |

```
hf download Qwen/Qwen3-32B-GGUF Qwen3-32B-Q8_0.gguf --local-dir .
hf download lmstudio-community/Qwen3.8-27B-GGUF Qwen3.8-27B-Q8_0.gguf --local-dir .
hf download lmstudio-community/Qwen3.6-35B-A3B-GGUF Qwen3.6-35B-A3B-Q8_0.gguf --local-dir .
```

### Twin layers within each model (`twin_layers.py`)

Commands that produce the files directly in `results/`:

```
python3 twin_layers.py Qwen3-32B-Q8_0.gguf scan
python3 twin_layers.py Qwen3-32B-Q8_0.gguf 24 8 31

python3 twin_layers.py Qwen3.8-27B-Q8_0.gguf scan
python3 twin_layers.py Qwen3.8-27B-Q8_0.gguf 16 8 23
python3 twin_layers.py Qwen3.8-27B-Q8_0.gguf 16 24 39
python3 twin_layers.py Qwen3.8-27B-Q8_0.gguf 32 8 23

python3 twin_layers.py Qwen3.6-35B-A3B-Q8_0.gguf scan
python3 twin_layers.py Qwen3.6-35B-A3B-Q8_0.gguf 12 8 19

python3 gallery_twins.py
```

The GGUF is memory-mapped and only the sampled rows are read, so a run takes about 5–15 s on a CPU and
needs little RAM. The sampled rows are seeded per tensor name, so the output is deterministic. These files
were produced with Python 3.14, gguf 0.19.0 and numpy 2.5.4; the same script run with llama.cpp's in-tree
gguf-py 0.17.0 gives identical output.

### Qwen3-32B vs Qwen3-14B (`qwen3_14b_parent.py`)

```
python3 qwen3_14b_parent.py Qwen3-32B-Q8_0.gguf --compare results/native/qwen3_14b_parent_bf16.npz
```

Writes `results/qwen3_14b/qwen3_14b_parent.txt` and `.npz`. It fetches about 1.4 GB of Qwen3-14B tensors
(BF16) with HTTP range requests; the fetched tensors are cached as float32 `.npy` files (about 2.6 GB) in a
temporary directory that is deleted at the end (`--cache DIR` keeps them). It took about 2.5 minutes with a
peak resident memory of 6.6 GB. `--compare` prints the difference from a reference run; the reference in
`results/native/` was made by an earlier version of the script that read the 32B side from the native BF16
safetensors instead of the GGUF.

### Follow-up tests on the GGUFs (`weight_tests/`)

The scripts find the GGUFs under their Hugging Face file names in the directory `GGUF_DIR` (default: the
current directory); `GGUF_32B`, `GGUF_27B` and `GGUF_MOE` override single paths. Output goes to
`results/weight_tests/` (override with `TWIN_OUT`). Each script's docstring describes the test in detail.

```
export GGUF_DIR=/path/to/ggufs
python3 weight_tests/s0_fresh_rows.py 32b        # also 27b, moe
python3 weight_tests/s2_neurons.py 32b           # also 27b, moe; writes per-neuron .npz used below
python3 weight_tests/s2_moe_experts.py
python3 weight_tests/s2_dead_units.py moe        # also 32b, 27b (no dead units, so no CSV)
python3 weight_tests/s2_followups.py             # needs s2_neurons.py output for 32b and 27b
python3 weight_tests/s2_width.py                 # needs s2_followups.py output
python3 weight_tests/s3_norms.py 32b             # also 27b, moe
python3 weight_tests/s4_spectral.py
python3 weight_tests/s5_router.py
python3 weight_tests/s6_triplets.py
```

All run on a CPU, deterministically (fixed seeds). Run time and peak resident memory on a 16-core Apple-silicon
machine: s2_neurons 5.5 min (32b, 7.3 GB) and 8.5 min (27b, 5.1 GB), a few seconds for moe; s4_spectral
5 min (4.4 GB); s2_width 3 min (3.2 GB); s2_moe_experts and s2_followups under 2 min (8.2 and 6.2 GB);
s0, s6 and s2_dead_units about 1 min per model (up to 5.8 GB); s3 and s5 under 10 s. The per-neuron `.npz`
arrays (up to 37 MB each) are intermediate files and are not committed; per-pair CSVs larger than 400 kB are
committed gzip-compressed (`*.csv.gz`). Every script in `weight_tests/` was re-run from this repository
(Python 3.14, gguf 0.19.0, numpy 2.5.4) into a separate directory, for all models it supports, and
reproduced the committed logs and CSVs byte for byte apart from the elapsed-time fields in the logs.

### Native-weight and model tests (`native/`)

These need the native BF16 checkpoint, which is not kept alongside this repository:

```
hf download Qwen/Qwen3-32B --revision 9216db5781bf21249d130ec9da846c4624c16137 --local-dir Qwen3-32B
python3 native/native_replication.py Qwen3-32B Qwen3-32B-Q8_0.gguf results/native
python3 native/functional_2x2.py Qwen3-32B results/native WEB.txt MIXED.txt CODE.txt
python3 native/gain_reset_gradients.py Qwen3-32B results/native WEB.txt MIXED.txt CODE.txt
```

`native_replication.py` runs on a CPU and reads the tensors memory-mapped. The two model scripts run the
full BF16 model on a GPU with enough memory for it (they were run on Apple-silicon MPS with 128 GB of
unified memory; the device is set in the script) with torch 2.13.0 and transformers 5.20.2.
`gain_reset_gradients.py` needs backward passes and uses 512-token sequences, because 1,024-token sequences
pushed a 128 GB machine into swap. WEB.txt, MIXED.txt and CODE.txt are text files of documents separated by
the ASCII record separator (0x1E): English web text from FineWeb (sample-10BT), a mixed sample (FineWeb-2 in
ten languages, English web and Wikipedia, GitHub code, open-web-math, UltraChat), and GitHub code
(codeparrot/github-code-clean). The scripts read documents from fixed offsets into these files; the exact
files used are not included, so a re-run with other text will give somewhat different numbers. The saved
outputs are in `results/native/`; these scripts were not re-run for this version of the repository.

### Loop hypothesis and family tree (`loop_hypothesis/`)

```
python3 loop_hypothesis/star_decomposition.py Qwen3-32B-Q8_0.gguf results/native/qwen3_14b_parent_bf16.npz \
    results/loop_hypothesis/star_decomposition.csv > results/loop_hypothesis/star_decomposition.txt
python3 loop_hypothesis/phylogeny.py Qwen3-14B Qwen3-32B-Q8_0.gguf results/loop_hypothesis
python3 loop_hypothesis/tree_fit.py results/loop_hypothesis/phylogeny.json > results/loop_hypothesis/tree_fit.txt
python3 loop_hypothesis/drift_profile.py Qwen3-14B Qwen3-32B-Q8_0.gguf results/loop_hypothesis
python3 loop_hypothesis/loop_tolerance.py Qwen3-14B results/loop_hypothesis WEB.txt MIXED.txt CODE.txt
```

`Qwen3-14B` is the BF16 checkpoint (`hf download Qwen/Qwen3-14B --revision
40c069824f4251a91eefaf281ebe4c544efd3e18 --local-dir Qwen3-14B`, 30 GB). `tree_fit.py` needs `scipy`.
`phylogeny.py` (70 s, 2.8 GB peak) and `drift_profile.py` (95 s, 2.4 GB) read only the rows they need;
`phylogeny.py` is a faster reimplementation of the memory-mapped script that produced the committed
`phylogeny.txt`, and reproduces it byte for byte. `loop_tolerance.py` runs the BF16 14B on a GPU (Apple MPS
by default, `TORCH_DEVICE` to change; torch 2.13.0, transformers 5.20.2), reading documents from fixed offsets
of text files as described for `native/` below; it was not re-run for this repository.

### The Qwen3 family (`family/`)

The scripts read local BF16 checkpoints from `QWEN_HF_DIR` (subdirectories `Qwen3-0.6B`, `Qwen3-1.7B`,
`Qwen3-4B`, `Qwen3-14B`; default the current directory) and the Qwen3-32B GGUF from `GGUF_32B` (default
`$GGUF_DIR/Qwen3-32B-Q8_0.gguf`). Qwen3-8B, Qwen3-30B-A3B and Qwen3-235B-A22B are not downloaded: the
tensors needed (about 4 GB as float32 in total) are fetched with HTTP range requests and cached in
`FAMILY_CACHE` (default `family/cache/`, not committed); the scripts that use them run offline once the cache
is filled. Revisions:

| Model | Source | Revision |
|---|---|---|
| Qwen3-0.6B | local BF16 | c1899de289a04d12100db370d81485cdf75e47ca |
| Qwen3-1.7B | local BF16 | 70d244cc86ccca08cf5af4e1e306ecf908b1ad5e |
| Qwen3-4B | local BF16 | 1cfa9a7208912126459214e8b04321603b3df60c |
| Qwen3-14B | local BF16 | 40c069824f4251a91eefaf281ebe4c544efd3e18 |
| Qwen3-32B | Q8_0 GGUF (above) | |
| Qwen3-8B | range requests, `main` | b968826d9c46dd6066d109eabc6255188de91218 |
| Qwen3-30B-A3B | range requests, `main` | ad44e777bcd18fa416d9da3bd8f70d33ebb85d39 |
| Qwen3-235B-A22B | range requests, `main` | 8efa61729e24bd65b1d152b5ab5409052aa80e65 |

```
export QWEN_HF_DIR=/path/to/checkpoints GGUF_32B=/path/to/Qwen3-32B-Q8_0.gguf
H=$QWEN_HF_DIR; R=results/family
for m in Qwen3-0.6B Qwen3-1.7B Qwen3-4B; do
  python3 family/analyze_small.py twins $H/$m > $R/twins_$m.txt
  python3 family/analyze_small.py width $H/$m > $R/width_$m.txt
done
python3 family/analyze_small.py embed 0.6B=$H/Qwen3-0.6B 1.7B=$H/Qwen3-1.7B 4B=$H/Qwen3-4B \
    8B=hf:Qwen/Qwen3-8B 14B=$H/Qwen3-14B 32B=$GGUF_32B > $R/embed_pairs.txt
python3 family/qknorm_screen.py > $R/qknorm_screen.txt
python3 family/qknorm_within.py > $R/qknorm_within.txt            # reads the gains cached by qknorm_screen.py
python3 family/moe_matrix_twins.py Qwen/Qwen3-30B-A3B Qwen/Qwen3-235B-A22B > $R/moe_matrix_twins.txt
python3 family/moe_experts.py > $R/moe_experts.txt
python3 family/pair_06_17.py > $R/pair_06_17.txt
python3 family/sortedness.py > $R/sortedness.txt
python3 family/pair_4_8.py > $R/pair_4_8.txt                      # reuses the embedding slices cached by `embed`
python3 family/weight_map.py > $R/weight_map.txt
python3 family/embed_origin.py > $R/embed_origin.txt
```

Each of these takes seconds to a few minutes on a CPU once the tensors are local or cached (at most 13 GB
resident, for `sortedness.py`); `pair_4_8.py`, `sortedness.py` and `pair_06_17.py` need `scipy`. All of them were
re-run from this repository (Python 3.14.6, numpy 2.5.4, gguf 0.19.0, scipy 1.18.1) against a copy of the
cache of range-fetched tensors, with the cached q/k-norm gains and embedding slices of the local models deleted
so that those were recomputed from the checkpoints. Every output reproduced the committed one byte for byte,
except that `moe_matrix_twins.py` prints its "fetched ..." lines only when it fetches (the committed file comes
from the cached run and adds the twin vs non-twin summary at the end of each model).

**Activations** (`acts_14_32.py`; not re-run for this repository). The committed numbers come from a 32B
residual-stream dump made with llama.cpp from the Q8_0 GGUF by a private tool that is not included. The same
`.bin` / `.idx` format (documented in `acts_14_32.py`) can be produced from the BF16 checkpoint with
`dump_residuals_hf.py`:

```
python3 family/acts_texts.py WEB.txt MIXED.txt texts.txt          # the 36 texts (documents at fixed offsets)
hf download Qwen/Qwen3-32B --revision 9216db5781bf21249d130ec9da846c4624c16137 --local-dir Qwen3-32B
python3 family/dump_residuals_hf.py Qwen3-32B texts.txt q32       # 6.7 GB dump; ~65 GB model in memory
python3 family/acts_14_32.py Qwen3-14B q32 results/family/acts_14_32 > results/family/acts_14_32.txt
```

Both model scripts use a GPU (MPS by default, `TORCH_DEVICE` to change). `dump_residuals_hf.py` was checked
only on Qwen3-0.6B on a CPU: it writes the expected format, its hooked tensors equal the model's hidden states
after each layer, and on the 36 texts the transformers tokenizer gives exactly the token ids (10,179) that
llama.cpp gave. A BF16 dump will differ slightly from the Q8_0 one used for the committed numbers.

## Files

- `twin_layers.py`: twin similarity within one GGUF (fixed-offset and scan modes).
- `qwen3_14b_parent.py`: Qwen3-14B (fetched from Hugging Face) vs Qwen3-32B (GGUF), layers and query heads.
- `gallery_twins.py`: twin pairs among Goodfire's gallery neurons.
- `weight_tests/`: follow-up tests on the GGUFs, with shared helpers in `common.py`.
- `native/`: tests on the native BF16 Qwen3-32B (weights, activations, gradients).
- `loop_hypothesis/`: the loop-hypothesis tests and the 14B/32B family tree, with the two pre-registration
  notes `predictions.md` and `phylogeny_predictions.md` (kept as written before the runs).
- `family/`: the Qwen3 family map (twin scans, cross-model maps, sortedness, MoE checks, activations);
  `paths.py` holds the configurable locations and `hf_remote.py` the range-request reader.
- `results/`: outputs of `twin_layers.py` and `gallery_twins.py` (each file starts with its command);
  `results/qwen3_14b/`, `results/weight_tests/`, `results/native/`, `results/loop_hypothesis/`,
  `results/family/`: outputs of the other scripts (`results/native/checkpoint_revision.txt` records the BF16
  revision).

## Corrections to earlier versions

The first version of this README (commit 8daa5d9) contained these errors or overstatements:

- It said that centring over n layers makes the correlations over all pairs average about −1/(n − 1). That
  holds only for the norm-weighted mean; the correct statement is under [norm gains](#norm-gains).
- It said that the Qwen3.8-27B similarities fit two successive stackings and also a single triple copy whose
  copies diverged unevenly. The covariance test shows that neither simple history fits the matrices without
  copy-specific shrinkage ([triple covariances](#qwen38-27b-triple-covariances)).
- It said that row j of every weight matrix in a twin layer resembles row j of its partner. That holds for
  most neurons and experts but unevenly ([per-neuron correspondence](#per-neuron-and-per-expert-correspondence)),
  and the sampled `ffn_down` rows were 64 residual-stream dimensions, not neurons.
- It reported the raw router similarity (0.79 vs 0.29); the routing-relevant, expert-centred figure is 0.74
  vs 0.000.
- It said that in Qwen3-32B and Qwen3.6-35B-A3B the twin pairs are tens to hundreds of standard deviations
  above the other pairs for every attention, linear-attention, FFN and expert matrix. With the SD taken
  across non-twin pairs, that holds for the large matrices (21–450 SDs) but not for the small
  linear-attention tensors (`ssm_alpha`, `ssm_beta`, `ssm_conv1d`: 8–16 in Qwen3.6-35B-A3B) or the raw
  router (4); and that SD is the spread of the non-twin pairs, not sampling error.
- It described the parent models as hypothetical with no released checkpoint identified; for Qwen3-32B the
  released Qwen3-14B lines up with the implied parent.

## License

MIT; see `LICENSE`.

