# Same-index twin layers in Qwen models

Three Qwen checkpoints contain blocks of layers whose weights line up index by index: for a layer i in the
block and its partner i + d, the weights attached to neuron, head or expert j in layer i resemble those of
the same j in layer i + d, and nothing comparable holds between other pairs of layers. We call such layers
same-index twins. In Qwen3-32B the alignment extends to a released model: Qwen3-32B lines up index by index
with Qwen3-14B along the map a width + depth expansion of the 14B would produce.

| Model | Layers | Twin blocks | Offset | Layers without a twin |
|---|---|---|---|---|
| Qwen3-32B (dense) | 64 | 8–31 ↔ 32–55 | 24 | 0–7, 56–63 |
| Qwen3.8-27B (dense FFN, hybrid attention) | 64 + 1 MTP block | 8–23 ↔ 24–39 ↔ 40–55 | 16, 32 | 0–7, 56–63, MTP block 64 |
| Qwen3.6-35B-A3B (MoE, 256 experts, hybrid attention) | 40 | 8–19 ↔ 20–31 | 12 | 0–7, 32–39 |

"Hybrid attention" means full attention in every fourth layer and linear attention (DeltaNet-style;
`attn_qkv`, `attn_gate`, `ssm_*` tensors) in the other three. All three offsets are multiples of 4, so
twins are always the same kind of layer.

Contents: [Qwen3-32B and Qwen3-14B](#qwen3-32b-lines-up-with-the-released-qwen3-14b) ·
[twin layers within each model](#twin-layers-within-each-model) · [follow-up tests](#follow-up-tests) ·
[what the weights show, and what they do not](#what-the-weights-show-and-what-they-do-not) ·
[Goodfire's neuron gallery](#twin-neurons-in-goodfires-neuron-gallery) · [reproducing](#reproducing) ·
[corrections to the first version](#corrections-to-the-first-version)

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
sibling trained from the same starting point. Nor do the weights say whether the width or the depth
expansion came first, or whether both happened at once. We found no Qwen publication that describes any of
this; the Qwen3 technical report (arXiv:2505.09388) lists Qwen3-32B as a 64-layer model pretrained on 36T
tokens and says nothing about its initialisation.

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

## What the weights show, and what they do not

**Established by the weights: the twins share an origin, and Qwen3-32B shares one with Qwen3-14B.** The
order of neurons, heads and experts within a layer is arbitrary: permuting them consistently across the
layer's matrices gives an equivalent network. Layers initialised independently therefore have no reason to
line up index by index, and non-twin pairs sit where unrelated vectors do (Qwen3-32B native weights, eight
control pairs: below 0.0011 in absolute value for six of the seven matrix families, 0.0025–0.0070 for
`ffn_gate`). The twin pairs are far outside that range for every attention, linear-attention projection,
FFN and expert matrix: in fresh row samples as well as the original one, in the native BF16 weights, neuron
by neuron and expert by expert, and, for the Qwen3-32B FFNs, in activations on identical inputs. The same
argument applies between models: Qwen3-14B and Qwen3-32B line up index by index exactly where a width and
depth expansion of a 14B-shaped network predicts, at cosines 0.11–0.93, and nowhere else (at most 0.003;
`gate_proj` 0.032).

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
  Qwen3-14B has that shape and lines up with it.
- Qwen3.6-35B-A3B = layers 0–19 followed by layers 8–27 of a 28-layer parent (parent layer k, 8 ≤ k ≤ 19,
  appears at positions k and k + 12). This parent is hypothetical; no released model was compared.
- Qwen3.8-27B: a 16-layer block appears three times, with partial width copies (parent FFN width 12,288).
  The parent is hypothetical. The covariances do not decide between one triple copy and two successive
  copies (see [triple covariances](#qwen38-27b-triple-covariances)).

Other histories are not excluded by the weights. A block whose weights were tied during training and later
untied produces the same endpoint pattern as a copy. Identical random initialisation of the blocks (for
example one seed per pipeline stage) followed by training is harder to reconcile with the identical
dead-unit sets, and for Qwen3-32B it would also require the 14B and the 32B to have started from the same
weights laid out along the expansion map; the weights alone still cannot rule it out.

**Not recoverable from the weights.**

- Which twin is the original. A stacking copies one parent layer into both positions, and both copies have
  since been trained. Neither twin is privileged.
- Whether the parameters were copied after training or trained while tied and then untied: a copied model
  and an untied model that reach the same parameters and optimiser state have identical subsequent
  trajectories.
- Which checkpoint was expanded into Qwen3-32B (the released Qwen3-14B, an earlier checkpoint of it, or a
  sibling), and whether width or depth grew first.
- The copying history of Qwen3.8-27B (see above).

**No public account.** We found no Qwen publication that describes this. The Qwen3 technical report
(arXiv:2505.09388) lists Qwen3-32B as a 64-layer model pretrained on 36T tokens and says nothing about how
it was initialised. A related public observation: Maciej Kilian asked on X whether Qwen3 uses depth-wise
upcycling, after seeing the same expert index highly active 12 and 24 layers apart in Qwen3-30B-A3B's
router statistics (https://x.com/kilian_maciej/status/1982612297874026731). Qwen3-30B-A3B was not
examined here.

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

Requirements: Python 3 with `numpy` and `gguf` (the GGUF reader from llama.cpp, published on PyPI) for
everything except the two model scripts in `native/`, which also need `torch` and `transformers`.
`gallery_twins.py` uses only the standard library.

```
pip install gguf numpy
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

## Files

- `twin_layers.py`: twin similarity within one GGUF (fixed-offset and scan modes).
- `qwen3_14b_parent.py`: Qwen3-14B (fetched from Hugging Face) vs Qwen3-32B (GGUF), layers and query heads.
- `gallery_twins.py`: twin pairs among Goodfire's gallery neurons.
- `weight_tests/`: follow-up tests on the GGUFs, with shared helpers in `common.py`.
- `native/`: tests on the native BF16 Qwen3-32B (weights, activations, gradients).
- `results/`: outputs of `twin_layers.py` and `gallery_twins.py` (each file starts with its command);
  `results/qwen3_14b/`, `results/weight_tests/`, `results/native/`: outputs of the other scripts
  (`results/native/checkpoint_revision.txt` records the BF16 revision).

## Corrections to the first version

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
