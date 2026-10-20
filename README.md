# Same-index twin layers in Qwen models

Three Qwen checkpoints contain blocks of layers whose weights line up index by index. For a layer i in the
block and its partner i + d, row j of every weight matrix in layer i resembles row j of the same matrix in
layer i + d (for MoE layers, neuron j of expert e resembles neuron j of expert e), and the per-channel norm
gains of the two layers are strongly correlated. Other pairs of layers show no such alignment. We call
such layers same-index twins.

| Model | Layers | Twin blocks | Offset | Layers without a twin |
|---|---|---|---|---|
| Qwen3-32B (dense) | 64 | 8–31 ↔ 32–55 | 24 | 0–7, 56–63 |
| Qwen3.8-27B (dense FFN, hybrid attention) | 64 + 1 MTP block | 8–23 ↔ 24–39 ↔ 40–55 | 16, 32 | 0–7, 56–63, MTP block 64 |
| Qwen3.6-35B-A3B (MoE, 256 experts, hybrid attention) | 40 | 8–19 ↔ 20–31 | 12 | 0–7, 32–39 |

"Hybrid attention" means full attention in every fourth layer and linear attention (DeltaNet-style;
`attn_qkv`, `attn_gate`, `ssm_*` tensors) in the other three. All three offsets are multiples of 4, so
twins are always the same kind of layer.

## Method

`twin_layers.py` reads a GGUF file and, for every per-layer tensor type, compares each pair of layers.

- **Matrices.** Signed cosine similarity between the flattened tensors of two layers. The script uses a
  fixed random sample of rows (the same rows in every layer: 64 rows if a row has more than 6000 entries,
  otherwise 256, or all rows if there are fewer) and dequantises them from Q8_0. For 3-D expert tensors a
  row is one (expert, neuron), so a match needs both indices to agree. The router (`ffn_gate_inp`) is
  treated as a matrix of per-expert rows.
- **Norm gains.** Pearson correlation of the gain vectors, raw and "centred" (after subtracting each
  channel's mean across layers, which removes the shape shared by all layers).
- **Fixed-offset mode** (`MODEL OFFSET FIRST LAST`) compares the twin pairs (i, i + OFFSET) for
  FIRST ≤ i ≤ LAST with all other pairs of layers that contain the tensor type. ± is the standard
  deviation across layer pairs. It also prints off-by-one pairs (i, i + OFFSET ± 1) and adjacent layers.
- **Scan mode** (`MODEL scan`) assumes no offset: for each tensor type it prints the mean similarity at
  every layer offset (the four highest are shown) and, for the mean over the tensor types present in every
  layer, each layer's most similar partner.

The results below come from the Q8_0 GGUFs listed under [Reproducing](#reproducing).

## Results

Numbers are copied from the files in `results/`. Matrix rows are signed cosine; norm rows are centred
Pearson r. "All matrices" is the mean over the tensor types present in every layer. Full-attention tensors
exist only in every fourth layer, so in the hybrid models they have fewer twin pairs than the other tensors.

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
layer (pooled cosine 0.15–0.31); for layers 0–7 and 56–63 the most similar layer is at 0.00–0.01.

### Qwen3.8-27B: three blocks 8–23, 24–39, 40–55

Twin pairs, mean ± SD (full attention: 4 pairs; linear attention: 12 pairs; FFN: 16 pairs).

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

In this model the "all other pairs" baseline of each fixed-offset run still contains the other two twin
relations, which widens it (for the matrices in the table, means 0.001–0.020 and SDs up to 0.080; see
`results/`). The scan, which averages each offset separately, finds no offset other than +16 and +32 with
a mean above 0.007 for any matrix in the table. By the pooled FFN cosine and the norm gains, the closest
pair of blocks is 24–39 ↔ 40–55 and the most distant is 8–23 ↔ 40–55.

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
| ffn_gate_inp (router) | 0.791 ± 0.028 | 0.286 ± 0.125 |
| all matrices (MoE + router) | 0.311 ± 0.020 | 0.042 ± 0.018 |
| attn_q_norm (centred r) | 0.807 ± 0.060 | −0.168 ± 0.233 |
| attn_k_norm (centred r) | 0.823 ± 0.034 | −0.171 ± 0.236 |
| ssm_norm (centred r) | 0.809 ± 0.105 | −0.051 ± 0.090 |

A routed-expert match requires the expert index and the neuron index to agree, so the experts kept their
positions. The router rows of twin layers correlate at 0.79 against 0.29 for other pairs: expert e at
layer i and expert e at layer i + 12 are routed alike. Centring over n layers makes the correlations over
all pairs average about −1/(n−1). Only 10 layers have full attention, and the twin pairs take a large
positive share, so the other q/k-norm pairs average about −0.17 here (Qwen3-32B, 64 layers: about −0.02).

The small linear-attention tensors (`ssm_conv1d`, `ssm_alpha`, `ssm_beta`) show the same pattern in both
hybrid models; see `results/`.

## What the weights show, and what they do not

**Established by the weights: the twins share an origin.** The order of neurons, heads and experts within
a layer is arbitrary: permuting them consistently across the layer's matrices gives an equivalent network.
Layers initialised independently therefore have no reason to line up index by index, and non-twin pairs
sit where unrelated vectors do (Qwen3-32B: 0.000 ± 0.001 for six of the seven matrix types, 0.004 ± 0.003
for `ffn_gate`). In Qwen3-32B and Qwen3.6-35B-A3B the twin pairs are tens to hundreds of standard
deviations above the other pairs for every attention, linear-attention, FFN and expert matrix; in
Qwen3.8-27B no offset other than +16 and +32 averages above 0.007 for these matrices. Each set of twins
therefore descends from common weights.

**Inferred: the shared origin was most likely a trained layer.** Norm gains start at 1 under standard
random initialisation, so a shared random start gives twins no shared gain pattern to begin with. Yet
centred twin q/k-norm gains correlate at about 0.8–0.9 (Qwen3-32B 0.88 / 0.93, Qwen3.6-35B-A3B
0.81 / 0.82, Qwen3.8-27B 24–39 ↔ 40–55 0.84 / 0.87), more strongly than the twin matrices do. This
fits depth up-scaling as in SOLAR (arXiv:2312.15166): a trained model's layers are stacked with an overlap
and the result is trained further. The twin positions match that recipe:

- Qwen3-32B = layers 0–31 followed by layers 8–39 of a 40-layer parent (parent layer k, 8 ≤ k ≤ 31,
  appears at positions k and k + 24).
- Qwen3.6-35B-A3B = layers 0–19 followed by layers 8–27 of a 28-layer parent (parent layer k, 8 ≤ k ≤ 19,
  appears at positions k and k + 12).
- Qwen3.8-27B: a 16-layer block appears three times. The uneven similarities (24–39 ↔ 40–55 closest,
  8–23 ↔ 40–55 furthest) fit two successive stackings and also a single triple copy whose copies diverged
  unevenly.

This is an inference, not an observation. The weights do not exclude other histories: a block whose
weights were tied early in training, or identical random initialisation of the blocks (for example one
seed per pipeline stage) followed by training from scratch, during which the aligned matrices could
themselves have driven correlated gains. The parent models described above are hypothetical; nothing
here identifies a released checkpoint as the parent.

**Not recoverable from the weights: which twin is the original.** A stacking copies one parent layer into
both positions, and both copies have since been trained. Neither twin is privileged.

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
so the chance baseline is indicative only.

## Reproducing

Requirements: Python 3 with `numpy` and `gguf` (the GGUF reader from llama.cpp, published on PyPI).
`gallery_twins.py` uses only the standard library.

```
pip install gguf numpy
```

Models (Q8_0 GGUF from Hugging Face; SHA-256 of the files used):

| Model | Hugging Face repo | File | Size | SHA-256 |
|---|---|---|---|---|
| Qwen3-32B | Qwen/Qwen3-32B-GGUF | Qwen3-32B-Q8_0.gguf | 34.8 GB | 2c50eb8aad05047dbf24fa014eb621adf552e14176cabe0c5db4ef38c91e2169 |
| Qwen3.8-27B | lmstudio-community/Qwen3.8-27B-GGUF | Qwen3.8-27B-Q8_0.gguf | 29.0 GB | f0fc43cfd802bd87bc7b95f2120553a6407848e3229126b623cc397427857e8a |
| Qwen3.6-35B-A3B | lmstudio-community/Qwen3.6-35B-A3B-GGUF | Qwen3.6-35B-A3B-Q8_0.gguf | 36.9 GB | e56476632cf65ab1a32c135aead9eba4278f3bd3eb0499aa74304566d6e27d5b |

```
hf download Qwen/Qwen3-32B-GGUF Qwen3-32B-Q8_0.gguf --local-dir .
hf download lmstudio-community/Qwen3.8-27B-GGUF Qwen3.8-27B-Q8_0.gguf --local-dir .
hf download lmstudio-community/Qwen3.6-35B-A3B-GGUF Qwen3.6-35B-A3B-Q8_0.gguf --local-dir .
```

Commands that produce the files in `results/`:

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
needs little RAM. The sampled rows are seeded per tensor name, so the output is deterministic. The files in
`results/` were produced with Python 3.14, gguf 0.19.0 and numpy 2.5.4; the same script run with
llama.cpp's in-tree gguf-py 0.17.0 gives identical output.

## Files

- `twin_layers.py`: the analysis script.
- `gallery_twins.py`: twin pairs among Goodfire's gallery neurons.
- `results/`: output of every command above. Each file starts with the command that produced it.

## License

MIT; see `LICENSE`.
