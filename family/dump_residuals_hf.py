#!/usr/bin/env python3
"""Dump a model's residual stream after every decoder layer, in the format acts_14_32.py reads.

    python3 family/dump_residuals_hf.py HF_MODEL_DIR TEXTS_TXT OUT_PREFIX

HF_MODEL_DIR: a transformers checkpoint, for acts_14_32.py the BF16 Qwen3-32B (about 65 GB in memory).
TEXTS_TXT: documents separated by the ASCII record separator 0x1E (acts_texts.py); documents that are only
whitespace are skipped, as are documents that tokenize to nothing. Each text is tokenized without special tokens
(Qwen3 adds no BOS) and cut at 2,048 tokens.
Writes OUT_PREFIX.idx (header "layers 0 1 ... L-1", then per text "i T tok_1 ... tok_T") and OUT_PREFIX.bin (per
text, per layer, T x hidden float16). The captured tensor is each decoder layer's output, i.e. the residual stream
after the layer and before the final norm. Runs on Apple-silicon MPS by default (TORCH_DEVICE to change).

The published acts_14_32 numbers were NOT made with this script: their 32B dump came from llama.cpp on the Q8_0
GGUF (the l_out-L tensors, same layout), with llama.cpp's tokenizer. This script is the standalone equivalent from
the BF16 weights; Q8_0 vs BF16 and the backend's arithmetic give slightly different activations, and the HF and
llama.cpp tokenizers may split rare strings differently.
"""
import os, sys
import numpy as np, torch
from transformers import AutoModelForCausalLM, AutoTokenizer

HF, TEXTS, OUT = sys.argv[1:4]
DEVICE = os.environ.get('TORCH_DEVICE', 'mps')
MAX_TOK = 2048

texts = [d for d in open(TEXTS, encoding='utf-8', errors='replace').read().split('\x1e') if d.strip(' \t\r\n')]
tok = AutoTokenizer.from_pretrained(HF)
model = AutoModelForCausalLM.from_pretrained(HF, dtype=torch.bfloat16, device_map={'': DEVICE}); model.eval()
L = len(model.model.layers)
cur = {}
hooks = [model.model.layers[l].register_forward_hook(
    lambda m, i, o, l=l: cur.__setitem__(l, (o[0] if isinstance(o, tuple) else o)[0].float().cpu().numpy()))
    for l in range(L)]
n_written = 0
with open(OUT + '.bin', 'wb') as fb, open(OUT + '.idx', 'w') as fi, torch.no_grad():
    fi.write('layers ' + ' '.join(str(l) for l in range(L)) + '\n')
    for i, text in enumerate(texts):
        ids = tok(text, add_special_tokens=False)['input_ids'][:MAX_TOK]
        if not ids: continue
        model.model(input_ids=torch.tensor([ids], device=DEVICE), use_cache=False)
        fi.write(f'{i} {len(ids)} ' + ' '.join(map(str, ids)) + '\n')
        for l in range(L):
            assert cur[l].shape == (len(ids), model.config.hidden_size)
            fb.write(cur[l].astype(np.float16).tobytes())
        n_written += 1
for h in hooks: h.remove()
print(f'wrote {OUT}.bin / .idx: {n_written} texts, {L} layers')
