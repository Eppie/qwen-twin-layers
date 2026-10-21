#!/usr/bin/env python3
"""Build the text file used for the Qwen3-14B / Qwen3-32B activation comparison (acts_14_32.py).

    python3 family/acts_texts.py WEB_TXT MIXED_TXT OUT_TXT

WEB_TXT, MIXED_TXT: text files of documents separated by the ASCII record separator 0x1E (English web text; mixed
multilingual web / English web / Wikipedia / code / math / chat), the same kind of files as in native/. Takes the
non-empty documents 1100-1123 of WEB_TXT and 3100-3111 of MIXED_TXT, cuts each to its first 1,500 characters, and
writes the 36 texts to OUT_TXT separated by 0x1E. With the files used for the published run this gave 10,179
tokens; those files are not included, so other text gives other (similar, if the claim holds) numbers.
"""
import sys

WEB, MIXED, OUT = sys.argv[1:4]


def docs(path):
    return [x for x in open(path, encoding='utf-8', errors='replace').read().split('\x1e') if x.strip()]


texts = [x[:1500] for x in docs(WEB)[1100:1124] + docs(MIXED)[3100:3112]]
with open(OUT, 'w', encoding='utf-8') as f:
    f.write('\x1e'.join(texts))
print(f'{len(texts)} texts written to {OUT}')
