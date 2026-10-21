"""Locations used by the scripts in family/; each can be overridden with an environment variable.

QWEN_HF_DIR   directory holding the local BF16 checkpoints, one subdirectory per model named as on Hugging Face
              (Qwen3-0.6B, Qwen3-1.7B, Qwen3-4B, Qwen3-14B). Default: the current directory.
GGUF_32B      path of Qwen3-32B-Q8_0.gguf. Default: $GGUF_DIR/Qwen3-32B-Q8_0.gguf, GGUF_DIR defaulting to the
              current directory (the same convention as weight_tests/).
FAMILY_CACHE  where tensors fetched with HTTP range requests, and embedding slices, are cached as .npy / .npz.
              Default: family/cache (not committed).
"""
import os

HERE = os.path.dirname(os.path.abspath(__file__))
HF_DIR = os.path.expanduser(os.environ.get('QWEN_HF_DIR', '.'))
GGUF_32B = os.path.expanduser(os.environ.get('GGUF_32B') or os.path.join(os.environ.get('GGUF_DIR', '.'), 'Qwen3-32B-Q8_0.gguf'))
CACHE = os.path.expanduser(os.environ.get('FAMILY_CACHE') or os.path.join(HERE, 'cache'))


def local(name):
    """Directory of a local checkpoint, e.g. local('Qwen3-4B')."""
    return os.path.join(HF_DIR, name)
