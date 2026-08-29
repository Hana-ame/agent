"""additive_rand_transformer — a tiny CPU GPT trained on additive-arithmetic expressions."""

from .model import TinyGPT, TinyGPTConfig, VOCAB_SIZE, TOKENS, TOK_TO_ID
from .data import gen_expression, pack_blocks, stream_batches, decode

__all__ = [
    "TinyGPT",
    "TinyGPTConfig",
    "VOCAB_SIZE",
    "TOKENS",
    "TOK_TO_ID",
    "gen_expression",
    "pack_blocks",
    "stream_batches",
    "decode",
]
