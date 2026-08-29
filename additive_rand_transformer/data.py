"""Dynamic arithmetic-expression dataset (additive random generation).

Task: predict the next token of an expression like

    <BOS> 1234 + 5678 = 6912 <EOS>

Rules enforced by the generator:
  * operands are 1..max_digits digit non-negative integers
    (1-4 digits are always covered; longer operands are also allowed)
  * NO leading zeros (except for the number 0 itself)
  * for "+": result = a + b  (may be longer than either operand)
  * for "-": a >= b so the result is non-negative
  * spacing is RANDOM — 0..max_spaces spaces around each operator, so both
    "1234+5678=6912" and "1234  +  5678 = 6912" are valid generator output
  * each expression is wrapped with <BOS> ... <EOS>
  * multiple expressions are packed into one block of length `block_size`
    (1024 by default) so the 1024-context model is fully exercised

Because the dataset is generated on the fly from a deterministic RNG, any
expression can be re-sampled and judged: an expression that matches this
distribution gets high log-likelihood, one that violates a generator rule
(wrong result, leading zero, negative result, wrong operator, ...) gets low.
"""

from __future__ import annotations

import random
from typing import Iterator, List, Tuple

import torch

from .model import TOK_TO_ID, VOCAB_SIZE

BOS = TOK_TO_ID["<BOS>"]
EOS = TOK_TO_ID["<EOS>"]
PLUS = TOK_TO_ID["+"]
MINUS = TOK_TO_ID["-"]
EQ = TOK_TO_ID["="]
SP = TOK_TO_ID[" "]


# Generator-wide defaults. 1-4 digits are always covered (per spec); the
# generator may emit longer operands too, and the model learns that.
DEFAULT_MIN_DIGITS = 1
DEFAULT_MAX_DIGITS = 6          # raises the ceiling above 4 while keeping it tiny
DEFAULT_MAX_SPACES = 3          # 0, 1, 2, or 3 spaces around each operator


def _int_to_tokens(n: int) -> List[int]:
    """Render a non-negative integer as a list of digit token ids (no separators)."""
    if n == 0:
        return [TOK_TO_ID["0"]]
    return [TOK_TO_ID[ch] for ch in str(n)]


def _random_int(rng: random.Random, min_digits: int = 1, max_digits: int = 6) -> int:
    """Uniform integer with `min_digits..max_digits` decimal digits (no leading zeros)."""
    if min_digits == 1:
        return rng.randint(0, 10 ** max_digits - 1)
    return rng.randint(10 ** (min_digits - 1), 10 ** max_digits - 1)


def _random_pad(rng: random.Random, max_spaces: int = 3) -> int:
    """Number of spaces around an operator — uniform over 0..max_spaces."""
    return rng.randint(0, max_spaces)


def gen_expression(
    rng: random.Random,
    min_digits: int = DEFAULT_MIN_DIGITS,
    max_digits: int = DEFAULT_MAX_DIGITS,
    max_spaces: int = DEFAULT_MAX_SPACES,
) -> List[int]:
    """One arithmetic expression, as a list of token ids.

    Returns: [<BOS>, a..., spaces, op, spaces, b..., spaces, =, spaces, c..., <EOS>]
    Spacing around each operator is independently random in 0..max_spaces.
    """
    a = _random_int(rng, min_digits=min_digits, max_digits=max_digits)
    b = _random_int(rng, min_digits=min_digits, max_digits=max_digits)

    if rng.random() < 0.5:
        op = PLUS
        c = a + b
    else:
        op = MINUS
        if a < b:                 # keep the result non-negative
            a, b = b, a
        c = a - b

    tokens: List[int] = [BOS]
    tokens += _int_to_tokens(a)
    tokens += [SP] * _random_pad(rng, max_spaces) + [op] + [SP] * _random_pad(rng, max_spaces)
    tokens += _int_to_tokens(b)
    tokens += [SP] * _random_pad(rng, max_spaces) + [EQ] + [SP] * _random_pad(rng, max_spaces)
    tokens += _int_to_tokens(c)
    tokens += [EOS]
    return tokens


def pack_blocks(
    rng: random.Random,
    block_size: int,
    batch_size: int,
    device: str = "cpu",
    min_digits: int = DEFAULT_MIN_DIGITS,
    max_digits: int = DEFAULT_MAX_DIGITS,
    max_spaces: int = DEFAULT_MAX_SPACES,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Build one (inputs, targets) batch by packing expressions into block_size blocks.

    Each block is a concatenation of complete <BOS>...<EOS> expressions padded to
    exactly block_size tokens. Padding is filled with EOS so the model never sees
    garbage in the right-padding region.
    """
    if block_size < 4:
        raise ValueError("block_size too small for any expression")

    inp: List[int] = []
    # Cap the number of gen_expression calls so packing never loops forever.
    for _ in range(block_size + 8):
        expr = gen_expression(rng, min_digits, max_digits, max_spaces)
        if len(inp) + len(expr) > block_size:
            break
        inp.extend(expr)
    if len(inp) < block_size:            # right-pad with EOS
        inp.extend([EOS] * (block_size - len(inp)))

    return _finalize_block(inp, device)


def _finalize_block(inp: List[int], device: str) -> Tuple[torch.Tensor, torch.Tensor]:
    inputs = torch.tensor(inp[:-1], dtype=torch.long, device=device)
    targets = torch.tensor(inp[1:], dtype=torch.long, device=device)
    return inputs, targets


def make_batch(
    rng: random.Random,
    block_size: int,
    batch_size: int,
    device: str = "cpu",
    min_digits: int = DEFAULT_MIN_DIGITS,
    max_digits: int = DEFAULT_MAX_DIGITS,
    max_spaces: int = DEFAULT_MAX_SPACES,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """A full (inputs, targets) batch, each row of shape (block_size,)."""
    inp_rows: List[torch.Tensor] = []
    tgt_rows: List[torch.Tensor] = []
    for _ in range(batch_size):
        i, t = pack_blocks(rng, block_size, 1, device, min_digits, max_digits, max_spaces)
        inp_rows.append(i)
        tgt_rows.append(t)
    return torch.stack(inp_rows), torch.stack(tgt_rows)


def stream_batches(
    block_size: int,
    batch_size: int,
    device: str = "cpu",
    seed: int = 0,
    min_digits: int = DEFAULT_MIN_DIGITS,
    max_digits: int = DEFAULT_MAX_DIGITS,
    max_spaces: int = DEFAULT_MAX_SPACES,
) -> Iterator[Tuple[torch.Tensor, torch.Tensor]]:
    """Endless stream of (inputs, targets) batches."""
    rng = random.Random(seed)
    while True:
        yield make_batch(rng, block_size, batch_size, device, min_digits, max_digits, max_spaces)


def decode(tokens: List[int]) -> str:
    """Decode a list of token ids back to a readable expression."""
    id_to_tok = {v: k for k, v in TOK_TO_ID.items()}
    return "".join(id_to_tok[t] for t in tokens)
