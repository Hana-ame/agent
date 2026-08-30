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


def _random_int(
    rng: random.Random,
    min_digits: int = 1,
    max_digits: int = 6,
    sparse_from: int = 3,
    density: float = 0.5,
) -> int:
    """Uniform integer with `min_digits..max_digits` decimal digits (no leading zeros).

    Digit *length* is weighted: 1-2 digit operands keep weight 1 (they fully
    cover the 0..99 space), and length d >= `sparse_from` gets weight
    `density ** (d - sparse_from + 1)` — so 3-digit is density, 4-digit is
    density^2, ... i.e. **progressively sparser**: we no longer enumerate every
    combination of longer operands, just sample a shrinking fraction of them.
    """
    weights = []
    for d in range(min_digits, max_digits + 1):
        if d < sparse_from:
            weights.append(1.0)
        else:
            weights.append(density ** (d - sparse_from + 1))
    n_digits = rng.choices(range(min_digits, max_digits + 1), weights=weights, k=1)[0]
    if n_digits == 1:
        return rng.randint(0, 9)
    return rng.randint(10 ** (n_digits - 1), 10 ** n_digits - 1)


def _random_pad(rng: random.Random, max_spaces: int = 3) -> int:
    """Number of spaces around an operator — uniform over 0..max_spaces."""
    return rng.randint(0, max_spaces)


def gen_expression(
    rng: random.Random,
    min_digits: int = DEFAULT_MIN_DIGITS,
    max_digits: int = DEFAULT_MAX_DIGITS,
    max_spaces: int = DEFAULT_MAX_SPACES,
    sparse_from: int = 3,
    density: float = 0.5,
) -> List[int]:
    """One arithmetic expression, as a list of token ids.

    Returns: [<BOS>, a..., spaces, op, spaces, b..., spaces, =, spaces, c..., <EOS>]
    Spacing around each operator is independently random in 0..max_spaces.
    `sparse_from`/`density`: digit lengths >= sparse_from are sampled at a
    shrinking rate (see `_random_int`), so longer operands do not enumerate
    the full combination space.
    """
    a = _random_int(rng, min_digits=min_digits, max_digits=max_digits,
                    sparse_from=sparse_from, density=density)
    b = _random_int(rng, min_digits=min_digits, max_digits=max_digits,
                    sparse_from=sparse_from, density=density)

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


def _col_tokens(parts: str) -> List[int]:
    """Tokenize a column string like '4+6+0=10' into [d,+,d,+,d,=,s...]."""
    out: List[int] = []
    for ch in parts:
        out.append(TOK_TO_ID[ch])
    return out


def gen_expression_cot(
    rng: random.Random,
    min_digits: int = DEFAULT_MIN_DIGITS,
    max_digits: int = DEFAULT_MAX_DIGITS,
    four_digit_bias: float = 0.0,
    sparse_from: int = 3,
    density: float = 0.5,
) -> List[int]:
    """Chain-of-thought arithmetic expression: problem + column-wise carries + answer.

    Format (uses only the 16 tokens; spaces separate columns):
      <BOS> a op b = d1 op d2 op carry = sum carry  d1 op d2 op carry = sum carry ... c <EOS>

    Columns run least-significant digit first; each is `da op db op carry = sum newcarry`
    where sum is the full (possibly 2-digit) sum/diff and newcarry the next carry/borrow.
    The trailing `c` is the actual answer. This decomposes multi-digit arithmetic into
    single-column operations the tiny model can actually learn.

    `four_digit_bias` (0..1): with this probability both operands are forced to
    `max_digits` digits — this oversamples the hardest carry/overflow cases
    (e.g. 4-digit + 4-digit with 5-digit results), which uniform digit-length
    sampling only produces ~6% of the time.

    `sparse_from`/`density`: as in `gen_expression`, operands with digit length
    >= sparse_from are sampled at a decaying rate (progressively sparse), so the
    datasource does not enumerate every long-operand combination.
    """
    if rng.random() < four_digit_bias:
        a = _random_int(rng, min_digits=max_digits, max_digits=max_digits,
                        sparse_from=sparse_from, density=density)
        b = _random_int(rng, min_digits=max_digits, max_digits=max_digits,
                        sparse_from=sparse_from, density=density)
    else:
        a = _random_int(rng, min_digits=min_digits, max_digits=max_digits,
                        sparse_from=sparse_from, density=density)
        b = _random_int(rng, min_digits=min_digits, max_digits=max_digits,
                        sparse_from=sparse_from, density=density)
    cols: List[Tuple[str, str]] = []  # (column_str, carry_after_this_column)
    if rng.random() < 0.5:
        op, opch = PLUS, "+"
        c = a + b
        carry = 0
        n = max(len(str(a)), len(str(b))) + 1   # include final carry position
        sa, sb = str(a), str(b)
        for i in range(n):
            da = int(sa[-1 - i]) if i < len(sa) else 0
            db = int(sb[-1 - i]) if i < len(sb) else 0
            s = da + db + carry
            cols.append((f"{da}+{db}+{carry}={s}", str(s // 10)))
            carry = s // 10
    else:
        op, opch = MINUS, "-"
        if a < b:
            a, b = b, a
        c = a - b
        borrow = 0
        n = len(str(a))
        sa, sb = str(a), str(b)
        for i in range(n):
            da = int(sa[-1 - i])
            db = int(sb[-1 - i]) if i < len(sb) else 0
            d = da - db - borrow
            if d < 0:
                d += 10
                nb = 1
            else:
                nb = 0
            cols.append((f"{da}-{db}-{borrow}={d}", str(nb)))
            borrow = nb

    tokens: List[int] = [BOS]
    tokens += _int_to_tokens(a)
    tokens += [SP, op, SP]
    tokens += _int_to_tokens(b)
    tokens += [SP, EQ, SP]
    for i, (col, after) in enumerate(cols):
        if i:
            tokens.append(SP)
        tokens += _col_tokens(col)
        tokens.append(SP)
        tokens.append(TOK_TO_ID[after])
    tokens.append(SP)
    tokens += _int_to_tokens(c)
    tokens.append(EOS)
    return tokens


def make_single_cot_batch(
    rng: random.Random,
    block_size: int,
    batch_size: int,
    device: str = "cpu",
    min_digits: int = DEFAULT_MIN_DIGITS,
    max_digits: int = DEFAULT_MAX_DIGITS,
    four_digit_bias: float = 0.0,
    sparse_from: int = 3,
    density: float = 0.5,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Batch of single CoT expressions, left-aligned + EOS padding.

    Rows are padded to a width that fits the longest possible CoT expression
    for this generator (not block_size), so the loss is not diluted by a huge
    EOS tail. Width estimate: problem part + (n+1) columns * ~10 tokens + answer.
    """
    # Per column: da+db+carry=sum (up to 9+9+1=19 -> 8 chars) + space + carry = ~10 tokens
    n_cols = max_digits + 1
    width = min(block_size, 2 + 4 + 3 + 4 + 3 + n_cols * 10 + max_digits + 2)
    if width < 8:
        raise ValueError("block_size too small for CoT mode")

    inp_rows: List[torch.Tensor] = []
    tgt_rows: List[torch.Tensor] = []
    for _ in range(batch_size):
        expr = gen_expression_cot(rng, min_digits, max_digits, four_digit_bias,
                                  sparse_from, density)
        if len(expr) > width:          # guard: never truncate
            expr = expr[:width - 1] + [EOS]
        pad = [EOS] * (width - len(expr))
        full = expr + pad
        inputs = torch.tensor(full[:-1], dtype=torch.long, device=device)
        targets = torch.tensor(full[1:], dtype=torch.long, device=device)
        inp_rows.append(inputs)
        tgt_rows.append(targets)
    return torch.stack(inp_rows), torch.stack(tgt_rows)


def pack_blocks(
    rng: random.Random,
    block_size: int,
    batch_size: int,
    device: str = "cpu",
    min_digits: int = DEFAULT_MIN_DIGITS,
    max_digits: int = DEFAULT_MAX_DIGITS,
    max_spaces: int = DEFAULT_MAX_SPACES,
    sparse_from: int = 3,
    density: float = 0.5,
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
        expr = gen_expression(rng, min_digits, max_digits, max_spaces,
                              sparse_from, density)
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
    sparse_from: int = 3,
    density: float = 0.5,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """A full (inputs, targets) batch, each row of shape (block_size,)."""
    inp_rows: List[torch.Tensor] = []
    tgt_rows: List[torch.Tensor] = []
    for _ in range(batch_size):
        i, t = pack_blocks(rng, block_size, 1, device, min_digits, max_digits, max_spaces,
                           sparse_from, density)
        inp_rows.append(i)
        tgt_rows.append(t)
    return torch.stack(inp_rows), torch.stack(tgt_rows)


def make_single_batch(
    rng: random.Random,
    block_size: int,
    batch_size: int,
    device: str = "cpu",
    min_digits: int = DEFAULT_MIN_DIGITS,
    max_digits: int = DEFAULT_MAX_DIGITS,
    max_spaces: int = DEFAULT_MAX_SPACES,
    sparse_from: int = 3,
    density: float = 0.5,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Batch of single expressions (no packing), left-aligned + EOS padding.

    Each row is ONE expression so the model attends only to that expression's
    tokens — a much stronger learning signal for the arithmetic itself, and
    ~15x fewer tokens per step than packed blocks. Rows are padded to the max
    possible expression length for this generator (not block_size), so the
    loss is not diluted by a huge EOS tail.
    """
    # Max token length of an expression: BOS + a + (sp op sp) + b + (sp = sp) + c + EOS
    max_spaces_tok = 1 + max_spaces + 1  # space*L + op + space*R
    width = min(block_size, 2 + 2 * max_digits + (max_digits + 1) + 2 * max_spaces_tok)
    if width < 8:
        raise ValueError("block_size too small for single-expression mode")

    inp_rows: List[torch.Tensor] = []
    tgt_rows: List[torch.Tensor] = []
    for _ in range(batch_size):
        expr = gen_expression(rng, min_digits, max_digits, max_spaces,
                              sparse_from, density)
        if len(expr) > width:          # guard: never truncate
            expr = expr[:width - 1] + [EOS]
        pad = [EOS] * (width - len(expr))
        full = expr + pad
        inputs = torch.tensor(full[:-1], dtype=torch.long, device=device)
        targets = torch.tensor(full[1:], dtype=torch.long, device=device)
        inp_rows.append(inputs)
        tgt_rows.append(targets)
    return torch.stack(inp_rows), torch.stack(tgt_rows)


def stream_batches(
    block_size: int,
    batch_size: int,
    device: str = "cpu",
    seed: int = 0,
    min_digits: int = DEFAULT_MIN_DIGITS,
    max_digits: int = DEFAULT_MAX_DIGITS,
    max_spaces: int = DEFAULT_MAX_SPACES,
    sparse_from: int = 3,
    density: float = 0.5,
) -> Iterator[Tuple[torch.Tensor, torch.Tensor]]:
    """Endless stream of (inputs, targets) batches."""
    rng = random.Random(seed)
    while True:
        yield make_batch(rng, block_size, batch_size, device, min_digits, max_digits, max_spaces,
                         sparse_from, density)


def decode(tokens: List[int]) -> str:
    """Decode a list of token ids back to a readable expression."""
    id_to_tok = {v: k for k, v in TOK_TO_ID.items()}
    return "".join(id_to_tok[t] for t in tokens)
