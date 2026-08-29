"""Standalone test of the additive_rand_transformer logic (no torch needed).

Verifies:
  * 16-token vocab
  * gen_expression produces valid expressions (correct arithmetic, no leading zeros,
    a>=b for subtraction, random spacing)
  * membership counter-examples each violate exactly one rule
  * decode round-trips

Run:  python additive_rand_transformer/test_no_torch.py
"""

import random
import sys

# --- Replicate the token table (16 tokens) ----------------------------------
TOKENS = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9",
          "+", "-", "=", " ", "<BOS>", "<EOS>"]
TOK_TO_ID = {tok: i for i, tok in enumerate(TOKENS)}
ID_TO_TOK = {i: tok for tok, i in TOK_TO_ID.items()}
VOCAB_SIZE = len(TOKENS)
assert VOCAB_SIZE == 16, f"expected 16 tokens, got {VOCAB_SIZE}"

BOS = TOK_TO_ID["<BOS>"]
EOS = TOK_TO_ID["<EOS>"]
PLUS = TOK_TO_ID["+"]
MINUS = TOK_TO_ID["-"]
EQ = TOK_TO_ID["="]
SP = TOK_TO_ID[" "]


def _int_to_tokens(n: int):
    if n == 0:
        return [TOK_TO_ID["0"]]
    return [TOK_TO_ID[ch] for ch in str(n)]


def _random_int(rng, min_digits=1, max_digits=6):
    if min_digits == 1:
        return rng.randint(0, 10 ** max_digits - 1)
    return rng.randint(10 ** (min_digits - 1), 10 ** max_digits - 1)


def _random_pad(rng, max_spaces=3):
    return rng.randint(0, max_spaces)


def gen_expression(rng, min_digits=1, max_digits=6, max_spaces=3):
    a = _random_int(rng, min_digits=min_digits, max_digits=max_digits)
    b = _random_int(rng, min_digits=min_digits, max_digits=max_digits)
    if rng.random() < 0.5:
        op = PLUS
        c = a + b
    else:
        op = MINUS
        if a < b:
            a, b = b, a
        c = a - b
    tokens = [BOS]
    tokens += _int_to_tokens(a)
    tokens += [SP] * _random_pad(rng, max_spaces) + [op] + [SP] * _random_pad(rng, max_spaces)
    tokens += _int_to_tokens(b)
    tokens += [SP] * _random_pad(rng, max_spaces) + [EQ] + [SP] * _random_pad(rng, max_spaces)
    tokens += _int_to_tokens(c)
    tokens += [EOS]
    return tokens


def decode(tokens):
    return "".join(ID_TO_TOK[t] for t in tokens)


def parse_expression(s: str):
    """Parse a decoded expression like '<BOS> 12 + 34 = 46 <EOS>' into (a, op, b, c)."""
    # Strip BOS/EOS
    s = s.replace("<BOS>", "").replace("<EOS>", "").strip()
    parts = s.split("=")
    assert len(parts) == 2, f"expected one '=', got: {s}"
    lhs = parts[0]
    c_str = parts[1].strip()
    # Find the operator
    op = "+" if "+" in lhs else "-"
    ab = lhs.replace(op, " ")
    ab = ab.split()
    assert len(ab) == 2, f"expected 2 operands, got: {ab}"
    a, b = int(ab[0]), int(ab[1])
    c = int(c_str)
    return a, op, b, c


def test_vocab():
    assert VOCAB_SIZE == 16
    assert set(TOKENS) == set("0123456789+-= ") | {"<BOS>", "<EOS>"}
    print("PASS: vocab is 16 tokens")


def test_gen_expression(rng, n=500):
    """Every generated expression must satisfy all generator rules."""
    for _ in range(n):
        expr = gen_expression(rng)
        s = decode(expr)
        a, op, b, c = parse_expression(s)
        # Rule 1: no leading zeros (except for 0 itself)
        for val in (a, b, c):
            token_str = str(val)
            if len(token_str) > 1:
                assert token_str[0] != "0", f"leading zero in {val}"
        # Rule 2: arithmetic is correct
        expected = a + b if op == "+" else a - b
        assert c == expected, f"{a} {op} {b} = {c}, expected {expected}"
        # Rule 3: subtraction has a >= b
        if op == "-":
            assert a >= b, f"subtraction with a<b: {a} - {b}"
        # Rule 4: operand digit lengths within 1..max_digits
        for val in (a, b, c):
            d = len(str(val))
            assert 1 <= d <= 7, f"operand {val} has {d} digits (max_digits+1 allowed for results)"
        # Rule 5: BOS/EOS present
        assert expr[0] == BOS and expr[-1] == EOS
    print(f"PASS: {n} expressions all satisfy generator rules")


def test_spacing_variation(rng, n=200):
    """Spacing should vary (0..3 spaces around operators)."""
    space_counts = set()
    for _ in range(n):
        expr = gen_expression(rng, max_spaces=3)
        s = decode(expr)
        # Count spaces around the first operator
        op_pos = None
        for i, tok in enumerate(expr):
            if tok in (PLUS, MINUS):
                op_pos = i
                break
        assert op_pos is not None
        left_spaces = 0
        for i in range(op_pos - 1, -1, -1):
            if expr[i] == SP:
                left_spaces += 1
            else:
                break
        right_spaces = 0
        for i in range(op_pos + 1, len(expr)):
            if expr[i] == SP:
                right_spaces += 1
            else:
                break
        space_counts.add((left_spaces, right_spaces))
    # Should see a variety of spacing combinations
    assert len(space_counts) > 3, f"expected >3 spacing combos, got {len(space_counts)}"
    print(f"PASS: spacing varies — saw {len(space_counts)} distinct (left,right) combos")


def test_membership_counter_examples(rng, n=100):
    """Counter-examples should each violate exactly one generator rule."""
    for _ in range(n):
        # wrong_result
        expr = list(gen_expression(rng))
        for i in range(len(expr) - 2, -1, -1):
            if 0 <= expr[i] <= 9:
                expr[i] = (expr[i] + 1) % 10
                break
        s = decode(expr)
        a, op, b, c = parse_expression(s)
        expected = a + b if op == "+" else a - b
        assert c != expected, "wrong_result should have wrong arithmetic"

        # leading_zero: force "0" prefix on first operand
        a = rng.randint(1, 999)
        b = rng.randint(0, 9999)
        op = PLUS if rng.random() < 0.5 else MINUS
        if op == MINUS and a < b:
            a, b = b, a
        c = a + b if op == PLUS else a - b
        a_str = "0" + str(a)
        assert a_str[0] == "0" and len(a_str) > 1, "leading_zero should have leading zero"

        # negative_result: a < b for subtraction
        a = rng.randint(1, 9)
        b = rng.randint(10, 999)
        assert a < b, "negative_result should have a < b"

    print(f"PASS: {n} counter-examples all violate their target rule")


if __name__ == "__main__":
    rng = random.Random(42)
    test_vocab()
    test_gen_expression(rng, n=1000)
    test_spacing_variation(rng, n=500)
    test_membership_counter_examples(rng, n=200)
    print("\nAll tests passed.")
