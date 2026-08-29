"""Validate gen_expression_cot via SP-segmented grammar (robust)."""
import random
from additive_rand_transformer.data import (gen_expression_cot, TOK_TO_ID,
                                            BOS, EOS, PLUS, MINUS, EQ, SP, decode)

def parse(expr):
    toks = [t for t in expr if t != EOS]
    # strip BOS
    assert toks[0] == BOS
    toks = toks[1:]
    # split by SP
    segs, cur = [], []
    for t in toks:
        if t == SP:
            segs.append(cur); cur = []
        else:
            cur.append(t)
    segs.append(cur)
    # segment 0 = "a op b = ..."  -> a_digits op b_digits EQ rest? no: BOS a SP op SP b SP EQ SP ...
    # Actually format: BOS a SP op SP b SP EQ SP col0 SP after0 SP col1 SP after1 ... SP answer
    # so segs[0]=a_digits, segs[1]=[op], segs[2]=b_digits, segs[3]=[EQ],
    # then alternating col / after, last seg = answer digits
    a = int(''.join(map(str, segs[0])))
    op = segs[1][0]
    b = int(''.join(map(str, segs[2])))
    assert segs[3] == [EQ]
    cols, afters = [], []
    i = 4
    while i + 1 < len(segs):
        col = segs[i]
        # col = da op db carry EQ sumdigits
        da, db, carry = col[0], col[2], col[4]
        assert col[1] == op
        assert col[3] == op
        eqpos = col.index(EQ)
        s = int(''.join(map(str, col[eqpos+1:])))
        after = int(''.join(map(str, segs[i+1])))
        cols.append((da, db, carry, s, after))
        afters.append(after)
        i += 2
    ans = int(''.join(map(str, segs[i])))
    return a, op, b, cols, ans

rng = random.Random(0)
N = 4000
for trial in range(N):
    expr = gen_expression_cot(rng, 1, 4)
    a, op, b, cols, ans = parse(expr)
    exp_ans = a + b if op == PLUS else a - b
    assert ans == exp_ans, f'ANS {a}{op}{b}={ans} exp {exp_ans} | {decode(expr)}'
    carry = 0
    if op == PLUS:
        for (da, db, c0, s, after) in cols:
            assert c0 == carry, f'carry-in {a}{op}{b}: {c0} != {carry}'
            assert s == da + db + carry, f'sum {a}{op}{b} col {da}+{db}+{carry}'
            assert after == s // 10
            carry = s // 10
    else:
        borrow = 0
        for (da, db, b0, d, after) in cols:
            assert b0 == borrow, f'borrow-in {a}{op}{b}: {b0} != {borrow}'
            expected = da - db - borrow
            if expected < 0:
                assert d == expected + 10 and after == 1, f'borrow col {a}{op}{b}'
                borrow = 1
            else:
                assert d == expected and after == 0, f'plain col {a}{op}{b}'
                borrow = 0
print(f'PASS: {N} CoT expressions — every column and answer correct')

rng = random.Random(1)
for _ in range(3):
    print(decode(gen_expression_cot(rng, 1, 4)))
