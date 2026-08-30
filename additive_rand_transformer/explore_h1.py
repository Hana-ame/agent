"""EXPLORE-H1: is the CoT carry column functionally necessary?  (corrected)

The model gets the CoT columns (operands + per-column sums + carries) as a
PREFIX, but NOT the final answer — it must generate the answer itself. Then we
corrupt one component of the prefix and check whether the regenerated answer
changes. Because the answer is never in the context, any change is real.

  * carry-flip:      corrupt the OUTGOING carry digit (the value the model
                     stored and must propagate to the next column)
  * sum-perturb:     corrupt a digit of one column's sum
  * operand-perturb: corrupt a digit of the left operand  (sanity control: a
                     computing model MUST change the answer)

Interpretation:
  - operand-perturb HIGH (near 100%) => model really computes from operands
  - carry-flip HIGH        => stored carry is functionally used in computation
  - carry-flip ~ operand   => carry matters as much as the operands themselves
  - carry-flip LOW while operand HIGH => model recomputes carry from operands
    each step (robust), OR ignores the stored carry column entirely
  - all LOW => memorized surface patterns (no real computation)

Usage: python -m additive_rand_transformer.explore_h1 --checkpoint <ckpt>
"""

from __future__ import annotations

import argparse
import random

import torch

from .data import BOS, EOS, PLUS, MINUS, EQ, SP, TOK_TO_ID, gen_expression_cot
from .model import TinyGPT, TinyGPTConfig


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--n_trials", type=int, default=150)
    p.add_argument("--max_digits", type=int, default=4)
    p.add_argument("--seed", type=int, default=7)
    return p.parse_args()


def greedy_from(model, prefix_ids, max_new=60):
    ids = list(prefix_ids)
    with torch.no_grad():
        for _ in range(max_new):
            x = torch.tensor([ids], dtype=torch.long)
            logits, _ = model(x, None)
            nxt = int(logits[0, -1].argmax())
            ids.append(nxt)
            if nxt == EOS:
                break
    return ids


def trailing_answer(ids):
    ds = []
    for t in reversed(ids):
        if t == EOS:
            continue
        if 0 <= t <= 9:
            ds.append(t)
        else:
            break
    return int("".join(str(t) for t in reversed(ds))) if ds else None


def split_prefix_answer(expr):
    """expr = [BOS, ..., columns..., SP, answer_digits, EOS].

    Returns (prefix_ids [BOS.. last carry], answer_digits_list).
    """
    # drop EOS, then walk back past the trailing answer digits to the last SP
    toks = list(expr)
    assert toks[-1] == EOS
    body = toks[:-1]
    k = len(body) - 1
    while k >= 0 and 0 <= body[k] <= 9:
        k -= 1
    # k now points at the SP right before the answer (or earlier if no SP)
    answer_digits = body[k + 1:]
    prefix = body[: k + 1]
    return prefix, answer_digits


def find_digit_positions(toks):
    """Return lists of positions for columns' sum spans, carry-out positions, and
    operand digits, all in prefix coordinates."""
    eqs = [i for i, t in enumerate(toks) if t == EQ]
    sums, carries = [], []
    for ei in eqs:
        j = ei + 1
        while j < len(toks) and toks[j] != SP:
            j += 1
        sums.append([p for p in range(ei + 1, j) if 0 <= toks[p] <= 9])
        carries.append(j + 1)      # single digit after the SP after sum
    op1 = next(i for i, t in enumerate(toks) if t in (PLUS, MINUS))
    operands = [i for i in range(1, op1) if 0 <= toks[i] <= 9]
    return sums, carries, operands


def main():
    args = parse_args()
    ck = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model = TinyGPT(TinyGPTConfig(**ck["config"]))
    model.load_state_dict(ck["model"])
    model.eval()

    rng = random.Random(args.seed)
    n = {"carry": 0, "sum": 0, "opnd": 0}
    chg = {"carry": 0, "sum": 0, "opnd": 0}
    base_correct = 0

    for _ in range(args.n_trials):
        expr = gen_expression_cot(rng, 1, args.max_digits)
        prefix, answer_digits = split_prefix_answer(expr)
        true_ans = int("".join(str(t) for t in answer_digits))
        if not prefix or prefix[0] != BOS:
            continue

        sums, carries, operands = find_digit_positions(prefix)

        # base answer from prefix (columns only, no answer)
        base_ids = greedy_from(model, prefix)
        base_ans = trailing_answer(base_ids)
        if base_ans == true_ans:
            base_correct += 1

        # --- carry flip ---
        ok_carry = [c for c in carries if c < len(prefix)
                    and prefix[c] in (TOK_TO_ID["0"], TOK_TO_ID["1"])]
        if ok_carry:
            m = list(prefix)
            p = rng.choice(ok_carry)
            m[p] = TOK_TO_ID["1"] if prefix[p] == TOK_TO_ID["0"] else TOK_TO_ID["0"]
            a2 = trailing_answer(greedy_from(model, m))
            n["carry"] += 1
            if a2 != base_ans:
                chg["carry"] += 1

        # --- sum perturb ---
        flat_sums = [p for s in sums for p in s]
        if flat_sums:
            m = list(prefix)
            p = rng.choice(flat_sums)
            m[p] = (prefix[p] + 1) % 10
            a2 = trailing_answer(greedy_from(model, m))
            n["sum"] += 1
            if a2 != base_ans:
                chg["sum"] += 1

        # --- operand perturb ---
        if operands:
            m = list(prefix)
            p = rng.choice(operands)
            m[p] = (prefix[p] + 1) % 10
            a2 = trailing_answer(greedy_from(model, m))
            n["opnd"] += 1
            if a2 != base_ans:
                chg["opnd"] += 1

    print(f"model: {args.checkpoint}  trials={args.n_trials}  "
          f"prefix-only base accuracy: {base_correct}/{args.n_trials} = {base_correct/args.n_trials:.1%}")
    print(f"{'corruption':<16} {'n':>4} {'answer changed':>16} {'rate':>8}")
    for k in ("opnd", "sum", "carry"):
        print(f"{k:<16} {n[k]:>4} {chg[k]:>16} {chg[k]/max(1,n[k]):>7.1%}")
    print("\nread: operand rate = sanity (computing model ~100%). "
          "carry rate vs operand rate => whether stored carry is functionally used.")


if __name__ == "__main__":
    main()