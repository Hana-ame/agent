"""Parameter scan: train every (n_layer, n_embd) config and record metrics.

Grid: n_layer in {1,2,4,6}, n_embd in {64,128,256}  (n_head=4 fixed).
Each config: single-expression mode, 4000 steps, batch 32, seed 0.
Records: final loss, membership per-class separation, and arithmetic
completion accuracy (addition/subtraction by digit length).

Writes runs/scan/<nl>_<d>/checkpoint_final.pt + a JSON summary, and prints a
markdown table for SCAN.md.

Usage: python -m additive_rand_transformer.scan
"""

from __future__ import annotations

import json
import os
import random
import time

import torch
import torch.nn.functional as F

from .data import (BOS, EOS, PLUS, MINUS, EQ, SP, TOK_TO_ID, _int_to_tokens,
                   make_single_batch, decode)
from .evaluate import membership_report
from .model import TinyGPT, TinyGPTConfig, VOCAB_SIZE

GRID = [(nl, d) for nl in (1, 2, 4, 6) for d in (64, 128, 256)]
STEPS = 4000
BATCH = 32
SEED = 0
SCAN_ROOT = "runs/scan"


def train_config(n_layer: int, n_embd: int, steps: int = STEPS):
    cfg = TinyGPTConfig(vocab_size=VOCAB_SIZE, block_size=1024,
                        n_layer=n_layer, n_head=4, n_embd=n_embd)
    model = TinyGPT(cfg)
    opt = model.configure_optimizers(weight_decay=0.1, learning_rate=3e-4,
                                     betas=(0.9, 0.95), device_type="cpu")
    rng = random.Random(SEED)
    final_loss = None
    for _ in range(steps):
        x, y = make_single_batch(rng, 1024, BATCH, "cpu", 1, 6, 3)
        logits, loss = model(x, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt["optimizer"].step()
        opt["optimizer"].zero_grad(set_to_none=True)
        final_loss = loss.item()
    return model, cfg, final_loss


def complete(prefix_ids, model, max_new=20):
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


def parse_result(ids):
    seen_eq = skip_spaces = False
    digits = []
    for t in ids:
        if t == EOS:
            break
        if t == BOS:
            continue
        if t == EQ:
            seen_eq = skip_spaces = True
            continue
        if seen_eq:
            if skip_spaces and t == SP:
                continue
            skip_spaces = False
            if 0 <= t <= 9:
                digits.append(str(t))
            elif t == SP:
                break
    return int("".join(digits)) if digits else None


def arithmetic_accuracy(model, n_trials=80):
    rng = random.Random(123)
    out = {}
    for op_name, op in (("add", PLUS), ("sub", MINUS)):
        for nd in (1, 2, 3, 4):
            correct = total = 0
            for _ in range(n_trials):
                a = rng.randint(10 ** (nd - 1) if nd > 1 else 0, 10 ** nd - 1)
                b = rng.randint(10 ** (nd - 1) if nd > 1 else 0, 10 ** nd - 1)
                if op == MINUS and a < b:
                    a, b = b, a
                c = a + b if op == PLUS else a - b
                pref = ([BOS] + _int_to_tokens(a) + [SP, op, SP]
                        + _int_to_tokens(b) + [SP, EQ, SP])
                pred = parse_result(complete(pref, model))
                total += 1
                if pred == c:
                    correct += 1
            out[f"{op_name}{nd}"] = round(correct / max(1, total), 3)
    return out


def main() -> None:
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--only", type=str, default="", help="comma list like 'L2_D64,L4_D128'; empty = full grid")
    p.add_argument("--root", type=str, default=SCAN_ROOT, help="output root dir")
    p.add_argument("--steps", type=int, default=STEPS, help="training steps per config")
    args = p.parse_args()

    grid = []
    if args.only:
        for s in args.only.split(","):
            nl_s, d_s = s.strip().lstrip("L").split("_")
            grid.append((int(nl_s.lstrip("L")), int(d_s.lstrip("D"))))
    else:
        grid = GRID
    os.makedirs(args.root, exist_ok=True)
    rows = []
    for n_layer, n_embd in grid:
        tag = f"L{n_layer}_D{n_embd}"
        t0 = time.time()
        model, cfg, loss = train_config(n_layer, n_embd, steps=args.steps)
        dt = time.time() - t0

        model.eval()
        rep = membership_report(model, random.Random(7), torch.device("cpu"), n_per_class=80)
        pos = rep["positive"][0]
        margins = {k: round(pos - v[0], 3) for k, v in rep.items() if k != "positive"}
        acc = arithmetic_accuracy(model)

        save_dir = os.path.join(args.root, tag)
        os.makedirs(save_dir, exist_ok=True)
        torch.save({"config": cfg.__dict__, "model": model.state_dict(),
                    "loss": loss, "step": args.steps},
                   os.path.join(save_dir, "checkpoint_final.pt"))

        row = {"n_layer": n_layer, "n_embd": n_embd,
               "params": model.num_parameters(), "final_loss": round(loss, 4),
               "steps": args.steps, "time_s": round(dt, 1), **margins, **acc}
        rows.append(row)
        print(f"{tag} done in {dt:.0f}s | loss {loss:.3f} | "
              f"margin wrong_op {margins['wrong_operator']:.2f} | "
              f"acc 1sub {acc['sub1']*100:.0f}% | 1add {acc['add1']*100:.0f}% | "
              f"2add {acc['add2']*100:.0f}% | 4add {acc['add4']*100:.0f}%", flush=True)

    tag_suffix = os.path.basename(args.root)
    with open(os.path.join(args.root, f"scan_{tag_suffix}.json"), "w") as f:
        json.dump(rows, f, indent=2)

    # Markdown table
    lines = ["| config | params | loss | wrong_op | lead_zero | neg_res | wrong_res | 1sub | 1add | 2add | 4add |",
             "|--------|--------|------|----------|-----------|---------|-----------|------|------|------|------|"]
    for r in rows:
        lines.append(
            f"| L{r['n_layer']}·D{r['n_embd']} | {r['params']:,} | {r['final_loss']:.3f} "
            f"| {r['wrong_operator']:.2f} | {r['leading_zero']:.2f} | {r['negative_result']:.2f} "
            f"| {r['wrong_result']:.2f} | {r['sub1']*100:.0f}% | {r['add1']*100:.0f}% "
            f"| {r['add2']*100:.0f}% | {r['add4']*100:.0f}% |")
    table = "\n".join(lines)
    print("\n" + table)
    print(f"\nJSON -> {args.root}/scan_{tag_suffix}.json")


if __name__ == "__main__":
    main()
