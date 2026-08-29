"""CoT vs plain-data comparison scan.

For every (n_layer, n_embd) in the grid we train TWO models with IDENTICAL
hyper-parameters and step count:
  * plain — single-expression data (no chain of thought): <BOS> a op b = c <EOS>
  * cot   — chain-of-thought column-wise data:     <BOS> a op b = cols... c <EOS>
Both are evaluated with the SAME greedy completion protocol: feed
'<BOS> a op b =' and parse the trailing digit run before EOS as the predicted
answer. This is the only fair Apples-to-Apples metric across the two formats.

Grid: n_layer in {1,2,4} x n_embd in {64,128,256} (n_head=4), max_digits=4,
3000 steps, batch 32, seed 0. Writes runs/cot_vs_plain/<mode>/<tag>/checkpoint +
JSON summary + markdown table for COT_VS_PLAIN.md.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import time

import torch

from .data import (BOS, EOS, PLUS, MINUS, EQ, SP, _int_to_tokens,
                   make_single_batch, make_single_cot_batch)
from .model import TinyGPT, TinyGPTConfig, VOCAB_SIZE

GRID = [(nl, d) for nl in (1, 2, 4) for d in (64, 128, 256)]
STEPS = 3000
BATCH = 32
SEED = 0
MAX_DIGITS = 4
ROOT = "runs/cot_vs_plain"


def train(mode: str, n_layer: int, n_embd: int, steps: int):
    cfg = TinyGPTConfig(vocab_size=VOCAB_SIZE, block_size=1024,
                        n_layer=n_layer, n_head=4, n_embd=n_embd)
    model = TinyGPT(cfg)
    opt = model.configure_optimizers(weight_decay=0.1, learning_rate=3e-4,
                                     betas=(0.9, 0.95), device_type="cpu")
    rng = random.Random(SEED)
    make_batch = (make_single_cot_batch if mode == "cot" else make_single_batch)
    final_loss = None
    for _ in range(steps):
        x, y = make_batch(rng, 1024, BATCH, "cpu", 1, MAX_DIGITS)
        logits, loss = model(x, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt["optimizer"].step()
        opt["optimizer"].zero_grad(set_to_none=True)
        final_loss = loss.item()
    return model, cfg, final_loss


def greedy_answer(model, a, b, op, max_new=70):
    """Feed '<BOS> a op b =', greedy-decode, parse trailing digit run = answer."""
    pref = ([BOS] + _int_to_tokens(a) + [SP, op, SP] + _int_to_tokens(b) + [SP, EQ, SP])
    ids = list(pref)
    with torch.no_grad():
        for _ in range(max_new):
            x = torch.tensor([ids], dtype=torch.long)
            logits, _ = model(x, None)
            nxt = int(logits[0, -1].argmax())
            ids.append(nxt)
            if nxt == EOS:
                break
    digits = []
    for t in reversed(ids):
        if t == EOS:
            continue
        if 0 <= t <= 9:
            digits.append(t)
        else:
            break
    return int("".join(str(t) for t in reversed(digits))) if digits else None


def completion_accuracy(model, n_trials=40):
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
                total += 1
                if greedy_answer(model, a, b, op) == c:
                    correct += 1
            out[f"{op_name}{nd}"] = round(correct / max(1, total), 3)
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["plain", "cot"], required=True)
    p.add_argument("--only", type=str, default="", help="comma list like 'L2_D64,L4_D128'")
    p.add_argument("--root", type=str, default=ROOT)
    p.add_argument("--steps", type=int, default=STEPS)
    args = p.parse_args()

    grid = []
    if args.only:
        for s in args.only.split(","):
            nl_s, d_s = s.strip().lstrip("L").split("_")
            grid.append((int(nl_s.lstrip("L")), int(d_s.lstrip("D"))))
    else:
        grid = GRID

    root = os.path.join(args.root, args.mode)
    os.makedirs(root, exist_ok=True)
    rows = []
    for n_layer, n_embd in grid:
        tag = f"L{n_layer}_D{n_embd}"
        t0 = time.time()
        model, cfg, loss = train(args.mode, n_layer, n_embd, args.steps)
        dt = time.time() - t0
        model.eval()
        acc = completion_accuracy(model)
        save_dir = os.path.join(root, tag)
        os.makedirs(save_dir, exist_ok=True)
        torch.save({"config": cfg.__dict__, "model": model.state_dict(),
                    "loss": loss, "step": args.steps},
                   os.path.join(save_dir, "checkpoint_final.pt"))
        row = {"mode": args.mode, "n_layer": n_layer, "n_embd": n_embd,
               "params": model.num_parameters(), "final_loss": round(loss, 4),
               "steps": args.steps, "time_s": round(dt, 1), **acc}
        rows.append(row)
        print(f"[{args.mode}] {tag} done {dt:.0f}s | loss {loss:.3f} | "
              f"sub1 {acc['sub1']*100:.0f}% | add1 {acc['add1']*100:.0f}% | "
              f"add2 {acc['add2']*100:.0f}% | add3 {acc['add3']*100:.0f}% | "
              f"add4 {acc['add4']*100:.0f}% | sub4 {acc['sub4']*100:.0f}%", flush=True)

    with open(os.path.join(root, f"cot_vs_plain_{args.mode}.json"), "w") as f:
        json.dump(rows, f, indent=2)
    print(f"JSON -> {root}/cot_vs_plain_{args.mode}.json")


if __name__ == "__main__":
    main()