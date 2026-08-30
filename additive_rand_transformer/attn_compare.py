"""Attention-variant accuracy comparison: train causal/linear/dsa at the SAME
small config and report CoT completion accuracy (add/sub by digit length).

Small config (L2_D64) + reduced steps keeps the Python-loop linear attention
fast enough on CPU. Each variant is trained identically (same steps, seed,
batches) so accuracy differences are attributable to the attention mechanism.

Usage: python -m additive_rand_transformer.attn_compare [--steps 2000] [--n_layer 2] [--n_embd 64]
"""

from __future__ import annotations

import argparse
import os
import random
import time

import torch

from .data import (BOS, EOS, PLUS, MINUS, EQ, SP, _int_to_tokens,
                   make_single_cot_batch)
from .model import TinyGPT, TinyGPTConfig, VOCAB_SIZE


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=2000)
    p.add_argument("--n_layer", type=int, default=2)
    p.add_argument("--n_head", type=int, default=4)
    p.add_argument("--n_embd", type=int, default=64)
    p.add_argument("--topk", type=int, default=8)
    p.add_argument("--max_digits", type=int, default=4)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--root", type=str, default="runs/attn_compare")
    return p.parse_args()


def greedy_answer(model, a, b, op, max_new=70):
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
            cor = tot = 0
            for _ in range(n_trials):
                a = rng.randint(10 ** (nd - 1) if nd > 1 else 0, 10 ** nd - 1)
                b = rng.randint(10 ** (nd - 1) if nd > 1 else 0, 10 ** nd - 1)
                if op == MINUS and a < b:
                    a, b = b, a
                c = a + b if op == PLUS else a - b
                tot += 1
                if greedy_answer(model, a, b, op) == c:
                    cor += 1
            out[f"{op_name}{nd}"] = round(cor / max(1, tot), 3)
    return out


def main():
    args = parse_args()
    rows = []
    for attn in ("causal", "linear", "dsa"):
        cfg = TinyGPTConfig(vocab_size=VOCAB_SIZE, block_size=1024,
                            n_layer=args.n_layer, n_head=args.n_head,
                            n_embd=args.n_embd, attn_type=attn,
                            attn_topk=args.topk)
        model = TinyGPT(cfg)
        opt = model.configure_optimizers(weight_decay=0.1, learning_rate=3e-4,
                                         betas=(0.9, 0.95), device_type="cpu")
        rng = random.Random(args.seed)
        t0 = time.time()
        final_loss = None
        for _ in range(args.steps):
            x, y = make_single_cot_batch(rng, 1024, args.batch_size, "cpu", 1, args.max_digits)
            logits, loss = model(x, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt["optimizer"].step()
            opt["optimizer"].zero_grad(set_to_none=True)
            final_loss = loss.item()
        dt = time.time() - t0
        model.eval()
        acc = completion_accuracy(model)
        save = os.path.join(args.root, attn)
        os.makedirs(save, exist_ok=True)
        torch.save({"config": cfg.__dict__, "model": model.state_dict(),
                    "loss": final_loss, "step": args.steps},
                   os.path.join(save, "checkpoint_final.pt"))
        rows.append((attn, model.num_parameters(), final_loss, dt, acc))
        print(f"[{attn}] {dt:.0f}s loss {final_loss:.3f} | "
              f"add1 {acc['add1']*100:.0f}% add2 {acc['add2']*100:.0f}% "
              f"add3 {acc['add3']*100:.0f}% add4 {acc['add4']*100:.0f}% | "
              f"sub1 {acc['sub1']*100:.0f}% sub4 {acc['sub4']*100:.0f}%", flush=True)

    print("\n=== attention comparison (L{}_{} params {}) ===".format(
        args.n_layer, args.n_embd, rows[0][1]))
    for attn, params, loss, dt, acc in rows:
        print(f"  {attn:<7} loss {loss:.3f} | add {acc['add1']*100:.0f}/{acc['add2']*100:.0f}/"
              f"{acc['add3']*100:.0f}/{acc['add4']*100:.0f}% | sub {acc['sub1']*100:.0f}/"
              f"{acc['sub2']*100:.0f}/{acc['sub3']*100:.0f}/{acc['sub4']*100:.0f}%")


if __name__ == "__main__":
    main()