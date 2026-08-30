"""Training-volume sweep: same config, different step counts (= tokens seen),
records completion accuracy vs training volume. Data source is the dynamic
sparse CoT generator (streaming, so 'epochs' over a fixed set does not apply —
every batch is fresh expressions; 'tokens seen' = steps x batch x seq_len).

Usage: python -m additive_rand_transformer.volume_sweep [--n_layer 4] [--n_embd 128]
"""

from __future__ import annotations

import argparse
import os
import random
import time

import torch

from .data import BOS, EOS, PLUS, MINUS, EQ, SP, _int_to_tokens, make_single_cot_batch
from .model import TinyGPT, TinyGPTConfig, VOCAB_SIZE


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--n_layer", type=int, default=4)
    p.add_argument("--n_head", type=int, default=4)
    p.add_argument("--n_embd", type=int, default=128)
    p.add_argument("--max_digits", type=int, default=4)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--steps_list", type=str, default="500,1000,2000,4000")
    p.add_argument("--sparse_from", type=int, default=3)
    p.add_argument("--density", type=float, default=0.5)
    p.add_argument("--root", type=str, default="runs/volume_sweep")
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


def completion_accuracy(model, n_trials=30):
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
    steps_list = [int(s) for s in args.steps_list.split(",")]
    cfg = TinyGPTConfig(vocab_size=VOCAB_SIZE, block_size=1024,
                        n_layer=args.n_layer, n_head=args.n_head, n_embd=args.n_embd)
    model = TinyGPT(cfg)
    opt = model.configure_optimizers(weight_decay=0.1, learning_rate=3e-4,
                                     betas=(0.9, 0.95), device_type="cpu")
    rng = random.Random(args.seed)
    print(f"config L{args.n_layer}_D{args.n_embd} params={model.num_parameters():,} "
          f"sparse(density={args.density},from={args.sparse_from}) max_digits={args.max_digits}")
    prev = 0
    tokens_per_step = None
    for step_target in steps_list:
        t0 = time.time()
        # train from prev to step_target (incremental)
        for _ in range(step_target - prev):
            x, y = make_single_cot_batch(rng, 1024, args.batch_size, "cpu", 1,
                                         args.max_digits, sparse_from=args.sparse_from,
                                         density=args.density)
            if tokens_per_step is None:
                tokens_per_step = x.numel()
            logits, loss = model(x, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt["optimizer"].step()
            opt["optimizer"].zero_grad(set_to_none=True)
        prev = step_target
        dt = time.time() - t0
        model.eval()
        acc = completion_accuracy(model)
        tokens_seen = step_target * tokens_per_step
        print(f"steps {step_target:5d} | tokens_seen {tokens_seen:>12,} | "
              f"add {acc['add1']*100:>3.0f}/{acc['add2']*100:>3.0f}/{acc['add3']*100:>3.0f}/{acc['add4']*100:>3.0f}% "
              f"sub {acc['sub1']*100:>3.0f}/{acc['sub2']*100:>3.0f}/{acc['sub3']*100:>3.0f}/{acc['sub4']*100:>3.0f}% "
              f"| +{dt:.0f}s", flush=True)
        save = os.path.join(args.root, f"steps_{step_target}")
        os.makedirs(save, exist_ok=True)
        torch.save({"config": cfg.__dict__, "model": model.state_dict(),
                    "step": step_target, "tokens_seen": tokens_seen,
                    "accuracy": acc}, os.path.join(save, "checkpoint.pt"))


if __name__ == "__main__":
    main()