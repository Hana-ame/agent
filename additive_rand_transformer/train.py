"""Training loop for the additive-arithmetic TinyGPT (CPU).

Run a quick smoke test (50 steps):
    python -m additive_rand_transformer.train --quick

Full run (a few thousand steps, checkpoint saved to runs/<ts>/checkpoint.pt):
    python -m additive_rand_transformer.train --steps 3000

Every `log_every` steps the script prints the mean log-likelihood of
generator output (`positive`) vs. five counter-example families that each
violate one generator rule. A growing margin means the model has learned
to tell generator output from non-generator output.
"""

from __future__ import annotations

import argparse
import math
import os
import random
import time

import torch

from .data import (BOS, EOS, PLUS, MINUS, EQ, SP, _int_to_tokens,
                   make_single_batch, make_single_cot_batch, stream_batches)
from .evaluate import membership_report, print_report
from .model import TinyGPT, TinyGPTConfig, VOCAB_SIZE

DEFAULT_RUNS_DIR = "runs"


def cot_completion_accuracy(model: TinyGPT, rng: random.Random, device: torch.device,
                            n_trials: int = 20, max_digits: int = 4) -> dict:
    """Greedy accuracy of the CoT model: given '<BOS> a op b =', does it emit the
    right trailing answer? Returns {label: fraction} for add/sub by digit count."""
    model.eval()
    results = {}
    with torch.no_grad():
        for op, opname in ((PLUS, "add"), (MINUS, "sub")):
            for nd in (1, 2, 3, 4):
                correct = total = 0
                for _ in range(n_trials):
                    a = rng.randint(10 ** (nd - 1) if nd > 1 else 0, 10 ** nd - 1)
                    b = rng.randint(10 ** (nd - 1) if nd > 1 else 0, 10 ** nd - 1)
                    if op == MINUS and a < b:
                        a, b = b, a
                    c = a + b if op == PLUS else a - b
                    prefix = ([BOS] + _int_to_tokens(a) + [SP, op, SP]
                              + _int_to_tokens(b) + [SP, EQ, SP])
                    ids = list(prefix)
                    for _ in range(60):
                        x = torch.tensor([ids], dtype=torch.long, device=device)
                        logits, _ = model(x, None)
                        nxt = int(logits[0, -1].argmax())
                        ids.append(nxt)
                        if nxt == EOS:
                            break
                    # trailing answer = digits between last '=' and EOS... but
                    # easier: find the digit run at the very end before EOS.
                    ans_digits = []
                    for t in reversed(ids):
                        if t == EOS:
                            continue
                        if 0 <= t <= 9:
                            ans_digits.append(t)
                        else:
                            break
                    ans = int("".join(str(t) for t in reversed(ans_digits))) if ans_digits else None
                    total += 1
                    if ans == c:
                        correct += 1
                results[f"{opname}{nd}"] = correct / max(1, total)
    model.train()
    return results


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=3000)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--block_size", type=int, default=1024)
    p.add_argument("--n_layer", type=int, default=2)
    p.add_argument("--n_head", type=int, default=4)
    p.add_argument("--n_embd", type=int, default=64)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--wd", type=float, default=0.1)
    p.add_argument("--warmup", type=int, default=200)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--log_every", type=int, default=25)
    p.add_argument("--eval_every", type=int, default=0,
                   help="run completion/membership eval every N steps (0 = same as log_every)")
    p.add_argument("--save_every", type=int, default=1000)
    p.add_argument("--runs_dir", type=str, default=DEFAULT_RUNS_DIR)
    p.add_argument("--quick", action="store_true", help="50-step smoke test, no checkpoint")
    p.add_argument("--max_digits", type=int, default=6,
                   help="max operand length (1-4 are always covered; longer allowed)")
    p.add_argument("--max_spaces", type=int, default=3,
                   help="max spaces around an operator (0..max_spaces, uniform)")
    p.add_argument("--single", action="store_true",
                   help="train on single expressions (no packing) — stronger arithmetic signal")
    p.add_argument("--cot", action="store_true",
                   help="train on chain-of-thought column-wise arithmetic (carry/borrow)")
    p.add_argument("--four_digit_bias", type=float, default=0.0,
                   help="CoT: fraction of samples with BOTH operands at max_digits "
                        "(oversamples hardest carry/overflow cases; 0.3 recommended)")
    return p.parse_args()


def lr_at(step: int, warmup: int, max_steps: int, base_lr: float) -> float:
    """Linear warmup + cosine decay."""
    if step < warmup:
        return base_lr * (step + 1) / warmup
    progress = (step - warmup) / max(1, max_steps - warmup)
    return base_lr * 0.5 * (1.0 + math.cos(math.pi * progress))


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    device = torch.device("cpu")

    cfg = TinyGPTConfig(
        vocab_size=VOCAB_SIZE,
        block_size=args.block_size,
        n_layer=args.n_layer,
        n_head=args.n_head,
        n_embd=args.n_embd,
    )
    model = TinyGPT(cfg).to(device)
    print(f"model: {cfg}  params={model.num_parameters():,}")

    opt = model.configure_optimizers(weight_decay=args.wd, learning_rate=args.lr,
                                     betas=(0.9, 0.95), device_type=str(device))

    steps = 50 if args.quick else args.steps
    if args.cot:
        rng = random.Random(args.seed)

        def _cot_iter():
            while True:
                yield make_single_cot_batch(rng, args.block_size, args.batch_size, device=str(device),
                                            max_digits=args.max_digits,
                                            four_digit_bias=args.four_digit_bias)
        data_iter = _cot_iter()
    elif args.single:
        rng = random.Random(args.seed)

        def _single_iter():
            while True:
                yield make_single_batch(rng, args.block_size, args.batch_size, device=str(device),
                                        max_digits=args.max_digits, max_spaces=args.max_spaces)
        data_iter = _single_iter()
    else:
        data_iter = stream_batches(args.block_size, args.batch_size, device=str(device), seed=args.seed,
                                   max_digits=args.max_digits, max_spaces=args.max_spaces)

    # Quick smoke test doesn't save anything.
    runs_dir: str | None = None
    if not args.quick:
        runs_dir = os.path.join(args.runs_dir, time.strftime("%Y%m%d_%H%M%S"))
        os.makedirs(runs_dir, exist_ok=True)

    t0 = time.time()
    for step in range(1, steps + 1):
        opt["optimizer"].param_groups[0]["lr"] = lr_at(step, args.warmup, steps, args.lr)
        opt["optimizer"].param_groups[1]["lr"] = lr_at(step, args.warmup, steps, args.lr)

        inputs, targets = next(data_iter)
        logits, loss = model(inputs, targets)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        opt["optimizer"].step()
        opt["optimizer"].zero_grad(set_to_none=True)

        eval_interval = args.eval_every or args.log_every
        if step % args.log_every == 0 or step == 1:
            model.eval()
            elapsed = time.time() - t0
            if args.cot:
                if step % eval_interval == 0 or step == 1:
                    acc = cot_completion_accuracy(model, random.Random(args.seed + step), device,
                                                  n_trials=10, max_digits=args.max_digits)
                    s = " | ".join(f"{k}{v*100:.0f}%" for k, v in acc.items())
                    print(f"step {step:5d} | loss {loss.item():.4f} | lr {opt['optimizer'].param_groups[0]['lr']:.2e} | "
                          f"cot_acc [{s}] | {elapsed:.1f}s")
                else:
                    print(f"step {step:5d} | loss {loss.item():.4f} | lr {opt['optimizer'].param_groups[0]['lr']:.2e} | {elapsed:.1f}s")
            else:
                report = membership_report(model, random.Random(args.seed + step), device, n_per_class=10)
                pos_ll = report["positive"][0]
                other_means = [v[0] for k, v in report.items() if k != "positive"]
                other_avg = sum(other_means) / len(other_means)
                margin = pos_ll - other_avg
                print(f"step {step:5d} | loss {loss.item():.4f} | lr {opt['optimizer'].param_groups[0]['lr']:.2e} | "
                      f"pos_ll {pos_ll:.2f} | other_avg {other_avg:.2f} | margin {margin:.2f} | {elapsed:.1f}s")
            model.train()

        if not args.quick and step % args.save_every == 0:
            ckpt = os.path.join(runs_dir, f"checkpoint_{step:06d}.pt")
            torch.save({"step": step, "config": cfg.__dict__, "model": model.state_dict(),
                        "optimizer": opt["optimizer"].state_dict()}, ckpt)
            print(f"saved {ckpt}")

    if not args.quick:
        final = os.path.join(runs_dir, "checkpoint_final.pt")
        torch.save({"step": steps, "config": cfg.__dict__, "model": model.state_dict(),
                    "optimizer": opt["optimizer"].state_dict()}, final)
        print(f"final checkpoint -> {final}")

    if args.cot:
        print("\n=== CoT completion accuracy (greedy, from '<BOS> a op b =') ===")
        acc = cot_completion_accuracy(model, random.Random(args.seed), device,
                                      n_trials=40, max_digits=args.max_digits)
        for k, v in acc.items():
            print(f"  {k:>5}: {v*100:5.1f}%")
    else:
        # Final membership report — shows the model can tell generator output apart
        # from a variety of counter-examples (wrong result, leading zero, format
        # violations, wrong operator).
        print_report(membership_report(model, random.Random(args.seed), device, n_per_class=20))


if __name__ == "__main__":
    main()
