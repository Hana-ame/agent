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

from .data import make_single_batch, pack_blocks, stream_batches
from .evaluate import membership_report, print_report
from .model import TinyGPT, TinyGPTConfig, VOCAB_SIZE

DEFAULT_RUNS_DIR = "runs"


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
    p.add_argument("--save_every", type=int, default=1000)
    p.add_argument("--runs_dir", type=str, default=DEFAULT_RUNS_DIR)
    p.add_argument("--quick", action="store_true", help="50-step smoke test, no checkpoint")
    p.add_argument("--max_digits", type=int, default=6,
                   help="max operand length (1-4 are always covered; longer allowed)")
    p.add_argument("--max_spaces", type=int, default=3,
                   help="max spaces around an operator (0..max_spaces, uniform)")
    p.add_argument("--single", action="store_true",
                   help="train on single expressions (no packing) — stronger arithmetic signal")
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
    if args.single:
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

        if step % args.log_every == 0 or step == 1:
            model.eval()
            report = membership_report(model, random.Random(args.seed + step), device, n_per_class=10)
            pos_ll = report["positive"][0]
            other_means = [v[0] for k, v in report.items() if k != "positive"]
            other_avg = sum(other_means) / len(other_means)
            margin = pos_ll - other_avg
            model.train()
            elapsed = time.time() - t0
            print(f"step {step:5d} | loss {loss.item():.4f} | lr {opt['optimizer'].param_groups[0]['lr']:.2e} | "
                  f"pos_ll {pos_ll:.2f} | other_avg {other_avg:.2f} | margin {margin:.2f} | {elapsed:.1f}s")

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

    # Final membership report — shows the model can tell generator output apart
    # from a variety of counter-examples (wrong result, leading zero, out of
    # range, format violations, wrong operator).
    print_report(membership_report(model, random.Random(args.seed), device, n_per_class=20))


if __name__ == "__main__":
    main()
