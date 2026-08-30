"""Token/s throughput benchmark for training and generation.

Measures on CPU:
  * TRAIN throughput: SFT-style forward+backward on CoT batches (tokens/s)
  * GEN throughput: greedy autoregressive generation, tokens/s (one token per
    full-context forward — this is the naive no-KV-cache path; the linear
    attention KV state and causal/DSa KV cache would be faster but this is the
    honest per-forward cost of the implemented models)

Usage: python -m additive_rand_transformer.bench [--n_layer N] [--n_embd D]
                                                [--attn_type TYPE] [--secs 5]
"""

from __future__ import annotations

import argparse
import random
import time

import torch

from .data import BOS, EOS, PLUS, EQ, SP, _int_to_tokens, make_single_cot_batch
from .model import TinyGPT, TinyGPTConfig, VOCAB_SIZE


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--n_layer", type=int, default=4)
    p.add_argument("--n_head", type=int, default=4)
    p.add_argument("--n_embd", type=int, default=128)
    p.add_argument("--attn_type", type=str, default="causal",
                   choices=["causal", "linear", "dsa"])
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--max_digits", type=int, default=4)
    p.add_argument("--secs", type=float, default=5.0, help="benchmark seconds each")
    return p.parse_args()


def train_tps(model, batch_size, max_digits, secs) -> tuple[float, int]:
    """Return (tokens_per_sec, tokens_per_step)."""
    cfg = model.cfg
    opt = model.configure_optimizers(weight_decay=0.1, learning_rate=3e-4,
                                     betas=(0.9, 0.95), device_type="cpu")
    rng = random.Random(0)
    x, y = make_single_cot_batch(rng, 1024, batch_size, "cpu", 1, max_digits)
    T = x.shape[1]
    tokens_per_step = x.numel()
    # warmup
    for _ in range(3):
        logits, loss = model(x, y)
        loss.backward()
        opt["optimizer"].step()
        opt["optimizer"].zero_grad(set_to_none=True)
    n = 0
    t0 = time.time()
    while time.time() - t0 < secs:
        logits, loss = model(x, y)
        loss.backward()
        opt["optimizer"].step()
        opt["optimizer"].zero_grad(set_to_none=True)
        n += 1
    dt = time.time() - t0
    return (n * tokens_per_step) / dt, tokens_per_step


def gen_tps(model, max_digits, secs) -> float:
    """Greedy one-token-at-a-time generation, tokens/s (full-context forward each step)."""
    rng = random.Random(1)
    # build a fixed prompt
    a = rng.randint(1, 9999)
    b = rng.randint(1, 9999)
    pref = ([BOS] + _int_to_tokens(a) + [SP, PLUS, SP]
            + _int_to_tokens(b) + [SP, EQ, SP])
    ids = list(pref)
    # warmup 10 steps
    for _ in range(10):
        x = torch.tensor([ids], dtype=torch.long)
        logits, _ = model(x, None)
        nxt = int(logits[0, -1].argmax())
        ids.append(nxt)
        if nxt == EOS:
            ids = list(pref)
    n_tokens = 0
    t0 = time.time()
    while time.time() - t0 < secs:
        x = torch.tensor([ids], dtype=torch.long)
        logits, _ = model(x, None)
        nxt = int(logits[0, -1].argmax())
        ids.append(nxt)
        n_tokens += 1
        if nxt == EOS or len(ids) > 90:
            ids = list(pref)
    dt = time.time() - t0
    return n_tokens / dt


def main() -> None:
    args = parse_args()
    cfg = TinyGPTConfig(vocab_size=VOCAB_SIZE, block_size=1024,
                        n_layer=args.n_layer, n_head=args.n_head,
                        n_embd=args.n_embd, attn_type=args.attn_type,
                        attn_topk=8)
    model = TinyGPT(cfg)
    print(f"config: L{cfg.n_layer}_D{cfg.n_embd}_H{cfg.n_head} attn={cfg.attn_type} "
          f"params={model.num_parameters():,}")
    train_tps_, tps_step = train_tps(model, args.batch_size, args.max_digits, args.secs)
    gen_tps_ = gen_tps(model, args.max_digits, args.secs)
    print(f"  TRAIN: {train_tps_:,.0f} tok/s  ({tps_step:,} tok/step)")
    print(f"  GEN:   {gen_tps_:,.0f} tok/s  (naive full-context per step)")


if __name__ == "__main__":
    main()