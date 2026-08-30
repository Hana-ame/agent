"""IMPROVE experiments: MoE / LoRA / curriculum-SFT comparisons.

Three independent experiments against matched baselines (CPU, tiny models):

A) MoE: dense MLP vs MoE(4 experts, top-2) at same config — total params grow
   ~4x, active per token ~2x. Reports accuracy + per-expert routing stats.
B) LoRA: from a fixed SFT checkpoint, compare (i) full fine-tune vs
   (ii) frozen base + LoRA rank r, at same steps on hard (4-digit biased) data.
   Reports add4 improvement and trainable-parameter fraction.
C) Curriculum SFT: stage1 easy (1-2 digit) then stage2 hard (3-4 digit) vs
   mixed training at equal total steps.

Usage: python -m additive_rand_transformer.improve [--experiment A|B|C|all]
"""

from __future__ import annotations

import argparse
import os
import random
import time

import torch

from .data import (BOS, EOS, PLUS, MINUS, EQ, SP, _int_to_tokens,
                   make_single_cot_batch)
from .model import TinyGPT, TinyGPTConfig, VOCAB_SIZE, apply_lora


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--experiment", type=str, default="all", choices=["A", "B", "C", "all"])
    p.add_argument("--steps", type=int, default=3000)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--max_digits", type=int, default=4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--root", type=str, default="runs/improve")
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
    ds = []
    for t in reversed(ids):
        if t == EOS:
            continue
        if 0 <= t <= 9:
            ds.append(t)
        else:
            break
    return int("".join(str(t) for t in reversed(ds))) if ds else None


def gen_tps(model, secs=4.0):
    """Greedy one-token-at-a-time generation throughput (tokens/s), CPU."""
    import time
    rng = random.Random(1)
    a = rng.randint(1, 9999)
    b = rng.randint(1, 9999)
    pref = ([BOS] + _int_to_tokens(a) + [SP, PLUS, SP]
            + _int_to_tokens(b) + [SP, EQ, SP])
    ids = list(pref)
    with torch.no_grad():
        for _ in range(8):                     # warmup
            x = torch.tensor([ids], dtype=torch.long)
            logits, _ = model(x, None)
            ids.append(int(logits[0, -1].argmax()))
            if ids[-1] == EOS or len(ids) > 60:
                ids = list(pref)
        n = 0
        t0 = time.time()
        while time.time() - t0 < secs:
            x = torch.tensor([ids], dtype=torch.long)
            logits, _ = model(x, None)
            ids.append(int(logits[0, -1].argmax()))
            n += 1
            if ids[-1] == EOS or len(ids) > 60:
                ids = list(pref)
        dt = time.time() - t0
    return n / dt


def completion_accuracy(model, n_trials=40):
    rng = random.Random(123)
    out = {}
    for opn, op in (("add", PLUS), ("sub", MINUS)):
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
            out[f"{opn}{nd}"] = round(cor / max(1, tot), 3)
    return out


def train_steps(model, rng, opt, steps, max_digits, bias=0.0, sparse_from=3, density=1.0):
    """Train `steps` batches; returns final loss."""
    fl = None
    for _ in range(steps):
        x, y = make_single_cot_batch(rng, 1024, 32, "cpu", 1, max_digits,
                                     four_digit_bias=bias, sparse_from=sparse_from,
                                     density=density)
        logits, loss = model(x, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt["optimizer"].step()
        opt["optimizer"].zero_grad(set_to_none=True)
        fl = loss.item()
    return fl


def exp_a_moe(args):
    print("=== A) MoE: dense vs MoE(4 experts, top2), L2_D64, steps", args.steps, "===")
    rng = random.Random(args.seed)
    cfg = TinyGPTConfig(vocab_size=VOCAB_SIZE, block_size=1024, n_layer=2,
                        n_head=4, n_embd=64)
    runs = {}
    for tag, moe_cfg in [("dense", None),
                         ("moe4_top2", TinyGPTConfig(vocab_size=VOCAB_SIZE,
                                                     block_size=1024, n_layer=2,
                                                     n_head=4, n_embd=64,
                                                     n_experts=4, moe_topk=2))]:
        cfg_use = moe_cfg or cfg
        model = TinyGPT(cfg_use)
        opt = model.configure_optimizers(weight_decay=0.1, learning_rate=3e-4,
                                         betas=(0.9, 0.95), device_type="cpu")
        t0 = time.time()
        train_steps(model, rng, opt, args.steps, 4)
        dt = time.time() - t0
        model.eval()
        acc = completion_accuracy(model)
        tps = gen_tps(model)
        runs[tag] = {"params": model.num_parameters(), "train_s": round(dt, 1),
                     "gen_tps": round(tps, 1), "acc": acc}
        print(f"  {tag}: params={model.num_parameters():,} train={dt:.0f}s "
              f"gen={tps:.0f} tok/s | "
              f"add {acc['add1']*100:.0f}/{acc['add2']*100:.0f}/{acc['add3']*100:.0f}/{acc['add4']*100:.0f}% "
              f"sub {acc['sub1']*100:.0f}/{acc['sub4']*100:.0f}%", flush=True)
    return runs


def exp_a_moe_routing(args):
    """MoE routing: do experts specialize by problem difficulty (1-digit easy vs
    4-digit hard with carries)? Compare expert hit-rate on easy vs hard samples."""
    print("=== A2) MoE expert routing specialization ===")
    cfg = TinyGPTConfig(vocab_size=VOCAB_SIZE, block_size=1024, n_layer=2,
                        n_head=4, n_embd=64, n_experts=4, moe_topk=2)
    model = TinyGPT(cfg)
    opt = model.configure_optimizers(weight_decay=0.1, learning_rate=3e-4,
                                     betas=(0.9, 0.95), device_type="cpu")
    rng = random.Random(args.seed)
    train_steps(model, rng, opt, args.steps, 4)

    import collections
    freq = collections.defaultdict(collections.Counter)  # expert -> {easy, hard}

    def tally(model, batch_x, label):
        B, T = batch_x.shape
        with torch.no_grad():
            h = model.drop(model.token_embedding(batch_x) +
                           model.position_embedding(torch.arange(T)))
            for blk in model.transformer:
                h = h + blk.attn(blk.ln_1(h))
                g = blk.mlp.router(blk.ln_2(h))
                _, idx = g.topk(2, dim=-1)
                for e in range(4):
                    freq[e][label] += int((idx == e).sum().item())
                # replace h after MLP path (only needed if more blocks — we use
                # router output only, so the block MLP value doesn't matter here)
                # still apply to keep residual flow valid for next level if any
                from .model import MoE
                if hasattr(blk.mlp, "experts"):
                    pass  # router-only tally; residual not needed after last block

    for _ in range(60):
        easy_x, _ = make_single_cot_batch(rng, 1024, 8, "cpu", 1, 1,  # 1-digit operands
                                          four_digit_bias=0.0, sparse_from=9, density=1.0)
        hard_x, _ = make_single_cot_batch(rng, 1024, 8, "cpu", 4, 4,  # both 4-digit
                                          four_digit_bias=1.0, sparse_from=9, density=1.0)
        tally(model, easy_x, "easy")
        tally(model, hard_x, "hard")
    print("  expert hit-rate by problem difficulty (1-digit vs 4-digit):")
    for e in range(4):
        s = freq[e]["easy"]; h = freq[e]["hard"]; tot = s + h
        print(f"    expert {e}: easy {s/max(1,tot):.0%}  hard {h/max(1,tot):.0%}"
              + ("  <- specialized" if max(s, h) / max(1, tot) > 0.6 else ""))


def exp_b_lora(args):
    print("=== B) LoRA vs full fine-tune from L4_D128 CoT ckpt (hard 4-digit data) ===")
    base_ckpt = "runs/20260830_044602/checkpoint_final.pt"  # L4_D128 bias0.5 SFT
    if not os.path.exists(base_ckpt):
        print(f"  ! missing {base_ckpt}, skipping (train it first)")
        return
    ck = torch.load(base_ckpt, map_location="cpu", weights_only=False)
    base_cfg = TinyGPTConfig(**ck["config"])
    results = {}
    for tag, rank in [("full_finetune", 0), ("lora_r8", 8), ("lora_r32", 32)]:
        model = TinyGPT(base_cfg)
        model.load_state_dict(ck["model"])
        rng = random.Random(args.seed)
        extra = {}
        if rank > 0:
            n_ad = apply_lora(model, rank=rank)
            trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
            extra = {"adapters": n_ad, "trainable": trainable,
                     "trainable_frac": round(trainable / model.num_parameters(), 3)}
            print(f"  {tag}: adapters={n_ad} trainable={trainable:,} ("
                  f"{trainable/model.num_parameters():.1%} of base)")
        lr = 3e-4
        opt = model.configure_optimizers(weight_decay=0.01, learning_rate=lr,
                                         betas=(0.9, 0.99), device_type="cpu")
        t0 = time.time()
        # hard data: bias towards 4-digit both operands (carry/overflow heavy)
        train_steps(model, rng, opt, 1500, 4, bias=1.0)
        dt = time.time() - t0
        model.eval()
        acc = completion_accuracy(model)
        tps = gen_tps(model)
        results[tag] = {"params": model.num_parameters(),
                        "train_s": round(dt, 1), "gen_tps": round(tps, 1),
                        "acc": acc, **extra}
        print(f"  {tag}: params={model.num_parameters():,} train={dt:.0f}s "
              f"gen={tps:.0f} tok/s | add4={acc['add4']*100:.0f}% "
              f"(SFT基线 add4≈38%) | add1-3 {acc['add1']*100:.0f}/{acc['add2']*100:.0f}/{acc['add3']*100:.0f}% | "
              f"sub4={acc['sub4']*100:.0f}%", flush=True)
    return results


def exp_c_curriculum(args):
    print("=== C) curriculum SFT: easy(1-2d) then hard(3-4d) vs mixed ===")
    steps = args.steps
    half = steps // 2
    results = {}
    for tag, plan in [("mixed", [("mix", steps)]),
                      ("curriculum", [("easy", half), ("hard", half)])]:
        cfg = TinyGPTConfig(vocab_size=VOCAB_SIZE, block_size=1024, n_layer=2,
                            n_head=4, n_embd=64)
        model = TinyGPT(cfg)
        opt = model.configure_optimizers(weight_decay=0.1, learning_rate=3e-4,
                                         betas=(0.9, 0.95), device_type="cpu")
        rng = random.Random(args.seed)
        t0 = time.time()
        for stage, n in plan:
            if stage == "mix":
                # mixed: uniform digit lengths, bias 0.3 toward 4-digit
                train_steps(model, rng, opt, n, 4, bias=0.3, density=0.6)
            elif stage == "easy":
                train_steps(model, rng, opt, n, 2, bias=0.0, density=1.0)
            else:
                train_steps(model, rng, opt, n, 4, bias=0.5, density=0.5)
        dt = time.time() - t0
        model.eval()
        acc = completion_accuracy(model)
        tps = gen_tps(model)
        results[tag] = {"params": model.num_parameters(), "train_s": round(dt, 1),
                        "gen_tps": round(tps, 1), "acc": acc}
        print(f"  {tag}: params={model.num_parameters():,} train={dt:.0f}s "
              f"gen={tps:.0f} tok/s | add {acc['add1']*100:.0f}/{acc['add2']*100:.0f}/"
              f"{acc['add3']*100:.0f}/{acc['add4']*100:.0f}% "
              f"sub {acc['sub1']*100:.0f}/{acc['sub2']*100:.0f}/{acc['sub3']*100:.0f}/{acc['sub4']*100:.0f}%",
              flush=True)
    return results


def main():
    import json
    args = parse_args()
    os.makedirs(args.root, exist_ok=True)
    summary = {}
    if args.experiment in ("A", "all"):
        summary["A_moe"] = exp_a_moe(args)
        exp_a_moe_routing(args)
    if args.experiment in ("B", "all"):
        summary["B_lora"] = exp_b_lora(args)
    if args.experiment in ("C", "all"):
        summary["C_curriculum"] = exp_c_curriculum(args)
    out = os.path.join(args.root, "improve_summary.json")
    with open(out, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\nsummary -> {out}")


if __name__ == "__main__":
    main()