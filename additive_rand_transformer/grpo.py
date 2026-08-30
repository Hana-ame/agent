"""GRPO (Group Relative Policy Optimization) for the CoT arithmetic model.

Following DeepSeek-R1 style GRPO:
  * for each problem, sample a GROUP of G completions
  * reward each by answer correctness (1/0)
  * advantage_i = (r_i - mean(r_group)) / std(r_group)   (group-relative baseline)
  * policy loss = -1/G * sum_i advantage_i * log p(gen_i)
    (no critic network; the group mean/std replaces the value function)
  * optional KL penalty to the frozen SFT reference, using the k3 estimator
      KL(pol||ref) ≈ exp(ref_logp - logp) - (ref_logp - logp) - 1   (Shao et al.)
  * optional PPO-style clipping on the ratio for stability

Usage:
  python -m additive_rand_transformer.grpo --checkpoint <SFT ckpt> \
      --grpo_steps 200 --group_size 8 --temperature 0.5 --kl_beta 0.05 \
      --sparse_from 3 --density 0.5 --max_digits 4
"""

from __future__ import annotations

import argparse
import os
import random
import time

import torch
import torch.nn.functional as F

from .data import BOS, EOS, PLUS, MINUS, EQ, SP, _int_to_tokens, _random_int
from .model import TinyGPT, TinyGPTConfig, VOCAB_SIZE


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--grpo_steps", type=int, default=200)
    p.add_argument("--batch_size", type=int, default=8, help="problems per step")
    p.add_argument("--group_size", type=int, default=8, help="G completions per problem")
    p.add_argument("--max_new", type=int, default=80)
    p.add_argument("--temperature", type=float, default=0.5)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--kl_beta", type=float, default=0.05)
    p.add_argument("--clip_ratio", type=float, default=0.2,
                   help="PPO-style ratio clip; 0 disables")
    p.add_argument("--min_digits", type=int, default=1)
    p.add_argument("--max_digits", type=int, default=4)
    p.add_argument("--sparse_from", type=int, default=3)
    p.add_argument("--density", type=float, default=0.5)
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--log_every", type=int, default=10)
    p.add_argument("--save_every", type=int, default=100)
    p.add_argument("--runs_dir", type=str, default="runs/grpo")
    return p.parse_args()


def sample_problem(rng, min_digits, max_digits, sparse_from, density):
    a = _random_int(rng, min_digits, max_digits, sparse_from, density)
    b = _random_int(rng, min_digits, max_digits, sparse_from, density)
    if rng.random() < 0.5:
        op, answer = PLUS, a + b
    else:
        op = MINUS
        if a < b:
            a, b = b, a
        answer = a - b
    pref = ([BOS] + _int_to_tokens(a) + [SP, op, SP]
            + _int_to_tokens(b) + [SP, EQ, SP])
    return a, b, op, answer, pref


def parse_trailing_answer(ids) -> int | None:
    digits = []
    for t in reversed(ids):
        if t == EOS:
            continue
        if 0 <= t <= 9:
            digits.append(t)
        else:
            break
    return int("".join(str(t) for t in reversed(digits))) if digits else None


def sample_group(model, prefix, answer, group_size, max_new, temperature):
    """Sample a group; returns (ids_list, logp_lists, pref_len, rewards)."""
    ids_list, lp_list, rewards, plen = [], [], [], None
    with torch.no_grad():
        for _ in range(group_size):
            ids = list(prefix)
            lps = []
            for _ in range(max_new):
                x = torch.tensor([ids], dtype=torch.long)
                logits, _ = model(x, None)
                logp = F.log_softmax(logits[0, -1], dim=-1)
                probs = torch.exp(logp / temperature)
                probs = probs / probs.sum()
                nxt = int(torch.multinomial(probs, 1).item())
                lps.append(float(logp[nxt].item()))
                ids.append(nxt)
                if nxt == EOS:
                    break
            pred = parse_trailing_answer(ids)
            ids_list.append(ids)
            lp_list.append(lps)
            rewards.append(1.0 if pred == answer else 0.0)
    return ids_list, lp_list, len(prefix), rewards


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    cfg = TinyGPTConfig(**ckpt["config"])
    model = TinyGPT(cfg)
    model.load_state_dict(ckpt["model"])
    print(f"loaded {args.checkpoint}: {cfg} params={model.num_parameters():,}")

    ref = None
    if args.kl_beta > 0.0:
        ref = TinyGPT(cfg)
        ref.load_state_dict(ckpt["model"])
        print(f"reference (frozen) for KL, beta={args.kl_beta}, clip={args.clip_ratio}")

    opt = model.configure_optimizers(weight_decay=0.01, learning_rate=args.lr,
                                     betas=(0.9, 0.99), device_type="cpu")
    runs_dir = os.path.join(args.runs_dir, time.strftime("%Y%m%d_%H%M%S"))
    os.makedirs(runs_dir, exist_ok=True)

    rng = random.Random(args.seed)
    t0 = time.time()
    rew_hist = []
    for step in range(1, args.grpo_steps + 1):
        opt["optimizer"].zero_grad(set_to_none=True)
        total_loss, rew_acc, n_rew = 0.0, 0.0, 0
        for _ in range(args.batch_size):
            _, _, _, answer, pref = sample_problem(
                rng, args.min_digits, args.max_digits, args.sparse_from, args.density)
            ids_list, lp_list, plen, rewards = sample_group(
                model, pref, answer, args.group_size, args.max_new, args.temperature)
            mean_r = sum(rewards) / len(rewards)
            std_r = (sum((r - mean_r) ** 2 for r in rewards) / len(rewards)) ** 0.5 + 1e-8

            for ids, lps, r in zip(ids_list, lp_list, rewards):
                adv = (r - mean_r) / std_r          # GRPO group-relative advantage
                gen = ids[plen:]
                x = torch.tensor([ids], dtype=torch.long)
                logits, _ = model(x[:, :-1], None)
                start = plen - 1
                targets = torch.tensor([gen], dtype=torch.long)
                logp = F.log_softmax(logits, dim=-1)
                logp_gen = logp[0, start:start + len(gen)].gather(1, targets).squeeze(1)
                if ref is not None:
                    with torch.no_grad():
                        ref_logits, _ = ref(x[:, :-1], None)
                        ref_lp = F.log_softmax(ref_logits, dim=-1)
                        ref_logp_gen = ref_lp[0, start:start + len(gen)].gather(1, targets).squeeze(1)
                    # k3 KL estimator per token, mean over trajectory
                    kl = torch.exp(ref_logp_gen - logp_gen) - (ref_logp_gen - logp_gen) - 1.0
                    kl_term = args.kl_beta * kl.mean()
                else:
                    kl_term = 0.0

                # GRPO policy loss: maximize advantage * log-prob of the trajectory,
                # plus KL penalty vs frozen reference (k3 estimator).
                loss = -adv * logp_gen.sum() + kl_term
                loss.backward()
                total_loss += float(loss.item())
            rew_acc += sum(rewards)
            n_rew += len(rewards)

        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        opt["optimizer"].step()
        mean_rew = rew_acc / max(1, n_rew)
        rew_hist.append(mean_rew)
        if step % args.log_every == 0 or step == 1:
            print(f"grpo step {step:4d} | loss {total_loss:9.2f} | mean_reward {mean_rew:.3f} "
                  f"| {time.time()-t0:.1f}s", flush=True)
        if step % args.save_every == 0:
            torch.save({"step": step, "config": cfg.__dict__, "model": model.state_dict(),
                        "reward_hist": rew_hist},
                       os.path.join(runs_dir, f"grpo_{step:06d}.pt"))
    final = os.path.join(runs_dir, "grpo_final.pt")
    torch.save({"step": args.grpo_steps, "config": cfg.__dict__, "model": model.state_dict(),
                "reward_hist": rew_hist}, final)
    print(f"final -> {final}")
    print(f"reward start {rew_hist[0]:.3f} -> end {rew_hist[-1]:.3f}")


if __name__ == "__main__":
    main()