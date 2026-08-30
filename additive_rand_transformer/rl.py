"""REINFORCE reinforcement learning for the CoT arithmetic model (robust version).

Agent: the CoT model turns a random problem from the SPARSE datasource into a
full chain of thought + answer. Reward = 1 if the trailing answer is correct.

Stability measures (learned from earlier unstable runs):
  * log-probs are summed ONLY over generated tokens (correct shift, no prefix
    contamination)
  * advantages are standardized per problem (minus baseline, divided by std)
  * KL penalty and entropy bonus use MEAN over the trajectory, not the sum, so
    trajectory length does not blow up the loss; KL is capped as a penalty on
    (log p_ref - log p) which pushes the policy back toward the SFT model
  * sampling temperature 0.8 keeps rollouts close to the greedy path

Usage:
  python -m additive_rand_transformer.rl --checkpoint <SFT ckpt> \
      --rl_steps 200 --n_samples 8 --sparse_from 3 --density 0.5 --max_digits 4 \
      --kl_beta 0.05 --entropy_bonus 0.01
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
    p.add_argument("--checkpoint", type=str, required=True, help="init weights (SFT model)")
    p.add_argument("--rl_steps", type=int, default=200)
    p.add_argument("--batch_size", type=int, default=8, help="problems per optimizer step")
    p.add_argument("--n_samples", type=int, default=8, help="rollouts per problem")
    p.add_argument("--max_new", type=int, default=80, help="max generated tokens per rollout")
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--lr", type=float, default=3e-5)
    p.add_argument("--entropy_bonus", type=float, default=0.01)
    p.add_argument("--kl_beta", type=float, default=0.05,
                   help="KL penalty vs reference (0=off); mean-based so safe at 0.05")
    p.add_argument("--min_digits", type=int, default=1)
    p.add_argument("--max_digits", type=int, default=4)
    p.add_argument("--sparse_from", type=int, default=3)
    p.add_argument("--density", type=float, default=0.5)
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--log_every", type=int, default=10)
    p.add_argument("--save_every", type=int, default=100)
    p.add_argument("--runs_dir", type=str, default="runs/rl")
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


def rollout_with_logp(model, prefix: list[int], max_new: int, temperature: float):
    """Sample one continuation, recording log-probs ONLY of generated tokens.

    Returns (full_ids, pref_len, gen_logp_tensor(list of floats as tensor)).
    """
    ids = list(prefix)
    logps: list[float] = []
    with torch.no_grad():
        for _ in range(max_new):
            x = torch.tensor([ids], dtype=torch.long)
            logits, _ = model(x, None)
            logp = F.log_softmax(logits[0, -1], dim=-1)
            probs = torch.exp(logp / temperature)
            probs = probs / probs.sum()
            nxt = int(torch.multinomial(probs, 1).item())
            logps.append(float(logp[nxt].item()))
            ids.append(nxt)
            if nxt == EOS:
                break
    return ids, len(prefix), logps


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    device = torch.device("cpu")

    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    cfg = TinyGPTConfig(**ckpt["config"])
    model = TinyGPT(cfg).to(device)
    model.load_state_dict(ckpt["model"])
    print(f"loaded {args.checkpoint}: {cfg} params={model.num_parameters():,}")

    ref = None
    if args.kl_beta > 0.0:
        ref = TinyGPT(cfg).to(device)
        ref.load_state_dict(ckpt["model"])
        print(f"reference (frozen) for KL, beta={args.kl_beta}")

    opt = model.configure_optimizers(weight_decay=0.01, learning_rate=args.lr,
                                     betas=(0.9, 0.99), device_type="cpu")
    runs_dir = os.path.join(args.runs_dir, time.strftime("%Y%m%d_%H%M%S"))
    os.makedirs(runs_dir, exist_ok=True)

    rng = random.Random(args.seed)
    t0 = time.time()
    reward_hist = []
    for step in range(1, args.rl_steps + 1):
        opt["optimizer"].zero_grad(set_to_none=True)
        total_loss = 0.0
        rew_acc, n_rew = 0.0, 0
        for _ in range(args.batch_size):
            a, b, op, answer, pref = sample_problem(
                rng, args.min_digits, args.max_digits, args.sparse_from, args.density)

            traj_ids: list[list[int]] = []
            traj_lps: list[list[float]] = []
            rewards: list[float] = []
            plen: int | None = None
            for _ in range(args.n_samples):
                ids, plen, lps = rollout_with_logp(model, pref, args.max_new, args.temperature)
                pred = parse_trailing_answer(ids)
                r = 1.0 if pred == answer else 0.0
                traj_ids.append(ids)
                traj_lps.append(lps)
                rewards.append(r)

            base = sum(rewards) / len(rewards)
            var = sum((r - base) ** 2 for r in rewards) / len(rewards)
            std = var ** 0.5 + 1e-8

            for ids, lps, r in zip(traj_ids, traj_lps, rewards):
                adv = (r - base) / std          # standardized advantage
                gen_tokens = ids[plen:]
                # re-forward to get gradients, gather ONLY generated-token logps
                x = torch.tensor([ids], dtype=torch.long)
                logits, _ = model(x[:, :-1], None)   # logits[i] predicts x[i+1]
                # positions plen-1 .. len-2 in logits predict gen_tokens[0..]
                pred_logp = F.log_softmax(logits, dim=-1)
                # targets = ids[plen:] aligned with logits index plen-1
                start = plen - 1
                targets = torch.tensor([gen_tokens], dtype=torch.long)
                logp_gen = pred_logp[0, start:start + len(gen_tokens)].gather(
                    1, targets).squeeze(1)
                mean_logp = logp_gen.mean()
                loss = -adv * mean_logp * logp_gen.numel()   # scale back is fine (adv std-normed)
                # entropy bonus (mean): encourage spread
                ent = -mean_logp
                loss = loss - args.entropy_bonus * ent
                if ref is not None:
                    with torch.no_grad():
                        ref_logits, _ = ref(x[:, :-1], None)
                        ref_lp = F.log_softmax(ref_logits, dim=-1)
                        ref_logp_gen = ref_lp[0, start:start + len(gen_tokens)].gather(
                            1, targets).squeeze(1)
                    # KL(pol || ref) = E[log pol - log ref]; +KL penalty pushes the
                    # policy BACK toward the frozen SFT model when it overconfidently
                    # drifts. (Sign: logp - ref_logp, NOT ref - logp.)
                    loss = loss + args.kl_beta * (logp_gen - ref_logp_gen).mean()
                loss.backward()
                total_loss += float(loss.item())
            rew_acc += sum(rewards)
            n_rew += len(rewards)

        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        opt["optimizer"].step()
        mean_rew = rew_acc / max(1, n_rew)
        reward_hist.append(mean_rew)
        if step % args.log_every == 0 or step == 1:
            print(f"rl step {step:4d} | loss {total_loss:9.2f} | mean_reward {mean_rew:.3f} "
                  f"| {time.time()-t0:.1f}s", flush=True)
        if step % args.save_every == 0:
            torch.save({"step": step, "config": cfg.__dict__, "model": model.state_dict(),
                        "reward_hist": reward_hist},
                       os.path.join(runs_dir, f"rl_{step:06d}.pt"))
            print(f"saved rl_{step:06d}.pt", flush=True)

    final = os.path.join(runs_dir, "rl_final.pt")
    torch.save({"step": args.rl_steps, "config": cfg.__dict__, "model": model.state_dict(),
                "reward_hist": reward_hist}, final)
    print(f"final -> {final}")
    print(f"reward start {reward_hist[0]:.3f} -> end {reward_hist[-1]:.3f}")


if __name__ == "__main__":
    main()