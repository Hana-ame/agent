"""SELF-PLAY RL: the model generates a FULL additive expression — question AND
answer — and a Python verifier checks only whether the trailing answer is
correct. Reward = 1 if correct.

This differs from the earlier external-prompt RL (rl.py/grpo.py) where the
datasource asked "a op b = ?" and the model only produced the answer. Here the
model must self-pose a valid arithmetic problem first — which exposes the
"cheating" failure mode: the model could game the reward by only writing
trivial 1-digit questions. We measure both reward AND question-difficulty
distribution (operand digit lengths) over training to detect this.

Key metrics per step:
  reward      — fraction of self-generated expressions whose answer verifies
  mean digits — mean operand digit length of the questions the model poses
  share 1-digit— fraction of questions with 1-digit operands (cheating proxy)
  params / train_s / gen_tps — the three headline metrics

Usage:
  python -m additive_rand_transformer.rl_selfplay --checkpoint <SFT ckpt> \
      --steps 150 --kl_beta 0.05 --temperature 0.6
"""

from __future__ import annotations

import argparse
import os
import random
import time

import torch
import torch.nn.functional as F

from .data import BOS, EOS, PLUS, MINUS, EQ, SP, TOK_TO_ID
from .model import TinyGPT, TinyGPTConfig


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=str, required=True, help="SFT init weights")
    p.add_argument("--steps", type=int, default=150)
    p.add_argument("--batch_size", type=int, default=8, help="rollouts per step")
    p.add_argument("--max_new", type=int, default=60)
    p.add_argument("--temperature", type=float, default=0.6)
    p.add_argument("--lr", type=float, default=3e-5)
    p.add_argument("--kl_beta", type=float, default=0.05)
    p.add_argument("--min_digits_reward", type=int, default=0,
                   help="only reward correct questions whose BOTH operands have "
                        ">= this many digits (difficulty anchor; 0 = no anchor)")
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--log_every", type=int, default=10)
    p.add_argument("--save_every", type=int, default=50)
    p.add_argument("--runs_dir", type=str, default="runs/selfplay")
    return p.parse_args()


def parse_expression(ids):
    """Parse a self-generated token list [BOS ... EOS] into (ok, a, op_sym, b, ans).

    ok = structurally well-formed; ans = trailing digit run; verified separately.
    """
    toks = [t for t in ids if t not in (BOS, EOS)]
    eq_pos = None
    for i, t in enumerate(toks):
        if t == EQ:
            eq_pos = i
            break
    if eq_pos is None:
        return False, None, None, None, None
    lhs = toks[:eq_pos]
    # lhs = a SP op SP b  (spaces may be repeated)
    op_pos = None
    for i, t in enumerate(lhs):
        if t in (PLUS, MINUS):
            op_pos = i
            break
    if op_pos is None:
        return False, None, None, None, None
    a_digits = [t for t in lhs[:op_pos] if 0 <= t <= 9]
    b_digits = [t for t in lhs[op_pos + 1:] if 0 <= t <= 9]
    if not a_digits or not b_digits:
        return False, None, None, None, None
    a = int("".join(str(t) for t in a_digits))
    b = int("".join(str(t) for t in b_digits))
    op_sym = PLUS if lhs[op_pos] == PLUS else MINUS
    # trailing answer = digit run at end before EOS
    ans_digits = []
    for t in reversed(toks):
        if 0 <= t <= 9:
            ans_digits.append(t)
        else:
            break
    if not ans_digits:
        return False, None, None, None, None
    ans = int("".join(str(t) for t in reversed(ans_digits)))
    return True, a, op_sym, b, ans


def digit_len(n: int) -> int:
    return len(str(n))


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    ck = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    cfg = TinyGPTConfig(**ck["config"])
    model = TinyGPT(cfg)
    model.load_state_dict(ck["model"])
    print(f"loaded {args.checkpoint}: {cfg} params={model.num_parameters():,}")

    ref = None
    if args.kl_beta > 0.0:
        ref = TinyGPT(cfg)
        ref.load_state_dict(ck["model"])

    opt = model.configure_optimizers(weight_decay=0.01, learning_rate=args.lr,
                                     betas=(0.9, 0.99), device_type="cpu")
    runs_dir = os.path.join(args.runs_dir, time.strftime("%Y%m%d_%H%M%S"))
    os.makedirs(runs_dir, exist_ok=True)

    rng = random.Random(args.seed)
    t0 = time.time()
    history = []
    for step in range(1, args.steps + 1):
        opt["optimizer"].zero_grad(set_to_none=True)
        total_loss = 0.0
        n_ok = n_total = 0
        digit_sums = 0
        for _ in range(args.batch_size):
            # self-pose: model generates the FULL expression starting from <BOS>
            ids = [BOS]
            logps = []
            with torch.no_grad():
                for _ in range(args.max_new):
                    x = torch.tensor([ids], dtype=torch.long)
                    logits, _ = model(x, None)
                    logp = F.log_softmax(logits[0, -1], dim=-1)
                    probs = torch.exp(logp / args.temperature)
                    probs = probs / probs.sum()
                    nxt = int(torch.multinomial(probs, 1).item())
                    logps.append(float(logp[nxt].item()))
                    ids.append(nxt)
                    if nxt == EOS:
                        break

            ok, a, op, b, ans = parse_expression(ids)
            n_total += 1
            if ok:
                correct = op == PLUS and ans == a + b or op == MINUS and ans == a - b
                if correct:
                    n_ok += 1
                digit_sums += digit_len(a) + digit_len(b)
            # difficulty anchor: if set, only correct questions with both operands
            # >= min_digits get reward (this fights the 'always pose 1-digit'
            # gaming collapse)
            ok_for_reward = ok
            if ok_for_reward and args.min_digits_reward > 1:
                ok_for_reward = (digit_len(a) >= args.min_digits_reward and
                                 digit_len(b) >= args.min_digits_reward)
            r = 1.0 if (ok_for_reward and correct) else 0.0

            # REINFORCE update on the full trajectory (BOS..EOS)
            x = torch.tensor([ids], dtype=torch.long)
            logits, _ = model(x[:, :-1], None)
            logp_all = F.log_softmax(logits, dim=-1)
            traj = x[:, 1:]
            logp_traj = logp_all.gather(-1, traj.unsqueeze(-1)).squeeze(-1)
            loss = -r * logp_traj.sum()
            if ref is not None:
                with torch.no_grad():
                    ref_logits, _ = ref(x[:, :-1], None)
                    ref_lp = F.log_softmax(ref_logits, dim=-1).gather(-1, traj.unsqueeze(-1)).squeeze(-1)
                # k3 KL penalty
                kl = torch.exp(ref_lp - logp_traj) - (ref_lp - logp_traj) - 1.0
                loss = loss + args.kl_beta * kl.mean()
            loss.backward()
            total_loss += float(loss.item())

        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        opt["optimizer"].step()

        reward = n_ok / max(1, n_total)
        mean_digits = digit_sums / (2 * max(1, n_total))
        history.append({"step": step, "reward": reward, "mean_digits": mean_digits})
        if step % args.log_every == 0 or step == 1:
            print(f"selfplay step {step:4d} | loss {total_loss:9.2f} | reward {reward:.3f} "
                  f"| mean_op_digits {mean_digits:.2f} | {time.time()-t0:.1f}s", flush=True)
        if step % args.save_every == 0:
            torch.save({"step": step, "config": cfg.__dict__, "model": model.state_dict(),
                        "history": history}, os.path.join(runs_dir, f"selfplay_{step:06d}.pt"))

    final = os.path.join(runs_dir, "selfplay_final.pt")
    torch.save({"step": args.steps, "config": cfg.__dict__, "model": model.state_dict(),
                "history": history}, final)
    print(f"final -> {final}")
    # headline summary
    gen_tps = _gen_tps(model)
    print(f"reward {history[0]['reward']:.3f} -> {history[-1]['reward']:.3f} | "
          f"mean_op_digits {history[0]['mean_digits']:.2f} -> {history[-1]['mean_digits']:.2f}")
    print(f"params={model.num_parameters():,} train={time.time()-t0:.0f}s gen={gen_tps:.0f} tok/s")


def _gen_tps(model, secs=4.0):
    import time as _t
    rng = random.Random(1)
    ids = [BOS]
    with torch.no_grad():
        for _ in range(8):
            x = torch.tensor([ids], dtype=torch.long)
            logits, _ = model(x, None)
            ids.append(int(logits[0, -1].argmax()))
            if ids[-1] == EOS or len(ids) > 50:
                ids = [BOS]
        n = 0
        t0 = _t.time()
        while _t.time() - t0 < secs:
            x = torch.tensor([ids], dtype=torch.long)
            logits, _ = model(x, None)
            ids.append(int(logits[0, -1].argmax()))
            n += 1
            if ids[-1] == EOS or len(ids) > 50:
                ids = [BOS]
    return n / (_t.time() - t0)


if __name__ == "__main__":
    main()