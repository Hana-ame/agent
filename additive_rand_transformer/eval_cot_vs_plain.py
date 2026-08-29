"""Re-evaluate ALL cot vs plain checkpoints from disk with one unified protocol,
so the JSON overwrite bug (parallel processes clobbering one file) is bypassed."""
import json
import os
import random

import torch

from additive_rand_transformer.data import (BOS, EOS, PLUS, MINUS, EQ, SP, _int_to_tokens)
from additive_rand_transformer.model import TinyGPT, TinyGPTConfig

ROOT = "runs/cot_vs_plain"
GRID = [(nl, d) for nl in (1, 2, 4) for d in (64, 128, 256)]
N_TRIALS = 60


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


def completion_accuracy(model, n_trials=N_TRIALS):
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
            out[f"{op_name}{nd}"] = correct / max(1, total)
    return out


def main():
    rows = []
    for mode in ("plain", "cot"):
        for nl, d in GRID:
            tag = f"L{nl}_D{d}"
            ckpt_path = os.path.join(ROOT, mode, tag, "checkpoint_final.pt")
            if not os.path.exists(ckpt_path):
                print(f"MISSING {ckpt_path}")
                continue
            ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
            cfg = TinyGPTConfig(**ck["config"])
            model = TinyGPT(cfg)
            model.load_state_dict(ck["model"])
            model.eval()
            acc = completion_accuracy(model)
            loss = ck.get("loss")
            row = {"mode": mode, "n_layer": nl, "n_embd": d,
                   "params": model.num_parameters(),
                   "final_loss": round(loss, 4) if loss is not None else None,
                   "steps": ck["step"], **{k: round(v, 3) for k, v in acc.items()}}
            rows.append(row)
            print(f"[{mode}] {tag} loss {row['final_loss']} | "
                  f"add1 {acc['add1']*100:.0f}% add2 {acc['add2']*100:.0f}% "
                  f"add3 {acc['add3']*100:.0f}% add4 {acc['add4']*100:.0f}% | "
                  f"sub1 {acc['sub1']*100:.0f}% sub4 {acc['sub4']*100:.0f}%", flush=True)

    with open(os.path.join(ROOT, "cot_vs_plain_final.json"), "w") as f:
        json.dump(rows, f, indent=2)
    print(f"\nJSON -> {ROOT}/cot_vs_plain_final.json")


if __name__ == "__main__":
    main()