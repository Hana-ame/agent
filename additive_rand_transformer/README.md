# additive_rand_transformer

A tiny GPT that runs entirely on CPU, trained on **additive-arithmetic expressions**
generated on the fly. After training, the model can judge whether a given
expression was produced by *this specific generator* — not merely whether the
arithmetic is correct.

## Vocab (16 tokens)

| id | token | id | token | id | token |
|----|-------|----|-------|----|-------|
| 0-9 | `0`..`9` | 10 | `+` | 11 | `-` |
| 12 | `=` | 13 | ` ` (space) | 14 | `<BOS>` |
| 15 | `<EOS>` | | | | |

## Generator rules

* operands: 1..`max_digits` digit non-negative integers (1-4 digits are always
  covered; longer operands are also allowed — default `max_digits=6`)
* **no leading zeros** (except for the number `0` itself)
* for `+`: `result = a + b` (no upper bound; the result may be longer than
  either operand)
* for `-`: `a >= b` so the result is non-negative (the generator swaps
  operands to enforce this)
* **spacing is random**: 0..`max_spaces` spaces around each operator
  (default `max_spaces=3`), so both `1234+5678=6912` and
  `1234  +  5678 = 6912` are valid output
* every expression is wrapped `<BOS> ... <EOS>` and multiple expressions are
  packed into a single block of length `block_size` (default 1024)

## Membership test

A model trained on the generator learns this whole distribution. Given ANY
expression, its log-likelihood tells you whether it plausibly came from THIS
generator. `evaluate.py` probes this with five counter-example families, each
violating exactly one generator rule:

| class | what it violates |
|-------|------------------|
| `positive`        | nothing — from the generator |
| `wrong_result`    | arithmetic result is wrong |
| `leading_zero`    | first operand has a leading zero |
| `negative_result` | subtraction where `a < b` (generator forbids) |
| `wrong_operator`  | uses `=` where an operator should be |

A well-trained model assigns **high** log-likelihood to `positive` and **low**
to all the others. The margin is the membership score.

## Model

Tiny GPT, CPU-only: `vocab=16, block_size=1024, n_layer=2, n_head=4,
n_embd=64` — about **200K parameters**. Weight-tied embedding/head, LayerNorm,
GELU, causal self-attention.

## Quick start

```bash
# 1. Install torch CPU-only (no CUDA needed)
pip install torch --index-url https://download.pytorch.org/whl/cpu

# 2. 50-step smoke test (no checkpoint, ~1 min on CPU)
python -m additive_rand_transformer.train --quick

# 3. Full run (checkpoint saved to runs/<timestamp>/checkpoint_final.pt)
python -m additive_rand_transformer.train --steps 3000
```

## CLI parameters

```bash
python -m additive_rand_transformer.train [options]
```

| option | default | meaning |
|--------|---------|---------|
| `--quick` | off | 50-step smoke test, no checkpoint |
| `--steps` | 3000 | total training steps |
| `--batch_size` | 8 | sequences per batch |
| `--block_size` | 1024 | context length (packed expressions) |
| `--n_layer` | 2 | transformer blocks |
| `--n_head` | 4 | attention heads |
| `--n_embd` | 64 | embedding dimension |
| `--lr` | 3e-4 | peak learning rate (warmup + cosine) |
| `--wd` | 0.1 | weight decay |
| `--warmup` | 200 | linear warmup steps |
| `--grad_clip` | 1.0 | gradient norm clip |
| `--seed` | 1337 | RNG seed (deterministic data + model init) |
| `--log_every` | 25 | print interval |
| `--save_every` | 1000 | checkpoint interval |
| `--max_digits` | 6 | max operand length (1-4 always covered) |
| `--max_spaces` | 3 | max spaces around an operator (0..N, uniform) |

## Understanding the output

Every `log_every` steps the script prints:

```
step    25 | loss 2.7134 | lr 3.75e-04 | pos_ll -18.32 | other_avg -22.50 | margin 4.18 | 12.3s
```

* `loss` — cross-entropy training loss (lower = better next-token prediction)
* `pos_ll` — mean log-likelihood of expressions **from the generator** (higher = better)
* `other_avg` — mean log-likelihood of 4 counter-example families (should be lower)
* `margin` — `pos_ll - other_avg` (should grow as the model learns the distribution)

At the end, a full membership report is printed:

```
========================================================================
MEMBERSHIP TEST  (mean log-likelihood, higher = more like generator)
========================================================================
class              mean_ll   sample
------------------------------------------------------------------------
positive             -12.30   <BOS>1234 + 5678 = 6912<EOS>
wrong_result         -28.41   <BOS>1234 + 5678 = 6913<EOS>
leading_zero         -19.87   <BOS>0123 + 4 = 127<EOS>
negative_result      -22.15   <BOS>5 - 234 = 229<EOS>
wrong_operator       -31.02   <BOS>1234 = 5678 = 6912<EOS>
------------------------------------------------------------------------
positive mean: -12.30   others avg: -25.36   margin: 13.06
========================================================================
```

A positive margin means the model can tell generator output apart from
non-generator output.

## Using a trained checkpoint

```python
import torch
from additive_rand_transformer.model import TinyGPT, TinyGPTConfig
from additive_rand_transformer.evaluate import sequence_logprob

ckpt = torch.load("runs/<timestamp>/checkpoint_final.pt", map_location="cpu")
cfg = TinyGPTConfig(**ckpt["config"])
model = TinyGPT(cfg)
model.load_state_dict(ckpt["model"])
model.eval()

# Judge any expression: high ll = looks like generator output
from additive_rand_transformer.data import gen_expression
import random
expr = gen_expression(random.Random(0))   # a valid expression
ll = sequence_logprob(model, expr, torch.device("cpu"))
print(f"log-likelihood: {ll:.2f}  (higher = more likely from this generator)")
```

## Files

* `model.py`     — `TinyGPT` and `TinyGPTConfig`
* `data.py`      — dynamic generator + block packing
* `evaluate.py`  — membership test (5 counter-example families)
* `train.py`     — CLI training loop with logging + checkpointing

## Design notes

* **Dynamic generation, bounded range.** Every batch is generated on the fly
  from a seeded RNG; the generator is fully deterministic given a seed, so any
  expression can be re-sampled and re-judged.
* **Random spacing.** Spacing is part of the distribution, not a formatting
  convention — the model must learn that 0..3 spaces around an operator are all
  equally likely.
* **CPU-first.** No CUDA, no fused kernels; `device="cpu"` everywhere. The
  model is small enough to train on a laptop in minutes.
* **Packed sequences.** Multiple `<BOS>...<EOS>` expressions are concatenated
  into one 1024-token block so the full context is exercised; the right tail is
  padded with `<EOS>` so the model never sees garbage.
