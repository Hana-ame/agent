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
pip install torch --index-url https://download.pytorch.org/whl/cpu

# 50-step smoke test (no checkpoint)
python -m additive_rand_transformer.train --quick

# full run, checkpoint saved to runs/<timestamp>/checkpoint_final.pt
python -m additive_rand_transformer.train --steps 3000
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
