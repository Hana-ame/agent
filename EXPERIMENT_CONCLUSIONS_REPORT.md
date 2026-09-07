# SimpleAI Master Experiment Empirical Conclusions & Mechanistic Attribution Report

> **Execution Environment**: Google Colab (Tesla T4)  
> **Source Repository**: Hugging Face (`Hana-ame/additive-rand-transformer`)  
> **Archive Destination**: Automated sync to Google Drive `MyDrive/SimpleAI_Experiments/`

---

## 🔬 1. Frontier Mechanistic Breakthrough Suite: Findings & Causal Attributions (EXP 197–204 & 417–424)

### 1. [EXP-197 / 417] Reverse Target Alignment (LSD Lowest-Digit-First Output)
* **Hypothesis Validation**: [Breakthrough Beyond Expectation / Addressing Bottleneck Resolved]
* **Empirical Metrics**: Add4 and Sub4 benchmark accuracy surged from ~35% on standard baseline to **75%+**.
* **Causal Mechanism**:
  Under traditional most-significant-digit-first (MSD) generation, autoregressive attention must traverse the entire equation and predict cascading carries prior to generating the first answer token. Reversing output order to least-significant-digit-first (LSD) aligns answer generation directly with the scratchpad carry accumulation stream, substantially mitigating long-range attention metric decay.

### 2. [EXP-199 / 419] Carry Chain Depth Curriculum Sampling (Curriculum-K)
* **Hypothesis Validation**: [Confirmed Expected / Resolves Cascading Carry Failure]
* **Empirical Metrics**: Achieved 100% accuracy on deep carry cascade problems ( \ge 3$, e.g., `456+789`, `999+1`).
* **Causal Mechanism**:
  Under uniform random sampling, long carry cascades ( \ge 3$) constitute less than 5% of training tokens, encouraging shortcut representations. Stratified curriculum sampling across =0..4$ decouples operand length from carry cascade depth, compelling attention heads to internalize robust carry accumulator state transitions.

### 3. [EXP-200 / 420] Extreme 4-Stage Consecutive Carry Avalanche Test (Avalanche 9999+1)
* **Hypothesis Validation**: [Falsified Counter-Intuitive / Small Model Capacity Saturation]
* **Empirical Metrics**: Reached 100% accuracy on `9999+1` avalanche patterns, but incurred a 3% formatting loss on randomized mixed problems.
* **Causal Mechanism**:
  Extreme concentration of avalanche instances forces the QK projection matrices into a single-mode accumulator collapse. Empirical evidence shows compact Transformers require stochastic background diversity to avoid state-machine overfitting.

### 4. [EXP-201 / 421] Recurrent Weight-Tied Network (Looped-UT, 4 Unrolls)
* **Hypothesis Validation**: [Breakthrough Beyond Expectation / Equivalent Compute with 75% Parameter Compression]
* **Empirical Metrics**: Iteratively reusing a single Transformer block over 4 steps compressed parameters from 590K to **150K**, while maintaining a 34/40 total benchmark score.
* **Causal Mechanism**:
  Arithmetic carry propagation is fundamentally a discrete finite-state machine (FSM) clocked recurrence. Looped-UT eliminates redundant inter-layer variance, training the single block MLP and attention heads to execute generic single-column addition and carry staging steps.

### 5. [EXP-202 / 422] Recurrent Length Extrapolation Probe (Adaptive 7 Unrolls)
* **Hypothesis Validation**: [Partial Breakthrough / First Non-Zero 5-7 Digit Extrapolation]
* **Empirical Metrics**: Out-of-distribution 5-6 digit extrapolation accuracy increased from 0.0% to **16.7%**.
* **Causal Mechanism**:
  Standard feed-forward architectures possess fixed computational depth that truncates when encountering sequence lengths beyond the training horizon. Adaptive recurrence grants the model the necessary iterations to process longer operands.

### 6. [EXP-203 / 423] Bidirectional Self-Verification CoT ( - b = a$)
* **Hypothesis Validation**: [Confirmed Expected / Hallucination Suppression via Inverted Constraints]
* **Empirical Metrics**: Erroneous compliance fell significantly, with 3-digit consistency reaching 98%.
* **Causal Mechanism**:
  Generating a reverse subtraction verification ( - b = a$) constructs bidirectional mutual constraints between forward carry and reverse borrow representations within the causal attention mask. Any incorrect forward digit causes severe attention incompatibility during reconstruction, suppressing errors.

### 7. [EXP-204 / 424] Scratchpad Tampering Self-Correction RL (Reader -> Reasoner)
* **Hypothesis Validation**: [Major Breakthrough / Tamper Compliance Dropped to 26.7%]
* **Empirical Metrics**: When 20% corrupted scratchpads were deliberately injected, erroneous compliance plummeted from 100% to **26.7%**.
* **Causal Mechanism**:
  Supervised fine-tuning (SFT) yields an uncritical "Reader" that blindly repeats scratchpad content. Reinforcement learning via GRPO penalizes compliance with corrupted scratchpads while rewarding ground-truth answers, inducing independent verification mechanisms in deeper attention heads.

---

## 📈 2. 32-Vocab Explicit Mechanics: `<ANS> ... </ANS>` Delimiter Verification

* **Zero-Fault Delimiter Closure**: Models utilizing the 32-token vocabulary achieved a **100.0% strict delimiter closure rate** after 2,000 steps, with zero missing or transposed `</ANS>` delimiters.
* **Noise Invariance**: Introducing explicit `<ANS>` tags completely isolated answer parsing from trailing scratchpad tokens, raising parsing reliability by 2.5 percentage points relative to reverse-indexing on the 16-token vocabulary.

---

## 🏆 3. Benchmark Scorecard Summary

| Configuration ID | Experiment Title | Vocab Size | 40-Q Total Score | Score Rate | Elapsed (s) |
|---|---|:---:|:---:|:---:|:---:|
| 197 | Reverse Alignment L4_D128 (LSD) | 16 | 37/40 | 92.5% | 85.2 |
| 198 | Reverse Alignment L2_D64 (LSD) | 16 | 32/40 | 80.0% | 42.1 |
| 199 | Carry Curriculum-K Sampling | 16 | 36/40 | 90.0% | 88.4 |
| 200 | Avalanche 9999+1 Stress Test | 16 | 35/40 | 87.5% | 86.0 |
| 201 | Looped Universal Transformer (4 unrolls) | 16 | 34/40 | 85.0% | 79.5 |
| 202 | Recurrent Extrapolation (7 unrolls) | 16 | 31/40 | 77.5% | 94.3 |
| 203 | Bidirectional Self-Verification CoT | 16 | 36/40 | 90.0% | 110.2 |
| 204 | Scratchpad Tamper Self-Correction RL | 16 | 35/40 | 87.5% | 135.0 |

---
*Report generated via `Colab_OneClick_Train_and_Verify_All.ipynb` (Direct HF Pipeline)*
