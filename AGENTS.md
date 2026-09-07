# AGENTS.md — SimpleAI Experiment Execution & Handover Protocol

> **Target Audience**: Successor AI Agents / Autonomous Pair Programmers.  
> **Core Objective**: Advance and execute pending experiments (such as items 197–220), backfill empirical measurements, regenerate Excel master ledgers, and author mechanistic attribution analyses.

---

## 🚨 Strict Prohibition (NEVER VIOLATE)

```
=============================================================================
  NEVER RUN ANY EXPERIMENTS ON THIS LOCAL MACHINE!
=============================================================================
```

1. **Strictly Prohibited**: Executing any training, evaluation, parameter sweeping, or heavy inference scripts on the local host (including but not limited to `train.py`, `evaluate.py`, `rl.py`, `arch_sweep.py`, `use_model.sh`, etc.).
2. **Local Machine Responsibilities**: Confined exclusively to code inspection, configuration file generation/editing, markdown documentation maintenance, executing `expand_and_update_all_excels.py` to compile workbooks, and staging Git commits.
3. **Execution Environment**: All compute-intensive training and evaluation workloads **must be executed exclusively on Google Colab** (GPU/CPU runtimes) or external cloud compute clusters.

---

## 🗺️ Project Topology & Key Assets

| Resource File | Relative Path | Role & Purpose |
|---|---|---|
| **Master Ledger** | `archive/ALL_DOCS_EXPERIMENTS_CONFIG_TO_RESULTS.xlsx` | Unified 49-column wide ledger archiving all additive and maze experiments. |
| **Additive Ledger** | `ADDITIVE_EXPERIMENTS_ALL.xlsx` | Dedicated additive panorama matrix tracking 18 method flags and granular metrics. |
| **Maze Ledger** | `MAZE_EXPERIMENTS_ALL.xlsx` | Dedicated maze navigation matrix tracking RL configurations and reach rates. |
| **Experiment Config Pool** | `additive-rand-transformer/configs/` | Standard launch configurations for experiments. |
| **Colab Batch Notebook** | `Colab_OneClick_Train_and_Verify_All.ipynb` | One-click sequential training and 40-question evaluation notebook. |
| **Excel Compiler Engine** | `expand_and_update_all_excels.py` | Local script to recompile workbooks and synchronize JSON configurations. |

---

## 🎯 Task Pool & Priorities

### 1. Highest Priority: Frontier Mechanistic Breakthrough Suite (EXP 197–204) 🌟🌟🌟

Key breakthrough probes targeting 4-digit overflow barriers, length extrapolation, and scratchpad reader limitations:

* **[197] Reverse Output Alignment L4_D128** (`configs/197_l4_d128_lsd.json`):
  * **Hypothesis**: Outputting answers in least-significant-digit (LSD) order removes reverse addressing overhead, elevating `add4` accuracy from ~35% to 75%+.
* **[198] Reverse Output Alignment L2_D64** (`configs/198_l2_d64_lsd.json`):
  * **Hypothesis**: Lightweight probe testing if low-digit alignment enables a 160K parameter model to resolve 3-digit carries earlier.
* **[199] Carry Curriculum Sampling** (`configs/199_k_0_4.json`):
  * **Hypothesis**: Stepped curriculum across consecutive carry count =0..4$ decouples digit length from carry cascade depth, resolving failure modes on  \ge 3$.
* **[200] Extreme 4-Stage Avalanche Test** (`configs/200_4_9999_1_100.json`):
  * **Hypothesis**: 100% exposure to 999+1$ continuous cascade carries to test small Transformer accumulator saturation boundaries.
* **[201] Looped Weight-Tied Network Looped-UT (4 Unrolls)** (`configs/201_looped-ut_block_4.json`):
  * **Hypothesis**: A single block reused iteratively across 4 steps compresses parameters by 75% while validating recursive algorithmic state-machine execution.
* **[202] Recurrent Length Extrapolation Probe (Adaptive 7 Unrolls)** (`configs/202_7.json`):
  * **Hypothesis**: Adaptively unrolling 7 steps on 5–7 digit problems breaks the 0.0% out-of-distribution extrapolation ceiling.
* **[203] Bidirectional Self-Verification CoT** (`configs/203_cot_c_c-b_a.json`):
  * **Hypothesis**: Appending reverse verification  - b = a$ establishes bidirectional constraints between forward carry and reverse borrow graphs.
* **[204] Tampered Scratchpad Self-Correction RL** (`configs/204_reader.json`):
  * **Hypothesis**: Injecting 20% corrupted scratchpads and rewarding correct final answers reduces erroneous scratchpad compliance from 100% to under 30% (Reader -> Reasoner).

### 2. Second Priority: Training Step Scaling Sweep (EXP 205–220) 🌟🌟

Systematic evaluation of compute scaling on the standard `L4_D128 CoT` recipe:
* **Micro-step Early Probes**: `205` (20 steps), `206` (50), `207` (100), `208` (200), `209` (500)
* **Mid-range Convergence Baselines**: `210` (1,000 steps), `211` (2,000), `212` (4,000 standard baseline)
* **High-Compute Long-horizon Scaling**: `213` (8,000 steps), `214` (16,000), `215` (32,000), `216` (64,000), `217` (128,000), `218` (256,000), `219` (512,000), `220` (1,024,000 steps)

---

## 💻 Google Colab Standard Operating Procedure (SOP)

### Step 1: Open Notebook in Colab
Open [`Colab_Run_Additive_Transformer.ipynb`](Colab_Run_Additive_Transformer.ipynb) or [`Colab_OneClick_Train_and_Verify_All.ipynb`](Colab_OneClick_Train_and_Verify_All.ipynb).

### Step 2: Set Up Workspace
In the initial notebook cell:
```python
from google.colab import drive
drive.mount('/content/drive')

!git clone https://github.com/Hana-ame/agent.git /content/workspace
%cd /content/workspace/additive-rand-transformer
!pip install -q torch openpyxl huggingface_hub pandas matplotlib
```

### Step 3: Execute Target Experiments

#### Option A: Single Configuration Run
```bash
python -m additive_rand_transformer.train --config configs/197_l4_d128_lsd.json
```

#### Option B: Sequential Batch Execution
```python
from additive_rand_transformer.batch_train import run_batch_experiments
reports = run_batch_experiments(run_mode="FRONTIER_197_204", max_experiments=8)
```

### Step 4: Record Output Metrics
Extract metrics from the evaluation summary:
```text
Final Evaluation Metrics:
  Loss: 0.1820
  Add Acc: [add1 100.0% | add2 100.0% | add3 93.3% | add4 76.7%]
  Sub Acc: [sub1 100.0% | sub2 100.0% | sub3 96.7% | sub4 90.0%]
  Extrapolate Acc: [add5 16.7% | sub5 13.3% | add6 0.0% | sub6 0.0%]
  Unique Exprs (RL): 58/60
  Elapsed Seconds: 412.5s
```

---

## 📊 Local Result Backfill & Excel Synchronization SOP

Once empirical metrics are obtained from Colab:

### 1. Update Generator Script
In `expand_and_update_all_excels.py`, locate `new_designed_items` or step rows.

### 2. Backfill Measurements & Attribution
Replace pending placeholders with quantitative values and provide mechanistic explanations:
```python
("EXP-REV-01", "[Frontier] Reverse Alignment L4_D128 (LSD output)", 4, 128, 4000, 32, "3e-4",
 ["SFT Supervised", "CoT Column", "4-Digit Biased", "Single Sample"],
 "100.0%", "100.0%", "93.3%", "76.7%", "100.0%", "100.0%", "96.7%", "90.0%", "—", "0.1820",
 "[Breakthrough Beyond Expectation] Reversing answer order to least-significant-digit first unlocked an add4 jump from 35% to 76.7%. Eliminates long-range backward addressing decay.")
```

### 3. Recompile Workbooks and Configurations
Run locally:
```bash
python3 expand_and_update_all_excels.py
```

### 4. Git Commit
```bash
git add archive/ ADDITIVE_EXPERIMENTS_ALL.xlsx MAZE_EXPERIMENTS_ALL.xlsx additive-rand-transformer/
git commit -m "feat(experiments): record completed runs and update master excel"
```

---

## ✍️ Mechanistic Attribution Writing Standards

When composing the `conclusion` field:
1. **Explicit Hypothesis Validation**: Begin with `[Confirmed Expected]`, `[Breakthrough Beyond Expectation]`, or `[Falsified Counter-Intuitive]`.
2. **Causal Attribution**: Deliver rigorous physical/algorithmic explanations (e.g., attention metric shifts, accumulator saturation, finite-state recurrence vs template pattern reading).
