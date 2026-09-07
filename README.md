# SimpleAI — Experiments & Training Comprehensive Guide (Google Colab & Cloud Pipelines)

> This document serves as the **unified training and operations manual** for the SimpleAI research workspace.
> 
> 💡 **Repository architecture, Git LFS protocols, 7 baseline checkpoint weight metadata, and legacy technical notes have been fully consolidated into the Excel master workbook under the `[Repo_Architecture_Legacy_Notes]` sheet.**

---

## Quick Navigation & Core Assets

| Resource | Path | Description |
|---|---|---|
| **Additive Experiments Master** | [`ADDITIVE_EXPERIMENTS_ALL.xlsx`](ADDITIVE_EXPERIMENTS_ALL.xlsx) | Dedicated additive workbook: 220 experiments, 18 method tags, and granular quantitative metrics |
| **Maze Experiments Master** | [`MAZE_EXPERIMENTS_ALL.xlsx`](MAZE_EXPERIMENTS_ALL.xlsx) | Dedicated maze workbook: 10 pure RL experiments and navigation success rates |
| **Unified Master Archive** | [`archive/ALL_DOCS_EXPERIMENTS_CONFIG_TO_RESULTS.xlsx`](archive/ALL_DOCS_EXPERIMENTS_CONFIG_TO_RESULTS.xlsx) | Historical cross-module unified 49-column wide archive sheet |
| **Agent Handover Protocol** | [`AGENTS.md`](AGENTS.md) | **Zero local compute rule**, 197–220 unrun experiment pool, Colab SOP, and data backfill standards |
| **Additive Colab Batch Pipeline** | [`Colab_OneClick_Train_and_Verify_All.ipynb`](Colab_OneClick_Train_and_Verify_All.ipynb) | One-click sequential batch training & 40-benchmark evaluation (robust, zero-crash stream) |
| **Additive Colab Single Run** | [`Colab_Run_Additive_Transformer.ipynb`](Colab_Run_Additive_Transformer.ipynb) | Interactive training, mechanistic probes, dynamic INT8 quantization, and Google Drive sync |
| **Maze Colab Manual** | [`Colab_Run_Maze_Transformer.ipynb`](Colab_Run_Maze_Transformer.ipynb) | Reactive 2D maze pure RL (GRPO) training and visualization |
| **Additive Trainer** | [`additive-rand-transformer/additive_rand_transformer/train.py`](additive-rand-transformer/additive_rand_transformer/train.py) | CLI trainer supporting `train.py --config config.json` |
| **Maze Trainer** | [`maze-transformer/maze_transformer/train.py`](maze-transformer/maze_transformer/train.py) | CLI trainer supporting `train.py --config maze_config.json` |
| **Quantization Evaluator** | [`additive-rand-transformer/additive_rand_transformer/quantize.py`](additive-rand-transformer/additive_rand_transformer/quantize.py) | PyTorch dynamic INT8 quantization, compression ratio, and throughput benchmarks |

---

## Method 1: Training on Google Colab (Recommended 🌟)

Run experiments on free/cloud GPUs/CPUs without consuming local compute, with automatic bidirectional sync to Google Drive:

### 1. Open Notebook
- In Google Cloud Shell or terminal:
  ```bash
  cloudshell download Colab_Run_Additive_Transformer.ipynb
  # or for the maze navigation notebook:
  cloudshell download Colab_Run_Maze_Transformer.ipynb
  ```
- Navigate to [Google Colab](https://colab.research.google.com/) -> Click **Upload** -> Select the `.ipynb` file.

### 2. One-Click Execution Pipeline
1. **Mount Google Drive**: Automatically connects to `/MyDrive/simpleAI_workspace/` so model checkpoints and logs persist across sessions.
2. **Environment & Checkpoint Setup**: Automatically downloads dependencies and official baseline pretrained weights (`.pt`).
3. **Customize `config.json`**: Edit depth, width, training steps, and datasource in the code cell.
4. **Launch Training**: Monitor real-time loss decay curves and the unlocking of 1–4 digit addition/subtraction capabilities.
5. **Mechanistic Diagnostics & Quantization**: Run H1 scratchpad reading probes and lossless dynamic INT8 quantization.
6. **Interactive REPL**: Input `1234 + 5678` to inspect step-by-step column-by-column reasoning.
7. **Automated Drive Backup**: Automatically archive the latest weights back to Google Drive.

---

## Method 2: Training via CLI (Remote Server / Cloud VM)

### 1. Write `config.json`
Define architecture, hyperparameters, and data generation settings via a JSON configuration:

```json
{
  "layers": 4,                  
  "d": 128,                     
  "heads": 4,                   
  "steps": 4000,                
  "batch_size": 32,             
  "lr": 3e-4,                   
  "wd": 0.1,                    
  "warmup": 200,                
  "datasource": {
    "type": "cot",              
    "max_digits": 4,            
    "bias": 0.5,                
    "max_spaces": 3,            
    "single": true              
  }
}
```

#### Native Parameter Alias Reference:
| JSON Field Alias | Internal Parameter | Description |
|---|---|---|
| `layers`, `layer`, `n_layers`, `num_layers` | `n_layer` | Model depth L (1-10) |
| `d`, `dim`, `d_model`, `embed_dim`, `width` | `n_embd` | Hidden channel width d (32-512) |
| `heads`, `head`, `n_heads` | `n_head` | Number of attention heads |
| `batch_size`, `bs`, `batch` | `batch_size` | Batch size |
| `steps`, `train_steps`, `max_steps`, `epochs` | `steps` | Training steps |
| `datasource.type: "cot"` / `"plain"` | `cot: True / False` | Enable Chain-of-Thought column scratchpad |
| `datasource.bias` | `four_digit_bias` | Weight ratio for difficult 4-digit pairs (0.5 is optimal) |
| `datasource.single` | `single` | Single-sequence training mode (unpacked) |

---

### 2. Run Training Commands

- **Additive Arithmetic Probe (Configuration-driven)**:
  ```bash
  cd additive-rand-transformer
  python -m additive_rand_transformer.train --config configs/001_transformer_base.json
  ```
  *(To run a 50-step quick smoke test: `python -m additive_rand_transformer.train --quick`)*

- **Maze Reactive Navigation (Pure RL GRPO Training)**:
  ```bash
  cd maze-transformer
  python -m maze_transformer.train --config configs/001_transformer_grpo.json
  ```

---

## Monitoring Training Metrics

During training, real-time metrics are logged to stdout every 25 steps:
```text
step   100 | loss 1.2140 | lr 1.50e-04 | cot_acc [add1 100% | add2 100% | add3 90% | add4 30% | sub1 100% | sub2 100% | sub3 95% | sub4 40%] | 12.3s
```

### Key Metrics & Capability Phase Transitions:
1. **`loss`**: Cross-entropy loss; normal convergence descends steadily from `2.50+` to approximately `0.17`.
2. **`cot_acc` Capability Phase Transitions**:
   - **L=1**: Captures only 1-digit addition and rudimentary 2-digit patterns.
   - **L=2**: Stably masters `add1 100%` and `add2 100%`.
   - **L=3 (Phase Transition 1)**: Unlocks `add3 93%`.
   - **L=4 (Phase Transition 2)**: Breakthrough in `sub4 96%` and `add4 35%` (completing multi-digit carry and borrow loops).
3. **Maze `solvability`**:
   - Under Transformer + GRPO within 120 steps, goal reach rate improves from 0% to **83.3%**, with wall-collision steps dropping from 50+ to 11.

---

## Post-Training Evaluation, Quantization & Interaction

### 1. Interactive Inference (REPL)
```bash
cd additive-rand-transformer
./use_model.sh -s "1234 + 5678"
./use_model.sh -s "9999 - 4321"
```

### 2. Dynamic INT8 Post-Training Quantization
Measure model compression, inference acceleration, and accuracy retention:
```bash
python -m additive_rand_transformer.quantize --checkpoint checkpoints/l4_d128_cot_bias05_final.pt
```
- **Quantization Results**: **3.8x** size compression (1.7MB -> 0.45MB), **1.4x** throughput speedup, with 1–4 digit accuracy **fully maintained at 100% (zero degradation)**.

### 3. Mechanistic Probe (H1 Scratchpad Tamper Test)
```bash
python -m additive_rand_transformer.explore_h1
```
- **Key Finding**: During answer generation, 88.7% of attention attends directly to intermediate sum columns, with 0% sensitivity to initial raw operands. When scratchpads are tampered with, the model conforms 100% to erroneous scratchpads, proving standard CoT acts as a Reader rather than an autonomous Reasoner.

---

## Archive Location

All legacy documentation, detailed configurations for the 7 baseline checkpoints, Git LFS, and cloud mirror specifications are preserved in the master workbook:
👉 Open [`archive/ALL_DOCS_EXPERIMENTS_CONFIG_TO_RESULTS.xlsx`](archive/ALL_DOCS_EXPERIMENTS_CONFIG_TO_RESULTS.xlsx) under the **`[Repo_Architecture_Legacy_Notes]`** worksheet.
