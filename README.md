# SimpleAI — 实验与训练完整使用指南 (Google Colab & 本地极速流水线)

> 本文档为 SimpleAI 研究工作区的**统一训练与操作手册**。
> 
> 💡 **原仓库架构、Git LFS 规范、7大 Checkpoint 权重元数据及历史说明已完整并入 Excel 工作簿之 `【仓库架构与项目说明_原README】` Sheet 中。**

---

## 快速导航与核心资产

| 资源 | 文件路径 | 说明 |
|---|---|---|
| **Excel 实验全景总库** | [`ALL_DOCS_EXPERIMENTS_CONFIG_TO_RESULTS.xlsx`](ALL_DOCS_EXPERIMENTS_CONFIG_TO_RESULTS.xlsx) | **全景宽表**：覆盖 220 项加法实验与 10 项迷宫实验，18 项训练方式打勾与细分定量指标 |
| **接棒 Agent 操作手册** | [`AGENT_EXPERIMENT_EXECUTION_GUIDE.md`](AGENT_EXPERIMENT_EXECUTION_GUIDE.md) | **零本地运行红线**、197–220待跑实验任务池、Colab 执行 SOP 与指标回填规范 |
| **加法 Colab 手册** | [`Colab_Run_Additive_Transformer.ipynb`](Colab_Run_Additive_Transformer.ipynb) | Google Colab 一键训练、探针诊断、INT8量化与 Drive 自动同步 |
| **迷宫 Colab 手册** | [`Colab_Run_Maze_Transformer.ipynb`](Colab_Run_Maze_Transformer.ipynb) | 反应式 2D 迷宫纯 RL (GRPO) 导航一键训练与可视化 |
| **加法训练入口** | [`additive-rand-transformer/additive_rand_transformer/train.py`](additive-rand-transformer/additive_rand_transformer/train.py) | 支持 `train.py --config config.json` 灵活拉起 |
| **迷宫训练入口** | [`maze-transformer/maze_transformer/train.py`](maze-transformer/maze_transformer/train.py) | 支持 `train.py --config maze_config.json` 灵活拉起 |
| **模型量化评测** | [`additive_rand_transformer/quantize.py`](additive-rand-transformer/additive_rand_transformer/quantize.py) | PyTorch 动态 INT8 量化、体积压缩比与推理吞吐基准 |

---

## 方式一：在 Google Colab 上训练（推荐 🌟）

无需占用本地算力，直接利用 Colab 免费 GPU/CPU，全自动与 Google Drive 双向同步：

### 1. 打开 Notebook
- 在 Cloud Shell 中下载 Notebook：
  ```bash
  cloudshell download Colab_Run_Additive_Transformer.ipynb
  # 或迷宫 Notebook：
  cloudshell download Colab_Run_Maze_Transformer.ipynb
  ```
- 访问 [Google Colab](https://colab.research.google.com/) -> 点击 **上传 (Upload)** -> 选择该 `.ipynb` 文件打开。

### 2. 一键运行全流程（点击左侧 ▶ 按钮）
1. **自动挂载 Google Drive**：连接 `/MyDrive/simpleAI_workspace/`，模型权重与日志断连不丢失。
2. **环境与权重准备**：自动拉取依赖与官方基准预训练权重 (`.pt`)。
3. **自定义 `config.json`**：在代码块中自由修改层数、宽度、步数与数据源。
4. **启动训练**：实时查看 Loss 下降曲线与 1–4 位加减法解锁过程。
5. **机制诊断与量化**：运行 H1 草稿纸读取机制探针与 INT8 动态量化无损验证。
6. **交互式求解 (REPL)**：输入 `1234 + 5678` 实时查看逐位竖式推理过程。
7. **自动备份回 Drive**：一键将最新权重同步保存至 Google Drive。

---

## 方式二：在本地 / 服务器命令行直接训练

### 1. 编写配置文件 `config.json`
通过 JSON 字典定义实验架构、超参以及数据源构建方式（支持各种常见别名）：

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

#### 原生参数别名映射表：
| JSON 字段别名 | 映射内部参数 | 作用说明 |
|---|---|---|
| `layers`, `layer`, `n_layers`, `num_layers` | `n_layer` | 模型深度 L (1-10) |
| `d`, `dim`, `d_model`, `embed_dim`, `width` | `n_embd` | 隐藏通道宽度 d (32-512) |
| `heads`, `head`, `n_heads` | `n_head` | 注意力头数 |
| `batch_size`, `bs`, `batch` | `batch_size` | 批量大小 |
| `steps`, `train_steps`, `max_steps`, `epochs` | `steps` | 训练步数 |
| `datasource.type: "cot"` / `"plain"` | `cot: True / False` | 是否启用思维链竖式草稿纸 |
| `datasource.bias` | `four_digit_bias` | 4 位高难度双操作数加权比例 (0.5 为黄金甜点) |
| `datasource.single` | `single` | 单样本训练模式（无打包） |

---

### 2. 拉起训练命令

- **【加法算术探针】（基于配置文件）**：
  ```bash
  cd additive-rand-transformer
  python -m additive_rand_transformer.train --config my_config.json
  ```
  *(如需 50 步极速冒烟测试验证环境，可加 `--quick`：`python -m additive_rand_transformer.train --quick`)*

- **【迷宫反应式导航】（纯 RL GRPO 训练）**：
  ```bash
  cd maze-transformer
  python -m maze_transformer.train --config maze_config.json
  ```

---

## 三、训练过程指标监控指南

训练启动后，终端每 25 步输出一次实时指标：
```text
step   100 | loss 1.2140 | lr 1.50e-04 | cot_acc [add1 100% | add2 100% | add3 90% | add4 30% | sub1 100% | sub2 100% | sub3 95% | sub4 40%] | 12.3s
```

### 关键指标与机制相变阶段：
1. **`loss`**：交叉熵损失，正常收敛由 `2.50+` 稳步下降至 `0.17` 左右。
2. **`cot_acc` 能力阶梯相变**：
   - **L=1**：仅掌握 1 位加法与粗糙 2 位加法。
   - **L=2**：稳定掌握 `add1 100%`、`add2 100%`。
   - **L=3（相变点 1）**：突破 `add3 93%`。
   - **L=4（相变点 2）**：突破 `sub4 96%` 与 `add4 35%`（完成多位进位与借位闭环）。
3. **迷宫 `solvability`**：
   - Transformer + GRPO 在 120 步内到达率由 0% 跃升至 **83.3%**，撞墙步数由 50+ 骤降至 11。

---

## 四、训练完成后的评估、量化与互动

### 1. 单题求解与交互式推理 (REPL)
```bash
cd additive-rand-transformer
./use_model.sh -s "1234 + 5678"
./use_model.sh -s "9999 - 4321"
```

### 2. 模型量化评测 (FP32 -> Dynamic INT8)
测量量化压缩比、推理吞吐加速与精度留存率：
```bash
python -m additive_rand_transformer.quantize --checkpoint checkpoints/l4_d128_cot_bias05_final.pt
```
- **量化实测结论**：体积压缩 **3.8x** (1.7MB $\to$ 0.45MB)，推理加速 **1.4x**，1–4 位加减法准确率 **100% 保持（零退化）**！

### 3. 学术机制探针 (H1 草稿纸篡改测试)
```bash
python -m additive_rand_transformer.explore_h1
```
- **核心结论**：答案生成阶段 88.7% 依赖读取中间和列，对初始操作数敏感度为 0%；草稿被篡改时 100% 顺从错误，证明 CoT 仅为 Reader 而非 Reasoner。

---

## 五、原 README 内容归档位置

原 README 中的全部历史说明、7 个 Checkpoint 详细配置、Git LFS 与镜像环境配置，已全部整理并入 Excel 工作簿：
👉 请打开 [`ALL_DOCS_EXPERIMENTS_CONFIG_TO_RESULTS.xlsx`](ALL_DOCS_EXPERIMENTS_CONFIG_TO_RESULTS.xlsx) 之 **`【仓库架构与项目说明_原README】`** 工作表进行查阅。
