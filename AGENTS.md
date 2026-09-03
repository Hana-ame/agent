# AGENTS.md — simpleAI 实验执行与交付指南 (Agent Handover Protocol)

> **面向对象**：后续接手此仓库的 AI Agent / Pair Programmer。  
> **核心任务**：推进并完成加法 Transformer 的未跑实验（序号 197–220），回填实测数据，更新 Excel 总库，并完成机制归因分析。

---

## 🚨 零号红线原则 (STRICT PROHIBITION - NEVER VIOLATE)

```
=============================================================================
  本地绝对不要跑任何实验！ (NEVER RUN ANY EXPERIMENTS ON THIS LOCAL MACHINE)
=============================================================================
```

1. **绝对禁止**在当前本地机器上执行任何训练、评测、参数扫描或重推理命令（包括但不限于 `train.py`, `evaluate.py`, `rl.py`, `arch_sweep.py`, `use_model.sh` 等）。
2. **本地环境职责**：仅允许进行代码检查、生成/修改配置文件、更新 Markdown 文档、执行 `expand_and_update_all_excels.py` 生成 Excel 工作簿，以及进行 Git 提交。
3. **实验执行环境**：所有需要消耗算力的训练与评测，**必须且只能在 Google Colab**（免费/付费 GPU/CPU）或独立云端集群上运行。

---

## 🗺️ 项目实验拓扑与关键文件

| 资源文件 | 绝对/相对路径 | 角色与用途 |
|---|---|---|
| **全实验总台账** | `ALL_DOCS_EXPERIMENTS_CONFIG_TO_RESULTS.xlsx` | 49 列超大宽表，统一记录加法（001–220）与迷宫（001–010）全部实验。 |
| **加法实验子表** | `additive-rand-transformer/EXPERIMENTS_ALL.xlsx` | 加法专用的单表全景矩阵，包含 18 项打勾方式与细分指标。 |
| **实验配置池** | `additive-rand-transformer/configs/` | 包含 `001_...json` 到 `220_...json` 的标准启动配置文件。 |
| **Colab 批量实测手册** | `Colab_OneClick_Train_and_Verify_All.ipynb` | 一键顺序批量训练与 40 题严格评测 Notebook（稳健防崩溃）。 |
| **Excel 编译引擎** | `expand_and_update_all_excels.py` | 本地执行，用于自动重新编译生成上述全部 Excel 表格与 JSON 配置。 |

---

## 🎯 待完成任务清单 (Unrun Experiments Pool)

总库共包含 220 项加法实验，其中 **001–196 为已完成实验**，已具备定量指标。  
**你的核心任务是完成以下 24 项待运行 (unrun) 实验**：

### 1. 优先级最高：前沿机制突破实验组（序号 197–204）🌟🌟🌟

这 8 个实验是探索“4位溢出阻滞”、“长度外推为0%”、“顺从错误草稿 Reader 局限”的核心突破口：

* **[197] 逆序目标对齐 L4_D128** (`configs/197_l4_d128_lsd.json`):
  * **假说**：将答案改为低位优先输出（如 `975` 代替 `579`），消解反向寻址开销。重点观察 `add4` 是否突破 45% 跃升至 80%+。
* **[198] 逆序目标对齐 L2_D64** (`configs/198_l2_d64_lsd.json`):
  * **假说**：轻量级模型低位对齐探针，检验能否让 16 万参数模型提前解锁 3 位进位。
* **[199] 进位链深度课程采样** (`configs/199_k_0_4.json`):
  * **假说**：按连续进位数 $K=0..4$ 阶梯采样，剥离“位数”与“进位级联深度”的干扰，解决 $K \ge 3$ 的级联进位失效。
* **[200] 极端 4 级连续进位雪崩测试** (`configs/200_4_9999_1_100.json`):
  * **假说**：100% 覆盖 $9999+1$ 类连续雪崩进位，检验极小 Transformer 进位累加器的极限饱和。
* **[201] 循环权重共享网络 Looped-UT (展开4步)** (`configs/201_looped-ut_block_4.json`):
  * **假说**：单 Block 权重循环复用 4 步，参数压缩 75%，验证算法状态机递归能力。
* **[202] 循环网络长度外推探针 (自适应展开7步)** (`configs/202_7.json`):
  * **假说**：UT 模型在测试 5–7 位题目时自适应迭代 7 步，检验能否首次打破 5–7 位外推 0.0% 的僵局。
* **[203] 正反双向自验算 CoT 验证器** (`configs/203_cot_c_c-b_a.json`):
  * **假说**：输出答案后反算 $c - b = a$，通过自注意力建立正向进位图与反向借位图的双向约束。
* **[204] 草稿篡改自纠错强化学习** (`configs/204_reader.json`):
  * **假说**：注入 20% 错误草稿并给纠错加分，重点观察顺从错误草稿率是否由 100% 降至 30% 以下（从 Reader 迈向 Reasoner）。

### 2. 优先级次高：训练步数跨数量级缩放扫描（序号 205–220）🌟🌟

在 `L4_D128 CoT` 标准配方下，验证训练步数对损失下探与算术泛化的极限影响：
* **微步数初期探针**：
  * `205` (20步), `206` (50步), `207` (100步), `208` (200步), `209` (500步)
* **中期对齐与标准收敛**：
  * `210` (1,000步), `211` (2,000步), `212` (4,000步标准基线)
* **长程与大算力规模化**：
  * `213` (8,000步), `214` (16,000步), `215` (32,000步), `216` (64,000步), `217` (128,000步), `218` (256,000步), `219` (512,000步), `220` (1,024,000步 百万步极限)

---

## 💻 Google Colab 执行标准作业程序 (SOP)

由于本地严禁运行实验，请引导用户或通过云端脚本按以下流程在 Colab 上执行：

### Step 1: 在 Colab 打开 Notebook
打开父仓库根目录下的 [`Colab_Run_Additive_Transformer.ipynb`](Colab_Run_Additive_Transformer.ipynb)。

### Step 2: 准备环境与拉取最新代码
在 Colab 代码单元格中执行：
```python
# 挂载 Google Drive 并克隆工作区
from google.colab import drive
drive.mount('/content/drive')

!git clone https://github.com/Hana-ame/agent.git /content/workspace
%cd /content/workspace/additive-rand-transformer
!pip install -q torch openpyxl huggingface_hub pandas matplotlib
```

### Step 3: 根据配置文件拉起指定实验

#### 选项 A：单实验串行调试（针对特定机制深度跟踪）
使用 `train.py` 直接读取 `configs/` 下对应的 JSON 配置文件：
```bash
# 示例 1：运行 197 号前沿机制突破实验
python -m additive_rand_transformer.train --config configs/197_l4_d128_lsd.json

# 示例 2：运行 201 号循环权重共享实验
python -m additive_rand_transformer.train --config configs/201_looped-ut_block_4.json
```

#### 选项 B：🚀 一键批量顺序实测（强烈推荐，单实验 8~10 秒，零崩溃）
直接打开 [`Colab_OneClick_Train_and_Verify_All.ipynb`](Colab_OneClick_Train_and_Verify_All.ipynb) 或在 Python 中调用批量引擎：
```python
from additive_rand_transformer.batch_train import run_batch_experiments

# 纯顺序流式执行，8 个实验约 80 秒跑完并现场严格评测 40 题
reports = run_batch_experiments(run_mode="FRONTIER_197_204", max_experiments=8)
```

### Step 4: 记录产出指标
训练完成后，训练器会自动调用测试集评测，输出如下格式日志：
```text
Final Evaluation Metrics:
  Loss: 0.1820
  Add Acc: [add1 100.0% | add2 100.0% | add3 93.3% | add4 76.7%]
  Sub Acc: [sub1 100.0% | sub2 100.0% | sub3 96.7% | sub4 90.0%]
  Extrapolate Acc: [add5 16.7% | sub5 13.3% | add6 0.0% | sub6 0.0%]
  Unique Exprs (RL): 58/60
  Elapsed Seconds: 412.5s
```
将这批测试指标收集并准备回填。

---

## 📊 本地回填结果与更新 Excel SOP

当你在 Colab 获得了实验指标后，回到本本地仓库进行数据同步（**仅执行构建代码，不跑模型！**）：

### 1. 修改本地构建脚本
打开 [`expand_and_update_all_excels.py`](expand_and_update_all_excels.py)，定位到 `new_designed_items`（前沿设计实验）或 `NEW_STEP_ROWS`（步数扫描）。

### 2. 填入实测数字并撰写机制归因
将对应项的 `"未跑"` 替换为真实数字，并撰写符合学术标准的归因分析：
```python
# 示例：回填 197 实验
("EXP-REV-01", "【机制突破】逆序目标对齐 L4_D128 (低位LSD优先输出)", 4, 128, 4000, 32, "3e-4",
 ["SFT监督", "CoT竖式", "4位加权", "单样本Single"],
 "100.0%", "100.0%", "93.3%", "76.7%", "100.0%", "100.0%", "96.7%", "90.0%", "—", "0.1820",
 "【超预期突破/成功破除寻址瓶颈】将答案调整为低位优先后，add4 准确率由原基线的 35% 跃升至 76.7%！实证表明高位进位丢失的主要根源在于长程反向寻址注意力衰减。")
```

### 3. 一键编译并刷新所有 Excel 与配置
在本地工作区根目录下执行：
```bash
uv run --with openpyxl python3 expand_and_update_all_excels.py
```
**编译引擎会自动完成**：
* 自动在 `ALL_DOCS_EXPERIMENTS_CONFIG_TO_RESULTS.xlsx` 中将该行底色由黄色（`未跑`）变为绿色（高准确率完成）；
* 自动同步更新 `additive-rand-transformer/EXPERIMENTS_ALL.xlsx`；
* 自动将 `additive-rand-transformer/configs/{seq_id}_...json` 中的 `"status"` 由 `"unrun"` 切换为 `"completed"`。

### 4. Git 提交成果
```bash
git add ALL_DOCS_EXPERIMENTS_CONFIG_TO_RESULTS.xlsx additive-rand-transformer/ configs/
git commit -m "feat(experiments): complete run 197-204 and update master excel"
```

---

## ✍️ 机制归因写作规范 (Conclusion Quality Standards)

在填写 `conclusion`（实测现象记载与机制归因）字段时，请严格遵守以下两点要求：
1. **必须显式检验假说（Hypothesis Check）**：使用 `【符合预期】`、`【超预期突破】` 或 `【反直觉证伪】` 作为开头。
2. **必须给出因果机制解释（Causal Explanation）**：拒绝单纯堆砌数字，必须解释“为什么会这样”（如：注意力度量偏移、进位累加器连续饱和、草稿纸模板未泛化为状态机等）。
