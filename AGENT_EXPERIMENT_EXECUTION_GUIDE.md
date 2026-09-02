# 🤖 simpleAI 实验执行与交付指南 (给接棒 Agent 的操作手册)

> **致接棒 Agent / 协作者**：  
> 本文档定义了在 `simpleAI` (父仓库) 与 `additive-rand-transformer` (子仓库) 中**执行、监控并回收未跑实验**的统一标准操作流程 (SOP)。  
> 请务必在开始工作前完整阅读本规范，特别是第一条**红线原则**。

---

## ⛔ 零号红线原则 (STRICT PROHIBITION)

> 🚨 **绝对禁止在本地机器 / 本地命令行运行任何模型训练、评估、扫描或推理实验！**  
> 
> * **本地环境定位**：仅用于**代码编写、文件审查、实验配置生成 (`configs/*.json`)、Excel 报表更新与 Git 版本维护**。  
> * **算力执行定位**：所有模型训练、评估与评测必须且只能在 **Google Colab**（免费/付费 GPU/CPU）或指定的云端环境中执行。

---

## 🗺️ 项目实验中枢与核心资产地图

| 资产名称 | 文件路径 | 作用说明 |
|---|---|---|
| **全实验总台账** | [`ALL_DOCS_EXPERIMENTS_CONFIG_TO_RESULTS.xlsx`](ALL_DOCS_EXPERIMENTS_CONFIG_TO_RESULTS.xlsx) | 49 列超大宽表，严格按 `001`–`220` 编号记录全部 220 项加法实验与 10 项迷宫实验。 |
| **加法实验子库** | [`additive-rand-transformer/EXPERIMENTS_ALL.xlsx`](additive-rand-transformer/EXPERIMENTS_ALL.xlsx) | 专供加法探针的单表全景矩阵，含 18 项打勾方式与细分指标。 |
| **实验配置集** | [`additive-rand-transformer/configs/`](additive-rand-transformer/configs/) | 包含 `001_...json` 至 `220_...json` 的标准化 JSON 启动配置文件。 |
| **加法 Colab 笔记本** | [`Colab_Run_Additive_Transformer.ipynb`](Colab_Run_Additive_Transformer.ipynb) | 云端全自动化运行流水线，支持驱动同步、一键拉取配置、训练并导出结果。 |
| **总表构建引擎** | [`expand_and_update_all_excels.py`](expand_and_update_all_excels.py) | 实验数据回收后，用于全自动刷新 Excel 工作簿并同步 JSON 状态的更新脚本。 |

---

## 🎯 待执行实验任务池 (按优先级排序)

在全量 220 项实验中，**序号 001–196 均为已完成并回填实测指标的实验**。接棒 Agent 的核心使命是完成以下两大类**待运行 (unrun) 实验**：

### 优先级 1：前沿机制突破实验组（序号 197–204）🌟🌟🌟
这 8 个实验旨在打破当前学术研究的三大硬瓶颈（`add4` 进位受阻、长度外推为 0%、草稿纸盲目顺从）：

| 序号 | 配置文件名 | 实验名称 | 核心科学假说与关注指标 |
|---|---|---|---|
| **197** | `197_l4_d128_lsd.json` | 逆序目标对齐 L4_D128 (LSD优先) | 将答案由高位先出改为低位先出（如 `975` 代替 `579`），消解注意力反向寻址开销。**重点观察：`add4` 是否突破 45% 跃升至 80%+**。 |
| **198** | `198_l2_d64_lsd.json` | 逆序目标对齐 L2_D64 轻量对照 | 验证极小参数（16万）下低位对齐能否提前解锁 3 位进位。 |
| **199** | `199_k_0_4.json` | 进位链深度课程采样 (K=0..4) | 按连续进位数 K 梯级退火，剥离“位数”与“进位级联深度”的混淆变量。 |
| **200** | `200_4_9999_1_100.json` | 极端 4 级雪崩进位压力测试 | 100% 采样 `9999+1` 类连续进位算式，探究进位累加器的极限饱和。 |
| **201** | `201_looped-ut_block_4.json` | 循环权重共享网络 (Looped UT-4步) | 单 Block 权重递归迭代 4 步，参数压缩 75%，检验状态机泛化。 |
| **202** | `202_7.json` | 循环网络跨长度外推探针 (7步) | 相同 UT 模型自适应展开 7 步，**重点观察：5–7 位外推是否打破 0.0% 僵局**。 |
| **203** | `203_cot_c_c-b_a.json` | 正反双向自验算 CoT 验证器 | 在 CoT 尾部增加反向验算式，建立双向约束。 |
| **204** | `204_reader.json` | 草稿篡改自纠错强化学习 | 注入 20% 错误草稿，给纠错加分。**重点观察：顺从错误率是否降至 30% 以下**。 |

### 优先级 2：训练步数规模化梯级扫描（序号 205–220）🌟🌟
在 `L4_D128 CoT` 基准模型上，进行训练步数的跨数量级缩放验证：
* **超微步数验证**：`205` (20步), `206` (50步), `207` (100步), `208` (200步), `209` (500步)
* **中期对齐验证**：`210` (1000步), `211` (2000步), `212` (4000步)
* **大算力长程扩展**：`213` (8000步), `214` (16000步), `215` (32000步), `216` (64000步), `217` (128000步), `218` (256000步), `219` (512000步), `220` (1024000步 百万步泛化)

---

## 🚀 Google Colab 执行流水线 (SOP)

接棒 Agent 请指引用户或通过自动化机制按以下流程在 Colab 执行：

### 第一步：准备与打开 Colab
1. 将 [`Colab_Run_Additive_Transformer.ipynb`](Colab_Run_Additive_Transformer.ipynb) 上传至 [Google Colab](https://colab.research.google.com/)。
2. 开启 Colab 运行环境（建议选择 **T4 GPU** 加速，或使用标准 CPU 亦可）。

### 第二步：克隆代码与加载配置
在 Colab 代码单元格中挂载 Google Drive 并拉取最新仓库：
```python
# Colab 运行单元格
!git clone https://github.com/Hana-ame/agent.git workspace
%cd workspace/additive-rand-transformer
!pip install -q torch openpyxl huggingface_hub pandas
```

### 第三步：拉起指定实验
使用统一启动入口 `run.py` 或 `train.py`，传入指定实验的 JSON 配置文件：
```bash
# 示例：运行 197 号实验（逆序目标对齐机制突破）
python -m additive_rand_transformer.train --config configs/197_l4_d128_lsd.json

# 示例：运行 201 号实验（循环权重共享 Looped-UT）
python -m additive_rand_transformer.train --config configs/201_looped-ut_block_4.json
```

### 第四步：执行标准化评测与结果保存
训练结束后，模型会在测试集上输出统一格式的评估读数：
```text
Evaluation Metrics:
- loss: 0.1824
- add1: 100.0% | add2: 100.0% | add3: 93.3% | add4: 76.7%
- sub1: 100.0% | sub2: 100.0% | sub3: 96.7% | sub4: 90.0%
- extrapolate (5-7 digits): add5 15.0%, add6 0.0%
- training_time: 412.5s
```
将该评测 JSON 结果保存回 Drive 或本地。

---

## 📊 回收结果与更新 Excel 台账流程

当 Colab 完成某项（或某批）实验后，接棒 Agent 需在本地将指标回填进系统，保持总库完整：

1. **修改数据源行项**：
   打开本地构建脚本 [`expand_and_update_all_excels.py`](expand_and_update_all_excels.py)，找到对应实验项（如 `EXP-REV-01` 或步数扫描项）。
2. **填入实测定量数据**：
   * 将 `"未跑"` 替换为真实指标：如 `"add1": "100.0%"`, `"add4": "80.0%"`, `"loss": "0.1824"`, `"time_s": "412.5"`。
   * 在 `"conclusion"` 字段中补全**实测现象分析与是否符合预期归因**。
3. **一键重新编译工作簿与配置**：
   在父仓库根目录下执行（仅执行脚本组装，不跑模型！）：
   ```bash
   uv run --with openpyxl python3 expand_and_update_all_excels.py
   ```
4. **验证成果**：
   * 确认 `ALL_DOCS_EXPERIMENTS_CONFIG_TO_RESULTS.xlsx` 中该行单元格由黄色（未跑）转为绿色（完成）；
   * 确认 `additive-rand-transformer/configs/{seq_id}_...json` 中的 `"status"` 由 `"unrun"` 变为 `"completed"`。

---

## 📝 机制归因分析写作模版 (Conclusion Guidelines)

回填结论时，必须回答以下核心机制问题，严禁空泛泛罗列数字：
1. **是否符合预期 (Hypothesis Check)**：
   * 符合预期 / 违背预期 / 发现新反直觉现象？
2. **机制成因归因 (Causal Attribution)**：
   * 性能提升归因于什么？（如“消除反向寻址延迟”、“提升进位链信噪比”、“循环状态机展开”等）。
   * 若失败，瓶颈卡在何处？（如“梯度阻断”、“局部模式坍缩”、“草稿纸容量饱和”等）。
