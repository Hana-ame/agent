# Google Colab T4 并行批量训练与评测执行指南 (T4 Parallel Training Guide)

> **定位**：利用 Colab 单卡 T4 算力，通过单卡多流并发（Multi-threading / Concurrency）实现 8~32 个加法 Transformer 实验同时训练与 40 题严格评测。

---

## 💡 为什么可以在 Colab 单卡 T4 上并行？

1. **硬件事实**：Google Colab（免费版及普通 Pro）单个运行时容器**仅提供 1 块单卡 NVIDIA T4（15~16GB 显存）**，平台本身不提供像 Kaggle 那样的物理双卡（2x T4）。
2. **算力瓶颈与浪费**：TinyGPT 进位模型非常小（$L=4, d=128$ 仅约 50~90 万参数，显存占用单模型仅 300MB~500MB）。如果单任务串行执行，T4 的 CUDA 核心利用率只有 5%~15%，显存闲置率超 95%。
3. **并发加速原理**：通过内存预生成数据（消除 CPU 瓶颈）+ Python `ThreadPoolExecutor` 启动 8~16 路独立模型并发训练，多个模型同时向单个 T4 的 CUDA 流提交计算图，将 GPU 算力吃满至 100%，整体批量扫描耗时从数小时缩减至数分钟。

---

## ⚡ 方法一：使用现成 Notebook 一键运行（最推荐）

仓库已内置开箱即用的 Notebook：[`Colab_T4_Parallel_Train.ipynb`](Colab_T4_Parallel_Train.ipynb)。

### 操作步骤：
1. 访问 [Google Colab](https://colab.research.google.com/)，上传或打开父目录下的 [`Colab_T4_Parallel_Train.ipynb`](Colab_T4_Parallel_Train.ipynb)。
2. 菜单栏选择：**代码执行程序 (Runtime)** -> **更改运行时类型 (Change runtime type)** -> 硬件加速器选择 **GPU (T4)**。
3. 在「步骤 4：T4 并行加速训练」单元格中配置：
   ```python
   # 运行模式选择：
   # - "FRONTIER_197_204": 优先执行 197–204 前沿机制突破组（8 个）
   # - "TINY_SCALING": 极小参数长训组 (221–241)
   # - "RUN_ALL_UNRUN": 自动扫描 configs/ 中全部 status="unrun" 的实验
   RUN_MODE = "FRONTIER_197_204"
   
   MAX_EXPERIMENTS = 32  # 本次执行实验上限
   PARALLEL = 8          # 并行度：推荐 8 路（平衡 GPU 吞吐与 CPU GIL 锁）
   ```
4. 点击菜单 **代码执行程序 -> 全部运行 (Ctrl+F9)**。
5. 运行完毕后，Checkpoint 与 40 题评测明细将自动同步备份至 Google Drive `MyDrive/SimpleAI_Experiments/`。

---

## 🐍 方法二：在 Colab 代码单元格调用 Python API

如果你已有正在运行的 Colab 会话，可直接运行如下代码块：

```python
# 1. 准备依赖与仓库
!pip install -q torch openpyxl huggingface_hub pandas matplotlib tabulate
!git clone https://huggingface.co/Hana-ame/additive-rand-transformer /content/additive-rand-transformer
%cd /content/additive-rand-transformer

# 2. 导入并行模块并发执行
from additive_rand_transformer.parallel_train import run_parallel_batch

reports = run_parallel_batch(
    run_mode="FRONTIER_197_204",  # 亦可使用 "RUN_ALL_UNRUN"
    max_experiments=8,
    parallel=8,                    # 8 路同时在 T4 上训练
    configs_dir="configs",
    device="cuda"
)

# 3. 打印 40 题得分总览
for r in reports:
    print(f"[{r['config']}] 得分: {r['score']}/40 ({r['score']/40*100:.1f}%) | 耗时: {r['duration']:.1f}s")
```

---

## 🐚 方法三：Bash 后台进程并发原生脚本 (`train.py`)

若需直接批量跑原生 `train.py`，可以在 Colab 中使用 Linux 后台作业符 `&` 并配合 `wait`：

```bash
%%bash
cd /content/additive-rand-transformer

# 4 路任务同时提交后台并发执行
python -m additive_rand_transformer.train --config configs/197_l4_d128_lsd.json > runs/197.log 2>&1 &
python -m additive_rand_transformer.train --config configs/198_l2_d64_lsd.json > runs/198.log 2>&1 &
python -m additive_rand_transformer.train --config configs/199_k_0_4.json > runs/199.log 2>&1 &
python -m additive_rand_transformer.train --config configs/200_4_9999_1_100.json > runs/200.log 2>&1 &

# 阻塞等待当前 4 个后台任务全部完成
wait
echo "第一批次 4 个实验已全部训练完成！"
```

---

## 🖥️ 补充：若运行环境为多物理卡 (如 Kaggle 2x T4 / 自建集群)

如果将代码迁移到具备多个独立物理 GPU（如 Kaggle 的 `GPU T4 x2`）的环境中，可按卡分配进程：

```bash
# GPU 0 与 GPU 1 独立无干扰并发
CUDA_VISIBLE_DEVICES=0 python -m additive_rand_transformer.train --config configs/197_l4_d128_lsd.json &
CUDA_VISIBLE_DEVICES=1 python -m additive_rand_transformer.train --config configs/198_l2_d64_lsd.json &
wait
```

---

## 🔄 数据回填闭环 (Local Data Backfill SOP)

在 Colab 获得实验输出数据后，切回本地仓库执行回填（**遵守零号红线，本地不跑训练**）：
1. 打开 `expand_and_update_all_excels.py`。
2. 将对应实验的指标及机制归因填入 `new_designed_items` 或 `NEW_STEP_ROWS`。
3. 执行编译引擎生成最新 Excel 与更新 JSON 配置状态：
   ```bash
   uv run --with openpyxl python3 expand_and_update_all_excels.py
   ```
4. 提交 Git 保存成果。
