# simpleAI — 研究本地工作区（HF submodule 集合）

本仓库是研究的**本地工作区**：父仓库极简，全部实际内容（模型代码、实验文档、权重）
挂在 Hugging Face submodule 上。

```
simpleAI/
├── .gitmodules                       # submodule 注册
├── additive-rand-transformer/        # ← HF submodule #1（加法算术探针）
│   ├── additive_rand_transformer/    # Python 包 (model/data/train/rl/...)
│   ├── checkpoints/                  # 7 个训练好的权重（已落真文件，非 LFS 指针）
│   ├── README.md                     # 项目使用手册 / model card
│   └── *.md                          # 全部研究文档 (RESEARCH/EXPLORE/IMPROVE/...)
├── maze-transformer/                 # ← HF submodule #2（反应式 2D 迷宫导航）
│   ├── maze_transformer/             # Python 包 (model/data/evaluate/train/...)
│   ├── checkpoints/                  # 训练好的权重（LFS）
│   └── README.md                     # 项目使用手册 / model card
├── README.md                         # 本文件
└── .env                              # 本地密钥（gitignored，永不提交）
```

> `maze-transformer` 的任务规格：智能体**每一步只看四周四格（路 `.` / 墙 `#`）**，
> 输出 **U/D/L/R** 之一；**指向墙 / 越界即为非法，非法则直接不动**（环境拒绝动作，
> 原地停留），到达终点 **G** 成功。训练目标是 BFS 最短路径动作序列，详见其 README。

---

## 模型权重（checkpoints/）

7 个权重**已是真文件**（从 HF 下载落地，非 131 字节 LFS 指针），全部 `torch.load` 实测可加载：

| 文件 | 大小 | 配置 | 用途 |
|---|---|---|---|
| `l4_d128_cot_bias05_final.pt` | 10.66 MB | 4L·128D causal | 机制研究主体（H1–H4） |
| `l4_d128_sft_nobias_final.pt` | 10.66 MB | 4L·128D causal | RL 基线（add4 起点 27%） |
| `l4_d128_grpo_final.pt` | 3.55 MB | 4L·128D + GRPO | add4 27%→32%，sub 无回退 |
| `l4_d128_reinforce_conservative.pt` | 3.55 MB | 4L·128D + REINFORCE | 保守 RL |
| `l4_d128_selfplay_anchor.pt` | 3.56 MB | 4L·128D 自问自答 | 难度锚定，模式坍缩 1/60 |
| `l2_d64_attn_dsa.pt` | 0.65 MB | 2L·64D DSA | 注意力变体最优 |
| `l2_d64_attn_causal.pt` | 0.65 MB | 2L·64D causal | 注意力变体基线 |

### 加载方式

直接本地加载（推荐，权重已在工作区）：

```python
import torch
from additive_rand_transformer.model import TinyGPT, TinyGPTConfig

ck = torch.load("additive-rand-transformer/checkpoints/l4_d128_cot_bias05_final.pt",
                map_location="cpu", weights_only=False)
cfg = TinyGPTConfig(**{k: v for k, v in ck["config"].items()
                       if k in TinyGPTConfig.__dataclass_fields__})
model = TinyGPT(cfg)
model.load_state_dict(ck["model"])
model.eval()
```

或从 HF 拉取（缓存到 `~/.cache/huggingface/hub`，不占工作区）：

```python
from huggingface_hub import hf_hub_download
CKPT = hf_hub_download("Hana-ame/additive-rand-transformer",
                       "checkpoints/l4_d128_cot_bias05_final.pt")
```

> 本机 `huggingface.co` 直连不通，需走镜像端点（见下）。

---

## Git LFS 说明

HF 仓库用 Git LFS 跟踪 `*.pt`（`filter=lfs`）。本机已装 **git-lfs 3.8.0**（conda：

```bash
conda install -y -c conda-forge git-lfs
```

当前 checkpoints 的**真文件 oid 与索引指针逐一核对一致**，`git status` 为 clean。
若日后 submodule 拉到新权重仍是指针，执行：

```bash
cd additive-rand-transformer
git config lfs.url "https://hf-mirror.com/Hana-ame/additive-rand-transformer.git/info/lfs"
git lfs pull
```

注意：submodule 的 `.git` 是文件（`gitdir: ../.git/modules/additive-rand-transformer`），
LFS filter 配置写在 `../.git/modules/additive-rand-transformer/config`。

---

## 更新 submodule

```bash
git submodule update --remote --depth 1    # 拉取 HF 最新内容
git -C additive-rand-transformer lfs pull  # 若出现指针文件
```

---

## 镜像端点

本网络环境 `huggingface.co` **不可达**，`hf-mirror.com` 可达。所有 HF 操作走镜像：

```bash
export HF_ENDPOINT=https://hf-mirror.com
```

git / LFS 仓库地址：`https://hf-mirror.com/Hana-ame/additive-rand-transformer.git`
（`maze-transformer` 同理：`https://hf-mirror.com/Hana-ame/maze-transformer.git`；
推送走 git 代理 `http://172.29.80.1:10809`，或对 API 设 `HTTPS_PROXY` 后用
`huggingface_hub` 上传。）

---

## 研究文档索引

全部实验报告在 `additive-rand-transformer/`（HF submodule）内，入口是
[`RESEARCH.md`](additive-rand-transformer/RESEARCH.md)（实验地图 + 报告索引 + 三要素总表）。

核心结论（详见 `EXPLORE.md`）：

1. **CoT 是草稿纸，不是推理** —— 答案读"和列"（88.7% 敏感），不回读操作数（0%）
2. **数据形态 > 模型容量** —— 不加 CoT 任意规模多位加法恒 0%
3. **绝不外推** —— 只训 1–4 位，测 5–7 位全 0%
4. **DSA 注意力 > causal > linear**；**GRPO > REINFORCE**；**LoRA 防灾难性遗忘**

---

## 安全

`.env` 已 gitignore；HF token 只从该文件读，永不入库。新仓库 `.gitignore` 覆盖
`*.db*`、`*.log`、`.env`、`*.egg-info`。

---

## 历史

- 从遗留 monorepo 剥离：原仓库 131 文件，本项目仅 30；根 `pyproject.toml` 打包的是
  `framework*`（无关的 vertex-edge-agent），本项目自带依赖（torch/numpy）
- 独立副本 `/mnt/d/Workplace/additive-rand-transformer` 已完成使命并删除（63M）
- HF 仓库最初由并行 agent 以扁平布局推送，已合并为 package 布局并补全 model card
