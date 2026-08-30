# simpleAI — HF submodule 父仓库

本仓库是**极简父仓库**：全部实际内容（模型代码、文档、权重）都挂在
Hugging Face submodule 上，本地不再维护一份拷贝。

```
simpleAI/
├── .gitmodules                    # submodule 注册
├── additive-rand-transformer/     # ← HF submodule (只读镜像)
│   ├── additive_rand_transformer/ # Python 包 (model/data/train/rl/...)
│   ├── checkpoints/               # 训练好的权重 (7 个 .pt)
│   ├── README.md                  # 项目使用手册
│   └── *.md                       # 全部研究文档 (RESEARCH/EXPLORE/IMPROVE/...)
└── .env                           # 本地密钥 (gitignored, 永不提交)
```

## 为什么用 submodule

之前的 `additive-rand-transformer` 分支把模型的 `runs/` 训练产物（803MB）、
各种 db/log 等垃圾文件卷进了本地仓库。改为 HF submodule 后：

- 父仓库只有 3 个文件（`.gitmodules` + `.gitignore` + `README.md`），**零垃圾**
- 内容与 HF Hub (`Hana-ame/additive-rand-transformer`) 单一来源，本地即镜像
- `checkpoints/` 走 Git LFS，不占父仓库历史

## 更新 submodule

```bash
git submodule update --remote --depth 1   # 拉取 HF 最新内容
```

## 安全

`.env` 已 gitignore；HF token 只从该文件读，永不入库。
