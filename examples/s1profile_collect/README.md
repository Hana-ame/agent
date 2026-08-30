# S1 Profile Collect（辅助数据目录）

> 辅助目录（gitignored 数据）。按「问题 / 方案 / 修改 / 测试」记录。

## 问题
S1 论坛抓取分析需要一个「楼主/用户画像」数据收集流程，产出数据量大、不值得进版本库。

## 方案
独立收集目录，`data/` 下的 json 数据被 `.gitignore` 忽略（`examples/s1profile_collect/data/`），
不污染仓库历史。

## 修改
- `.gitignore`：`examples/s1profile_collect/data/`（路径下所有 json 不入库）。

## 测试
**测试方案**：数据目录被 git 忽略。**测试方法**：`git status --ignored --short | grep s1profile_collect`。
**测试结果**：`data/` 出现在 ignored 列表，仓库历史不含抓取数据（这也是「本地 383 文件 vs 远端 116」差异的来源）。