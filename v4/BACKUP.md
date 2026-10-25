# v4 备份（Vertex-Edge Agent V4）

- 来源分支：`origin/vertex-edge-agent-v4`
- 来源提交：`3c8b8e5`
- 状态：**历史设计备份，不参与当前 v7/v8/v9 运行**
- 内容：vertices/edges 状态图、Guard + cascading abort、动态工具库、examples/docs

## 为什么留

v8 已取材其中的 **settlement barrier**（`join: "any"`）。后续可能继续取材：

- 动态工具库：子图当工具，router 运行时选择
- cascading abort：分支剪枝向下游传播
- 自定义 edge / guard 路由

## 运行

原项目独立依赖，按本目录的 `README.md` / `requirements.txt` 安装；不要用根目录的 pytest 收集它。
