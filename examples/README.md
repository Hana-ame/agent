# 框架运行示例 (Examples)

本目录包含了四个独立的示例流水线，用于展示 `vertex-edge-agent` 框架从基础概念到生产级高级应用的各项核心能力。

所有的示例均享有统一的启动方式。你可以通过根目录的 `run.py` 直接运行对应的配置文件：

```bash
# 语法
python examples/run.py examples/<示例目录>/config.json
```

### 示例总览 (Examples Overview)

| 示例名称 (Directory) | 核心特点与展示能力 (Key Features) | 主要做了什么 (What it does) |
| :--- | :--- | :--- |
| **`simple/`** | 基础流水线<br>*(Linear Pipeline)* | 演示了最基本的 **输入 -> 处理 -> 输出** 的 3 个节点串行。展示了最简单的 `Vertex` 状态流转和 `Edge` 处理。 |
| **`complex/`** | 复杂图拓扑与钩子<br>*(DAG & Script Hooks)* | 演示了多路并发的图计算。包含 **Fan-out** (单节点数据拆分多路) 和 **Fan-in** (多路数据汇合等待依赖)。同时演示了如何使用基础的模块级钩子(Module Hooks)对数据进行前置/后置清洗。 |
| **`custom_classes/`** | 动态面向对象子类<br>*(Native Subclassing)* | 抛弃了传统的模块钩子，展示了更强大的 **OOP 架构**。演示框架如何利用 `inspect` 模块动态识别并实例化你在外部脚本中写的 `Vertex` 和 `Edge` 子类，从而优雅地实现数据验证和处理逻辑注入。 |
| **`real_llm/`** | 接驳真实大模型接口<br>*(Real API Integration)* | 演示如何通过覆写 `Edge` 子类完全架空框架自带的 Mock 测试体系。直接使用 `urllib` + `asyncio.to_thread` 向真正的外部云端 API (如 `opencode.ai`) 发起异步的 HTTP POST 请求并提取真实的大模型回复。 |

> **提示：** 每个示例文件夹内均有单独的 `README.md`，内附有该示例专属的 Mermaid 网络拓扑结构图及详细的代码说明，欢迎进入对应的子目录深入查阅！
