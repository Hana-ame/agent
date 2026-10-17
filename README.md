# DAG Agent (v7.0)

一个基于标签流驱动（Tag-driven）、分层优先级调度与元图（Meta-DAG）自修改的模块化 Agent 框架。

彻底解决传统工作流「单文件泥球、脏上下文隐式拼接污染、缺乏单步原子调度与处理计数控制、元图自举晦涩难懂」等问题。

---

## 🌟 核心设计理念

1. **三大核心规范文件**
   - `models.json`: 模型注册配置 `{key: {api/endpoint, key/apikey, model, params}}`。
   - `contents.json`: 状态内容项集合 `[{tags: [str], contents: str, gen: int}]`，支持预加载与 Python 上下文安全读写。
   - `edges.json`: 处理边集合 `[{input_tags, processor: {model, prompt}, output_tags, layer: int, count: int}]`。

2. **单步调度 `dag.step()`**
   - **代际递增 (`gen`)**: 每经过一条处理边或一次合成，新内容的 `gen = max(input.gen) + 1`。
   - **分层优先级 (`layer`)**: `layer=0` 最优先，数值越大越往后执行。
   - **执行计数 (`count`)**: 记录边被触发的次数，每次执行 `edge.count += 1`。
   - **调度顺序**: `(layer 升序, input.gen 升序, edge.count 升序, 语义相似度 降序)`。

3. **默认处理器 `dag.default_processor` (N in 1 out)**
   - **Mode 1 (定向拼凑满足边)**: 当单条内容无法触发任何边时，优先选取 `count` 较小的边，从现有内容池中拼凑出其 `input_tags`，合成单条输出并驱动该边。
   - **Mode 2 (自主路由 + 纯净上下文)**: 当无可满足边时，两阶段自主接管：
     - *Phase 1 (路由选择)*: 自主决定选择哪几个 input 索引以及赋予什么 output_tags。
     - *Phase 2 (正式输出)*: 发起 **100% 上下文纯净** 的生成请求（只包含被选中的 input 内容与提示词，绝无未选条目或路由元数据噪音）。

4. **元 DAG (`Meta-DAG`)**
   - 元 DAG 自身就是一个标准 DAG。
   - 输入：待修改目标 DAG (`target_dag` 的 JSON) + 一段修改指令 `prompt`。
   - 输出：修改并验证后的新 DAG（JSON / hydrated 实例）。
   - 支持自然语言修改指令、结构化 JSON Patch，以及元 DAG 修改元 DAG 自身。

---

## 🚀 架构演进七大版本 (Versions A ~ G)

| 版本 | 类名 | 核心特性 | 典型场景 |
| :--- | :--- | :--- | :--- |
| **Version A** | `VersionA_MinimalDAG` | 极简精确标签匹配（关闭向量近似）、严格 `layer`/`gen`/`count` 调度 | 强确定性规则流水线 |
| **Version B** | `VersionB_SemanticEmbeddingDAG` | 向量与 N-gram 余弦相似度阈值匹配 (`threshold`) | 标签存在同义词或自然语言偏差 |
| **Version C** | `VersionC_CleanContextDefaultDAG` | 两阶段干净上下文 `default_processor` (低 count 优先拼凑 + 自主隔离) | 复杂长文本多步聚合 |
| **Version D** | `VersionD_MetaSelfModifyingDAG` | 元图自省与自修改 (`modify_with_prompt`) | 动态自进化 Agent、Agent 编写 Agent |
| **Version E** | `VersionE_AsyncParallelDAG` | 异步 Frontier 并发批处理 (`step_parallel_frontier`) | 树状或多分支并行评估、并发检索 |
| **Version F** | `VersionF_TransactionalReactiveDAG` | 事务快照与回滚 (`checkpoint`, `rollback`, `transaction`) + 环路上限防死循环 | 关键金融事务、试错与回退机制 |
| **Version G** | `VersionG_ProductionCompleteDAG` | 终极整合版：融合 A~F 全部能力，提供结构化 Trace 导出与 Mermaid 图生成 | 工业级生产部署与全链路可视化审计 |

---

## 🛠 快速上手

### 1. 快速使用
```python
from dag_agent import DAG

# 从当前目录加载 models.json, contents.json, edges.json
dag = DAG.from_directory(".")

# 单步推进
step_result = dag.step()
print("Step progressed:", step_result.progressed)

# 循环运行至目标标签或最大步数
dag.run(until_tag="final_report", max_steps=10)

# 获取最终结果
print("Result:", dag.get_result("final_report"))
```

### 2. Python 上下文中读写 contents
```python
with dag.contents.edit(auto_save=True) as store:
    store[0].contents += " (补充审计备注)"
    store.add({"tags": ["extra_note"], "contents": "审计完成", "gen": 0})
```

### 3. 元 DAG 修改 DAG
```python
from dag_agent import transform_dag_with_meta_dag

modified_dag = transform_dag_with_meta_dag(
    target_dag=dag,
    prompt="添加边: [final_report] -> (laptop-6.8: 请将最终报告翻译为英文) -> [english_report] layer=2"
)
```

---

## 🧪 运行测试

```bash
# 运行全部测试套件
python3 -m unittest tests/test_dag_agent_all.py -v

# 运行 A ~ G 版本效果演示
python3 examples/03_versions_a_to_g_showcase.py
```
