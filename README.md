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

### 3. 工具挂载与调用 (`tools.json`)
```python
# tools.json: {"calculator": {"handler": "builtin:eval_math"}, ...}
dag = DAG.from_directory(".")

# 调用已挂载的计算器或自定义工具
margin = dag.call_tool("calculator", expression="(52000000 - 32000000) / 52000000 * 100")
print(f"Gross Margin: {margin:.2f}%")
```

### 4. 元 DAG 修改 DAG
```python
from dag_agent import transform_dag_with_meta_dag

modified_dag = transform_dag_with_meta_dag(
    target_dag=dag,
    prompt="添加边: [final_report] -> (laptop-6.8: 请将最终报告翻译为英文) -> [english_report] layer=2"
)
```

---

## 📂 丰富的业务场景示例 (20+ Examples)

`examples/` 目录下提供了 20 个开箱即用的工业级业务 DAG 范例：

| 编号 | 脚本名称 | 业务场景与技术亮点 |
| :--- | :--- | :--- |
| `ex01` | `ex01_financial_report_analysis.py` | 财报多季度数据提取与对齐汇总分析 |
| `ex02` | `ex02_code_review_security_audit.py` | 代码质量静态分析、SQL注入漏洞扫描与修复建议 |
| `ex03` | `ex03_map_reduce_market_research.py` | 竞品情报多路并发提取与 SWOT 矩阵汇聚 |
| `ex04` | `ex04_tool_augmented_calculator_dag.py` | 挂载 `tools.json` 的数学计算工具调用图 |
| `ex05` | `ex05_rag_document_qa_pipeline.py` | 长文档分块、多跳检索与问答合成流水线 |
| `ex06` | `ex06_customer_intent_dispatcher.py` | 客户工单多意图智能识别与专席分流处理 |
| `ex07` | `ex07_iterative_prompt_evolution.py` | 提示词多代际 (`gen 0->1->2->3`) 遗传迭代演化 |
| `ex08` | `ex08_meta_dag_hot_patching.py` | 运行时元 DAG 根据自然语言指令对业务图热修补 |
| `ex09` | `ex09_multi_agent_adversarial_debate.py` | 正反方多智能体对抗辩论与中立裁判综合裁决 |
| `ex10` | `ex10_database_nl2sql_and_validator.py` | 自然语言转 SQL、AST 安全检查与只读校验 |
| `ex11` | `ex11_academic_literature_survey.py` | 学术文献方法提炼、基准对比与研究空白分析 |
| `ex12` | `ex12_multilingual_content_localization.py` | 跨语种文案本地化、文化禁忌过滤与 SEO 优化 |
| `ex13` | `ex13_ecommerce_order_fraud_detection.py` | 电商风控：IP、卡频次与支付行为多维欺诈拦截 |
| `ex14` | `ex14_healthcare_triage_clinical_decision.py` | 医疗症状初步分诊、红旗警告筛查与专科导诊 |
| `ex15` | `ex15_data_etl_cleaning_normalization.py` | 数据 ETL 管道：脏数据清洗、时区转换与入库校验 |
| `ex16` | `ex16_interactive_story_branching.py` | 交互式支线小说：多分支叙事探索与主线结局收束 |
| `ex17` | `ex17_microservice_incident_rca.py` | 微服务故障根因排查：日志、指标与调用链三合一汇聚 |
| `ex18` | `ex18_technical_interview_generator.py` | 岗位技术面试题库生成、参考答案与面试官评分标准 |
| `ex19` | `ex19_legal_contract_risk_audit.py` | 商业合同合规审查：排他条款、违约责任与风险高亮 |
| `ex20` | `ex20_game_npc_dialogue_state_machine.py` | 游戏 NPC 对话状态机：好感度转移与动态支线任务解锁 |

---

## 🧪 运行测试与并发基准

```bash
# 1. 一键并发并行运行全部 20 个 DAG 业务示例 (耗时仅 ~3.5 秒)
python3 examples/run_all_examples_parallel.py

# 2. 运行真实端点 (laptop-6.8 & sensenova-6.8) 在线全链路实测
python3 test_live_endpoints.py

# 3. 运行 6 种不同 DAG 拓扑机制的综合评测
python3 examples/04_diverse_dags_evaluation.py

# 4. 运行完整单元测试套件 (覆盖全模块与 Versions A ~ G)
python3 -m unittest tests/test_dag_agent_all.py -v
```

