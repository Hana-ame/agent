# Agent

> 本仓库在 2026-10-20 合并了两条**无共同祖先**的历史：
> `dag_agent` v7 线（远端 `main`）与 `vea_v6` v6 自举线（原先只存在于本地）。
> 两个框架代码互不重叠（`dag_agent/`+`examples/ex*.py` 与 `vea_v6/`+`tools/`），
> 因此是**并存**而非取舍 —— 两边的 README 原文完整保留在下面两大部分里，
> 只把标题各降一级以便嵌套。

| 线 | 代码 | 测试入口 |
|---|---|---|
| DAG Agent v7 | `dag_agent/`, `examples/ex*.py` | `python -m pytest tests/ -q`（**推荐**）、`python examples/run_all_examples_parallel.py`、`python test_live_endpoints.py`（真实端点） |
| VEA v6 自举线 | `vea_v6/`, `tools/` | `python tests/test_vea_v6_all.py`、`python tools/accept_run.py <run>` |

### 测试怎么跑（2026-10-22 分层，默认离线）

```bash
pytest tests/ -q                     # 默认：离线。network / llm 标记的用例自动跳过
pytest tests/ -q --run-network       # 联网：stage1st 抓取 + 真实端点（CI 里是独立手动 job）
pytest tests/test_dag_contracts.py   # 只验所有已注册 DAG 的契约
python -m dags list                  # 注册表能否发现 dag_defs/
python -m dags run <name> --dry-run  # 跑完整条 DAG 但不落盘
```

| 层 | 文件 | 说明 |
|---|---|---|
| 库 | `tests/test_dags_library.py` | 注册 / 自动发现 / CLI / run_dag，全离线 |
| 契约 | `tests/test_dag_contracts.py` | 参数化**所有**已注册 DAG：种子可注入、`final_tag` 从种子可达、产物落 `out/`、默认 `max_steps` 够用 |
| 定义 | `tests/test_dag_defs_offline.py` | 解析 / 兜底纯函数（时间、模型序号、抽取式摘要、META 块） |
| 引擎 | `tests/test_dag_agent_all.py`、`test_dynamic_and_discontinuous.py`、`test_watcher_loop.py` | v7 引擎 |
| 自举 | `tests/test_vea_v6_all.py` | v6 线；`TestVEAV6Suite` 只是 unittest 兼容包装，pytest 下自动跳过（否则同一逻辑跑两遍） |
| 联网 | `tests/test_stage1st_dag.py` | 打 `network` 标记，默认跳过 |

**新增 DAG 不用改测试**：契约测试参数化自动带上；只要 `@register` 的元数据齐全、
`final_tag` 从 `seed_tag` 可达，就会自动纳入闸门。

> **测试入口的坑（实测）**：下面 Part I 文档里写的
> `python -m unittest tests/test_dag_agent_all.py -v` **只跑得到 9 个测试** ——
> v7 线另有 12 个测试（`test_dynamic_and_discontinuous.py` 6 个、
> `test_stage1st_dag.py` 3 个、`test_watcher_loop.py` 3 个）是 **pytest 风格**
> （普通类 + `test_*` 方法，不继承 `unittest.TestCase`），`unittest` 收集不到，
> 会静默显示 `Ran 0 tests` 而不报错。要跑全请用 `python -m pytest tests/ -q`。

> ⚠️ **`models.json` 不入库**（内含端点凭据）：先
> `cp models.json.example models.json` 再跑真实端点脚本。
> 该文件在 2026-10-17 被 v7 线提交过一次，其中的 sensenova 凭据**已随公开仓库
> 暴露**，必须轮换（详见 `.workbuddy/memory/2026-10-20.md`）。

---

# 第一部分：DAG Agent v7

## DAG Agent (v7.0)

一个基于标签流驱动（Tag-driven）、分层优先级调度与元图（Meta-DAG）自修改的模块化 Agent 框架。

彻底解决传统工作流「单文件泥球、脏上下文隐式拼接污染、缺乏单步原子调度与处理计数控制、元图自举晦涩难懂」等问题。

---

### 🌟 核心设计理念

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

### 🚀 架构演进七大版本 (Versions A ~ G)

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

### 🛠 快速上手

#### 1. 快速使用
```python
from dag_agent import DAG

## 从当前目录加载 models.json, contents.json, edges.json
dag = DAG.from_directory(".")

## 单步推进
step_result = dag.step()
print("Step progressed:", step_result.progressed)

## 循环运行至目标标签或最大步数
dag.run(until_tag="final_report", max_steps=10)

## 获取最终结果
print("Result:", dag.get_result("final_report"))
```

#### 2. Python 上下文中读写 contents
```python
with dag.contents.edit(auto_save=True) as store:
    store[0].contents += " (补充审计备注)"
    store.add({"tags": ["extra_note"], "contents": "审计完成", "gen": 0})
```

#### 3. 工具挂载与调用 (`tools.json`)
```python
## tools.json: {"calculator": {"handler": "builtin:eval_math"}, ...}
dag = DAG.from_directory(".")

## 调用已挂载的计算器或自定义工具
margin = dag.call_tool("calculator", expression="(52000000 - 32000000) / 52000000 * 100")
print(f"Gross Margin: {margin:.2f}%")
```

#### 4. 元 DAG 修改 DAG
```python
from dag_agent import transform_dag_with_meta_dag

modified_dag = transform_dag_with_meta_dag(
    target_dag=dag,
    prompt="添加边: [final_report] -> (laptop-6.8: 请将最终报告翻译为英文) -> [english_report] layer=2"
)
```

---

### 📂 丰富的业务场景示例 (20+ Examples)

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

### 🧪 运行测试与并发基准

```bash
## 1. 一键并发并行运行全部 20 个 DAG 业务示例 (耗时仅 ~3.5 秒)
python3 examples/run_all_examples_parallel.py

## 2. 运行真实端点 (laptop-6.8 & sensenova-6.8) 在线全链路实测
python3 test_live_endpoints.py

## 3. 运行 6 种不同 DAG 拓扑机制的综合评测
python3 examples/04_diverse_dags_evaluation.py

## 4. 运行完整单元测试套件 (覆盖全模块与 Versions A ~ G)
python3 -m unittest tests/test_dag_agent_all.py -v
```


---

# 第二部分：VEA v6（自举线，原 `main`）

## Vertex-Edge Agent V6

> **图即 `{contents: [], edges: []}`；只加处理边（必有输入/输出 tag，按 layer 从小到大跑）；找不到对应 tag 时由预设边从相近信息合成「输出 tag = 所需输入 tag」的条目；外部检测 `tag=result` 终止。**

V6 是 VEA 框架的纯标签信息流重写——彻底抛弃了命名顶点、状态机与 SQLite 同步样板，用一个极简的 `{contents, edges}` 结构表达一切。

详见 [docs/ARCHITECTURE_V6.md](docs/ARCHITECTURE_V6.md)。

---

### 安装

```bash
## Python 3.10+
pip install -e .
```

---

### 快速上手

```python
from vea_v6 import DEFAULT_PRESET_EDGE, DAG

## 构造时显式传入默认预设边常量 DEFAULT_PRESET_EDGE；
## models 以 model 名为 KEY 注册 api/key/model/params，边里 {"模型": KEY} 按 KEY 查询注入
dag = DAG(
    preset_edge=DEFAULT_PRESET_EDGE,
    models={
        "sensenova-v6": {"api": "https://api.sensenova.cn/v1", "key": "sk-xxx",
                         "model": "sensenova-v6", "params": {"temperature": 0.2}},
        "sensenova-v6-reasoner": {"api": "https://api.sensenova.cn/v1", "key": "sk-xxx",
                                  "model": "sensenova-v6-reasoner", "params": {"temperature": 0.0}},
    },
)
dag.add_edge(["raw_report"], "提取营收明细: {content}", ["revenue_part"], layer=1)
dag.add_edge(["raw_report"], {"模型": "sensenova-v6", "处理prompt": "提取成本明细: {content}"}, ["cost_part"], layer=1)
## 需汇聚两个上游 tag（不允许没有输入 tag）；找不到时自动通过 dag.preset_edge 从相近信息合成 输出tag=所需输入tag 的条目
dag.add_edge(["revenue_part", "cost_part"], {"模型": "sensenova-v6-reasoner", "处理prompt": "计算净利润并生成结论"}, ["result"], layer=2)

dag.send({"tags": ["raw_report"], "content": "Q4总营收5000万，总成本3000万", "gen": 0})

while not any("result" in item["tags"] for item in dag.context):
    dag.run()

print(dag.get_result())
```

---

### 核心概念

| 概念 | 说明 |
|---|---|
| **DAG** | `{"contents": [], "edges": []}`，`dag.contents` / `dag.edges` 属性访问 + `dag["contents"]` 字典访问 + `dag.to_dict()` / `DAG.from_dict()` 序列化 |
| **处理边** | `(input_tags: list[str], processor, output_tags: list[str], layer: int)`；**不允许没有输入 tag / 输出 tag**；processor 支持纯字符串 `prompt: str`、指定模型字典 `{"模型": str, "处理prompt": str}`（兼容 `{"model": ..., "prompt": ...}`）、或 Python 可调用函数；`dag.run()` 总是从 `layer` 小的边开始跑起 |
| **模型注册表** | `DAG(models={...})`：以 model 名为 KEY 保存 `{"api", "key", "model", "params"}`；边里 `{"模型": KEY}` 调用 LLM 时按 KEY 查询并注入 api/key/参数；设置了注册表却查不到 KEY 会抛 `KeyError` |
| **信息条目** | `{"tags": list[str], "content": str, "image"?: path\|url, "gen": 0}`；初始送入 `gen=0`，每经过一次边处理代数自动递增，无图片时自动省略 `image` 键 |
| **预设边合成** | `dag.preset_edge`：默认为模块级常量 `DEFAULT_PRESET_EDGE`（构造 DAG 时显式传入）。当某条边在 run 时找不到对应输入 tag 的条目时，自动从 `contents` 中找**相近的一条或多条**信息，合成一条**输出 tag = 该边所需输入 tag** 的新条目后再执行该边 |
| **外部终止** | `dag.run()` 执行一次推进，`dag.context` 列出全部上下文，由外部循环检测 `tag="result"` 终止 |
| **元图构建器** | `create_dag_builder_dag()`：3 边 V6 DAG 构建另一个 V6 DAG，边 3（layer=2）通过 `preset_edge` 合成产出目标业务 DAG |
| **边内联 API 配置** | processor 可直接写 `{"model": ..., "api": ..., "key": ..., "params": {...}, "处理prompt": ...}`，**优先级高于注册表**；也可只覆盖部分字段（如 `{"模型": KEY, "参数": {...}}`） |
| **自举** | `create_bootstrap_dag()` / `bootstrap()`：输入描述 → `tag="result"` 产出**完整框架源码**；生成物内嵌自身 SPEC 与源码，调其 `generate_framework()` 能逐字节重建自身（quine 固定点） |

---

### 示例

| 示例 | 说明 |
|---|---|
| `examples/05_vea_v6_quickstart.py` | 直接创建并运行 V6 DAG，模型注册表 + 预设边自动合成 |
| `examples/06_vea_v6_meta_builder.py` | 元图构建器：V6 DAG 构建 V6 DAG |
| `examples/07_vea_v6_tag_reverse_openai.py` | 按 tag 取内容（`todo`）→ 反转 → 写回 `contents`；OpenAI 兼容边 |
| `examples/08_vea_v6_openai_edge_configs.py` | OpenAI 兼容边的三种配置：注册表 / 内联 / 混合（内联优先） |
| `examples/09_vea_v6_bootstrap.py` | 自举：DAG 生成 DAG 框架，并证明生成物能生成自己 |

```bash
python3 examples/05_vea_v6_quickstart.py
python3 examples/06_vea_v6_meta_builder.py
python3 examples/07_vea_v6_tag_reverse_openai.py
python3 examples/08_vea_v6_openai_edge_configs.py
python3 examples/09_vea_v6_bootstrap.py
```

---

### 测试

```bash
python3 tests/test_vea_v6_all.py
## ALL 18 VEA V6 TEST SUITES PASSED SUCCESSFULLY!

python3 -m unittest discover -s tests -v
python3 -m pytest tests/test_vea_v6_all.py -v
```

18 组测试覆盖：

1. `{contents, edges}` 结构、`prompt: str` 与 `{"模型": ..., "处理prompt": ...}` 边格式
2. `{tags, content, image?, gen}` 信息结构、多模态 `image` 透传与 `gen` 代数演进
3. 多分支汇聚边、自定义 `preset_edge` 合成链路、空输入/输出 tag 校验
4. 多轮循环外部检测 `tag="result"` 终止
5. 字典字面量初始化 DAG
6. 元图通过 `preset_edge` 合成产出业务 DAG，跑出 `{"final_pay": 220}`
7. 自然语言句式、箭头句式及无输入边句式解析
8. 模型注册表按 KEY 查询注入、未注册 KEY 抛错、`layer` 升序执行、单条相近信息合成
9. 按 tag 取内容 → 反转 → 写回 `dag.contents`
10. OpenAI 兼容 API 边（model + key + params）
11. 多模型注册表切换
12. 自举 DAG：输入描述 → `tag="result"` 的完整框架源码（含预设边合成的汇聚条目）
13. 固定点：同一 SPEC 恒得同一份代码；生成物当模块执行后能逐字节复现自身
14. 同一段提示词换描述 → 生成另一个自稳定的新框架
15. 生成的框架可当普通库用（建 DAG、按 tag 取内容、跑出 `result`）
16. 内联 OpenAI 配置边；只写 `{"模型": KEY}` 不误判为内联
17. 注册表 KEY + 边内参数覆盖时不得把 KEY 当模型名
18. 推理模型 `content` 为空时回退 `reasoning_content` / `reasoning`

---

### 许可证

MIT
