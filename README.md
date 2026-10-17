# Vertex-Edge Agent V6

> **图即 `{contents: [], edges: []}`；只加处理边，无输入则由预设边从现有信息找两条拼凑；外部检测 `tag=result` 终止。**

V6 是 VEA 框架的纯标签信息流重写——彻底抛弃了命名顶点、状态机与 SQLite 同步样板，用一个极简的 `{contents, edges}` 结构表达一切。

详见 [docs/ARCHITECTURE_V6.md](docs/ARCHITECTURE_V6.md)。

---

## 安装

```bash
# Python 3.10+
pip install -e .
```

---

## 快速上手

```python
from vea_v6 import DAG

dag = DAG()
dag.add_edge(["raw_report"], "提取营收明细: {content}", ["revenue_part"])
dag.add_edge(["raw_report"], {"模型": "sensenova-v6", "处理prompt": "提取成本明细: {content}"}, ["cost_part"])
# 无单条直接输入，运行时自动通过 dag.preset_edge 从现有条目找两条拼凑
dag.add_edge([], {"模型": "sensenova-v6-reasoner", "处理prompt": "计算净利润并生成结论"}, ["result"])

dag.send({"tags": ["raw_report"], "content": "Q4总营收5000万，总成本3000万", "gen": 0})

while not any("result" in item["tags"] for item in dag.context):
    dag.run()

print(dag.get_result())
```

---

## 核心概念

| 概念 | 说明 |
|---|---|
| **DAG** | `{"contents": [], "edges": []}`，`dag.contents` / `dag.edges` 属性访问 + `dag["contents"]` 字典访问 + `dag.to_dict()` / `DAG.from_dict()` 序列化 |
| **处理边** | `(input_tags: list[str], processor, output_tags: list[str])`；processor 支持纯字符串 `prompt: str`、指定模型字典 `{"模型": str, "处理prompt": str}`（兼容 `{"model": ..., "prompt": ...}`）、或 Python 可调用函数 |
| **信息条目** | `{"tags": list[str], "content": str, "image"?: path\|url, "gen": 0}`；初始送入 `gen=0`，每经过一次边处理代数自动递增，无图片时自动省略 `image` 键 |
| **预设边拼凑** | `dag.preset_edge`：DAG 初始化时自带。当某条边没有单条直接匹配输入时，自动从 `contents` 中挑选最匹配的两条信息拼凑后再处理 |
| **外部终止** | `dag.run()` 执行一次推进，`dag.context` 列出全部上下文，由外部循环检测 `tag="result"` 终止 |
| **元图构建器** | `create_dag_builder_dag()`：3 边 V6 DAG 构建另一个 V6 DAG，边 3 通过 `preset_edge` 拼凑产出目标业务 DAG |

---

## 示例

| 示例 | 说明 |
|---|---|
| `examples/05_vea_v6_quickstart.py` | 直接创建并运行 V6 DAG，预设边自动拼凑 |
| `examples/06_vea_v6_meta_builder.py` | 元图构建器：V6 DAG 构建 V6 DAG |

```bash
python3 examples/05_vea_v6_quickstart.py
python3 examples/06_vea_v6_meta_builder.py
```

---

## 测试

```bash
python3 tests/test_vea_v6_all.py
# ALL 7 VEA V6 TEST SUITES PASSED SUCCESSFULLY!

python3 -m pytest tests/test_vea_v6_all.py -v
```

7 组测试覆盖：

1. `{contents, edges}` 结构、`prompt: str` 与 `{"模型": ..., "处理prompt": ...}` 边格式
2. `{tags, content, image?, gen}` 信息结构、多模态 `image` 透传与 `gen` 代数演进
3. 空输入边、多分支汇聚边、自定义 `preset_edge` 拼凑链路
4. 多轮循环外部检测 `tag="result"` 终止
5. 字典字面量初始化 DAG
6. 元图通过 `preset_edge` 合成产出业务 DAG，跑出 `{"final_pay": 220}`
7. 自然语言句式、箭头句式及无输入边句式解析

---

## 许可证

MIT
