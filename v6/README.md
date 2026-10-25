# Vertex-Edge Agent V6

> **图即 `{contents: [], edges: []}`；只加处理边（必有输入/输出 tag，按 layer 从小到大跑）；找不到对应 tag 时由预设边从相近信息合成「输出 tag = 所需输入 tag」的条目；外部检测 `tag=result` 终止。**

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
from vea_v6 import DEFAULT_PRESET_EDGE, DAG

# 构造时显式传入默认预设边常量 DEFAULT_PRESET_EDGE；
# models 以 model 名为 KEY 注册 api/key/model/params，边里 {"模型": KEY} 按 KEY 查询注入
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
# 需汇聚两个上游 tag（不允许没有输入 tag）；找不到时自动通过 dag.preset_edge 从相近信息合成 输出tag=所需输入tag 的条目
dag.add_edge(["revenue_part", "cost_part"], {"模型": "sensenova-v6-reasoner", "处理prompt": "计算净利润并生成结论"}, ["result"], layer=2)

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
| **处理边** | `(input_tags: list[str], processor, output_tags: list[str], layer: int)`；**不允许没有输入 tag / 输出 tag**；processor 支持纯字符串 `prompt: str`、指定模型字典 `{"模型": str, "处理prompt": str}`（兼容 `{"model": ..., "prompt": ...}`）、或 Python 可调用函数；`dag.run()` 总是从 `layer` 小的边开始跑起 |
| **模型注册表** | `DAG(models={...})`：以 model 名为 KEY 保存 `{"api", "key", "model", "params"}`；边里 `{"模型": KEY}` 调用 LLM 时按 KEY 查询并注入 api/key/参数；设置了注册表却查不到 KEY 会抛 `KeyError` |
| **信息条目** | `{"tags": list[str], "content": str, "image"?: path\|url, "gen": 0}`；初始送入 `gen=0`，每经过一次边处理代数自动递增，无图片时自动省略 `image` 键 |
| **预设边合成** | `dag.preset_edge`：默认为模块级常量 `DEFAULT_PRESET_EDGE`（构造 DAG 时显式传入）。当某条边在 run 时找不到对应输入 tag 的条目时，自动从 `contents` 中找**相近的一条或多条**信息，合成一条**输出 tag = 该边所需输入 tag** 的新条目后再执行该边 |
| **外部终止** | `dag.run()` 执行一次推进，`dag.context` 列出全部上下文，由外部循环检测 `tag="result"` 终止 |
| **元图构建器** | `create_dag_builder_dag()`：3 边 V6 DAG 构建另一个 V6 DAG，边 3（layer=2）通过 `preset_edge` 合成产出目标业务 DAG |
| **边内联 API 配置** | processor 可直接写 `{"model": ..., "api": ..., "key": ..., "params": {...}, "处理prompt": ...}`，**优先级高于注册表**；也可只覆盖部分字段（如 `{"模型": KEY, "参数": {...}}`） |
| **自举** | `create_bootstrap_dag()` / `bootstrap()`：输入描述 → `tag="result"` 产出**完整框架源码**；生成物内嵌自身 SPEC 与源码，调其 `generate_framework()` 能逐字节重建自身（quine 固定点） |

---

## 示例

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

## 测试

```bash
python3 tests/test_vea_v6_all.py
# ALL 18 VEA V6 TEST SUITES PASSED SUCCESSFULLY!

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

## 许可证

MIT
