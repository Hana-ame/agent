"""VEA V6 (Vertex-Edge Agent Framework v6.0) — 纯标签信息流与预设边拼凑的极简 DAG 框架.

核心特性：
1. **极简 DAG 结构 (`{"contents": [], "edges": []}`)**：
   - 一个 `DAG` 实例只需添加处理边：`(输入的 tag list[str], 处理方式 (prompt: str | {"模型": ..., "处理prompt": ...}), 输出的 tag list[str], layer: int)`；
     处理边不允许没有输入 tag / 输出 tag，`dag.run()` 总是从 `layer` 小的边开始跑起。
2. **标准信息条目 (`ContentItem`)**：
   - 送入 DAG 的信息结构为 `{"tags": list[str], "content": str, "image"?: path|url, "gen": 0}`。
3. **模型注册表与预设边 (`dag.models` / `dag.preset_edge`)**：
   - `DAG(models={...})` 以 model 名为 KEY 保存 `{"api", "key", "model", "params"}`，
     边里的 `{"模型": <KEY>, ...}` 在调用 LLM 时按 KEY 查询注册表并注入配置。
   - 当某条 `edge` 在 run 时找不到对应输入 tag 的信息，则自动从现有条目中找相近的一条或多条信息，
     通过预设边对象（默认为模块级常量 `DEFAULT_PRESET_EDGE`，构造 DAG 时建议显式传入
     `preset_edge=DEFAULT_PRESET_EDGE`）合成一条 输出 tag = 该 edge 所需输入 tag 的新信息。
4. **单步运行与外部条件终止**：
   - 一个 `dag` 代表一次任务：`dag.run()` = 跑一次，`dag.context` = 所有的 context 列出。
   - 通过在外部检测是否达成某个条件（一般而言是找一条 `tag="result"` 的 `content`）来作为终止信号和最后回应。
5. **附赠用来构建 DAG 的 DAG (`create_dag_builder_dag` / `MetaDAGBuilder`)**：
   - 构建器自身即是一个标准 V6 `DAG`，接收任务/句子描述并通过自身处理边与预设边运行，在 `tag="result"` 中产出构建好的目标 V6 `DAG`。
"""

from vea_v6.builder import (
    BuilderResult,
    MetaDAGBuilder,
    create_dag_builder_dag,
    parse_edges_from_description,
)
from vea_v6.dag import DEFAULT_PRESET_EDGE, DEFAULT_PRESET_EDGE_PROMPT, DAG, default_text_similarity
from vea_v6.llm import LLMProtocol, OpenAICompatibleLLM, SmartMockLLM
from vea_v6.primitives import ContentItem, Edge, ProcessorType, content_to_str, normalize_tags

# 兼容别名
SentenceToDAGMerger = MetaDAGBuilder
Message = ContentItem

__version__ = "6.0.0"

__all__ = [
    "__version__",
    "ContentItem",
    "Message",
    "Edge",
    "ProcessorType",
    "normalize_tags",
    "content_to_str",
    "DAG",
    "DEFAULT_PRESET_EDGE",
    "DEFAULT_PRESET_EDGE_PROMPT",
    "default_text_similarity",
    "LLMProtocol",
    "OpenAICompatibleLLM",
    "SmartMockLLM",
    "create_dag_builder_dag",
    "MetaDAGBuilder",
    "SentenceToDAGMerger",
    "BuilderResult",
    "parse_edges_from_description",
]
