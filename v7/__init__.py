"""DAG Framework (`v7`).

支持两种无缝调用方式：

1. 模块级直接调用::

    import v7

    dag.load(json_or_path)
    dag.step()
    dag.contents
    dag.edges

2. 实例级调用::

    import v7

    d = dag.load(json_or_path)
    d.step()
    d.contents
    d.edges
"""

from __future__ import annotations

from typing import Any, List, Optional, Sequence, Union

from v7.builder import (
    build_dag_from_description,
    parse_edges_from_description,
    parse_json_stream,
)
from v7.default_processor import (
    DefaultProcessor,
    DefaultProcessorOutcome,
    merge_clean_contents,
)
from v7.engine import DAG, StepResult
from v7.llm import (
    LLMCallRecord,
    LLMClientProtocol,
    MockLLMClient,
    OpenAICompatibleClient,
    deterministic_semantic_embedding,
)
from v7.matcher import (
    EdgeMatchCandidate,
    TagMatcher,
    cosine_similarity,
    ngram_similarity,
)
from v7.meta import (
    apply_prompt_to_dag_dict,
    create_meta_dag,
    transform_dag_with_meta_dag,
)
from v7.schema import (
    ContentItem,
    ContentStore,
    EdgeSpec,
    ModelConfig,
    ModelRegistry,
    Processor,
    ToolRegistry,
    ToolSpec,
    normalize_tags,
    serialize_content,
)

__version__ = "7.0.0"

# 模块级默认 DAG 实例，支持直接 `dag.load(...)` -> `dag.step()` -> `dag.contents` / `dag.edges`
_current_dag: DAG = DAG()


def get_current_dag() -> DAG:
    """返回当前模块级激活的 DAG 实例。"""
    return _current_dag


def load(source: Any, **kwargs: Any) -> DAG:
    """加载 DAG（支持传入目录路径、JSON 文件路径、JSON 字符串或 dict/list）。

    同时更新模块级当前 DAG 并返回该 `DAG` 实例，因此以下两种写法完全等价::

        import v7
        dag.load(json_data)
        dag.step()
        print(dag.contents, dag.edges)

        # 或
        d = dag.load(json_data)
        d.step()
        print(d.contents, d.edges)
    """
    global _current_dag
    _current_dag = DAG().load(source, **kwargs)
    return _current_dag


def step() -> StepResult:
    """推进当前模块级 DAG 一步（`dag.step()`）。"""
    return _current_dag.step()


def run(*, until_tag: str = "result", max_steps: int = 20) -> List[StepResult]:
    """循环推进当前模块级 DAG 直至出现 `until_tag` 或达到 `max_steps`。"""
    return _current_dag.run(until_tag=until_tag, max_steps=max_steps)


def add_edge(
    input_tags: Any,
    processor: Any = None,
    output_tags: Optional[Union[str, Sequence[str]]] = None,
    *,
    layer: int = 0,
    count: int = 0,
    default_id: str = "",
) -> EdgeSpec:
    """向当前模块级 DAG 添加处理边。"""
    return _current_dag.add_edge(
        input_tags,
        processor,
        output_tags,
        layer=layer,
        count=count,
        default_id=default_id,
    )


def add_content(
    contents: Any,
    tags: Optional[Union[str, Sequence[str]]] = None,
    gen: int = 0,
    **extra: Any,
) -> ContentItem:
    """向当前模块级 DAG 添加内容项。"""
    return _current_dag.add_content(contents, tags=tags, gen=gen, **extra)


def set(tag: str, content: Any) -> ContentItem:
    """把外部处理结果写回当前 DAG（`dag.set("tag", "内容")`）。"""
    return _current_dag.add_content(contents=content, tags=[tag])


def get_result(tag: str = "result") -> Optional[str]:
    """获取当前模块级 DAG 指定标签的最新结果内容。"""
    return _current_dag.get_result(tag)


def has_tag(tag: str = "result") -> bool:
    """检查当前模块级 DAG 是否存在指定标签。"""
    return _current_dag.has_tag(tag)


def __getattr__(name: str) -> Any:
    if name in (
        "contents",
        "edges",
        "models",
        "tools",
        "history",
        "step_count",
        "default_processor",
        "matcher",
    ):
        return getattr(_current_dag, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "DAG",
    "StepResult",
    "load",
    "step",
    "run",
    "add_edge",
    "add_content",
    "set",
    "get_result",
    "has_tag",
    "get_current_dag",
    "contents",
    "edges",
    "models",
    "tools",
    "ModelConfig",
    "ModelRegistry",
    "ContentItem",
    "ContentStore",
    "Processor",
    "EdgeSpec",
    "ToolRegistry",
    "ToolSpec",
    "TagMatcher",
    "EdgeMatchCandidate",
    "DefaultProcessor",
    "DefaultProcessorOutcome",
    "LLMClientProtocol",
    "OpenAICompatibleClient",
    "MockLLMClient",
    "LLMCallRecord",
    "create_meta_dag",
    "transform_dag_with_meta_dag",
    "apply_prompt_to_dag_dict",
    "parse_edges_from_description",
    "build_dag_from_description",
    "parse_json_stream",
    "cosine_similarity",
    "ngram_similarity",
    "deterministic_semantic_embedding",
    "normalize_tags",
    "serialize_content",
    "merge_clean_contents",
]
