"""DAG Agent Framework (`dag_agent`).

Modular, high-cohesion, low-coupling DAG execution framework with:
- `models.json`, `contents.json`, `edges.json` persistence & live Python context
- Tag similarity & embedding threshold matching (`layer` 0-first, `gen` 0-first)
- Edge execution counter (`count`) and generation tracking (`gen + 1`)
- Two-mode `default_processor` (N-in-1-out targeted puzzle assembly prioritizing low-`count` edges,
  plus two-phase autonomous routing with 100% clean-context formal output requests)
- First-class Meta-DAG (`create_meta_dag`, `transform_dag_with_meta_dag`)
- Architectural Versions A through G (`VersionA_MinimalDAG` .. `VersionG_ProductionCompleteDAG`)
"""

from dag_agent.default_processor import DefaultProcessor, DefaultProcessorOutcome, merge_clean_contents
from dag_agent.engine import DAG, StepResult


def load(path, **kwargs) -> "DAG":
    """一行加载 DAG：传入目录（含 models/contents/edges/tools.json）或单个 JSON 文件。

    用法::

        import dag_agent as dag

        d = dag.load("path/to/dir_or_json")
        d.step()
        # ... 其他逻辑 ...
        d.step()
    """
    from pathlib import Path

    p = Path(path)
    if p.is_dir():
        return DAG.from_directory(p, **kwargs)
    if p.is_file():
        return DAG.from_json(p.read_text(encoding="utf-8"), **kwargs)
    raise FileNotFoundError(f"dag.load: 路径不存在 {path!r}")
from dag_agent.llm import (
    LLMCallRecord,
    LLMClientProtocol,
    MockLLMClient,
    OpenAICompatibleClient,
    deterministic_semantic_embedding,
)
from dag_agent.matcher import EdgeMatchCandidate, TagMatcher, cosine_similarity, ngram_similarity
from dag_agent.meta import apply_prompt_to_dag_dict, create_meta_dag, transform_dag_with_meta_dag
from dag_agent.schema import (
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
from dag_agent.watcher import (
    BenchmarkTestCase,
    EvaluationReport,
    PerformanceProfile,
    QualityProfile,
    WatcherOptimizerLoop,
    create_watcher_dag,
)
from dag_agent.versions import (
    VersionA_MinimalDAG,
    VersionB_SemanticEmbeddingDAG,
    VersionC_CleanContextDefaultDAG,
    VersionD_MetaSelfModifyingDAG,
    VersionE_AsyncParallelDAG,
    VersionF_TransactionalReactiveDAG,
    VersionG_ProductionCompleteDAG,
)

__version__ = "7.0.0"

__all__ = [
    "DAG",
    "StepResult",
    "load",
    "ModelConfig",
    "ModelRegistry",
    "ContentItem",
    "ContentStore",
    "Processor",
    "EdgeSpec",
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
    "cosine_similarity",
    "ngram_similarity",
    "deterministic_semantic_embedding",
    "normalize_tags",
    "serialize_content",
    "merge_clean_contents",
    "create_watcher_dag",
    "WatcherOptimizerLoop",
    "BenchmarkTestCase",
    "EvaluationReport",
    "PerformanceProfile",
    "QualityProfile",
    "VersionA_MinimalDAG",
    "VersionB_SemanticEmbeddingDAG",
    "VersionC_CleanContextDefaultDAG",
    "VersionD_MetaSelfModifyingDAG",
    "VersionE_AsyncParallelDAG",
    "VersionF_TransactionalReactiveDAG",
    "VersionG_ProductionCompleteDAG",
]
