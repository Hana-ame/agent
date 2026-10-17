"""VEA V6 公开入口（SPEC 文件）：只做 re-export。

生成物对本文件的去留只按 `__all__` 签名判定（见 `parts/bootstrap.py::signature`），
实现体可以整体重写。
"""

__version__ = "6.0.0"

__all__ = [
    "ContentItem",
    "Edge",
    "DAG",
    "DEFAULT_PRESET_EDGE",
    "SmartMockLLM",
    "OpenAICompatibleLLM",
    "load_models_file",
    "create_dag_builder_dag",
    "MetaDAGBuilder",
    "create_generation_dag",
    "bootstrap",
    "generate_framework",
    "render_framework_code",
    "manifest",
    "verify_fixed_point",
    "verify_self_reproduction",
    "parse_edges_from_description",
    "create_bootstrap_dag",
    "FRAMEWORK_SPEC",
    "FRAME_FILES",
    "MAIN_SOURCE",
]

from vea_v6.parts.bootstrap import (
    FRAME_FILES,
    FRAMEWORK_SPEC,
    MAIN_SOURCE,
    bootstrap,
    create_bootstrap_dag,
    create_generation_dag,
    generate_framework,
    manifest,
    render_framework_code,
    verify_fixed_point,
    verify_self_reproduction,
)
from vea_v6.parts.builder import MetaDAGBuilder, create_dag_builder_dag, parse_edges_from_description
from vea_v6.parts.dag import DEFAULT_PRESET_EDGE, DAG, load_models_file
from vea_v6.parts.llm import OpenAICompatibleLLM, SmartMockLLM
from vea_v6.parts.primitives import ContentItem, Edge
