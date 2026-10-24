"""Core DAG Execution Engine (`DAG` & `dag.step()`).

Coordinates:
- Loading/saving `models.json`, `contents.json`, `edges.json`
- Single-step atomic progression via `dag.step()`
- Priority scheduling (`layer` 0-first ascending, `gen` 0-first ascending, `count` ascending)
- Fallback to `dag.default_processor` (Mode 1: N-in-1-out assembly for low-`count` edges;
  Mode 2: Autonomous input/output selection + Clean-Context execution)
"""

from __future__ import annotations

import inspect
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple, Union

from dag.default_processor import DefaultProcessor
from dag.llm import LLMClientProtocol, MockLLMClient
from dag.matcher import EdgeMatchCandidate, TagMatcher
from dag.schema import (
    ContentItem,
    ContentStore,
    EdgeSpec,
    ModelRegistry,
    ToolRegistry,
    ToolSpec,
    normalize_tags,
    serialize_content,
)


@dataclass
class StepResult:
    """Structured record of what happened during a single `dag.step()` call."""

    step_num: int
    action: str  # "edge" | "default_targeted_assembly" | "default_autonomous" | "noop"
    edge_id: Optional[str] = None
    edge_layer: Optional[int] = None
    edge_count: Optional[int] = None
    consumed_indices: Tuple[int, ...] = ()
    produced_items: List[ContentItem] = field(default_factory=list)
    similarity: float = 0.0

    @property
    def progressed(self) -> bool:
        return self.action != "noop" and bool(self.produced_items)


class DAG:
    """Tag-driven Directed Acyclic Graph Agent."""

    def __init__(
        self,
        name: str = "dag",
        *,
        models: Optional[Union[str, Path, Dict[str, Any], ModelRegistry]] = None,
        contents: Optional[Union[str, Path, Sequence[Any], ContentStore]] = None,
        edges: Optional[Union[str, Path, Sequence[Any]]] = None,
        tools: Optional[Union[str, Path, Dict[str, Any], ToolRegistry]] = None,
        llm_client: Optional[LLMClientProtocol] = None,
        default_processor: Optional[DefaultProcessor] = None,
        similarity_threshold: float = 0.70,
        use_embeddings: bool = True,
    ) -> None:
        self.name = name
        self.llm_client: LLMClientProtocol = llm_client or MockLLMClient()

        # 1. Load models.json
        if isinstance(models, ModelRegistry):
            self.models = models
        elif isinstance(models, (str, Path)):
            self.models = ModelRegistry(path=models)
        else:
            self.models = ModelRegistry(models=models)

        # 2. Load contents.json
        if isinstance(contents, ContentStore):
            self.contents = contents
        elif isinstance(contents, (str, Path)):
            self.contents = ContentStore(path=contents)
        else:
            self.contents = ContentStore(items=contents)

        # 3. Load edges.json
        self.edges_path: Optional[Path] = None
        self.edges: List[EdgeSpec] = []
        if isinstance(edges, (str, Path)):
            self.load_edges(edges)
        elif edges:
            for idx, e in enumerate(edges):
                self.add_edge(e, default_id=f"edge_{idx}")

        # 4. Load tools.json
        if isinstance(tools, ToolRegistry):
            self.tools = tools
        elif isinstance(tools, (str, Path)):
            self.tools = ToolRegistry(path=tools)
        else:
            self.tools = ToolRegistry(tools=tools)

        # 5. Matcher & Default Processor
        self.matcher = TagMatcher(
            llm_client=self.llm_client,
            threshold=similarity_threshold,
            use_embeddings=use_embeddings,
        )
        self.default_processor = default_processor or DefaultProcessor()

        # 6. Execution state tracking
        self.step_count: int = 0
        self.history: List[StepResult] = []
        self._executed_edge_sigs: Set[Tuple[int, Tuple[int, ...]]] = set()
        self._assembled_edge_sigs: Set[Tuple[int, Tuple[int, ...]]] = set()
        self._autonomous_sigs: Set[Tuple[int, ...]] = set()

    # ---- File I/O (`models.json`, `contents.json`, `edges.json`, `tools.json`) ------------
    @classmethod
    def from_directory(
        cls,
        directory: Union[str, Path],
        *,
        llm_client: Optional[LLMClientProtocol] = None,
        similarity_threshold: float = 0.70,
    ) -> "DAG":
        """Load a DAG from a directory containing `models.json`, `contents.json`, `edges.json`, and optionally `tools.json`."""
        d = Path(directory)
        return cls(
            name=d.name,
            models=d / "models.json" if (d / "models.json").exists() else None,
            contents=d / "contents.json" if (d / "contents.json").exists() else None,
            edges=d / "edges.json" if (d / "edges.json").exists() else None,
            tools=d / "tools.json" if (d / "tools.json").exists() else None,
            llm_client=llm_client,
            similarity_threshold=similarity_threshold,
        )


    def load(
        self,
        source: Any,
        *,
        llm_client: Optional[LLMClientProtocol] = None,
        similarity_threshold: Optional[float] = None,
    ) -> "DAG":
        """Load DAG state (models, contents, edges, tools) in-place from a directory path,
        JSON file path, JSON string, dict, or list, and return `self`.
        """
        if llm_client is not None:
            self.llm_client = llm_client
            self.matcher.llm_client = llm_client
        if similarity_threshold is not None:
            self.matcher.threshold = similarity_threshold

        # Reset execution tracking when loading new state
        self.step_count = 0
        self.history.clear()
        self._executed_edge_sigs.clear()
        self._assembled_edge_sigs.clear()
        self._autonomous_sigs.clear()

        if isinstance(source, dict):
            self._load_from_dict_payload(source)
            return self

        if isinstance(source, list):
            self._load_from_list_payload(source)
            return self

        if isinstance(source, (str, Path)):
            s_str = str(source).strip()
            # Check if it's a raw JSON string first when it starts with { or [
            if s_str.startswith("{") or s_str.startswith("["):
                try:
                    parsed = json.loads(s_str)
                    if isinstance(parsed, dict):
                        self._load_from_dict_payload(parsed)
                    elif isinstance(parsed, list):
                        self._load_from_list_payload(parsed)
                    return self
                except json.JSONDecodeError:
                    pass

            p = Path(source)
            if p.is_dir():
                if (p / "models.json").exists():
                    self.models = ModelRegistry(path=p / "models.json")
                if (p / "contents.json").exists():
                    self.contents = ContentStore(path=p / "contents.json")
                if (p / "edges.json").exists():
                    self.load_edges(p / "edges.json")
                if (p / "tools.json").exists():
                    self.tools = ToolRegistry(path=p / "tools.json")
                return self

            if p.is_file():
                raw = json.loads(p.read_text(encoding="utf-8"))
                if isinstance(raw, dict):
                    self._load_from_dict_payload(raw)
                elif isinstance(raw, list):
                    self._load_from_list_payload(raw, source_path=p)
                return self

        raise FileNotFoundError(f"dag.load: 无法识别或路径不存在 {source!r}")

    def _load_from_dict_payload(self, data: Dict[str, Any]) -> None:
        if "name" in data:
            self.name = str(data["name"])
        if "models" in data:
            self.models = ModelRegistry(models=data["models"])
        if "tools" in data:
            self.tools = ToolRegistry(tools=data["tools"])
        if "contents" in data:
            self.contents = ContentStore(items=data["contents"])
        if "edges" in data:
            self.edges.clear()
            for idx, e in enumerate(data["edges"]):
                self.add_edge(e, default_id=f"edge_{idx}")

    def _load_from_list_payload(self, items: List[Any], source_path: Optional[Path] = None) -> None:
        if not items:
            return
        first = items[0] if isinstance(items[0], dict) else {}
        if any(k in first for k in ("input_tags", "output_tags", "processor")):
            self.edges.clear()
            if source_path is not None:
                self.edges_path = source_path
            for idx, spec in enumerate(items):
                self.add_edge(spec, default_id=f"edge_{idx}")
        else:
            self.contents = ContentStore(items=items, path=source_path)

    def save_directory(self, directory: Union[str, Path]) -> None:
        """Save `models.json`, `contents.json`, `edges.json`, and `tools.json` into `directory`."""
        d = Path(directory)
        d.mkdir(parents=True, exist_ok=True)
        self.models.save(d / "models.json")
        self.contents.save(d / "contents.json")
        self.save_edges(d / "edges.json")
        if self.tools:
            self.tools.save(d / "tools.json")

    def call_tool(self, tool_name: str, **kwargs: Any) -> Any:
        """Execute a registered tool on the DAG."""
        return self.tools.execute(tool_name, **kwargs)

    def load_edges(self, path: Union[str, Path]) -> "DAG":
        p = Path(path)
        self.edges_path = p
        raw = json.loads(p.read_text(encoding="utf-8"))
        edge_list = raw.get("edges", raw) if isinstance(raw, dict) else raw
        self.edges.clear()
        for idx, spec in enumerate(edge_list):
            self.add_edge(spec, default_id=f"edge_{idx}")
        return self

    def save_edges(self, path: Optional[Union[str, Path]] = None) -> None:
        target = Path(path) if path else self.edges_path
        if not target:
            raise ValueError("No file path specified for saving edges.json")
        target.write_text(
            json.dumps([e.to_dict() for e in self.edges], ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    # ---- Graph Manipulation -------------------------------------------------
    def add_edge(
        self,
        input_tags: Any,
        processor: Any = None,
        output_tags: Optional[Union[str, Sequence[str]]] = None,
        *,
        layer: int = 0,
        count: int = 0,
        default_id: str = "",
    ) -> EdgeSpec:
        if processor is None and output_tags is None:
            edge = EdgeSpec.from_any(input_tags, default_id=default_id or f"edge_{len(self.edges)}")
        else:
            edge = EdgeSpec(
                input_tags=input_tags,
                processor=processor,
                output_tags=output_tags or [],
                layer=layer,
                count=count,
                id=default_id or f"edge_{len(self.edges)}",
            )
        if not edge.id:
            edge.id = f"edge_{len(self.edges)}"
        self.edges.append(edge)
        return edge

    def add_content(
        self,
        contents: Any,
        tags: Optional[Union[str, Sequence[str]]] = None,
        gen: int = 0,
        **extra: Any,
    ) -> ContentItem:
        if isinstance(contents, dict) and tags is None:
            return self.contents.add(contents)
        return self.contents.add(contents=contents, tags=tags or ["input"], gen=gen, **extra)

    # ---- Single-Edge Execution ----------------------------------------------
    def _execute_edge(self, edge: EdgeSpec, input_items: Sequence[ContentItem]) -> ContentItem:
        """Run `edge.processor` on `input_items`, increment `edge.count`, and return output `ContentItem` (`gen + 1`)."""
        next_gen = (max(it.gen for it in input_items) + 1) if input_items else 1
        out_tags = normalize_tags(edge.output_tags) or ["output"]
        proc = edge.processor

        if proc.fn is not None:
            sig = inspect.signature(proc.fn)
            params = [
                p
                for p in sig.parameters.values()
                if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)
            ]
            if len(params) >= 2:
                raw_out = proc.fn(
                    input_items[0].contents if len(input_items) == 1 else [i.contents for i in input_items],
                    input_items[0] if len(input_items) == 1 else list(input_items),
                )
            elif len(params) == 1:
                # 单参数 fn 只收文本；多输入时收文本列表。
                # 旧版在这里 try/except 后拿 ContentItem 再调一次：
                # ① 会吞掉 fn 自己的异常（KeyError 被换成难懂的 TypeError）；
                # ② 有副作用的 fn（发请求、写库）会被执行两遍。
                raw_out = proc.fn(
                    input_items[0].contents if len(input_items) == 1
                    else [i.contents for i in input_items]
                )
            else:
                raw_out = proc.fn()
            out_text = serialize_content(raw_out)
        else:
            prompt = proc.format_prompt(input_items)
            model_cfg = self.models.resolve(proc.model)
            out_text = self.llm_client.complete(
                prompt,
                model_config=model_cfg,
                phase="edge",
                is_clean_context=True,
                selected_inputs=[it.contents for it in input_items],
            )

        edge.count += 1
        return ContentItem(tags=out_tags, contents=out_text, gen=next_gen)

    # ---- Core Atomic Step (`dag.step()`) ------------------------------------
    def step(self) -> StepResult:
        """Execute one atomic step of the DAG:

        1. **Direct / Threshold Match**:
           - Find all `(edge, content)` candidates whose tag similarity / embedding cosine >= `threshold`.
           - Pick highest priority: `layer` ascending (0 first), then `gen` ascending (0 first), then `count` ascending.
           - Execute edge (`edge.count += 1`), save output with `output_tags` and `gen = input.gen + 1` in `dag.contents`.
        2. **Fallback to `dag.default_processor`** when no candidate matches above threshold:
           - **Mode 2a (Targeted N-in-1-out Assembly)**: Find which tags in `dag.contents` can be pieced together
             to form an edge's `input_tags`, **prioritizing edges with smaller `count`**. Run clean-context synthesis
             and execute that edge.
           - **Mode 2b (Autonomous Free Routing + Clean Context)**: If no edge was run in this round,
             `default_processor` selects which `N` inputs to use and decides what `output_tags` to produce
             (Phase 1 routing), then issues a **context-clean formal output request** (Phase 2) and appends the result.
        """
        self.step_count += 1

        # Phase 1: Try Direct / Embedding-Threshold Edge Match
        candidate: Optional[EdgeMatchCandidate] = self.matcher.find_best_candidate(
            self.edges,
            self.contents,
            self._executed_edge_sigs,
        )
        if candidate is not None:
            self._executed_edge_sigs.add((candidate.edge_index, candidate.content_indices))
            matched_items = [self.contents[i] for i in candidate.content_indices]
            out_item = self._execute_edge(candidate.edge, matched_items)
            self.contents.add(out_item)
            res = StepResult(
                step_num=self.step_count,
                action="edge",
                edge_id=candidate.edge.id,
                edge_layer=candidate.edge.layer,
                edge_count=candidate.edge.count,
                consumed_indices=candidate.content_indices,
                produced_items=[out_item],
                similarity=candidate.similarity,
            )
            self.history.append(res)
            return res

        # Phase 2a: Default Processor — Targeted N-in-1-out Puzzle Assembly (prioritizing lowest `edge.count`)
        assembly = self.default_processor.find_puzzle_assembly_for_edges(
            self.edges,
            self.contents,
            self.matcher,
            self._assembled_edge_sigs,
        )
        if assembly is not None:
            e_idx, target_edge, combo_indices = assembly
            self._assembled_edge_sigs.add((e_idx, combo_indices))
            selected_items = [self.contents[i] for i in combo_indices]

            # Synthesize N-in-1-out item with clean context carrying `target_edge.input_tags`
            synthesized_item = self.default_processor.execute_clean_synthesis(
                selected_items=selected_items,
                output_tags=target_edge.input_tags,
                llm_client=self.llm_client,
                models=self.models,
            )
            self.contents.add(synthesized_item)
            synth_idx = len(self.contents) - 1
            produced = [synthesized_item]

            if self.default_processor.execute_target_edge_immediately:
                self._executed_edge_sigs.add((e_idx, (synth_idx,)))
                edge_out = self._execute_edge(target_edge, [synthesized_item])
                self.contents.add(edge_out)
                produced.append(edge_out)

            res = StepResult(
                step_num=self.step_count,
                action="default_targeted_assembly",
                edge_id=target_edge.id,
                edge_layer=target_edge.layer,
                edge_count=target_edge.count,
                consumed_indices=combo_indices,
                produced_items=produced,
            )
            self.history.append(res)
            return res

        # Phase 2b: Default Processor — Autonomous Input/Output Selection + Clean-Context Synthesis
        if len(self.contents) > 0:
            selected_indices, decided_output_tags = self.default_processor.route_autonomous_inputs_and_tags(
                contents=self.contents,
                edges=self.edges,
                llm_client=self.llm_client,
                models=self.models,
                autonomous_history=self._autonomous_sigs,
            )
            if selected_indices:
                self._autonomous_sigs.add(selected_indices)
                selected_items = [self.contents[i] for i in selected_indices]
                out_item = self.default_processor.execute_clean_synthesis(
                    selected_items=selected_items,
                    output_tags=decided_output_tags,
                    llm_client=self.llm_client,
                    models=self.models,
                )
                self.contents.add(out_item)
                res = StepResult(
                    step_num=self.step_count,
                    action="default_autonomous",
                    consumed_indices=selected_indices,
                    produced_items=[out_item],
                )
                self.history.append(res)
                return res

        res = StepResult(step_num=self.step_count, action="noop")
        self.history.append(res)
        return res

    def run(
        self,
        *,
        until_tag: str = "result",
        max_steps: int = 20,
    ) -> List[StepResult]:
        """Repeatedly call `self.step()` until `until_tag` appears in `self.contents` or `max_steps` is reached."""
        results: List[StepResult] = []
        for _ in range(max_steps):
            if until_tag and self.has_tag(until_tag):
                break
            step_res = self.step()
            results.append(step_res)
            if not step_res.progressed:
                break
        return results

    # ---- Query & Serialization Helpers --------------------------------------
    def has_tag(self, tag: str = "result") -> bool:
        return any(tag in item.tags for item in self.contents)

    def get_latest(self, tag: str = "result") -> Optional[ContentItem]:
        matches = self.contents.by_tag(tag)
        return matches[-1] if matches else None

    def get_result(self, tag: str = "result") -> Optional[str]:
        latest = self.get_latest(tag)
        return latest.contents if latest is not None else None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "models": self.models.to_dict(),
            "tools": self.tools.to_dict(),
            "contents": self.contents.to_list(),
            "edges": [e.to_dict() for e in self.edges],
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=indent)

    @classmethod
    def from_dict(
        cls,
        data: Dict[str, Any],
        *,
        llm_client: Optional[LLMClientProtocol] = None,
        similarity_threshold: float = 0.70,
    ) -> "DAG":
        return cls(
            name=str(data.get("name", "dag")),
            models=data.get("models", {}),
            tools=data.get("tools", {}),
            contents=data.get("contents", []),
            edges=data.get("edges", []),
            llm_client=llm_client,
            similarity_threshold=similarity_threshold,
        )

    @classmethod
    def from_json(
        cls,
        raw_json: str,
        *,
        llm_client: Optional[LLMClientProtocol] = None,
        similarity_threshold: float = 0.70,
    ) -> "DAG":
        return cls.from_dict(json.loads(raw_json), llm_client=llm_client, similarity_threshold=similarity_threshold)

    def __repr__(self) -> str:
        return (
            f"DAG(name={self.name!r}, models={len(self.models)}, "
            f"contents={len(self.contents)}, edges={len(self.edges)}, steps={self.step_count})"
        )
