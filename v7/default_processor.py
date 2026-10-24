"""Default Processor (`dag.default_processor`) for DAG Agent.

Implements the two fallback modes when no edge directly matches above the threshold:

1. **Targeted Puzzle Assembly (拼凑满足 Edge 输入)**:
   - Prioritizes edges with smaller `count` (`(edge.count, edge.layer)` ascending).
   - Finds `N` existing content items whose tags/contents can be pieced together to form
     a target edge's `input_tags`.
   - Synthesizes 1 combined `ContentItem` (`n in 1 out`, `gen = max(inputs.gen) + 1`)
     carrying `edge.input_tags`, and immediately executes that target edge (or deposits the item).

2. **Autonomous Free Routing & Synthesis (自主决定 Input 与 Output Tags)**:
   - Triggered when no edge could be executed or assembled in the current round.
   - **Phase 1 (Routing / Selection Call)**: Selects which `N` input items from `dag.contents`
     to combine and decides what `output_tags` the new item should have.
   - **Phase 2 (Formal Execution Call — Clean Context!)**: Sends a brand-new, **context-clean**
     request containing ONLY the selected `N` inputs and the synthesis instruction (free of
     unselected items or routing metadata) to produce the 1 output item (`n in 1 out`).
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence, Set, Tuple

from v7.llm import LLMClientProtocol
from v7.matcher import TagMatcher
from v7.schema import ContentItem, EdgeSpec, ModelRegistry, normalize_tags


@dataclass
class DefaultProcessorOutcome:
    """Result of running `dag.default_processor`."""

    mode: str  # "targeted_assembly" | "autonomous_free"
    selected_indices: Tuple[int, ...]
    output_item: ContentItem
    target_edge_index: Optional[int] = None
    edge_output_item: Optional[ContentItem] = None


def merge_clean_contents(items: Sequence[ContentItem]) -> str:
    """Deterministically merge N content items cleanly (merges JSON dicts or joins with newlines)."""
    if not items:
        return ""
    if len(items) == 1:
        return items[0].contents
    parsed_dicts: List[Dict[str, Any]] = []
    for it in items:
        s = it.contents.strip()
        if s.startswith("{") and s.endswith("}"):
            try:
                obj = json.loads(s)
                if isinstance(obj, dict):
                    parsed_dicts.append(obj)
                    continue
            except Exception:
                pass
        parsed_dicts = []
        break
    if parsed_dicts:
        merged: Dict[str, Any] = {}
        for d in parsed_dicts:
            merged.update(d)
        return json.dumps(merged, ensure_ascii=False)
    return "\n".join(it.contents for it in items if it.contents.strip())


class DefaultProcessor:
    """Tag-free N-in-1-out processor group with Two-Phase Clean-Context execution."""

    def __init__(
        self,
        model: str = "default",
        synthesis_prompt: str = "请将以下输入信息整合为一条结构清晰、语义完整的输出：\n{inputs}",
        routing_prompt: str = (
            "请从候选内容中选择最相关的若干条输入（给出索引列表 selected_indices），"
            "并决定合成后的输出标签列表（output_tags）。仅返回 JSON: "
            '{"selected_indices": [0], "output_tags": ["synthesized"]}'
        ),
        assembly_similarity_floor: float = 0.25,
        execute_target_edge_immediately: bool = True,
        custom_router: Optional[Callable[[Sequence[ContentItem], Sequence[EdgeSpec]], Tuple[List[int], List[str]]]] = None,
        custom_synthesizer: Optional[Callable[[Sequence[ContentItem], Sequence[str]], str]] = None,
    ) -> None:
        self.model = model
        self.synthesis_prompt = synthesis_prompt
        self.routing_prompt = routing_prompt
        self.assembly_similarity_floor = float(assembly_similarity_floor)
        self.execute_target_edge_immediately = bool(execute_target_edge_immediately)
        self.custom_router = custom_router
        self.custom_synthesizer = custom_synthesizer
        self.count: int = 0

    def find_puzzle_assembly_for_edges(
        self,
        edges: Sequence[EdgeSpec],
        contents: Sequence[ContentItem],
        matcher: TagMatcher,
        assembled_history: Set[Tuple[int, Tuple[int, ...]]],
    ) -> Optional[Tuple[int, EdgeSpec, Tuple[int, ...]]]:
        """Find which existing content items can be pieced together (`N in 1 out`) to satisfy an edge's `input_tags`.

        Prioritizes edges with smaller `count` first (`(edge.count, edge.layer, idx)`).
        """
        if not contents or not edges:
            return None

        # Sort edges by (count ascending, layer ascending, original index ascending)
        ranked_edges = sorted(
            enumerate(edges),
            key=lambda pair: (pair[1].count, pair[1].layer, pair[0]),
        )

        for e_idx, edge in ranked_edges:
            if not edge.input_tags:
                continue
            # fn 边是强 schema 的代码，只接受精确 tag 的输入（见 TagMatcher.exact_match_for_fn）；
            # 让它们进 LLM 拼装会把结构不对的东西喂给代码。
            if getattr(matcher, "exact_match_for_fn", False) and getattr(edge.processor, "fn", None) is not None:
                continue
            max_limit = getattr(edge, "max_count", None) or getattr(self, "max_edge_count", None)
            if max_limit is not None and edge.count >= max_limit:
                continue
            selected: List[int] = []
            covered_tags = 0

            for req_tag in edge.input_tags:
                # Find the best content item that covers or semantically relates to `req_tag`
                best_idx: Optional[int] = None
                best_score = -1.0
                for c_idx, item in enumerate(contents):
                    # Avoid output of the same edge
                    if set(item.tags) == set(edge.output_tags):
                        continue
                    tag_sim = max((matcher.tag_similarity(req_tag, t) for t in item.tags), default=0.0)
                    # Also check if req_tag is a substring/component of item.tags or content
                    content_hint = 0.35 if req_tag.lower() in item.contents.lower() else 0.0
                    score = max(tag_sim, content_hint)
                    # Prefer items not yet in `selected` so multi-tag edges gather distinct pieces
                    diversity_bonus = 0.05 if c_idx not in selected else 0.0
                    if score >= self.assembly_similarity_floor and (score + diversity_bonus) > best_score:
                        best_score = score + diversity_bonus
                        best_idx = c_idx

                if best_idx is not None:
                    covered_tags += 1
                    if best_idx not in selected:
                        selected.append(best_idx)

            if covered_tags == len(edge.input_tags) and selected:
                combo = tuple(selected)
                if (e_idx, combo) not in assembled_history:
                    return (e_idx, edge, combo)

        return None

    def route_autonomous_inputs_and_tags(
        self,
        contents: Sequence[ContentItem],
        edges: Sequence[EdgeSpec],
        llm_client: LLMClientProtocol,
        models: ModelRegistry,
        autonomous_history: Set[Tuple[int, ...]],
    ) -> Tuple[Tuple[int, ...], List[str]]:
        """Phase 1 (Routing Decision): Select which N inputs to process and what `output_tags` to assign.

        This routing call is completely separate from the Phase 2 clean-context synthesis call.
        """
        if not contents:
            return ((), ["empty"])

        if self.custom_router is not None:
            idxs, out_tags = self.custom_router(contents, edges)
            return (tuple(idxs), normalize_tags(out_tags) or ["synthesized"])

        # Build routing overview (only used in Phase 1 routing call, NEVER in Phase 2 formal output call)
        catalog_lines = [
            f"[{i}] tags={it.tags}, gen={it.gen}, preview={it.contents[:80]!r}"
            for i, it in enumerate(contents)
        ]
        unrun_edges = sorted(edges, key=lambda e: (e.count, e.layer))
        edge_hints = [
            f"Edge(input_tags={e.input_tags} -> output_tags={e.output_tags}, count={e.count}, layer={e.layer})"
            for e in unrun_edges
        ]
        routing_query = (
            f"{self.routing_prompt}\n\n"
            f"当前候选内容 (Contents):\n" + "\n".join(catalog_lines) + "\n\n"
            f"当前待满足边 (Edges, 优先考虑 count 小的边):\n" + ("\n".join(edge_hints) if edge_hints else "无")
        )

        model_cfg = models.resolve(self.model)
        raw_decision = llm_client.complete(
            routing_query,
            model_config=model_cfg,
            system_prompt="You are a DAG routing planner. Output JSON with selected_indices and output_tags.",
            phase="default_routing",
            is_clean_context=False,
        )

        # Parse JSON decision if LLM returned structured routing JSON
        selected_idxs, decided_tags = self._parse_routing_response(raw_decision, len(contents))
        if not selected_idxs:
            # Heuristic fallback: pick unconsumed / highest-gen non-result items
            non_result = [i for i, it in enumerate(contents) if "result" not in it.tags]
            pool = non_result if non_result else list(range(len(contents)))
            selected_idxs = pool[-2:] if len(pool) >= 2 else pool[-1:]

        if not decided_tags:
            # If there is an edge with low count whose input_tags we haven't produced yet, aim for its input_tags
            existing_tags = {t for it in contents for t in it.tags}
            for e in unrun_edges:
                missing = [t for t in e.input_tags if t not in existing_tags]
                if missing:
                    decided_tags = list(e.input_tags)
                    break
            if not decided_tags:
                decided_tags = ["default_out"]

        combo = tuple(selected_idxs)
        if combo in autonomous_history and len(contents) > 1:
            # Shift selection or tag so we don't repeat the exact same autonomous step forever
            combo = (len(contents) - 1,)
        return (combo, normalize_tags(decided_tags))

    @staticmethod
    def _parse_routing_response(raw: str, num_contents: int) -> Tuple[List[int], List[str]]:
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        if not match:
            return ([], [])
        try:
            data = json.loads(match.group(0))
            if not isinstance(data, dict):
                return ([], [])
            raw_idxs = data.get("selected_indices", data.get("inputs", []))
            idxs = [int(i) for i in raw_idxs if isinstance(i, (int, str)) and str(i).isdigit() and 0 <= int(i) < num_contents]
            tags = normalize_tags(data.get("output_tags", data.get("tags", [])))
            return (idxs, tags)
        except Exception:
            return ([], [])

    def execute_clean_synthesis(
        self,
        selected_items: Sequence[ContentItem],
        output_tags: Sequence[str],
        llm_client: LLMClientProtocol,
        models: ModelRegistry,
    ) -> ContentItem:
        """Phase 2 (Formal Output Request — Clean Context!):

        Constructs a strictly clean request containing ONLY `selected_items` (no unselected
        items, no routing catalog, no tag-selection boilerplate) and returns 1 synthesized `ContentItem`.
        """
        out_tags = normalize_tags(output_tags) or ["output"]
        next_gen = (max(it.gen for it in selected_items) + 1) if selected_items else 1

        if self.custom_synthesizer is not None:
            text = self.custom_synthesizer(selected_items, out_tags)
            self.count += 1
            return ContentItem(tags=out_tags, contents=text, gen=next_gen)

        # Builtin preservation when assembling meta inputs (target_dag + prompt)
        has_target_dag = any("target_dag" in it.tags for it in selected_items)
        has_prompt = any("prompt" in it.tags for it in selected_items)
        if has_target_dag and has_prompt:
            t_text = next((it.contents for it in selected_items if "target_dag" in it.tags), "")
            p_text = next((it.contents for it in selected_items if "prompt" in it.tags), "")
            self.count += 1
            return ContentItem(tags=out_tags, contents=f"{t_text}\n\n{p_text}", gen=next_gen)

        clean_inputs_block = "\n\n".join(
            f"--- 输入 {i + 1} ---\n{it.contents}" for i, it in enumerate(selected_items)
        )
        if "{inputs}" in self.synthesis_prompt:
            clean_prompt = self.synthesis_prompt.replace("{inputs}", clean_inputs_block)
        elif "{contents}" in self.synthesis_prompt:
            clean_prompt = self.synthesis_prompt.replace("{contents}", clean_inputs_block)
        else:
            clean_prompt = f"{self.synthesis_prompt}\n\n{clean_inputs_block}"

        model_cfg = models.resolve(self.model)
        response = llm_client.complete(
            clean_prompt,
            model_config=model_cfg,
            system_prompt="",
            phase="default_clean_output",
            is_clean_context=True,
            selected_inputs=[it.contents for it in selected_items],
        )

        # If using default mock echo and multiple structured JSON inputs exist, provide clean merged payload
        if response.startswith(f"[{model_cfg.key if model_cfg else 'default'}]") and len(selected_items) >= 1:
            response = merge_clean_contents(selected_items)

        self.count += 1
        return ContentItem(tags=out_tags, contents=response, gen=next_gen)
