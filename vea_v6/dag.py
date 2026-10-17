"""VEA V6 核心有向无环图模块 (DAG).

核心设计规范（V6 全新架构）：
1. **DAG 结构**：`{"contents": [], "edges": []}`
   - `dag.contents`: 存储所有信息条目的列表
   - `dag.edges`: 存储所有处理边的列表
   - `dag.context`: 列出当前所有的 context 条目（即 `dag.contents`）
2. **只用加处理边**：
   每条边为 `(输入的 tag list[str], 处理方式 (prompt: str | {"模型": ..., "处理prompt": ...}), 输出的 tag list[str])`
3. **信息条目结构**：
   `{"tags": list[str], "content": str, "image"?: path|url, "gen": 0}`
4. **无输入边的双条目拼凑与预设边 (`preset_edge`)**：
   对一条 edge，如果没有输入（`input_tags` 为空，或现有条目中没有单条信息能直接匹配其输入 tag），
   则从现有的所有条目的信息中找到两条，通过一条预设边（`dag.preset_edge`，在 `DAG` 初始化时即自带的对象）
   拼凑出一条信息，再交由该边完成处理。
5. **运行与外部条件终止**：
   - 一个 `dag` 代表一次任务
   - `dag.run()` = 跑一次
   - `dag.context` = 所有的 context 列出
   - 通过在外部检测是否达成某个条件（一般而言是找到一条 `tag="result"` 的 `content`）来作为终止信号和最后回应。
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import inspect
import json
import math
import re
from collections import Counter
from typing import Any, Callable, Dict, Iterator, List, Optional, Sequence, Set, Tuple, Union

from vea_v6.llm import LLMProtocol, SmartMockLLM
from vea_v6.primitives import ContentItem, Edge, ProcessorType, content_to_str, normalize_tags


DEFAULT_PRESET_EDGE_PROMPT = (
    "请将以下两条信息拼凑整合为一条语义连贯的完整信息：\n"
    "信息1 [{tags_1}]: {content_1}\n"
    "信息2 [{tags_2}]: {content_2}"
)


def default_text_similarity(text_a: str, text_b: str) -> float:
    """计算两个标签或文本之间的字符级 1-gram + 2-gram 余弦与包含相似度（0.0 ~ 1.0）。"""
    a = re.sub(r"[\s_\-:：,，。()（）\[\]]+", "", str(text_a).lower().strip())
    b = re.sub(r"[\s_\-:：,，。()（）\[\]]+", "", str(text_b).lower().strip())
    if not a or not b:
        return 0.0
    if a == b:
        return 1.0

    def _ngrams(s: str) -> Counter:
        grams = list(s) + [s[i : i + 2] for i in range(len(s) - 1)]
        return Counter(grams)

    vec_a = _ngrams(a)
    vec_b = _ngrams(b)
    common = set(vec_a.keys()) & set(vec_b.keys())
    dot = sum(vec_a[k] * vec_b[k] for k in common)
    norm_a = math.sqrt(sum(v * v for v in vec_a.values()))
    norm_b = math.sqrt(sum(v * v for v in vec_b.values()))
    cosine = (dot / (norm_a * norm_b)) if (norm_a and norm_b) else 0.0

    shorter, longer = (a, b) if len(a) <= len(b) else (b, a)
    if len(shorter) >= 2 and shorter in longer:
        containment = len(shorter) / len(longer)
        cosine = max(cosine, 0.55 + 0.35 * containment)
    return round(min(1.0, cosine), 4)


def _default_preset_combine_contents(c1: str, c2: str) -> str:
    """默认预设边的智能内容拼凑逻辑：若两条内容均为 JSON 字典则合并键值，否则拼接文本。"""
    s1 = c1.strip()
    s2 = c2.strip()
    if s1.startswith("{") and s1.endswith("}") and s2.startswith("{") and s2.endswith("}"):
        try:
            d1 = json.loads(s1)
            d2 = json.loads(s2)
            if isinstance(d1, dict) and isinstance(d2, dict):
                merged = dict(d1)
                merged.update(d2)
                return json.dumps(merged, ensure_ascii=False)
        except Exception:
            pass
    if not s1:
        return s2
    if not s2 or s2 == s1:
        return s1
    return f"{s1}\n{s2}"


class DAG:
    """VEA V6 核心有向无环图实例 (`{contents: [], edges: []}`)."""

    def __init__(
        self,
        name_or_data: Union[str, Dict[str, Any], None] = None,
        *,
        contents: Optional[Sequence[Union[ContentItem, Dict[str, Any]]]] = None,
        edges: Optional[Sequence[Union[Edge, Tuple[Any, ...], List[Any], Dict[str, Any]]]] = None,
        llm: Optional[LLMProtocol] = None,
        preset_edge: Optional[Union[Edge, Tuple[Any, ...], Dict[str, Any]]] = None,
    ) -> None:
        if isinstance(name_or_data, dict):
            self.name = str(name_or_data.get("name", "vea_v6_dag"))
            init_contents = name_or_data.get("contents", contents)
            init_edges = name_or_data.get("edges", edges)
        else:
            self.name = str(name_or_data or "vea_v6_dag")
            init_contents = contents
            init_edges = edges

        self.llm: LLMProtocol = llm or SmartMockLLM()

        # 初始化时自带的预设边对象（用于当某条 edge 没有直接输入时，从现有信息中找两条拼凑一条信息）
        if preset_edge is not None:
            pe = Edge.from_any(preset_edge)
            pe.is_preset = True
            self.preset_edge: Edge = pe
        else:
            self.preset_edge = Edge(
                input_tags=[],
                processor={
                    "模型": "default",
                    "处理prompt": DEFAULT_PRESET_EDGE_PROMPT,
                },
                output_tags=["merged"],
                is_preset=True,
                edge_id="preset_edge",
            )
        # 别名，方便不同命名习惯访问预设边
        self.default_edge: Edge = self.preset_edge

        self.contents: List[ContentItem] = []
        self.edges: List[Edge] = []

        # 记录已执行过的 (edge_index, input_signature) 以免在多轮 run() 循环中重复处理同一输入
        self._executed_records: Set[Tuple[int, str]] = set()
        # 记录哪些信息条目（按其在 contents 中的索引）已经被下游边作为直接输入消费过
        self._consumed_indices: Set[int] = set()
        # 记录已经通过 preset_edge 拼凑过的条目索引对 (i, j)，避免重复拼凑同一对
        self._merged_pairs_by_edge: Dict[int, Set[Tuple[int, int]]] = {}

        if init_contents:
            for item in init_contents:
                self.send(item)

        if init_edges:
            for edge_spec in init_edges:
                self.add_edge(edge_spec)

    # =========================================================================
    # 1. 核心属性与字典接口 (`dag.context`, `dag["contents"]`, `dag["edges"]`)
    # =========================================================================

    @property
    def context(self) -> List[ContentItem]:
        """列出当前 DAG 中所有的 context 信息条目 (`dag.context = 所有的context列出`)。"""
        self._normalize_state()
        return self.contents

    def _normalize_state(self) -> None:
        """确保外部直接操作 `dag.contents.append(...)` 或 `dag.edges.append(...)` 的元素也被规范化。"""
        for i, item in enumerate(self.contents):
            if not isinstance(item, ContentItem):
                self.contents[i] = ContentItem(item if isinstance(item, dict) else {"content": str(item), "tags": [], "gen": 0})
        for i, edge in enumerate(self.edges):
            if not isinstance(edge, Edge):
                self.edges[i] = Edge.from_any(edge)

    def __getitem__(self, key: str) -> Any:
        """支持以 `dag["contents"]` 和 `dag["edges"]` 的字典方式访问。"""
        self._normalize_state()
        if key == "contents":
            return self.contents
        if key == "context":
            return self.context
        if key == "edges":
            return self.edges
        if key == "preset_edge":
            return self.preset_edge
        raise KeyError(key)

    def __contains__(self, key: object) -> bool:
        return key in ("contents", "edges", "context", "preset_edge")

    def keys(self) -> List[str]:
        return ["contents", "edges"]

    def __iter__(self) -> Iterator[str]:
        return iter(("contents", "edges"))

    # =========================================================================
    # 2. 添加处理边与送入信息条目
    # =========================================================================

    def add_edge(
        self,
        input_tags: Union[Edge, Tuple[Any, ...], List[Any], Dict[str, Any], str],
        processor: Optional[ProcessorType] = None,
        output_tags: Optional[Union[str, Sequence[str]]] = None,
    ) -> Edge:
        """向 DAG 实例添加一条处理边：`([input_tags], processor, [output_tags])`。

        既支持三参数调用：
            `dag.add_edge(["query"], "提取关键词: {content}", ["keywords"])`
            `dag.add_edge(["keywords"], {"模型": "gpt-4o", "处理prompt": "总结: {content}"}, ["result"])`
        也支持直接传入三元组或 `Edge` 对象：
            `dag.add_edge((["query"], "提取关键词", ["keywords"]))`
        """
        if processor is None and output_tags is None:
            edge = Edge.from_any(input_tags)  # type: ignore[arg-type]
        else:
            edge = Edge(
                input_tags=input_tags if not isinstance(input_tags, (Edge, dict)) else [],  # type: ignore[arg-type]
                processor=processor,
                output_tags=output_tags,
            )
        if not edge.id:
            edge.id = f"e_{len(self.edges) + 1}"
        self.edges.append(edge)
        return edge

    def send(
        self,
        info: Union[ContentItem, Dict[str, Any], str],
        tags: Optional[Union[str, Sequence[str]]] = None,
        image: Optional[str] = None,
        gen: int = 0,
    ) -> ContentItem:
        """向 DAG 送入一条信息。

        信息结构：`{"tags": list[str], "content": str, "image"?: path|url, "gen": 0}`。
        支持直接传入字典：
            `dag.send({"tags": ["input"], "content": "你好", "gen": 0})`
        或分参传入：
            `dag.send("你好", tags=["input"], image="path/to/img.png", gen=0)`
        """
        if isinstance(info, ContentItem):
            item = ContentItem(
                info,
                tags=tags if tags is not None else info.tags,
                image=image if image is not None else info.image,
                gen=info.gen if "gen" in info else gen,
            )
        elif isinstance(info, dict):
            item = ContentItem(
                info,
                tags=tags if tags is not None else info.get("tags", []),
                content=info.get("content", ""),
                image=image if image is not None else info.get("image"),
                gen=int(info.get("gen", gen)),
            )
        else:
            item = ContentItem(
                tags=tags if tags is not None else ["input"],
                content=content_to_str(info),
                image=image,
                gen=gen,
            )
        self.contents.append(item)
        return item

    # 常用别名，方便不同调用习惯
    add_content = send
    input = send
    feed = send
    push = send

    # =========================================================================
    # 3. 预设边双信息拼凑机制 (Preset Edge Two-Item Synthesis)
    # =========================================================================

    def _select_two_items_for_edge(
        self,
        edge_idx: int,
        edge: Edge,
    ) -> Optional[Tuple[int, int, ContentItem, ContentItem]]:
        """当某条 `edge` 没有单条匹配输入时，从现有的所有 `contents` 条目中找到最合适的两条。

        挑选优先级：
        1. 优先排除已被当前 `edge` 拼凑过的条目对 `(i, j)`；
        2. 优先排除由预设边刚刚生成的纯中间过渡条目（避免无限套娃），除非没有其它条目可用；
        3. 若 `edge.input_tags` 非空，按两条条目合并后对 `edge.input_tags` 的精确覆盖率 + 语义相似度打分；
        4. 优先选择尚未被其它边作为单输入消费过的“叶子”条目，以及 `gen` 更高（更新）的条目。
        """
        self._normalize_state()
        n = len(self.contents)
        if n < 2:
            return None

        used_pairs = self._merged_pairs_by_edge.setdefault(edge_idx, set())
        req_tags = set(edge.input_tags)

        best_score: Optional[Tuple[float, float, int, int, int]] = None
        best_pair: Optional[Tuple[int, int]] = None

        for i in range(n):
            for j in range(i + 1, n):
                if (i, j) in used_pairs:
                    continue
                item_i = self.contents[i]
                item_j = self.contents[j]

                # 避免拿已经带有 result 终止标签的条目去拼凑（除非别无选择）
                has_result_penalty = -10.0 if ("result" in item_i.tags or "result" in item_j.tags) else 0.0
                # 避免拿 preset_edge 合成出的中间条目再去和它的父条目重复拼凑
                is_preset_synthetic = ("_preset_synthesized" in item_i or "_preset_synthesized" in item_j)
                synthetic_penalty = -5.0 if is_preset_synthetic else 0.0

                combined_tags = set(item_i.tags) | set(item_j.tags)

                # 1. 精确标签覆盖率（并且要求两条信息各自至少提供了一部分贡献，或互补覆盖）
                exact_cov = 0.0
                sim_score = 0.0
                if req_tags:
                    covered = len(req_tags & combined_tags)
                    exact_cov = covered / len(req_tags)
                    # 语义相似度加分（处理标签名略有差异的情况）
                    sim_vals = [
                        max(
                            (default_text_similarity(rt, ct) for ct in combined_tags),
                            default=0.0,
                        )
                        for rt in req_tags
                    ]
                    sim_score = sum(sim_vals) / len(sim_vals)

                # 2. 尚未被作为单输入消费过的叶子条目优先
                unconsumed_bonus = (1.0 if i not in self._consumed_indices else 0.0) + (
                    1.0 if j not in self._consumed_indices else 0.0
                )

                # 3. 代数 (gen) 与位置新鲜度
                gen_sum = item_i.gen + item_j.gen

                score_tuple = (
                    round(has_result_penalty + synthetic_penalty + exact_cov * 4.0 + sim_score * 2.0 + unconsumed_bonus, 4),
                    float(gen_sum),
                    i + j,
                    i,
                    j,
                )
                if best_score is None or score_tuple > best_score:
                    best_score = score_tuple
                    best_pair = (i, j)

        if best_pair is None:
            return None

        idx1, idx2 = best_pair
        return idx1, idx2, self.contents[idx1], self.contents[idx2]

    async def _run_preset_edge_on_pair(
        self,
        item1: ContentItem,
        item2: ContentItem,
        target_input_tags: Sequence[str],
    ) -> ContentItem:
        """通过 `self.preset_edge` 将两条信息拼凑为一条新信息。"""
        pe = self.preset_edge
        merged_tags = normalize_tags(list(item1.tags) + list(item2.tags) + list(pe.output_tags) + list(target_input_tags))
        merged_image = item1.image or item2.image
        merged_gen = max(item1.gen, item2.gen) + 1

        if pe.fn is not None:
            sig = inspect.signature(pe.fn)
            params = [
                p for p in sig.parameters.values() if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)
            ]
            if len(params) >= 2:
                res = pe.fn(item1, item2)
            else:
                res = pe.fn([item1, item2])
            if asyncio.iscoroutine(res):
                res = await res
            merged_content = content_to_str(res)
        elif isinstance(self.llm, SmartMockLLM) and not self._mock_llm_matches_preset(pe, item1, item2):
            # 在默认模拟环境下，使用无损的 JSON/文本智能拼凑，使下游边能直接读取两条信息的全部内容
            merged_content = _default_preset_combine_contents(item1.content, item2.content)
        else:
            prompt_str = pe.render_pair_prompt(item1, item2)
            merged_content = await self._call_llm(prompt_str, model=pe.model, image=merged_image)

        synthesized = ContentItem(
            tags=merged_tags,
            content=merged_content,
            image=merged_image,
            gen=merged_gen,
        )
        synthesized["_preset_synthesized"] = True
        return synthesized

    def _mock_llm_matches_preset(self, pe: Edge, item1: ContentItem, item2: ContentItem) -> bool:
        """检查 SmartMockLLM 是否专门配置了针对 preset_edge 的规则。"""
        if not isinstance(self.llm, SmartMockLLM):
            return True
        prompt_str = pe.render_pair_prompt(item1, item2)
        for kw in self.llm.rules:
            if kw in prompt_str:
                return True
        if pe.model and pe.model in self.llm.model_rules:
            return True
        return self.llm.fallback_fn is not None

    # =========================================================================
    # 4. 单条处理边的执行逻辑
    # =========================================================================

    async def _call_llm(
        self,
        prompt: str,
        model: Optional[str] = None,
        image: Optional[str] = None,
    ) -> str:
        """调用大模型后端（自动适配 `ask(prompt, model, image)` 或单参数 `ask(prompt)`）。"""
        ask_fn = getattr(self.llm, "ask", None)
        if ask_fn is None and callable(self.llm):
            ask_fn = self.llm
        if ask_fn is None:
            raise ValueError("当前 DAG 配置的 llm 对象未实现 ask 方法且不可调用")

        sig = inspect.signature(ask_fn)
        params = [
            p for p in sig.parameters.values() if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)
        ]
        has_var_kw = any(p.kind == p.VAR_KEYWORD for p in sig.parameters.values())

        if len(params) >= 3 or has_var_kw:
            res = ask_fn(prompt, model=model, image=image)
        elif len(params) == 2:
            res = ask_fn(prompt, model)
        else:
            res = ask_fn(prompt)

        if asyncio.iscoroutine(res):
            res = await res
        return content_to_str(res)

    async def _execute_edge_on_item(self, edge: Edge, item: ContentItem) -> ContentItem:
        """使用 `edge` 处理单条输入信息 `item`，生成一条带有 `edge.output_tags` 的新信息。"""
        out_tags = normalize_tags(edge.output_tags) or ["output"]
        out_image = item.image
        out_gen = item.gen + 1

        if edge.fn is not None:
            sig = inspect.signature(edge.fn)
            params = [
                p for p in sig.parameters.values() if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)
            ]
            if len(params) >= 2:
                res = edge.fn(item.content, item)
            elif len(params) == 1:
                # 若参数名提示期望完整条目字典，则传入 item，否则传入 item.content（若函数内部当 dict 访问也能自适应）
                param_name = params[0].name.lower()
                if param_name in ("item", "msg", "message", "entry", "info", "context_item"):
                    res = edge.fn(item)
                else:
                    try:
                        res = edge.fn(item.content)
                    except (TypeError, KeyError, AttributeError):
                        res = edge.fn(item)
            else:
                res = edge.fn()

            if asyncio.iscoroutine(res):
                res = await res

            # 如果函数直接返回了一个带 tags/content 的字典，允许动态覆写输出 tags（非常适合条件循环边）
            if isinstance(res, dict) and "content" in res and "tags" in res:
                return ContentItem(
                    tags=res["tags"],
                    content=res["content"],
                    image=res.get("image", out_image),
                    gen=int(res.get("gen", out_gen)),
                )
            out_content = content_to_str(res)
        elif not edge.prompt:
            # 若未提供任何 prompt 或函数，直接透传内容并打上 output_tags
            out_content = item.content
        else:
            rendered = edge.render_prompt(item)
            out_content = await self._call_llm(rendered, model=edge.model, image=out_image)

        return ContentItem(
            tags=out_tags,
            content=out_content,
            image=out_image,
            gen=out_gen,
        )

    # =========================================================================
    # 5. 执行引擎 (`dag.run()` = 跑一次)
    # =========================================================================

    def _has_upstream_producer(self, required_tags: Sequence[str], exclude_edge_idx: int) -> bool:
        """检查图中是否存在其它单条边能够直接产出包含全部 `required_tags` 的信息。"""
        req = set(normalize_tags(required_tags))
        if not req:
            return False
        for idx, e in enumerate(self.edges):
            if idx == exclude_edge_idx:
                continue
            if req.issubset(set(e.output_tags)):
                return True
        return False

    async def run_async(self) -> List[ContentItem]:
        """异步执行一次 DAG (`dag.run()` 的协程实现)。

        执行逻辑：
        - 遍历当前 `dag.edges`：
          1. 若某条边 `edge` 具有 `input_tags` 且在 `dag.contents` 中能找到尚未被该边处理过的匹配信息条目，
             则直接执行该边并把产出的新信息追加到 `dag.contents`。
          2. 若某条边 `edge` 没有输入（`edge.input_tags` 为空，或在 `dag.contents` 中找不到任何单条匹配信息，
             例如需要汇聚两条分支标签或标签未对齐），且当前 `len(dag.contents) >= 2`，
             则从现有 `dag.contents` 中找到两条信息，通过 `self.preset_edge`（预设边）拼凑成一条信息，
             再交由 `edge` 处理产出新信息。
        - 返回本次 `run()` 新产出的所有 `ContentItem` 列表。
        """
        self._normalize_state()
        newly_created: List[ContentItem] = []

        # 第一阶段：按拓扑/定义顺序推进所有能直接匹配单条输入的边，以及需要双条目拼凑的无输入边
        for edge_idx, edge in enumerate(self.edges):
            # 情况 A：edge.input_tags 非空，尝试在现有 contents 中寻找单条直接匹配的输入
            matched_single = False
            if edge.input_tags:
                for c_idx, item in enumerate(list(self.contents)):
                    record_key = (edge_idx, f"single:{c_idx}")
                    if record_key in self._executed_records:
                        continue
                    if item.matches_tags(edge.input_tags):
                        self._executed_records.add(record_key)
                        self._consumed_indices.add(c_idx)
                        out_item = await self._execute_edge_on_item(edge, item)
                        self.contents.append(out_item)
                        newly_created.append(out_item)
                        matched_single = True

            if matched_single:
                continue

            # 情况 B：该边没有单条匹配输入（input_tags 为空，或现有条目中无单条匹配）
            # 检查是否应当触发“从现有的所有条目的信息中找到两条，通过预设边拼凑一个信息”
            should_fallback_merge = False
            if not edge.input_tags:
                should_fallback_merge = True
            elif not self._has_upstream_producer(edge.input_tags, exclude_edge_idx=edge_idx):
                # 没有任何单条上游边能独立产出该 edge 所需的全部 input_tags（例如多路分支标签汇聚，或标签名未对齐）
                should_fallback_merge = True

            if should_fallback_merge and len(self.contents) >= 2:
                pair_sel = self._select_two_items_for_edge(edge_idx, edge)
                if pair_sel is not None:
                    idx1, idx2, item1, item2 = pair_sel
                    self._merged_pairs_by_edge.setdefault(edge_idx, set()).add((idx1, idx2))
                    self._consumed_indices.add(idx1)
                    self._consumed_indices.add(idx2)

                    # 通过初始化时自带的 preset_edge 拼凑两条信息为一条信息
                    synthesized_item = await self._run_preset_edge_on_pair(
                        item1, item2, target_input_tags=edge.input_tags
                    )
                    self.contents.append(synthesized_item)
                    synth_idx = len(self.contents) - 1
                    self._consumed_indices.add(synth_idx)
                    newly_created.append(synthesized_item)

                    # 标记该合成条目已被当前 edge 消费，随后执行当前 edge
                    self._executed_records.add((edge_idx, f"single:{synth_idx}"))
                    out_item = await self._execute_edge_on_item(edge, synthesized_item)
                    self.contents.append(out_item)
                    newly_created.append(out_item)

        # 兜底：如果本轮没有任何边通过情况 A 或情况 B 触发，但存在尚未跑过的有输入边且已有 >=2 条信息，
        # 允许该无输入就绪的边通过预设边从现有信息中找两条拼凑执行
        if not newly_created and len(self.contents) >= 2:
            for edge_idx, edge in enumerate(self.edges):
                pair_sel = self._select_two_items_for_edge(edge_idx, edge)
                if pair_sel is not None:
                    idx1, idx2, item1, item2 = pair_sel
                    self._merged_pairs_by_edge.setdefault(edge_idx, set()).add((idx1, idx2))
                    self._consumed_indices.add(idx1)
                    self._consumed_indices.add(idx2)

                    synthesized_item = await self._run_preset_edge_on_pair(
                        item1, item2, target_input_tags=edge.input_tags
                    )
                    self.contents.append(synthesized_item)
                    synth_idx = len(self.contents) - 1
                    self._consumed_indices.add(synth_idx)
                    newly_created.append(synthesized_item)

                    self._executed_records.add((edge_idx, f"single:{synth_idx}"))
                    out_item = await self._execute_edge_on_item(edge, synthesized_item)
                    self.contents.append(out_item)
                    newly_created.append(out_item)
                    break

        return newly_created

    def run(self) -> List[ContentItem]:
        """跑一次 DAG (`dag.run() = 跑一次`)。

        可直接在同步脚本或外部 `while` 循环中调用；若当前已处于运行中的事件循环内，自动切换至工作线程执行。
        """
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop and loop.is_running():
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                fut = pool.submit(asyncio.run, self.run_async())
                return fut.result()
        return asyncio.run(self.run_async())

    # =========================================================================
    # 6. 外部条件检测与结果提取辅助方法 (一般找 tag="result" 的 content)
    # =========================================================================

    def find_by_tag(self, tag: str = "result") -> List[ContentItem]:
        """在 `dag.context` 中查找所有包含指定 `tag`（默认为 `"result"`）的信息条目。"""
        self._normalize_state()
        return [item for item in self.contents if tag in item.tags]

    def has_tag(self, tag: str = "result") -> bool:
        """检测 `dag.context` 中是否已经出现包含指定 `tag` 的信息条目。"""
        return len(self.find_by_tag(tag)) > 0

    def has_result(self) -> bool:
        """检测是否已达成默认终止条件（存在 `tag="result"` 的条目）。"""
        return self.has_tag("result")

    def get_result(self, tag: str = "result") -> Optional[str]:
        """获取最新一条 `tag="result"` 的 `content` 作为最后回应；若尚未生成则返回 `None`。"""
        matches = self.find_by_tag(tag)
        if not matches:
            return None
        return matches[-1].content

    def get_result_item(self, tag: str = "result") -> Optional[ContentItem]:
        """获取最新一条 `tag="result"` 的完整信息条目对象。"""
        matches = self.find_by_tag(tag)
        if not matches:
            return None
        return matches[-1]

    def to_dict(self) -> Dict[str, Any]:
        """序列化为 V6 标准结构 `{"contents": [...], "edges": [...]}`。"""
        self._normalize_state()
        return {
            "contents": [c.to_dict() for c in self.contents],
            "edges": [e.to_dict() for e in self.edges],
        }

    @classmethod
    def from_dict(
        cls,
        data: Dict[str, Any],
        *,
        llm: Optional[LLMProtocol] = None,
        edge_bindings: Optional[Dict[str, ProcessorType]] = None,
    ) -> "DAG":
        """从 `{"contents": [...], "edges": [...]}` 字典反序列化构造可运行的 `DAG` 实例。"""
        dag = cls(name_or_data=data.get("name", "vea_v6_dag"), llm=llm)
        for c in data.get("contents", []):
            dag.send(c)
        bindings = dict(edge_bindings or {})
        for e_spec in data.get("edges", []):
            edge = Edge.from_any(e_spec)
            # 若提供了动作绑定（例如把某段处理描述绑定到具体的 Python 函数或模型字典），自动替换 processor
            if isinstance(edge.processor, str) and edge.processor in bindings:
                edge.processor = bindings[edge.processor]
            elif edge.prompt in bindings:
                edge.processor = bindings[edge.prompt]
            dag.add_edge(edge)
        return dag

    def __repr__(self) -> str:
        return f"DAG(contents={len(self.contents)}, edges={len(self.edges)})"
