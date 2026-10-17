"""VEA V6 核心有向无环图模块 (DAG).

核心设计规范（V6 全新架构）：
1. **DAG 结构**：`{"contents": [], "edges": []}`
   - `dag.contents`: 存储所有信息条目的列表
   - `dag.edges`: 存储所有处理边的列表
   - `dag.context`: 列出当前所有的 context 条目（即 `dag.contents`）
2. **只用加处理边**：
   每条边为 `(输入的 tag list[str], 处理方式 (prompt: str | {"模型": ..., "处理prompt": ...}), 输出的 tag list[str], layer: int)`；
   处理边不允许没有输入 tag / 输出 tag；`dag.run()` 总是从 `layer` 小的边开始跑起。
3. **信息条目结构**：
   `{"tags": list[str], "content": str, "image"?: path|url, "gen": 0}`
4. **模型注册表与预设边 (`dag.models` / `dag.preset_edge`)**：
   - `DAG(models={...})` 以 model 名为 KEY 保存 `{"api", "key", "model", "params"}`；
     边里的 `{"模型": <KEY>, ...}` 在调用 LLM 时按 KEY 查询注册表并注入配置。
   - 当某条 edge 在 run 时找不到对应输入 tag 的信息，
     则从现有条目中找相近的一条或多条信息，通过预设边（`dag.preset_edge`，默认为模块级常量
     `DEFAULT_PRESET_EDGE`，构造 DAG 时建议显式传入 `preset_edge=DEFAULT_PRESET_EDGE`）
     合成一条 **输出 tag = 该 edge 所需输入 tag** 的新信息，再交由该边完成处理。
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

from vea_v6.llm import LLMProtocol, OpenAICompatibleLLM, SmartMockLLM
from vea_v6.primitives import ContentItem, Edge, ProcessorType, content_to_str, normalize_tags


DEFAULT_PRESET_EDGE_PROMPT = (
    "请将以下两条信息拼凑整合为一条语义连贯的完整信息：\n"
    "信息1 [{tags_1}]: {content_1}\n"
    "信息2 [{tags_2}]: {content_2}"
)

# 模块级默认预设边常量：保存 V6 的默认预设边对象。
# 当某条 edge 在 run 时找不到对应输入 tag 的信息时，用它从现有 contents 中找相近的
# 一条或多条信息，合成一条 输出 tag = 该 edge 所需输入 tag 的新信息。
# 构造 DAG 时应显式传入：`DAG(preset_edge=DEFAULT_PRESET_EDGE, ...)`。
DEFAULT_PRESET_EDGE = Edge(
    input_tags=[],
    processor={
        "模型": "default",
        "处理prompt": DEFAULT_PRESET_EDGE_PROMPT,
    },
    output_tags=["merged"],
    is_preset=True,
    edge_id="preset_edge",
)


def load_models_file(path: Optional[Union[str, "Path"]] = None) -> Dict[str, Dict[str, Any]]:
    """从 `models.json` 读取模型注册表（provider endpoint + api key），供 DAG 直接使用。

    注册表条目格式与 `DAG(models={...})` 一致：`{"api": base_url, "key": api_key, "model": 模型名, "params": {...}}`。
    - 显式传入 `path`：读取指定文件，文件不存在则抛 `FileNotFoundError`。
    - 不传 `path`：按顺序自动查找 `vea_v6/models.json`（包目录）与当前工作目录的 `models.json`；
      都找不到时返回空字典（等价于 `DAG()` 不带注册表）。

    用法：:

        from vea_v6 import DAG, OpenAICompatibleLLM, load_models_file
        dag = DAG(llm=OpenAICompatibleLLM())   # models 自动从 models.json 读取
    """
    from pathlib import Path as _Path

    if path is not None:
        p = _Path(path)
        if not p.exists():
            raise FileNotFoundError(f"模型注册表文件不存在: {p}")
        return json.loads(p.read_text(encoding="utf-8"))
    for candidate in (_Path(__file__).with_name("models.json"), _Path("models.json")):
        if candidate.exists():
            return json.loads(candidate.read_text(encoding="utf-8"))
    return {}


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
        models: Optional[Dict[str, Dict[str, Any]]] = None,
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

        # 模型注册表：以 model 名为 KEY，保存 {"api": base_url, "key": api_key, "model": 模型名, "params": {...}}
        # `add_edge` 中 `{"模型": <KEY>, "处理prompt": ...}` 会在调用 LLM 时按该 KEY 查询此表注入配置。
        # 使用真实 LLM（OpenAICompatibleLLM）且未显式传 `models` 时，自动从 `models.json`
        # （包目录或 CWD）读取 provider endpoint + api key；SmartMockLLM 等测试桩不受影响。
        if models is None and isinstance(llm, OpenAICompatibleLLM):
            models = load_models_file()
        self.models: Dict[str, Dict[str, Any]] = dict(models or {})

        # 预设边对象（默认引用模块级常量 DEFAULT_PRESET_EDGE）：当某条 edge 在 run 时找不到
        # 对应输入 tag 的信息时，用它从现有信息中找相近的一条或多条，合成一条
        # 输出 tag = 该 edge 所需输入 tag 的新信息。推荐构造时显式传入 `preset_edge=DEFAULT_PRESET_EDGE`。
        if preset_edge is not None:
            pe = Edge.from_any(preset_edge)
            pe.is_preset = True
            self.preset_edge: Edge = pe
        else:
            self.preset_edge = DEFAULT_PRESET_EDGE
        # 别名，方便不同命名习惯访问预设边
        self.default_edge: Edge = self.preset_edge

        self.contents: List[ContentItem] = []
        self.edges: List[Edge] = []

        # 记录已执行过的 (edge_index, input_signature) 以免在多轮 run() 循环中重复处理同一输入
        self._executed_records: Set[Tuple[int, str]] = set()
        # 记录哪些信息条目（按其在 contents 中的索引）已经被下游边作为直接输入消费过
        self._consumed_indices: Set[int] = set()
        # 记录已经为哪条 edge 的哪组输入 tag 用预设边合成过条目，避免重复合成
        self._preset_fabricated: Set[Tuple[int, Tuple[str, ...]]] = set()

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
        layer: int = 0,
    ) -> Edge:
        """向 DAG 实例添加一条处理边：`([input_tags], processor, [output_tags], layer=0)`。

        既支持三/四参数调用：
            `dag.add_edge(["query"], "提取关键词: {content}", ["keywords"])`
            `dag.add_edge(["keywords"], {"模型": "gpt-4o", "处理prompt": "总结: {content}"}, ["result"], layer=2)`
        也支持直接传入三元组/四元组或 `Edge` 对象：
            `dag.add_edge((["query"], "提取关键词", ["keywords"], 1))`

        处理边不允许没有输入 tag 或输出 tag（预设边模板除外）；
        若需要汇聚多条信息，请显式写出所需输入 tag，运行时由预设边自动从相近信息合成。
        """
        if processor is None and output_tags is None:
            edge = Edge.from_any(input_tags)  # type: ignore[arg-type]
            if layer:
                edge.layer = int(layer)
        else:
            edge = Edge(
                input_tags=input_tags if not isinstance(input_tags, (Edge, dict)) else [],  # type: ignore[arg-type]
                processor=processor,
                output_tags=output_tags,
                layer=layer,
            )
        if not edge.is_preset:
            if not edge.input_tags:
                raise ValueError(
                    "处理边不允许没有输入 tag（input_tags 不能为空）。"
                    "若需汇聚多条信息，请显式写出所需输入 tag，运行时由预设边自动从相近信息拼凑合成。"
                )
            if not edge.output_tags:
                raise ValueError("处理边不允许没有输出 tag（output_tags 不能为空）。")
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
    # 3. 预设边信息合成机制 (Preset Edge Synthesis：输出 tag = 所需输入 tag)
    # =========================================================================

    def _select_similar_items_for_edge(
        self,
        edge: Edge,
        max_items: int = 2,
    ) -> List[int]:
        """当某条 `edge` 在 run 时找不到对应输入 tag 的信息时，
        从现有的所有 `contents` 条目中找出与其所需输入 tag 最相近的一个或多个条目索引。

        挑选规则：
        1. 按条目 tags 对 `edge.input_tags` 的精确覆盖率 + 语义相似度打分；
        2. 若单条最佳条目已能（精确或语义相近地）覆盖全部所需 tag，只取这一条；
        3. 否则取打分最高且标签互补的前两条（“相近的一个或者多个”）；
        4. 优先选择尚未被消费的叶子条目与 gen 更高的条目；避开预设边合成的中间条目与 result 条目。
        """
        self._normalize_state()
        n = len(self.contents)
        if n == 0:
            return []
        req_tags = list(edge.input_tags)

        def _single_score(i: int) -> Tuple[float, float, int]:
            item = self.contents[i]
            if req_tags:
                exact = len(set(req_tags) & set(item.tags)) / len(req_tags)
                sim = sum(
                    max((default_text_similarity(rt, ct) for ct in item.tags), default=0.0)
                    for rt in req_tags
                ) / len(req_tags)
            else:
                exact = sim = 0.0
            penalty = 0.0
            if "result" in item.tags:
                penalty -= 10.0
            if "_preset_synthesized" in item:
                penalty -= 5.0
            unconsumed = 1.0 if i not in self._consumed_indices else 0.0
            return (round(penalty + exact * 4.0 + sim * 2.0 + unconsumed, 4), float(item.gen), i)

        ranked = sorted(range(n), key=_single_score, reverse=True)
        best_i = ranked[0]

        # 情况 1：边只需要单个输入 tag，且最佳条目已（精确或语义相近地）覆盖它 → 只取这一条
        if len(req_tags) == 1:
            best_tags = set(self.contents[best_i].tags)
            rt = req_tags[0]
            if rt in best_tags or any(default_text_similarity(rt, ct) >= 0.5 for ct in best_tags):
                return [best_i]

        if max_items < 2 or n < 2:
            return [best_i]

        # 情况 2：选一对标签互补、与所需 tag 最相近的条目
        best_score: Optional[Tuple[float, float, int, int, int]] = None
        best_pair: Optional[Tuple[int, int]] = None
        for a in range(n):
            for b in range(a + 1, n):
                item_a, item_b = self.contents[a], self.contents[b]
                combined = set(item_a.tags) | set(item_b.tags)
                if req_tags:
                    exact = len(set(req_tags) & combined) / len(req_tags)
                    sim = sum(
                        max((default_text_similarity(rt, ct) for ct in combined), default=0.0)
                        for rt in req_tags
                    ) / len(req_tags)
                else:
                    exact = sim = 0.0
                penalty = 0.0
                if "result" in combined:
                    penalty -= 10.0
                if "_preset_synthesized" in item_a or "_preset_synthesized" in item_b:
                    penalty -= 5.0
                unconsumed = (1.0 if a not in self._consumed_indices else 0.0) + (
                    1.0 if b not in self._consumed_indices else 0.0
                )
                st = (
                    round(penalty + exact * 4.0 + sim * 2.0 + unconsumed, 4),
                    float(item_a.gen + item_b.gen),
                    a + b,
                    a,
                    b,
                )
                if best_score is None or st > best_score:
                    best_score = st
                    best_pair = (a, b)

        if best_pair is None:
            return [best_i]
        return [best_pair[0], best_pair[1]]

    def _render_preset_prompt(self, pe: Edge, items: List[ContentItem]) -> str:
        """为预设边构造合成提示词（支持 1 条或多条参考信息）。"""
        if len(items) >= 2:
            return pe.render_pair_prompt(items[0], items[1])
        item = items[0]
        c = content_to_str(item.get("content", ""))
        t = ", ".join(item.get("tags", []))
        template = pe.prompt or "请基于以下参考信息生成所需内容："
        return f"{template}\n\n参考信息 [{t}]：\n{c}"

    async def _run_preset_edge_on_items(
        self,
        items: List[ContentItem],
        target_tags: Sequence[str],
    ) -> ContentItem:
        """通过 `self.preset_edge` 把一条或多条相近信息合成一条新信息。

        合成条目的输出 tag = 目标边所需的输入 tag（`target_tags`），
        使该边随后可以把这条合成信息当作自己的直接输入来执行。
        """
        pe = self.preset_edge
        out_tags = normalize_tags(target_tags)
        merged_image = next((it.image for it in items if it.image), None)
        merged_gen = max(it.gen for it in items) + 1

        if pe.fn is not None:
            sig = inspect.signature(pe.fn)
            params = [
                p for p in sig.parameters.values() if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)
            ]
            if len(params) >= 2 and len(items) >= 2:
                res = pe.fn(items[0], items[1])
            else:
                res = pe.fn(items if len(items) > 1 else items[0])
            if asyncio.iscoroutine(res):
                res = await res
            merged_content = content_to_str(res)
        elif isinstance(self.llm, SmartMockLLM) and not self._mock_llm_matches_preset(pe, items):
            # 在默认模拟环境下，使用无损的 JSON/文本智能拼凑，使下游边能直接读取全部参考内容
            merged_content = items[0].content
            for it in items[1:]:
                merged_content = _default_preset_combine_contents(merged_content, it.content)
        else:
            prompt_str = self._render_preset_prompt(pe, items)
            merged_content = await self._call_llm(prompt_str, model=pe.model, image=merged_image)

        synthesized = ContentItem(
            tags=out_tags,
            content=merged_content,
            image=merged_image,
            gen=merged_gen,
        )
        synthesized["_preset_synthesized"] = True
        return synthesized

    def _mock_llm_matches_preset(self, pe: Edge, items: List[ContentItem]) -> bool:
        """检查 SmartMockLLM 是否专门配置了针对 preset_edge 的规则。"""
        if not isinstance(self.llm, SmartMockLLM):
            return True
        prompt_str = self._render_preset_prompt(pe, items)
        for kw in self.llm.rules:
            if kw in prompt_str:
                return True
        if pe.model and pe.model in self.llm.model_rules:
            return True
        return self.llm.fallback_fn is not None

    # =========================================================================
    # 4. 单条处理边的执行逻辑
    # =========================================================================

    def _resolve_model_config(
        self,
        model: Optional[str],
        edge: Optional[Edge] = None,
    ) -> Optional[Dict[str, Any]]:
        """解析一次 LLM 调用的连接配置（注册表 + 边内联，后者覆盖前者）。

        解析顺序：
        1. `edge.inline_llm_config`（边内联的 api/key/model/params）—— 最高优先级；
        2. `dag.models[model]`（模型注册表，`{"api", "key", "model", "params"}`）；
        3. 两者都没有 → 返回 `None`，调用方用 `llm` 对象的默认连接。

        `api`/`base_url`、`key`/`api_key` 两组键名等价；注册表其余键透传为请求参数。
        仅当「注册表里查不到该 KEY，且该边没有内联配置」时才抛 `KeyError`
        （`"default"` 与非注册表形态的真实模型名除外，后者直接当模型名用）。
        """
        inline = edge.inline_llm_config if edge is not None else None
        registry_cfg = self.models.get(model) if (model and self.models) else None

        if registry_cfg is None and inline is None:
            if not model or not self.models:
                # 没有注册表、也没有内联配置：交给 llm 默认值，model 原样传递
                return {"model": model} if model else None
            if model == "default":
                return None
            # 注册表存在但查不到：若 model 看起来是真实模型名（含分隔符或已注册 provider 前缀），
            # 仍允许直连；否则视为配置错误，尽早失败。
            if "/" in model or "." in model or ":" in model:
                return {"model": model}
            raise KeyError(
                f"模型 '{model}' 未在 dag.models 中注册"
                f"（已注册的 KEY: {sorted(self.models)}）；"
                "若这是真实模型名，请在边里内联 api/key，或改用 'provider/model' 形式。"
            )

        merged: Dict[str, Any] = {}
        if registry_cfg:
            params = dict(registry_cfg.get("params") or {})
            for k, v in registry_cfg.items():
                if k not in ("api", "base_url", "key", "api_key", "model", "params"):
                    params[k] = v
            merged["params"] = params
            merged["api_key"] = registry_cfg.get("key", registry_cfg.get("api_key"))
            merged["base_url"] = registry_cfg.get("api", registry_cfg.get("base_url"))
            merged["model"] = registry_cfg.get("model", model)
        else:
            merged = {"params": {}, "model": model}

        if inline:
            # 边里的 model/模型 只有在「该名称不是注册表 KEY」时才算内联模型名覆盖。
            # 否则 {"模型": "laptop-6.8", "参数": {...}} 这种「引用注册表 + 覆盖参数」写法
            # 会把 KEY 当成真实模型名发给 provider（实测导致 404）。
            inline_model = inline.get("model")
            model_is_registry_key = bool(inline_model and self.models.get(inline_model))
            if inline_model and not model_is_registry_key:
                merged["model"] = inline_model
            if inline.get("api_key"):
                merged["api_key"] = inline["api_key"]
            if inline.get("base_url"):
                merged["base_url"] = inline["base_url"]
            if inline.get("params"):
                merged["params"] = {**merged.get("params", {}), **inline["params"]}

        return merged

    async def _call_llm(
        self,
        prompt: str,
        model: Optional[str] = None,
        image: Optional[str] = None,
        edge: Optional[Edge] = None,
    ) -> str:
        """调用大模型后端。

        连接配置按 `edge.inline_llm_config` → `dag.models[model]` → `llm` 默认值 依次解析，
        并把 api_key、base_url 与额外参数注入 `ask(...)`（按 `ask` 签名过滤不支持的参数）。
        """
        ask_fn = getattr(self.llm, "ask", None)
        if ask_fn is None and callable(self.llm):
            ask_fn = self.llm
        if ask_fn is None:
            raise ValueError("当前 DAG 配置的 llm 对象未实现 ask 方法且不可调用")

        cfg = self._resolve_model_config(model, edge=edge)
        target_model = (cfg or {}).get("model", model)
        extra_kwargs: Dict[str, Any] = {}
        if cfg:
            if cfg.get("api_key"):
                extra_kwargs["api_key"] = cfg["api_key"]
            if cfg.get("base_url"):
                extra_kwargs["base_url"] = cfg["base_url"]
            extra_kwargs.update(cfg.get("params") or {})

        sig = inspect.signature(ask_fn)
        params = [
            p for p in sig.parameters.values() if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)
        ]
        has_var_kw = any(p.kind == p.VAR_KEYWORD for p in sig.parameters.values())

        base_kwargs: Dict[str, Any] = {"model": target_model, "image": image}
        if has_var_kw:
            call_kwargs = {**base_kwargs, **extra_kwargs}
        else:
            accepted = {p.name for p in params} - {"prompt"}
            call_kwargs = {k: v for k, v in {**base_kwargs, **extra_kwargs}.items() if k in accepted}

        res = ask_fn(prompt, **call_kwargs)
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
            out_content = await self._call_llm(rendered, model=edge.model, image=out_image, edge=edge)

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

        执行逻辑（**edge 总是从 `layer` 参数小的开始跑起**，同 layer 按定义顺序）：
        1. 若某条边 `edge` 的 `input_tags` 能在 `dag.contents` 中找到尚未被该边处理过的匹配条目，
           则直接执行该边并把产出的新信息追加到 `dag.contents`。
        2. 若本次 run 没有找到 `edge` 所需输入 tag 的条目：
           - 若还有未跑过的上游边能直接产出这些 tag，则先等待（本轮跳过，下一轮再试）；
           - 否则用 `self.preset_edge`（预设边）从现有 `contents` 中找到相近的一条或多条信息，
             合成一条 **输出 tag = 该 edge 所需输入 tag** 的新信息，再交由该 edge 执行。
        - 返回本次 `run()` 新产出的所有 `ContentItem` 列表。
        """
        self._normalize_state()
        newly_created: List[ContentItem] = []

        ordered_indices = sorted(range(len(self.edges)), key=lambda i: (self.edges[i].layer, i))
        for edge_idx in ordered_indices:
            edge = self.edges[edge_idx]
            if edge.is_preset:
                continue

            # 情况 1：存在单条直接匹配输入的条目 → 直接执行
            matched_single = False
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

            # 情况 2：没有找到对应 tag 的信息
            # 若还有未跑过的上游边能直接产出该 edge 所需的全部 tag，本轮先跳过等它产出
            if self._has_upstream_producer(edge.input_tags, exclude_edge_idx=edge_idx):
                continue
            if not self.contents:
                continue
            # 每组输入 tag 只为该 edge 合成一次，避免多轮 run() 重复合成
            fab_key = (edge_idx, tuple(sorted(edge.input_tags)))
            if fab_key in self._preset_fabricated:
                continue

            # 用预设边从现有 contents 中找相近的一条或多条信息，
            # 合成一条 输出 tag = 该 edge 所需输入 tag 的新信息
            sel_indices = self._select_similar_items_for_edge(edge)
            if not sel_indices:
                continue
            items = [self.contents[i] for i in sel_indices]
            synthesized_item = await self._run_preset_edge_on_items(items, edge.input_tags)
            self.contents.append(synthesized_item)
            self._preset_fabricated.add(fab_key)
            self._consumed_indices.update(sel_indices)
            synth_idx = len(self.contents) - 1
            self._consumed_indices.add(synth_idx)
            newly_created.append(synthesized_item)

            # 随后把合成条目作为该 edge 的直接输入执行
            self._executed_records.add((edge_idx, f"single:{synth_idx}"))
            out_item = await self._execute_edge_on_item(edge, synthesized_item)
            self.contents.append(out_item)
            newly_created.append(out_item)

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
        preset_edge: Optional[Union[Edge, Tuple[Any, ...], Dict[str, Any]]] = None,
        models: Optional[Dict[str, Dict[str, Any]]] = None,
        edge_bindings: Optional[Dict[str, ProcessorType]] = None,
    ) -> "DAG":
        """从 `{"contents": [...], "edges": [...]}` 字典反序列化构造可运行的 `DAG` 实例。"""
        dag = cls(
            name_or_data=data.get("name", "vea_v6_dag"),
            llm=llm,
            preset_edge=preset_edge,
            models=models if models is not None else data.get("models"),
        )
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
