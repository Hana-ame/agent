"""VEA V6 核心：`DAG`（纯标签信息流图）。

结构只有两个字段：`{"contents": [], "edges": []}`。

- `dag.send(...)` 送入信息条目；`dag.add_edge(...)` 加处理边（layer 小者先跑）；
- `dag.run()` 跑一次；外部循环检测 `tag="result"` 终止；
- 某条边缺所需输入 tag 时，由**预设边**（`preset_edge`）从相近条目合成一条
  「输出 tag = 所需输入 tag」的中间条目，再交给该边处理；
- `dag.models` 是模型注册表（KEY → `{api, key, model, params}`），边里写 `{"模型": KEY}` 引用，
  也可把 `api/key/model/params` 直接内联在边里（内联优先）。
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

from vea_v6.parts.llm import OpenAICompatibleLLM, SmartMockLLM
from vea_v6.parts.primitives import ContentItem, Edge, ProcessorType, content_to_str, normalize_tags

__all__ = [
    "DAG",
    "DEFAULT_PRESET_EDGE",
    "DEFAULT_PRESET_EDGE_PROMPT",
    "default_text_similarity",
    "load_models_file",
    "combine_contents",
]

DEFAULT_PRESET_EDGE_PROMPT = (
    "请将下面若干条信息拼凑整合为一条语义连贯的完整信息：\n"
    "{pairs}"
)

DEFAULT_PRESET_EDGE = Edge(
    input_tags=[],
    processor={"模型": "default", "处理prompt": DEFAULT_PRESET_EDGE_PROMPT},
    output_tags=["merged"],
    is_preset=True,
    edge_id="preset_edge",
)


def load_models_file(path: Optional[str] = None) -> Dict[str, Dict[str, Any]]:
    """读取模型注册表 `models.json`（包目录优先，其次 CWD）。

    条目格式：`{"api": base_url, "key": api_key, "model": 真实模型名, "params": {...}}`。
    显式给 `path` 且文件不存在时抛 `FileNotFoundError`；自动查找失败返回 `{}`。
    """
    from pathlib import Path

    if path is not None:
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"模型注册表文件不存在: {p}")
        return json.loads(p.read_text(encoding="utf-8"))
    here = Path(__file__).resolve().parent
    for candidate in (here.parent / "models.json", Path.cwd() / "models.json"):
        if candidate.exists():
            return json.loads(candidate.read_text(encoding="utf-8"))
    return {}


def default_text_similarity(text_a: str, text_b: str) -> float:
    """字符 1-gram + 2-gram 余弦相似度，带包含惩罚修正（0..1）。"""
    a = re.sub(r"[\s_\-:：,，。()（）\[\]]+", "", str(text_a).lower().strip())
    b = re.sub(r"[\s_\-:：,，。()（）\[\]]+", "", str(text_b).lower().strip())
    if not a or not b:
        return 0.0
    if a == b:
        return 1.0

    def grams(s: str) -> Counter:
        return Counter(list(s) + [s[i : i + 2] for i in range(len(s) - 1)])

    va, vb = grams(a), grams(b)
    common = set(va) & set(vb)
    dot = sum(va[k] * vb[k] for k in common)
    na = math.sqrt(sum(v * v for v in va.values()))
    nb = math.sqrt(sum(v * v for v in vb.values()))
    cosine = dot / (na * nb) if (na and nb) else 0.0
    shorter, longer = (a, b) if len(a) <= len(b) else (b, a)
    if len(shorter) >= 2 and shorter in longer:
        cosine = max(cosine, 0.55 + 0.35 * len(shorter) / len(longer))
    return round(min(1.0, cosine), 4)


def combine_contents(c1: str, c2: str) -> str:
    """默认合成策略：两个 JSON 字典合并键；否则换行拼接。"""
    s1, s2 = c1.strip(), c2.strip()
    if s1.startswith("{") and s1.endswith("}") and s2.startswith("{") and s2.endswith("}"):
        try:
            d1, d2 = json.loads(s1), json.loads(s2)
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
    """V6 DAG 实例。"""

    def __init__(
        self,
        name_or_data: Union[str, Dict[str, Any], None] = None,
        *,
        contents: Optional[Sequence[Any]] = None,
        edges: Optional[Sequence[Any]] = None,
        llm: Optional[Any] = None,
        preset_edge: Optional[Any] = None,
        models: Optional[Dict[str, Dict[str, Any]]] = None,
    ) -> None:
        if isinstance(name_or_data, dict):
            self.name = str(name_or_data.get("name", "vea_v6_dag"))
            init_contents = name_or_data.get("contents", contents)
            init_edges = name_or_data.get("edges", edges)
        else:
            self.name = str(name_or_data or "vea_v6_dag")
            init_contents, init_edges = contents, edges

        self.llm = llm or SmartMockLLM()
        if models is None and isinstance(llm, OpenAICompatibleLLM):
            models = load_models_file()
        self.models: Dict[str, Dict[str, Any]] = dict(models or {})

        if preset_edge is not None:
            pe = Edge.from_any(preset_edge)
            pe.is_preset = True
            self.preset_edge = pe
        else:
            self.preset_edge = DEFAULT_PRESET_EDGE
        self.default_edge = self.preset_edge

        self.contents: List[ContentItem] = []
        self.edges: List[Edge] = []
        self._executed_records: Set[Tuple[int, str]] = set()
        self._consumed_indices: Set[int] = set()
        self._preset_fabricated: Set[Tuple[int, Tuple[str, ...]]] = set()

        for item in init_contents or []:
            self.send(item)
        for spec in init_edges or []:
            self.add_edge(spec)

    # ---- 视图与字典接口 ---------------------------------------------------
    @property
    def context(self) -> List[ContentItem]:
        self._normalize_state()
        return self.contents

    def _normalize_state(self) -> None:
        for i, item in enumerate(self.contents):
            if not isinstance(item, ContentItem):
                self.contents[i] = ContentItem(
                    item if isinstance(item, dict) else {"content": str(item), "tags": [], "gen": 0}
                )
        for i, e in enumerate(self.edges):
            if not isinstance(e, Edge):
                self.edges[i] = Edge.from_any(e)

    def __getitem__(self, key: str) -> Any:
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

    # ---- 加边 / 送信息 ---------------------------------------------------
    def add_edge(
        self,
        input_tags: Any,
        processor: Optional[ProcessorType] = None,
        output_tags: Optional[Union[str, Sequence[str]]] = None,
        layer: int = 0,
    ) -> Edge:
        """加处理边：`([inputs], processor, [outputs], layer)` 或直接传 `Edge`/元组/字典。"""
        if processor is None and output_tags is None:
            edge = Edge.from_any(input_tags)
            if layer:
                edge.layer = int(layer)
        else:
            edge = Edge(input_tags, processor, output_tags, layer)
        if not edge.is_preset:
            if not edge.input_tags:
                raise ValueError(
                    "处理边不允许没有输入 tag；若需多条信息汇聚，请显式写出所需输入 tag，"
                    "运行时由预设边从相近信息合成。"
                )
            if not edge.output_tags:
                raise ValueError("处理边不允许没有输出 tag。")
        if not edge.id:
            edge.id = f"e_{len(self.edges) + 1}"
        self.edges.append(edge)
        return edge

    def send(
        self,
        info: Any,
        tags: Optional[Union[str, Sequence[str]]] = None,
        image: Optional[str] = None,
        gen: int = 0,
    ) -> ContentItem:
        """送入一条信息：`send({"tags": [...], "content": ...})` 或 `send("文本", tags=[...])`。"""
        if isinstance(info, dict):
            item = ContentItem(
                info,
                tags=tags if tags is not None else info.get("tags", []),
                content=info.get("content", ""),
                image=image if image is not None else info.get("image"),
                gen=int(info.get("gen", gen)),
            )
        else:
            item = ContentItem(tags=tags if tags is not None else ["input"], content=content_to_str(info), image=image, gen=gen)
        self.contents.append(item)
        return item

    add_content = send
    input = send
    feed = send
    push = send

    # ---- 预设边：找相近信息 ----------------------------------------------
    def _select_similar_items_for_edge(self, edge: Edge, max_items: int = 2) -> List[int]:
        """从现有条目里挑出与该边所需输入 tag 最相近的 1~2 条。"""
        self._normalize_state()
        n = len(self.contents)
        if n == 0:
            return []
        req = list(edge.input_tags)

        def score(i: int) -> Tuple[float, float, int]:
            item = self.contents[i]
            if req:
                exact = len(set(req) & set(item.tags)) / len(req)
                sim = sum(max((default_text_similarity(r, t) for t in item.tags), default=0.0) for r in req) / len(req)
            else:
                exact = sim = 0.0
            penalty = -10.0 if "result" in item.tags else 0.0
            if "_preset_synthesized" in item:
                penalty -= 5.0
            fresh = 1.0 if i not in self._consumed_indices else 0.0
            return (round(penalty + exact * 4.0 + sim * 2.0 + fresh, 4), float(item.gen), i)

        ranked = sorted(range(n), key=score, reverse=True)
        best = ranked[0]
        if len(req) == 1:
            if req[0] in self.contents[best].tags or any(
                default_text_similarity(req[0], t) >= 0.5 for t in self.contents[best].tags
            ):
                return [best]
        if max_items < 2 or n < 2:
            return [best]

        best_key: Optional[Tuple[float, float, int]] = None
        best_pair: Optional[Tuple[int, int]] = None
        for a in range(n):
            for b in range(a + 1, n):
                ia, ib = self.contents[a], self.contents[b]
                combined = set(ia.tags) | set(ib.tags)
                if req:
                    exact = len(set(req) & combined) / len(req)
                    sim = sum(max((default_text_similarity(r, t) for t in combined), default=0.0) for r in req) / len(req)
                else:
                    exact = sim = 0.0
                penalty = -10.0 if "result" in combined else 0.0
                if "_preset_synthesized" in ia or "_preset_synthesized" in ib:
                    penalty -= 5.0
                fresh = (a not in self._consumed_indices) + (b not in self._consumed_indices)
                key = (round(penalty + exact * 4.0 + sim * 2.0 + fresh, 4), float(ia.gen + ib.gen), a + b)
                if best_key is None or key > best_key:
                    best_key, best_pair = key, (a, b)
        return list(best_pair) if best_pair else [best]

    def _render_preset_prompt(self, pe: Edge, items: List[ContentItem]) -> str:
        pairs = "\n".join(
            f"信息{i + 1} [{', '.join(it.tags)}]:\n{it.content}" for i, it in enumerate(items)
        )
        template = pe.prompt or DEFAULT_PRESET_EDGE_PROMPT
        if "{pairs}" in template:
            return template.replace("{pairs}", pairs)
        return f"{template}\n\n{pairs}"

    async def _run_preset_edge_on_items(self, items: List[ContentItem], target_tags: Sequence[str]) -> ContentItem:
        """用预设边把 1~n 条相近信息合成一条「输出 tag = 目标边所需输入 tag」的条目。"""
        pe = self.preset_edge
        out_tags = normalize_tags(target_tags)
        image = next((it.image for it in items if it.image), None)
        gen = max(it.gen for it in items) + 1

        if pe.fn is not None:
            res = pe.fn(items if len(items) > 1 else items[0])
            if asyncio.iscoroutine(res):
                res = await res
            merged = content_to_str(res)
        elif isinstance(self.llm, SmartMockLLM) and not self._mock_matches_preset(pe, items):
            merged = items[0].content
            for it in items[1:]:
                merged = combine_contents(merged, it.content)
        else:
            merged = await self._call_llm(self._render_preset_prompt(pe, items), model=pe.model, image=image, edge=pe)

        item = ContentItem(tags=out_tags, content=merged, image=image, gen=gen)
        item["_preset_synthesized"] = True
        return item

    def _mock_matches_preset(self, pe: Edge, items: List[ContentItem]) -> bool:
        if not isinstance(self.llm, SmartMockLLM):
            return True
        prompt = self._render_preset_prompt(pe, items)
        if any(kw in prompt for kw in self.llm.rules):
            return True
        if pe.model and pe.model in self.llm.model_rules:
            return True
        return self.llm.fallback_fn is not None

    # ---- 模型配置解析 ----------------------------------------------------
    def _resolve_model_config(self, model: Optional[str], edge: Optional[Edge] = None) -> Optional[Dict[str, Any]]:
        """解析一次调用要用的连接配置：内联 > 注册表 > llm 默认值。

        返回 `{"model", "api_key", "base_url", "params"}` 或 `None`。
        `{"模型": KEY, "参数": {...}}` 里的 KEY 只有在注册表里查得到时才当 KEY 用，
        否则（真实模型名）直接作为模型名。
        """
        inline = edge.inline_llm_config if edge is not None else None
        registry_cfg = self.models.get(model) if (model and self.models) else None

        if registry_cfg is None and inline is None:
            if not model or not self.models:
                return {"model": model} if model else None
            if model == "default":
                return None
            if "/" in model or "." in model or ":" in model:
                return {"model": model}
            raise KeyError(
                f"模型 '{model}' 未在 dag.models 中注册（已注册: {sorted(self.models)}）；"
                "若这是真实模型名，请在边里内联 api/key。"
            )

        merged: Dict[str, Any] = {"params": {}, "model": model, "api_key": None, "base_url": None}
        if registry_cfg:
            params = dict(registry_cfg.get("params") or {})
            for k, v in registry_cfg.items():
                if k not in ("api", "base_url", "key", "api_key", "model", "params"):
                    params[k] = v
            merged["params"] = params
            merged["api_key"] = registry_cfg.get("key", registry_cfg.get("api_key"))
            merged["base_url"] = registry_cfg.get("api", registry_cfg.get("base_url"))
            merged["model"] = registry_cfg.get("model", model)

        if inline:
            inline_model = inline.get("model")
            is_registry_key = bool(inline_model and self.models.get(inline_model))
            if inline_model and not is_registry_key:
                merged["model"] = inline_model
            if inline.get("key"):
                merged["api_key"] = inline["key"]
            if inline.get("api"):
                merged["base_url"] = inline["api"]
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
        """调用模型；按 `edge.inline_llm_config → dag.models → llm 默认值` 注入连接配置。"""
        ask_fn = getattr(self.llm, "ask", None) or (self.llm if callable(self.llm) else None)
        if ask_fn is None:
            raise ValueError("当前 llm 对象既没有 ask 方法也不可调用")

        cfg = self._resolve_model_config(model, edge=edge)
        target = (cfg or {}).get("model", model)
        extra: Dict[str, Any] = {}
        if cfg:
            if cfg.get("api_key"):
                extra["api_key"] = cfg["api_key"]
            if cfg.get("base_url"):
                extra["base_url"] = cfg["base_url"]
            extra.update(cfg.get("params") or {})

        sig = inspect.signature(ask_fn)
        has_var_kw = any(p.kind == p.VAR_KEYWORD for p in sig.parameters.values())
        accepted = {p.name for p in sig.parameters.values() if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)}
        kv = {"model": target, "image": image, **extra}
        if not has_var_kw:
            kv = {k: v for k, v in kv.items() if k in accepted}
        res = ask_fn(prompt, **kv)
        if asyncio.iscoroutine(res):
            res = await res
        return content_to_str(res)

    # ---- 单边执行 --------------------------------------------------------
    async def _execute_edge_on_item(self, edge: Edge, item: ContentItem) -> ContentItem:
        out_tags = normalize_tags(edge.output_tags) or ["output"]
        out_image = item.image
        out_gen = item.gen + 1

        if edge.fn is not None:
            sig = inspect.signature(edge.fn)
            params = [p for p in sig.parameters.values() if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)]
            if len(params) >= 2:
                res = edge.fn(item.content, item)
            elif len(params) == 1:
                name = params[0].name.lower()
                if name in ("item", "msg", "message", "entry", "info", "context_item"):
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
            if isinstance(res, dict) and "content" in res and "tags" in res:
                return ContentItem(
                    tags=res["tags"],
                    content=res["content"],
                    image=res.get("image", out_image),
                    gen=int(res.get("gen", out_gen)),
                )
            out_content = content_to_str(res)
        elif not edge.prompt and not edge.model:
            out_content = item.content
        else:
            out_content = await self._call_llm(edge.render_prompt(item), model=edge.model, image=out_image, edge=edge)

        return ContentItem(tags=out_tags, content=out_content, image=out_image, gen=out_gen)

    # ---- 运行引擎 --------------------------------------------------------
    def _has_upstream_producer(self, required: Sequence[str], exclude_edge_idx: int) -> bool:
        req = set(normalize_tags(required))
        if not req:
            return False
        for idx, e in enumerate(self.edges):
            if idx != exclude_edge_idx and req.issubset(set(e.output_tags)):
                return True
        return False

    async def run_async(self) -> List[ContentItem]:
        """跑一次。返回本轮新产出的条目。"""
        self._normalize_state()
        created: List[ContentItem] = []
        order = sorted(range(len(self.edges)), key=lambda i: (self.edges[i].layer, i))

        for idx in order:
            edge = self.edges[idx]
            if edge.is_preset:
                continue

            matched = False
            for c_idx, item in enumerate(list(self.contents)):
                if set(edge.input_tags) & set(item.tags):
                    sig = f"{c_idx}:{item.gen}"
                    if (idx, sig) in self._executed_records:
                        continue
                    self._executed_records.add((idx, sig))
                    self._consumed_indices.add(c_idx)
                    out = await self._execute_edge_on_item(edge, item)
                    self.contents.append(out)
                    created.append(out)
                    matched = True
                    break
            if matched:
                continue

            # 没有直接匹配：若上游还能产出所需 tag，则本轮先等
            if self._has_upstream_producer(edge.input_tags, exclude_edge_idx=idx):
                pending = any(
                    (idx, f"{c}:{it.gen}") not in self._executed_records
                    for c, it in enumerate(self.contents)
                    if set(edge.input_tags) & set(it.tags)
                )
                if pending:
                    continue

            key = (idx, tuple(edge.input_tags))
            if key in self._preset_fabricated:
                continue
            picks = self._select_similar_items_for_edge(edge)
            if not picks:
                continue
            self._preset_fabricated.add(key)
            items = [self.contents[i] for i in picks]
            synth = await self._run_preset_edge_on_items(items, edge.input_tags)
            self.contents.append(synth)
            created.append(synth)
            self._executed_records.add((idx, f"single:{len(self.contents) - 1}"))
            out = await self._execute_edge_on_item(edge, synth)
            self.contents.append(out)
            created.append(out)
        return created

    def run(self) -> List[ContentItem]:
        """同步跑一次（已有事件循环时自动切线程）。"""
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None
        if loop and loop.is_running():
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                return pool.submit(asyncio.run, self.run_async()).result()
        return asyncio.run(self.run_async())

    def run_sync(self, coro: Any) -> Any:
        """在同步上下文里跑一个协程（已有事件循环时切线程）。"""
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(coro)
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            return pool.submit(asyncio.run, coro).result()

    def ask_sync(self, prompt: str, model: Optional[str] = None) -> str:
        """同步问一句（内部起事件循环；只给确定性边 / 脚本用）。"""
        return self.run_sync(self._call_llm(prompt, model=model))

    # ---- 结果查询 --------------------------------------------------------
    def find_by_tag(self, tag: str = "result") -> List[ContentItem]:
        self._normalize_state()
        return [it for it in self.contents if tag in it.tags]

    def has_tag(self, tag: str = "result") -> bool:
        return bool(self.find_by_tag(tag))

    def has_result(self) -> bool:
        return self.has_tag("result")

    def get_result(self, tag: str = "result") -> Optional[str]:
        items = self.find_by_tag(tag)
        return items[-1].content if items else None

    def get_result_item(self, tag: str = "result") -> Optional[ContentItem]:
        items = self.find_by_tag(tag)
        return items[-1] if items else None

    # ---- 序列化 ----------------------------------------------------------
    def to_dict(self) -> Dict[str, Any]:
        self._normalize_state()
        return {"contents": [c.to_dict() for c in self.contents], "edges": [e.to_dict() for e in self.edges]}

    @classmethod
    def from_dict(cls, data: Dict[str, Any], **kwargs: Any) -> "DAG":
        return cls(data, **kwargs)

    def __repr__(self) -> str:
        return f"DAG(name={self.name!r}, contents={len(self.contents)}, edges={len(self.edges)})"

