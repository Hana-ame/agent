"""VEA V6 元构建器：一个"用来构建 DAG 的 DAG"（`meta_builder_dag`）。

输入的描述被解析成边列表与标签对齐信息，两者由预设边拼成一条中间条目，
再编译成目标 DAG 的 `{"contents": [], "edges": [...]}`。
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Union

from vea_v6.parts.dag import DEFAULT_PRESET_EDGE, DAG, default_text_similarity
from vea_v6.parts.primitives import ContentItem, Edge, ProcessorType

__all__ = [
    "parse_edges_from_description",
    "parse_json_stream",
    "create_dag_builder_dag",
    "BuilderResult",
    "MetaDAGBuilder",
]


def parse_json_stream(text: str) -> List[Any]:
    """从可能含多段 JSON / 前后夹杂自然语言的文本里，顺序解出所有 JSON 值。

    预设边合并两条信息后，正文常是"JSON\\nJSON"或"提示词\\n\\n输入数据：\\nJSON\\nJSON"。
    直接 `json.loads` 会因为 Extra data 失败；这里用 `raw_decode` 扫一遍，逐段取。
    """
    out: List[Any] = []
    i, n = 0, len(text or "")
    dec = json.JSONDecoder()
    while i < n:
        while i < n and text[i] not in "{[":
            i += 1
        if i >= n:
            break
        try:
            obj, end = dec.raw_decode(text, i)
        except json.JSONDecodeError:
            i += 1
            continue
        out.append(obj)
        i = end
    return out

_V6_EDGE_SENTENCE_PATTERNS = [
    re.compile(
        r"^(?:创建|添加|步骤\d*[:：.\s]*)?(?:从|由|把|将)\s*(?P<src>(?:\[[^\]]*\]|[^\s,，。;；])+?)\s*"
        r"(?:经过|通过|经由|利用|使用|执行|进行|做)\s*(?P<act>.+?)\s*"
        r"(?:到|至|得到|生成|产出|输出为|输出|变为|转化为)\s*(?P<dst>(?:\[[^\]]*\]|[^\s,，。;；\}])+?)$"
    ),
    re.compile(
        r"^(?:创建|添加|步骤\d*[:：.\s]*)?(?:无输入|空输入)?\s*"
        r"(?:经过|通过|经由|利用|使用|执行|进行|做)\s*(?P<act>.+?)\s*"
        r"(?:到|至|得到|生成|产出|输出为|输出|变为|转化为)\s*(?P<dst>(?:\[[^\]]*\]|[^\s,，。;；\}])+?)$"
    ),
    re.compile(r"^(?P<src>[^\-\=\>]*?)\s*[\-\=]+\[\s*(?P<act>.+?)\s*\][\-\=]*\>\s*(?P<dst>.+?)$"),
    re.compile(r"^(?P<src>[^\-\>]*?)\s*\-\>\s*(?P<act>[^\-\>]+?)\s*\-\>\s*(?P<dst>[^\-\>]+?)$"),
]


def _parse_tag_list_expr(raw: str) -> List[str]:
    s = raw.strip().strip(" '\"\t")
    if not s or s in ("[]", "无", "无输入", "空", "空输入", "None", "none", "null"):
        return []
    if s.startswith("[") and s.endswith("]"):
        s = s[1:-1].strip()
    if not s:
        return []
    if re.search(r"[,，、;；\+]+", s):
        return [p.strip(" '\"\t") for p in re.split(r"[,，、;；\+]+", s) if p.strip(" '\"\t")]
    return [s]


def _parse_processor_expr(raw: str) -> Union[str, Dict[str, Any]]:
    s = raw.strip().strip(" '\"\t")
    if s.startswith("{") and s.endswith("}"):
        try:
            parsed = json.loads(s)
            if isinstance(parsed, dict):
                return parsed
        except Exception:
            d: Dict[str, Any] = {}
            for pair in re.split(r"[,，;；]+", s[1:-1].strip()):
                if ":" in pair or "：" in pair:
                    k, v = re.split(r"[:：]", pair, maxsplit=1)
                    d[k.strip(" '\"\t")] = v.strip(" '\"\t")
            if d:
                return d
    return s


def parse_edges_from_description(description: Union[str, Sequence[Any], Dict[str, Any]]) -> List[Dict[str, Any]]:
    """把自然语言 / 句子列表 / JSON 解析成 V6 边定义列表。"""
    if isinstance(description, dict):
        if "edges" in description:
            return [Edge.from_any(e).to_dict() for e in description["edges"]]
        return [Edge.from_any(description).to_dict()]
    if isinstance(description, (list, tuple)) and description and not isinstance(description[0], str):
        return [Edge.from_any(e).to_dict() for e in description]
    if isinstance(description, str):
        stripped = description.strip()
        if stripped.startswith(("{", "[")) and stripped.endswith(("}", "]")):
            try:
                parsed = json.loads(stripped)
                if isinstance(parsed, dict) and "edges" in parsed:
                    return [Edge.from_any(e).to_dict() for e in parsed["edges"]]
                if isinstance(parsed, list):
                    return [Edge.from_any(e).to_dict() for e in parsed]
            except Exception:
                pass
        normalized = re.sub(
            r"(?<=[a-zA-Z0-9_\]\u4e00-\u9fa5])\s+(?=从\s+[^\s]+\s+(?:经过|通过|经由))", "\n", stripped
        )
        lines = re.split(r"[\n;；。]+", normalized)
    else:
        lines = []
        for item in description:
            lines.extend(re.split(r"[\n;；。]+", str(item)))

    out: List[Dict[str, Any]] = []
    for line in lines:
        clean = re.sub(r"^(?:\d+[\.\、\)\s]+|[\-\*\u2022]\s+)", "", line.strip()).strip()
        if not clean:
            continue
        for pat in _V6_EDGE_SENTENCE_PATTERNS:
            m = pat.match(clean)
            if m:
                gd = m.groupdict()
                out.append(
                    Edge(
                        input_tags=_parse_tag_list_expr(gd.get("src") or ""),
                        processor=_parse_processor_expr(gd.get("act") or ""),
                        output_tags=_parse_tag_list_expr(gd.get("dst") or ""),
                    ).to_dict()
                )
                break
        else:
            raise ValueError(
                f"无法从描述中解析出 V6 处理边: '{clean}'。"
                "请用 '从 [tag1] 经过 处理方式 到 [result]' 或 '[tag1] -> 处理方式 -> [result]' 格式。"
            )
    return out


def create_dag_builder_dag(similarity_threshold: float = 0.45, ensure_result_tag: bool = True) -> DAG:
    """创建元构建 DAG（本身是标准 V6 DAG）。"""
    builder = DAG("V6_Meta_DAG_Builder", preset_edge=DEFAULT_PRESET_EDGE)

    def _edge1_parse_edges(content: str) -> str:
        return json.dumps({"parsed_edges": parse_edges_from_description(content)}, ensure_ascii=False)

    def _edge2_analyze_alignment(content: str) -> str:
        parsed = parse_edges_from_description(content)
        produced: List[str] = []
        consumed: List[str] = []
        for e in parsed:
            for t in e["output_tags"]:
                if t not in produced:
                    produced.append(t)
            for t in e["input_tags"]:
                if t not in consumed:
                    consumed.append(t)
        aliases: Dict[str, str] = {}
        logs: List[str] = []
        for c in consumed:
            if c in produced:
                continue
            components = [p for p in produced if len(p) >= 2 and p in c and p != c]
            if len(components) >= 2:
                logs.append(f"复合输入标签 '{c}' 拆解为 {components}（运行时由预设边拼凑）")
                continue
            best, best_sim = None, 0.0
            for p in produced:
                sim = default_text_similarity(p, c)
                if sim >= similarity_threshold and sim > best_sim:
                    best, best_sim = p, sim
            if best:
                aliases[c] = best
                logs.append(f"相似标签对齐: '{c}' ↔ '{best}' (sim={best_sim:.2f})")
        return json.dumps(
            {"tag_aliases": aliases, "alignment_logs": logs, "ensure_result_tag": ensure_result_tag},
            ensure_ascii=False,
        )

    def _edge3_compile(merged: str) -> str:
        # 合并后的正文可能是"若干段 JSON 顺序拼接"（预设边把两条都塞进 `输入数据：`），
        # 只 `json.loads` 整段会报 Extra data。逐个 JSON 对象解出来再按 key 合并。
        parsed = parse_json_stream(merged)
        data: Dict[str, Any] = {}
        for obj in parsed:
            if isinstance(obj, dict):
                data.update(obj)
        raw_edges: List[Dict[str, Any]] = data.get("parsed_edges", [])
        aliases: Dict[str, str] = data.get("tag_aliases", {})
        logs: List[str] = data.get("alignment_logs", [])
        produce_all = {t for e in raw_edges for t in e.get("output_tags", [])}

        compiled: List[Dict[str, Any]] = []
        for e in raw_edges:
            new_in: List[str] = []
            for t in e.get("input_tags", []):
                comps = [p for p in produce_all if len(p) >= 2 and p in t and p != t]
                targets = sorted(comps) if len(comps) >= 2 else [aliases.get(t, t)]
                for x in targets:
                    if x not in new_in:
                        new_in.append(x)
            compiled.append(
                {
                    "input_tags": new_in,
                    "processor": e.get("processor", ""),
                    "output_tags": list(e.get("output_tags", [])),
                    "layer": len(compiled) + 1,
                }
            )
        if data.get("ensure_result_tag", True) and compiled:
            if not any("result" in e["output_tags"] for e in compiled):
                consumed_after = {t for e in compiled for t in e["input_tags"]}
                sinks = [e for e in compiled if not any(o in consumed_after for o in e["output_tags"])]
                (sinks[-1] if sinks else compiled[-1])["output_tags"].append("result")
        return json.dumps(
            {"name": "Built_V6_DAG", "contents": [], "edges": compiled, "alignment_logs": logs},
            ensure_ascii=False,
        )

    builder.add_edge(["input"], _edge1_parse_edges, ["parsed_edges"], layer=1)
    builder.add_edge(["input"], _edge2_analyze_alignment, ["tag_alignment"], layer=1)
    # 两路汇聚：没有单条信息同时带这两个 tag → 运行时由预设边合成
    builder.add_edge(["parsed_edges", "tag_alignment"], _edge3_compile, ["result"], layer=2)
    return builder


@dataclass
class BuilderResult:
    compiled_dag: DAG
    dag_dict: Dict[str, Any]
    alignment_logs: List[str] = field(default_factory=list)
    builder_context: List[ContentItem] = field(default_factory=list)


class MetaDAGBuilder:
    """用元构建 DAG 构建目标业务 DAG。"""

    def __init__(
        self,
        similarity_threshold: float = 0.45,
        ensure_result_tag: bool = True,
        edge_bindings: Optional[Dict[str, ProcessorType]] = None,
        llm: Optional[Any] = None,
    ) -> None:
        self.similarity_threshold = similarity_threshold
        self.ensure_result_tag = ensure_result_tag
        self.edge_bindings = dict(edge_bindings or {})
        self.llm = llm
        self.meta_dag = create_dag_builder_dag(similarity_threshold, ensure_result_tag)

    def build(self, description: Any, initial_contents: Optional[Sequence[Any]] = None) -> BuilderResult:
        self.meta_dag = create_dag_builder_dag(self.similarity_threshold, self.ensure_result_tag)
        if isinstance(description, str):
            text = description
        elif isinstance(description, (list, tuple)) and all(isinstance(x, str) for x in description):
            text = "\n".join(description)
        else:
            text = json.dumps(description, ensure_ascii=False)
        self.meta_dag.send({"tags": ["input"], "content": text, "gen": 0})
        ticks = 0
        while not self.meta_dag.has_result() and ticks < 10:
            self.meta_dag.run()
            ticks += 1
        raw = self.meta_dag.get_result()
        if not raw:
            raise RuntimeError("元构建 DAG 未能产出 tag='result'")
        data = json.loads(raw)
        compiled = DAG({"contents": [], "edges": data.get("edges", [])}, llm=self.llm)
        for item in initial_contents or []:
            compiled.send(item)
        return BuilderResult(
            compiled_dag=compiled,
            dag_dict={"contents": compiled.to_dict()["contents"], "edges": compiled.to_dict()["edges"]},
            alignment_logs=data.get("alignment_logs", []),
            builder_context=list(self.meta_dag.context),
        )

    merge = build
