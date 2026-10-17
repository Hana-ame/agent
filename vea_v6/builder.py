"""VEA V6 “用来构建 DAG 的 DAG” (Meta-DAG Builder).

对应需求：
> “和 V5 一样，这次的完全新版，也需要附赠一个 dag 是用来构建 dag 的 dag。”

本模块提供一个标准的 VEA V6 `DAG` 实例（由 `create_dag_builder_dag()` 创建），
它的任务就是接收自然语言描述、分立句子或边配置信息，通过自身的 V6 处理边与预设边（`preset_edge`）运行，
最终在 `tag="result"` 的条目中产出构建好的目标 V6 `DAG`（`{"contents": [], "edges": [...]}`）！

元图 `builder_dag` 的内部拓扑（自身即是一个纯粹的 V6 DAG）：
1. **边 1 (`["input"]` -> `["parsed_edges"]`)**：
   解析输入文本/句子，提取出每条处理边的 `(input_tags, processor, output_tags)`。
2. **边 2 (`["input"]` -> `["tag_alignment"]`)**：
   分析全局标签流向、相似标签对齐关系（例如上下游名字相近的 tag 自动对齐）以及终止标签 `"result"` 保障策略。
3. **边 3 (`[]` / `["parsed_edges", "tag_alignment"]` -> `["result"]`)**：
   该边没有单条直接输入，因此自动通过 `builder_dag.preset_edge`（预设边）将前两条边产出的
   `parsed_edges` 与 `tag_alignment` 两条信息拼凑为一条完整规格信息，并编译生成目标 V6 `DAG` 的
   `{"contents": [], "edges": [...]}` JSON 结构，打上 `tags=["result"]` 作为终止信号和最终回应！
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

from vea_v6.dag import DAG, default_text_similarity
from vea_v6.llm import LLMProtocol
from vea_v6.primitives import ContentItem, Edge, ProcessorType, normalize_tags


# 支持多种中英文句子描述 V6 边的正则模式
_V6_EDGE_SENTENCE_PATTERNS = [
    # 1. 标准句式：从 X 经过/通过/使用 Y 到/输出/得到 Z
    re.compile(
        r"^(?:创建|添加|步骤\d*[:：.\s]*)?(?:从|由|把|将)\s*(?P<src>(?:\[[^\]]*\]|[^\s,，。;；])+?)\s*"
        r"(?:经过|通过|经由|利用|使用|执行|进行|做)\s*(?P<act>.+?)\s*"
        r"(?:到|至|得到|生成|产出|输出为|输出|变为|转化为)\s*(?P<dst>(?:\[[^\]]*\]|[^\s,，。;；\}])+?)$"
    ),
    # 2. 无输入句式（触发预设边）：无输入/空输入 经过 Y 到 Z，或 直接“经过 Y 到 Z”
    re.compile(
        r"^(?:创建|添加|步骤\d*[:：.\s]*)?(?:无输入|空输入)?\s*"
        r"(?:经过|通过|经由|利用|使用|执行|进行|做)\s*(?P<act>.+?)\s*"
        r"(?:到|至|得到|生成|产出|输出为|输出|变为|转化为)\s*(?P<dst>(?:\[[^\]]*\]|[^\s,，。;；\}])+?)$"
    ),
    # 3. 箭头三段式：[a, b] --[prompt]--> [c] 或 a -> prompt -> b
    re.compile(
        r"^(?P<src>[^\-\=\>]*?)\s*[\-\=]+\[\s*(?P<act>.+?)\s*\][\-\=]*\>\s*(?P<dst>.+?)$"
    ),
    re.compile(
        r"^(?P<src>[^\-\>]*?)\s*\-\>\s*(?P<act>[^\-\>]+?)\s*\-\>\s*(?P<dst>[^\-\>]+?)$"
    ),
]


def _parse_tag_list_expr(raw_tag_str: str) -> List[str]:
    """解析句子中的标签列表表达式，如 `'[a, b]'`、`'a与b'`、`'a, b'`、`'无'`/`'[]'`。"""
    s = raw_tag_str.strip().strip(" '\"\t")
    if not s or s in ("[]", "无", "无输入", "空", "空输入", "None", "none", "null"):
        return []
    if s.startswith("[") and s.endswith("]"):
        s = s[1:-1].strip()
    if not s:
        return []
    # 如果包含逗号、顿号、分号或加号，按分隔符拆分为多个 tag
    if re.search(r"[,，、;；\+]+", s):
        parts = re.split(r"[,，、;；\+]+", s)
        return normalize_tags([p.strip(" '\"\t") for p in parts if p.strip(" '\"\t")])
    return normalize_tags([s])


def _parse_processor_expr(raw_act_str: str) -> Union[str, Dict[str, Any]]:
    """解析句子中的处理方式表达式：支持普通 prompt 字符串，或 `{模型: ..., 处理prompt: ...}` 字典写法。"""
    s = raw_act_str.strip().strip(" '\"\t")
    if s.startswith("{") and s.endswith("}"):
        try:
            parsed = json.loads(s)
            if isinstance(parsed, dict):
                return parsed
        except Exception:
            # 支持简写中文格式如 {模型: gpt-4o, 处理prompt: 提取摘要}
            inner = s[1:-1].strip()
            kv_pairs = re.split(r"[,，;；]+", inner)
            d: Dict[str, Any] = {}
            for pair in kv_pairs:
                if ":" in pair or "：" in pair:
                    k, v = re.split(r"[:：]", pair, maxsplit=1)
                    d[k.strip(" '\"\t")] = v.strip(" '\"\t")
            if d:
                return d
    return s


def parse_edges_from_description(
    description: Union[str, Sequence[Any], Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """从自然语言描述、分立句子列表或 JSON 结构中解析出 V6 边定义列表。"""
    if isinstance(description, dict):
        if "edges" in description:
            return [Edge.from_any(e).to_dict() for e in description["edges"]]
        return [Edge.from_any(description).to_dict()]

    if isinstance(description, (list, tuple)) and description and not isinstance(description[0], str):
        return [Edge.from_any(e).to_dict() for e in description]

    if isinstance(description, str):
        stripped = description.strip()
        if (stripped.startswith("{") and stripped.endswith("}")) or (
            stripped.startswith("[") and stripped.endswith("]")
        ):
            try:
                parsed_json = json.loads(stripped)
                if isinstance(parsed_json, dict) and "edges" in parsed_json:
                    return [Edge.from_any(e).to_dict() for e in parsed_json["edges"]]
                if isinstance(parsed_json, list):
                    return [Edge.from_any(e).to_dict() for e in parsed_json]
            except Exception:
                pass

        normalized = re.sub(
            r"(?<=[a-zA-Z0-9_\]\u4e00-\u9fa5])\s+(?=从\s+[^\s]+\s+(?:经过|通过|经由))",
            "\n",
            stripped,
        )
        raw_lines = re.split(r"[\n;；。]+", normalized)
    else:
        raw_lines = []
        for item in description:
            normalized = re.sub(
                r"(?<=[a-zA-Z0-9_\]\u4e00-\u9fa5])\s+(?=从\s+[^\s]+\s+(?:经过|通过|经由))",
                "\n",
                str(item),
            )
            raw_lines.extend(re.split(r"[\n;；。]+", normalized))

    edges_list: List[Dict[str, Any]] = []
    for line in raw_lines:
        clean = line.strip()
        clean = re.sub(r"^(?:\d+[\.\、\)\s]+|[\-\*\•]\s+)", "", clean).strip()
        if not clean:
            continue

        matched = False
        for pat in _V6_EDGE_SENTENCE_PATTERNS:
            m = pat.match(clean)
            if m:
                gd = m.groupdict()
                src_raw = gd.get("src") or ""
                act_raw = gd.get("act") or ""
                dst_raw = gd.get("dst") or ""
                in_tags = _parse_tag_list_expr(src_raw)
                proc = _parse_processor_expr(act_raw)
                out_tags = _parse_tag_list_expr(dst_raw)
                edges_list.append(
                    Edge(input_tags=in_tags, processor=proc, output_tags=out_tags).to_dict()
                )
                matched = True
                break

        if not matched:
            raise ValueError(
                f"无法从描述中解析出 V6 处理边: '{clean}'。"
                "请使用类似 '从 [tag1] 经过 处理方式 到 [result]' 或 '[tag1] -> 处理方式 -> [result]' 的格式。"
            )

    return edges_list


def create_dag_builder_dag(
    similarity_threshold: float = 0.45,
    ensure_result_tag: bool = True,
) -> DAG:
    """创建并返回一个“用来构建 DAG 的 V6 DAG” (`meta_builder_dag`)。

    运行方式与普通 V6 DAG 完全一致：
        ```python
        builder_dag = create_dag_builder_dag()
        builder_dag.send({"tags": ["input"], "content": "从 x1 经过 提取摘要 到 result", "gen": 0})
        while not builder_dag.has_result():
            builder_dag.run()
        built_dag_dict = json.loads(builder_dag.get_result())
        target_dag = DAG.from_dict(built_dag_dict)
        ```
    """
    builder_dag = DAG("V6_Meta_DAG_Builder")

    # 边 1：从 ["input"] 解析出分立的 V6 处理边列表 ["parsed_edges"]
    def _edge1_parse_edges(content: str) -> str:
        parsed_edges = parse_edges_from_description(content)
        return json.dumps({"parsed_edges": parsed_edges}, ensure_ascii=False)

    # 边 2：从 ["input"] 分析标签流向、相似标签对齐与终止标签 ["tag_alignment"]
    def _edge2_analyze_alignment(content: str) -> str:
        parsed_edges = parse_edges_from_description(content)
        produced_tags: List[str] = []
        consumed_tags: List[str] = []
        for e in parsed_edges:
            for t in e["output_tags"]:
                if t not in produced_tags:
                    produced_tags.append(t)
            for t in e["input_tags"]:
                if t not in consumed_tags:
                    consumed_tags.append(t)

        # 找出单标签输入中名称不完全一致但高度相似的上下游 tag 对（例如 "商品价格列表" vs "商品价格清单"）
        tag_aliases: Dict[str, str] = {}
        alignment_logs: List[str] = []
        for c_tag in consumed_tags:
            if c_tag in produced_tags:
                continue
            # 检查是否是多分支组合描述（如 "原价小计与优惠折扣信息" 同时包含两个上游产出 tag）
            matched_components = [
                p_tag for p_tag in produced_tags if len(p_tag) >= 2 and p_tag in c_tag and p_tag != c_tag
            ]
            if len(matched_components) >= 2:
                # 拆分为多标签输入，以便在 V6 运行时自然触发 preset_edge 拼凑两条上游信息！
                alignment_logs.append(
                    f"复合输入标签 '{c_tag}' 拆解为多标签输入 {matched_components}（运行时由预设边拼凑两条信息）"
                )
                continue

            best_p: Optional[str] = None
            best_sim = 0.0
            for p_tag in produced_tags:
                sim = default_text_similarity(p_tag, c_tag)
                if sim >= similarity_threshold and sim > best_sim:
                    best_sim = sim
                    best_p = p_tag
            if best_p is not None:
                tag_aliases[c_tag] = best_p
                alignment_logs.append(
                    f"相似标签对齐: 输入标签 '{c_tag}' 与上游输出标签 '{best_p}' 语义相似 (sim={best_sim:.2f})"
                )

        return json.dumps(
            {
                "tag_aliases": tag_aliases,
                "alignment_logs": alignment_logs,
                "ensure_result_tag": ensure_result_tag,
            },
            ensure_ascii=False,
        )

    # 边 3：无单条匹配输入（需要组合 ["parsed_edges"] 与 ["tag_alignment"] 两条信息），
    # 自动通过 builder_dag.preset_edge 拼凑后，编译产出目标 DAG 结构（输出 tag=["result"]）
    def _edge3_compile_target_dag(merged_content: str) -> str:
        data = json.loads(merged_content)
        raw_edges: List[Dict[str, Any]] = data.get("parsed_edges", [])
        tag_aliases: Dict[str, str] = data.get("tag_aliases", {})
        alignment_logs: List[str] = data.get("alignment_logs", [])
        should_ensure_result: bool = bool(data.get("ensure_result_tag", True))

        # 收集所有上游产出标签
        all_produced = {t for e in raw_edges for t in e.get("output_tags", [])}

        compiled_edges: List[Dict[str, Any]] = []
        for e in raw_edges:
            in_tags = list(e.get("input_tags", []))
            out_tags = list(e.get("output_tags", []))
            proc = e.get("processor", "")

            new_in_tags: List[str] = []
            for t in in_tags:
                # 若单个标签包含多个上游产出标签名（如 "原价小计与优惠折扣信息"），自动展开为两个输入 tag，
                # 从而在目标 V6 DAG 运行时触发 preset_edge 从现有信息中找到这两条并拼凑！
                components = [p for p in all_produced if len(p) >= 2 and p in t and p != t]
                if len(components) >= 2:
                    for comp in sorted(components):
                        if comp not in new_in_tags:
                            new_in_tags.append(comp)
                elif t in tag_aliases:
                    mapped = tag_aliases[t]
                    if mapped not in new_in_tags:
                        new_in_tags.append(mapped)
                else:
                    if t not in new_in_tags:
                        new_in_tags.append(t)

            compiled_edges.append(
                {
                    "input_tags": new_in_tags,
                    "processor": proc,
                    "output_tags": out_tags,
                }
            )

        # 确保最终汇点边带有 "result" tag，方便外部循环通过检测 tag="result" 终止
        if should_ensure_result and compiled_edges:
            has_any_result = any("result" in e["output_tags"] for e in compiled_edges)
            if not has_any_result:
                consumed_after = {t for e in compiled_edges for t in e["input_tags"]}
                sink_edges = [
                    e
                    for e in compiled_edges
                    if not any(ot in consumed_after for ot in e["output_tags"])
                ]
                target_sink = sink_edges[-1] if sink_edges else compiled_edges[-1]
                if "result" not in target_sink["output_tags"]:
                    target_sink["output_tags"].append("result")

        target_dag_dict = {
            "name": "Built_V6_DAG",
            "contents": [],
            "edges": compiled_edges,
            "alignment_logs": alignment_logs,
        }
        return json.dumps(target_dag_dict, ensure_ascii=False)

    builder_dag.add_edge(["input"], _edge1_parse_edges, ["parsed_edges"])
    builder_dag.add_edge(["input"], _edge2_analyze_alignment, ["tag_alignment"])
    # 第 3 条边不设单条直接输入（input_tags=[]），运行时自动触发 builder_dag.preset_edge 拼凑前两条边的输出！
    builder_dag.add_edge([], _edge3_compile_target_dag, ["result"])

    return builder_dag


@dataclass
class BuilderResult:
    """“用来构建 DAG 的 DAG”运行产出结果封装。"""

    compiled_dag: DAG
    dag_dict: Dict[str, Any]
    alignment_logs: List[str] = field(default_factory=list)
    builder_context: List[ContentItem] = field(default_factory=list)


class MetaDAGBuilder:
    """VEA V6 元图构建器：使用内部的 `builder_dag`（一个真实的 V6 DAG）来构建目标业务 `DAG`。"""

    def __init__(
        self,
        similarity_threshold: float = 0.45,
        ensure_result_tag: bool = True,
        edge_bindings: Optional[Dict[str, ProcessorType]] = None,
        llm: Optional[LLMProtocol] = None,
    ) -> None:
        self.similarity_threshold = similarity_threshold
        self.ensure_result_tag = ensure_result_tag
        self.edge_bindings = dict(edge_bindings or {})
        self.llm = llm
        self.meta_dag: DAG = create_dag_builder_dag(
            similarity_threshold=similarity_threshold,
            ensure_result_tag=ensure_result_tag,
        )

    def build(
        self,
        description: Union[str, Sequence[Any], Dict[str, Any]],
        initial_contents: Optional[Sequence[Union[ContentItem, Dict[str, Any]]]] = None,
    ) -> BuilderResult:
        """通过运行内部的 `meta_dag` 构建并返回一个新的 V6 `DAG` 实例。"""
        # 每次构建使用全新的元图实例，保证上下文干净
        self.meta_dag = create_dag_builder_dag(
            similarity_threshold=self.similarity_threshold,
            ensure_result_tag=self.ensure_result_tag,
        )
        content_str = (
            description
            if isinstance(description, str)
            else (
                "\n".join(str(x) for x in description)
                if isinstance(description, (list, tuple)) and all(isinstance(x, str) for x in description)
                else json.dumps(description, ensure_ascii=False)
            )
        )
        # 1. 向用来构建 DAG 的 meta_dag 送入一条初始信息 (gen=0)
        self.meta_dag.send({"tags": ["input"], "content": content_str, "gen": 0})

        # 2. 在外部检测是否达成 tag="result" 条件来终止
        max_ticks = 10
        ticks = 0
        while not self.meta_dag.has_result() and ticks < max_ticks:
            self.meta_dag.run()
            ticks += 1

        res_str = self.meta_dag.get_result()
        if not res_str:
            raise RuntimeError("用来构建 DAG 的元图未能生成 tag='result' 的构建结果")

        dag_dict = json.loads(res_str)
        compiled = DAG.from_dict(
            dag_dict,
            llm=self.llm,
            edge_bindings=self.edge_bindings,
        )
        if initial_contents:
            for item in initial_contents:
                compiled.send(item)

        return BuilderResult(
            compiled_dag=compiled,
            dag_dict={"contents": compiled.to_dict()["contents"], "edges": compiled.to_dict()["edges"]},
            alignment_logs=dag_dict.get("alignment_logs", []),
            builder_context=list(self.meta_dag.context),
        )

    # 兼容 V5 命名习惯
    merge = build
