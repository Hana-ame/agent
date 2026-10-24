"""从自然语言描述构建 v7 DAG（自 v6 的 `MetaDAGBuilder` 解析层移植，2026-10-22）。

v7 原有 `dag.meta` 是「用 DAG 改一个已存在的 DAG」，缺的是**从零生成**：
给一段自然语言或句子列表，解析成边定义，组装成可运行的 DAG。

## 支持的写法

```
从 [raw_report] 经过 提取关键指标 到 [metrics]
[metrics] -> 计算同比 -> [yoy]
无输入 经过 生成初始问题 到 [question]
创建步骤1：从 [a,b] 经过 汇总 到 [c]
```

也接受 JSON（`{"edges": [...]}` 或 `[{...}]`）与现成的 dict 列表。

## 与 v6 的差别

- 边定义用 v7 的 `EdgeSpec`（`Processor(model, prompt)`），不是 v6 的 `Edge`
- 不依赖 v6 的预设边 / DAG 实现，纯解析 + 组装，可离线单测
"""
from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional, Sequence, Union

from v7.engine import DAG
from v7.schema import EdgeSpec

__all__ = [
    "parse_edges_from_description",
    "parse_json_stream",
    "build_dag_from_description",
    "EDGE_SENTENCE_PATTERNS",
]


def parse_json_stream(text: str) -> List[Any]:
    """从可能含多段 JSON / 前后夹杂自然语言的文本里，顺序解出所有 JSON 值。

    预设边合并两条信息后，正文常是「JSON\\nJSON」或「提示词\\n\\n输入数据：\\nJSON」。
    直接 `json.loads` 会因 Extra data 失败，这里用 `raw_decode` 逐段取。
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


EDGE_SENTENCE_PATTERNS = [
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
    re.compile(r"^(?P<src>[^\-=\>]*?)\s*[\-=]+\[\s*(?P<act>.+?)\s*\][\-=]*\>\s*(?P<dst>.+?)$"),
    re.compile(r"^(?P<src>[^\-]+?)\s*->\s*(?P<act>[^\-]+?)\s*->\s*(?P<dst>[^\-]+?)$"),
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


def _spec_from_any(x: Any) -> Dict[str, Any]:
    return EdgeSpec.from_any(x).to_dict()


def parse_edges_from_description(
    description: Union[str, Sequence[Any], Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """把自然语言 / 句子列表 / JSON 解析成 v7 边定义列表（`EdgeSpec.to_dict()` 形态）。"""
    if isinstance(description, dict):
        if "edges" in description:
            return [_spec_from_any(e) for e in description["edges"]]
        return [_spec_from_any(description)]
    if isinstance(description, (list, tuple)) and description and not isinstance(description[0], str):
        return [_spec_from_any(e) for e in description]
    if isinstance(description, str):
        stripped = description.strip()
        if stripped.startswith(("{", "[")) and stripped.endswith(("}", "]")):
            try:
                parsed = json.loads(stripped)
                if isinstance(parsed, dict) and "edges" in parsed:
                    return [_spec_from_any(e) for e in parsed["edges"]]
                if isinstance(parsed, list):
                    return [_spec_from_any(e) for e in parsed]
            except Exception:
                pass
        # 同一行里连着写多条：「…到 [x] 从 [y] 经过…」→ 在第二个「从」前断行
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
        for pat in EDGE_SENTENCE_PATTERNS:
            m = pat.match(clean)
            if m:
                gd = m.groupdict()
                out.append(
                    EdgeSpec(
                        input_tags=_parse_tag_list_expr(gd.get("src") or ""),
                        processor=_parse_processor_expr(gd.get("act") or ""),
                        output_tags=_parse_tag_list_expr(gd.get("dst") or ""),
                    ).to_dict()
                )
                break
        else:
            raise ValueError(
                f"无法从描述中解析出处理边: {clean!r}。"
                "请用「从 [tag1] 经过 处理方式 到 [result]」或「[tag1] -> 处理方式 -> [result]」。"
            )
    return out


def build_dag_from_description(
    description: Union[str, Sequence[Any], Dict[str, Any]],
    *,
    name: str = "built_from_description",
    ensure_final_tag: Optional[str] = "result",
    llm_client: Any = None,
) -> DAG:
    """解析描述并组装成可直接 `add_content()` + `run(until_tag=...)` 的 v7 DAG。

    `ensure_final_tag`：若没有任何边产出该 tag，就追加到最末一条边的 output_tags 上，
    保证调用方总有终止条件可用（传 None 关掉）。
    """
    edges = parse_edges_from_description(description)
    if not edges:
        raise ValueError("描述里没有解析出任何边")

    dag = DAG(name=name, llm_client=llm_client) if llm_client is not None else DAG(name=name)
    for edge in edges:
        dag.add_edge(
            input_tags=edge["input_tags"],
            processor=edge["processor"],
            output_tags=edge["output_tags"],
            layer=int(edge.get("layer", 0)),
        )

    if ensure_final_tag:
        produced = {t for e in dag.edges for t in e.output_tags}
        if ensure_final_tag not in produced:
            last = max(dag.edges, key=lambda e: (e.layer, dag.edges.index(e)))
            last.output_tags = list(last.output_tags) + [ensure_final_tag]
    return dag
