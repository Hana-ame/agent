"""v7 图 → v8 配置：把顺序执行的 v7 子图摊平成可波次并行的 v8 图。

v7 的边是 LLM/fn，引擎一次只跑一条；v8 的 pi 节点 + 波次调度可以并行无依赖的边。
转换规则：

- `input_tags` / `output_tags` / `layer` / `prompt` 原样保留
- v7 的 `processor.model` 若是 models.json 里的 key，映射成 `provider/model` 覆盖
- v7 的 `contents` 第一条作为 v8 的 seed（tag + content）
- `result_tag`：优先参数，其次 v7 图自己的，最后取最后一条边的输出 tag

v7 的 fn 边无法序列化，JSON 里本来也不会有；command 节点需要手工补。
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

__all__ = ["convert_v7", "convert_file"]


def _tags(edge: Dict[str, Any], key: str, alias: str) -> List[str]:
    raw = edge.get(key, edge.get(alias, []))
    if isinstance(raw, str):
        return [raw]
    return [str(x) for x in (raw or [])]


def convert_v7(data: Dict[str, Any], *, provider: str = "laptop",
               result_tag: Optional[str] = None,
               name: Optional[str] = None) -> Dict[str, Any]:
    edges_raw = data.get("edges")
    if not isinstance(edges_raw, list) or not edges_raw:
        raise ValueError("v7 图没有 edges 数组")

    models = data.get("models") or {}
    out_edges: List[Dict[str, Any]] = []
    for i, e in enumerate(edges_raw):
        proc = e.get("processor") or e.get("Processor") or {}
        if isinstance(proc, str):
            proc = {"prompt": proc}
        prompt = str(proc.get("prompt") or "{contents}")
        model_key = str(proc.get("model") or "default")
        meta = models.get(model_key)
        model_name = meta.get("model") if isinstance(meta, dict) else None

        edge = {
            "id": str(e.get("id") or f"edge{i}"),
            "layer": int(e.get("layer", 0)),
            "input_tags": _tags(e, "input_tags", "inputs"),
            "output_tags": _tags(e, "output_tags", "outputs"),
            "prompt": prompt,
        }
        if model_name and model_name != "default":
            edge["model"] = f"{provider}/{model_name}"
        out_edges.append(edge)

    contents = data.get("contents") or []
    seed_tag = data.get("seed_tag")
    seed_content = data.get("seed_content") or ""
    if contents and isinstance(contents[0], dict):
        c0 = contents[0]
        seed_tag = seed_tag or ((c0.get("tags") or [None])[0])
        seed_content = seed_content or str(c0.get("contents", ""))

    if not seed_tag:
        ins = out_edges[0]["input_tags"]
        seed_tag = ins[0] if ins else "input"
    result = (result_tag or data.get("result_tag")
              or (out_edges[-1]["output_tags"][0] if out_edges[-1]["output_tags"] else "result"))

    return {
        "name": name or f"{data.get('name', 'v7')}_v8",
        "_comment": "由 dag8 convert 从 v7 图转换：边按原 layer 保留，v8 波次调度可并行无依赖的边。",
        "seed_tag": str(seed_tag),
        "seed_content": str(seed_content),
        "result_tag": str(result),
        "pi": {
            "provider": provider,
            "model": "deepseek-v4-flash",
            "tools": False,
            "timeout": 240,
            "retries": 3,
        },
        "edges": out_edges,
    }


def convert_v7_to_graphs(
    data: Dict[str, Any],
    *,
    subdir: str = "graphs",
    provider: str = "laptop",
    result_tag: Optional[str] = None,
    name: Optional[str] = None,
) -> tuple[Dict[str, Any], Dict[str, Dict[str, Any]]]:
    """把 v7 的每条边变成一张单边 v7 子图，v8 波次并行调度它们。

    这样 v8 拿到的是 v7 直连客户端（快），而不是 pi 子进程（慢），
    同时保留 v8 的 tag 路由/并行/缓存/契约。
    """
    edges_raw = data.get("edges")
    if not isinstance(edges_raw, list) or not edges_raw:
        raise ValueError("v7 图没有 edges 数组")

    models = data.get("models") or {}
    subgraphs: Dict[str, Dict[str, Any]] = {}
    parent_edges: List[Dict[str, Any]] = []

    for i, e in enumerate(edges_raw):
        eid = str(e.get("id") or f"edge{i}")
        proc = e.get("processor") or e.get("Processor") or {}
        if isinstance(proc, str):
            proc = {"prompt": proc}
        ins = _tags(e, "input_tags", "inputs")
        outs = _tags(e, "output_tags", "outputs")
        if not ins or not outs:
            raise ValueError(f"v7 edge {eid!r} 缺 input/output tags")

        sub_edge = {
            "id": eid,
            "layer": 0,
            "input_tags": [ins[0]],
            "output_tags": outs,
            "processor": proc,
        }
        subgraphs[f"{eid}.json"] = {
            "name": f"{data.get('name', 'v7')}_{eid}",
            "models": models,
            "seed_tag": ins[0],
            "contents": [],
            "result_tag": outs[-1],
            "edges": [sub_edge],
        }
        parent_edges.append({
            "id": eid,
            "kind": "graph",
            "layer": int(e.get("layer", 0)),
            "input_tags": ins,
            "output_tags": outs,
            "graph": {"path": f"{subdir}/{eid}.json", "timeout": 300,
                      "seed_tag": ins[0], "result_tag": outs[-1]},
        })

    contents = data.get("contents") or []
    seed_tag = data.get("seed_tag")
    seed_content = data.get("seed_content") or ""
    if contents and isinstance(contents[0], dict):
        c0 = contents[0]
        seed_tag = seed_tag or ((c0.get("tags") or [None])[0])
        seed_content = seed_content or str(c0.get("contents", ""))
    if not seed_tag:
        seed_tag = parent_edges[0]["input_tags"][0]
    result = (result_tag or data.get("result_tag")
              or (parent_edges[-1]["output_tags"][0] if parent_edges[-1]["output_tags"] else "result"))

    parent = {
        "name": name or f"{data.get('name', 'v7')}_v8_graphs",
        "_comment": "由 dag8 convert --mode graph 生成：每条边是一张单边 v7 子图，v8 负责并行。",
        "seed_tag": str(seed_tag),
        "seed_content": str(seed_content),
        "result_tag": str(result),
        "pi": {"provider": provider, "model": "deepseek-v4-flash",
               "tools": False, "timeout": 240, "retries": 3},
        "edges": parent_edges,
    }
    return parent, subgraphs


def convert_file(source: str | Path, out: str | Path, **kwargs: Any) -> Path:
    data = json.loads(Path(source).read_text(encoding="utf-8"))
    cfg = convert_v7(data, **kwargs)
    p = Path(out)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(cfg, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return p
