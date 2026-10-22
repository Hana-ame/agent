"""v8 配置层：DAG 与 prompt 全部是纯 JSON。

一个配置文件就是一张图：

```json
{
  "name": "demo",
  "result_tag": "answer",
  "pi": {"provider": "laptop", "model": "deepseek-v4-flash"},
  "edges": [
    {
      "id": "answer",
      "layer": 0,
      "input_tags": ["question"],
      "output_tags": ["answer"],
      "prompt": "请回答：\\n{contents}"
    }
  ]
}
```

## prompt 占位符

| 占位符 | 替换成 |
|---|---|
| `{contents}` | 该边所有输入 tag 对应内容的拼接（`---` 分隔） |
| `{content}` | 同上但只取最新一条 |
| `{tag:名称}` | 指定 tag 的全部内容拼接 |
| `{tag1,tag2}` | 多 tag 简写 |

未定义的占位符原样保留（多半是模型自己需要的字面量，比如 JSON 示例里的 `{"ai": [...]}`），
不去猜、不报错 —— 只有「占位符键名恰好等于某个 tag 名」时才替换。
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

PLACEHOLDER = re.compile(r"\{([a-zA-Z0-9_:,\-]+)\}")


@dataclass
class PiConfig:
    provider: str = "laptop"
    model: str = "deepseek-v4-flash"
    tools: bool = False           # 默认关工具：edge 是纯文本变换，不该改磁盘
    timeout: float = 300.0
    retries: int = 3
    ssh_host: Optional[str] = None   # 形如 lumin@127.0.0.1；None = 本机直接调 pi
    ssh_key: Optional[str] = None


@dataclass
class EdgeConfig:
    id: str
    layer: int
    input_tags: List[str]
    output_tags: List[str]
    prompt: str
    model: Optional[str] = None      # 覆盖 DAG 级 provider/model，形如 "provider/id"
    timeout: Optional[float] = None
    tools: Optional[bool] = None


@dataclass
class DAGConfig:
    name: str
    edges: List[EdgeConfig]
    result_tag: str
    seed_tag: Optional[str] = None
    seed_content: Optional[str] = None
    db_path: Optional[str] = None
    pi: PiConfig = field(default_factory=PiConfig)
    path: Optional[str] = None
    raw: Dict[str, Any] = field(default_factory=dict)

    def sorted_edges(self) -> List[EdgeConfig]:
        return sorted(self.edges, key=lambda e: (e.layer, self.edges.index(e)))


def _edge_from_dict(d: Dict[str, Any], i: int) -> EdgeConfig:
    for key in ("input_tags", "output_tags", "prompt"):
        if key not in d:
            raise ValueError(f"第 {i + 1} 条 edge 缺字段 {key!r}：{d}")
    return EdgeConfig(
        id=str(d.get("id") or f"edge{i + 1}"),
        layer=int(d.get("layer", 0)),
        input_tags=[str(t) for t in d["input_tags"]],
        output_tags=[str(t) for t in d["output_tags"]],
        prompt=str(d["prompt"]),
        model=d.get("model"),
        timeout=d.get("timeout"),
        tools=d.get("tools"),
    )


def load(path: str | Path) -> DAGConfig:
    p = Path(path)
    data = json.loads(p.read_text(encoding="utf-8"))

    edges_raw = data.get("edges")
    if not isinstance(edges_raw, list) or not edges_raw:
        raise ValueError(f"{p} 里没有 edges 数组")
    edges = [_edge_from_dict(e, i) for i, e in enumerate(edges_raw)]

    pi_raw = data.get("pi") or {}
    pi = PiConfig(
        provider=pi_raw.get("provider", "laptop"),
        model=pi_raw.get("model", "deepseek-v4-flash"),
        tools=bool(pi_raw.get("tools", False)),
        timeout=float(pi_raw.get("timeout", 300.0)),
        retries=int(pi_raw.get("retries", 3)),
        ssh_host=pi_raw.get("ssh_host"),
        ssh_key=pi_raw.get("ssh_key"),
    )

    result_tag = data.get("result_tag") or edges[-1].output_tags[0]
    return DAGConfig(
        name=str(data.get("name") or p.stem),
        edges=edges,
        result_tag=str(result_tag),
        seed_tag=data.get("seed_tag"),
        seed_content=data.get("seed_content"),
        db_path=data.get("db_path"),
        pi=pi,
        path=str(p.resolve()),
        raw=data,
    )


def render_prompt(prompt: str, inputs: Dict[str, List[str]]) -> str:
    """把占位符换成实际内容。

    `inputs` 是 `{tag: [内容, ...]}`。只替换键里出现过的 tag，
    其余占位符原样保留（例如 prompt 里给模型看的 JSON 示例）。
    """
    contents_all = [c for tag in inputs for c in inputs[tag]]
    latest = contents_all[-1] if contents_all else ""

    def _sub(m: re.Match) -> str:
        key = m.group(1)
        if key == "contents":
            return "\n\n---\n\n".join(contents_all)
        if key == "content":
            return latest
        if key.startswith("tag:"):
            tag = key[4:]
            return "\n\n---\n\n".join(inputs.get(tag, [])) if tag in inputs else m.group(0)
        parts = [p.strip() for p in key.split(",") if p.strip()]
        if parts and all(p in inputs for p in parts):
            return "\n\n---\n\n".join(c for p in parts for c in inputs[p])
        return m.group(0)

    return PLACEHOLDER.sub(_sub, prompt)
