"""v8 配置层：DAG 与 prompt 全部是纯 JSON。

一个配置文件就是一张图。节点默认是 pi agent 调用（`kind: "pi"`），
也可以把一个 v7 图当节点（`kind: "graph"`，实验特性），或把一段确定性
shell 命令当节点（`kind: "command"`，不调模型）：

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
    },
    {
      "id": "deep_dive",
      "kind": "graph",
      "layer": 1,
      "input_tags": ["answer"],
      "output_tags": ["deep"],
      "graph": {
        "path": "graphs/v7_deep_dive.json",
        "seed_tag": "question",
        "result_tag": "final"
      }
    }
  ]
}
```

`graph` 节点的字段：

| 字段 | 必填 | 说明 |
|---|---|---|
| `path` | 是 | v7 图 JSON，相对本配置文件所在目录 |
| `seed_tag` | 是 | 节点输入注入到 v7 图的哪个 tag |
| `result_tag` | 是 | 从 v7 图的哪个 tag 取输出 |
| `models` | 否 | v7 用的 models.json；不填则用图 JSON 内联的 `models` |
| `max_steps` | 否 | v7 单次运行的最大步数，默认 20 |

`command` 节点的字段：

| 字段 | 必填 | 说明 |
|---|---|---|
| `command` | 是 | 一段 shell 命令；可用下方 prompt 占位符 |
| `stdin` | 否 | `true` 时把 `{contents}` 渲染后的文本喂给命令的 stdin |
| `timeout` | 否 | 秒；默认 600，超时即节点失败 |

命令的工作目录是配置文件所在目录；非零退出码即节点失败，stderr 进 `edges.error`。
dry-run 不执行命令，只渲染命令文本。

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

from dag8.contracts import normalize_spec

PLACEHOLDER = re.compile(r"\{([a-zA-Z0-9_:,\-]+)\}")

NODE_KINDS = ("pi", "graph", "command", "http", "db", "approval")


@dataclass
class PiConfig:
    provider: str = "laptop"
    model: str = "deepseek-v4-flash"
    tools: bool = False           # 默认关工具：edge 是纯文本变换，不该改磁盘
    timeout: float = 300.0
    retries: int = 3
    thinking: Optional[str] = None   # 传给 pi --thinking（off/minimal/low/...）；None = 不传
    ssh_host: Optional[str] = None   # 形如 lumin@127.0.0.1；None = 本机直接调 pi
    ssh_key: Optional[str] = None
    fallbacks: List[str] = field(default_factory=list)  # 备用模型，如 ["glm-5.2", "sensenova/..."]


@dataclass
class GraphConfig:
    """把一个 v7 图当节点跑所需的引用信息。"""

    path: str
    seed_tag: str
    result_tag: str
    models: Optional[str] = None
    max_steps: int = 20
    timeout: float = 300.0           # 整张子图的总预算（秒）
    engine: str = "v7"               # v7 = 顺序子图；v8 = 递归 v8 图（可波次并行）
    workers: int = 1                 # engine=v8 时子图内部的并发


@dataclass
class EdgeConfig:
    id: str
    layer: int
    input_tags: List[str]
    output_tags: List[str]
    prompt: str
    kind: str = "pi"                 # pi | graph | command
    graph: Optional[GraphConfig] = None
    command: Optional[str] = None    # 仅 command 节点：一段 shell 命令；stdin=True 时输入走 stdin
    stdin: bool = False              # 仅 command 节点：把输入文本喂给命令的 stdin
    model: Optional[str] = None      # 覆盖 DAG 级 provider/model，形如 "provider/id"（仅 pi 节点）
    timeout: Optional[float] = None
    tools: Optional[bool] = None
    expect: Dict[str, Any] = field(default_factory=dict)  # 输出契约（见 dag8/contracts.py）
    repair: int = 0                  # 契约失败后把原因回灌给 pi 节点重跑的次数
    optional_inputs: List[str] = field(default_factory=list)  # 缺失也不阻塞的输入 tag
    cache: Optional[bool] = None     # False = 该边禁用缓存（易变节点，如扫描当前文件）
    when: Optional[Dict[str, Any]] = None  # 条件边：输入就绪后按 tag 内容决定是否执行
    url: Optional[str] = None         # 仅 http 节点
    method: str = "GET"               # 仅 http 节点
    headers: Dict[str, str] = field(default_factory=dict)
    body: Optional[str] = None        # 仅 http 节点：请求体（支持占位符）
    max_chars: int = 20000            # 仅 http 节点：响应截断
    judge: Optional[Dict[str, Any]] = None  # 质量闸门：输出后调一次评审，低于 min_score 判失败
    db_path: Optional[str] = None      # 仅 db 节点：sqlite 路径（相对配置文件）
    query: Optional[str] = None        # 仅 db 节点：SQL
    params: List[str] = field(default_factory=list)  # SQL 绑定参数（可含占位符）
    max_rows: int = 1000
    read_only: bool = True
    marker: Optional[str] = None      # 仅 approval 节点：批准标记文件（相对配置文件）


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


def _parse_graph(raw: Any, i: int) -> GraphConfig:
    if not isinstance(raw, dict):
        raise ValueError(f"第 {i + 1} 条 edge 的 graph 必须是对象：{raw!r}")
    engine = str(raw.get("engine", "v7"))
    required = ("path",) if engine == "v8" else ("path", "seed_tag", "result_tag")
    for key in required:
        if not raw.get(key):
            raise ValueError(f"第 {i + 1} 条 edge 的 graph 缺字段 {key!r}：{raw}")
    return GraphConfig(
        path=str(raw["path"]),
        seed_tag=str(raw.get("seed_tag") or ""),
        result_tag=str(raw.get("result_tag") or ""),
        models=str(raw["models"]) if raw.get("models") else None,
        max_steps=int(raw.get("max_steps", 20)),
        timeout=float(raw.get("timeout", 300.0)),
        engine=engine,
        workers=int(raw.get("workers", 1)),
    )


def _edge_from_dict(d: Dict[str, Any], i: int) -> EdgeConfig:
    for key in ("input_tags", "output_tags"):
        if key not in d:
            raise ValueError(f"第 {i + 1} 条 edge 缺字段 {key!r}：{d}")

    # kind 缺省时：写了 graph 就是 graph 节点，写了 command 就是 command 节点，否则是 pi 节点
    kind = str(
        d.get("kind")
        or ("graph" if "graph" in d else ("command" if "command" in d
            else ("http" if "url" in d else ("db" if "db_path" in d
                  else ("approval" if "marker" in d else "pi")))))
    ).lower()
    if kind not in NODE_KINDS:
        raise ValueError(
            f"第 {i + 1} 条 edge 的 kind={kind!r} 不支持（只允许 {' / '.join(NODE_KINDS)}）"
        )

    prompt = str(d.get("prompt", ""))
    graph: Optional[GraphConfig] = None
    command: Optional[str] = None
    if kind == "pi" and "prompt" not in d:
        raise ValueError(f"第 {i + 1} 条 edge 缺字段 'prompt'：{d}")
    if kind == "command":
        command = str(d.get("command") or "").strip()
        if not command:
            raise ValueError(f"第 {i + 1} 条 edge 缺字段 'command'：{d}")
    if kind == "graph":
        graph = _parse_graph(d.get("graph"), i)

    try:
        expect = normalize_spec(d.get("expect"))
    except ValueError as exc:
        raise ValueError(f"第 {i + 1} 条 edge 的 expect 非法：{exc}") from exc
    repair = int(d.get("repair", 0))
    if repair < 0:
        raise ValueError(f"第 {i + 1} 条 edge 的 repair 必须是非负整数：{repair}")

    input_tags = [str(t) for t in d["input_tags"]]
    optional = [str(t) for t in (d.get("optional_inputs") or [])]
    unknown_opt = [t for t in optional if t not in input_tags]
    if unknown_opt:
        raise ValueError(
            f"第 {i + 1} 条 edge 的 optional_inputs 不在 input_tags 里：{unknown_opt}"
        )
    url = d.get("url")
    method = str(d.get("method") or "GET").upper()
    headers = d.get("headers") or {}
    body = d.get("body")
    max_chars = int(d.get("max_chars", 20000))
    if kind == "http":
        if not url:
            raise ValueError(f"第 {i + 1} 条 http edge 缺 url")
        if method not in ("GET", "POST", "PUT", "PATCH", "DELETE", "HEAD"):
            raise ValueError(f"第 {i + 1} 条 http edge 的 method 非法：{method}")
        if not isinstance(headers, dict) or not all(
                isinstance(k, str) and isinstance(v, (str, int, float))
                for k, v in headers.items()):
            raise ValueError(f"第 {i + 1} 条 http edge 的 headers 必须是字符串映射")
        headers = {str(k): str(v) for k, v in headers.items()}
        if body is not None and not isinstance(body, str):
            raise ValueError(f"第 {i + 1} 条 http edge 的 body 必须是字符串")
        if max_chars <= 0:
            raise ValueError(f"第 {i + 1} 条 http edge 的 max_chars 必须为正")
    db_path = d.get("db_path")
    query = d.get("query")
    params = d.get("params") or []
    max_rows = int(d.get("max_rows", 1000))
    read_only = bool(d.get("read_only", True))
    if kind == "db":
        if not db_path:
            raise ValueError(f"第 {i + 1} 条 db edge 缺 db_path")
        if not query:
            raise ValueError(f"第 {i + 1} 条 db edge 缺 query")
        if not isinstance(params, list) or not all(isinstance(x, str) for x in params):
            raise ValueError(f"第 {i + 1} 条 db edge 的 params 必须是字符串数组")
        if max_rows <= 0:
            raise ValueError(f"第 {i + 1} 条 db edge 的 max_rows 必须为正")
    marker = d.get("marker")
    if kind == "approval" and not marker:
        raise ValueError(f"第 {i + 1} 条 approval edge 缺 marker")
    judge = d.get("judge")
    if judge is not None:
        if not isinstance(judge, dict):
            raise ValueError(f"第 {i + 1} 条 edge 的 judge 必须是对象")
        unknown_judge = set(judge) - {"min_score", "prompt", "model"}
        if unknown_judge:
            raise ValueError(f"第 {i + 1} 条 edge 的 judge 有未知字段：{sorted(unknown_judge)}")
        min_score = float(judge.get("min_score", 7))
        if not 0 <= min_score <= 10:
            raise ValueError(f"第 {i + 1} 条 edge 的 judge.min_score 必须在 0~10")
        judge = {**judge, "min_score": min_score}
    when = d.get("when")
    if when is not None:
        if not isinstance(when, dict):
            raise ValueError(f"第 {i + 1} 条 edge 的 when 必须是对象")
        unknown_when = set(when) - {"tag", "present", "absent", "pattern", "equals"}
        if unknown_when:
            raise ValueError(f"第 {i + 1} 条 edge 的 when 有未知字段：{sorted(unknown_when)}")
        tag = str(when.get("tag") or "")
        if not tag:
            raise ValueError(f"第 {i + 1} 条 edge 的 when 缺 tag")
        if tag not in input_tags:
            raise ValueError(
                f"第 {i + 1} 条 edge 的 when.tag={tag!r} 不在 input_tags 里"
            )
        if "pattern" in when:
            try:
                re.compile(str(when["pattern"]))
            except re.error as exc:
                raise ValueError(f"第 {i + 1} 条 edge 的 when.pattern 非法：{exc}") from exc
        when = dict(when)

    return EdgeConfig(
        id=str(d.get("id") or f"edge{i + 1}"),
        layer=int(d.get("layer", 0)),
        input_tags=input_tags,
        output_tags=[str(t) for t in d["output_tags"]],
        prompt=prompt,
        kind=kind,
        graph=graph,
        command=command,
        stdin=bool(d.get("stdin", False)),
        model=d.get("model"),
        timeout=d.get("timeout"),
        tools=d.get("tools"),
        expect=expect,
        repair=repair,
        optional_inputs=optional,
        cache=(None if "cache" not in d else bool(d["cache"])),
        when=when,
        url=str(url) if url is not None else None,
        method=method,
        headers=headers,
        body=body,
        max_chars=max_chars,
        judge=judge,
        db_path=str(db_path) if db_path is not None else None,
        query=str(query) if query is not None else None,
        params=[str(x) for x in params],
        max_rows=max_rows,
        read_only=read_only,
        marker=str(marker) if marker is not None else None,
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
        thinking=pi_raw.get("thinking"),
        ssh_host=pi_raw.get("ssh_host"),
        ssh_key=pi_raw.get("ssh_key"),
        fallbacks=[str(x) for x in (pi_raw.get("fallbacks") or [])],
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
