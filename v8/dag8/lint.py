"""v8 配置静态检查：不跑图，先找出「跑起来才会发现」的问题。

检查项：

- 重复 edge id
- `result_tag` 永远没有生产者
- 输入 tag 既不是 seed、也没有任何边生产（必然 skipped；optional 除外）
- 同一个输出 tag 有多个生产者（允许，但可能带来意外拼接）
- graph 节点引用的子图文件不存在
- producer 的 layer 不低于 consumer（v8 按波次跑，不是错误，只是建议）
- 边之间的 tag 依赖成环（会互相等待）

错误会让 `lint` 返回 1；警告默认不失败，`--strict` 才失败。
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Set

from dag8.config import DAGConfig

__all__ = ["Issue", "lint_config"]


@dataclass
class Issue:
    level: str          # error | warning
    message: str
    edge_id: str = ""


def _find_cycles(edges) -> List[str]:
    """边依赖图上的环；返回环里的边 id（空 = 无环）。"""
    producers: Dict[str, List[str]] = defaultdict(list)
    for e in edges:
        for t in e.output_tags:
            producers[t].append(e.id)

    deps: Dict[str, Set[str]] = {e.id: set() for e in edges}
    for e in edges:
        for t in e.input_tags:
            for pid in producers.get(t, []):
                deps[e.id].add(pid)

    color: Dict[str, int] = {}   # 0=未访问 1=栈中 2=完成
    stack: List[str] = []
    cycles: List[str] = []

    def dfs(node: str) -> None:
        color[node] = 1
        stack.append(node)
        for nxt in deps.get(node, ()):
            if color.get(nxt, 0) == 1:
                i = stack.index(nxt)
                cycles.extend(stack[i:])
            elif color.get(nxt, 0) == 0:
                dfs(nxt)
        stack.pop()
        color[node] = 2

    for eid in deps:
        if color.get(eid, 0) == 0:
            dfs(eid)
    return sorted(set(cycles))


def lint_config(cfg: DAGConfig) -> List[Issue]:
    issues: List[Issue] = []

    # 1) 重复 id
    seen: Set[str] = set()
    for e in cfg.edges:
        if e.id in seen:
            issues.append(Issue("error", f"重复 edge id：{e.id}", e.id))
        seen.add(e.id)

    producers: Dict[str, List[str]] = defaultdict(list)
    for e in cfg.edges:
        for t in e.output_tags:
            producers[t].append(e.id)

    # 2) result_tag 无生产者
    if cfg.result_tag not in producers and cfg.result_tag != cfg.seed_tag:
        issues.append(Issue("error", f"result_tag={cfg.result_tag!r} 没有任何边生产"))

    # 3) 不可达输入（extra_seed_tags 由调用方在 run 时注入）
    seeds = {cfg.seed_tag, *cfg.extra_seed_tags}
    for e in cfg.edges:
        for t in e.input_tags:
            if t in seeds or t in producers or t in e.optional_inputs:
                continue
            issues.append(Issue(
                "error",
                f"输入 tag {t!r} 既不是 seed 也没有生产者，这条边会一直 skipped",
                e.id,
            ))

    # 4) 多生产者
    for t, owners in producers.items():
        if len(owners) > 1:
            issues.append(Issue(
                "warning",
                f"输出 tag {t!r} 有多个生产者 {owners}，下游会拿到全部内容",
            ))

    # 5) graph 子图文件
    for e in cfg.edges:
        if e.kind != "graph" or e.graph is None:
            continue
        base = Path(cfg.path).resolve().parent if cfg.path else Path.cwd()
        p = Path(e.graph.path)
        p = p if p.is_absolute() else base / p
        if not p.is_file():
            issues.append(Issue("error", f"graph 子图不存在：{p}", e.id))

    # 6) layer 建议
    id_to_edge = {e.id: e for e in cfg.edges}
    for e in cfg.edges:
        for t in e.input_tags:
            for pid in producers.get(t, []):
                if pid == e.id:
                    continue
                pe = id_to_edge[pid]
                if pe.layer >= e.layer:
                    issues.append(Issue(
                        "warning",
                        f"{e.id}(layer {e.layer}) 的输入来自 {pid}(layer {pe.layer})，"
                        "建议生产者在更低的 layer",
                        e.id,
                    ))

    # 7) 环
    cyc = _find_cycles(cfg.edges)
    if cyc:
        issues.append(Issue("error", f"边依赖成环：{cyc}"))

    # 去重（同一问题可能被多个输入重复触发）
    out: List[Issue] = []
    seen_msg: Set[str] = set()
    for it in issues:
        key = f"{it.level}|{it.edge_id}|{it.message}"
        if key not in seen_msg:
            seen_msg.add(key)
            out.append(it)
    return out
