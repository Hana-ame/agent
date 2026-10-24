"""解释一次运行为什么失败/卡住，并给出可执行建议。

    python -m dag8 explain RUN_ID
    python -m dag8 --db path explain RUN_ID --json

不调网关，只读 sqlite。
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from dag8.store import Store

__all__ = ["explain", "format_report"]


def _hint(error: str) -> str:
    low = error.lower()
    if "429" in error or "rate_limit" in low or "too many requests" in low or "限流" in error:
        return "限流：加 --rpm；配置 pi.fallbacks；graph 子图会放大调用数，先降并发"
    if "超时" in error or "timeout" in low or "timed out" in low:
        return "超时：检查 --max-seconds / 节点 timeout；可 --resume 只补未完成的边"
    if "输出契约" in error or "schema" in low or "不是合法 json" in low:
        return "格式不配：看 expect/schema/repair；先 dag8 lint 再收紧或放宽契约"
    if "缺输入" in error or "不存在" in error or "not found" in low:
        return "输入/路径问题：先 dag8 lint；检查上游 output_tags 或 optional_inputs"
    return ""


def explain(store: Store, run_id: str) -> Dict[str, Any]:
    info = dict(store.run_info(run_id))
    edges = store.edge_latest(run_id)

    rows: List[Dict[str, Any]] = []
    hints: List[str] = []
    for eid in sorted(edges, key=lambda x: (int(edges[x].get("layer") or 0), x)):
        e = edges[eid]
        rows.append({
            "edge": eid,
            "layer": e.get("layer"),
            "status": e.get("status"),
            "elapsed": e.get("elapsed"),
            "model": e.get("model"),
            "error": e.get("error"),
        })
        err = e.get("error") or ""
        if e.get("status") == "error" and err:
            h = _hint(err)
            if h:
                hints.append(f"{eid}: {h}")
        elif e.get("status") == "skipped" and err:
            hints.append(f"{eid}: 被跳过（{err}）；补上游或把该 tag 标进 optional_inputs")

    result = store.result(run_id, info.get("result_tag") or "")
    if info.get("status") == "over_budget":
        hints.append("超出成本预算：调高 --max-cost；已完成边保留，可 --resume-last 续跑")
    if info.get("status") in ("error", "incomplete", "timeout", "partial",
                              "interrupted", "over_budget") and rows:
        if info.get("status") == "partial":
            hints.append("部分失败：可 --resume-last 补未成功的边；或用 dag8 diff 对比其它运行")
        if info.get("status") == "interrupted":
            hints.append("运行被中断：可直接 --resume-last 续跑")
        hints.append(
            f"可续跑：python -m dag8 --db <db> run <config> --resume {run_id}"
            "（只补未完成的边）"
        )
    if info.get("status") == "ok":
        hints.append("运行成功；可用 dag8 diff 与其它运行对比，或 dag8 cost 看花费")

    deduped: List[str] = []
    for h in hints:
        if h not in deduped:
            deduped.append(h)

    return {
        "run": info,
        "result_chars": len(result or "") if result is not None else None,
        "edges": rows,
        "hints": deduped,
    }


def format_report(data: Dict[str, Any]) -> str:
    info = data["run"]
    lines = [
        f"run {info['run_id']} · {info.get('dag_name')} · {info.get('status')}",
        f"seed: {info.get('seed_tag')} {str(info.get('seed_content') or '')[:60]!r}",
        f"result: {info.get('result_tag')} "
        f"({data['result_chars'] if data['result_chars'] is not None else '缺失'})",
        "",
    ]
    for e in data["edges"]:
        mark = {"ok": "✓", "error": "✗", "skipped": "·"}.get(e["status"], "?")
        elapsed = f"{e['elapsed']:.1f}s" if e.get("elapsed") is not None else "-"
        lines.append(f"{mark} {e['edge'][:24]:24} {e['status']:8} {elapsed:>7}  {e.get('error') or ''}")
    if data["hints"]:
        lines.append("")
        lines.append("建议：")
        for h in data["hints"]:
            lines.append(f"  - {h}")
    return "\n".join(lines)
