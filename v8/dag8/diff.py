"""两次运行的对比：边状态、时延、模型、结果长度。

用于 A/B（换模型、v7 vs v8、调并发前后）：

    python -m dag8 diff RUN_A RUN_B
    python -m dag8 --db path diff RUN_A RUN_B --json
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from dag8.store import Store

__all__ = ["diff_runs", "format_report"]


def _result_chars(store: Store, info: Dict[str, Any]) -> Optional[int]:
    cid = info.get("result_content_id")
    if not cid:
        return None
    row = store.conn.execute(
        "SELECT LENGTH(content) FROM contents WHERE id=?", (cid,)
    ).fetchone()
    return int(row[0]) if row else None


def _duration(info: Dict[str, Any]) -> Optional[float]:
    if info.get("finished_at") and info.get("started_at"):
        return round(info["finished_at"] - info["started_at"], 2)
    return None


def diff_runs(store: Store, run_a: str, run_b: str) -> Dict[str, Any]:
    info_a = store.run_info(run_a)
    info_b = store.run_info(run_b)
    ea = store.edge_latest(run_a)
    eb = store.edge_latest(run_b)

    def layer_of(eid: str) -> int:
        for src in (ea, eb):
            if eid in src:
                return int(src[eid].get("layer") or 0)
        return 0

    rows: List[Dict[str, Any]] = []
    for eid in sorted(set(ea) | set(eb), key=lambda x: (layer_of(x), x)):
        a, b = ea.get(eid), eb.get(eid)
        ta = a["elapsed"] if a and a["elapsed"] is not None else None
        tb = b["elapsed"] if b and b["elapsed"] is not None else None
        rows.append({
            "edge": eid,
            "status_a": a["status"] if a else "-",
            "status_b": b["status"] if b else "-",
            "elapsed_a": round(ta, 2) if ta is not None else None,
            "elapsed_b": round(tb, 2) if tb is not None else None,
            "delta": round(tb - ta, 2) if ta is not None and tb is not None else None,
            "model_a": a["model"] if a else None,
            "model_b": b["model"] if b else None,
        })

    def side(info: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "run_id": info["run_id"],
            "status": info["status"],
            "duration": _duration(info),
            "result_chars": _result_chars(store, info),
        }

    return {"a": side(info_a), "b": side(info_b), "edges": rows}


def format_report(data: Dict[str, Any]) -> str:
    a, b = data["a"], data["b"]
    lines = [
        f"A {a['run_id']}  {a['status']}  {a['duration'] if a['duration'] is not None else '-'}s"
        f"  result {a['result_chars'] if a['result_chars'] is not None else '-'} chars",
        f"B {b['run_id']}  {b['status']}  {b['duration'] if b['duration'] is not None else '-'}s"
        f"  result {b['result_chars'] if b['result_chars'] is not None else '-'} chars",
        "",
        f"{'edge':20} {'A':10} {'A(s)':>7} {'B':10} {'B(s)':>7} {'Δ(s)':>7}",
        "-" * 66,
    ]
    for r in data["edges"]:
        ea = "-" if r["elapsed_a"] is None else f"{r['elapsed_a']:.1f}"
        eb = "-" if r["elapsed_b"] is None else f"{r['elapsed_b']:.1f}"
        dd = "-" if r["delta"] is None else f"{r['delta']:+.1f}"
        lines.append(f"{r['edge'][:20]:20} {r['status_a']:10} {ea:>7} "
                     f"{r['status_b']:10} {eb:>7} {dd:>7}")
    return "\n".join(lines)
