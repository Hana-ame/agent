"""历史运行统计：按模型看成功率、时延、缓存命中与错误 Top。

数据来自 sqlite，不调网关：

    python -m dag8 stats
    python -m dag8 --db path stats --json

用途：决定「哪个模型值得用」「错误是不是 429」「缓存有没有真的省」。
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from dag8.store import Store

__all__ = ["collect", "format_report"]


def _pct(values: List[float], q: float) -> float:
    if not values:
        return 0.0
    xs = sorted(values)
    idx = min(len(xs) - 1, max(0, int(round((len(xs) - 1) * q))))
    return xs[idx]


def collect(store: Store, limit: int = 50) -> Dict[str, Any]:
    runs = store.conn.execute(
        "SELECT run_id, dag_name, status, started_at, finished_at, error "
        "FROM runs ORDER BY started_at DESC LIMIT ?", (limit,)
    ).fetchall()

    durations = [r["finished_at"] - r["started_at"] for r in runs
                 if r["finished_at"] and r["started_at"]]
    status: Dict[str, int] = {}
    for r in runs:
        status[r["status"]] = status.get(r["status"], 0) + 1

    ids = [r["run_id"] for r in runs]
    edges: List[Any] = []
    if ids:
        q = ",".join("?" * len(ids))
        edges = store.conn.execute(
            f"SELECT model, status, elapsed, error, cached FROM edges "
            f"WHERE run_id IN ({q})", ids
        ).fetchall()

    by_model: Dict[str, Dict[str, Any]] = {}
    errors: Dict[str, int] = {}
    for e in edges:
        model = e["model"] or "(none)"
        m = by_model.setdefault(model, {"n": 0, "ok": 0, "cached": 0, "lat": []})
        m["n"] += 1
        if e["status"] == "ok":
            m["ok"] += 1
            if e["cached"]:
                m["cached"] += 1
            elif e["elapsed"] is not None and e["elapsed"] > 0.05:
                m["lat"].append(e["elapsed"])
        elif e["status"] == "error" and e["error"]:
            key = str(e["error"]).replace("\n", " ")[:80]
            errors[key] = errors.get(key, 0) + 1

    models = []
    for name, m in by_model.items():
        models.append({
            "model": name,
            "n": m["n"],
            "ok_rate": round(m["ok"] / m["n"], 3) if m["n"] else 0.0,
            "p50": round(_pct(m["lat"], 0.5), 2),
            "p90": round(_pct(m["lat"], 0.9), 2),
            "cached": m["cached"],
            "n_uncached": len(m["lat"]),
        })
    models.sort(key=lambda x: (-x["n"], x["model"]))

    return {
        "runs": {
            "n": len(runs),
            "by_status": status,
            "p50_duration": round(_pct(durations, 0.5), 2),
        },
        "models": models,
        "errors": [{"message": k, "n": v}
                   for k, v in sorted(errors.items(), key=lambda kv: -kv[1])[:5]],
    }


def format_report(data: Dict[str, Any]) -> str:
    r = data["runs"]
    lines = [
        f"运行：{r['n']} 次  {r['by_status']}  时延 p50 {r['p50_duration']:.1f}s",
        "",
        f"{'模型':28} {'n':>4} {'ok%':>6} {'p50':>8} {'p90':>8} {'缓存':>4}",
        "-" * 64,
    ]
    for m in data["models"]:
        lines.append(
            f"{m['model'][:28]:28} {m['n']:>4} {m['ok_rate'] * 100:>5.0f}% "
            f"{m['p50']:>7.1f}s {m['p90']:>7.1f}s {m['cached']:>4}"
        )
    if data["errors"]:
        lines.append("")
        lines.append("错误 Top：")
        for e in data["errors"]:
            lines.append(f"  {e['n']}× {e['message']}")
    return "\n".join(lines)
