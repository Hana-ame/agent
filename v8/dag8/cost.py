"""按 sqlite 运行记录估算 tokens 与费用。

pi 不返回 usage，所以 token 用字符估算（chars/2），并在报告里明确标注。
价格来自 configs/providers.json（`price_in` / `price_out`，单位：每百万 token）。

    python -m dag8 cost                 # 最近 20 次运行
    python -m dag8 --db path cost --json
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from dag8.store import Store

V8 = Path(__file__).resolve().parent.parent
PROVIDERS = V8 / "configs" / "providers.json"

__all__ = ["load_prices", "run_costs", "format_report"]

Price = Tuple[float, float]


def load_prices(path: Optional[str | Path] = None) -> Dict[Tuple[str, str], Price]:
    p = Path(path) if path else PROVIDERS
    data = json.loads(p.read_text(encoding="utf-8"))
    out: Dict[Tuple[str, str], Price] = {}
    for provider, cfg in data.items():
        if str(provider).startswith("_") or not isinstance(cfg, dict):
            continue
        for model, meta in (cfg.get("models") or {}).items():
            meta = meta or {}
            out[(str(provider), str(model))] = (
                float(meta.get("price_in", 0) or 0),
                float(meta.get("price_out", 0) or 0),
            )
    return out


def run_costs(store: Store, prices: Dict[Tuple[str, str], Price],
              limit: int = 20) -> List[Dict[str, Any]]:
    rows = store.conn.execute(
        "SELECT run_id, dag_name, status, started_at FROM runs "
        "ORDER BY started_at DESC LIMIT ?", (limit,)
    ).fetchall()

    out: List[Dict[str, Any]] = []
    for r in rows:
        edges = store.conn.execute(
            "SELECT e.model, e.prompt_chars, LENGTH(c.content) AS out_chars "
            "FROM edges e LEFT JOIN contents c ON c.id = e.content_id "
            "WHERE e.run_id=? AND e.status='ok'", (r["run_id"],)
        ).fetchall()
        tin = tout = 0.0
        cost = 0.0
        unknown = []
        for e in edges:
            ti = (e["prompt_chars"] or 0) / 2.0
            to = (e["out_chars"] or 0) / 2.0
            tin += ti
            tout += to
            model = str(e["model"] or "")
            prov, _, name = model.partition("/")
            if name and (prov, name) in prices:
                pi, po = prices[(prov, name)]
                cost += (ti * pi + to * po) / 1e6
            elif model:
                unknown.append(model)
        out.append({
            "run_id": r["run_id"],
            "dag": r["dag_name"],
            "status": r["status"],
            "tokens_in": round(tin),
            "tokens_out": round(tout),
            "cost": round(cost, 6),
            "unknown_models": sorted(set(unknown)),
        })
    return out


def format_report(rows: List[Dict[str, Any]]) -> str:
    lines = [
        f"{'run_id':14} {'dag':20} {'status':10} {'tokens≈':>9} {'cost':>10}",
        "-" * 70,
    ]
    total = 0.0
    for r in rows:
        total += r["cost"]
        lines.append(
            f"{r['run_id']:14} {r['dag'][:20]:20} {r['status']:10} "
            f"{r['tokens_in'] + r['tokens_out']:>9} {r['cost']:>10.6f}"
        )
    lines.append("-" * 70)
    lines.append(f"{'合计':14} {'':20} {'':10} {'':>9} {total:>10.6f}")
    lines.append("")
    lines.append("注：tokens 由字符/2 估算（pi 不返回 usage）；价格见 configs/providers.json（0 = 免费/未配）")
    return "\n".join(lines)
