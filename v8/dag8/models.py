"""模型记分卡：provider 价格 + 实测成功率/时延。

价格来自 configs/providers.json；成功率/时延来自 sqlite 的历史 edges。
没有调用记录的模型也会列出（n=0），方便比较。

    python -m dag8 models
    python -m dag8 --db path models --json
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from dag8 import stats
from dag8.store import Store

V8 = Path(__file__).resolve().parent.parent
PROVIDERS = V8 / "configs" / "providers.json"

__all__ = ["load_catalog", "scorecard", "format_report"]


def load_catalog(path: Optional[str | Path] = None) -> List[Dict[str, Any]]:
    p = Path(path) if path else PROVIDERS
    data = json.loads(p.read_text(encoding="utf-8"))
    out: List[Dict[str, Any]] = []
    for provider, cfg in data.items():
        if str(provider).startswith("_") or not isinstance(cfg, dict):
            continue
        for model, meta in (cfg.get("models") or {}).items():
            meta = meta or {}
            out.append({
                "provider": str(provider),
                "model": str(model),
                "price_in": float(meta.get("price_in", 0) or 0),
                "price_out": float(meta.get("price_out", 0) or 0),
            })
    return out


def scorecard(store: Store, catalog: List[Dict[str, Any]],
              limit: int = 50) -> List[Dict[str, Any]]:
    data = stats.collect(store, limit)
    measured = {m["model"]: m for m in data["models"]}

    rows: List[Dict[str, Any]] = []
    seen = set()
    for c in catalog:
        key = f"{c['provider']}/{c['model']}"
        seen.add(key)
        m = measured.get(key)
        rows.append({
            "key": key,
            "provider": c["provider"],
            "model": c["model"],
            "price_in": c["price_in"],
            "price_out": c["price_out"],
            "n": m["n"] if m else 0,
            "ok_rate": m["ok_rate"] if m else None,
            "p50": m["p50"] if m else None,
            "p90": m["p90"] if m else None,
            "cached": m["cached"] if m else 0,
        })

    for key, m in measured.items():
        if key in seen:
            continue
        rows.append({
            "key": key, "provider": "?", "model": key,
            "price_in": None, "price_out": None,
            "n": m["n"], "ok_rate": m["ok_rate"],
            "p50": m["p50"], "p90": m["p90"], "cached": m["cached"],
        })

    rows.sort(key=lambda r: (-(r["n"] or 0), r["key"]))
    return rows


def format_report(rows: List[Dict[str, Any]]) -> str:
    lines = [
        f"{'模型':32} {'in/1M':>8} {'out/1M':>9} {'n':>4} {'ok%':>6} {'p50':>8} {'p90':>8}",
        "-" * 82,
    ]
    for r in rows:
        price_in = "-" if r["price_in"] is None else f"{r['price_in']:.2f}"
        price_out = "-" if r["price_out"] is None else f"{r['price_out']:.2f}"
        ok = "-" if r["ok_rate"] is None else f"{r['ok_rate'] * 100:.0f}%"
        p50 = "-" if r["p50"] is None else f"{r['p50']:.1f}s"
        p90 = "-" if r["p90"] is None else f"{r['p90']:.1f}s"
        lines.append(
            f"{r['key'][:32]:32} {price_in:>8} {price_out:>9} {r['n']:>4} "
            f"{ok:>6} {p50:>8} {p90:>8}"
        )
    lines.append("")
    lines.append("注：价格为每百万 token，来自 configs/providers.json；n/ok/p50 来自历史运行")
    return "\n".join(lines)
