"""按历史实测 + 价格选模型（provider 自动路由）。

score = ok_rate - p50/120 - price_weight * (price_in + price_out)

数据来自同一个 sqlite 的 stats；没有调用记录的模型不参与。
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from dag8 import stats
from dag8.models import load_catalog
from dag8.store import Store

__all__ = ["choose_model"]


def choose_model(store: Store, catalog: Optional[List[Dict[str, Any]]] = None,
                 *, min_calls: int = 1,
                 price_weight: float = 0.0) -> Optional[Dict[str, Any]]:
    catalog = catalog if catalog is not None else load_catalog()
    measured = {m["model"]: m for m in stats.collect(store)["models"]}

    best: Optional[Dict[str, Any]] = None
    for c in catalog:
        key = f"{c['provider']}/{c['model']}"
        m = measured.get(key)
        if not m or m.get("n_uncached", 0) < max(1, int(min_calls)):
            continue
        ok = float(m["ok_rate"] or 0)
        p50 = float(m["p50"] or 0)
        price = float(c.get("price_in") or 0) + float(c.get("price_out") or 0)
        score = ok - p50 / 120.0 - float(price_weight) * price
        row = {**c, "key": key, "n": m["n"], "n_uncached": m["n_uncached"], "ok_rate": ok,
               "p50": p50, "score": round(score, 4)}
        if best is None or row["score"] > best["score"]:
            best = row
    return best
