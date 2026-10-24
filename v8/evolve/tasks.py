"""Task：一个可评测的任务描述 + 便宜的确定性检查。"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional


DEFAULT_CRITERIA = [
    "覆盖：是否覆盖源材料的核心论点、案例与结论",
    "准确：人名/典故/术语/因果链是否与源材料一致",
    "结构：标题、表格、阶段、支柱、勘误表是否齐全",
    "可读：是否像给人读的笔记，而不是提纲或逐字稿",
]


@dataclass
class Task:
    id: str
    title: str
    config: str
    seed: str = ""
    description: str = ""
    criteria: List[str] = field(default_factory=lambda: list(DEFAULT_CRITERIA))
    checks: Dict[str, Any] = field(default_factory=dict)
    source_path: str = ""

    @classmethod
    def load(cls, path: str | Path) -> "Task":
        p = Path(path).resolve()
        data = json.loads(p.read_text(encoding="utf-8"))
        cfg = Path(str(data["config"]))
        if not cfg.is_absolute():
            cfg = (p.parent / cfg).resolve()
        src = Path(str(data.get("source_path") or ""))
        if src and not src.is_absolute():
            src = (p.parent / src).resolve()
        return cls(
            id=str(data.get("id") or p.stem),
            title=str(data.get("title") or p.stem),
            config=str(cfg),
            seed=str(data.get("seed") or ""),
            description=str(data.get("description") or ""),
            criteria=[str(x) for x in (data.get("criteria") or DEFAULT_CRITERIA)],
            checks=dict(data.get("checks") or {}),
            source_path=str(src) if src else "",
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "title": self.title,
            "config": self.config,
            "seed": self.seed,
            "description": self.description,
            "criteria": self.criteria,
            "checks": self.checks,
            "source_path": self.source_path,
        }


def check_output(text: str, checks: Dict[str, Any],
                 leakage: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """确定性检查。返回 {checks:[...], passed, total, score, score_raw, leakage}。

    如果泄漏审计命中，score 直接乘以 0.1（用户规则）。
    """
    body = text or ""
    items: List[Dict[str, Any]] = []

    def add(name: str, ok: bool, detail: str) -> None:
        items.append({"name": name, "ok": bool(ok), "detail": detail})

    if "min_chars" in checks:
        n = len(body.strip())
        add("min_chars", n >= int(checks["min_chars"]),
            f"{n} >= {checks['min_chars']}")
    if "max_chars" in checks:
        n = len(body.strip())
        add("max_chars", n <= int(checks["max_chars"]),
            f"{n} <= {checks['max_chars']}")
    for heading in checks.get("required_headings", []):
        add(f"heading:{heading}", heading in body,
            "found" if heading in body else "missing")
    for term in checks.get("required_terms", []):
        add(f"term:{term}", term in body,
            "found" if term in body else "missing")
    for term in checks.get("forbidden_terms", []):
        add(f"forbidden:{term}", term not in body,
            "absent" if term not in body else "present")
    pattern = checks.get("pattern")
    if pattern:
        add("pattern", re.search(pattern, body, re.S) is not None,
            f"re.search({pattern!r})")

    total = len(items)
    passed = sum(1 for it in items if it["ok"])
    raw = round(100.0 * passed / total, 1) if total else 0.0
    leak = bool((leakage or {}).get("leak"))
    score = round(raw * 0.1, 1) if leak else raw
    return {
        "checks": items,
        "passed": passed,
        "total": total,
        "score": score,
        "score_raw": raw,
        "leakage": leakage or {"enabled": False, "leak": False, "multiplier": 1.0},
    }


def generic_checks() -> Dict[str, Any]:
    """没有任务特定 checks 时的结构检查（用于回看历史 run）。"""
    return {
        "min_chars": 1000,
        "max_chars": 20000,
        "required_headings": [],
        "required_terms": [],
        "forbidden_terms": ["A 国", "B 国"],
    }
