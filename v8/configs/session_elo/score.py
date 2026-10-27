#!/usr/bin/env python3
"""给每个 session 的最终结论打确定性质量分。"""
from __future__ import annotations
import json, math, os, re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]

def clamp(x: float) -> float:
    return max(0.0, min(1.0, x))

def bigrams(t: str):
    t = re.sub(r"\s+", "", t)
    return {t[i:i+2] for i in range(max(0, len(t) - 1))}

def score_item(s: dict) -> dict:
    text = s.get("conclusion") or ""
    req = s.get("request") or ""
    n = len(text)
    length = clamp(math.log1p(n) / math.log1p(4000))
    digits = clamp(len(re.findall(r"\d+", text)) / 20)
    code = 1.0 if "```" in text else 0.0
    paths = 1.0 if re.search(r"(/[\w.\-]+){2,}", text) else 0.0
    specificity = (digits + code + paths) / 3
    headings = clamp(len(re.findall(r"(?m)^#{1,6}\s", text)) / 6)
    lists = clamp(len(re.findall(r"(?m)^\s*([-*]|\d+\.)\s", text)) / 10)
    structure = (headings + lists) / 2
    rb, cb = bigrams(req), bigrams(text)
    fit = len(rb & cb) / max(1, len(rb))
    action = clamp(sum(k in text for k in ("下一步", "建议", "风险", "结论", "因为", "所以", "TODO")) / 4)
    completion = 1.0 if s.get("status") == "completed" else (0.5 if s.get("status") else 0.0)
    substance = (length + specificity) / 2
    score = (0.30 * substance + 0.25 * fit + 0.20 * structure
             + 0.15 * action + 0.10 * completion)
    return {**s, "score": round(clamp(score), 4),
            "components": {"substance": round(substance, 3), "fit": round(fit, 3),
                           "structure": round(structure, 3), "action": round(action, 3),
                           "completion": round(completion, 3)}}

def main() -> int:
    data = json.loads(os.environ["DAG8_INPUTS"])
    scan = json.loads(data["scan"])
    src = json.loads(Path(scan["path"]).read_text(encoding="utf-8"))["items"]
    items = [score_item(x) for x in src]
    items.sort(key=lambda x: x["score"], reverse=True)
    out = ROOT / "v8" / "data" / "session_elo" / "scored.json"
    out.write_text(json.dumps({"count": len(items), "items": items}, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"path": str(out), "count": len(items),
                      "avg": round(sum(x["score"] for x in items) / max(1, len(items)), 3),
                      "top": [(x["session_id"][:20], x["score"]) for x in items[:5]]}, ensure_ascii=False))
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
