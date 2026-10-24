"""人工评价包：review.md / review.json / ratings.json + 评分/选择。"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from evolve.tasks import Task


def _read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, data: Dict[str, Any]) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def build_review(run_dir: str | Path, task: Task, records: List[Dict[str, Any]]) -> Path:
    """生成人看的 review.md 和机读的 review.json / ratings.json。"""
    base = Path(run_dir)
    base.mkdir(parents=True, exist_ok=True)

    candidates = []
    for rec in records:
        output_path = Path(rec.get("output_path", ""))
        output = output_path.read_text(encoding="utf-8") if output_path.is_file() else ""
        candidates.append({**rec, "output": output})

    review = {
        "run_id": base.name,
        "task": task.to_dict(),
        "criteria": task.criteria,
        "candidates": candidates,
        "ratings": {c["candidate_id"]: {"score": None, "note": ""} for c in candidates},
    }
    _write_json(base / "review.json", review)

    ratings = {
        "run_id": base.name,
        "task_id": task.id,
        "scale": "1-5（1=差，3=可用，5=很好）",
        "criteria": task.criteria,
        "ratings": {c["candidate_id"]: {"score": None, "note": ""} for c in candidates},
    }
    _write_json(base / "ratings.json", ratings)
    (base / "review.md").write_text(_render_md(review), encoding="utf-8")
    return base / "review.md"


def _render_md(review: Dict[str, Any]) -> str:
    lines: List[str] = []
    task = review["task"]
    lines.append(f"# 人工评价：{task['title']}")
    lines.append("")
    lines.append(f"- task: `{task['id']}`")
    lines.append(f"- run: `{review['run_id']}`")
    lines.append(f"- 候选数: {len(review['candidates'])}")
    lines.append("")
    lines.append("## 评分标准（每题 1~5 分）")
    lines.append("")
    for c in review["criteria"]:
        lines.append(f"- {c}")
    lines.append("")
    lines.append("## 评分表")
    lines.append("")
    lines.append("| 候选 | 自动检查 | 泄漏 | 人工分 | 备注 | 耗时 | 错误 |")
    lines.append("|---|---:|---|---:|---|---:|---|")
    ratings = review.get("ratings", {})
    for c in review["candidates"]:
        auto = c.get("auto", {})
        rating = ratings.get(c["candidate_id"], {})
        human = rating.get("score")
        raw = auto.get("score_raw", auto.get("score", 0))
        score = auto.get("score", 0)
        auto_txt = f"{score}"
        if raw != score:
            auto_txt += f"（原 {raw}）"
        auto_txt += f" ({auto.get('passed',0)}/{auto.get('total',0)})"
        leak = c.get("leakage") or {}
        if not leak.get("enabled"):
            leak_txt = "未启用"
        elif leak.get("leak"):
            leak_txt = "是 ×0.1"
        else:
            leak_txt = "否"
        lines.append(
            f"| `{c['candidate_id']}` · {c.get('name','')} "
            f"| {auto_txt} | {leak_txt} "
            f"| {human if human is not None else ''} | {rating.get('note','')} "
            f"| {c.get('elapsed',0)}s | {(c.get('error') or '')[:80]} |"
        )
    lines.append("")
    lines.append("> 评分方式：编辑 `ratings.json`，或运行 "
                 "`python -m evolve rate --run <dir> --candidate <id> --score 1-5 --note '...'`。")
    lines.append("")

    for i, c in enumerate(review["candidates"], 1):
        lines.append(f"## 候选 {i}：`{c['candidate_id']}` · {c.get('name','')}")
        lines.append("")
        lines.append(f"- 状态：`{c.get('status')}`；耗时 {c.get('elapsed',0)}s；"
                     f"输出 {c.get('output_chars',0)} 字符")
        lines.append(f"- 变异：`{c.get('summary',{}).get('lineage',{})}`")
        auto = c.get("auto", {})
        lines.append(f"- 自动检查：{auto.get('score',0)}"
                     f"（{auto.get('passed',0)}/{auto.get('total',0)}）")
        leak = c.get("leakage") or {}
        if leak.get("enabled"):
            lines.append(f"- 泄漏审计：{'命中 ×0.1' if leak.get('leak') else '未命中'}；"
                         f"{leak.get('reason','')}")
            for m in (leak.get("matches") or [])[:5]:
                lines.append(f"  - `{m.get('where')}`: {m.get('segment','')[:120]}")
        elif leak:
            lines.append(f"- 泄漏审计：未启用；{leak.get('reason','')}")
        if c.get("error"):
            lines.append(f"- 错误：`{c['error']}`")
        lines.append("")
        lines.append("<details><summary>展开输出全文</summary>")
        lines.append("")
        lines.append("~~~markdown")
        lines.append(c.get("output", "").rstrip())
        lines.append("~~~")
        lines.append("")
        lines.append("</details>")
        lines.append("")
    return "\n".join(lines) + "\n"


def rate_candidate(run_dir: str | Path, candidate_id: str, score: int,
                   note: str = "") -> Dict[str, Any]:
    base = Path(run_dir)
    path = base / "ratings.json"
    if not path.is_file():
        raise FileNotFoundError(f"没有 ratings.json：{path}")
    data = _read_json(path)
    if candidate_id not in data["ratings"]:
        raise KeyError(f"未知候选 {candidate_id!r}；可选：{list(data['ratings'])}")
    score = int(score)
    if score < 1 or score > 5:
        raise ValueError("score 必须在 1~5")
    data["ratings"][candidate_id] = {"score": score, "note": note}
    _write_json(path, data)
    # 同步 review.json，方便后续 select 只读一个文件
    review_path = base / "review.json"
    if review_path.is_file():
        review = _read_json(review_path)
        review.setdefault("ratings", {})[candidate_id] = {"score": score, "note": note}
        _write_json(review_path, review)
        (base / "review.md").write_text(_render_md(review), encoding="utf-8")
    return data


def select_candidate(run_dir: str | Path, by: str = "human") -> Dict[str, Any]:
    base = Path(run_dir)
    review = _read_json(base / "review.json")
    ratings = _read_json(base / "ratings.json")["ratings"] if (base / "ratings.json").is_file() else {}
    by = by.lower()
    if by not in ("human", "auto", "combined"):
        raise ValueError("--by 只支持 human / auto / combined")

    rows = []
    for c in review["candidates"]:
        cid = c["candidate_id"]
        human = (ratings.get(cid) or {}).get("score")
        auto = float(c.get("auto", {}).get("score", 0) or 0)
        if by in ("human", "combined") and human is None:
            raise ValueError(f"候选 {cid} 还没人工评分；先 rate 或改用 --by auto")
        score = float(human) if by == "human" else (
            auto if by == "auto" else 0.7 * (float(human) / 5.0 * 100.0) + 0.3 * auto
        )
        rows.append({
            "candidate_id": cid,
            "name": c.get("name", ""),
            "human": human,
            "auto": auto,
            "score": round(score, 2),
            "note": (ratings.get(cid) or {}).get("note", ""),
        })
    rows.sort(key=lambda r: (r["score"], r["auto"]), reverse=True)
    winner = rows[0] if rows else None
    result = {
        "run_id": base.name,
        "by": by,
        "winner": winner["candidate_id"] if winner else None,
        "ranking": rows,
    }
    _write_json(base / "selection.json", result)
    return result


def rebuild_review(run_dir: str | Path) -> Path:
    """用 review.json 重新渲染 review.md（改了展示逻辑后不用重跑候选）。"""
    base = Path(run_dir)
    review = _read_json(base / "review.json")
    out = base / "review.md"
    out.write_text(_render_md(review), encoding="utf-8")
    return out


def load_review(run_dir: str | Path) -> Dict[str, Any]:
    return _read_json(Path(run_dir) / "review.json")


def render_table(run_dir: str | Path) -> str:
    base = Path(run_dir)
    review = _read_json(base / "review.json")
    ratings = _read_json(base / "ratings.json")["ratings"] if (base / "ratings.json").is_file() else {}
    lines = ["| 候选 | 自动 | 泄漏 | 人工分 | 备注 |", "|---|---:|---|---:|---|"]
    for c in review["candidates"]:
        cid = c["candidate_id"]
        human = (ratings.get(cid) or {}).get("score")
        leak = c.get("leakage") or {}
        leak_txt = "未启用" if not leak.get("enabled") else ("是" if leak.get("leak") else "否")
        lines.append(
            f"| `{cid}` · {c.get('name','')} | "
            f"{c.get('auto',{}).get('score',0)} | "
            f"{leak_txt} | "
            f"{human if human is not None else '-'} | "
            f"{(ratings.get(cid) or {}).get('note','')} |"
        )
    return "\n".join(lines)
