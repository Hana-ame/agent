"""浸泡 DAG（dag-soak）：跑 X 小时，在稳定性达标后交出「可信结果」。

和 `dag-eval` 的区别：

| | dag-eval | dag-soak |
|---|---|---|
| 目的 | 跑 N 次，给被测 DAG 打分 | **跑 X 小时**，用时间换确定性 |
| 终止 | 固定 N 次 | 时间预算到 / 连续稳定 / 达 max_runs |
| 产出 | PASS/WARN/FAIL | **收敛后的代表性输出** + 全程轨迹 |
| 合适场景 | 回归门禁 | 交差：要一份可以拿去用的结果 |

## 判定「稳定」的三个条件（全满足才算这一轮稳）

1. 最近 `window` 份输出的两两 3-gram Jaccard 均值 >= `stability`
2. 最近 `window` 份全部通过结构硬检查（非空/无 Traceback/结尾完整/到达 final_tag）
3. 目标 DAG 若导出 `verify_report`，也要全过

连续 `window` 轮都稳 → 判定收敛，立刻停（不烧完预算）。

## 用法

```bash
# 跑 0.5 小时（1800 秒），连续 5 份输出 Jaccard >= 0.85 就提前收敛
python -m dags run dag-soak --seed '{"target":"selftest-stable","hours":0.5}'

# 离线自证（3.6 秒预算）
python -m dags run dag-soak --seed '{"target":"selftest-stable","hours":0.001,"window":3}'
```

请求字段全部可选：

```json
{
  "target": "selftest-stable",
  "hours": 1.0,
  "window": 5,
  "stability": 0.85,
  "min_runs": 3,
  "max_runs": 500,
  "max_steps": 30
}
```
"""
from __future__ import annotations

import json
import statistics
import time
from collections import deque
from typing import Any, Dict, List, Optional

import dags  # noqa: F401  (触发 dag_defs 自动加载)
from dags import registry
from dags.registry import register, timed

from dag_defs._metrics import as_text as _as_text
from dag_defs._metrics import pair_jaccard, p95 as _p95
from dag_defs._metrics import structure as _structure
from dag_defs.dag_eval import _custom_verifier

__all__ = ["build_dag_soak", "DEFAULT_REQUEST"]

DEFAULT_REQUEST: Dict[str, Any] = {
    "target": "selftest-stable",
    "hours": 0.001,
    "window": 3,
    "stability": 0.85,
    "min_runs": 3,
    "max_runs": 200,
    "max_steps": 30,
}


# ---------------------------------------------------------------------------
# 边 1：浸泡
# ---------------------------------------------------------------------------
def _hard_checks(text: str, reached: bool) -> Dict[str, bool]:
    s = _structure(text)
    return {
        "non_empty": bool(text.strip()),
        "no_error_marker": not s["has_error"],
        "ends_clean": s["ends_clean"],
        "reached_final_tag": reached,
    }


@timed
def soak_run(payload: Any) -> str:
    """在时间预算内反复跑目标 DAG，直到连续稳定或预算耗尽。"""
    raw = _as_text(payload).strip()
    try:
        req = json.loads(raw) if raw else dict(DEFAULT_REQUEST)
    except json.JSONDecodeError:
        req = dict(DEFAULT_REQUEST, target=raw)

    req = {**DEFAULT_REQUEST, **req}
    name = str(req["target"])
    budget = max(0.0, float(req["hours"])) * 3600.0
    window = max(2, int(req["window"]))
    threshold = float(req["stability"])
    min_runs = max(2, int(req["min_runs"]))
    max_runs = max(min_runs, int(req["max_runs"]))
    max_steps = max(1, int(req["max_steps"]))

    spec = registry.DAG_SPECS.get(name)
    if spec is None:
        raise KeyError(f"未知 DAG: {name!r}；可用: {sorted(registry.DAG_SPECS)}")
    verifier = _custom_verifier(name)

    print(f"  [soak] 目标 {name}｜预算 {budget:.1f}s｜窗口 {window}｜Jaccard>={threshold}")
    t0 = time.time()
    records: List[Dict[str, Any]] = []
    recent: deque = deque(maxlen=window)
    streak = 0
    converged = False
    stop_reason = "budget"

    while len(records) < max_runs:
        if len(records) >= min_runs and (time.time() - t0) >= budget:
            stop_reason = "budget"
            break

        registry._timings.clear()
        rt = time.perf_counter()
        entry: Dict[str, Any] = {"i": len(records)}
        try:
            dag = spec.build()
            dag.add_content(contents=spec.seed_content, tags=[spec.seed_tag], gen=0)
            steps = dag.run(until_tag=spec.final_tag, max_steps=max_steps)
            report = dag.get_result(spec.final_tag) or ""
            entry.update(
                elapsed=time.perf_counter() - rt,
                steps=len(steps),
                report=report,
                reached=bool(dag.has_tag(spec.final_tag)),
                timings=dict(registry._timings),
            )
        except Exception as exc:  # noqa: BLE001 - 单轮崩了不该终止整场浸泡
            entry.update(
                elapsed=time.perf_counter() - rt,
                steps=0,
                report="",
                reached=False,
                timings={},
                error=f"{type(exc).__name__}: {exc}",
            )

        recent.append(entry)
        checks = _hard_checks(entry["report"], entry["reached"])
        if verifier is not None:
            try:
                checks["custom_verify"] = bool(verifier(entry["report"])[0])
            except Exception:  # noqa: BLE001
                checks["custom_verify"] = False

        # 窗口没满之前不判稳
        window_ready = len(recent) >= min_runs
        if window_ready:
            texts = [r["report"] for r in recent]
            jac = pair_jaccard(texts)
            all_ok = all(all(_hard_checks(r["report"], r["reached"]).values()) for r in recent)
        else:
            jac, all_ok = None, False

        entry["window_jaccard"] = None if jac is None else round(jac, 4)
        entry["window_ok"] = bool(all_ok and jac is not None and jac >= threshold)
        entry["checks"] = checks
        records.append(entry)

        if entry["window_ok"]:
            streak += 1
        else:
            streak = 0

        print(f"  [soak] run {entry['i'] + 1:>3}: {entry['elapsed']:.1f}s "
              f"jaccard={'n/a' if jac is None else f'{jac:.3f}'} "
              f"ok={entry['window_ok']} streak={streak}"
              + (f" ERROR {entry.get('error')}" if entry.get("error") else ""))

        if streak >= window and len(recent) >= window:
            converged = True
            stop_reason = "converged"
            break

    if not converged and len(records) >= max_runs:
        stop_reason = "max_runs"

    elapsed = time.time() - t0
    print(f"  [soak] 结束：{len(records)} 轮 / {elapsed:.1f}s / {stop_reason}")
    return json.dumps(
        {
            "target": name,
            "request": req,
            "converged": converged,
            "stop_reason": stop_reason,
            "runs": len(records),
            "elapsed": elapsed,
            "budget_seconds": budget,
            "records": [
                {k: v for k, v in r.items() if k != "report"} | {"report": r["report"]}
                for r in records
            ],
        },
        ensure_ascii=False,
    )


# ---------------------------------------------------------------------------
# 边 2：分析
# ---------------------------------------------------------------------------
@timed
def soak_analyze(payload: Any) -> str:
    data = json.loads(_as_text(payload))
    records = data["records"]
    reports = [r["report"] for r in records]
    elapsed = [r["elapsed"] for r in records]

    # 收敛窗口 = 最后 window 份输出；代表输出取其中与其它最相似的那份（最「中心」）
    window = int(data["request"]["window"])
    tail = records[-window:] if len(records) >= window else records
    tail_texts = [r["report"] for r in tail]
    from dag_defs._metrics import jaccard as _jaccard

    if len(tail_texts) >= 2:
        centrality = [
            (sum(_jaccard(t, o) for j, o in enumerate(tail_texts) if j != i) / (len(tail_texts) - 1), i)
            for i, t in enumerate(tail_texts)
        ]
        best_i = max(centrality)[1]
    else:
        best_i = 0
    representative = tail_texts[best_i] if tail_texts else ""

    fail_reasons = [r.get("error") for r in records if r.get("error")]
    jac_curve = [r.get("window_jaccard") for r in records]

    return json.dumps(
        {
            "target": data["target"],
            "converged": data["converged"],
            "stop_reason": data["stop_reason"],
            "runs": data["runs"],
            "elapsed": round(data["elapsed"], 2),
            "budget_seconds": data["budget_seconds"],
            "window": window,
            "threshold": data["request"]["stability"],
            "stability_curve": jac_curve,
            "tail_jaccard": round(
                sum(v for v in jac_curve[-window:] if v is not None)
                / max(1, len([v for v in jac_curve[-window:] if v is not None])),
                4,
            ),
            "speed": {
                "min": round(min(elapsed), 3),
                "median": round(statistics.median(elapsed), 3),
                "p95": round(_p95(elapsed), 3),
                "max": round(max(elapsed), 3),
            },
            "structure": _structure(representative),
            "representative": representative,
            "failures": len(fail_reasons),
            "failure_samples": fail_reasons[:5],
            "check_pass_rate": {
                k: round(
                    sum(1 for r in records if r["checks"].get(k)) / max(1, len(records)), 4
                )
                for k in ("non_empty", "no_error_marker", "ends_clean", "reached_final_tag", "custom_verify")
                if any(k in r["checks"] for r in records)
            },
        },
        ensure_ascii=False,
    )


# ---------------------------------------------------------------------------
# 边 3：出报告
# ---------------------------------------------------------------------------
@timed
def soak_render(payload: Any) -> str:
    m = json.loads(_as_text(payload))
    verdict = "**已收敛**" if m["converged"] else "**未收敛**"
    reason = {
        "converged": "连续窗口达标，提前停止",
        "budget": "时间预算耗尽",
        "max_runs": "达到 max_runs 上限",
    }.get(m["stop_reason"], m["stop_reason"])

    out: List[str] = [
        f"# DAG 浸泡报告 · {m['target']}",
        "",
        f"- 结论：{verdict}（{reason}）",
        f"- 预算 {m['budget_seconds']:.1f}s ｜ 实跑 **{m['runs']}** 轮 ｜ 用时 **{m['elapsed']:.1f}s**",
        f"- 收敛条件：最近 **{m['window']}** 份输出两两 Jaccard ≥ **{m['threshold']}**，且结构检查与验收器全过",
        f"- 末窗平均 Jaccard：**{m['tail_jaccard']}**",
        "",
        "## 速度",
        "",
        "| 指标 | 秒 |",
        "|---|--:|",
        f"| min | {m['speed']['min']} |",
        f"| median | {m['speed']['median']} |",
        f"| p95 | {m['speed']['p95']} |",
        f"| max | {m['speed']['max']} |",
        "",
        "## 稳定性轨迹",
        "",
        "| 轮次 | 窗口 Jaccard |",
        "|---|--:|",
    ]
    for i, v in enumerate(m["stability_curve"], 1):
        out.append(f"| {i} | {'n/a' if v is None else v} |")

    out += [
        "",
        "## 检查通过率",
        "",
        "| 检查项 | 通过率 |",
        "|---|--:|",
    ]
    for k, v in m["check_pass_rate"].items():
        out.append(f"| {k} | {v:.0%} |")

    s = m["structure"]
    out += [
        "",
        "## 代表输出（收敛窗口内最中心的一份）",
        "",
        f"字符 {s['chars']} ｜ 行 {s['lines']} ｜ 标题 {s['sections']} ｜ 结尾完整 {s['ends_clean']}",
        "",
    ]
    if m["failures"]:
        out += ["## 失败", "", f"共 {m['failures']} 轮异常：", ""]
        out += [f"- {x}" for x in m["failure_samples"]]
        out.append("")

    body = m["representative"]
    out += ["```markdown", body[:3000] + ("\n…（截断）" if len(body) > 3000 else ""), "```", ""]
    return "\n".join(out)


# ---------------------------------------------------------------------------
# 构建
# ---------------------------------------------------------------------------
@register(
    "dag-soak",
    "浸泡其他 DAG：跑 X 小时，连续 window 份输出稳定后收敛，交出代表性结果 + 全程轨迹",
    seed_tag="soak_request",
    seed_content=json.dumps(DEFAULT_REQUEST, ensure_ascii=False),
    final_tag="soak_report",
    default_output="dag_defs/out/soak_report.md",
)
def build_dag_soak():
    from dag_agent import DAG

    d = DAG(name="dag_soak")
    d.add_edge(input_tags=["soak_request"], processor=soak_run, output_tags=["soak_raw"], layer=0)
    d.add_edge(input_tags=["soak_raw"], processor=soak_analyze, output_tags=["soak_metrics"], layer=1)
    d.add_edge(input_tags=["soak_metrics"], processor=soak_render, output_tags=["soak_report"], layer=2)
    return d
