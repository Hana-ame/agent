"""评测 DAG：把其他已注册 DAG 当作被测量对象，量化稳定性 / 速度 / 正确率。

## 怎么用

```bash
python -m dags run dag-eval                                   # 默认测标定桩，秒级出结果
python -m dags run dag-eval --seed '{"target":"stage1st-ai-24h","runs":3}'   # 测真 DAG（要外网+key）
```

或者在别的 DAG 里把它当前置步骤：种子内容是 JSON

```json
{"target": "stage1st-ai-24h", "runs": 5, "max_steps": 30}
```

## 三段式

| 边 | 输入 → 输出 | 干什么 |
|---|---|---|
| `_collect` | `eval_request` → `eval_raw` | 对目标 DAG 连跑 N 次，记录耗时/步数/报告原文 |
| `_analyze` | `eval_raw` → `eval_metrics` | 算三项指标 |
| `_render`  | `eval_metrics` → `eval_report` | 出 Markdown 报告 + PASS/WARN/FAIL |

## 三项指标怎么算

**稳定性**（同样的输入，跑 N 次结果像不像）
- 结构字段一致率：chars / lines / nonempty_lines / sections / ends_clean / has_error
  逐字段取众数占比，1.0 = 每次都一样
- 文本相似度：所有两两组合的 3-gram Jaccard 均值（0~1，模型输出天然抖，0.6 以上算稳）
- 长度变异系数 CV = std / mean

**速度**
- elapsed 的 min / 中位数 / p95 / max
- 每条 edge 的平均耗时（取自目标 DAG 里 `timed()` 包装的处理器）

**正确率**（离线可判的硬条件，不需要人看）
- `non_empty`：报告非空
- `no_error_marker`：不含 Traceback / Error: / Exception
- `ends_clean`：以句末标点或闭合符号结尾（防截断）
- `min_lines`：非空行 >= 5
- `has_sections`：至少一个 `#` 标题

综合判定：三项各自有阈值，任一 WARN 整体就 WARN，任一 FAIL 就 FAIL。
"""
from __future__ import annotations

import json
import math
import re
import statistics
import sys
import time
from typing import Any, Dict, List, Optional

import dags  # noqa: F401  (触发 dag_defs 自动加载)
from dags import registry
from dags.registry import register, timed

__all__ = ["build_dag_eval", "DEFAULT_REQUEST"]

# 默认目标用标定桩：不带参数直接跑必须能立刻出结果。
# 想评真 DAG 就显式指定，例如：
#   python -m dags run dag-eval --seed '{"target":"stage1st-ai-24h","runs":3}'
DEFAULT_REQUEST: Dict[str, Any] = {"target": "selftest-stable", "runs": 3, "max_steps": 30}

# 阈值：任一 FAIL 判 FAIL，任一 WARN 判 WARN
THRESHOLDS = {
    "field_agreement_warn": 0.99,
    "jaccard_warn": 0.60,
    "cv_warn": 0.35,
    "p95_seconds_warn": 120.0,
    "correctness_warn": 1.0,
}


# ---------------------------------------------------------------------------
# 工具（指标实现见 dag_defs/_metrics.py，评测与浸泡共用）
# ---------------------------------------------------------------------------
from dag_defs._metrics import (  # noqa: E402
    agreement as _agreement,
    as_text as _as_text,
    jaccard as _jaccard,
    p95 as _p95,
    structure as _structure,
)


def _custom_verifier(target: str):
    """取目标 DAG 自带的正確率判据 `verify_report(text) -> (bool, reason)`；没有返回 None。"""
    if not target:
        return None
    spec = registry.DAG_SPECS.get(target)
    if spec is None:
        return None
    module = sys.modules.get(getattr(spec.build, "__module__", ""))
    fn = getattr(module, "verify_report", None)
    return fn if callable(fn) else None


# ---------------------------------------------------------------------------
# 边 1：采集
# ---------------------------------------------------------------------------
@timed
def eval_collect(payload: Any) -> str:
    """对目标 DAG 连跑 N 次，采集耗时/步数/报告。"""
    raw = _as_text(payload).strip()
    try:
        req = json.loads(raw) if raw else dict(DEFAULT_REQUEST)
    except json.JSONDecodeError:
        req = dict(DEFAULT_REQUEST, target=raw)

    name = str(req.get("target") or DEFAULT_REQUEST["target"])
    runs = max(1, int(req.get("runs", 3)))
    max_steps = max(1, int(req.get("max_steps", 30)))

    spec = registry.DAG_SPECS.get(name)
    if spec is None:
        raise KeyError(f"未知 DAG: {name!r}；可用: {sorted(registry.DAG_SPECS)}")

    records: List[Dict[str, Any]] = []
    for i in range(runs):
        registry._timings.clear()
        t0 = time.perf_counter()
        entry: Dict[str, Any] = {"i": i}
        try:
            dag = spec.build()
            dag.add_content(contents=spec.seed_content, tags=[spec.seed_tag], gen=0)
            steps = dag.run(until_tag=spec.final_tag, max_steps=max_steps)
            report = dag.get_result(spec.final_tag) or ""
            entry.update(
                elapsed=time.perf_counter() - t0,
                steps=len(steps),
                report=report,
                reached=bool(dag.has_tag(spec.final_tag)),
                timings=dict(registry._timings),
            )
        except Exception as exc:  # noqa: BLE001 - 被测 DAG 崩了也要出报告
            entry.update(
                elapsed=time.perf_counter() - t0,
                steps=0,
                report="",
                reached=False,
                timings={},
                error=f"{type(exc).__name__}: {exc}",
            )
        records.append(entry)
        print(f"  [eval] run {i + 1}/{runs}: {entry['elapsed']:.1f}s "
              f"{len(entry['report'])} chars steps={entry['steps']}"
              + (f" ERROR {entry.get('error')}" if entry.get("error") else ""))

    print(f"  [eval] 采集完成：{name} × {runs}")
    return json.dumps({"target": name, "runs": runs, "records": records}, ensure_ascii=False)


# ---------------------------------------------------------------------------
# 边 2：分析
# ---------------------------------------------------------------------------
@timed
def eval_analyze(payload: Any) -> str:
    data = json.loads(_as_text(payload))
    records = data["records"]
    reports = [r["report"] for r in records]
    structs = [_structure(t) for t in reports]

    # --- 稳定性 ---
    fields = [
        "chars", "lines", "nonempty_lines", "sections", "ends_clean", "has_error",
    ]
    field_agreement = {f: round(_agreement([s[f] for s in structs]), 4) for f in fields}

    pairs = [
        _jaccard(reports[i], reports[j])
        for i in range(len(reports))
        for j in range(i + 1, len(reports))
    ]
    mean_jaccard = round(statistics.fmean(pairs), 4) if pairs else 1.0

    lens = [len(t) for t in reports]
    mean_len = statistics.fmean(lens) if lens else 0.0
    cv = round((statistics.pstdev(lens) / mean_len), 4) if mean_len else 0.0

    # --- 速度 ---
    elapsed = [r["elapsed"] for r in records]
    per_edge: Dict[str, List[float]] = {}
    for r in records:
        for k, v in (r.get("timings") or {}).items():
            per_edge.setdefault(k, []).append(float(v))
    speed = {
        "min": round(min(elapsed), 3),
        "median": round(statistics.median(elapsed), 3),
        "p95": round(_p95(elapsed), 3),
        "max": round(max(elapsed), 3),
        "per_edge_mean": {k: round(statistics.fmean(v), 3) for k, v in per_edge.items()},
    }

    # --- 正确率 ---
    checks = {
        "non_empty": [bool(t.strip()) for t in reports],
        "no_error_marker": [not s["has_error"] and not r.get("error") for s, r in zip(structs, records)],
        "ends_clean": [s["ends_clean"] for s in structs],
        "min_lines": [s["nonempty_lines"] >= 5 for s in structs],
        "has_sections": [s["sections"] >= 1 for s in structs],
        "reached_final_tag": [bool(r.get("reached")) for r in records],
    }

    # 目标 DAG 可以自带验收器：在定义模块里导出 verify_report(text) -> (bool, reason)。
    # 有就用它算「语义正确率」，没有就只能靠上面的结构硬检查。
    custom = _custom_verifier(data.get("target", ""))
    if custom is not None:
        results = []
        for t in reports:
            try:
                ok, _why = custom(t)
                results.append(bool(ok))
            except Exception:  # noqa: BLE001 - 验收器自己崩了算不通过
                results.append(False)
        checks["custom_verify"] = results

    pass_rate = {k: round(sum(v) / len(v), 4) for k, v in checks.items()}

    return json.dumps(
        {
            "target": data["target"],
            "runs": data["runs"],
            "stability": {
                "field_agreement": field_agreement,
                "mean_jaccard": mean_jaccard,
                "length_cv": cv,
                "lengths": lens,
            },
            "speed": speed,
            "correctness": pass_rate,
            "errors": [r.get("error") for r in records if r.get("error")],
        },
        ensure_ascii=False,
    )


# ---------------------------------------------------------------------------
# 边 3：出报告
# ---------------------------------------------------------------------------
def _verdict(metrics: Dict[str, Any]) -> tuple[str, List[str]]:
    notes: List[str] = []
    level = "PASS"

    def bump(new: str, why: str) -> None:
        nonlocal level
        order = {"PASS": 0, "WARN": 1, "FAIL": 2}
        if order[new] > order[level]:
            level = new
        notes.append(why)

    st, sp, co = metrics["stability"], metrics["speed"], metrics["correctness"]

    low_fields = [k for k, v in st["field_agreement"].items() if v < THRESHOLDS["field_agreement_warn"]]
    if low_fields:
        bump("WARN", f"结构字段不稳定：{', '.join(low_fields)}")
    if st["mean_jaccard"] < THRESHOLDS["jaccard_warn"]:
        bump("WARN", f"文本相似度 {st['mean_jaccard']} < {THRESHOLDS['jaccard_warn']}")
    if st["length_cv"] > THRESHOLDS["cv_warn"]:
        bump("WARN", f"长度变异 CV {st['length_cv']} > {THRESHOLDS['cv_warn']}")

    if sp["p95"] > THRESHOLDS["p95_seconds_warn"]:
        bump("WARN", f"p95 {sp['p95']}s > {THRESHOLDS['p95_seconds_warn']}s")

    hard = {"non_empty", "no_error_marker", "ends_clean", "reached_final_tag"}
    for k, v in co.items():
        if v < 1.0:
            bump("FAIL" if k in hard else "WARN", f"正确率检查未全过：{k} = {v}")
    if metrics["errors"]:
        bump("FAIL", f"运行期异常 {len(metrics['errors'])} 次")
    return level, notes


@timed
def eval_render(payload: Any) -> str:
    m = json.loads(_as_text(payload))
    st, sp, co = m["stability"], m["speed"], m["correctness"]
    level, notes = _verdict(m)

    out: List[str] = [
        f"# DAG 评测报告 · {m['target']}",
        "",
        f"- 运行次数：**{m['runs']}**",
        f"- 判定：**{level}**",
        "",
    ]
    if notes:
        out += ["## 判定依据", ""] + [f"- {n}" for n in notes] + [""]
    else:
        out += ["## 判定依据", "", "- 全部指标在阈值内", ""]

    out += [
        "## 稳定性",
        "",
        "| 字段 | 一致率 |",
        "|---|--:|",
    ]
    for k, v in st["field_agreement"].items():
        out.append(f"| {k} | {v:.3f} |")
    out += [
        "",
        f"- 文本 3-gram Jaccard 均值：**{st['mean_jaccard']:.3f}**",
        f"- 长度变异系数 CV：**{st['length_cv']:.3f}**",
        f"- 各次长度：{', '.join(str(x) for x in st['lengths'])}",
        "",
        "## 速度",
        "",
        "| 指标 | 秒 |",
        "|---|--:|",
        f"| min | {sp['min']:.3f} |",
        f"| median | {sp['median']:.3f} |",
        f"| p95 | {sp['p95']:.3f} |",
        f"| max | {sp['max']:.3f} |",
        "",
    ]
    if sp["per_edge_mean"]:
        out += ["| 处理器 | 平均耗时(秒) |", "|---|--:|"]
        for k, v in sorted(sp["per_edge_mean"].items(), key=lambda kv: -kv[1]):
            out.append(f"| {k} | {v:.3f} |")
        out.append("")

    out += [
        "## 正确率",
        "",
        "| 检查项 | 通过率 |",
        "|---|--:|",
    ]
    for k, v in co.items():
        out.append(f"| {k} | {v:.0%} |")
    out.append("")

    if m["errors"]:
        out += ["## 异常", ""] + [f"- {e}" for e in m["errors"]] + [""]

    out += ["## 阈值", "", "```json", json.dumps(THRESHOLDS, ensure_ascii=False, indent=2), "```", ""]
    return "\n".join(out)


# ---------------------------------------------------------------------------
# 构建
# ---------------------------------------------------------------------------
@register(
    "dag-eval",
    "评测其他 DAG：跑 N 次量化稳定性（字段一致率/文本相似度/长度变异）、速度（min/median/p95）与正确率（结构硬检查）",
    seed_tag="eval_request",
    seed_content=json.dumps(DEFAULT_REQUEST, ensure_ascii=False),
    final_tag="eval_report",
    default_output="dag_defs/out/eval_report.md",
)
def build_dag_eval():
    from dag_agent import DAG

    d = DAG(name="dag_eval")
    d.add_edge(
        input_tags=["eval_request"],
        processor=eval_collect,
        output_tags=["eval_raw"],
        layer=0,
    )
    d.add_edge(
        input_tags=["eval_raw"],
        processor=eval_analyze,
        output_tags=["eval_metrics"],
        layer=1,
    )
    d.add_edge(
        input_tags=["eval_metrics"],
        processor=eval_render,
        output_tags=["eval_report"],
        layer=2,
    )
    return d
