#!/usr/bin/env python3
"""调度搜索：在 v8 图上枚举「节点粒度 × 拓扑 × 并发」，找速度/质量更优的调度形式。

## 搜索流程（预算内 successive halving）

1. **候选生成**：每种调度形式生成一张 v8 配置（必要时附带 v7 子图）
   - `single`   1 个 pi 节点
   - `chain2`   2 个 pi 节点串行（要点 → 成文）
   - `chain3`   3 个 pi 节点串行（初稿 → 批评 → 定稿）
   - `fanout`   2 个 pi 节点并行（定义 / 好处）→ 合并节点
   - `v7_only`  1 个 graph 节点，内部是 2 条边串行的 v7 图
   - `hybrid`   pi 节点（要点）→ v7 图节点（成文）
2. **粗筛**：每个候选 × `--workers` 网格各跑 1 次，得 (质量, 时延)
3. **复测**：按 `质量 / 时延` 取 top-3，补跑到 3 次，取质量均值 / 时延中位数
4. **输出**：Pareto 前沿 + 排名表 + 最佳配置目录

## 指标

- 时延：整次 `dag8.run` 的 wall time（评审时间不计入）
- 质量：固定评审 prompt 打三项（覆盖 / 准确 / 简洁），各 1~5，合计 3~15
- 效率：质量 / 时延；同质量比速度，同速度比质量

## 用法

```bash
cd v8
python scripts/search_schedule.py --workers 1,3 --max-runs 14
python scripts/search_schedule.py --forms single,fanout --dry-run   # 只自检配置
```

真实搜索会调用 pi 与评审模型（免 key 网关），一次完整搜索约数分钟。
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import re
import statistics
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

V8 = Path(__file__).resolve().parent.parent
ROOT = V8.parent
for p in (str(V8), str(ROOT)):
    if p not in sys.path:
        sys.path.insert(0, p)

import dag8  # noqa: E402
from dag.llm import OpenAICompatibleClient  # noqa: E402
from dag.schema import ModelConfig  # noqa: E402
from dag8.pi import PiAgent, PiError  # noqa: E402

GATEWAY = "https://laptop-8000.moonchan.xyz/v1"
GEN_MODEL = "deepseek-v4-flash"
JUDGE_MODEL = "deepseek-v4-flash"
PROVIDERS_PATH = V8 / "configs" / "providers.json"

def load_providers() -> Dict[str, Any]:
    try:
        return json.loads(PROVIDERS_PATH.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def apply_model(cand: Candidate, provider: str, model: str,
                providers: Dict[str, Any]) -> Candidate:
    """复制一份候选并把模型换成指定值（pi 节点 + v7 图内联 models 一起换）。"""
    c = copy.deepcopy(cand)
    c.name = f"{cand.name}@{model}"
    c.config["pi"]["provider"] = provider
    c.config["pi"]["model"] = model
    base_url = (providers.get(provider) or {}).get("base_url", GATEWAY)
    for graph in c.files.values():
        for m in (graph.get("models") or {}).values():
            m["model"] = model
            m["api"] = base_url
            m["key"] = "not-needed"
    return c


PI_BASE = {
    "provider": "laptop",
    "model": GEN_MODEL,
    "tools": False,
    "timeout": 240,
    "retries": 3,
}

TASK = (
    "写一篇 150~250 字的中文短文，解释「标签驱动的 DAG」。必须覆盖三点："
    "① 用标签代替显式连线；② 调度器按标签匹配依赖、决定执行顺序；"
    "③ 这种设计的好处（解耦、易演化）。只输出正文。"
)

JUDGE_SYSTEM = "你是严格的写作评审，只输出 JSON。"
JUDGE_PROMPT = """任务：{task}

候选答案：
{answer}

按三项各打 1~5 分：
1) 覆盖：是否覆盖 ①标签替代显式连线 ②标签匹配决定调度 ③好处
2) 准确：概念与术语是否准确
3) 简洁：是否 150~250 字、无注水

只输出 JSON：{{"coverage":1-5,"accuracy":1-5,"concision":1-5,"note":"一句话"}}"""


# ---------------------------------------------------------------------------
# 候选：每种调度形式 = 一张 v8 配置（可选 v7 子图）
# ---------------------------------------------------------------------------
@dataclass
class Candidate:
    name: str
    config: Dict[str, Any]
    files: Dict[str, Dict[str, Any]] = field(default_factory=dict)  # 相对路径 → v7 图 JSON


def _v7(edges: List[Dict[str, Any]]) -> Dict[str, Any]:
    return {
        "name": "v7_sub",
        "models": {
            "laptop": {
                "api": GATEWAY,
                "key": "not-needed",
                "model": GEN_MODEL,
                "params": {"temperature": 0.2},
            }
        },
        "edges": edges,
    }


def _edge(eid: str, layer: int, ins: List[str], outs: List[str], prompt: str) -> Dict[str, Any]:
    return {"id": eid, "layer": layer, "input_tags": ins, "output_tags": outs, "prompt": prompt}


def _base(name: str, edges: List[Dict[str, Any]], files: Optional[Dict[str, Dict[str, Any]]] = None) -> Candidate:
    return Candidate(
        name=name,
        config={
            "name": name,
            "seed_tag": "topic",
            "seed_content": TASK,
            "result_tag": "final",
            "pi": PI_BASE,
            "edges": edges,
        },
        files=files or {},
    )


def cand_single() -> Candidate:
    return _base("single", [
        _edge("write", 0, ["topic"], ["final"],
              "任务：{contents}\n\n直接输出成品正文，不要开场白。"),
    ])


def cand_chain2() -> Candidate:
    return _base("chain2", [
        _edge("points", 0, ["topic"], ["points"],
              "把下面的任务拆成 3 条要点，每条一行：\n{contents}"),
        _edge("write", 1, ["points"], ["final"],
              "根据下面的要点写一篇 150~250 字短文，覆盖 ①标签替代显式连线 "
              "②标签匹配决定调度 ③好处：\n{contents}\n\n只输出正文。"),
    ])


def cand_chain3() -> Candidate:
    return _base("chain3", [
        _edge("draft", 0, ["topic"], ["draft"],
              "任务：{contents}\n\n写出初稿。"),
        _edge("critique", 1, ["draft"], ["critique"],
              "下面初稿只挑 1 个最关键的不足，一句话说清，不要改写：\n{contents}"),
        _edge("final", 2, ["draft", "critique"], ["final"],
              "根据初稿和批评给出定稿，只输出正文：\n【初稿】\n{tag:draft}\n\n【批评】\n{tag:critique}"),
    ])


def cand_fanout() -> Candidate:
    return _base("fanout", [
        _edge("definition", 0, ["topic"], ["definition"],
              "用一句话给「标签驱动的 DAG」下定义。任务背景：{contents}"),
        _edge("benefit", 0, ["topic"], ["benefit"],
              "用一句话说明这种设计最大的好处（解耦/易演化）。任务背景：{contents}"),
        _edge("merge", 1, ["definition", "benefit"], ["final"],
              "把下面两句话扩写成 150~250 字短文，覆盖 ①用标签代替显式连线 "
              "②调度器按标签匹配 ③好处；只输出正文：\n"
              "【定义】{tag:definition}\n【好处】{tag:benefit}"),
    ])


def cand_v7_only() -> Candidate:
    graph = _v7([
        {"id": "outline", "layer": 0, "input_tags": ["topic"], "output_tags": ["outline"],
         "processor": {"model": "laptop",
                       "prompt": "给下面的任务列 3 条要点，每条一行：\n{contents}"}},
        {"id": "write", "layer": 1, "input_tags": ["outline"], "output_tags": ["final"],
         "processor": {"model": "laptop",
                       "prompt": "根据要点写 150~250 字短文，覆盖 ①标签替代显式连线 "
                                 "②标签匹配决定调度 ③好处；只输出正文：\n{contents}"}},
    ])
    return _base("v7_only", [
        {"id": "graph", "kind": "graph", "layer": 0,
         "input_tags": ["topic"], "output_tags": ["final"],
         "graph": {"path": "v7.json", "seed_tag": "topic", "result_tag": "final", "max_steps": 10}},
    ], files={"v7.json": graph})


def cand_hybrid() -> Candidate:
    graph = _v7([
        {"id": "write", "layer": 0, "input_tags": ["points"], "output_tags": ["final"],
         "processor": {"model": "laptop",
                       "prompt": "根据下面的要点写 150~250 字短文，覆盖 ①标签替代显式连线 "
                                 "②标签匹配决定调度 ③好处；只输出正文：\n{contents}"}},
    ])
    return _base("hybrid", [
        _edge("points", 0, ["topic"], ["points"],
              "把下面的任务拆成 3 条要点，每条一行：\n{contents}"),
        {"id": "write", "kind": "graph", "layer": 1,
         "input_tags": ["points"], "output_tags": ["final"],
         "graph": {"path": "v7.json", "seed_tag": "points", "result_tag": "final", "max_steps": 5}},
    ], files={"v7.json": graph})


FORMS = {
    "single": cand_single,
    "chain2": cand_chain2,
    "chain3": cand_chain3,
    "fanout": cand_fanout,
    "v7_only": cand_v7_only,
    "hybrid": cand_hybrid,
}


# ---------------------------------------------------------------------------
# 评审
# ---------------------------------------------------------------------------
def make_judge(retries: int = 4, sleep_base: float = 4.0, gap: float = 3.0,
               limiter=None):
    """评审客户端：网关共享 RPM，被限流时指数退避；仍失败则返回 None（不中断搜索）。"""
    client = OpenAICompatibleClient(timeout=120)
    model = ModelConfig(key="judge", endpoint=GATEWAY, apikey="not-needed",
                        model=JUDGE_MODEL, params={"temperature": 0.0})
    cache: Dict[str, Any] = {}
    last_call = {"t": 0.0}

    def _wait_gap() -> None:
        wait = gap - (time.perf_counter() - last_call["t"])
        if wait > 0:
            time.sleep(wait)
        last_call["t"] = time.perf_counter()

    def judge(answer: str):
        key = hashlib.sha256(answer.encode("utf-8")).hexdigest()
        if key in cache:
            return cache[key]

        _wait_gap()
        raw = ""
        last_exc: Optional[Exception] = None
        for attempt in range(retries):
            try:
                if limiter is not None:
                    limiter.acquire()
                raw = client.complete(
                    JUDGE_PROMPT.format(task=TASK, answer=answer),
                    model_config=model, system_prompt=JUDGE_SYSTEM)
                break
            except Exception as exc:  # noqa: BLE001 - 限流/网络都退避
                last_exc = exc
                if attempt < retries - 1:
                    time.sleep(sleep_base * (2 ** attempt))
        else:
            note = f"judge-error: {last_exc}"
            cache[key] = (None, note[:120])
            return cache[key]

        m = re.search(r"\{.*\}", raw, re.S)
        try:
            data = json.loads(m.group(0)) if m else {}
            score = (float(data["coverage"]) + float(data["accuracy"])
                     + float(data["concision"]))
            if not 3.0 <= score <= 15.0:
                raise ValueError("score out of range")
            note = str(data.get("note", ""))[:80]
        except Exception:  # noqa: BLE001 - 评审解析失败就当 0 分并留痕
            score, note = 0.0, f"judge-parse-failed: {raw[:80]}"
        cache[key] = (score, note)
        return score, note

    return judge


# ---------------------------------------------------------------------------
# 单次评测
# ---------------------------------------------------------------------------
@dataclass
class Trial:
    candidate: Candidate
    cfg_path: Path
    db_path: Path
    workers: int
    latencies: List[float] = field(default_factory=list)
    qualities: List[float] = field(default_factory=list)
    outputs: List[str] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)
    failures: int = 0
    judge_fails: int = 0
    model: str = ""
    price_in: float = 0.0
    price_out: float = 0.0
    tokens: List[float] = field(default_factory=list)
    costs: List[float] = field(default_factory=list)

    @property
    def latency(self) -> float:
        return statistics.median(self.latencies) if self.latencies else float("inf")

    @property
    def quality(self) -> float:
        return statistics.mean(self.qualities) if self.qualities else 0.0

    @property
    def efficiency(self) -> float:
        return self.quality / max(self.latency, 0.05)

    @property
    def runs(self) -> int:
        return len(self.latencies)

    @property
    def tokens_run(self) -> float:
        return statistics.median(self.tokens) if self.tokens else 0.0

    @property
    def cost_run(self) -> float:
        return statistics.median(self.costs) if self.costs else 0.0


def write_candidate(cand: Candidate, root: Path) -> Path:
    d = root / cand.name
    d.mkdir(parents=True, exist_ok=True)
    for rel, data in cand.files.items():
        p = d / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    cfg_path = d / "dag.json"
    cfg_path.write_text(json.dumps(cand.config, ensure_ascii=False, indent=2) + "\n",
                        encoding="utf-8")
    return cfg_path


def evaluate(cand: Candidate, cfg_path: Path, db_path: Path, workers: int,
             judge, runs: int, *, model: str = "", price_in: float = 0.0,
             price_out: float = 0.0) -> Trial:
    t = Trial(candidate=cand, cfg_path=cfg_path, db_path=db_path, workers=workers,
              model=model, price_in=price_in, price_out=price_out)
    for _ in range(runs):
        t0 = time.perf_counter()
        res = dag8.run(str(cfg_path), db_path=str(db_path), max_workers=workers)
        t.latencies.append(time.perf_counter() - t0)
        tin = sum(e.prompt_chars for e in res.edges) / 2.0
        tout = sum(e.output_chars for e in res.edges) / 2.0
        t.tokens.append(tin + tout)
        t.costs.append((tin * price_in + tout * price_out) / 1e6)
        if res.status != "ok" or not res.result:
            t.failures += 1
            t.qualities.append(0.0)
            t.outputs.append("")
            t.notes.append(f"run status={res.status}")
            continue
        q, note = judge(res.result)
        t.outputs.append(res.result)
        t.notes.append(note)
        if q is None:  # 评审被限流/失败：保留时延，质量记为缺失
            t.judge_fails += 1
        else:
            t.qualities.append(q)
    return t


# ---------------------------------------------------------------------------
# 搜索 + 报告
# ---------------------------------------------------------------------------
def pareto_front(trials: List[Trial]) -> List[Trial]:
    """质量越高、时延越低越好；返回未被支配的点。"""
    front = []
    for t in trials:
        dominated = any(
            other is not t
            and other.quality >= t.quality
            and other.latency <= t.latency
            and (other.quality > t.quality or other.latency < t.latency)
            for other in trials
        )
        if not dominated:
            front.append(t)
    return sorted(front, key=lambda t: t.latency)


def fmt_table(trials: List[Trial]) -> str:
    ranked = sorted(trials, key=lambda t: -t.efficiency)
    lines = [
        "| 调度形式 | 模型 | workers | 质量(mean) | 时延 p50 | tokens/次 | 成本/次 | 效率(质量/s) | runs | 备注 |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for t in ranked:
        note = t.notes[-1] if t.notes else ""
        if t.judge_fails:
            note = f"{note}（评审失败 {t.judge_fails} 次）"
        form = t.candidate.name.split("@")[0]
        lines.append(
            f"| {form} | {t.model or '-'} | {t.workers} | {t.quality:.2f} | {t.latency:.2f}s | "
            f"{t.tokens_run:.0f} | {t.cost_run:.2f} | {t.efficiency:.3f} | {t.runs} | {note} |"
        )
    return "\n".join(lines)


def max_wave_width(config: Dict[str, Any]) -> int:
    """同一 layer 最多有几个节点 —— 只有 >1 时 workers 才可能产生影响。"""
    per_layer: Dict[int, int] = {}
    for e in config.get("edges", []):
        layer = int(e.get("layer", 0))
        per_layer[layer] = per_layer.get(layer, 0) + 1
    return max(per_layer.values(), default=1)


def search(cands: List[Candidate], workers_grid: List[int], max_runs: int,
           judge, outdir: Path, repeats: int = 2, target: int = 3,
           models: Optional[List[Tuple[str, str]]] = None,
           providers: Optional[Dict[str, Any]] = None) -> List[Trial]:
    trials: List[Trial] = []
    used = 0
    skipped: List[str] = []
    model_specs = models or [("laptop", GEN_MODEL)]
    providers = providers or {}

    # 阶段 0：候选 × 模型 × 并发；只保留「workers 可能有用」的组合
    combos: List[Tuple[Candidate, int, str, float, float]] = []
    for base in cands:
        for provider, model in model_specs:
            cand = apply_model(base, provider, model, providers)
            width = max_wave_width(cand.config)
            price = ((providers.get(provider) or {}).get("models", {}).get(model) or {})
            p_in = float(price.get("price_in", 0) or 0)
            p_out = float(price.get("price_out", 0) or 0)
            for w in workers_grid:
                if w > 1 and width <= 1:
                    skipped.append(f"{cand.name}@w{w}")
                    continue
                combos.append((cand, w, model, p_in, p_out))

    # 阶段 1：每个组合跑 repeats 次，先拿中位数
    for cand, w, model, p_in, p_out in combos:
        cfg_path = write_candidate(cand, outdir / "candidates")
        db_path = outdir / f"db_{cand.name}_w{w}.sqlite3"
        n = min(repeats, max_runs - used)
        if n <= 0:
            break
        trials.append(evaluate(cand, cfg_path, db_path, w, judge, runs=n,
                               model=model, price_in=p_in, price_out=p_out))
        used += n

    # 阶段 2：按效率取 top-2，补跑到 target 次（预算内）
    ranked = sorted(trials, key=lambda t: -t.efficiency)
    for t in ranked[:2]:
        need = min(target - t.runs, max_runs - used)
        if need <= 0:
            continue
        extra = evaluate(t.candidate, t.cfg_path, t.db_path, t.workers, judge, runs=need,
                         model=t.model, price_in=t.price_in, price_out=t.price_out)
        t.latencies += extra.latencies
        t.qualities += extra.qualities
        t.outputs += extra.outputs
        t.notes += extra.notes
        t.judge_fails += extra.judge_fails
        t.tokens += extra.tokens
        t.costs += extra.costs
        used += need

    if skipped:
        print(f"跳过无意义组合（单节点，workers 不影响时延）：{', '.join(skipped)}", flush=True)
    return trials


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="搜索 v8 调度顺序/形式")
    ap.add_argument("--forms", default=",".join(FORMS),
                    help=f"要比较的调度形式，逗号分隔（可选：{','.join(FORMS)}）")
    ap.add_argument("--workers", default="1,3", help="并发数网格，逗号分隔")
    ap.add_argument("--max-runs", type=int, default=14, help="总执行预算（次）")
    ap.add_argument("--repeats", type=int, default=2, help="每个组合首轮重复次数（抗单次噪声）")
    ap.add_argument("--target", type=int, default=3, help="top 组合补跑到的目标次数")
    ap.add_argument("--judge-sleep", type=float, default=3.0,
                    help="两次评审的最小间隔秒数（网关 RPM 很紧，别调太小）")
    ap.add_argument("--models", default=GEN_MODEL,
                    help="模型网格，逗号分隔（provider 固定 laptop，如 deepseek-v4-flash,glm-5.2）")
    ap.add_argument("--out", default=None, help="输出目录（默认 v8/data/search/<时间戳>）")
    ap.add_argument("--dry-run", action="store_true", help="只生成并自检配置，不调模型")
    ap.add_argument("--no-warmup", action="store_true", help="跳过 pi 预热")
    args = ap.parse_args(argv)

    forms = [f.strip() for f in args.forms.split(",") if f.strip()]
    unknown = [f for f in forms if f not in FORMS]
    if unknown:
        ap.error(f"未知形式：{unknown}；可选 {list(FORMS)}")
    workers_grid = [int(w) for w in args.workers.split(",") if w.strip()]

    outdir = Path(args.out) if args.out else (
        V8 / "data" / "search" / time.strftime("%Y%m%d-%H%M%S"))
    outdir.mkdir(parents=True, exist_ok=True)
    cands = [FORMS[f]() for f in forms]

    if args.dry_run:
        for cand in cands:
            cfg_path = write_candidate(cand, outdir / "candidates")
            res = dag8.run(str(cfg_path), db_path=str(outdir / f"dry_{cand.name}.sqlite3"),
                           dry_run=True)
            print(f"✓ {cand.name:10s} 配置自检 {res.status}  "
                  f"edges={len(res.edges)}  files={list(cand.files)}")
        print(f"\n候选已写入 {outdir / 'candidates'}")
        return 0

    if not args.no_warmup:
        pi_cfg = dag8.PiConfig(provider="laptop", model=GEN_MODEL, timeout=120, retries=2)
        t0 = time.perf_counter()
        try:
            PiAgent(pi_cfg).run("只回复两个字：就绪")
            print(f"pi 预热完成（{time.perf_counter() - t0:.1f}s）", flush=True)
        except PiError as exc:
            # 预热只是把冷启动成本挪走，失败不该让整场搜索崩掉
            print(f"⚠ pi 预热失败（继续搜索）：{exc}", flush=True)

    providers = load_providers()
    model_specs = [("laptop", m.strip()) for m in args.models.split(",") if m.strip()]
    judge = make_judge(gap=args.judge_sleep)
    t0 = time.perf_counter()
    trials = search(cands, workers_grid, args.max_runs, judge, outdir,
                    repeats=args.repeats, target=args.target,
                    models=model_specs, providers=providers)
    elapsed = time.perf_counter() - t0

    front = pareto_front(trials)
    best_eff = max(trials, key=lambda t: t.efficiency)
    # 质量优先：同分时取更快的
    best_q = max(trials, key=lambda t: (round(t.quality, 2), -t.latency))
    report = [
        "# v8 调度搜索结果",
        "",
        f"- 任务：{TASK}",
        f"- 模型：{[m for _, m in model_specs]}；价格：见 configs/providers.json（免 key 网关为 0）",
        f"- 形式：{forms}；workers：{workers_grid}；预算：{args.max_runs} 次；实际：{sum(t.runs for t in trials)} 次；"
        f"每组合首轮 {args.repeats} 次，top 补到 {args.target} 次",
        f"- 搜索耗时：{elapsed:.1f}s",
        "",
        "## 排名（按效率）",
        "",
        fmt_table(trials),
        "",
        "## Pareto 前沿（质量高、时延低）",
        "",
        "| 调度形式 | workers | 质量 | 时延 p50 |",
        "|---|---:|---:|---:|",
    ]
    for t in front:
        report.append(f"| {t.candidate.name} | {t.workers} | {t.quality:.2f} | {t.latency:.2f}s |")
    report += [
        "",
        f"**最优（效率）**：`{best_eff.candidate.name}` @ workers={best_eff.workers}，"
        f"质量 {best_eff.quality:.2f}，时延 {best_eff.latency:.2f}s",
        "",
        f"**最优（质量）**：`{best_q.candidate.name}` @ workers={best_q.workers}，"
        f"质量 {best_q.quality:.2f}，时延 {best_q.latency:.2f}s",
        "",
        "## 最佳配置",
        "",
        f"```\n{json.dumps(best_eff.candidate.config, ensure_ascii=False, indent=2)}\n```",
    ]
    (outdir / "report.md").write_text("\n".join(report) + "\n", encoding="utf-8")

    best_dir = outdir / "best"
    write_candidate(best_eff.candidate, best_dir)
    best_cfg = best_dir / best_eff.candidate.name / "dag.json"
    print("\n".join(report[:40]))
    print(f"\n报告：{outdir / 'report.md'}")
    print(f"最佳配置：{best_cfg}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
