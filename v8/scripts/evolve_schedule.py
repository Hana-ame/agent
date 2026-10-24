#!/usr/bin/env python3
"""遗传算法搜索调度：基因 + 适应度 + **交互账本**。

## 基因（genome）

| gene | alleles |
|---|---|
| `topology` | single / chain2 / fanout / v7_only |
| `model` | deepseek-flash / deepseek-v4-flash |
| `prompt` | terse / strict（strict 追加 150~250 字硬约束） |
| `workers` | 1 / 3（只有 fanout 这类并行形式才可能生效） |

## 适应度

    fitness = quality - latency_penalty * latency_seconds

quality 是评审 3~15 分；默认 latency_penalty=0.1（1 分 ≈ 10 秒）。
Pareto（质量、时延）与 tokens 也一并记录。

## 账本（核心）

每次评估都进账本、跨轮累积，然后算：

- **单基因边际效应**：带该基因 vs 不带的平均适应度差
  → `生效`（Δ>阈值）、`失效`（中性）、`负向`
- **两基因交互（2×2 因子）**：
  - `eff_a = E[A,¬B] - E[¬A,¬B]`
  - `eff_b = E[¬A,B] - E[¬A,¬B]`
  - `interaction = (E[A,B]-E[A,¬B]) - (E[¬A,B]-E[¬A,¬B])`
  → `结合才生效`（各自单独 ≤0，组合增益 >阈值）
  → `结合会失效`（各自单独 >0，但 interaction 显著为负）
  → `相互放大`（各自 >0，且 interaction 显著为正）

只有 2×2 四格都有样本才分析；任一格 n<2 标 `低置信`。
账本是假设生成器，不是显著性检验。

## 用法

```bash
cd v8
python scripts/evolve_schedule.py --population 6 --generations 3 --max-evals 18
python scripts/evolve_schedule.py --ledger-only   # 只重算账本，不跑模型
```

产物：
    data/evolution/<时间戳>/report.md     本轮报告
    data/evolution/<时间戳>/best.json     最佳基因
    data/evolution/ledger.json            跨轮累积账本
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import statistics
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

V8 = Path(__file__).resolve().parent.parent
ROOT = V8.parent
for p in (str(V8), str(V8 / "scripts"), str(ROOT)):
    if p not in sys.path:
        sys.path.insert(0, p)

import dag8  # noqa: E402
import search_schedule as ss  # noqa: E402
from dag8.ratelimit import shared_limiter  # noqa: E402

EVOLVE_DIR = V8 / "data" / "evolution"
LEDGER_PATH = EVOLVE_DIR / "ledger.json"

GENES: Dict[str, List[Any]] = {
    "topology": ["single", "chain2", "fanout", "v7_only"],
    "model": ["deepseek-flash", "deepseek-v4-flash"],
    "prompt": ["terse", "strict"],
    "workers": [1, 3],
    "contract": ["off", "on"],
}

STRICT_SUFFIX = "\n\n硬约束：正文必须 150~250 字；不足或超出都算不合格。"
LATENCY_CAP = 120.0   # 超过 120s 的挂钟按 120s 计入适应度（多半是限流重试）

SEED_GENOMES = [
    {"topology": "single", "model": "deepseek-flash", "prompt": "terse",
     "workers": 1, "contract": "off"},
    {"topology": "single", "model": "deepseek-flash", "prompt": "terse",
     "workers": 1, "contract": "on"},
]


# ---------------------------------------------------------------------------
# 基因
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Genome:
    topology: str
    model: str
    prompt: str
    workers: int
    contract: str = "off"

    def as_dict(self) -> Dict[str, Any]:
        return {"topology": self.topology, "model": self.model,
                "prompt": self.prompt, "workers": self.workers,
                "contract": self.contract}

    def id(self) -> str:
        raw = json.dumps(self.as_dict(), sort_keys=True)
        return hashlib.sha256(raw.encode()).hexdigest()[:10]

    @property
    def key(self) -> str:
        return self.id()

    def __str__(self) -> str:
        return (f"{self.topology}/{self.model}/{self.prompt}"
                f"/w{self.workers}/c{self.contract}")


def _as_genome(d: Dict[str, Any]) -> Genome:
    return Genome(topology=d["topology"], model=d["model"],
                  prompt=d["prompt"], workers=int(d["workers"]),
                  contract=d.get("contract", "off"))


def _workers_for(topology: str, rng: random.Random) -> int:
    """workers 只有在同一 layer 有多个节点时才可能生效；串行拓扑固定为 1。"""
    return rng.choice(GENES["workers"]) if topology == "fanout" else 1


def _normalize(g: Genome) -> Genome:
    if g.topology != "fanout":
        return Genome(g.topology, g.model, g.prompt, 1)
    return g


def random_genome(rng: random.Random) -> Genome:
    topology = rng.choice(GENES["topology"])
    return Genome(
        topology=topology,
        model=rng.choice(GENES["model"]),
        prompt=rng.choice(GENES["prompt"]),
        workers=_workers_for(topology, rng),
        contract=rng.choice(GENES["contract"]),
    )


def crossover(a: Genome, b: Genome, rng: random.Random) -> Genome:
    da, db = a.as_dict(), b.as_dict()
    return _normalize(_as_genome(
        {k: (da[k] if rng.random() < 0.5 else db[k]) for k in GENES}))


def mutate(g: Genome, rng: random.Random, rate: float = 0.25) -> Genome:
    d = g.as_dict()
    for gene, alleles in GENES.items():
        if rng.random() < rate:
            d[gene] = rng.choice([a for a in alleles if a != d[gene]])
    child = _normalize(_as_genome(d))
    if child == g:  # 保证变异至少发生一次
        gene = rng.choice(list(GENES))
        d = g.as_dict()
        if gene == "workers" and d["topology"] != "fanout":
            gene = "topology"          # workers 在串行拓扑上无效，改变异别的
        d[gene] = rng.choice([a for a in GENES[gene] if a != d[gene]])
        child = _normalize(_as_genome(d))
    return child


def init_population(size: int, rng: random.Random) -> List[Genome]:
    pop: List[Genome] = [_as_genome(d) for d in SEED_GENOMES]
    while len(pop) < size:
        g = random_genome(rng)
        if g not in pop:
            pop.append(g)
    return pop[:size]


# ---------------------------------------------------------------------------
# 基因 → 可执行配置
# ---------------------------------------------------------------------------
def decode(g: Genome, providers: Dict[str, Any]) -> ss.Candidate:
    cand = ss.FORMS[g.topology]()
    cand = ss.apply_model(cand, "laptop", g.model, providers)
    cand.name = f"{g.topology}-{g.model}-{g.prompt}-w{g.workers}"

    if g.contract == "on":
        for edge in cand.config.get("edges", []):
            if "final" in edge.get("output_tags", []):
                edge["expect"] = {"min_chars": 150, "max_chars": 400}
                edge["repair"] = 1

    if g.prompt == "strict":
        for edge in cand.config.get("edges", []):
            if edge.get("kind") != "graph" and "prompt" in edge:
                edge["prompt"] += STRICT_SUFFIX
        for graph in cand.files.values():
            for edge in graph.get("edges", []):
                proc = edge.get("processor") or {}
                if "prompt" in proc:
                    proc["prompt"] += STRICT_SUFFIX
    return cand


def evaluate(g: Genome, outdir: Path, judge, providers: Dict[str, Any],
             latency_penalty: float, rpm: Optional[int] = None) -> Dict[str, Any]:
    cand = decode(g, providers)
    cfg_path = ss.write_candidate(cand, outdir / "candidates")
    db_path = outdir / f"db_{cand.name}.sqlite3"

    attempts = 0
    while True:
        attempts += 1
        t0 = time.perf_counter()
        res = dag8.run(str(cfg_path), db_path=str(db_path),
                       max_workers=g.workers, rpm=rpm)
        latency = time.perf_counter() - t0
        if res.status == "ok" and res.result:
            break
        if attempts >= 2:   # 失败重试一次，别把一次 429 当成基因效应
            break

    tin = sum(e.prompt_chars for e in res.edges) / 2.0
    tout = sum(e.output_chars for e in res.edges) / 2.0
    tokens = tin + tout

    quality = 0.0
    note = f"run status={res.status}"
    if res.status != "ok":
        errs = [x.error for x in res.edges if x.error]
        if errs:
            note = f"run status={res.status}: {errs[0][:120]}"
    if res.status == "ok" and res.result:
        q, note = judge(res.result)
        if q is not None:
            quality = q

    capped = min(latency, LATENCY_CAP)
    failed = res.status != "ok" or quality <= 0
    return {
        "genome": g.as_dict(),
        "genome_id": g.id(),
        "quality": quality,
        "latency": latency,
        "latency_capped": capped,
        "tokens": tokens,
        "fitness": quality - latency_penalty * capped,
        "note": note,
        "status": res.status,
        "attempts": attempts,
        "failed": failed,
        "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }


def tournament(pop: List[Genome], results: Dict[str, Dict[str, Any]],
               rng: random.Random, k: int = 3) -> Genome:
    picks = rng.sample(pop, min(k, len(pop)))
    return max(picks, key=lambda g: results[g.id()]["fitness"])


def evolve(pop_size: int, generations: int, max_evals: int, seed: int,
           outdir: Path, judge, providers: Dict[str, Any],
           latency_penalty: float,
           rpm: Optional[int] = None) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    rng = random.Random(seed)
    pop = init_population(pop_size, rng)
    results: Dict[str, Dict[str, Any]] = {}
    history: List[Dict[str, Any]] = []
    used = 0

    for gen in range(generations):
        for g in pop:
            if g.id() in results:
                continue
            if used >= max_evals:
                break
            results[g.id()] = evaluate(g, outdir, judge, providers, latency_penalty, rpm)
            used += 1
            r = results[g.id()]
            print(f"  [gen {gen}] {g}  quality={r['quality']:.1f} "
                  f"lat={r['latency']:.1f}s", flush=True)

        if not results:
            break
        ranked = sorted((results[g.id()] for g in pop if g.id() in results),
                        key=lambda r: r["fitness"], reverse=True)
        history.append({
            "gen": gen,
            "best": ranked[0]["genome"],
            "best_fitness": ranked[0]["fitness"],
            "mean_fitness": statistics.mean(r["fitness"] for r in ranked),
            "n": len(ranked),
        })
        if used >= max_evals or gen == generations - 1:
            break

        # 精英保留 + 锦标赛选择 + 交叉/变异
        elites = [_as_genome(r["genome"]) for r in ranked[:2]]
        newpop = list(elites)
        evaluated = [g for g in pop if g.id() in results]
        while len(newpop) < pop_size:
            child = mutate(crossover(tournament(evaluated, results, rng),
                                     tournament(evaluated, results, rng), rng), rng)
            if child not in newpop:
                newpop.append(child)
        pop = newpop

    return list(results.values()), history


# ---------------------------------------------------------------------------
# 账本
# ---------------------------------------------------------------------------
def _mean(xs: List[float]) -> float:
    return statistics.mean(xs) if xs else 0.0


def _adjusted_effects(evaluations: List[Dict[str, Any]]) -> Dict[str, Dict[str, float]]:
    """OLS 估计每个等位在控制其他基因后的边际效应（剥离混淆）。

    workers 不参与全局回归：它只在 fanout 上结构性有效，直接进全局模型会与拓扑共线。
    """
    try:
        import numpy as np
    except ImportError:  # pragma: no cover - numpy 缺失时退回原始边际
        return {}

    genes = ["topology", "model", "prompt", "contract"]
    cols = [(g, a) for g in genes for a in GENES[g][1:]]
    if not cols or len(evaluations) < len(cols) + 2:
        return {}

    def row(e):
        r = [1.0]
        for g, a in cols:
            r.append(1.0 if str(e["genome"][g]) == str(a) else 0.0)
        return r

    X = np.array([row(e) for e in evaluations], dtype=float)
    y = np.array([e["fitness"] for e in evaluations], dtype=float)
    try:
        beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    except np.linalg.LinAlgError:
        return {}

    overall = float((X @ beta).mean())
    out: Dict[str, Dict[str, float]] = {}
    for g in genes:
        out[g] = {}
        idxs = [i for i, (g2, _) in enumerate(cols, start=1) if g2 == g]
        for a in GENES[g]:
            Xa = X.copy()
            for i in idxs:
                Xa[:, i] = 0.0
            for i, (g2, a2) in enumerate(cols, start=1):
                if g2 == g and str(a2) == str(a):
                    Xa[:, i] = 1.0
            out[g][str(a)] = round(float((Xa @ beta).mean() - overall), 4)
    return out


def _fitness(e: Dict[str, Any]) -> float:
    """统一适应度口径；旧账本没有 latency_capped 的按 120s 截断回算。"""
    if "fitness" in e and "latency_capped" in e:
        return float(e["fitness"])
    return float(e.get("quality", 0.0)) - 0.1 * min(float(e.get("latency", 0.0)), LATENCY_CAP)


def _fit_residuals(evals: List[Dict[str, Any]], genes: List[str]) -> Optional[List[float]]:
    """拟合主效应并返回残差；样本不足或 numpy 缺失返回 None。

    交互分析用残差而不是原始 fitness：否则「single + flash + terse」会和
    chain2/fanout 混在同一个 2×2 格子里，把组合优势算没。
    """
    try:
        import numpy as np
    except ImportError:  # pragma: no cover
        return None

    cols = [(g, a) for g in genes for a in GENES[g][1:]]
    if not cols or len(evals) < len(cols) + 2:
        return None

    def row(e):
        r = [1.0]
        for g, a in cols:
            r.append(1.0 if str(e["genome"][g]) == str(a) else 0.0)
        return r

    X = np.array([row(e) for e in evals], dtype=float)
    y = np.array([e["fitness"] for e in evals], dtype=float)
    try:
        beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    except np.linalg.LinAlgError:
        return None
    return [float(v) for v in (y - X @ beta)]


def build_ledger(evaluations: List[Dict[str, Any]], min_effect: float = 0.5) -> Dict[str, Any]:
    if not evaluations:
        return {"updated_at": time.strftime("%Y-%m-%dT%H:%M:%S"), "n_evaluations": 0,
                "n_valid": 0, "n_failed": 0,
                "overall_mean_fitness": 0.0, "gene_stats": {}, "pairs": [],
                "buckets": {"effective": [], "neutral": [], "harmful": [],
                            "combined_only": [], "combined_harm": [], "amplified": []}}

    evaluations = [
        {**e, "fitness": _fitness(e),
         "genome": {g: e["genome"].get(g, GENES[g][0]) for g in GENES}}
        for e in evaluations
    ]

    # 失败样本（重试后仍 error / 评审解析失败）不参与基因效应统计
    valid = [e for e in evaluations if not e.get("failed") and e.get("quality", 0) > 0]
    n_failed = len(evaluations) - len(valid)
    if not valid:
        return {"updated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
                "n_evaluations": len(evaluations), "n_valid": 0, "n_failed": n_failed,
                "overall_mean_fitness": 0.0, "gene_stats": {}, "pairs": [],
                "buckets": {"effective": [], "neutral": [], "harmful": [],
                            "combined_only": [], "combined_harm": [], "amplified": []}}

    adjusted = _adjusted_effects(valid)
    overall = _mean([e["fitness"] for e in valid])

    main_resid = _fit_residuals(valid, ["topology", "model", "prompt", "contract"])
    fanout_idx = [i for i, e in enumerate(valid) if e["genome"]["topology"] == "fanout"]
    fanout_resid = _fit_residuals([valid[i] for i in fanout_idx], ["model", "prompt"])
    fanout_resid_map = dict(zip(fanout_idx, fanout_resid)) if fanout_resid else {}

    # 单基因边际。workers 只在 fanout 上结构性有效，其他拓扑的样本一律排除，
    # 否则「单节点 + 并发 3」会被噪声算成生效/协同（第一轮账本真踩过）。
    gene_stats: Dict[str, Dict[str, Any]] = {}
    for gene, alleles in GENES.items():
        gene_stats[gene] = {}
        pool = ([e for e in valid if e["genome"]["topology"] == "fanout"]
                if gene == "workers" else valid)
        if not pool:
            continue
        for allele in alleles:
            with_ = [e["fitness"] for e in pool
                     if str(e["genome"][gene]) == str(allele)]
            without = [e["fitness"] for e in pool
                       if str(e["genome"][gene]) != str(allele)]
            if not with_:
                continue
            gene_stats[gene][str(allele)] = {
                "n": len(with_),
                "mean": _mean(with_),
                "delta": _mean(with_) - (overall if not without else _mean(without)),
                "adj_delta": adjusted.get(gene, {}).get(str(allele)),
                "low_conf": len(with_) < 3,
            }

    # 两基因交互：2×2 因子
    features: List[Tuple[str, str]] = []
    for gene, alleles in GENES.items():
        pool = ([e for e in valid if e["genome"]["topology"] == "fanout"]
                if gene == "workers" else valid)
        for allele in alleles:
            if any(str(e["genome"][gene]) == str(allele) for e in pool):
                features.append((gene, str(allele)))

    pairs: List[Dict[str, Any]] = []
    for i in range(len(features)):
        for j in range(i + 1, len(features)):
            g1, a1 = features[i]
            g2, a2 = features[j]
            if g1 == g2:
                continue

            workers_pair = "workers" in (g1, g2)
            subset_idx = fanout_idx if workers_pair else list(range(len(valid)))

            def cell(p1: bool, p2: bool, _g1=g1, _a1=a1, _g2=g2, _a2=a2,
                     _idx=subset_idx) -> List[int]:
                return [
                    i for i in _idx
                    if (str(valid[i]["genome"][_g1]) == _a1) == p1
                    and (str(valid[i]["genome"][_g2]) == _a2) == p2
                ]

            c11, c10, c01, c00 = (cell(True, True), cell(True, False),
                                  cell(False, True), cell(False, False))
            ns = {"11": len(c11), "10": len(c10), "01": len(c01), "00": len(c00)}
            if min(ns.values()) < 1 or ns["11"] < 2:
                continue

            def raw_mean(idx):
                return _mean([valid[i]["fitness"] for i in idx])

            def resid_mean(idx):
                if workers_pair:
                    vals = [fanout_resid_map.get(i, 0.0) for i in idx]
                elif main_resid:
                    vals = [main_resid[i] for i in idx]
                else:
                    vals = [valid[i]["fitness"] for i in idx]
                return _mean(vals)

            m11, m10, m01, m00 = (raw_mean(c11), raw_mean(c10),
                                  raw_mean(c01), raw_mean(c00))
            r11, r10, r01, r00 = (resid_mean(c11), resid_mean(c10),
                                  resid_mean(c01), resid_mean(c00))
            eff_a, eff_b = m10 - m00, m01 - m00
            interaction_raw = (m11 - m10) - (m01 - m00)
            interaction = (r11 - r10) - (r01 - r00)   # 残差交互 = 校正其他基因后的交互

            pairs.append({
                "pair": [f"{g1}={a1}", f"{g2}={a2}"],
                "n": ns,
                "mean": {"11": m11, "10": m10, "01": m01, "00": m00},
                "eff_a": eff_a,
                "eff_b": eff_b,
                "interaction": interaction,
                "interaction_raw": interaction_raw,
                "low_conf": min(ns.values()) < 2,
                "adjusted": bool(main_resid) if not workers_pair else bool(fanout_resid_map),
            })

    # 单基因分桶
    effective, neutral, harmful = [], [], []
    for gene, stats in gene_stats.items():
        for allele, s in stats.items():
            item = {"feature": f"{gene}={allele}", **s}
            effect = s["adj_delta"] if s.get("adj_delta") is not None else s["delta"]
            if effect > min_effect:
                effective.append(item)
            elif effect < -min_effect:
                harmful.append(item)
            else:
                neutral.append(item)

    # 交互分桶
    combined_only, combined_harm, redundant, amplified = [], [], [], []
    for p in pairs:
        combo_gain = p["mean"]["11"] - p["mean"]["00"]
        if p["interaction"] < -min_effect and p["eff_a"] > min_effect and p["eff_b"] > min_effect:
            # 组合绝对低于任一单独 → 真·结合失效；否则只是次可加（冗余）
            worse_than_solo = p["mean"]["11"] < min(p["mean"]["10"], p["mean"]["01"])
            (combined_harm if worse_than_solo else redundant).append(p)
        elif combo_gain > min_effect and p["eff_a"] <= min_effect and p["eff_b"] <= min_effect:
            combined_only.append(p)      # 单独都不行，合起来才涨
        elif p["interaction"] > min_effect and p["eff_a"] > min_effect and p["eff_b"] > min_effect:
            amplified.append(p)          # 各自有效，合起来更强

    by_gene_delta = lambda x: (x.get("delta", 0), x.get("n", 0))
    by_pair_int = lambda x: (x.get("interaction", 0), x.get("n", {}).get("11", 0))

    return {
        "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "n_evaluations": len(evaluations),
        "n_valid": len(valid),
        "n_failed": n_failed,
        "overall_mean_fitness": overall,
        "gene_stats": gene_stats,
        "pairs": sorted(pairs, key=by_pair_int, reverse=True),
        "buckets": {
            "effective": sorted(effective, key=by_gene_delta, reverse=True),
            "neutral": sorted(neutral, key=by_gene_delta, reverse=True),
            "harmful": sorted(harmful, key=by_gene_delta),
            "combined_only": sorted(combined_only, key=by_pair_int, reverse=True),
            "combined_harm": sorted(combined_harm, key=by_pair_int),
            "redundant": sorted(redundant, key=by_pair_int),
            "amplified": sorted(amplified, key=by_pair_int, reverse=True),
        },
    }


def load_ledger() -> Dict[str, Any]:
    try:
        return json.loads(LEDGER_PATH.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {"evaluations": []}


def save_ledger(ledger: Dict[str, Any]) -> None:
    LEDGER_PATH.parent.mkdir(parents=True, exist_ok=True)
    LEDGER_PATH.write_text(json.dumps(ledger, ensure_ascii=False, indent=2) + "\n",
                           encoding="utf-8")


# ---------------------------------------------------------------------------
# 报告
# ---------------------------------------------------------------------------
def _fmt_items(items: List[Dict[str, Any]], kind: str) -> str:
    if not items:
        return "_无（样本不足或未观察到）_"
    lines = []
    for it in items[:12]:
        if kind == "gene":
            conf = " 低置信" if it.get("low_conf") else ""
            adj = it.get("adj_delta")
            adj_txt = f" Δadj={adj:+.2f}" if adj is not None else ""
            lines.append(f"- `{it['feature']}` Δraw={it['delta']:+.2f}{adj_txt}（n={it['n']}）{conf}")
        else:
            conf = " 低置信" if it.get("low_conf") else ""
            n = it["n"]
            lines.append(f"- `{it['pair'][0]}` + `{it['pair'][1]}` "
                         f"interaction(校正)={it['interaction']:+.2f}"
                         f"（raw {it['interaction_raw']:+.2f}；"
                         f"单独 {it['eff_a']:+.2f} / {it['eff_b']:+.2f}；"
                         f"n={n['11']}/{n['10']}/{n['01']}/{n['00']}）{conf}")
    return "\n".join(lines)


def fmt_report(history: List[Dict[str, Any]], evaluations: List[Dict[str, Any]],
               ledger: Dict[str, Any], best: Dict[str, Any],
               elapsed: float, n_total: int) -> str:
    b = ledger["buckets"]
    lines = [
        "# 遗传算法调度搜索",
        "",
        f"- 本轮评估：{len(evaluations)} 次；账本累计：{n_total} 次"
        f"（有效 {ledger.get('n_valid', n_total)} / 失败 {ledger.get('n_failed', 0)}）；耗时 {elapsed:.0f}s",
        f"- 适应度 = quality - latency_penalty × latency；overall mean = "
        f"{ledger['overall_mean_fitness']:.2f}",
        "",
        "## 代际",
        "",
        "| 代 | 最佳基因 | fitness | 群体均值 | n |",
        "|---:|---|---:|---:|---:|",
    ]
    for h in history:
        g = h["best"]
        name = (f"{g['topology']}/{g['model']}/{g['prompt']}"
                f"/w{g['workers']}/c{g.get('contract', 'off')}")
        lines.append(f"| {h['gen']} | {name} | {h['best_fitness']:.2f} | "
                     f"{h['mean_fitness']:.2f} | {h['n']} |")

    if best:
        bg = best["genome"]
        lines += [
            "",
            "## 最佳基因",
            "",
            f"`{bg['topology']}/{bg['model']}/{bg['prompt']}"
            f"/w{bg['workers']}/c{bg.get('contract', 'off')}` — "
            f"quality {best['quality']:.2f}，时延 {best['latency']:.2f}s，"
            f"tokens≈{best['tokens']:.0f}，fitness {best['fitness']:.2f}",
        ]

    lines += [
        "",
        "## 账本（累积后）",
        "",
        "### 生效（单独就涨）",
        _fmt_items(b["effective"], "gene"),
        "",
        "### 失效（单独接近 0）",
        _fmt_items(b["neutral"], "gene"),
        "",
        "### 负向（单独就掉）",
        _fmt_items(b["harmful"], "gene"),
        "",
        "### 结合才生效（协同）",
        _fmt_items(b["combined_only"], "pair"),
        "",
        "### 结合会失效（拮抗）",
        _fmt_items(b["combined_harm"], "pair"),
        "",
        "### 次可加（组合仍可用，但增益低于相加）",
        _fmt_items(b["redundant"], "pair"),
        "",
        "### 相互放大（两个都生效，组合更强）",
        _fmt_items(b["amplified"], "pair"),
        "",
        "## 注意",
        "",
        "- workers 只在 fanout 上参与统计（串行拓扑的并发是结构性无效）",
        "- 账本是假设生成器，不是显著性检验；n<3 已标「低置信」",
        "- 单次评估受模型/网关噪声影响，结论要跨轮复现",
        "- 删除 `data/evolution/ledger.json` 可重置账本",
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="遗传算法搜索调度 + 基因交互账本")
    ap.add_argument("--population", type=int, default=6)
    ap.add_argument("--generations", type=int, default=3)
    ap.add_argument("--max-evals", type=int, default=18)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--latency-penalty", type=float, default=0.1)
    ap.add_argument("--rpm", type=int, default=None,
                    help="调度器 RPM 限速（躲 429；默认不限）")
    ap.add_argument("--min-effect", type=float, default=0.5)
    ap.add_argument("--judge-sleep", type=float, default=3.0)
    ap.add_argument("--out", default=None)
    ap.add_argument("--ledger-only", action="store_true",
                    help="不跑模型，只从已有账本重算分桶并写报告")
    args = ap.parse_args(argv)

    outdir = Path(args.out) if args.out else (EVOLVE_DIR / time.strftime("%Y%m%d-%H%M%S"))
    outdir.mkdir(parents=True, exist_ok=True)
    providers = ss.load_providers()

    if args.ledger_only:
        ledger = load_ledger()
        evals = ledger.get("evaluations", [])
        if not evals:
            print("账本为空，先跑一轮")
            return 1
        rebuilt = {**ledger, **build_ledger(evals, args.min_effect)}
        save_ledger(rebuilt)
        (outdir / "report.md").write_text(
            fmt_report([], evals, rebuilt, max(evals, key=lambda e: e["fitness"]),
                       0.0, len(evals)) + "\n", encoding="utf-8")
        print(f"账本已重算：{LEDGER_PATH}")
        return 0

    judge = ss.make_judge(gap=args.judge_sleep,
                          limiter=shared_limiter(args.rpm) if args.rpm else None)
    t0 = time.perf_counter()
    evaluations, history = evolve(args.population, args.generations, args.max_evals,
                                  args.seed, outdir, judge, providers,
                                  args.latency_penalty, args.rpm)
    elapsed = time.perf_counter() - t0

    ledger = load_ledger()
    ledger.setdefault("evaluations", []).extend(evaluations)
    ledger = {**ledger, **build_ledger(ledger["evaluations"], args.min_effect)}
    save_ledger(ledger)

    best = max(evaluations, key=lambda e: e["fitness"]) if evaluations else {}
    (outdir / "best.json").write_text(json.dumps(best, ensure_ascii=False, indent=2) + "\n",
                                      encoding="utf-8")
    report = fmt_report(history, evaluations, ledger, best, elapsed,
                        len(ledger["evaluations"]))
    (outdir / "report.md").write_text(report + "\n", encoding="utf-8")

    print("\n" + report)
    print(f"\n账本：{LEDGER_PATH}")
    print(f"本轮报告：{outdir / 'report.md'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
