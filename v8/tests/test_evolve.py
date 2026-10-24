"""遗传算法 + 基因账本的离线测试：变异/解码/交互分桶。"""
from __future__ import annotations

import sys
from pathlib import Path

V8 = Path(__file__).resolve().parent.parent
for p in (str(V8), str(V8 / "scripts")):
    if p not in sys.path:
        sys.path.insert(0, p)

import evolve_schedule as ev  # noqa: E402
import search_schedule as ss  # noqa: E402


def _ev(topology: str, prompt: str, workers: int, fitness: float,
        model: str = "deepseek-flash") -> dict:
    return {
        "genome": {"topology": topology, "model": model,
                   "prompt": prompt, "workers": workers},
        "fitness": fitness,
        "quality": fitness,
        "status": "ok",
        "failed": False,
    }


def test_mutation_stays_in_alleles_and_changes():
    import random
    rng = random.Random(1)
    g = ev.Genome(topology="single", model="deepseek-flash", prompt="terse", workers=1)
    child = ev.mutate(g, rng)
    assert child != g
    assert child.topology in ev.GENES["topology"]
    assert child.model in ev.GENES["model"]
    assert child.prompt in ev.GENES["prompt"]
    assert child.workers in ev.GENES["workers"]


def test_decode_strict_touches_pi_and_v7_prompts():
    providers = ss.load_providers()

    pi_cand = ev.decode(ev.Genome("single", "deepseek-flash", "strict", 1), providers)
    assert ev.STRICT_SUFFIX.strip() in pi_cand.config["edges"][0]["prompt"]

    graph_cand = ev.decode(ev.Genome("v7_only", "deepseek-flash", "strict", 1), providers)
    prompts = [e["processor"]["prompt"] for e in graph_cand.files["v7.json"]["edges"]]
    assert prompts and all(ev.STRICT_SUFFIX.strip() in p for p in prompts)


def test_ledger_finds_combined_only():
    """单独都没效果，组合 +5 → 结合才生效。"""
    evals = (
        [_ev("fanout", "terse", 1, 10.0) for _ in range(2)]
        + [_ev("fanout", "terse", 3, 10.0) for _ in range(2)]
        + [_ev("fanout", "strict", 1, 10.0) for _ in range(2)]
        + [_ev("fanout", "strict", 3, 15.0) for _ in range(2)]
    )
    led = ev.build_ledger(evals)
    pairs = {tuple(p["pair"]) for p in led["buckets"]["combined_only"]}
    assert ("prompt=strict", "workers=3") in pairs or ("workers=3", "prompt=strict") in pairs


def test_ledger_finds_combined_harm():
    """各自单独 +3，组合反而 -4 → 结合会失效。"""
    evals = (
        [_ev("fanout", "terse", 1, 10.0) for _ in range(2)]
        + [_ev("fanout", "terse", 3, 13.0) for _ in range(2)]
        + [_ev("fanout", "strict", 1, 13.0) for _ in range(2)]
        + [_ev("fanout", "strict", 3, 9.0) for _ in range(2)]
    )
    led = ev.build_ledger(evals)
    pairs = {tuple(p["pair"]) for p in led["buckets"]["combined_harm"]}
    assert ("prompt=strict", "workers=3") in pairs or ("workers=3", "prompt=strict") in pairs


def test_ledger_buckets_effective_and_neutral():
    evals = (
        [_ev("single", "terse", 1, 15.0) for _ in range(3)]
        + [_ev("fanout", "terse", 1, 9.0) for _ in range(3)]
    )
    led = ev.build_ledger(evals)
    effective = {x["feature"] for x in led["buckets"]["effective"]}
    assert "topology=single" in effective
    assert "topology=fanout" in {x["feature"] for x in led["buckets"]["harmful"]}


def test_ledger_excludes_structurally_inert_workers():
    """单节点拓扑上的 workers=3 是结构性无效，不能算协同/生效。"""
    evals = (
        [_ev("fanout", "terse", 1, 10.0) for _ in range(2)]
        + [_ev("single", "terse", 3, 15.0) for _ in range(2)]   # 噪声样本
        + [_ev("fanout", "terse", 1, 10.0) for _ in range(2)]
        + [_ev("fanout", "terse", 3, 10.0) for _ in range(2)]
    )
    led = ev.build_ledger(evals)

    pairs = {tuple(p["pair"]) for p in led["buckets"]["combined_only"]}
    assert ("topology=single", "workers=3") not in pairs
    assert ("workers=3", "topology=single") not in pairs

    # workers 统计只来自 fanout：2 个 w1、2 个 w3
    assert led["gene_stats"]["workers"]["3"]["n"] == 2


def test_failed_evaluations_excluded_from_ledger():
    failed = {**_ev("single", "terse", 1, 0.0), "quality": 0.0,
              "status": "error", "failed": True}
    evals = [_ev("single", "terse", 1, 15.0)] * 2 + [failed]
    led = ev.build_ledger(evals)
    assert led["n_failed"] == 1
    assert led["n_valid"] == 2
    assert led["gene_stats"]["topology"]["single"]["n"] == 2


def test_adjusted_effects_available():
    combos = [
        ("single", "deepseek-flash", "terse", 1, 15.0),
        ("single", "deepseek-flash", "strict", 1, 14.0),
        ("single", "deepseek-v4-flash", "terse", 1, 12.0),
        ("chain2", "deepseek-flash", "terse", 1, 11.0),
        ("chain2", "deepseek-flash", "strict", 1, 10.0),
        ("chain2", "deepseek-v4-flash", "terse", 1, 9.0),
        ("fanout", "deepseek-flash", "terse", 1, 8.0),
        ("fanout", "deepseek-v4-flash", "strict", 1, 7.0),
    ]
    evals = [_ev(t, p, w, f, model=m) for t, m, p, w, f in combos]
    led = ev.build_ledger(evals)
    stat = led["gene_stats"]["topology"]["single"]
    assert stat["adj_delta"] is not None
    assert stat["adj_delta"] > 0


def test_contract_gene_sets_expect():
    cand = ev.decode(ev.Genome("single", "deepseek-flash", "terse", 1, "on"),
                     ss.load_providers())
    final = [e for e in cand.config["edges"] if "final" in e["output_tags"]][0]
    assert final["expect"]["min_chars"] == 150
    assert final["repair"] == 1


def test_contract_defaults_off():
    assert ev.Genome("single", "deepseek-flash", "terse", 1).contract == "off"
