"""稳定性测试：同一输入重复跑，量化输出是否稳定。

用法：
  python3 -m dag_defs.stage1st_stability classify 4     # 分类环节跑 4 次
  python3 -m dag_defs.stage1st_stability full 2         # 完整 DAG 跑 2 次，比较结构完整性
"""
from __future__ import annotations

import itertools
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import dags  # noqa: E402,F401  (触发 dag_defs 自动加载)
from dags import registry  # noqa: E402

# 定义模块的 sys.modules 键名由 registry 维护（带目录指纹），不要自己猜：
# 早先写死 sys.modules["dag_defs_stage1st_ai"]，换个加载顺序或从包路径导入就 KeyError。
s1 = registry.get_definition_module("stage1st_ai")


def _jaccard(a, b) -> float:
    sa, sb = set(a), set(b)
    if not sa and not sb:
        return 1.0
    return len(sa & sb) / len(sa | sb)


def test_classify(runs: int = 4) -> None:
    print(f"=== 分类稳定性：{runs} 次 ===")
    threads = s1.scrape_catalog()
    print(f"目录固定 {len(threads)} 帖，反复送入同一 prompt\n")

    recs = []
    for i in range(runs):
        s1.LAST_CLASSIFY.clear()
        picked = s1.classify_ai_threads(threads)
        rec = dict(s1.LAST_CLASSIFY)
        recs.append(rec)
        print(f"run {i + 1}: model={rec.get('model')} 选中={rec.get('count')} "
              f"耗时={rec.get('elapsed')}s raw={rec.get('raw_len')}B")

    print("\n两两 Jaccard 相似度（1.0 = 完全一致）：")
    for (i, a), (j, b) in itertools.combinations(list(enumerate(recs)), 2):
        print(f"  run{i + 1} vs run{j + 1}: {_jaccard(a['tids'], b['tids']):.3f}")

    sets = [set(r["tids"]) for r in recs]
    core = set.intersection(*sets) if sets else set()
    union = set.union(*sets) if sets else set()
    print(f"\n每次都选中（稳定核）：{len(core)} 帖")
    print(f"至少一次选中：{len(union)} 帖")
    print(f"只出现部分次数（抖动）：{len(union - core)} 帖")
    title_of = {}
    for r in recs:
        for tid, title in r.get("pairs", []):
            title_of[tid] = title
    for tid in sorted(union - core):
        hits = sum(1 for st in sets if tid in st)
        print(f"    [{hits}/{runs}] {tid} {title_of.get(tid, '')[:50]}")
    print(f"\n一致选中：")
    for tid in sorted(core):
        print(f"    {tid} {title_of.get(tid, '')[:50]}")


def _structure_report(text: str) -> dict:
    """检查摘要段本身是否完整（不是看报告尾部，报告尾部是索引）。"""
    i = text.find("## 主题索引")
    summary = text[text.find("## 摘要"): i] if i > 0 else text
    ends_ok = bool(re.search(r"[。！？.|]\s*$", summary.rstrip()))
    failed = ("LLM 摘要不可用" in summary) or ("抽取式摘要" in summary)
    return {
        "lines": len(text.splitlines()),
        "summary_chars": len(summary),
        "summary_ends_properly": ends_ok,
        "summary_fallback": failed,
        "has_index_section": "## 主题索引" in text,
        "index_items": len(re.findall(r"^- \[", text, re.M)),
        "sections_count": len(re.findall(r"^## ", text, re.M)),
    }


def test_full(runs: int = 2) -> None:
    print(f"=== 完整 DAG 稳定性：{runs} 次 ===")
    reports = []
    for i in range(runs):
        # dry_run：只跑到取结果，正式产物 dag_defs/out/ 留给 `python -m dags run` 写
        res = registry.run_dag("stage1st-ai-24h", dry_run=True)
        text = str(res["report"])
        snap = ROOT / "dag_defs" / "out" / f"stability_run{i + 1}.md"
        snap.parent.mkdir(parents=True, exist_ok=True)
        snap.write_text(text, encoding="utf-8")
        st = _structure_report(text)
        reports.append(st)
        print(f"run {i + 1}: {json.dumps(st, ensure_ascii=False)}  elapsed={res['elapsed']:.1f}s")

    print("\n结构一致性：")
    keys = reports[0].keys()
    for k in keys:
        vals = [r[k] for r in reports]
        same = all(v == vals[0] for v in vals)
        print(f"  {k:24} {vals}  {'稳定' if same else '不一致'}")


def main() -> None:
    mode = sys.argv[1] if len(sys.argv) > 1 else "classify"
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 4
    if mode == "classify":
        test_classify(n)
    elif mode == "full":
        test_full(n)
    else:
        raise SystemExit("mode: classify | full")


if __name__ == "__main__":
    main()
