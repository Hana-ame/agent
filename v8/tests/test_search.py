"""调度搜索脚本的离线测试：候选生成、并发宽度判定、Pareto/报告渲染。

不碰网络：候选只用 `dry_run=True` 自检，评审与真实 pi 调用不在单测范围内。
"""
from __future__ import annotations

import sys
from pathlib import Path

V8 = Path(__file__).resolve().parent.parent
for p in (str(V8), str(V8 / "scripts")):
    if p not in sys.path:
        sys.path.insert(0, p)

import dag8  # noqa: E402
import search_schedule as ss  # noqa: E402


def test_all_forms_generate_runnable_configs(tmp_path):
    for name, factory in ss.FORMS.items():
        cand = factory()
        cfg_path = ss.write_candidate(cand, tmp_path)
        res = dag8.run(str(cfg_path), db_path=tmp_path / f"{name}.sqlite3", dry_run=True)
        assert res.status == "ok", (name, res.summary())


def test_max_wave_width_flags_parallel_candidates():
    assert ss.max_wave_width(ss.cand_single().config) == 1
    assert ss.max_wave_width(ss.cand_chain3().config) == 1
    assert ss.max_wave_width(ss.cand_fanout().config) == 2


def test_pareto_front_and_table():
    a = ss.Trial(candidate=ss.cand_single(), cfg_path=Path("a"), db_path=Path("d"),
                 workers=1, latencies=[10.0], qualities=[12.0])
    b = ss.Trial(candidate=ss.cand_chain2(), cfg_path=Path("b"), db_path=Path("d"),
                 workers=1, latencies=[20.0], qualities=[12.0])   # 被 a 支配
    c = ss.Trial(candidate=ss.cand_chain3(), cfg_path=Path("c"), db_path=Path("d"),
                 workers=1, latencies=[30.0], qualities=[9.0])    # 被 a 支配

    front = ss.pareto_front([a, b, c])
    assert [t.candidate.name for t in front] == ["single"]

    table = ss.fmt_table([a, b, c])
    assert "调度形式" in table and "single" in table
    assert a.efficiency == 1.2


def test_apply_model_overrides_pi_and_v7_models():
    providers = ss.load_providers()
    c = ss.apply_model(ss.cand_hybrid(), "laptop", "glm-5.2", providers)
    assert c.config["pi"]["model"] == "glm-5.2"
    assert c.name.endswith("@glm-5.2")
    for graph in c.files.values():
        for m in graph["models"].values():
            assert m["model"] == "glm-5.2"
            assert m["api"] == providers["laptop"]["base_url"]
