"""dags 库本身的单元测试：注册表 / 自动发现 / 单次运行 / CLI。

全部离线，不碰 models.json 里的真端点。
"""
from __future__ import annotations

import warnings
from pathlib import Path

import pytest

import dags  # noqa: F401  (触发 dag_defs 自动加载)
from dag_agent import DAG, MockLLMClient
from dags.__main__ import main as cli_main
from dags.registry import (
    DAG_SPECS,
    DagSpec,
    get_definition_module,
    list_dags,
    load_definitions,
    register,
    run_dag,
    timed,
)


# ---------------------------------------------------------------------------
# 辅助
# ---------------------------------------------------------------------------
def _fake_dag() -> DAG:
    d = DAG(name="fake", models={}, llm_client=MockLLMClient())
    d.add_edge(input_tags=["seed"], processor=lambda x: "ok", output_tags=["final"], layer=0)
    return d


def _spec(name: str = "tmp-fake") -> DagSpec:
    return DagSpec(
        name=name,
        description="测试用假 DAG",
        build=_fake_dag,
        seed_tag="seed",
        seed_content="hello",
        final_tag="final",
        default_output="out/tmp_fake.md",
    )


def _def_module(register_name: str, hits: Path | None = None) -> str:
    """生成一个最小的定义模块源码（只注册，不真跑）。"""
    tail = f'Path(r"{hits}").open("a", encoding="utf-8").write("hit\\n")' if hits else ""
    return (
        "from pathlib import Path\n"
        "from dags.registry import register\n\n"
        f'register("{register_name}", "auto", seed_tag="s", seed_content="c",\n'
        '         final_tag="f", default_output="out/x.md")(lambda: None)\n'
        f"{tail}\n"
    )


# ---------------------------------------------------------------------------
# register
# ---------------------------------------------------------------------------
def test_register_writes_spec(clean_specs):
    @register("tmp-reg", "描述", seed_tag="s", seed_content="c", final_tag="f", default_output="out/x.md")
    def build():
        return None

    assert "tmp-reg" in DAG_SPECS
    spec = DAG_SPECS["tmp-reg"]
    assert spec.seed_tag == "s"
    assert spec.final_tag == "f"
    assert spec.build is build


def test_register_same_build_no_warning(clean_specs):
    def build():
        return None

    register("tmp-same", "描述", seed_tag="s", seed_content="c", final_tag="f", default_output="out/x.md")(build)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        register("tmp-same", "描述2", seed_tag="s", seed_content="c", final_tag="f", default_output="out/x.md")(build)
    assert not [w for w in caught if issubclass(w.category, RuntimeWarning)]


def test_register_different_build_warns(clean_specs):
    register("tmp-dup", "描述", seed_tag="s", seed_content="c", final_tag="f", default_output="out/x.md")(
        lambda: None
    )
    with pytest.warns(RuntimeWarning, match="重复注册"):
        register("tmp-dup", "描述", seed_tag="s", seed_content="c", final_tag="f", default_output="out/x.md")(
            lambda: None
        )


# ---------------------------------------------------------------------------
# load_definitions
# ---------------------------------------------------------------------------
def test_load_definitions_skips_underscore_and_ignores_init(tmp_path, clean_specs):
    d = tmp_path / "defs"
    d.mkdir()
    (d / "__init__.py").write_text("raise SystemExit('不该被加载')\n", encoding="utf-8")
    (d / "_helper.py").write_text("raise SystemExit('不该被加载')\n", encoding="utf-8")
    (d / "good.py").write_text(_def_module("tmp-good"), encoding="utf-8")

    assert load_definitions(d) == ["good"]
    assert "tmp-good" in DAG_SPECS


def test_load_definitions_is_idempotent(tmp_path, clean_specs):
    d = tmp_path / "defs"
    d.mkdir()
    hits = tmp_path / "hits.txt"
    (d / "good.py").write_text(_def_module("tmp-good2", hits=hits), encoding="utf-8")

    load_definitions(d)
    load_definitions(d)
    assert hits.read_text(encoding="utf-8").count("hit") == 1


def test_load_definitions_force_reexecutes(tmp_path, clean_specs):
    d = tmp_path / "defs"
    d.mkdir()
    hits = tmp_path / "hits.txt"
    (d / "good.py").write_text(_def_module("tmp-good3", hits=hits), encoding="utf-8")

    load_definitions(d)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # force 重载会触发重复注册警告，属预期
        load_definitions(d, force=True)
    assert hits.read_text(encoding="utf-8").count("hit") == 2


def test_load_definitions_same_stem_in_different_dirs(tmp_path, clean_specs):
    """回归：旧实现用 `dag_defs_{stem}` 当 sys.modules 键，第二个目录的同名模块会被静默跳过。"""
    a, b = tmp_path / "a", tmp_path / "b"
    a.mkdir()
    b.mkdir()
    (a / "x.py").write_text(_def_module("tmp-from-a"), encoding="utf-8")
    (b / "x.py").write_text(_def_module("tmp-from-b"), encoding="utf-8")

    load_definitions(a)
    load_definitions(b)
    assert {"tmp-from-a", "tmp-from-b"} <= set(DAG_SPECS)


def test_load_definitions_missing_dir_returns_empty(tmp_path):
    assert load_definitions(tmp_path / "nope") == []


def test_get_definition_module_unknown_raises():
    with pytest.raises(KeyError, match="定义模块未加载"):
        get_definition_module("definitely_not_a_module")


def test_get_definition_module_finds_stage1st():
    mod = get_definition_module("stage1st_ai")
    assert hasattr(mod, "build_stage1st_ai_dag")


# ---------------------------------------------------------------------------
# list / run
# ---------------------------------------------------------------------------
def test_list_dags_sorted():
    names = [s.name for s in list_dags()]
    assert names == sorted(names)
    assert len(names) == len(DAG_SPECS)


def test_run_dag_unknown_name_lists_available():
    with pytest.raises(KeyError) as ei:
        run_dag("no-such-dag")
    assert "no-such-dag" in str(ei.value)


def test_run_dag_dry_run_does_not_write(tmp_path, clean_specs):
    out = tmp_path / "out.md"
    DAG_SPECS["tmp-fake"] = _spec()
    res = run_dag("tmp-fake", output=str(out), dry_run=True)

    assert res["written"] is False
    assert res["report"] == "ok"
    assert len(res["steps"]) >= 1  # run() 返回本轮新产出的条目列表，不是步数
    assert not out.exists()


def test_run_dag_writes_output(tmp_path, clean_specs):
    out = tmp_path / "sub" / "out.md"
    DAG_SPECS["tmp-fake"] = _spec()
    res = run_dag("tmp-fake", output=str(out))

    assert res["written"] is True
    assert out.read_text(encoding="utf-8") == res["report"]


def test_run_dag_injects_seed_and_reaches_final_tag(tmp_path, clean_specs):
    DAG_SPECS["tmp-fake"] = _spec()
    res = run_dag("tmp-fake", output=str(tmp_path / "o.md"), dry_run=True)

    dag = res["dag"]
    assert dag.has_tag("seed") and dag.has_tag("final")
    assert dag.get_result("final") == "ok"


def test_run_dag_records_edge_timings(tmp_path, clean_specs):
    def build():
        d = DAG(name="timed", models={}, llm_client=MockLLMClient())
        d.add_edge(input_tags=["seed"], processor=timed(lambda x: "ok"), output_tags=["final"])
        return d

    spec = _spec()
    spec.build = build
    DAG_SPECS["tmp-fake"] = spec
    res = run_dag("tmp-fake", output=str(tmp_path / "o.md"), dry_run=True)

    assert "<lambda>" in res["timings"]
    assert res["timings"]["<lambda>"] >= 0.0


def test_timed_preserves_metadata():
    @timed
    def foo(x):
        return x + 1

    assert foo.__name__ == "foo"
    assert foo(1) == 2


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def test_cli_list(capsys):
    assert cli_main(["list"]) == 0
    out = capsys.readouterr().out
    assert "stage1st-ai-24h" in out


def test_cli_run_dry_run(capsys, tmp_path, clean_specs):
    DAG_SPECS["tmp-fake"] = _spec()
    out_file = tmp_path / "cli.md"
    assert cli_main(["run", "tmp-fake", "--dry-run", "-o", str(out_file)]) == 0
    captured = capsys.readouterr().out
    assert "dry-run" in captured
    assert not out_file.exists()


def test_run_dag_seed_override(tmp_path, clean_specs):
    """--seed / run_dag(seed=...) 必须覆盖注册时写死的 seed_content。"""
    seen = {}

    def build():
        d = DAG(name="seed_probe", models={}, llm_client=MockLLMClient())

        def echo(x):
            seen["got"] = getattr(x, "contents", x)
            return "ok"

        d.add_edge(input_tags=["seed"], processor=echo, output_tags=["final"])
        return d

    spec = _spec()
    spec.build = build
    DAG_SPECS["tmp-seed"] = spec
    run_dag("tmp-seed", output=str(tmp_path / "o.md"), dry_run=True, seed="替换后的种子")
    assert seen["got"] == "替换后的种子"


def test_cli_run_seed(capsys, tmp_path, clean_specs):
    seen = {}

    def build():
        d = DAG(name="seed_probe2", models={}, llm_client=MockLLMClient())

        def echo(x):
            seen["got"] = getattr(x, "contents", x)
            return "ok"

        d.add_edge(input_tags=["seed"], processor=echo, output_tags=["final"])
        return d

    spec = _spec()
    spec.build = build
    DAG_SPECS["tmp-seed2"] = spec
    assert cli_main(["run", "tmp-seed2", "--dry-run", "--seed", "from-cli", "-o", str(tmp_path / "o.md")]) == 0
    assert seen["got"] == "from-cli"
