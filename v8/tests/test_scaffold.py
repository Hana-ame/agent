"""项目结构 → 图 的离线测试：扫描、生成配置、dry-run、指纹脚本。"""
from __future__ import annotations

import json
import shutil
import subprocess
import sys

import pytest
from pathlib import Path

V8 = Path(__file__).resolve().parent.parent
if str(V8) not in sys.path:
    sys.path.insert(0, str(V8))

import dag8  # noqa: E402
from dag8 import config as cfg_mod
from dag8 import scaffold
from dag8.lint import lint_config  # noqa: E402
from dag8.__main__ import main as dag8_main  # noqa: E402


def _make_project(tmp_path: Path) -> Path:
    root = tmp_path / "proj"
    (root / "pkg_a").mkdir(parents=True)
    (root / "pkg_a" / "__init__.py").write_text("", encoding="utf-8")
    (root / "pkg_a" / "core.py").write_text("def foo():\n    return 1\n", encoding="utf-8")
    (root / "pkg_b").mkdir()
    (root / "pkg_b" / "__init__.py").write_text("", encoding="utf-8")
    (root / "pkg_b" / "b.py").write_text("from pkg_a.core import foo\n", encoding="utf-8")
    (root / "plain").mkdir()
    (root / "plain" / "x.txt").write_text("x", encoding="utf-8")
    return root


def test_scan_packages(tmp_path):
    mods = scaffold.scan_modules(_make_project(tmp_path), depth=1, mode="package")
    assert [p.name for p in mods] == ["pkg_a", "pkg_b"]


def test_scan_dir_fallback(tmp_path):
    root = tmp_path / "nopkg"
    (root / "a").mkdir(parents=True)
    (root / "b").mkdir()
    mods = scaffold.scan_modules(root, depth=1, mode="package")
    assert sorted(p.name for p in mods) == ["a", "b"]


def test_build_config_shape(tmp_path):
    root = _make_project(tmp_path)
    cfg = scaffold.build_config(root, scaffold.scan_modules(root))
    edges = cfg["edges"]
    assert len(edges) == 3
    assert [e["kind"] for e in edges[:-1]] == ["command", "command"]
    assert edges[-1].get("kind", "pi") == "pi"
    assert edges[0]["expect"]["type"] == "json"
    assert edges[0]["cache"] is False
    assert cfg["result_tag"] == "report"
    assert edges[-1]["input_tags"] == [e["output_tags"][0] for e in edges[:-1]]


def test_scaffold_config_dry_run(tmp_path):
    root = _make_project(tmp_path)
    cfg = scaffold.build_config(root, scaffold.scan_modules(root))
    out = scaffold.write_config(cfg, tmp_path / "arch.json")
    res = dag8.run(out, db_path=tmp_path / "arch.sqlite3", dry_run=True, max_workers=2)
    assert res.status == "ok", res.summary()
    assert len(res.edges) == 3


def test_inspector_hash_changes(tmp_path):
    root = _make_project(tmp_path) / "pkg_a"
    script = V8 / "scripts" / "inspect_module.py"

    def run():
        r = subprocess.run([sys.executable, str(script), str(root)],
                           capture_output=True, text=True, check=True)
        return json.loads(r.stdout)

    first = run()
    assert first["module"] == "pkg_a"
    assert first["loc"] >= 2
    assert len(first["hash"]) == 16

    (root / "core.py").write_text("def foo():\n    return 2\n", encoding="utf-8")
    assert run()["hash"] != first["hash"]


def test_scaffold_cli(tmp_path):
    root = _make_project(tmp_path)
    out = tmp_path / "arch.json"
    assert dag8_main(["scaffold", str(root), "--out", str(out)]) == 0
    cfg = json.loads(out.read_text(encoding="utf-8"))
    assert len(cfg["edges"]) == 3


def _git_repo(tmp_path: Path) -> Path:
    root = _make_project(tmp_path)

    def git(*args):
        subprocess.run(["git", "-C", str(root), *args], check=True,
                       capture_output=True, text=True)

    git("init", "-q")
    git("add", ".")
    subprocess.run(["git", "-C", str(root), "-c", "user.email=t@t",
                    "-c", "user.name=t", "commit", "-qm", "init"],
                   check=True, capture_output=True, text=True)
    return root


@pytest.mark.skipif(shutil.which("git") is None, reason="需要 git")
def test_changed_modules_picks_touched(tmp_path):
    root = _git_repo(tmp_path)
    (root / "pkg_a" / "core.py").write_text("def foo():\n    return 3\n", encoding="utf-8")
    (root / "pkg_b" / "new.py").write_text("x = 1\n", encoding="utf-8")

    mods = scaffold.changed_modules(root)
    assert {p.name for p in mods} == {"pkg_a", "pkg_b"}


def test_changed_modules_requires_git(tmp_path):
    root = _make_project(tmp_path)
    with pytest.raises(ValueError, match="git"):
        scaffold.changed_modules(root)


@pytest.mark.skipif(shutil.which("git") is None, reason="需要 git")
def test_scaffold_cli_changed(tmp_path):
    root = _git_repo(tmp_path)
    (root / "pkg_a" / "core.py").write_text("def foo():\n    return 4\n", encoding="utf-8")

    out = tmp_path / "changed.json"
    assert dag8_main(["scaffold", str(root), "--changed", "--out", str(out)]) == 0
    cfg = json.loads(out.read_text(encoding="utf-8"))
    assert len(cfg["edges"]) == 2   # 1 个模块 + report


@pytest.mark.skipif(shutil.which("git") is None, reason="需要 git")
def test_scaffold_review_config(tmp_path):
    root = _git_repo(tmp_path)
    (root / "pkg_a" / "core.py").write_text("def foo():\n    return 5\n", encoding="utf-8")
    mods = scaffold.changed_modules(root)
    assert mods

    cfg = scaffold.build_review_config(root, mods)
    cmds = [e["command"] for e in cfg["edges"] if e.get("kind") == "command"]
    assert cmds and all("git -C" in c and "pkg_a" in c for c in cmds)
    assert cfg["result_tag"] == "review"

    path = scaffold.write_config(cfg, tmp_path / "review.json")
    res = dag8.run(path, db_path=tmp_path / "review.sqlite3", dry_run=True)
    assert res.status == "ok"


@pytest.mark.skipif(shutil.which("git") is None, reason="需要 git")
def test_scaffold_cli_review(tmp_path):
    root = _git_repo(tmp_path)
    (root / "pkg_a" / "core.py").write_text("def foo():\n    return 6\n", encoding="utf-8")

    out = tmp_path / "review.json"
    assert dag8_main(["scaffold", str(root), "--changed", "--review",
                      "--out", str(out)]) == 0
    cfg = json.loads(out.read_text(encoding="utf-8"))
    assert len(cfg["edges"]) == 2   # diff + review


def test_scaffold_compose(tmp_path):
    root = _make_project(tmp_path)
    mods = scaffold.scan_modules(root)
    parent, subs = scaffold.build_composed_config(root, mods)
    assert len(subs) == len(mods)

    graph_edges = [e for e in parent["edges"] if e.get("kind") == "graph"]
    assert len(graph_edges) == len(mods)
    assert all(e["graph"]["engine"] == "v8" for e in graph_edges)
    assert parent["edges"][-1]["id"] == "report"

    out = tmp_path / "parent.json"
    graphs = tmp_path / "graphs"
    graphs.mkdir()
    for name, sub in subs.items():
        scaffold.write_config(sub, graphs / name)
    scaffold.write_config(parent, out)

    loaded = cfg_mod.load(out)
    assert [i for i in lint_config(loaded) if i.level == "error"] == []
    res = dag8.run(out, db_path=tmp_path / "c.sqlite3", dry_run=True)
    assert res.status == "ok"


def test_scaffold_cli_compose(tmp_path):
    root = _make_project(tmp_path)
    out = tmp_path / "parent.json"
    assert dag8_main(["scaffold", str(root), "--compose", "--out", str(out)]) == 0
    assert out.exists()
    cfg = json.loads(out.read_text(encoding="utf-8"))
    assert any(e.get("kind") == "graph" for e in cfg["edges"])
    graphs = tmp_path / "graphs"
    assert graphs.is_dir() and list(graphs.glob("*.json"))


@pytest.mark.skipif(shutil.which("git") is None, reason="需要 git")
def test_scaffold_compose_review(tmp_path):
    root = _git_repo(tmp_path)
    (root / "pkg_a" / "core.py").write_text("def foo():\n    return 7\n", encoding="utf-8")
    mods = scaffold.changed_modules(root)

    parent, subs = scaffold.build_composed_review_config(root, mods)
    graph_edges = [e for e in parent["edges"] if e.get("kind") == "graph"]
    assert len(graph_edges) == len(mods)
    assert all(e["graph"]["engine"] == "v8" for e in graph_edges)
    assert parent["result_tag"] == "review"

    sub = next(iter(subs.values()))
    assert any(e.get("kind") == "command" for e in sub["edges"])

    out = tmp_path / "parent.json"
    graphs = tmp_path / "graphs"
    graphs.mkdir()
    for name, cfg in subs.items():
        scaffold.write_config(cfg, graphs / name)
    scaffold.write_config(parent, out)

    loaded = cfg_mod.load(out)
    assert [i for i in lint_config(loaded) if i.level == "error"] == []
    assert dag8.run(out, db_path=tmp_path / "rc.sqlite3", dry_run=True).status == "ok"


@pytest.mark.skipif(shutil.which("git") is None, reason="需要 git")
def test_scaffold_cli_compose_review(tmp_path):
    root = _git_repo(tmp_path)
    (root / "pkg_a" / "core.py").write_text("def foo():\n    return 8\n", encoding="utf-8")

    out = tmp_path / "parent.json"
    assert dag8_main(["scaffold", str(root), "--compose", "--review",
                      "--out", str(out)]) == 0
    cfg = json.loads(out.read_text(encoding="utf-8"))
    assert cfg["result_tag"] == "review"
    assert any(e.get("kind") == "graph" for e in cfg["edges"])
