"""配置静态检查的离线测试。"""
from __future__ import annotations

import json
import sys
from pathlib import Path

V8 = Path(__file__).resolve().parent.parent
if str(V8) not in sys.path:
    sys.path.insert(0, str(V8))

from dag8 import config as cfg_mod  # noqa: E402
from dag8.__main__ import main as dag8_main  # noqa: E402
from dag8.lint import lint_config  # noqa: E402


def _write(tmp_path: Path, data: dict) -> Path:
    p = tmp_path / "dag.json"
    p.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    return p


def _lint(tmp_path: Path, data: dict):
    return lint_config(cfg_mod.load(_write(tmp_path, data)))


def _errors(issues):
    return [i for i in issues if i.level == "error"]


def test_demo_config_is_clean():
    cfg = cfg_mod.load(V8 / "configs" / "demo.json")
    assert _errors(lint_config(cfg)) == []


def test_unknown_input_is_error(tmp_path):
    data = {
        "name": "t", "seed_tag": "a", "seed_content": "x", "result_tag": "out",
        "edges": [{"id": "e", "input_tags": ["a", "nope"],
                   "output_tags": ["out"], "prompt": "x"}],
    }
    assert any("nope" in i.message for i in _errors(_lint(tmp_path, data)))


def test_missing_result_is_error(tmp_path):
    data = {
        "name": "t", "seed_tag": "a", "seed_content": "x", "result_tag": "zzz",
        "edges": [{"id": "e", "input_tags": ["a"], "output_tags": ["b"], "prompt": "x"}],
    }
    assert any("zzz" in i.message for i in _errors(_lint(tmp_path, data)))


def test_duplicate_edge_id(tmp_path):
    data = {
        "name": "t", "seed_tag": "a", "seed_content": "x", "result_tag": "out",
        "edges": [
            {"id": "x", "input_tags": ["a"], "output_tags": ["b"], "prompt": "1"},
            {"id": "x", "input_tags": ["a"], "output_tags": ["out"], "prompt": "2"},
        ],
    }
    assert any("重复" in i.message for i in _errors(_lint(tmp_path, data)))


def test_cycle_detected(tmp_path):
    data = {
        "name": "t", "seed_tag": "a", "seed_content": "x", "result_tag": "a",
        "edges": [
            {"id": "e1", "input_tags": ["a"], "output_tags": ["b"], "prompt": "1"},
            {"id": "e2", "input_tags": ["b"], "output_tags": ["a"], "prompt": "2"},
        ],
    }
    assert any("成环" in i.message for i in _errors(_lint(tmp_path, data)))


def test_optional_unknown_not_flagged(tmp_path):
    data = {
        "name": "t", "seed_tag": "a", "seed_content": "x", "result_tag": "out",
        "edges": [{"id": "e", "input_tags": ["a", "b"], "optional_inputs": ["b"],
                   "output_tags": ["out"], "prompt": "x"}],
    }
    issues = _lint(tmp_path, data)
    assert not any("'b'" in i.message for i in issues)


def test_missing_graph_file(tmp_path):
    data = {
        "name": "t", "seed_tag": "a", "seed_content": "x", "result_tag": "out",
        "edges": [{"id": "g", "kind": "graph",
                   "input_tags": ["a"], "output_tags": ["out"],
                   "graph": {"path": "nope.json", "seed_tag": "a", "result_tag": "out"}}],
    }
    assert any("子图不存在" in i.message for i in _errors(_lint(tmp_path, data)))


def test_lint_cli_exit_codes(tmp_path):
    good = _write(tmp_path, {
        "name": "t", "seed_tag": "a", "seed_content": "x", "result_tag": "out",
        "edges": [{"id": "e", "input_tags": ["a"], "output_tags": ["out"], "prompt": "x"}],
    })
    assert dag8_main(["lint", str(good)]) == 0

    bad = _write(tmp_path, {
        "name": "t", "seed_tag": "a", "seed_content": "x", "result_tag": "out",
        "edges": [{"id": "e", "input_tags": ["nope"], "output_tags": ["out"], "prompt": "x"}],
    })
    assert dag8_main(["lint", str(bad)]) == 1


def test_strict_warnings(tmp_path):
    # 两个边产出同一个 tag → warning；非 strict 不失败
    data = {
        "name": "t", "seed_tag": "a", "seed_content": "x", "result_tag": "out",
        "edges": [
            {"id": "e1", "input_tags": ["a"], "output_tags": ["out"], "prompt": "1"},
            {"id": "e2", "input_tags": ["a"], "output_tags": ["out"], "prompt": "2"},
        ],
    }
    p = _write(tmp_path, data)
    assert any(i.level == "warning" for i in _lint(tmp_path, data))
    assert dag8_main(["lint", str(p)]) == 0
    assert dag8_main(["lint", str(p), "--strict"]) == 1


def test_run_lint_blocks_bad_config(tmp_path):
    bad = _write(tmp_path, {
        "name": "t", "seed_tag": "a", "seed_content": "x", "result_tag": "out",
        "edges": [{"id": "e", "input_tags": ["nope"], "output_tags": ["out"], "prompt": "x"}],
    })
    assert dag8_main(["run", str(bad), "--lint", "--dry-run"]) == 1


def test_run_lint_allows_good_config(tmp_path):
    good = _write(tmp_path, {
        "name": "t", "seed_tag": "a", "seed_content": "x", "result_tag": "out",
        "edges": [{"id": "e", "input_tags": ["a"], "output_tags": ["out"], "prompt": "x {contents}"}],
    })
    assert dag8_main(["run", str(good), "--lint", "--dry-run"]) == 0


def test_run_accepts_db_after_subcommand(tmp_path):
    good = _write(tmp_path, {
        "name": "t", "seed_tag": "a", "seed_content": "x", "result_tag": "out",
        "edges": [{"id": "e", "input_tags": ["a"], "output_tags": ["out"], "prompt": "x {contents}"}],
    })
    db = tmp_path / "after.sqlite3"
    assert dag8_main(["run", str(good), "--db", str(db), "--dry-run"]) == 0
    assert db.exists()


def test_run_json_output(tmp_path, capsys):
    good = _write(tmp_path, {
        "name": "t", "seed_tag": "a", "seed_content": "x", "result_tag": "out",
        "edges": [{"id": "e", "input_tags": ["a"], "output_tags": ["out"],
                   "prompt": "x {contents}"}],
    })
    db = tmp_path / "json.sqlite3"
    assert dag8_main(["run", str(good), "--dry-run", "--db", str(db),
                      "--json"]) == 0
    data = json.loads(capsys.readouterr().out)
    assert data["status"] == "ok"
    assert data["edges"][0]["edge"] == "e"
    assert data["db"] == str(db)


def test_run_json_error_exit(tmp_path, capsys):
    bad = _write(tmp_path, {
        "name": "t", "seed_tag": "a", "seed_content": "x", "result_tag": "out",
        "edges": [{"id": "e", "input_tags": ["missing"], "output_tags": ["out"],
                   "prompt": "x"}],
    })
    assert dag8_main(["run", str(bad), "--dry-run", "--json"]) == 1
    data = json.loads(capsys.readouterr().out)
    assert data["status"] != "ok"


def test_lint_json(tmp_path, capsys):
    good = _write(tmp_path, {
        "name": "t", "seed_tag": "a", "seed_content": "x", "result_tag": "out",
        "edges": [{"id": "e", "input_tags": ["a"], "output_tags": ["out"],
                   "prompt": "x {contents}"}],
    })
    assert dag8_main(["lint", str(good), "--json"]) == 0
    data = json.loads(capsys.readouterr().out)
    assert data["errors"] == 0
    assert data["issues"] == []

    bad = _write(tmp_path, {
        "name": "t", "seed_tag": "a", "seed_content": "x", "result_tag": "out",
        "edges": [{"id": "e", "input_tags": ["missing"], "output_tags": ["out"],
                   "prompt": "x"}],
    })
    assert dag8_main(["lint", str(bad), "--json"]) == 1
    data = json.loads(capsys.readouterr().out)
    assert data["errors"] >= 1


def test_cli_invalid_config_exit_2(tmp_path, capsys):
    bad = _write(tmp_path, {
        "name": "t", "seed_tag": "a", "seed_content": "x", "result_tag": "out",
        "edges": [{"id": "g", "kind": "graph", "input_tags": ["a"],
                   "output_tags": ["out"],
                   "graph": {"path": "x.json", "engine": "v9"}}],
    })
    assert dag8_main(["lint", str(bad)]) == 2
    assert "配置加载失败" in capsys.readouterr().out

    assert dag8_main(["run", str(bad), "--dry-run"]) == 2
    assert "运行失败" in capsys.readouterr().out


def test_run_preflight_blocks(tmp_path, monkeypatch, capsys):
    import dag8.doctor as doctor

    monkeypatch.setattr(
        doctor, "_check_http",
        lambda provider, timeout: doctor.Check("x", False, "HTTP 530", "源站故障"),
    )
    good = _write(tmp_path, {
        "name": "t", "seed_tag": "a", "seed_content": "x", "result_tag": "out",
        "edges": [{"id": "e", "input_tags": ["a"], "output_tags": ["out"],
                   "prompt": "x {contents}"}],
    })
    assert dag8_main(["run", str(good), "--dry-run", "--preflight"]) == 1
    out = capsys.readouterr().out
    assert "预检失败" in out
    assert "530" in out


def test_run_preflight_passes(tmp_path, monkeypatch):
    import dag8.doctor as doctor

    monkeypatch.setattr(
        doctor, "_check_http",
        lambda provider, timeout: doctor.Check("x", True, "HTTP 200"),
    )
    good = _write(tmp_path, {
        "name": "t", "seed_tag": "a", "seed_content": "x", "result_tag": "out",
        "edges": [{"id": "e", "input_tags": ["a"], "output_tags": ["out"],
                   "prompt": "x {contents}"}],
    })
    assert dag8_main(["run", str(good), "--dry-run", "--preflight"]) == 0


def test_preflight_json_keeps_stdout_clean(tmp_path, monkeypatch, capsys):
    import dag8.doctor as doctor

    monkeypatch.setattr(
        doctor, "_check_http",
        lambda provider, timeout: doctor.Check("x", True, "HTTP 200"),
    )
    good = _write(tmp_path, {
        "name": "t", "seed_tag": "a", "seed_content": "x", "result_tag": "out",
        "edges": [{"id": "e", "input_tags": ["a"], "output_tags": ["out"],
                   "prompt": "x {contents}"}],
    })
    assert dag8_main(["run", str(good), "--dry-run", "--preflight", "--json"]) == 0
    out = capsys.readouterr().out
    data = json.loads(out)   # 预检提示不应污染 stdout
    assert data["status"] == "ok"


def test_cli_max_cost_blocks(tmp_path, capsys):
    cfg = _write(tmp_path, {
        "name": "t", "seed_tag": "a", "seed_content": "x", "result_tag": "out",
        "edges": [{"id": "e", "input_tags": ["a"], "output_tags": ["out"],
                   "prompt": "x {contents}"}],
    })
    assert dag8_main(["run", str(cfg), "--dry-run", "--max-cost", "0",
                      "--json"]) == 1
    assert json.loads(capsys.readouterr().out)["status"] == "over_budget"


def test_cli_auto_model_routes(tmp_path):
    from dag8.store import Store

    db = tmp_path / "h.sqlite3"
    with Store(db) as st:
        for i in range(3):
            r = st.new_run(f"d{i}", "cfg", "seed", "S")
            st.log_edge(r, "e", 0, ["seed"], ["out"],
                        model="laptop/deepseek-flash", status="ok", elapsed=5.0)
            st.finish_run(r, "ok")
    cfg = _write(tmp_path, {
        "name": "t", "seed_tag": "a", "seed_content": "x", "result_tag": "out",
        "pi": {"provider": "laptop", "model": "glm-5.2",
               "timeout": 10, "retries": 1},
        "edges": [{"id": "e", "input_tags": ["a"], "output_tags": ["out"],
                   "prompt": "x {contents}"}],
    })
    assert dag8_main(["run", str(cfg), "--dry-run", "--db", str(db),
                      "--auto-model", "--json"]) == 0
    with Store(db) as st:
        last = st.conn.execute("SELECT model FROM edges ORDER BY id DESC LIMIT 1").fetchone()
    assert last["model"] == "laptop/deepseek-flash"
