"""Tests for the V4 manifest runner (``framework.run_v4`` / ``vea-run-v4``).

The V1 runner ``examples/run.py`` is deliberately left untouched; these tests
cover the new V4 entry point: the one-step API, the CLI, the V1-manifest guard
and the script-root wiring that makes a manifest outside the repository usable.
"""

from __future__ import annotations

import asyncio
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from framework.run_v4 import (
    _looks_like_v1_manifest,
    load_v4_manifest,
    main,
    parse_args,
    run_from_manifest,
    run_manifest_async,
)
from framework.utils.script_loader import load_script
from framework.vertex_v4 import VertexStoreV4

REPO_ROOT = Path(__file__).resolve().parent.parent
CUSTOM_EDGE_MANIFEST = REPO_ROOT / "examples" / "custom_edge" / "config.json"
V1_MANIFEST = REPO_ROOT / "examples" / "simple" / "config.json"

EDGE_SOURCE = """
from framework.edges.base import EdgeResultV4, EdgeV4
from framework.vertex_v4 import VertexStateV4


class UpperEdge(EdgeV4):
    async def run(self, session_id, store, agent=None, auto_transition=True, **kwargs):
        ok, reason, in_v, out_v = self.check_handshake(session_id, store)
        if not ok:
            return EdgeResultV4(edge_id=self.id, success=False, skipped=True, reason=reason)
        out = str(in_v.content).upper()
        store.apply_merge_strategy(
            session_id=session_id, name=self.output_vertex, incoming_content=out
        )
        if auto_transition:
            store.update_vertex_state(
                session_id, self.output_vertex, VertexStateV4.DATA_READY.value
            )
        return EdgeResultV4(edge_id=self.id, success=True, output=out)
"""


def _runner_env() -> dict:
    """Environment for subprocess invocations (repo + installed deps on PYTHONPATH)."""
    env = dict(os.environ)
    existing = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = str(REPO_ROOT) + (os.pathsep + existing if existing else "")
    return env


def write_manifest(base: Path, spec: str) -> Path:
    """Write a two-vertex / one-edge V4 manifest into ``base``."""
    path = base / "graph.json"
    path.write_text(
        json.dumps(
            {
                "version": "4.0",
                "metadata": {"name": "run_v4_test"},
                "vertices": [
                    {"name": "v_in", "content": "hello runner", "state": "data ready"},
                    {"name": "v_out", "content": "", "state": "todo"},
                ],
                "edges": [
                    {
                        "id": "e1",
                        "type": spec,
                        "input_vertex": "v_in",
                        "output_vertex": "v_out",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    return path


class TestOneStepApi:
    def test_run_from_manifest_loads_seeds_and_executes(self) -> None:
        result = run_from_manifest(str(CUSTOM_EDGE_MANIFEST), session_id="runner_api")

        assert result.success
        assert result.session_id == "runner_api"
        assert result.completed_edges == ["e_count", "e_upper", "e_passthrough"]
        assert result.vertex_contents["v_out"] == "5 WORDS"
        assert result.errors == []

    def test_run_manifest_async_works_inside_a_running_loop(self) -> None:
        async def run() -> object:
            return await run_manifest_async(str(CUSTOM_EDGE_MANIFEST), session_id="runner_async")

        result = asyncio.run(run())
        assert result.success
        assert result.vertex_contents["v_out"] == "5 WORDS"

    def test_run_from_manifest_refuses_a_nested_event_loop(self) -> None:
        async def nested() -> None:
            with pytest.raises(RuntimeError, match="cannot be called from a running event loop"):
                run_from_manifest(str(CUSTOM_EDGE_MANIFEST), session_id="nested")

        asyncio.run(nested())

    def test_caller_supplied_store_is_not_closed(self) -> None:
        store = VertexStoreV4(":memory:")
        result = run_from_manifest(str(CUSTOM_EDGE_MANIFEST), session_id="kept", store=store)

        assert result.success
        names = {v.name for v in store.list_vertices("kept")}
        assert {"v_in", "v_count", "v_upper", "v_out"} <= names
        store.close()

    def test_missing_manifest_raises_file_not_found(self) -> None:
        with pytest.raises(FileNotFoundError, match="not found"):
            run_from_manifest("/tmp/definitely-not-a-manifest.json")


class TestManifestGuard:
    def test_v1_manifest_is_refused_with_an_actionable_message(self) -> None:
        with pytest.raises(ValueError, match="V1 manifest") as excinfo:
            load_v4_manifest(str(V1_MANIFEST))
        assert "examples/run.py" in str(excinfo.value)
        assert "vea-run-v4" in str(excinfo.value)

    def test_v4_manifest_is_accepted(self) -> None:
        graph = load_v4_manifest(str(CUSTOM_EDGE_MANIFEST), override_session_id="guard")
        assert graph.session_id == "guard"
        assert len(graph.edges) == 3

    def test_detection_is_conservative(self) -> None:
        assert _looks_like_v1_manifest(
            {"vertices": [{"id": "a", "initial_data": []}, {"id": "b", "settings": {}}]}
        )
        assert not _looks_like_v1_manifest(
            {"vertices": [{"name": "a", "state": "data ready"}]}
        )
        # version 4 wins over vertex shape
        assert not _looks_like_v1_manifest({"version": "4.0", "vertices": [{"id": "a"}]})
        assert not _looks_like_v1_manifest({"vertices": []})


class TestScriptRoots:
    def test_manifest_adjacent_script_loads(self, tmp_path: Path) -> None:
        (tmp_path / "upper.py").write_text(EDGE_SOURCE, encoding="utf-8")
        manifest = write_manifest(tmp_path, "upper.py:UpperEdge")

        result = run_from_manifest(str(manifest), session_id="adjacent")
        assert result.success
        assert result.vertex_contents["v_out"] == "HELLO RUNNER"

    def test_manifest_adjacent_script_loads_when_other_roots_configured(
        self, tmp_path: Path
    ) -> None:
        # Regression: configuring --script-root must not break a script that lives
        # next to the manifest.
        (tmp_path / "upper.py").write_text(EDGE_SOURCE, encoding="utf-8")
        other = tmp_path / "other"
        other.mkdir()
        manifest = write_manifest(tmp_path, "upper.py:UpperEdge")

        graph = load_v4_manifest(str(manifest), script_roots=[other])
        assert type(graph.edges["e1"]).__name__ == "UpperEdge"

    def test_script_found_only_in_a_configured_root(self, tmp_path: Path) -> None:
        lib = tmp_path / "lib"
        lib.mkdir()
        (lib / "upper.py").write_text(EDGE_SOURCE, encoding="utf-8")
        manifest = write_manifest(tmp_path, "upper.py:UpperEdge")

        graph = load_v4_manifest(str(manifest), script_roots=[lib])
        assert type(graph.edges["e1"]).__name__ == "UpperEdge"

        result = run_from_manifest(
            str(manifest), session_id="from_root", script_roots=[lib]
        )
        assert result.success
        assert result.vertex_contents["v_out"] == "HELLO RUNNER"

    def test_missing_script_reports_not_found_not_confinement(self, tmp_path: Path) -> None:
        lib = tmp_path / "lib"
        lib.mkdir()
        with pytest.raises(FileNotFoundError, match="not found"):
            load_script("does-not-exist.py", allowed_roots=[lib])


class TestCli:
    def test_parse_args_defaults(self) -> None:
        args = parse_args(["graph.json"])
        assert args.manifest == "graph.json"
        assert args.session is None
        assert args.db == ":memory:"
        assert args.concurrency == 4
        assert args.timeout == 120.0
        assert args.script_roots is None
        assert not args.json and not args.quiet and not args.print_events

    def test_script_root_is_repeatable(self) -> None:
        args = parse_args(
            ["graph.json", "--script-root", "a", "--script-root", "b", "--concurrency", "2"]
        )
        assert args.script_roots == ["a", "b"]
        assert args.concurrency == 2

    def test_main_success_summary(self, tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
        (tmp_path / "upper.py").write_text(EDGE_SOURCE, encoding="utf-8")
        manifest = write_manifest(tmp_path, "upper.py:UpperEdge")

        code = main([str(manifest), "--session", "cli_ok"])

        assert code == 0
        output = capsys.readouterr().out
        assert "success:     True" in output
        assert "v_out: HELLO RUNNER" in output
        assert "completed:   e1" in output

    def test_main_json_output(self, tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
        (tmp_path / "upper.py").write_text(EDGE_SOURCE, encoding="utf-8")
        manifest = write_manifest(tmp_path, "upper.py:UpperEdge")

        code = main([str(manifest), "--json", "--quiet"])

        assert code == 0
        payload = json.loads(capsys.readouterr().out)
        assert payload["success"] is True
        assert payload["vertex_contents"]["v_out"] == "HELLO RUNNER"

    def test_main_v1_manifest_returns_one_and_points_at_the_legacy_runner(
        self, capsys: pytest.CaptureFixture
    ) -> None:
        code = main([str(V1_MANIFEST)])

        assert code == 1
        error = capsys.readouterr().err
        assert "V1 manifest" in error
        assert "examples/run.py" in error
        assert capsys.readouterr().out == ""

    def test_main_missing_manifest_returns_one(self, capsys: pytest.CaptureFixture) -> None:
        code = main(["/tmp/no-such-manifest.json"])
        assert code == 1
        assert "not found" in capsys.readouterr().err

    def test_main_reports_a_bad_edge_spec_as_a_failure(
        self, tmp_path: Path, capsys: pytest.CaptureFixture
    ) -> None:
        manifest = write_manifest(tmp_path, "missing_edge.py:Missing")

        assert main([str(manifest), "--quiet"]) == 1
        assert "Script not found" in capsys.readouterr().err

    def test_main_script_root_is_honoured(self, tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
        lib = tmp_path / "lib"
        lib.mkdir()
        (lib / "upper.py").write_text(EDGE_SOURCE, encoding="utf-8")
        manifest = write_manifest(tmp_path, "upper.py:UpperEdge")

        code = main([str(manifest), "--script-root", str(lib), "--session", "cli_root"])

        assert code == 0
        assert "success:     True" in capsys.readouterr().out

    def test_main_file_database_persists(self, tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
        (tmp_path / "upper.py").write_text(EDGE_SOURCE, encoding="utf-8")
        manifest = write_manifest(tmp_path, "upper.py:UpperEdge")
        db_path = tmp_path / "run.db"

        code = main([str(manifest), "--db", str(db_path), "--session", "cli_db", "--quiet"])

        assert code == 0
        assert db_path.exists()
        import sqlite3

        with sqlite3.connect(str(db_path)) as conn:
            vertices = dict(
                conn.execute("SELECT name, state FROM vertices WHERE session_id = 'cli_db'")
            )
            edge_type = conn.execute(
                "SELECT edge_type FROM edges WHERE session_id = 'cli_db'"
            ).fetchone()[0]
        assert vertices["v_out"] == "data ready"
        assert edge_type == "upper.py:UpperEdge"


class TestEntryPoints:
    def test_console_script_is_declared(self) -> None:
        pyproject = (REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8")
        assert 'vea-run-v4 = "framework.run_v4:main"' in pyproject
        assert 'vea-server = "framework.server.app:main"' in pyproject

    def test_package_export_is_available(self) -> None:
        import framework

        assert callable(framework.run_from_manifest)
        assert callable(framework.run_manifest_async)
        assert callable(framework.load_v4_manifest)
        for name in ("run_from_manifest", "run_manifest_async", "load_v4_manifest"):
            assert name in framework.__all__
        # The runner module is not imported eagerly by the package: it is resolved
        # through the PEP 562 ``__getattr__`` on first attribute access.
        proc = subprocess.run(
            [
                sys.executable,
                "-c",
                "import sys, framework; print('framework.run_v4' in sys.modules);"
                "callable(framework.run_from_manifest)",
            ],
            capture_output=True,
            text=True,
            timeout=120,
            cwd=REPO_ROOT,
            env=_runner_env(),
        )
        assert proc.returncode == 0, proc.stderr
        assert proc.stdout.strip() == "False"

    def test_module_form_has_no_runpy_warning(self) -> None:
        proc = subprocess.run(
            [sys.executable, "-m", "framework.run_v4", str(CUSTOM_EDGE_MANIFEST), "--session", "runpy"],
            capture_output=True,
            text=True,
            timeout=120,
            cwd=REPO_ROOT,
            env=_runner_env(),
        )
        assert proc.returncode == 0, proc.stderr
        assert "RuntimeWarning" not in proc.stderr
        assert "5 WORDS" in proc.stdout

    def test_v1_runner_is_untouched(self) -> None:
        # Compatibility: the legacy V1 entry point keeps its own stack.
        run_v1 = (REPO_ROOT / "examples" / "run.py").read_text(encoding="utf-8")
        assert "from framework import Graph, Executor" in run_v1
        assert "framework.run_v4" not in run_v1
