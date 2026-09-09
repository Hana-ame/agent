"""Tests: the standalone edge CLI accepts custom edges as ``script.py:ClassName``.

``--type`` used to be pinned to ``choices=edge_type_choices()``, so argparse
rejected any script spec with ``invalid choice`` and exit code 2. The type is now
passed through untouched: built-in names and script specs are decided by
``EdgeV4.from_config`` (see ``is_dynamic_edge_type``), which is what every other
entry point (config JSON, ``vea-run-v4``, the HTTP API) already does.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from framework.edges.cli import parse_args

REPO_ROOT = Path(__file__).resolve().parent.parent

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


def _cli_env() -> dict:
    env = dict(os.environ)
    existing = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = str(REPO_ROOT) + (os.pathsep + existing if existing else "")
    return env


class TestTypeArgument:
    def test_script_spec_is_accepted(self) -> None:
        args = parse_args(["--type", "my_edges.py:UpperEdge", "--session", "s"])
        assert args.type == "my_edges.py:UpperEdge"

    def test_py_suffix_alone_is_accepted(self) -> None:
        args = parse_args(["--type", "my_edges.py", "--session", "s"])
        assert args.type == "my_edges.py"

    def test_built_in_name_still_accepted(self) -> None:
        for name in ("code", "llm", "tool", "llm_tool", "reflexive", "sensenova"):
            assert parse_args(["--type", name]).type == name

    def test_help_explains_the_script_spec_format(self) -> None:
        import contextlib

        import io

        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer):
            with pytest.raises(SystemExit) as excinfo:
                parse_args(["--help"])
        assert excinfo.value.code == 0
        assert "script.py:ClassName" in buffer.getvalue()

    def test_cli_runs_a_custom_edge_by_script_spec(self, tmp_path: Path) -> None:
        (tmp_path / "upper.py").write_text(EDGE_SOURCE, encoding="utf-8")
        proc = subprocess.run(
            [
                sys.executable,
                "-m",
                "framework.edges.cli",
                "--type",
                "upper.py:UpperEdge",
                "--session",
                "cli_type",
                "--input",
                "v_in",
                "--output",
                "v_out",
                "--seed-input",
                "hello runner",
                "--dir",
                str(tmp_path),
            ],
            capture_output=True,
            text=True,
            timeout=120,
            cwd=REPO_ROOT,
            env=_cli_env(),
        )
        assert proc.returncode == 0, proc.stderr
        payload = json.loads(proc.stdout)
        assert payload["success"] is True
        assert payload["output"] == "HELLO RUNNER"

    def test_unknown_name_fails_at_the_edge_layer_not_argparse(
        self, tmp_path: Path
    ) -> None:
        (tmp_path / "upper.py").write_text(EDGE_SOURCE, encoding="utf-8")
        proc = subprocess.run(
            [
                sys.executable,
                "-m",
                "framework.edges.cli",
                "--type",
                "no_such_edge",
                "--session",
                "cli_type",
                "--input",
                "v_in",
                "--output",
                "v_out",
                "--seed-input",
                "hello",
                "--dir",
                str(tmp_path),
            ],
            capture_output=True,
            text=True,
            timeout=120,
            cwd=REPO_ROOT,
            env=_cli_env(),
        )
        assert proc.returncode == 1
        assert "invalid choice" not in proc.stderr
        assert "script spec" in proc.stderr


class TestLightImport:
    """``import framework`` must not require the web stack."""

    def test_core_import_without_fastapi(self) -> None:
        env = dict(os.environ)
        env["PYTHONPATH"] = str(REPO_ROOT)  # no /tmp deps target
        code = (
            "import framework\n"
            "print(framework.GraphV4.__name__, framework.ExecutorV4.__name__)\n"
            "print(callable(framework.run_from_manifest))\n"
            "try:\n"
            "    framework.create_v4_server\n"
            "except ModuleNotFoundError as exc:\n"
            "    print(exc.name)\n"
        )
        proc = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
            timeout=120,
            cwd=REPO_ROOT,
            env=env,
        )
        assert proc.returncode == 0, proc.stderr
        assert proc.stdout.strip().splitlines() == ["GraphV4 ExecutorV4", "True", "fastapi"]

    def test_web_symbols_resolve_lazily(self) -> None:
        env = dict(os.environ)
        existing = env.get("PYTHONPATH", "")
        env["PYTHONPATH"] = str(REPO_ROOT) + (os.pathsep + existing if existing else "")
        code = (
            "import sys, framework\n"
            "print('framework.server_v4' in sys.modules)\n"
            "framework.create_v4_server\n"
            "print('framework.server_v4' in sys.modules)\n"
            "framework.SSEExecutorV4\n"
            "print('framework.sse_executor_v4' in sys.modules)\n"
        )
        proc = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
            timeout=120,
            cwd=REPO_ROOT,
            env=env,
        )
        assert proc.returncode == 0, proc.stderr
        assert proc.stdout.strip().split() == ["False", "True", "True"]

    def test_star_import_still_lists_everything(self) -> None:
        env = dict(os.environ)
        existing = env.get("PYTHONPATH", "")
        env["PYTHONPATH"] = str(REPO_ROOT) + (os.pathsep + existing if existing else "")
        code = (
            "from framework import *\n"
            "print(callable(create_v4_server), callable(SSEExecutorV4))\n"
            "print(callable(run_from_manifest), callable(WorkflowExecutorV4))\n"
        )
        proc = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
            timeout=120,
            cwd=REPO_ROOT,
            env=env,
        )
        assert proc.returncode == 0, proc.stderr
        assert proc.stdout.strip().split() == ["True", "True", "True", "True"]

    def test_unknown_attribute_is_still_attribute_error(self) -> None:
        env = dict(os.environ)
        env["PYTHONPATH"] = str(REPO_ROOT)
        proc = subprocess.run(
            [sys.executable, "-c", "import framework; framework.nope_xyz"],
            capture_output=True,
            text=True,
            timeout=120,
            cwd=REPO_ROOT,
            env=env,
        )
        assert proc.returncode != 0
        assert "AttributeError" in proc.stderr


class TestCliModuleFormIsWarningFree:
    """``python -m framework.edges.cli`` must not print runpy's RuntimeWarning.

    ``framework.edge_v4`` used to do ``from framework.edges.cli import main,
    parse_args`` at module level. Since ``framework/__init__.py`` imports
    ``edge_v4``, the CLI module landed in ``sys.modules`` before runpy executed
    it, which printed a RuntimeWarning on stderr.
    """

    def test_stderr_is_clean_when_running_the_cli(self, tmp_path: Path) -> None:
        (tmp_path / "upper.py").write_text(EDGE_SOURCE, encoding="utf-8")
        proc = subprocess.run(
            [
                sys.executable,
                "-m",
                "framework.edges.cli",
                "--type",
                "upper.py:UpperEdge",
                "--session",
                "quiet",
                "--input",
                "v_in",
                "--output",
                "v_out",
                "--seed-input",
                "hello runner",
                "--dir",
                str(tmp_path),
            ],
            capture_output=True,
            text=True,
            timeout=120,
            cwd=REPO_ROOT,
            env=_cli_env(),
        )
        assert proc.returncode == 0
        assert "RuntimeWarning" not in proc.stderr
        assert proc.stderr.strip() == ""
        assert json.loads(proc.stdout)["output"] == "HELLO RUNNER"

    def test_edge_v4_imports_the_cli_lazily(self) -> None:
        env = dict(os.environ)
        existing = env.get("PYTHONPATH", "")
        env["PYTHONPATH"] = str(REPO_ROOT) + (os.pathsep + existing if existing else "")
        code = (
            "import sys, framework.edge_v4 as ev\n"
            "print('framework.edges.cli' in sys.modules)\n"
            "ev.parse_args(['--type', 'a.py:B'])\n"
            "print('framework.edges.cli' in sys.modules)\n"
            "from framework.edge_v4 import main\n"
            "print(main.__name__)\n"
            "print('main' in ev.__all__ and 'parse_args' in ev.__all__)\n"
        )
        proc = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
            timeout=120,
            cwd=REPO_ROOT,
            env=env,
        )
        assert proc.returncode == 0, proc.stderr
        assert proc.stdout.strip().split() == ["False", "True", "main", "True"]


class TestFlagJsonKeyMapping:
    """CLI flags line up with the config JSON keys: ``--id`` ↔ ``"id"``, and so on.

    ``--input`` / ``--output`` keep their short names and map onto ``"input_vertex"``
    / ``"output_vertex"``; ``--edge-id`` is now spelled ``--id`` (the old spelling
    still works); the canonical config key for the session is ``"session"``.
    """

    def test_id_flag_is_the_edge_identifier(self, tmp_path: Path) -> None:
        (tmp_path / "upper.py").write_text(EDGE_SOURCE, encoding="utf-8")
        proc = subprocess.run(
            [
                sys.executable,
                "-m",
                "framework.edges.cli",
                "--dir",
                str(tmp_path),
                "--id",
                "custom_id",
                "--type",
                "upper.py:UpperEdge",
                "--session",
                "id_flag",
                "--input",
                "v_in",
                "--output",
                "v_out",
                "--seed-input",
                "flagged",
            ],
            capture_output=True,
            text=True,
            timeout=120,
            cwd=REPO_ROOT,
            env=_cli_env(),
        )
        assert proc.returncode == 0, proc.stderr
        payload = json.loads(proc.stdout)
        assert payload["edge_id"] == "custom_id"
        assert payload["output"] == "FLAGGED"

    def test_edge_id_is_a_deprecated_alias(self) -> None:
        assert parse_args(["--id", "e1"]).edge_id == "e1"
        assert parse_args(["--edge-id", "e1"]).edge_id == "e1"

    def test_seed_flag_is_the_canonical_name(self, tmp_path: Path) -> None:
        (tmp_path / "upper.py").write_text(EDGE_SOURCE, encoding="utf-8")
        proc = subprocess.run(
            [
                sys.executable,
                "-m",
                "framework.edges.cli",
                "--dir",
                str(tmp_path),
                "--id",
                "seed_edge",
                "--type",
                "upper.py:UpperEdge",
                "--session",
                "seed_flag",
                "--input",
                "v_in",
                "--output",
                "v_out",
                "--seed",
                "seed me",
            ],
            capture_output=True,
            text=True,
            timeout=120,
            cwd=REPO_ROOT,
            env=_cli_env(),
        )
        assert proc.returncode == 0, proc.stderr
        assert json.loads(proc.stdout)["output"] == "SEED ME"

    def test_seed_input_flag_is_a_deprecated_alias(self) -> None:
        assert parse_args(["--seed", "x"]).seed_input == "x"
        assert parse_args(["--seed-input", "y"]).seed_input == "y"

    def test_help_names_both_spellings(self) -> None:
        import contextlib

        import io

        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer):
            with pytest.raises(SystemExit) as excinfo:
                parse_args(["--help"])
        assert excinfo.value.code == 0
        help_text = buffer.getvalue()
        assert "--id" in help_text
        assert "--edge-id" in help_text
        assert "--input" in help_text and "input_vertex" in help_text

    def test_config_json_session_key_is_canonical(self, tmp_path: Path) -> None:
        (tmp_path / "upper.py").write_text(EDGE_SOURCE, encoding="utf-8")
        config = tmp_path / "edge.json"
        config.write_text(
            json.dumps(
                {
                    "session": "from_json",
                    "id": "cfg_edge",
                    "type": "upper.py:UpperEdge",
                    "input_vertex": "v_in",
                    "output_vertex": "v_out",
                    "seed": "from json",
                }
            ),
            encoding="utf-8",
        )
        proc = subprocess.run(
            [
                sys.executable,
                "-m",
                "framework.edges.cli",
                "--config",
                str(config),
                "--dir",
                str(tmp_path),
            ],
            capture_output=True,
            text=True,
            timeout=120,
            cwd=REPO_ROOT,
            env=_cli_env(),
        )
        assert proc.returncode == 0, proc.stderr
        payload = json.loads(proc.stdout)
        assert payload["edge_id"] == "cfg_edge"
        assert payload["output"] == "FROM JSON"

    def test_config_json_session_id_alias_still_works(self, tmp_path: Path) -> None:
        (tmp_path / "upper.py").write_text(EDGE_SOURCE, encoding="utf-8")
        config = tmp_path / "edge.json"
        config.write_text(
            json.dumps(
                {
                    "session_id": "from_json_alias",
                    "type": "upper.py:UpperEdge",
                    "input_vertex": "v_in",
                    "output_vertex": "v_out",
                    "seed_input": "alias",
                }
            ),
            encoding="utf-8",
        )
        proc = subprocess.run(
            [
                sys.executable,
                "-m",
                "framework.edges.cli",
                "--config",
                str(config),
                "--dir",
                str(tmp_path),
            ],
            capture_output=True,
            text=True,
            timeout=120,
            cwd=REPO_ROOT,
            env=_cli_env(),
        )
        assert proc.returncode == 0, proc.stderr
        assert json.loads(proc.stdout)["output"] == "ALIAS"

    def test_missing_session_mentions_the_canonical_key(self, tmp_path: Path) -> None:
        config = tmp_path / "edge.json"
        config.write_text(
            json.dumps({"type": "code", "input_vertex": "a", "output_vertex": "b"}),
            encoding="utf-8",
        )
        proc = subprocess.run(
            [
                sys.executable,
                "-m",
                "framework.edges.cli",
                "--config",
                str(config),
                "--dir",
                str(tmp_path),
            ],
            capture_output=True,
            text=True,
            timeout=120,
            cwd=REPO_ROOT,
            env=_cli_env(),
        )
        assert proc.returncode == 1
        assert "--session" in proc.stderr
        assert "'session'" in proc.stderr
        assert "deprecated alias" in proc.stderr
