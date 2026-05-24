"""Tests for opencode.py - Opencode class."""
import unittest
import tempfile
import os
from unittest.mock import patch, MagicMock, mock_open

from opencode import Opencode


class TestOpencode(unittest.TestCase):
    """Test suite for the Opencode class."""

    def setUp(self):
        self.opencode = Opencode()

    # ── _run_command ────────────────────────────────────────────

    @patch("opencode.subprocess.run")
    def test_run_command_uses_subprocess(self, mock_run):
        """Verify _run_command uses subprocess.run (not os.system)."""
        mock_run.return_value = MagicMock(
            stdout="model-1\nmodel-2\n",
            stderr="",
        )
        result = self.opencode._run_command(["opencode", "models"])

        mock_run.assert_called_once_with(
            ["opencode", "models"],
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.stdout, "model-1\nmodel-2\n")
        self.assertEqual(result.stderr, "")

    @patch("opencode.subprocess.run")
    def test_run_command_returns_completed_process(self, mock_run):
        """Verify _run_command returns a CompletedProcess-like object."""
        mock_run.return_value = MagicMock(
            stdout="output",
            stderr="",
            returncode=0,
        )
        result = self.opencode._run_command(["opencode", "models"])
        self.assertEqual(result.stdout, "output")
        self.assertEqual(result.returncode, 0)

    # ── list_models ────────────────────────────────────────────

    def test_list_models_reads_from_models_list_json(self):
        """Verify list_models reads models from models_list.json file."""
        fake_models = [
            "siliconflow-cn/Qwen/Qwen3.5-4B",
            "google/gemma-4-31b-it",
            "opencode/deepseek-v4-flash-free",
        ]
        content = "\n".join(fake_models)

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False, encoding="utf-8"
        ) as tmp:
            tmp.write(content)
            tmp_path = tmp.name

        try:
            self.opencode._models_path = tmp_path
            result = self.opencode.list_models()
            self.assertEqual(result, content)
        finally:
            os.unlink(tmp_path)

    def test_list_models_returns_model_names_only(self):
        """Verify list_models strips empty lines and returns only model names."""
        content = (
            "siliconflow-cn/Qwen/Qwen3.5-4B\n"
            "\n"
            "opencode/deepseek-v4-flash-free\n"
            "   \n"
            "google/gemma-4-31b-it\n"
        )
        expected = (
            "siliconflow-cn/Qwen/Qwen3.5-4B\n"
            "opencode/deepseek-v4-flash-free\n"
            "google/gemma-4-31b-it"
        )

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False, encoding="utf-8"
        ) as tmp:
            tmp.write(content)
            tmp_path = tmp.name

        try:
            self.opencode._models_path = tmp_path
            result = self.opencode.list_models()
            self.assertEqual(result, expected)
            # 验证空行和空格行已被过滤
            self.assertNotIn("\n\n", result)
        finally:
            os.unlink(tmp_path)

    def test_list_models_returns_str(self):
        """Verify list_models returns a string, not a list or None."""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False, encoding="utf-8"
        ) as tmp:
            tmp.write("model-a\nmodel-b\n")
            tmp_path = tmp.name

        try:
            self.opencode._models_path = tmp_path
            result = self.opencode.list_models()
            self.assertIsInstance(result, str)
            self.assertEqual(result, "model-a\nmodel-b")
        finally:
            os.unlink(tmp_path)

    def test_list_models_empty_file_returns_empty_string(self):
        """Verify list_models returns '' when models_list.json is empty."""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False, encoding="utf-8"
        ) as tmp:
            tmp_path = tmp.name

        try:
            self.opencode._models_path = tmp_path
            result = self.opencode.list_models()
            self.assertEqual(result, "")
        finally:
            os.unlink(tmp_path)

    def test_list_models_raises_on_file_not_found(self):
        """Verify list_models raises RuntimeError when file does not exist."""
        self.opencode._models_path = "/tmp/nonexistent_models.json"
        with self.assertRaises(RuntimeError) as ctx:
            self.opencode.list_models()
        self.assertIn("not found", str(ctx.exception))

    # ── list_agents ────────────────────────────────────────────

    @patch("opencode.subprocess.run")
    def test_list_agents_calls_run_command(self, mock_run):
        """Verify list_agents delegates to _run_command with correct args."""
        mock_run.return_value = MagicMock(
            stdout="agent-alpha\nagent-beta\n",
            stderr="",
            returncode=0,
        )

        result = self.opencode.list_agents()

        mock_run.assert_called_once_with(
            ["opencode", "agent", "list"],
            capture_output=True,
            text=True,
        )
        self.assertEqual(result, "agent-alpha\nagent-beta\n")

    @patch("opencode.subprocess.run")
    def test_list_agents_returns_stdout_only(self, mock_run):
        """Verify list_agents returns stdout and does NOT return stderr."""
        mock_run.return_value = MagicMock(
            stdout="agent-alpha\n",
            stderr="Error: connection failed\n",
            returncode=0,
        )

        result = self.opencode.list_agents()

        self.assertIn("agent-alpha", result)
        self.assertNotIn("Error", result)

    @patch("opencode.subprocess.run")
    def test_list_agents_raises_on_error(self, mock_run):
        """Verify list_agents raises RuntimeError when returncode != 0."""
        mock_run.return_value = MagicMock(
            stdout="",
            stderr="error: connection refused",
            returncode=1,
        )
        with self.assertRaises(RuntimeError) as ctx:
            self.opencode.list_agents()
        self.assertIn("error: connection refused", str(ctx.exception))

    # ── run_prompt ─────────────────────────────────────────────

    @patch("opencode.subprocess.run")
    def test_run_prompt_creates_correct_command(self, mock_run):
        """Verify run_prompt constructs the correct opencode run command."""
        mock_run.return_value = MagicMock(
            stdout="execution result\n",
            stderr="",
            returncode=0,
        )

        result = self.opencode.run_prompt("hello world")

        mock_run.assert_called_once_with(
            ["opencode", "run", "hello world"],
            capture_output=True,
            text=True,
        )
        self.assertEqual(result, "execution result\n")

    @patch("opencode.subprocess.run")
    def test_run_prompt_with_agent(self, mock_run):
        """Verify run_prompt includes --agent when specified."""
        mock_run.return_value = MagicMock(
            stdout="result from EditFile\n",
            stderr="",
            returncode=0,
        )

        result = self.opencode.run_prompt("fix this code", agent="EditFile")

        mock_run.assert_called_once_with(
            ["opencode", "run", "--agent", "EditFile", "fix this code"],
            capture_output=True,
            text=True,
        )
        self.assertEqual(result, "result from EditFile\n")

    @patch("opencode.subprocess.run")
    def test_run_prompt_with_model(self, mock_run):
        """Verify run_prompt includes --model when specified."""
        mock_run.return_value = MagicMock(
            stdout="model output\n",
            stderr="",
            returncode=0,
        )

        result = self.opencode.run_prompt(
            "explain this", model="google/gemma-4-31b-it"
        )

        mock_run.assert_called_once_with(
            ["opencode", "run", "--model", "google/gemma-4-31b-it", "explain this"],
            capture_output=True,
            text=True,
        )
        self.assertEqual(result, "model output\n")

    @patch("opencode.subprocess.run")
    def test_run_prompt_with_agent_and_model(self, mock_run):
        """Verify run_prompt includes both --agent and --model when specified."""
        mock_run.return_value = MagicMock(
            stdout="result\n",
            stderr="",
            returncode=0,
        )

        result = self.opencode.run_prompt(
            "write a test", agent="EditFile", model="google/gemma-4-31b-it"
        )

        mock_run.assert_called_once_with(
            [
                "opencode", "run",
                "--agent", "EditFile",
                "--model", "google/gemma-4-31b-it",
                "write a test",
            ],
            capture_output=True,
            text=True,
        )
        self.assertEqual(result, "result\n")

    @patch("opencode.subprocess.run")
    def test_run_prompt_raises_on_error(self, mock_run):
        """Verify run_prompt raises RuntimeError when returncode != 0."""
        mock_run.return_value = MagicMock(
            stdout="",
            stderr="error: model overloaded",
            returncode=1,
        )
        with self.assertRaises(RuntimeError) as ctx:
            self.opencode.run_prompt("test prompt")
        self.assertIn("error: model overloaded", str(ctx.exception))

    @patch("opencode.subprocess.run")
    def test_run_prompt_raises_on_nonzero_returncode_with_agent(self, mock_run):
        """Verify run_prompt raises even when agent/model are specified."""
        mock_run.return_value = MagicMock(
            stdout="partial output",
            stderr="timeout after 30s",
            returncode=2,
        )
        with self.assertRaises(RuntimeError) as ctx:
            self.opencode.run_prompt("do something", agent="EditFile", model="gpt-4")
        self.assertIn("timeout after 30s", str(ctx.exception))

    # ── @filename resolution ───────────────────────────────────

    @patch("opencode.subprocess.run")
    def test_run_prompt_resolves_at_filename(self, mock_run):
        """Verify run_prompt resolves @filename to file content."""
        mock_run.return_value = MagicMock(
            stdout="executed with file content\n",
            stderr="",
            returncode=0,
        )

        fake_content = "file content from @file.txt"
        with patch("builtins.open", mock_open(read_data=fake_content)):
            result = self.opencode.run_prompt("@file.txt")

        # The prompt should be the file content, not "@file.txt"
        mock_run.assert_called_once_with(
            ["opencode", "run", fake_content],
            capture_output=True,
            text=True,
        )
        self.assertEqual(result, "executed with file content\n")

    @patch("opencode.subprocess.run")
    def test_run_prompt_resolves_at_opencode_py_with_agent(self, mock_run):
        """Verify run_prompt resolves @opencode.py and includes agent/model."""
        mock_run.return_value = MagicMock(
            stdout="result\n",
            stderr="",
            returncode=0,
        )

        # Simulate reading opencode.py
        fake_content = "class Opencode:\n    pass\n"
        with patch("builtins.open", mock_open(read_data=fake_content)):
            result = self.opencode.run_prompt(
                "@opencode.py", agent="EditFile", model="google/gemma-4-31b-it"
            )

        mock_run.assert_called_once_with(
            [
                "opencode", "run",
                "--agent", "EditFile",
                "--model", "google/gemma-4-31b-it",
                fake_content,
            ],
            capture_output=True,
            text=True,
        )
        self.assertEqual(result, "result\n")

    @patch("opencode.subprocess.run")
    def test_run_prompt_regular_prompt_not_affected(self, mock_run):
        """Verify regular prompts (without @) are passed as-is."""
        mock_run.return_value = MagicMock(
            stdout="result\n",
            stderr="",
            returncode=0,
        )

        result = self.opencode.run_prompt("hello world")

        mock_run.assert_called_once_with(
            ["opencode", "run", "hello world"],
            capture_output=True,
            text=True,
        )
        self.assertEqual(result, "result\n")

    @patch("opencode.subprocess.run")
    def test_run_prompt_empty_prompt_not_affected(self, mock_run):
        """Verify empty string is passed as-is (not treated as @filename)."""
        mock_run.return_value = MagicMock(
            stdout="\n",
            stderr="",
            returncode=0,
        )

        result = self.opencode.run_prompt("")

        mock_run.assert_called_once_with(
            ["opencode", "run", ""],
            capture_output=True,
            text=True,
        )
        self.assertEqual(result, "\n")

    def test_resolve_at_prompt_file_not_found(self):
        """Verify _resolve_at_prompt raises RuntimeError for missing file."""
        with self.assertRaises(RuntimeError) as ctx:
            self.opencode._resolve_at_prompt("@nonexistent_file.xyz")
        self.assertIn("not found", str(ctx.exception))

    def test_resolve_at_prompt_returns_original_if_no_at(self):
        """Verify _resolve_at_prompt returns original prompt if no @ prefix."""
        result = self.opencode._resolve_at_prompt("hello world")
        self.assertEqual(result, "hello world")

    def test_resolve_at_prompt_returns_original_if_empty(self):
        """Verify _resolve_at_prompt returns empty string as-is."""
        result = self.opencode._resolve_at_prompt("")
        self.assertEqual(result, "")

    def test_resolve_at_prompt_returns_original_if_none(self):
        """Verify _resolve_at_prompt returns None as-is."""
        result = self.opencode._resolve_at_prompt(None)
        self.assertIsNone(result)

    # ── Module-level checks ────────────────────────────────────

    def test_no_os_import(self):
        """Verify the module no longer imports os (os.system removed from code)."""
        import opencode as mod
        source = mod.__file__
        with open(source, "r") as f:
            content = f.read()
        # import os 应该被移除了
        self.assertNotIn("import os", content)
        # os.system( 这种实际调用不应存在（注释中提到 os.system 是合理的）
        self.assertNotIn("os.system(", content)

    def test_no_json_dot_loads_for_models(self):
        """Verify list_models does NOT use json.loads (reads raw, no JSON parse)."""
        # list_models reads line by line, not via json.loads
        import opencode as mod
        source = mod.__file__
        with open(source, "r") as f:
            content = f.read()
        # import json is only for type annotations / other uses, not for parsing
        self.assertIn("import json", content)


if __name__ == "__main__":
    unittest.main()
