# test_run_shell_command.py
import unittest
import subprocess
import platform
import os
import tempfile
import shutil
from execute_command import run_shell_command   # your function

class TestRunShellCommand(unittest.TestCase):

    def setUp(self):
        """Create a temporary directory for file tests."""
        self.test_dir = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.test_dir)

    def test_basic_command(self):
        """Test a simple echo command and capture stdout."""
        result = run_shell_command("echo 'Hello, World!'")
        self.assertEqual(result.returncode, 0)
        self.assertEqual(result.stdout.strip(), "Hello, World!")
        self.assertEqual(result.stderr, "")

    def test_command_with_error(self):
        """Test a command that writes to stderr and returns non-zero."""
        result = run_shell_command("ls /nonexistent_path_12345")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("No such file or directory", result.stderr)

    def test_multi_line_command(self):
        """Test a multi-line shell script."""
        cmd = """
            echo "line1"
            echo "line2"
            echo "line3"
        """
        result = run_shell_command(cmd)
        expected = "line1\nline2\nline3"
        self.assertEqual(result.stdout.strip(), expected)

    def test_here_document(self):
        """Test writing a file using a here-document (EOF)."""
        outfile = os.path.join(self.test_dir, "test.txt")
        # Use raw string to avoid Python interpreting backslashes
        cmd = f"""
        cat > {outfile} << 'EOF'
This is line 1
This is line 2
EOF
        """
        result = run_shell_command(cmd)
        self.assertEqual(result.returncode, 0)

        # Verify file contents
        with open(outfile, "r") as f:
            content = f.read().strip()
        self.assertEqual(content, "This is line 1\nThis is line 2")

    def test_variable_expansion_without_quoting(self):
        """Test that $VAR is expanded when delimiter is unquoted."""
        outfile = os.path.join(self.test_dir, "var_test.txt")
        os.environ["TEST_VAR"] = "expanded_value"
        cmd = f"""
        cat > {outfile} << EOF
Variable = $TEST_VAR
EOF
        """
        result = run_shell_command(cmd)
        self.assertEqual(result.returncode, 0)
        with open(outfile, "r") as f:
            content = f.read().strip()
        self.assertEqual(content, "Variable = expanded_value")

    def test_variable_expansion_with_quoting(self):
        """Test that $VAR is NOT expanded when delimiter is quoted."""
        outfile = os.path.join(self.test_dir, "var_test_quoted.txt")
        os.environ["TEST_VAR"] = "expanded_value"  # set but should be ignored
        cmd = f"""
        cat > {outfile} << 'EOF'
Variable = $TEST_VAR
EOF
        """
        result = run_shell_command(cmd)
        self.assertEqual(result.returncode, 0)
        with open(outfile, "r") as f:
            content = f.read().strip()
        self.assertEqual(content, "Variable = $TEST_VAR")

    def test_platform_specific_default(self):
        """Check that the correct default shell is chosen for the platform."""
        # We can't easily test the actual default without mocking,
        # but we can run a command that uses shell features.
        if platform.system() == "Windows":
            # On Windows with Git Bash, 'which' exists
            result = run_shell_command("which bash")
            self.assertEqual(result.returncode, 0)
            self.assertIn("bash.exe", result.stdout)
        else:
            # On Linux/macOS, /bin/sh should exist
            result = run_shell_command("echo $0")
            # $0 often shows the shell name; but it's not guaranteed.
            # Simpler: check that a basic command works.
            result = run_shell_command("pwd")
            self.assertEqual(result.returncode, 0)

    def test_custom_shell_path(self):
        """Test specifying a different shell executable."""
        if platform.system() == "Windows":
            # Use cmd.exe as an alternative shell on Windows
            # Note: cmd.exe uses different syntax.
            result = run_shell_command("echo %CD%", shell_path="cmd.exe")
            self.assertEqual(result.returncode, 0)
            self.assertTrue(len(result.stdout) > 0)
        else:
            # On Unix, use bash explicitly
            result = run_shell_command("echo $BASH", shell_path="/bin/bash")
            self.assertEqual(result.returncode, 0)
            # $BASH should be set in bash, but not in sh; not foolproof.
            # Alternative: test a bash-specific feature
            result = run_shell_command("echo ${PWD##*/}", shell_path="/bin/bash")
            self.assertEqual(result.returncode, 0)

    def test_stdout_and_stderr_separate(self):
        """Verify that stdout and stderr are captured separately."""
        # Command that prints to both stdout and stderr
        cmd = "echo 'out'; echo 'err' >&2"
        result = run_shell_command(cmd)
        self.assertEqual(result.stdout.strip(), "out")
        self.assertEqual(result.stderr.strip(), "err")

    def test_return_code(self):
        """Test that returncode is captured correctly."""
        result = run_shell_command("exit 42")
        self.assertEqual(result.returncode, 42)

    # ----------------------------------------------------------------------
    # Security demonstration (not an actual test, but a warning)
    # ----------------------------------------------------------------------
    def test_shell_injection_danger(self):
        """Demonstrate why shell=True is dangerous with untrusted input."""
        # Simulate user input containing a malicious command
        user_input = "hello; rm -rf /tmp/danger"
        # If we naively build the command, the injected part runs!
        dangerous_cmd = f"echo {user_input}"
        # We expect this to actually try to delete /tmp/danger (if it exists)
        # In a test we won't create that file; we just show the risk.
        # Instead, we'll run a harmless injection and check that it was interpreted.
        user_input = "hello; echo 'INJECTED'"
        dangerous_cmd = f"echo {user_input}"
        result = run_shell_command(dangerous_cmd)
        # The output will be "hello" on one line and "INJECTED" on another
        # because the semicolon ends the echo command and starts a new one.
        output_lines = result.stdout.strip().split('\n')
        self.assertIn("INJECTED", output_lines)

        # Correct way: use shlex.quote() to escape
        import shlex
        safe_cmd = f"echo {shlex.quote(user_input)}"
        result_safe = run_shell_command(safe_cmd)
        # Now the entire string is treated as one argument to echo
        self.assertEqual(result_safe.stdout.strip(), "hello; echo 'INJECTED'")

if __name__ == '__main__':
    unittest.main()