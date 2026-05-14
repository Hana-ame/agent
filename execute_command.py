# coding: utf-8
import subprocess
import platform
from typing import Optional

def _shell_path(
    shell_path: Optional[str] = None
):
    if shell_path:
        return shell_path
    
    # Set platform‑specific default shell if not provided
    if platform.system() == "Windows":
        # Default to Git Bash on Windows
        return r'C:\Program Files\Git\usr\bin\bash.exe'
    else:
        # On Unix-like systems, use the standard Bourne shell
        return '/bin/bash'

def run_shell_command(
    command: str,
    shell_path: Optional[str] = None
) -> subprocess.CompletedProcess:
    r"""
    Execute a string as a shell command.

    On Linux/macOS, the default system shell is used.
    On Windows, if no shell_path is provided, it defaults to
    C:\Program Files\Git\usr\bin\bash.exe (Git Bash).

    Args:
        command (str): The shell command to run.
        shell_path (str, optional): Path to the shell executable.
            If None, platform‑specific default is chosen.

    Returns:
        subprocess.CompletedProcess: An object containing return code,
                                     stdout, and stderr.

    Raises:
        FileNotFoundError: If the specified shell_path does not exist.
        subprocess.CalledProcessError: If check=True is used (see note).
    """
    

    # Run the command with shell=True and the chosen executable
    result = subprocess.run(
        command,
        shell=True,
        executable=_shell_path(shell_path=shell_path),
        capture_output=True,
        text=True
    )
    return result

# Example usage:
if __name__ == "__main__":
    # Linux/macOS example
    if platform.system() != "Windows":
        out = run_shell_command("echo 'Hello from Linux'")
        print(out)

    # Windows example (using Git Bash)
    else:
        # Use forward slashes for paths inside the command (Bash style)
        out = run_shell_command("echo 'Hello from Windows Git Bash'")
        print(out)

    cmd = r"""
cat > /tmp/myfile.txt << 'EOF'
This is line 1
This is line 2
This is line 3
EOF
    """
    result = run_shell_command(cmd)
    print(result.stdout)          # usually empty if successful
    print(result.stderr)          # check for errors
        
if __name__ == "__main__":
    result = run_shell_command("python3 -m unittest test.py -v")
    print(result.stdout)
    print(result.stderr)