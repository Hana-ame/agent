# ```python
#!/usr/bin/env python3
"""
utils.py - Utility script for file and git operations.

Usage:
    python utils.py get-file <file_path>
        Read the content of <file_path> and write it to .MESSAGE.txt

    python utils.py get-tree
        Generate a directory tree of the current directory (like 'tree' command)
        and write it to MESSAGE.txt

    python utils.py commit <commit_message>
        Stage all changes (git add .) and commit with the given message.
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path


def write_file_content(file_path: str, output_file: str = ".MESSAGE.txt") -> None:
    """
    Read the content of file_path and write it to output_file.
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as src:
            content = src.read()
    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found.", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error reading file '{file_path}': {e}", file=sys.stderr)
        sys.exit(1)

    try:
        with open(output_file, 'w', encoding='utf-8') as dst:
            dst.write(content)
        print(f"Content written to {output_file}")
    except Exception as e:
        print(f"Error writing to {output_file}: {e}", file=sys.stderr)
        sys.exit(1)


def generate_tree(start_path: Path = Path("."), prefix: str = "", output_lines: list = None) -> None:
    """
    Recursively generate a tree-like listing of the directory structure.
    """
    if output_lines is None:
        output_lines = []
        # Add the root directory name (current directory's name or '.')
        root_name = start_path.resolve().name or '.'
        output_lines.append(root_name)

    # Get all entries, sort: directories first, then files, both alphabetically
    try:
        entries = list(start_path.iterdir())
    except PermissionError:
        # Skip directories we can't read
        return

    # Separate directories and files
    dirs = sorted([e for e in entries if e.is_dir()])
    files = sorted([e for e in entries if e.is_file()])

    # Combine: all directories then all files
    sorted_entries = dirs + files
    total = len(sorted_entries)

    for idx, entry in enumerate(sorted_entries):
        is_last = (idx == total - 1)

        # Choose the connector based on whether it's the last item
        connector = "������ " if is_last else "������ "
        output_lines.append(prefix + connector + entry.name)

        # If it's a directory, recurse into it
        if entry.is_dir():
            # New prefix for children: add "    " if last, else "��   "
            new_prefix = prefix + ("    " if is_last else "��   ")
            generate_tree(entry, new_prefix, output_lines)


def write_tree(output_file: str = "MESSAGE.txt") -> None:
    """
    Generate a tree of the current directory and write it to output_file.
    """
    output_lines = []
    generate_tree(Path("."), output_lines=output_lines)
    tree_str = "\n".join(output_lines)

    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(tree_str)
        print(f"Directory tree written to {output_file}")
    except Exception as e:
        print(f"Error writing to {output_file}: {e}", file=sys.stderr)
        sys.exit(1)


def git_commit(message: str) -> None:
    """
    Stage all changes (git add .) and commit with the given message.
    """
    # Check if we're inside a git repository
    try:
        subprocess.run(
            ["git", "rev-parse", "--is-inside-work-tree"],
            check=True,
            capture_output=True,
            text=True,
        )
    except subprocess.CalledProcessError:
        print("Error: Not inside a git repository.", file=sys.stderr)
        sys.exit(1)
    except FileNotFoundError:
        print("Error: git command not found. Please install git.", file=sys.stderr)
        sys.exit(1)

    # Stage all changes
    try:
        subprocess.run(["git", "add", "."], check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as e:
        print(f"Error during 'git add .': {e.stderr}", file=sys.stderr)
        sys.exit(1)

    # Commit with the provided message
    try:
        result = subprocess.run(
            ["git", "commit", "-m", message],
            check=True,
            capture_output=True,
            text=True,
        )
        print(result.stdout)
        if result.stderr:
            print(result.stderr, file=sys.stderr)
    except subprocess.CalledProcessError as e:
        print(f"Error during commit: {e.stderr}", file=sys.stderr)
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description="Utility script for file and git operations."
    )
    subparsers = parser.add_subparsers(dest="command", required=True, help="Subcommand to run")

    # get-file command
    parser_getfile = subparsers.add_parser("get-file", help="Read a file and write to .MESSAGE.txt")
    parser_getfile.add_argument("file_path", help="Path to the file to read")

    # get-tree command
    parser_gettree = subparsers.add_parser("get-tree", help="Generate directory tree and write to MESSAGE.txt")

    # commit command
    parser_commit = subparsers.add_parser("commit", help="Stage all and commit with a message")
    parser_commit.add_argument("commit_message", help="Commit message")

    args = parser.parse_args()

    if args.command == "get-file":
        write_file_content(args.file_path)
    elif args.command == "get-tree":
        write_tree()
    elif args.command == "commit":
        git_commit(args.commit_message)
    else:
        # This should not happen due to required=True
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
# ```