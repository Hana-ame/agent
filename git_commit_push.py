#!/usr/bin/env python3
"""
git_commit_push.py - 根据 Board 666 no.190229 指令执行
指令: "commit/push当前文件夹" + "需要所有内容"

执行步骤:
1. git add 所有修改
2. git commit
3. git push
"""

import subprocess
import sys
import os
from datetime import datetime, timezone

WORKDIR = "/mnt/d/WorkPlace/simpleAI"

def run_cmd(cmd, cwd=WORKDIR, capture=True, timeout=None):
    """Run a shell command and return output."""
    print(f"$ {cmd}", flush=True)
    if capture:
        result = subprocess.run(
            cmd, shell=True, cwd=cwd, capture_output=True, text=True, timeout=timeout
        )
    else:
        result = subprocess.run(
            cmd, shell=True, cwd=cwd, timeout=timeout
        )
    if result.returncode != 0:
        print(f"  ERROR (rc={result.returncode}): {result.stderr.strip()}", flush=True)
        return False, result.stdout, result.stderr
    if capture:
        print(f"  OK: {result.stdout.strip()[:200]}", flush=True)
    else:
        print(f"  OK (rc=0)", flush=True)
    return True, result.stdout, result.stderr


def main():
    print("=" * 60)
    print(f"git_commit_push.py - 执行时间: {datetime.now(timezone.utc).isoformat()}")
    print("=" * 60)

    # Step 0: Show status before
    print("\n[Step 0] 当前 Git 状态:")
    ok, out, _ = run_cmd("git status --short")
    if ok:
        print(f"  Changes:\n{out}")

    # Show diff summary
    ok, out, _ = run_cmd("git diff --stat")
    if ok:
        print(f"  Diff stat:\n{out}")

    # Step 1: git add all changes
    print("\n[Step 1] git add -A")
    ok, out, err = run_cmd("git add -A")
    if not ok:
        print("FATAL: git add failed", flush=True)
        sys.exit(1)

    # Verify what's staged
    ok, out, _ = run_cmd("git diff --cached --stat")
    if ok:
        print(f"  Staged:\n{out}")

    # Step 2: git commit
    print("\n[Step 2] git commit")
    commit_msg = f"Auto666: commit/push 当前文件夹所有内容 (Board 666 no.190229)"
    ok, out, err = run_cmd(f'git commit -m "{commit_msg}"')
    if not ok:
        if "nothing to commit" in err.lower() or "nothing to commit" in out.lower():
            print("  Nothing to commit, skipping commit.", flush=True)
        else:
            print(f"FATAL: git commit failed", flush=True)
            sys.exit(1)

    # Step 3: git push
    print("\n[Step 3] git push to origin/cell-agent")
    ok, out, err = run_cmd("git push origin cell-agent", timeout=60000)
    if not ok:
        print(f"FATAL: git push failed", flush=True)
        sys.exit(1)

    # Step 4: Verify final state
    print("\n[Step 4] 验证最终状态")
    run_cmd("git status")
    run_cmd("git log --oneline -3")

    print("\n" + "=" * 60)
    print("✅ 完成: 已成功 commit + push 所有内容")
    print("=" * 60)


if __name__ == "__main__":
    main()
