#!/usr/bin/env python3
"""演示用「模块作者」：在独立的 clone 里创建 / 升级单一功能分支，并 push 到 origin。

每个分支只做一件事：在 modules/<lang>/<name>/ 下新增一个模块。
—— 这正是「各个 branch 实现单一功能，merge 之后作为模块」的工作方式。

用法：
    python3 demo/author_modules.py create <clone 目录>
    python3 demo/author_modules.py bump   <clone 目录> <版本号>
"""
from __future__ import annotations

import json
import os
import subprocess
import sys

MODULES = [
    {"name": "greet-go", "language": "go", "branch": "feat/greet-go", "entry": "module.so"},
    {"name": "greet-js", "language": "js", "branch": "feat/greet-js", "entry": "index.mjs"},
    {"name": "greet-py", "language": "python", "branch": "feat/greet-py", "entry": "impl.py"},
]


def run(args: list[str], cwd: str) -> str:
    proc = subprocess.run(args, cwd=cwd, text=True, capture_output=True)
    if proc.returncode != 0:
        raise SystemExit(f"命令失败: {' '.join(args)}\n{proc.stdout}\n{proc.stderr}")
    return proc.stdout.strip()


def git(clone: str, *args: str) -> str:
    return run(["git", *args], cwd=clone)


def manifest(m: dict, version: str) -> str:
    return json.dumps(
        {
            "name": m["name"],
            "language": m["language"],
            "interface": "bml/module-v1",
            "version": version,
            "entry": m["entry"],
        },
        ensure_ascii=False,
        indent=2,
    ) + "\n"


def files_for(m: dict, version: str) -> dict[str, str]:
    n = version.split(".")[0]  # 1.0.0 -> 1
    path = f"modules/{m['language']}/{m['name']}"
    if m["language"] == "go":
        return {
            f"{path}/module.json": manifest(m, version),
            f"{path}/greet.go": f'''package main

import "example.com/bml/go/sdk"

type mod struct{{}}

func (mod) Name() string {{ return "{m["name"]}" }}

func (mod) Describe() string {{ return "Go 插件模块：生成问候语（v{n}）" }}

func (mod) Run(payload string) (string, error) {{
	return "hello, " + payload + " (go v{n})", nil
}}

// New 是宿主约定的导出符号；必须返回 sdk.Module。
func New() sdk.Module {{ return mod{{}} }}
''',
        }
    if m["language"] == "js":
        return {
            f"{path}/module.json": manifest(m, version),
            f"{path}/index.mjs": f'''// JS 模块：默认导出统一接口 {{ name, describe, run }}
export default {{
  name: "{m["name"]}",
  describe() {{
    return "JS ESM 模块：生成问候语（v{n}）";
  }},
  run(payload) {{
    return `hello, ${{payload}} (js v{n})`;
  }},
}};
''',
        }
    return {
        f"{path}/module.json": manifest(m, version),
        f"{path}/impl.py": f'''"""Python 模块：模块级 NAME / describe() / run(payload)。"""

NAME = "{m["name"]}"


def describe() -> str:
    return "Python 模块：生成问候语（v{n}）"


def run(payload: str) -> str:
    return f"hello, {{payload}} (python v{n})"
''',
    }


def write_and_commit(clone: str, m: dict, version: str, message: str, push: bool) -> None:
    for rel, content in files_for(m, version).items():
        path = os.path.join(clone, rel)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)
    git(clone, "add", "-A")
    git(clone, "commit", "-q", "-m", message)
    if push:
        git(clone, "push", "-q", "-u", "origin", m["branch"])
    print(f"[author] {m['branch']}: {message} → {git(clone, 'rev-parse', '--short', 'HEAD')}")


def create(clone: str) -> None:
    git(clone, "fetch", "-q", "origin")
    for m in MODULES:
        git(clone, "checkout", "-q", "-B", m["branch"], "origin/main")
        write_and_commit(clone, m, "1.0.0", f"feat({m['name']}): 新增单一功能模块 v1", push=True)
    git(clone, "checkout", "-q", "main")  # 作者回到 main，等待下一轮


def bump(clone: str, version: str) -> None:
    n = version.split(".")[0]
    for m in MODULES:
        git(clone, "checkout", "-q", m["branch"])
        git(clone, "pull", "-q", "--ff-only", "origin", m["branch"])
        write_and_commit(clone, m, version, f"feat({m['name']}): 升级到 v{n}", push=True)
    git(clone, "checkout", "-q", "main")


def main() -> int:
    if len(sys.argv) < 3:
        print(__doc__)
        return 2
    cmd, clone = sys.argv[1], sys.argv[2]
    if cmd == "create":
        create(clone)
    elif cmd == "bump":
        if len(sys.argv) < 4:
            print("用法: author_modules.py bump <clone> <version>")
            return 2
        bump(clone, sys.argv[3])
    else:
        print(f"未知命令: {cmd}")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
