#!/usr/bin/env python3
"""输出一个目录（模块）的结构指纹 JSON，供 v8 command 节点调用。

只读、确定性：同一份代码得到同一个 `hash`；改了文件 `hash` 就变。
越过 `__pycache__` / `.git` / `data` / `node_modules` 等目录。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

IGNORE = {
    "__pycache__", ".git", ".venv", "venv", "node_modules",
    "data", "out", ".pytest_cache", ".mypy_cache", ".ruff_cache",
}

DEF_RE = re.compile(r"(?m)^\s*(?:async\s+)?def\s+\w+|^\s*class\s+\w+")
IMPORT_RE = re.compile(r"(?m)^\s*(?:from\s+([\w.]+)\s+import|import\s+([\w.]+))")


def inspect_module(root: Path) -> dict:
    files = []
    for p in root.rglob("*"):
        if not p.is_file():
            continue
        rel = p.relative_to(root)
        if any(part in IGNORE for part in rel.parts):
            continue
        if p.suffix == ".pyc":
            continue
        files.append(p)
    files.sort()

    digest = hashlib.sha256()
    loc = py_files = defs = 0
    imports: set[str] = set()
    for p in files:
        data = p.read_bytes()
        digest.update(p.relative_to(root).as_posix().encode())
        digest.update(data)
        if p.suffix == ".py":
            py_files += 1
            text = data.decode("utf-8", "replace")
            for line in text.splitlines():
                s = line.strip()
                if s and not s.startswith("#"):
                    loc += 1
            defs += len(DEF_RE.findall(text))
            for m in IMPORT_RE.finditer(text):
                imports.add(m.group(1) or m.group(2))

    return {
        "module": root.name,
        "path": str(root),
        "files": len(files),
        "py_files": py_files,
        "loc": loc,
        "defs": defs,
        "hash": digest.hexdigest()[:16],
        "imports": sorted(imports)[:30],
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="模块结构指纹")
    ap.add_argument("path")
    args = ap.parse_args(argv)
    root = Path(args.path).resolve()
    if not root.is_dir():
        print(f"不是目录：{root}", file=sys.stderr)
        return 2
    print(json.dumps(inspect_module(root), ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
