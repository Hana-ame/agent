#!/usr/bin/env python
"""`bootstrap_runs/` 的回档工具 —— 把每一代产物当成一次提交来浏览与取回。

结构（`tools/oneshot_chain.py` 写入）：

```
bootstrap_runs/
  <run-id>/
    gen0/{HEAD.json, vea_v6/...}    ← 父提交
    gen1/{HEAD.json, vea_v6/...}    ← parent = gen0 的 combined_sha256
    gen2/...
```

HEAD.json 里有 `parent` / `combined_sha256` / 各文件 sha256 与字节数，
所以这是一条可校验的链，不是散落的目录快照。

## 用法

```bash
python tools/boot_rollback.py list                 # 列出所有 run 与各代
python tools/boot_rollback.py show <run> [gen]     # 看某一代详情（默认最新）
python tools/boot_rollback.py diff <run> <a> <b>   # 比较两代（谁改了哪个文件）
python tools/boot_rollback.py checkout <run> <gen> # 把那一代覆盖回工作区（自动备份当前）
python tools/boot_rollback.py verify <run>         # 校验整条链的父指针与文件 hash
```

`checkout` **不删除任何东西**：它先把当前 `vea_v6/` 整体复制到
`bootstrap_runs/_before_checkout_<时间戳>/`，再写入目标代。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parent.parent
RUNS = ROOT / "bootstrap_runs"
FRAME_DIRS = ("vea_v6",)


def sha_file(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def read_head(gen_dir: Path) -> Dict[str, Any]:
    return json.loads((gen_dir / "HEAD.json").read_text(encoding="utf-8"))


def gens_of(run: Path) -> List[Path]:
    out = [d for d in run.iterdir() if d.is_dir() and (d / "HEAD.json").exists()]
    return sorted(out, key=lambda d: read_head(d).get("gen", 0))


def pick_run(name: Optional[str]) -> Path:
    if not RUNS.exists():
        raise SystemExit("还没有任何 run（先跑 tools/oneshot_chain.py）")
    if name:
        p = RUNS / name
        if not p.exists():
            raise SystemExit(f"没有这个 run: {name}")
        return p
    cands = sorted([d for d in RUNS.iterdir() if d.is_dir() and not d.name.startswith("_")],
                   key=lambda d: d.stat().st_mtime)
    if not cands:
        raise SystemExit("run 目录为空")
    return cands[-1]


def pick_gen(run: Path, which: Optional[str]) -> Path:
    gens = gens_of(run)
    if not gens:
        raise SystemExit(f"{run.name} 里没有代")
    if which is None:
        return gens[-1]
    for g in gens:
        if g.name == which or g.name == f"gen{which}":
            return g
    raise SystemExit(f"{run.name} 里没有 {which}")


def cmd_list(_args: argparse.Namespace) -> int:
    if not RUNS.exists():
        print("还没有任何 run")
        return 0
    runs = sorted([d for d in RUNS.iterdir() if d.is_dir() and not d.name.startswith("_")])
    if not runs:
        print("还没有任何 run")
        return 0
    for run in runs:
        print(f"\n{run.name}")
        for g in gens_of(run):
            h = read_head(g)
            flags = []
            if h.get("byte_identical") is not None:
                flags.append("字节一致" if h["byte_identical"] else "字节不同")
            if h.get("self_verify") is not None:
                flags.append("自校验OK" if h["self_verify"] else "自校验FAIL")
            print(f"  {g.name:<6} {h.get('combined_sha256','')[:16]}  "
                  f"{h.get('total_bytes',0):>7}B  parent={str(h.get('parent'))[:16]:<16} "
                  f"{' '.join(flags)}")
    return 0


def cmd_show(args: argparse.Namespace) -> int:
    run = pick_run(args.run)
    g = pick_gen(run, args.gen)
    h = read_head(g)
    print(f"{run.name}/{g.name}\n" + json.dumps(h, ensure_ascii=False, indent=2))
    return 0


def cmd_diff(args: argparse.Namespace) -> int:
    run = pick_run(args.run)
    a = pick_gen(run, args.a)
    b = pick_gen(run, args.b)
    ha, hb = read_head(a), read_head(b)
    if ha["combined_sha256"] == hb["combined_sha256"]:
        print(f"{a.name} 与 {b.name} 整份框架完全相同")
        return 0
    print(f"{a.name} -> {b.name}")
    for p in sorted(set(ha["files"]) | set(hb["files"])):
        sa = ha["files"].get(p, {}).get("sha256", "-")
        sb = hb["files"].get(p, {}).get("sha256", "-")
        if sa != sb:
            ba = ha["files"].get(p, {}).get("bytes", 0)
            bb = hb["files"].get(p, {}).get("bytes", 0)
            print(f"  M {p}  {ba}B -> {bb}B  {sa[:12]} -> {sb[:12]}")
    return 0


def cmd_verify(args: argparse.Namespace) -> int:
    run = pick_run(args.run)
    gens = gens_of(run)
    bad = 0
    prev = None
    for g in gens:
        h = read_head(g)
        # 文件 hash 是否与 HEAD 记录一致
        for p, info in h["files"].items():
            f = g / p
            if not f.exists():
                print(f"  MISSING {g.name}: {p}")
                bad += 1
                continue
            if sha_file(f) != info["sha256"]:
                print(f"  HASH    {g.name}: {p}")
                bad += 1
        if prev is not None and h.get("parent") not in (None, prev):
            print(f"  PARENT  {g.name}: parent={str(h.get('parent'))[:16]} 但上一代是 {prev[:16]}")
            bad += 1
        prev = h["combined_sha256"]
        print(f"  {g.name:<6} {h['combined_sha256'][:16]}  ok")
    print("\n链完整" if not bad else f"\n发现 {bad} 处问题")
    return 0 if not bad else 1


def cmd_checkout(args: argparse.Namespace) -> int:
    run = pick_run(args.run)
    g = pick_gen(run, args.gen)
    h = read_head(g)

    backup = RUNS / f"_before_checkout_{time.strftime('%Y%m%d-%H%M%S')}"
    backup.mkdir(parents=True, exist_ok=False)
    for d in FRAME_DIRS:
        if (ROOT / d).exists():
            shutil.copytree(ROOT / d, backup / d)
    print(f"已备份当前工作区到 {backup.relative_to(ROOT)}")

    for p in h["files"]:
        src = g / p
        dst = ROOT / p
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, dst)
        print(f"  恢复 {p}")
    print(f"\n已回档到 {run.name}/{g.name}  ({h['combined_sha256'][:16]})")
    print("复核：python tools/fixpoint_v3.py --check")
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="bootstrap_runs 的 git 式回档")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("list", help="列出所有 run 与各代")
    p.set_defaults(func=cmd_list)

    p = sub.add_parser("show", help="查看某一代详情")
    p.add_argument("run", nargs="?", default=None)
    p.add_argument("gen", nargs="?", default=None)
    p.set_defaults(func=cmd_show)

    p = sub.add_parser("diff", help="比较同一 run 的两代")
    p.add_argument("run", nargs="?", default=None)
    p.add_argument("a", nargs="?", default="0")
    p.add_argument("b", nargs="?", default=None)
    p.set_defaults(func=cmd_diff)

    p = sub.add_parser("verify", help="校验整条链")
    p.add_argument("run", nargs="?", default=None)
    p.set_defaults(func=cmd_verify)

    p = sub.add_parser("checkout", help="回档某一代到工作区（自动备份）")
    p.add_argument("run", nargs="?", default=None)
    p.add_argument("gen", nargs="?", default=None)
    p.set_defaults(func=cmd_checkout)

    args = ap.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
