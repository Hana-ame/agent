#!/usr/bin/env python
"""`bootstrap_runs/` 的回档工具 —— 把每一代产物当成一次提交来浏览与取回。

结构（`tools/oneshot_chain.py` 与 `tools/meta_dag.py` 都会写 HEAD.json）：

```
bootstrap_runs/
  20261018-003215/gen0/{HEAD.json, vea_v6/...}   ← 老 run（顶层）
  meta/s13/gen1/{HEAD.json, vea_v6/, steps/}      ← 元 DAG 的 run（嵌在 meta/ 下）
```

HEAD.json 里有 `parent` / `combined_sha256` / 各文件 sha256 与字节数，
所以这是一条可校验的链，不是散落的目录快照。元 DAG 的 HEAD.json 里还带
`self_verify`（自举冒烟）/ `demo`（自选任务）/ `steps`（8 步快照文件名）——
链上每一环都是通过两道闸门的一代。

## 用法

```bash
python tools/boot_rollback.py list                 # 列出所有 run 与各代
python tools/boot_rollback.py show <run> [gen]     # 看某一代详情（默认最新）
python tools/boot_rollback.py diff <run> <a> <b>   # 比较两代（谁改了哪个文件）
python tools/boot_rollback.py checkout <run> <gen> # 把那一代覆盖回工作区（自动备份当前）
python tools/boot_rollback.py verify <run>         # 校验整条链的父指针与文件 hash
```

run 名可以带子目录，例如 `meta/s13`。

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


def is_run_dir(d: Path) -> bool:
    """这个目录是一个 run：它的子目录里至少有 gen*/HEAD.json。

    `bootstrap_runs/` 下有两种布局：老 run（`oneshot_chain.py` 写的）直接躺在
    顶层；元 DAG 的 run 嵌在 `meta/` 下面（`meta/s13/gen1/...`）。两种都要能
    被 list / verify / diff / checkout 找到。
    """
    return any(c.is_dir() and (c / "HEAD.json").exists() for c in d.iterdir())


def find_runs() -> List[Path]:
    """所有 run 目录（含 `meta/` 下面那一层），按目录名排序。"""
    out: List[Path] = []
    for d in sorted(RUNS.iterdir()):
        if not d.is_dir() or d.name.startswith("_"):
            continue
        if is_run_dir(d):
            out.append(d)
            continue
        for c in sorted(d.iterdir()):
            if c.is_dir() and not c.name.startswith("_") and is_run_dir(c):
                out.append(c)
    return out


def read_head(gen_dir: Path) -> Dict[str, Any]:
    return json.loads((gen_dir / "HEAD.json").read_text(encoding="utf-8"))


def gens_of(run: Path) -> List[Path]:
    out = [d for d in run.iterdir() if d.is_dir() and (d / "HEAD.json").exists()]
    return sorted(out, key=lambda d: read_head(d).get("gen", 0))


def pick_run(name: Optional[str]) -> Path:
    if not RUNS.exists():
        raise SystemExit("还没有任何 run（先跑 tools/oneshot_chain.py 或 tools/meta_dag.py）")
    if name:
        # 支持 "meta/s13" 这种带子目录的写法（Path 会按 / 拼）
        p = RUNS / name
        if not p.exists():
            raise SystemExit(f"没有这个 run: {name}")
        return p
    cands = find_runs()
    if not cands:
        raise SystemExit("run 目录为空")
    return cands[-1]


def rel(run: Path) -> str:
    """run 的显示名：相对 bootstrap_runs/ 的路径（meta 下面的 run 带上前缀）。"""
    return run.relative_to(RUNS).as_posix()


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
    runs = find_runs()
    if not runs:
        print("还没有任何 run")
        return 0
    for run in runs:
        print(f"\n{rel(run)}")
        for g in gens_of(run):
            h = read_head(g)
            flags = []
            if h.get("byte_identical") is not None:
                flags.append("字节一致" if h["byte_identical"] else "字节不同")
            if h.get("self_verify") is not None:
                flags.append("自校验OK" if h["self_verify"] else "自校验FAIL")
            if h.get("demo") is not None:
                flags.append("自选任务OK" if h["demo"] else "自选任务FAIL")
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
    if not gens:
        raise SystemExit(f"{rel(run)} 里没有代")
    print(f"校验 {rel(run)}")
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
    # 🔴 回档必须当场核对：HEAD.json 里存着每个文件的 sha256，不核对就等于
    # 只信"copy 没报错"。旧版只打印一句"复核：fixpoint_v3 --check"当建议，
    # 而那是验不动点，根本不是验"这一代有没有被正确写回工作区"。
    bad = verify_against_head(ROOT, h)
    if bad:
        print("\n回档校验失败（工作区与这一代的记录不符）：")
        for p, got, want in bad:
            print(f"  {p}: 落盘 {got[:16]} 应为 {want[:16]}")
        print(f"备份在 {backup.relative_to(ROOT)}")
        return 1
    print(f"\n已回档到 {run.name}/{g.name}  ({h['combined_sha256'][:16]})")
    print(f"  落盘 {len(h['files'])} 个文件逐个核对 sha256 通过")
    print("复核：python tools/fixpoint_v3.py --check")
    return 0


def verify_against_head(root: Path, h: Dict[str, Any]) -> List[tuple]:
    """核对 root 下这些文件是否与 HEAD 记录的 sha256 相符。

    返回 `[(相对路径, 落盘值, 记录值)]`；空列表 = 全部相符。
    """
    bad: List[tuple] = []
    for p, info in h["files"].items():
        f = root / p
        if not f.exists():
            bad.append((p, "<缺失>", info["sha256"]))
            continue
        got = sha_file(f)
        if got != info["sha256"]:
            bad.append((p, got, info["sha256"]))
    return bad


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
