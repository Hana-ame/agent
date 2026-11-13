#!/usr/bin/env python3
"""同步器：把「单一功能分支」自动合并成 main 上的「模块」，并刷新 lock。

核心循环（幂等）：
    git fetch                      # 拉取各 feature 分支的最新提交
    git merge --no-ff origin/<b>   # 每个分支只改 modules/<lang>/<name>/，互不冲突
    go build -buildmode=plugin ... # Go 模块需要编译成 module.so（JS/Python 直接加载源码）
    写 modules.lock.json           # 记录每个模块的 branch / commit / version / artifact

命令：
    python3 sync.py status [--json]         # 查看模块与分支的同步状态
    python3 sync.py once [--no-commit]      # 同步一次（合并 + 构建 + lock）
    python3 sync.py watch [--interval 2]    # 轮询远端分支，有变化就自动同步

前提：当前在 main 分支、已跟踪文件无改动（sync 自己会提交 lock）。
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone

ROOT = os.path.dirname(os.path.abspath(__file__))
INTERFACE = "bml/module-v1"


# ---------------------------------------------------------------- git 基础封装
def run(args: list[str], cwd: str = ROOT, check: bool = True) -> subprocess.CompletedProcess:
    proc = subprocess.run(args, cwd=cwd, text=True, capture_output=True)
    if check and proc.returncode != 0:
        raise SystemExit(f"[sync] 命令失败: {' '.join(args)}\n{proc.stdout}\n{proc.stderr}")
    return proc


def git(*args: str, check: bool = True) -> subprocess.CompletedProcess:
    return run(["git", *args], check=check)


def git_ok(*args: str) -> bool:
    return git(*args, check=False).returncode == 0


# ---------------------------------------------------------------- 配置 / 状态
def load_json(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def load_spec() -> dict:
    return load_json(os.path.join(ROOT, "modules.json"))


def load_lock() -> dict:
    path = os.path.join(ROOT, "modules.lock.json")
    if not os.path.isfile(path):
        return {"interface": INTERFACE, "generated_at": None, "modules": []}
    return load_json(path)


def save_lock(lock: dict) -> None:
    with open(os.path.join(ROOT, "modules.lock.json"), "w", encoding="utf-8") as f:
        json.dump(lock, f, ensure_ascii=False, indent=2)
        f.write("\n")


def branch_of_head() -> str:
    return git("rev-parse", "--abbrev-ref", "HEAD").stdout.strip()


def ensure_ready(spec: dict) -> None:
    if branch_of_head() != "main":
        raise SystemExit("[sync] 必须在 main 分支上运行（main 只做集成）")
    dirty = git("status", "--porcelain", "--untracked-files=no").stdout.strip()
    if dirty:
        raise SystemExit(f"[sync] 已跟踪文件有未提交改动，先处理：\n{dirty}")


def remote_ref(spec: dict, module: dict) -> str:
    return f"{spec.get('remote', 'origin')}/{module['branch']}"


def remote_head(spec: dict, module: dict) -> str | None:
    proc = git("rev-parse", "--verify", remote_ref(spec, module), check=False)
    return proc.stdout.strip() if proc.returncode == 0 else None


# ---------------------------------------------------------------- 构建
def newest_source_mtime(module_dir: str) -> float:
    newest = 0.0
    for base, dirs, files in os.walk(module_dir):
        dirs[:] = [d for d in dirs if d not in {"__pycache__", ".git"}]
        for name in files:
            if name.endswith((".go", ".mod", ".sum")):
                newest = max(newest, os.path.getmtime(os.path.join(base, name)))
    return newest


def needs_go_build(module_dir: str, artifact: str) -> bool:
    if not os.path.isfile(artifact):
        return True
    return newest_source_mtime(module_dir) > os.path.getmtime(artifact)


def build_modules(spec: dict) -> None:
    for m in spec["modules"]:
        if m["language"] != "go":
            continue
        module_dir = os.path.join(ROOT, "modules", "go", m["name"])
        if not os.path.isdir(module_dir):
            print(f"[sync] 跳过构建 {m['name']}：目录还没合并进来")
            continue
        artifact = os.path.join(module_dir, "module.so")
        if not needs_go_build(module_dir, artifact):
            continue
        rel = os.path.relpath(module_dir, ROOT)
        tmp = artifact + ".tmp"
        print(f"[sync] go build -buildmode=plugin ./{rel}")
        proc = run(["go", "build", "-buildmode=plugin", "-o", tmp, f"./{rel}"], check=False)
        if proc.returncode != 0:
            if os.path.exists(tmp):
                os.unlink(tmp)
            raise SystemExit(f"[sync] ✗ 构建 {m['name']} 失败:\n{proc.stdout}\n{proc.stderr}")
        # 原子替换：正在运行的宿主 mmap 的是旧 inode，不会被写坏
        os.replace(tmp, artifact)
        print(f"[sync] built {os.path.relpath(artifact, ROOT)}")


# ---------------------------------------------------------------- 同步一次
def sync_once(commit: bool = True, verbose: bool = True) -> bool:
    spec = load_spec()
    ensure_ready(spec)
    remote = spec.get("remote", "origin")
    print(f"[sync] fetch {remote} …")
    git("fetch", remote, "--prune")

    merged_any = False
    for m in spec["modules"]:
        ref = remote_ref(spec, m)
        if not git_ok("rev-parse", "--verify", ref):
            print(f"[sync] ✗ 分支不存在，跳过：{ref}")
            continue
        msg = f"merge module {m['name']} from {m['branch']}"
        proc = git("merge", "--no-ff", "--no-edit", "-m", msg, ref, check=False)
        if proc.returncode != 0:
            git("merge", "--abort", check=False)
            raise SystemExit(f"[sync] ✗ 合并 {ref} 冲突，已回滚：\n{proc.stdout}\n{proc.stderr}")
        first = (proc.stdout.strip().splitlines() or [""])[0]
        if "Already up to date" in proc.stdout:
            if verbose:
                print(f"[sync] = {m['name']} 已是最新（{ref}）")
        else:
            merged_any = True
            print(f"[sync] ✔ 合并 {ref} → {first}")

    build_modules(spec)

    # 生成 lock：只有模块集合 / commit / version 真的变了才更新时间戳并提交
    old_lock = load_lock()
    modules = []
    for m in spec["modules"]:
        head = remote_head(spec, m)
        module_dir = os.path.join(ROOT, "modules", m["language"], m["name"])
        manifest = {}
        manifest_path = os.path.join(module_dir, "module.json")
        if os.path.isfile(manifest_path):
            manifest = load_json(manifest_path)
        modules.append(
            {
                "name": m["name"],
                "language": m["language"],
                "branch": m["branch"],
                "commit": head,
                "version": manifest.get("version"),
                "artifact": os.path.relpath(module_dir, ROOT) if os.path.isdir(module_dir) else None,
            }
        )
    modules.sort(key=lambda x: (x["language"], x["name"]))
    same = [(x["name"], x["commit"], x["version"], x["artifact"]) for x in modules] == [
        (x["name"], x["commit"], x["version"], x["artifact"]) for x in old_lock.get("modules", [])
    ]
    changed = merged_any or not same
    if not same:
        save_lock(
            {
                "interface": INTERFACE,
                "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "remote": remote,
                "modules": modules,
            }
        )
        changed = True

    if changed and commit:
        git("add", "modules.lock.json")
        staged = git("diff", "--cached", "--quiet", check=False).returncode != 0
        if staged:
            git("commit", "-m", "chore(sync): refresh modules.lock.json")
            print("[sync] 已提交 modules.lock.json")
    elif not changed and verbose:
        print("[sync] 无变化")
    return changed


# ---------------------------------------------------------------- status
def status(as_json: bool = False) -> int:
    spec = load_spec()
    lock = load_lock()
    lock_by_name = {m["name"]: m for m in lock.get("modules", [])}
    rows = []
    for m in spec["modules"]:
        ref = remote_ref(spec, m)
        head = remote_head(spec, m)
        merged = lock_by_name.get(m["name"], {}).get("commit")
        if head is None:
            state = "分支不存在"
        elif merged == head:
            state = "已同步"
        elif merged is None:
            state = "未合并"
        else:
            state = "有更新待合并"
        rows.append(
            {
                "name": m["name"],
                "language": m["language"],
                "branch": m["branch"],
                "merged_commit": merged,
                "remote_head": head,
                "state": state,
            }
        )
    if as_json:
        print(json.dumps({"interface": lock.get("interface", INTERFACE), "modules": rows}, ensure_ascii=False, indent=2))
        return 0
    print(f"[sync] 接口契约: {lock.get('interface', INTERFACE)}")
    print(f"[sync] {'模块':<12}{'语言':<8}{'分支':<22}{'已合并':<10}{'远端HEAD':<10}状态")
    for r in rows:
        print(
            f"[sync] {r['name']:<12}{r['language']:<8}{r['branch']:<22}"
            f"{(r['merged_commit'] or '-')[:7]:<10}{(r['remote_head'] or '-')[:7]:<10}{r['state']}"
        )
    return 0


# ---------------------------------------------------------------- watch
def watch(interval: float) -> int:
    spec = load_spec()
    print(f"[sync] watch 模式：每 {interval:g}s 检查一次远端分支（Ctrl-C 退出）")
    tick = 0
    try:
        while True:
            if branch_of_head() != "main":
                raise SystemExit("[sync] watch 必须在 main 分支上运行")
            git("fetch", spec.get("remote", "origin"), "--prune", check=False)
            lock = load_lock()
            lock_by_name = {m["name"]: m.get("commit") for m in lock.get("modules", [])}
            pending = []
            for m in spec["modules"]:
                head = remote_head(spec, m)
                if head is None:
                    continue
                if lock_by_name.get(m["name"]) != head:
                    pending.append(f"{m['name']}@{head[:7]}")
            if pending:
                print(f"[sync] 检测到分支更新: {', '.join(pending)} → 自动同步")
                sync_once(commit=True, verbose=False)
            elif tick % 20 == 0:
                print(f"[sync] 无变化（{len(spec['modules'])} 个模块分支）")
            tick += 1
            time.sleep(interval)
    except KeyboardInterrupt:
        print("\n[sync] watch 退出")
    return 0


def main() -> int:
    global ROOT  # noqa: PLW0603 - 允许 --root 覆盖默认的脚本目录
    parser = argparse.ArgumentParser(description="branch → module 同步器")
    parser.add_argument("--root", default=ROOT, help="仓库根目录（默认脚本所在目录）")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_status = sub.add_parser("status", help="查看同步状态")
    p_status.add_argument("--json", action="store_true", help="输出 JSON")

    p_once = sub.add_parser("once", help="同步一次")
    p_once.add_argument("--no-commit", action="store_true", help="不自动提交 lock")

    p_watch = sub.add_parser("watch", help="持续轮询并自动同步")
    p_watch.add_argument("--interval", type=float, default=2.0, help="轮询间隔（秒）")

    args = parser.parse_args()
    ROOT = os.path.abspath(args.root)

    if args.cmd == "status":
        return status(as_json=args.json)
    if args.cmd == "once":
        sync_once(commit=not args.no_commit)
        return 0
    if args.cmd == "watch":
        return watch(args.interval)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
