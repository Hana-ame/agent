#!/usr/bin/env python3
"""Python 语言的模块宿主。

扫描 modules/python/*/module.json → importlib 按文件路径加载 → 调用统一接口
NAME / describe() / run(payload)。--watch 时轮询 mtime，变更后重新加载并执行。

零第三方依赖。
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
LAB_ROOT = os.path.dirname(HERE)


def short(sha: str | None) -> str:
    return (sha or "?")[:7]


def read_lock(root: str) -> dict[str, dict]:
    path = os.path.join(root, "modules.lock.json")
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        return {m["name"]: m for m in data.get("modules", [])}
    except (OSError, ValueError):
        return {}


def modules_dir(root: str) -> str:
    return os.path.join(root, "modules", "python")


def discover(root: str) -> list[str]:
    base = modules_dir(root)
    if not os.path.isdir(base):
        return []
    out = []
    for name in sorted(os.listdir(base)):
        d = os.path.join(base, name)
        if os.path.isdir(d) and os.path.isfile(os.path.join(d, "module.json")):
            out.append(d)
    return out


def entry_of(d: str) -> str:
    with open(os.path.join(d, "module.json"), encoding="utf-8") as f:
        manifest = json.load(f)
    return os.path.join(d, manifest.get("entry", "impl.py"))


def load_one(d: str) -> tuple[dict, object]:
    """按文件路径加载模块，模块名唯一，不与任何已安装包冲突。"""
    with open(os.path.join(d, "module.json"), encoding="utf-8") as f:
        manifest = json.load(f)
    entry = entry_of(d)
    mod_name = f"bml_module_{manifest.get('name') or os.path.basename(d)}"
    spec = importlib.util.spec_from_file_location(mod_name, entry)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"无法为 {entry} 创建 import spec")
    module = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = module  # 支持模块内部相对 import
    spec.loader.exec_module(module)
    return manifest, module


def describe_and_run(root: str, d: str, lock: dict[str, dict], payload: str) -> None:
    manifest, module = load_one(d)
    name = getattr(module, "NAME", manifest.get("name", os.path.basename(d)))
    info = lock.get(name, {})
    describe = getattr(module, "describe", lambda: "")
    print(
        f"[python] {name} v{manifest.get('version', '?')} commit={short(info.get('commit'))} "
        f"| {describe()}"
    )
    run = getattr(module, "run", None)
    if run is None:
        raise RuntimeError(f"{name} 缺少 run(payload) 函数")
    print(f"[python]   run({payload!r}) => {run(payload)}")


def snapshot(root: str) -> dict[str, str]:
    out = {}
    for d in discover(root):
        entry = entry_of(d)
        try:
            st = os.stat(entry)
            out[d] = f"{st.st_mtime_ns}:{st.st_size}"
        except OSError:
            out[d] = "missing"
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description="Python 模块宿主")
    parser.add_argument("--root", default=LAB_ROOT, help="仓库根目录")
    parser.add_argument("--payload", default="world", help="传给模块 run() 的输入")
    parser.add_argument("--watch", action="store_true", help="监听模块文件变更并热重载")
    args = parser.parse_args()
    root = os.path.abspath(args.root)

    dirs = discover(root)
    if not dirs:
        print("[python] 没有发现 Python 模块（先运行：python3 sync.py once）")
        return 0
    print(f"[python] 发现 {len(dirs)} 个 Python 模块")
    lock = read_lock(root)
    for d in dirs:
        try:
            describe_and_run(root, d, lock, args.payload)
        except Exception as err:  # noqa: BLE001 - 演示宿主：单个模块失败不拖垮整体
            print(f"[python] ✗ {os.path.basename(d)} 加载失败: {err}")
    if not args.watch:
        return 0

    print("[python] watch 模式：文件变更后自动重新加载（无需重启）")
    prev = snapshot(root)
    try:
        while True:
            time.sleep(0.5)
            cur = snapshot(root)
            for d, sig in cur.items():
                if prev.get(d) != sig:
                    print(f"[python] 检测到模块更新: {os.path.basename(d)}，重新加载")
                    try:
                        describe_and_run(root, d, read_lock(root), args.payload)
                    except Exception as err:  # noqa: BLE001
                        print(f"[python] ✗ 重载 {os.path.basename(d)} 失败: {err}")
            prev = cur
    except KeyboardInterrupt:
        print("\n[python] watch 退出")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
