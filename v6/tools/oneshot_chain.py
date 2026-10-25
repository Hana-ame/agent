#!/usr/bin/env python
"""一次 shot 连续自举 N 代框架，每代落盘 + git 式回档点。

## 判据（用户定义，不可放宽）

一次 shot（同一条提示词、不人工干预）必须连续输出 **至少三代** 完全一样的整个
`vea_v6` 框架：M0 由真模型生成，M1 由 M0 自己的 `generate_framework()` 重建，
M2 由 M1 重建 …… 每一代都与上一代 **逐字节相同**，且每一代都能独立通过完整自校验
（清单 sha / bytes 逐项吻合 + 二代复现）。

## 用法

```bash
# 用 laptop-6.8（本地 8000），连跑 3 代
python tools/oneshot_chain.py --generations 3

# laptop 卡住时换 model、不换 api
python tools/oneshot_chain.py --generations 3 --model-key sensenova-6.8

# 不调模型，只验"当前磁盘框架能否连续自举 N 代"（离线闭环检查）
python tools/oneshot_chain.py --generations 3 --offline
```

每代产物写进 `bootstrap_runs/<run-id>/gen<N>/`，并带一个 `HEAD.json` 记录父代、
提示词、模型、各文件 sha256 与合并 sha，构成可回档的提交链。
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
sys.path.insert(0, str(ROOT))

RUNS = ROOT / "bootstrap_runs"


def sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def combined_sha(files: Dict[str, str]) -> str:
    """整份框架的合并指纹：按路径排序后逐文件 hash，再整体 hash。"""
    h = hashlib.sha256()
    for p in sorted(files):
        h.update(p.encode("utf-8"))
        h.update(b"\0")
        h.update(hashlib.sha256(files[p].encode("utf-8")).digest())
    return h.hexdigest()


def snapshot(files: Dict[str, str], dest: Path, meta: Dict[str, Any]) -> Dict[str, Any]:
    """把一代产物落盘，并写 HEAD.json 作为可回档的"提交"。"""
    dest.mkdir(parents=True, exist_ok=True)
    for path, text in files.items():
        target = dest / path
        target.parent.mkdir(parents=True, exist_ok=True)
        # 显式写 \n 并指定 newline，避免 Windows 上被转成 CRLF 污染字节数
        target.write_text(text, encoding="utf-8", newline="")
    head = dict(meta)
    head["files"] = {p: {"sha256": sha(t), "bytes": len(t.encode("utf-8"))} for p, t in sorted(files.items())}
    head["combined_sha256"] = combined_sha(files)
    head["total_bytes"] = sum(len(t.encode("utf-8")) for t in files.values())
    (dest / "HEAD.json").write_text(
        json.dumps(head, ensure_ascii=False, indent=2), encoding="utf-8", newline=""
    )
    return head


def verify_local(files: Dict[str, str]) -> tuple[bool, str]:
    """对一份内存中的框架做完整自校验（不依赖磁盘）。"""
    from vea_v6.parts.bootstrap import MAIN_SOURCE, _probe

    try:
        ns = _probe(files[MAIN_SOURCE])
        return ns["verify_self_reproduction"]()
    except Exception as exc:  # noqa: BLE001 - 校验跑不起来就是不闭合
        return False, f"校验执行失败: {exc}"


def rebuild(files: Dict[str, str]) -> Dict[str, str]:
    """用这一代的 `generate_framework()` 重建下一代。"""
    from vea_v6.parts.bootstrap import MAIN_SOURCE, _probe

    return _probe(files[MAIN_SOURCE])["generate_framework"]()


def run_offline(generations: int, run_id: str) -> int:
    """离线闭环：不做生成，直接把磁盘框架连续自举 N 代并逐字节比对。"""
    from vea_v6.parts.bootstrap import framework_sources

    cur = framework_sources()
    base = snapshot(cur, RUNS / run_id / "gen0", {
        "gen": 0, "parent": None, "mode": "offline-disk",
        "prompt": None, "model_key": None,
        "note": "gen0 = 磁盘现状，未调用模型",
    })
    print(f"gen0  {base['combined_sha256'][:16]}  {base['total_bytes']}B  7 files  (磁盘现状)")

    for i in range(1, generations + 1):
        nxt = rebuild(cur)
        ok = nxt == cur
        okv, why = verify_local(nxt)
        head = snapshot(nxt, RUNS / run_id / f"gen{i}", {
            "gen": i, "parent": base["combined_sha256"], "mode": "offline-disk",
            "prompt": None, "model_key": None,
            "byte_identical": ok, "self_verify": okv, "self_verify_reason": why,
        })
        mark = "==" if ok else "!!"
        print(f"gen{i}  {head['combined_sha256'][:16]}  {head['total_bytes']}B  {mark} gen{i-1}  "
              f"自校验={'OK' if okv else 'FAIL'}")
        if not ok or not okv:
            print(f"FAIL 第 {i} 代不闭合：{'字节不同' if not ok else why}")
            return 1
        cur, base = nxt, head
    print(f"\n通过：一次 shot 连续 {generations} 代逐字节一致，且每代自校验 OK")
    return 0


def run_llm(generations: int, run_id: str, model_key: str, prompt: str, max_ticks: int) -> int:
    """真模型一次 shot：M0 由模型生成，之后各代由上一代自己重建。

    **必须真发 HTTP**：这里给 `_post_sync` 挂钩子计数，跑完若为 0 直接判失败。
    `DAG()` 在 `llm=None` 时静默退化成 `SmartMockLLM`，那种"1 秒跑通"是假通过。
    """
    from vea_v6.parts import llm as llm_mod
    from vea_v6.parts.bootstrap import bootstrap

    calls: List[Dict[str, Any]] = []
    real_post = llm_mod.OpenAICompatibleLLM._post_sync

    def counting_post(self, prompt, **kw):  # noqa: ANN001, ANN003 - 透传
        calls.append({"base_url": kw.get("base_url"), "model": kw.get("model"), "chars": len(prompt)})
        return real_post(self, prompt, **kw)

    llm_mod.OpenAICompatibleLLM._post_sync = counting_post
    try:
        t0 = time.time()
        r = bootstrap(prompt, model_key=model_key, max_ticks=max_ticks)
        elapsed = time.time() - t0
    finally:
        llm_mod.OpenAICompatibleLLM._post_sync = real_post

    if not calls:
        print("FAIL 一次 HTTP 请求都没发出（DAG 走了本地 mock，属假通过）")
        return 1
    bases = sorted({str(c["base_url"]) for c in calls})
    print(f"HTTP {len(calls)} 次，端点 {bases}，耗时 {elapsed:.1f}s")

    from vea_v6.parts.bootstrap import _LAST_VERDICT

    rejected = _LAST_VERDICT.get("rejected") or {}
    if rejected:
        print(f"注意：{len(rejected)} 个文件被模型写坏、已按原文件兜底：")
        for p, why in rejected.items():
            print(f"  - {p}: {why}")

    gen0 = dict(r.files)
    head = snapshot(gen0, RUNS / run_id / "gen0", {
        "gen": 0, "parent": None, "mode": f"llm:{model_key}",
        "prompt": prompt, "self_verify": r.okay, "self_verify_reason": r.reason,
        "http_calls": len(calls), "endpoints": bases,
        "rejected": rejected,
    })
    print(f"gen0  {head['combined_sha256'][:16]}  {head['total_bytes']}B  自校验={'OK' if r.okay else 'FAIL'}")
    if not r.okay:
        print(f"FAIL M0 本身不闭合：{r.reason}")
        return 1

    cur = gen0
    for i in range(1, generations + 1):
        nxt = rebuild(cur)
        same = nxt == cur
        okv, why = verify_local(nxt)
        head = snapshot(nxt, RUNS / run_id / f"gen{i}", {
            "gen": i, "parent": head["combined_sha256"], "mode": f"llm:{model_key}",
            "prompt": prompt, "byte_identical": same, "self_verify": okv, "self_verify_reason": why,
        })
        print(f"gen{i}  {head['combined_sha256'][:16]}  {head['total_bytes']}B  "
              f"{'==' if same else '!!'} gen{i-1}  自校验={'OK' if okv else 'FAIL'}")
        if not same or not okv:
            print(f"FAIL 第 {i} 代不闭合：{'字节不同' if not same else why}")
            return 1
        cur = nxt
    print(f"\n通过：一次 shot 连续 {generations} 代逐字节一致（模型 {model_key}）")
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="一次 shot 连续自举整个 vea_v6 框架")
    ap.add_argument("--generations", type=int, default=3, help="连续自举代数（判据要求 >= 3）")
    ap.add_argument("--model-key", default="laptop-6.8", help="models.json 里的 KEY")
    ap.add_argument("--prompt", default="", help="传给生成 DAG 的描述（留空即用默认 SPEC）")
    ap.add_argument("--max-ticks", type=int, default=12, help="DAG 最多跑几轮")
    ap.add_argument("--offline", action="store_true", help="不调模型，只验磁盘框架能否连续自举")
    ap.add_argument("--run-id", default=None, help="产物目录名；默认按时间戳")
    args = ap.parse_args(argv)

    if args.generations < 3:
        print("判据要求至少 3 代（--generations >= 3）", file=sys.stderr)
        return 2

    run_id = args.run_id or time.strftime("%Y%m%d-%H%M%S")
    (RUNS / run_id).mkdir(parents=True, exist_ok=True)
    print(f"run-id = {run_id}\n")

    if args.offline:
        return run_offline(args.generations, run_id)
    return run_llm(args.generations, run_id, args.model_key, args.prompt, args.max_ticks)


if __name__ == "__main__":
    sys.exit(main())
