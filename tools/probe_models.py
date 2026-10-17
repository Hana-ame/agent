#!/usr/bin/env python
"""探测本机/远端有哪些 OpenAI 兼容端点可用，以及它们暴露哪些模型。

不猜端口、不猜模型名——一律去问 `/v1/models`。每个端点分三态：

  OK      200 且能解析出模型列表     -> 可用
  AUTH    401/403                   -> 服务在，但缺 key（端点是对的）
  DOWN    连不上 / 非本协议          -> 可用性未知或不是 LLM 服务

用法：

```bash
python tools/probe_models.py                 # 探默认端口集
python tools/probe_models.py --ports 8000 8326 18326
python tools/probe_models.py --url https://api.sensenova.cn/v1
python tools/probe_models.py --write-models  # 把探到的端口写进 models.json 的 api 字段
```
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.error
import urllib.request
from typing import Any, Dict, List, Optional, Tuple

#: 常见本地推理/网关端口（vLLM / llama.cpp / LM Studio / Ollama / 各类 relay）
DEFAULT_PORTS = (8000, 8080, 8326, 18326, 18488, 18627, 18899, 11434, 1234, 5000, 20241, 45623)


def probe(base: str, timeout: float = 3.0) -> Tuple[str, Any]:
    """探测一个 base_url，返回 (状态, 详情)。"""
    url = base.rstrip("/")
    if not url.endswith("/models"):
        url += "/models"
    req = urllib.request.Request(url, headers={"Accept": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read().decode("utf-8", "replace")
            try:
                data = json.loads(raw)
            except json.JSONDecodeError:
                return "OK", {"raw": raw[:200]}
            ids: List[str] = []
            for item in data.get("data") or data.get("models") or []:
                if isinstance(item, dict):
                    ids.append(str(item.get("id") or item.get("name") or ""))
                elif isinstance(item, str):
                    ids.append(item)
            return "OK", {"models": [i for i in ids if i]}
    except urllib.error.HTTPError as exc:
        body = ""
        try:
            body = exc.read().decode("utf-8", "replace")[:200]
        except Exception:  # noqa: BLE001 - 只为展示
            pass
        if exc.code in (401, 403):
            return "AUTH", {"code": exc.code, "body": body}
        return "ERR", {"code": exc.code, "body": body}
    except Exception as exc:  # noqa: BLE001 - 连不上就是连不上
        return "DOWN", {"err": str(exc)[:120]}


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="探测 OpenAI 兼容端点的 /v1/models")
    ap.add_argument("--ports", type=int, nargs="*", default=list(DEFAULT_PORTS))
    ap.add_argument("--url", action="append", default=[], help="额外的完整 base_url")
    ap.add_argument("--timeout", type=float, default=3.0)
    args = ap.parse_args(argv)

    targets = [f"http://127.0.0.1:{p}/v1" for p in args.ports] + list(args.url)
    found: Dict[str, List[str]] = {}

    print(f"{'base_url':<42} {'状态':<6} 说明")
    print("-" * 100)
    for base in targets:
        status, info = probe(base, args.timeout)
        if status == "OK":
            models = info.get("models") or []
            found[base] = models
            desc = ", ".join(models[:6]) + (" …" if len(models) > 6 else "")
            if not models:
                desc = f"200 但无模型列表：{info.get('raw','')[:60]}"
        elif status == "AUTH":
            desc = f"服务在，缺 key（HTTP {info.get('code')}）：{info.get('body','')[:70]}"
        elif status == "ERR":
            desc = f"HTTP {info.get('code')}：{info.get('body','')[:70]}"
        else:
            desc = f"连不上：{info.get('err','')[:60]}"
        print(f"{base:<42} {status:<6} {desc}")

    print()
    if not found:
        print("没有可直接使用的端点（无 200 + 模型列表）。")
        print("AUTH 的那些说明**端点是对的**，补上 key 即可；把 key 填进 models.json。")
        return 1
    print("可用端点：")
    for base, models in found.items():
        print(f"  {base}  ->  {len(models)} 个模型")
        for m in models:
            print(f"      {m}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
