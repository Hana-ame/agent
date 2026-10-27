#!/usr/bin/env python3
"""单 session 复盘报告入口。

用法：
    python3 v8/configs/dsh_session_report/run.py --list
    python3 v8/configs/dsh_session_report/run.py --session <id> [--backend http|pi]
    python3 v8/configs/dsh_session_report/run.py            # 用 dag.json 里的默认 session
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
V8 = ROOT / "v8"
for p in (str(ROOT), str(V8)):
    if p not in sys.path:
        sys.path.insert(0, p)

import dag8  # noqa: E402
from v7.llm import OpenAICompatibleClient  # noqa: E402

class _LocalClient:
    """网关不可用时的本地降级：从 prompt 里的 session JSON 生成模板报告。"""

    @staticmethod
    def _session(prompt: str):
        marker = prompt.find('"session_id"')
        if marker < 0:
            return {}
        start = prompt.rfind("{", 0, marker)
        if start < 0:
            return {}
        import json as _json
        try:
            session, _ = _json.JSONDecoder().raw_decode(prompt[start:])
            return session if isinstance(session, dict) else {}
        except ValueError:
            return {}

    def complete(self, prompt: str, **_kwargs) -> str:
        import json as _json
        session = self._session(prompt)
        if "ASPECT:" in prompt:
            aspect = prompt.split("ASPECT:", 1)[1].split("\n", 1)[0].strip()
            if aspect == "背景与用户诉求":
                body = (f"- 标题：{session.get('title') or '（无）'}\n"
                        f"- 工作目录：`{session.get('cwd')}`\n"
                        f"- 更新时间：{session.get('updated_iso')}\n"
                        f"- 用户诉求：{session.get('request') or '（无）'}\n")
            elif aspect == "过程与关键动作":
                body = (f"- turns：{session.get('turns')}\n"
                        f"- tools：{session.get('tools')}\n"
                        f"- 状态：{session.get('status')}\n")
            elif aspect == "最终结论与证据":
                body = f"{session.get('conclusion') or '（无）'}\n"
            elif aspect == "技术决策与判断":
                body = ("（本地降级未做决策归纳；结论原文见“最终结论与证据”）\n")
            else:
                body = "（本地降级未生成；网关恢复后重跑即可补全）\n"
            return f"## {aspect}\n\n{body}"

        if "MERGE_REPORT" in prompt:
            body = prompt.split("维度材料：", 1)[-1].strip()
            return ("# DSH Session 复盘报告（本地合并）\n\n"
                    "> 本地模板：网关不可用，未调用模型；各维度来自 session 记录。\n\n"
                    + body + "\n")

        # 兼容旧的整篇报告 prompt
        if not session:
            return "# 本地报告\n\n> 未解析到 session JSON。\n"
        return (
            f"# {session.get('title') or session.get('session_id')} 复盘报告\n\n"
            "> 本地模板：网关不可用，未调用模型。\n\n"
            f"## 最终结论\n{session.get('conclusion') or '（无）'}\n"
        )

    def embed(self, text: str, **_kwargs):
        return [0.0] * 32


HERE = Path(__file__).resolve().parent
CFG = HERE / "dag.json"
STATE = Path(os.environ.get("DSH_HOME", str(Path.home() / ".dsh"))) / "session-manager" / "state.json"


def list_sessions(limit: int = 20) -> int:
    state = json.loads(STATE.read_text(encoding="utf-8"))
    rows = []
    for sid, meta in state["byId"].items():
        if meta.get("kind") != "top-level":
            continue
        rows.append((meta.get("updatedAt") or 0, sid, meta.get("cwd") or "",
                     meta.get("title") or ""))
    rows.sort(reverse=True)
    for ts, sid, cwd, title in rows[:limit]:
        when = time.strftime("%m-%d %H:%M", time.localtime(ts / 1000))
        print(f"{sid}  {when}  {title[:42]:42s}  {cwd}")
    return 0


def main() -> int:
    default = json.loads(CFG.read_text(encoding="utf-8")).get("seed_content", "")
    ap = argparse.ArgumentParser()
    ap.add_argument("--session", default=default, help="完整 id 或唯一前缀")
    ap.add_argument("--backend", choices=("pi", "http", "local"), default="http",
                    help="graph 节点后端：http/pi/local；local 为网关不可用时模板报告")
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--rpm", type=int, default=1)
    ap.add_argument("--max-seconds", type=float, default=900)
    ap.add_argument("--list", action="store_true")
    args = ap.parse_args()
    if args.list:
        return list_sessions()

    kwargs = {}
    if args.backend == "http":
        kwargs["graph_llm_client"] = OpenAICompatibleClient(timeout=300, max_retries=5)
    elif args.backend == "local":
        kwargs["graph_llm_client"] = _LocalClient()
    res = dag8.run(
        str(CFG), seed_content=args.session,
        db_path=str(ROOT / "v8" / "data" / "dsh_session_report.sqlite3"),
        max_workers=args.workers,
        rpm=(None if args.backend == "local" else args.rpm),
        use_cache=False,
        max_seconds=args.max_seconds, **kwargs,
    )
    print(res.summary())
    report_path = ""
    if res.result:
        try:
            report_path = json.loads(res.result).get("report", "")
        except ValueError:
            report_path = ""
    if not report_path:
        latest = ROOT / "reports" / "dsh_session_report_latest.md"
        report_path = str(latest) if latest.exists() else ""
    print("report:", report_path)
    if report_path:
        p = Path(report_path)
        if p.exists():
            print("chars:", len(p.read_text(encoding="utf-8")))
    return 0 if res.status == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())