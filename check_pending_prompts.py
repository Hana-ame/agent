#!/usr/bin/env python3
"""
check_pending_prompts.py — 检查 Board 666 是否有尚未回复的 Prompt

用途：
  1. 获取 Board 666 全部帖子
  2. 按 ts 降序排序（主帖 + 回复）
  3. 识别尚无 Auto666 或 Loop666 回复的帖子（= 未处理的 Prompt）
  4. 输出结构化 JSON 结果

输出格式（JSON）：
  {
    "total_posts": <int>,
    "latest_ts": "<ISO timestamp>",
    "pending": [ { "no": <int>, "ts": "<...>", "txt": "<...>", "thread_id": "<...>" }, ... ],
    "has_pending": true/false,
    "summary": "<人类可读摘要>"
  }

返回码：
  0 — 有未处理的需求（pending 非空）
  1 — 无未处理的需求
  2 — 获取/解析失败
"""

import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

# ── 配置 ──────────────────────────────────────────────────────────
BOARD_ID = 666
MOONCHAN_SCRIPT = Path(
    os.environ.get(
        "MOONCHAN_SCRIPT",
        "/home/lumin/.claude/skills/moonchan-forum/scripts/moonchan.py",
    )
)

# 视为"已处理"的昵称列表
REPLIED_NICKNAMES = {"Auto666", "Loop666"}


# ── 数据获取 ──────────────────────────────────────────────────────


def fetch_board_data(bid: int = BOARD_ID) -> list | None:
    """通过 moonchan.py list 获取板块数据，返回 Python 对象"""
    cmd = [
        sys.executable or "python3",
        str(MOONCHAN_SCRIPT),
        "list",
        str(bid),
        "--pn",
        "0",
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
    except subprocess.TimeoutExpired:
        print("[check_pending_prompts] 错误: 获取板块数据超时", file=sys.stderr)
        return None

    if result.returncode != 0:
        print(
            f"[check_pending_prompts] 错误: moonchan.py 返回码 {result.returncode}",
            file=sys.stderr,
        )
        print(f"  stderr: {result.stderr[:500]}", file=sys.stderr)
        return None

    try:
        data = json.loads(result.stdout)
    except json.JSONDecodeError as e:
        print(
            f"[check_pending_prompts] 错误: JSON 解析失败 — {e}", file=sys.stderr
        )
        return None

    if not isinstance(data, list):
        print(
            f"[check_pending_prompts] 错误: 返回数据不是数组 (type={type(data).__name__})",
            file=sys.stderr,
        )
        return None

    return data


# ── 排序 ──────────────────────────────────────────────────────────


def sort_posts(data: list) -> list:
    """按 ts 降序排列主帖数组及每个帖子的回复列表"""
    data.sort(key=lambda p: p["ts"], reverse=True)
    # 验证
    assert data[0]["ts"] == max(p["ts"] for p in data), "主帖排序验证失败"

    for post in data:
        replies = post.get("list")
        if replies:
            replies.sort(key=lambda r: r["ts"], reverse=True)
            if replies:
                assert replies[0]["ts"] == max(
                    r["ts"] for r in replies
                ), f"回复排序验证失败: no={post.get('no')}"
    return data


# ── 检查未处理 ────────────────────────────────────────────────────


def has_reply_from(nickname: str, post: dict) -> bool:
    """检查某帖子是否已被指定昵称回复"""
    for reply in post.get("list") or []:
        if reply.get("n") == nickname:
            return True
    return False


def classify_txt(txt: str) -> str:
    """粗分类：code / instruction / upload / rant / unknown"""
    txt = txt.strip()
    if not txt:
        return "empty"
    code_indicators = [
        "```python",
        "```py",
        "def ",
        "import ",
        "class ",
        "timeout",
        "retry",
        "requests.",
    ]
    for ind in code_indicators:
        if ind in txt:
            return "code"
    instruction_indicators = [
        "ssh", "add ", "修改", "创建", "修复", "执行", "检查",
        "which branch", "git ", "汇报", "测试", "写一个",
    ]
    for ind in instruction_indicators:
        if ind in txt.lower():
            return "instruction"
    if txt.startswith("http") or txt.startswith("[Preview]") or txt.startswith("[Download]") or txt.startswith("Download:"):
        return "upload"
    if any(c in txt for c in ["操", "傻逼", "死"]):
        return "rant"
    return "unknown"


def find_pending_prompts(data: list) -> list[dict]:
    """
    遍历所有帖子，找出尚无 Auto666 / Loop666 回复的帖子。
    返回列表，每项含 {no, ts, txt, thread_id, type, thread_no, thread_ts}
    """
    pending = []
    for post in data:
        # 用所有已处理昵称检查
        already_replied = any(
            has_reply_from(nick, post) for nick in REPLIED_NICKNAMES
        )
        if already_replied:
            continue

        # 分类
        ptype = classify_txt(post.get("txt", ""))

        pending.append(
            {
                "no": post.get("no"),
                "ts": post.get("ts"),
                "txt": post.get("txt", ""),
                "thread_id": post.get("id"),
                "type": ptype,
                "thread_no": post.get("no"),  # 主帖 no 即 thread ID
            }
        )
    return pending


# ── 结果输出 ──────────────────────────────────────────────────────


def build_output(data: list, pending: list) -> dict:
    """构建结构化 JSON 输出"""
    latest_ts = data[0]["ts"] if data else None
    has_pending = len(pending) > 0

    # 摘要
    total = len(data)
    lines = [
        f"Board 666 检查结果 ({datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')})",
        f"总帖子数: {total}",
        f"最新帖子: ts={latest_ts}",
        f"未处理需求: {len(pending)} 个",
    ]
    if pending:
        for p in pending:
            lines.append(
                f"  · no={p['no']} [{p['type']}] {p['txt'][:80]}..."
            )
    else:
        lines.append("  ✅ 所有帖子均已回复，无未处理需求。")

    return {
        "total_posts": total,
        "latest_ts": latest_ts,
        "pending": pending,
        "has_pending": has_pending,
        "summary": "\n".join(lines),
    }


def main() -> int:
    # 1. 获取数据
    data = fetch_board_data()
    if data is None:
        print(json.dumps({"error": "获取板块数据失败", "has_pending": False}))
        return 2

    # 2. 排序
    sort_posts(data)

    # 3. 查找未处理
    pending = find_pending_prompts(data)

    # 4. 输出 JSON
    output = build_output(data, pending)
    print(json.dumps(output, ensure_ascii=False, indent=2))

    return 0 if pending else 1


if __name__ == "__main__":
    sys.exit(main())
