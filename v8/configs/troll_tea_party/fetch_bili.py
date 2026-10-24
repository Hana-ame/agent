#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""fetch_bili.py — DAG 的确定性取数节点：B 站 URL → 元信息 + ASR 转写。

这个脚本只做“拿数据”，不做任何总结。它把结果写成一段固定格式的文本，
后面的 pi 节点（6.8 模型）只负责阅读、纠错、组织与写作。

用法：
    python3 fetch_bili.py "https://www.bilibili.com/video/BV1W5nw6PExs/"
    echo "https://www.bilibili.com/video/BV1W5nw6PExs/" | python3 fetch_bili.py

输出（stdout）：
    <header>...</header>         # 笔记头，定稿节点原样复制
    # 视频元信息
    ...
    # 转写原文
    ...

缓存：
    cache/<BV>.json              # 元信息 + 转写；同一 BV 重复跑不再打接口
    cache/cookie.txt             # 上次登录态可用的 cookie 文件名（不存 cookie 值）

为什么不用 pi 的工具：
    6.8 模型（sensenova-6.8-flash-lite）走本网关时带工具会 400
    （tools 的 schema 上游不认）；而且“取字幕”是确定性任务，本就该由代码做。
"""
from __future__ import annotations

import glob
import gzip
import json
import os
import re
import subprocess
import sys
import tempfile
import urllib.request
from datetime import datetime, timedelta, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
CACHE = HERE / "cache"
COOKIE_CACHE = CACHE / "cookie.txt"
BILI_SUBTITLE = Path.home() / "script" / "bili-subtitle.py"
UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36"
)
CST = timezone(timedelta(hours=8))


def log(msg: str) -> None:
    print(f"[fetch] {msg}", file=sys.stderr, flush=True)


def parse_bvid(url: str) -> str:
    m = re.search(r"(BV[0-9A-Za-z]{10})", url or "")
    if not m:
        raise SystemExit(f"URL 里没有 BV 号：{url!r}")
    return m.group(1)


def http_text(url: str, timeout: float = 25.0) -> str:
    req = urllib.request.Request(
        url,
        headers={
            "User-Agent": UA,
            "Accept": "text/html,application/xhtml+xml,application/json;q=0.9,*/*;q=0.8",
            "Accept-Encoding": "gzip",
            "Accept-Language": "zh-CN,zh;q=0.9",
        },
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        raw = resp.read()
        if (resp.headers.get("Content-Encoding") or "").lower() == "gzip":
            raw = gzip.decompress(raw)
        return raw.decode("utf-8", "replace")


def fetch_view(bvid: str) -> dict:
    data = json.loads(http_text(f"https://api.bilibili.com/x/web-interface/view?bvid={bvid}"))
    if data.get("code") != 0 or not data.get("data"):
        raise RuntimeError(f"view 接口失败：{data.get('code')} {data.get('message')}")
    return data["data"]


def iso_to_cst(value: str) -> str:
    s = value.strip()
    if s.endswith("Z"):
        s = s[:-1] + "+00:00"
    dt = datetime.fromisoformat(s)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(CST).strftime("%Y-%m-%d %H:%M:%S")


def fetch_page_meta(bvid: str) -> dict:
    """从视频页 JSON-LD 里取 dateCreated。

    注意：view 接口的 `pubdate` 有时是后续编辑/定时发布的时间，
    和页面上展示的首次发布时间不一致；页面 JSON-LD 的 dateCreated 才是
    “视频真正创建/发布”的那个时间（本视频实测差一天多）。
    """
    html = http_text(f"https://www.bilibili.com/video/{bvid}/")
    out = {}
    m = re.search(r'"dateCreated":"([^"]+)"', html)
    if m:
        out["date_created"] = iso_to_cst(m.group(1))
    m = re.search(r'"uploadDate":"([^"]+)"', html)
    if m:
        out["upload_date"] = iso_to_cst(m.group(1))
    return out


def duration_text(seconds: int) -> str:
    seconds = max(0, int(seconds or 0))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"


def cookie_candidates() -> list[str]:
    out: list[str] = []
    if COOKIE_CACHE.is_file():
        cached = COOKIE_CACHE.read_text(encoding="utf-8", errors="replace").strip()
        if cached and Path(cached).is_file():
            out.append(cached)
    for p in sorted(
        glob.glob(os.path.expanduser("~/Downloads/cookies*.txt")),
        key=os.path.getmtime,
        reverse=True,
    ):
        if p not in out:
            out.append(p)
    return out


def fetch_transcript(url: str, bvid: str, refresh: bool = False) -> tuple[str, str]:
    """返回 (transcript, 使用的 cookie 文件名)。失败抛 RuntimeError。"""
    if not BILI_SUBTITLE.is_file():
        raise RuntimeError(f"取字幕脚本不存在：{BILI_SUBTITLE}")

    errors: list[str] = []
    for cookie in cookie_candidates():
        # 先用 --list 快速判登录态，避免每次都为过期 cookie 走完整请求。
        env = dict(os.environ)
        env["BILI_COOKIE_FILE"] = cookie
        try:
            probe = subprocess.run(
                [str(BILI_SUBTITLE), "--list", bvid],
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=45,
                env=env,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            errors.append(f"{Path(cookie).name}: {type(exc).__name__}")
            continue
        probe_text = (probe.stdout or "") + (probe.stderr or "")
        if "isLogin=True" not in probe_text:
            errors.append(f"{Path(cookie).name}: 登录态失效")
            continue

        # 登录态可用：正式取文本。--fallback 默认开启；这里显式写上，行为更可读。
        with tempfile.NamedTemporaryFile(
            prefix=f"{bvid}-", suffix=".txt", dir=str(CACHE), delete=False
        ) as fh:
            tmp = Path(fh.name)
        try:
            proc = subprocess.run(
                [str(BILI_SUBTITLE), "--format", "txt", "--out", str(tmp), url],
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=120,
                env=env,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            errors.append(f"{Path(cookie).name}: {type(exc).__name__}")
            tmp.unlink(missing_ok=True)
            continue

        try:
            if proc.returncode == 0 and tmp.is_file():
                text = tmp.read_text(encoding="utf-8", errors="replace").strip()
                if len(text) >= 500:
                    COOKIE_CACHE.write_text(cookie, encoding="utf-8")
                    log(f"用 {Path(cookie).name} 取到 {len(text)} 字符（{proc.returncode}）")
                    return text, Path(cookie).name
                errors.append(f"{Path(cookie).name}: 文本过短（{len(text)}）")
            else:
                detail = (proc.stderr or proc.stdout or "").strip().splitlines()
                errors.append(
                    f"{Path(cookie).name}: exit={proc.returncode}"
                    + (f" {detail[-1][:120]}" if detail else "")
                )
        finally:
            tmp.unlink(missing_ok=True)

        # 2 = 无字幕轨道（终态，换 cookie 也没用）；5 = 视频已删；6 = 无中文语言。
        if proc.returncode in (2, 5, 6):
            raise RuntimeError(f"该视频没有可用字幕（exit={proc.returncode}）：{errors[-1]}")

    raise RuntimeError(
        "所有候选 cookie 都取不到字幕（可能全部过期）。"
        "请重新导出登录态 cookie 到 ~/Downloads/cookies.txt。"
        f" 尝试记录：{errors[:6]}"
    )


def build_payload(bvid: str) -> dict:
    view = fetch_view(bvid)
    page = {}
    try:
        page = fetch_page_meta(bvid)
    except Exception as exc:  # noqa: BLE001 - 页面拿不到就退回 view 时间
        log(f"页面元信息失败，退回 view 时间：{type(exc).__name__}: {exc}")

    title = str(view.get("title") or "").strip()
    owner = str((view.get("owner") or {}).get("name") or "").strip()
    url = f"https://www.bilibili.com/video/{bvid}/"
    published = page.get("date_created") or page.get("upload_date")
    if not published:
        published = datetime.fromtimestamp(int(view.get("pubdate") or 0), CST).strftime(
            "%Y-%m-%d %H:%M:%S"
        )

    live_date = ""
    m = re.search(r"(20\d{2})\.(\d{1,2})\.(\d{1,2})", title)
    if m:
        live_date = f"{m.group(1)}.{int(m.group(2)):02d}.{int(m.group(3)):02d}"

    short_title = re.split(r"——（\d+）", title, maxsplit=1)[0].strip() or title
    header_lines = [f"# {short_title}——深度阅读笔记", ""]
    header_lines.append(f"**视频来源**：[{title}]({url})")
    header_lines.append(f"**UP主**：{owner}")
    if live_date:
        header_lines.append(f"**发布时间**：{published}（节选自 {live_date} 直播）")
    else:
        header_lines.append(f"**发布时间**：{published}")
    header = "\n".join(header_lines)

    meta = {
        "bvid": bvid,
        "title": title,
        "short_title": short_title,
        "owner": owner,
        "published": published,
        "live_date": live_date,
        "duration": duration_text(int(view.get("duration") or 0)),
        "url": url,
        "header": header,
    }
    return meta


def build_source(meta: dict, transcript: str) -> str:
    lines = [
        "<header>",
        meta["header"],
        "</header>",
        "",
        "# 视频元信息",
        f"- 标题：{meta['title']}",
        f"- BV：{meta['bvid']}",
        f"- UP主：{meta['owner']}",
        f"- 发布时间：{meta['published']}",
    ]
    if meta.get("live_date"):
        lines.append(f"- 直播日期：{meta['live_date']}")
    lines += [
        f"- 时长：{meta['duration']}",
        f"- 链接：{meta['url']}",
        "- 字幕：B 站 AI 自动字幕（ai-zh，ASR，含同音错字与口语）",
        "",
        "# 转写原文",
        transcript.strip(),
        "",
    ]
    return "\n".join(lines)


def main() -> int:
    argv = [a for a in sys.argv[1:] if a != "--refresh"]
    refresh = "--refresh" in sys.argv[1:]
    url = argv[0].strip() if argv else sys.stdin.read().strip()
    if not url:
        print("用法：fetch_bili.py <视频 URL> 或从 stdin 读 URL", file=sys.stderr)
        return 2
    if not url.startswith("http"):
        # 允许直接传 BV 号，DAG 里仍然只用 URL，但手工调试方便。
        url = f"https://www.bilibili.com/video/{url}/"
    bvid = parse_bvid(url)
    CACHE.mkdir(parents=True, exist_ok=True)

    cache_file = CACHE / f"{bvid}.json"
    if cache_file.is_file() and not refresh:
        try:
            cached = json.loads(cache_file.read_text(encoding="utf-8"))
            if cached.get("transcript") and cached.get("meta"):
                log(f"命中缓存 {cache_file.name}")
                print(build_source(cached["meta"], cached["transcript"]))
                return 0
        except (OSError, ValueError):
            pass

    log(f"抓取 {bvid} 的元信息与字幕")
    meta = build_payload(bvid)
    transcript, cookie_name = fetch_transcript(url, bvid, refresh=refresh)
    cache_file.write_text(
        json.dumps({"meta": meta, "transcript": transcript}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    log(f"写入缓存 {cache_file.name}（cookie={cookie_name}）")
    print(build_source(meta, transcript))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
