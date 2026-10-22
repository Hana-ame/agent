"""DAG: Stage1st（S1）157 归墟板块 · 近 24 小时 AI 相关情报摘要。

Edge 拓扑：

  edge_0  s1_board_url    → [scrape_catalog]  → s1_ai_threads
          抓板块目录（多页）→ **批量一次 LLM 判定**哪些帖与 AI 相关（不用关键词）

  edge_1  s1_ai_threads   → [fetch_24h]       → s1_ai_posts_24h
          对每个活跃帖，**从最后一页往回翻**，一直翻到覆盖满 24 小时为止

  edge_2  s1_ai_posts_24h → [build_digest]    → s1_ai_digest_md
          结构化 Markdown

  edge_3  s1_ai_digest_md → [summarize_report]→ s1_ai_report
          laptop-6.8 生成中文情报摘要 + 主题索引

每次运行 = 2 次 LLM 调用（分类 1 次 + 摘要 1 次）。无本地缓存。
"""
from __future__ import annotations

import datetime as dt
import html as _html
import json
import re
import sys
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import functools
import time

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parents[1]
sys.path.insert(0, str(ROOT))

import dag  # noqa: E402
from dag import DAG  # noqa: E402

EDGE_TIMINGS: Dict[str, float] = {}


def timed(fn):
    """记录每条边处理函数的耗时。"""
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        t0 = time.perf_counter()
        try:
            return fn(*args, **kwargs)
        finally:
            EDGE_TIMINGS[fn.__name__] = time.perf_counter() - t0
    return wrapper

BOARD_ID = "157"
BOARD_NAME = "归墟"
BASE = "https://stage1st.com/2b"
CLASSIFY_MODEL_KEY = "sensenova-6.8"   # 分流：目录 AI 相关性判定走 sensenova（快）
SUMMARY_MODEL_KEY = "laptop-6.8"      # 分流：长文摘要走 6.8（质量）

# 埋点：记录最近一次分类（供稳定性测试读取）
LAST_CLASSIFY: Dict[str, Any] = {}

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
}

HOURS = 24
MAX_BOARD_PAGES = 2
MAX_THREADS = 8
MAX_BACKFILL_PAGES = 15
MAX_POSTS_PER_THREAD = 15
MAX_TEXT_CHARS = 300


# ---------------------------------------------------------------------------
# 基础工具
# ---------------------------------------------------------------------------
def _http(url: str, timeout: float = 15.0) -> Tuple[str, str]:
    req = urllib.request.Request(url, headers=HEADERS)
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.geturl(), resp.read().decode("utf-8", errors="ignore")


def _parse_dt(s: str) -> Optional[dt.datetime]:
    s = (s or "").strip()
    for fmt in ("%Y-%m-%d %H:%M", "%Y-%m-%d %H:%M:%S"):
        try:
            return dt.datetime.strptime(s, fmt)
        except ValueError:
            continue
    return None


def _as_text(payload: Any) -> str:
    """引擎可能传入 str / ContentItem / list，统一成文本。"""
    if isinstance(payload, str):
        return payload
    if isinstance(payload, (list, tuple)):
        return "\n\n".join(_as_text(x) for x in payload)
    contents = getattr(payload, "contents", None)
    if contents is not None:
        return contents if isinstance(contents, str) else json.dumps(contents, ensure_ascii=False)
    return str(payload)


def _llm_text(
    prompt: str,
    *,
    model_key: str = SUMMARY_MODEL_KEY,
    max_tokens: int = 4000,
    timeout: float = 150.0,
) -> str:
    """调指定模型，只取 message.content；代理/直连切换 + 429 重试。"""
    import os as _os
    import time as _t

    import requests

    from dag.schema import ModelRegistry

    cfg = ModelRegistry(path=str(ROOT / "models.json")).resolve(model_key)
    url = cfg.endpoint.rstrip("/")
    if not url.endswith("/chat/completions"):
        url += "/chat/completions"
    headers = {"Content-Type": "application/json"}
    if cfg.apikey and cfg.apikey != "not-needed":
        headers["Authorization"] = f"Bearer {cfg.apikey}"
    body = {
        "model": cfg.model or cfg.key,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": float(cfg.params.get("temperature", 0.2)),
    }

    def _once() -> str:
        resp = requests.post(url, headers=headers, json=body, timeout=timeout)
        resp.raise_for_status()
        data = resp.json()
        msg = (data.get("choices") or [{}])[0].get("message") or {}
        return (msg.get("content") or "").strip()

    proxy_keys = ["HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy", "ALL_PROXY", "all_proxy"]
    last: Optional[Exception] = None

    # 阶段 1：按环境（通常走代理）
    for attempt in range(2):
        try:
            out = _once()
            if out:
                return out
            last = RuntimeError("content 为空")
        except requests.exceptions.HTTPError as exc:
            code = getattr(getattr(exc, "response", None), "status_code", None)
            last = exc
            if code == 429 and attempt == 0:
                _t.sleep(3)
                continue
            break
        except (requests.exceptions.ProxyError, requests.exceptions.ConnectionError) as exc:
            last = exc
            _t.sleep(1.0)
            continue

    # 阶段 2：绕开代理直连
    saved = {k: _os.environ.pop(k) for k in proxy_keys if k in _os.environ}
    try:
        for attempt in range(2):
            try:
                out = _once()
                if out:
                    return out
                last = RuntimeError("content 为空")
            except requests.exceptions.HTTPError as exc:
                code = getattr(getattr(exc, "response", None), "status_code", None)
                last = exc
                if code == 429 and attempt == 0:
                    _t.sleep(5)
                    continue
                break
            except Exception as exc:  # noqa: BLE001
                last = exc
                _t.sleep(1.0)
    finally:
        _os.environ.update(saved)

    raise last or RuntimeError("LLM 调用失败")


# ---------------------------------------------------------------------------
# 目录抓取 + LLM 判定 AI 相关性
# ---------------------------------------------------------------------------
def scrape_catalog(max_pages: int = MAX_BOARD_PAGES) -> List[Dict[str, Any]]:
    threads: Dict[str, Dict[str, Any]] = {}
    for page in range(1, max_pages + 1):
        url = f"{BASE}/forum-{BOARD_ID}-{page}.html"
        try:
            _, page_html = _http(url)
        except Exception as ex:
            print(f"  ⚠️ 板块第 {page} 页抓取失败: {ex}")
            break
        rows = re.findall(r'<tbody id="normalthread_(\d+)">(.*?)</tbody>', page_html, re.S)
        if not rows:
            break
        new_count = 0
        for tid, block in rows:
            m_title = re.search(r'class="s xst"[^>]*>([^<]+)</a>', block)
            if not m_title:
                continue
            title = _html.unescape(m_title.group(1).strip())
            m_last = re.search(r"goto=lastpost#lastpost\">([^<]+)</a>", block)
            last_post = m_last.group(1).strip() if m_last else ""
            m_rep = re.search(r'class="xi2">(\d+)</a>', block)
            replies = int(m_rep.group(1)) if m_rep else 0
            if tid not in threads:
                new_count += 1
            threads[tid] = {
                "tid": tid,
                "title": title,
                "url": f"{BASE}/thread-{tid}-1-1.html",
                "last_post": last_post,
                "replies": replies,
            }
        if new_count == 0:
            break
    return list(threads.values())


def _parse_index_list(text: str, n: int) -> List[int]:
    """从模型输出里解析出序号列表（0-based 转 1-based 由调用方处理）。"""
    m = re.search(r"\{.*\}", text, re.S)
    raw = m.group(0) if m else text
    nums: List[int] = []
    try:
        obj = json.loads(raw)
        if isinstance(obj, dict):
            for v in obj.values():
                if isinstance(v, list):
                    nums.extend(int(x) for x in v if str(x).strip().lstrip("-").isdigit())
        elif isinstance(obj, list):
            nums.extend(int(x) for x in obj if str(x).strip().lstrip("-").isdigit())
    except Exception:
        nums = [int(x) for x in re.findall(r"\d+", text)]
    return sorted({x for x in nums if 1 <= x <= n})


def classify_ai_threads(threads: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """批量一次 LLM 判定：哪些帖与 AI 相关。不用关键词。"""
    if not threads:
        return []
    lines = [f"{i + 1}. {t['title']}" for i, t in enumerate(threads)]
    prompt = (
        "下面是一个中文论坛板块的帖子目录（每行「序号. 标题」）。\n"
        "请判断哪些帖子与 AI 相关。判定标准：主题必须是人工智能、大模型/LLM、机器学习、"
        "生成式 AI（文本/图像/视频/音频）、AI 芯片与算力、AI 公司或产品、"
        "AI 应用/Agent/智能体、AI 监管与伦理等。\n"
        "注意：只是标题里出现 AI 字样但实际在聊社会话题、时政、生活杂谈的，不算；"
        "宁缺毋滥，拿不准就不要选。\n"
        "要求：不要逐条分析，直接给最终答案。只输出 JSON，不要解释，"
        "格式：{\"ai\": [序号, ...]}\n\n"
        "目录：\n" + "\n".join(lines)
    )
    used_model = CLASSIFY_MODEL_KEY
    t_start = __import__("time").perf_counter()
    try:
        out = _llm_text(prompt, model_key=CLASSIFY_MODEL_KEY, max_tokens=8000)
    except Exception as ex:  # noqa: BLE001
        print(f"           ⚠️ {CLASSIFY_MODEL_KEY} 不可用（{type(ex).__name__}），回退 {SUMMARY_MODEL_KEY}")
        used_model = SUMMARY_MODEL_KEY
        try:
            out = _llm_text(prompt, model_key=SUMMARY_MODEL_KEY, max_tokens=8000)
        except Exception as ex2:  # noqa: BLE001
            raise RuntimeError(f"AI 相关性判定失败（两个模型均不可用）: {ex2}") from ex2

    idx = _parse_index_list(out, len(threads))
    if not idx:
        # 再给一次更严格的机会
        retry = prompt + "\n\n注意：只输出形如 {\"ai\": [1,3,5]} 的 JSON，不要任何其他文字。"
        try:
            out2 = _llm_text(retry, model_key=CLASSIFY_MODEL_KEY, max_tokens=8000)
            idx = _parse_index_list(out2, len(threads))
        except Exception:  # noqa: BLE001
            idx = []
    if not idx:
        raise RuntimeError(f"AI 相关性判定失败：无法解析模型输出 {out[:200]!r}")

    picked = [threads[i - 1] for i in idx]
    LAST_CLASSIFY.clear()
    LAST_CLASSIFY.update({
        "model": used_model,
        "elapsed": round(__import__("time").perf_counter() - t_start, 2),
        "count": len(picked),
        "raw_len": len(out),
        "tids": sorted(t["tid"] for t in picked),
        # 成对保存，避免排序后与标题错位
        "pairs": [[t["tid"], t["title"]] for t in picked],
    })
    return picked


# ---------------------------------------------------------------------------
# 从最后一页往回翻，翻满 24 小时
# ---------------------------------------------------------------------------
def resolve_last_page(tid: str) -> int:
    url = f"{BASE}/forum.php?mod=redirect&tid={tid}&goto=lastpost"
    try:
        final_url, _ = _http(url)
        m = re.search(r"thread-\d+-(\d+)-1\.html", final_url)
        if m:
            return int(m.group(1))
        m = re.search(r"[?&]page=(\d+)", final_url)
        if m:
            return int(m.group(1))
    except Exception:
        pass
    return 1


def posts_on_page(tid: str, page: int) -> List[Dict[str, Any]]:
    """解析一页的全部楼层（带 datetime），不过滤时间。"""
    url = f"{BASE}/thread-{tid}-{page}-1.html"
    _, page_html = _http(url)
    out: List[Dict[str, Any]] = []
    for ch in re.split(r'(?=<div id="post_\d+")', page_html)[1:]:
        m_pid = re.match(r'<div id="post_(\d+)"', ch)
        if not m_pid:
            continue
        m_time = re.search(r"发表于\s*(\d{4}-\d{1,2}-\d{1,2}\s+\d{1,2}:\d{2})", ch)
        ts = _parse_dt(m_time.group(1)) if m_time else None
        if ts is None:
            continue
        m_body = re.search(r'<td class="t_f"[^>]*>(.*?)</td>', ch, re.S)
        if not m_body:
            continue
        text = re.sub(r"<[^>]+>", " ", m_body.group(1))
        text = re.sub(r"\s+", " ", text).strip()
        if len(text) < 6:
            continue
        out.append({
            "pid": m_pid.group(1),
            "dt": ts,
            "time": ts.strftime("%Y-%m-%d %H:%M"),
            "text": text[:MAX_TEXT_CHARS],
        })
    return out


def fetch_thread_24h(tid: str, since: dt.datetime, max_pages: int = MAX_BACKFILL_PAGES) -> Dict[str, Any]:
    """从最后一页开始往回翻，直到最早楼层早于 since（覆盖满 24h）或翻满 max_pages。"""
    last_page = resolve_last_page(tid)
    collected: List[Dict[str, Any]] = []
    seen = set()
    pages_read = 0
    page = last_page
    while page >= 1 and pages_read < max_pages:
        try:
            posts = posts_on_page(tid, page)
        except Exception:
            break
        pages_read += 1
        if posts:
            for p in posts:
                if p["dt"] >= since and p["pid"] not in seen:
                    seen.add(p["pid"])
                    collected.append(p)
            oldest = min(p["dt"] for p in posts)
            if oldest < since:
                # 这一页已经越过 24h 边界，更早的页不必再看
                break
        page -= 1
    collected.sort(key=lambda p: p["dt"])
    kept = collected[-MAX_POSTS_PER_THREAD:]
    return {
        "last_page": last_page,
        "pages_read": pages_read,
        # 保留窗口内最新的 N 条（升序排序后取尾部），并去掉不可序列化的 datetime
        "posts": [{k: v for k, v in p.items() if k != "dt"} for p in kept],
        "posts_total_24h": len(collected),
    }


def collect_24h(
    hours: int = HOURS,
    limit: int = MAX_THREADS,
    threads: Optional[List[Dict[str, Any]]] = None,
    catalog_total: Optional[int] = None,
) -> Dict[str, Any]:
    now = dt.datetime.now()
    since = now - dt.timedelta(hours=hours)

    all_threads = threads if threads is not None else scrape_catalog()
    active: List[Tuple[dt.datetime, Dict[str, Any]]] = []
    for t in all_threads:
        lp = _parse_dt(t.get("last_post", ""))
        if lp is not None and lp >= since:
            active.append((lp, t))
    active.sort(key=lambda x: x[0], reverse=True)
    active = active[:limit]

    def _work(item: Tuple[dt.datetime, Dict[str, Any]]) -> Dict[str, Any]:
        _, t = item
        info = fetch_thread_24h(t["tid"], since)
        return {"thread": t, **info}

    if active:
        with ThreadPoolExecutor(max_workers=min(6, len(active))) as ex:
            collected = list(ex.map(_work, active))
    else:
        collected = []

    return {
        "generated_at": now.strftime("%Y-%m-%d %H:%M:%S"),
        "window_start": since.strftime("%Y-%m-%d %H:%M"),
        "hours": hours,
        "board": f"{BASE}/forum-{BOARD_ID}-1.html",
        "catalog_total": catalog_total if catalog_total is not None else len(all_threads),
        "ai_total": len(all_threads),
        "active_total": len(active),
        "threads": collected,
    }


# ---------------------------------------------------------------------------
# fn 边
# ---------------------------------------------------------------------------
def s1_scrape_catalog(seed: Any) -> str:
    """edge_0：抓目录 → LLM 判定 AI 相关性 → AI 帖 JSON。"""
    _as_text(seed)
    print("  [edge_0] 抓取 S1 157 板块目录…")
    threads = scrape_catalog()
    print(f"           目录 {len(threads)} 帖，交给 LLM 判定 AI 相关性…")
    ai = classify_ai_threads(threads)
    print(f"           LLM 判定 AI 相关：{len(ai)} 帖")
    return json.dumps({"catalog_total": len(threads), "threads": ai}, ensure_ascii=False)


def s1_fetch_24h(threads_json: Any) -> str:
    """edge_1：从最后一页往回翻，翻满 24 小时。"""
    threads_json = _as_text(threads_json)
    print("  [edge_1] 从各帖最后一页往回翻，抓满近 24 小时…")
    try:
        payload = json.loads(threads_json)
        threads = payload.get("threads", [])
        cat_total = payload.get("catalog_total")
    except Exception:
        threads, cat_total = [], None
    data = collect_24h(threads=threads, catalog_total=cat_total)
    n_posts = sum(t["posts_total_24h"] for t in data["threads"])
    print(f"           活跃帖 {data['active_total']} 个，24h 回帖 {n_posts} 条")
    return json.dumps(data, ensure_ascii=False)


def s1_build_digest(posts_json: Any) -> str:
    """edge_2：24h 回帖 → 结构化 Markdown（含机读 META）。"""
    posts_json = _as_text(posts_json)
    print("  [edge_2] 生成 Markdown 汇总…")
    d = json.loads(posts_json)
    lines = [
        f"# Stage1st {BOARD_ID}（{BOARD_NAME}）近 {d['hours']} 小时 AI 讨论",
        "",
        f"- 采集时间：{d['generated_at']}",
        f"- 时间窗口：{d['window_start']} ~ {d['generated_at']}",
        f"- 板块：{d['board']}",
        f"- 目录 {d['catalog_total']} 帖，AI 相关 {d['ai_total']} 帖，24h 内活跃 {d['active_total']} 帖",
        "",
    ]
    total_posts = 0
    for item in d["threads"]:
        th = item["thread"]
        posts = item["posts"]
        total_posts += len(posts)
        lines.append(f"## {th['title']}")
        lines.append(
            f"- 链接：{th['url']}（最后回复 {th['last_post']}；"
            f"末页 {item['last_page']}，往回读 {item['pages_read']} 页，24h 内 {item['posts_total_24h']} 条）"
        )
        for p in posts:
            lines.append(f"  - [{p['time']}] {p['text']}")
        lines.append("")
    lines.insert(6, f"- 24h 回帖合计：{total_posts} 条")

    meta = {
        "generated_at": d["generated_at"],
        "window_start": d["window_start"],
        "board": d["board"],
        "catalog_total": d["catalog_total"],
        "ai_total": d["ai_total"],
        "active_total": d["active_total"],
        "threads": [
            {
                "title": it["thread"]["title"],
                "url": it["thread"]["url"],
                "last_post": it["thread"]["last_post"],
                "last_page": it["last_page"],
                "pages_read": it["pages_read"],
                "post_count": it["posts_total_24h"],
            }
            for it in d["threads"]
        ],
    }
    marker = "<!--META:" + json.dumps(meta, ensure_ascii=False) + "-->"
    return marker + "\n" + "\n".join(lines)


def _extractive(digest: str, note: str = "") -> str:
    lines = ["（LLM 摘要不可用，以下为抽取式摘要）"] if note else []
    cur = None
    for line in digest.splitlines():
        if line.startswith("## "):
            cur = line[3:].strip()
            lines.append(f"- **{cur}**")
        elif cur is not None and line.strip().startswith("- ["):
            lines.append(f"  - {line.strip()[2:][:160]}")
    return "\n".join([x for x in lines if x]).strip()


def _digest_meta(digest: str) -> str:
    m = re.search(r"<!--META:(.*?)-->", digest, re.S)
    if not m:
        return json.dumps(
            {"generated_at": "", "window_start": "", "board": "",
             "catalog_total": 0, "ai_total": 0, "active_total": 0, "threads": []},
            ensure_ascii=False,
        )
    return m.group(1)


def s1_summarize_report(digest: Any) -> str:
    """edge_3：laptop-6.8 生成情报摘要，拼成终版报告。"""
    digest = _as_text(digest)
    print("  [edge_3] 生成 AI 情报摘要（laptop-6.8）…")
    summary = ""
    meta0 = json.loads(_digest_meta(digest))
    n_threads = len(meta0["threads"])
    n_posts = sum(it.get("post_count", 0) for it in meta0["threads"])
    try:
        prompt = (
            "你是科技情报分析师。下面是 Stage1st（S1）157 归墟板块最近 24 小时"
            "与 AI 相关的论坛讨论原文。请输出结构化的中文情报摘要：\n"
            "1) 一句话总览（今天社区在聊什么）\n"
            "2) 核心议题（按热度排序，每条 1-2 句，注明涉及的技术/产品/公司）\n"
            "3) 社区观点与争议（正反方）\n"
            "4) 值得关注的信号/风险\n"
            "只基于给定原文，不要编造；信息不足就说明不足。\n"
            f"【统计口径】本窗口共 {n_threads} 个活跃帖、{n_posts} 条回帖；"
            "下面给出的是每帖最新若干条样本，不是全量。凡需引用数量，一律使用"
            f"{n_posts} 条这个总数，不要自行统计样本条数。\n\n"
            f"【原文】\n{digest}"
        )
        summary = _llm_text(prompt, model_key=SUMMARY_MODEL_KEY, max_tokens=8000)
    except Exception as ex:  # noqa: BLE001
        print(f"           ⚠️ LLM 不可用（{type(ex).__name__}: {str(ex)[:120]}），降级为抽取式摘要")
        summary = _extractive(digest, note=str(ex))

    if not summary or not summary.strip():
        summary = _extractive(digest, note="empty")

    meta = json.loads(_digest_meta(digest))
    total_posts = sum(it.get("post_count", 0) for it in meta["threads"])
    head = [
        f"# Stage1st 157（{BOARD_NAME}）近 24 小时 AI 情报摘要",
        "",
        f"- 生成时间：{meta['generated_at']}",
        f"- 时间窗口：{meta['window_start']} ~ {meta['generated_at']}",
        f"- 数据源：{meta['board']}",
        f"- 覆盖：目录 {meta['catalog_total']} 帖 / AI 相关 {meta['ai_total']} 帖 / "
        f"24h 活跃 {meta['active_total']} 帖 / 回帖 {total_posts} 条",
        "",
        "## 摘要",
        "",
        summary.strip(),
        "",
        "## 主题索引",
        "",
    ]
    for it in meta["threads"]:
        head.append(
            f"- [{it['title']}]({it['url']}) — 最后回复 {it['last_post']}，"
            f"24h 回帖 {it['post_count']} 条（末页 {it['last_page']}，回溯 {it['pages_read']} 页）"
        )
    return "\n".join(head)


# ---------------------------------------------------------------------------
# 构建
# ---------------------------------------------------------------------------
SEED_TAG = "s1_board_url"
SEED_CONTENT = "https://stage1st.com/2b/forum-157-1.html"
FINAL_TAG = "s1_ai_report"
DEFAULT_OUTPUT = SCRIPT_DIR / "out" / "stage1st_ai_24h.md"


def build_stage1st_ai_dag() -> DAG:
    d = DAG(name="stage1st_ai_24h", models=str(ROOT / "models.json"))
    d.add_edge(
        input_tags=["s1_board_url"],
        processor=timed(s1_scrape_catalog),
        output_tags=["s1_ai_threads"],
        layer=0,
    )
    d.add_edge(
        input_tags=["s1_ai_threads"],
        processor=timed(s1_fetch_24h),
        output_tags=["s1_ai_posts_24h"],
        layer=1,
    )
    d.add_edge(
        input_tags=["s1_ai_posts_24h"],
        processor=timed(s1_build_digest),
        output_tags=["s1_ai_digest_md"],
        layer=2,
    )
    d.add_edge(
        input_tags=["s1_ai_digest_md"],
        processor=timed(s1_summarize_report),
        output_tags=["s1_ai_report"],
        layer=3,
    )
    return d


def run_once(
    *,
    output: Optional[str | Path] = None,
    max_steps: int = 30,
    dry_run: bool = False,
) -> Dict[str, Any]:
    """构建并单次运行 Stage1st 24h AI 情报摘要 DAG。"""
    EDGE_TIMINGS.clear()
    t0 = time.perf_counter()
    d = build_stage1st_ai_dag()
    d.add_content(contents=SEED_CONTENT, tags=[SEED_TAG], gen=0)
    steps = d.run(until_tag=FINAL_TAG, max_steps=max_steps)
    elapsed = time.perf_counter() - t0

    report = d.get_result(FINAL_TAG) or ""
    out_path = Path(output) if output else DEFAULT_OUTPUT
    written = False
    if not dry_run:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(report, encoding="utf-8")
        written = True

    return {
        "name": "stage1st-ai-24h",
        "dag": d,
        "report": report,
        "output": str(out_path),
        "written": written,
        "elapsed": elapsed,
        "steps": steps,
        "timings": dict(EDGE_TIMINGS),
    }


def main(argv: Optional[List[str]] = None) -> int:
    import argparse

    ap = argparse.ArgumentParser(description="Stage1st 157 归墟板块 · 近 24 小时 AI 相关情报摘要 DAG")
    ap.add_argument("-o", "--output", default=None, help="输出文件路径（默认 scripts/s1/out/stage1st_ai_24h.md）")
    ap.add_argument("--max-steps", type=int, default=30)
    ap.add_argument("--dry-run", action="store_true", help="只运行不写盘")
    args = ap.parse_args(argv)

    res = run_once(output=args.output, max_steps=args.max_steps, dry_run=args.dry_run)
    print(f"\n=== {res['name']} ===")
    print(f"输出：{res['output']}" + ("" if res["written"] else "（dry-run，未写盘）"))
    print(f"行数：{len(str(res['report']).splitlines())}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
