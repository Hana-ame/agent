"""Edge script: chatto「去 stage1st 收集帖子」任务(s1profile 每日收集)的 vertex-edge 实现。

对应 chatto-bot 的定时任务 plugins/s1profile.py + plugins/s1parse.py:
  1. 抓配置版块(默认 forum-157)最新 N 个帖的全部楼层,按 uid 归并(by_uid);
  2. 对每个发言 uid 再补抓其「发的主题」(home.php?mod=space&uid=<uid>&do=thread&view=me),
     把 uid 自己的主题帖楼层并入;
  3. 与已有档案 data/profiles/<uid>.json 合并(按 uid+page+正文前40字去重),写回 JSON。

本脚本以两条 edge 挂到 graph 上:
  e_collect  stage="collect"  pre_process: 版块最新 N 帖 → by_uid
  e_enrich   stage="enrich"   pre_process: 每人 do=thread 主题补抓 → 最终 by_uid
                             post_process: 合并进档案(profile_dir)并写回 JSON

踩坑记(沿用 s1parse/s1profile 的经验):
- stage1st 匿名态 do=reply 回帖列表要登录,只抓「帖子楼层 + 用户发的主题」,不依赖登录。
- 楼层 div[id^="post_"];uid 在 a[href*="uid="];用户名在 .authi;正文在 td.t_f/[id^="postmessage_"];
  时间在「发表于/回复于」后,新回复是相对时间(刚刚/x分钟前/…),统一解析成时间戳。
- 版块清单只选标题链接 a.xst(一个帖一个),避开回帖数/最后回复锚点等干扰链接。
- 档案去重键 (uid, page, content[:40]),与 s1profile._merge_posts 一致。
- stage1st 是境内站,trust_env=False 直连,不跟随环境代理(否则页面错乱/时间全旧)。
"""

import hashlib
import json
import os
import re
from datetime import datetime, timedelta

import httpx
from bs4 import BeautifulSoup, Tag

_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
    )
}
_BASE = "https://stage1st.com/2b/"
_THREAD_RE = re.compile(r"thread-(\d+)-(\d+)-(\d+)\.html")


# ── 解析层(镜像 plugins/s1parse.py)────────────────────────────────
def _client() -> httpx.AsyncClient:
    """直连客户端:stage1st 走 env 代理会拿到错乱页面,故 trust_env=False。"""
    return httpx.AsyncClient(
        headers=_HEADERS, timeout=30, trust_env=False, follow_redirects=True
    )


def _parse_stage1st_time(text: str | None) -> datetime | None:
    """stage1st 时间字符串 → datetime,支持绝对(2026-8-23 19:01)与相对(刚刚/x分钟前/今天/昨天)。"""
    if not text:
        return None
    text = text.strip()
    m = re.search(r"(\d{4})-(\d{1,2})-(\d{1,2})\s+(\d{1,2}):(\d{2})", text)
    if m:
        try:
            y, mo, d, h, mi = (int(x) for x in m.groups())
            return datetime(y, mo, d, h, mi)
        except ValueError:
            return None
    now = datetime.now()
    if "刚刚" in text:
        return now
    mm = re.search(r"(\d+)\s*分钟前", text)
    if mm:
        return now - timedelta(minutes=int(mm.group(1)))
    hh = re.search(r"(\d+)\s*小时前", text)
    if hh:
        return now - timedelta(hours=int(hh.group(1)))
    dd = re.search(r"(\d+)\s*天前", text)
    if dd:
        return now - timedelta(days=int(dd.group(1)))
    if "昨天" in text:
        base = now - timedelta(days=1)
        tm = re.search(r"(\d{1,2}):(\d{2})", text)
        return base.replace(hour=int(tm.group(1)), minute=int(tm.group(2)), second=0, microsecond=0) if tm else base
    if "今天" in text:
        tm = re.search(r"(\d{1,2}):(\d{2})", text)
        return now.replace(hour=int(tm.group(1)), minute=int(tm.group(2)), second=0, microsecond=0) if tm else now
    return None


def _parse_post(block: Tag) -> dict | None:
    """从单个楼层块抠出 (uid, username, time, content);抠不到返回 None。"""
    a = block.select_one('a[href*="uid="]')
    if not a or not a.get("href"):
        return None
    href = str(a["href"])
    m = re.search(r"uid=(\d+)", href)
    if not m:
        return None
    uid = m.group(1)
    authi = block.select_one(".authi")
    username = authi.get_text(strip=True) if authi else None
    posted_at = None
    tm = re.search(r"(?:发表于|回复于)\s*(.+)", block.get_text("\n", strip=True))
    if tm:
        seg = tm.group(1).strip()
        dt = _parse_stage1st_time(seg)
        if dt is not None and dt.year > 2000:
            posted_at = dt.strftime("%Y-%m-%d %H:%M")
        else:
            posted_at = seg[:40]
    msg = block.select_one("td.t_f, [id^='postmessage_']")
    content = msg.get_text("\n", strip=True) if msg else ""
    if not content:
        return None
    return {"uid": uid, "username": username, "time": posted_at, "content": content}


def _cur_tid(html: str) -> str | None:
    m = _THREAD_RE.search(html)
    return m.group(1) if m else None


def parse_thread_html(html: str, page: int = 1) -> tuple[list[dict], int]:
    """解析一页帖子:返回 (本页楼层列表, 总页数)。"""
    soup = BeautifulSoup(html, "html.parser")
    posts: list[dict] = []
    for block in soup.select('div[id^="post_"]'):
        p = _parse_post(block)
        if p:
            p["page"] = page
            posts.append(p)
    total = page
    tid = _cur_tid(html)
    for href in soup.select('a[href*=".html"]'):
        h = str(href.get("href", ""))
        m = _THREAD_RE.search(h)
        if m and tid and m.group(1) == tid:
            total = max(total, int(m.group(2)))
    return posts, total


async def collect_thread(url: str, max_pages: int | None = None) -> list[dict]:
    """抓一个帖子的所有页,返回去重后的楼层列表。"""
    async with _client() as c:
        r = await c.get(url)
        r.raise_for_status()
        posts, total = parse_thread_html(r.text, 1)
        tid = _cur_tid(r.text)
        if not tid:
            return posts
        pages = total if max_pages is None else min(total, max_pages)
        for n in range(2, pages + 1):
            u = f"{_BASE}thread-{tid}-{n}-1.html"
            rr = await c.get(u)
            if rr.status_code != 200:
                break
            pg, _ = parse_thread_html(rr.text, n)
            posts.extend(pg)
    seen = set()
    out = []
    for p in posts:
        key = (p["uid"], p["page"], hashlib.md5(p["content"].encode()).hexdigest()[:12])
        if key in seen:
            continue
        seen.add(key)
        out.append(p)
    return out


def parse_index(html: str) -> list[dict]:
    """解析版块/个人主页的帖子清单:只选标题链接 a.xst,避免回帖数/锚点干扰。"""
    soup = BeautifulSoup(html, "html.parser")
    out: list[dict] = []
    seen = set()
    for href in soup.select("a.xst"):
        h = str(href.get("href", ""))
        m = _THREAD_RE.search(h)
        if not m or m.group(2) != "1":
            continue
        tid = m.group(1)
        if tid in seen:
            continue
        seen.add(tid)
        title = href.get_text(strip=True)
        if not title:
            continue
        u = h if h.startswith("http") else _BASE + h
        last_reply = None
        node: Tag | None = href
        for _ in range(8):
            node = node.parent
            if node is None:
                break
            rtext = node.get_text(" ", strip=True)
            dates = re.findall(r"\d{4}-\d{1,2}-\d{1,2}\s+\d{1,2}:\d{2}", rtext)
            if dates:
                last_reply = dates[-1]
                break
        out.append({"tid": tid, "title": title, "url": u, "last_reply": last_reply})
    return out


# ── 收集层(镜像 plugins/s1profile._collect)────────────────────────
async def collect_forum_posts(forum_url: str, max_threads: int = 15) -> dict[str, list[dict]]:
    """版块最新 N 帖全部楼层,按 uid 归并(by_uid)。对应 s1profile 的 forum_to_json 部分。"""
    async with _client() as c:
        r = await c.get(forum_url)
        r.raise_for_status()
        threads = parse_index(r.text)[:max_threads]
    by_uid: dict[str, list[dict]] = {}
    for t in threads:
        try:
            posts = await collect_thread(t["url"])
        except Exception as e:
            print(f"# warn: {t['url']} failed: {e}", flush=True)
            continue
        for p in posts:
            by_uid.setdefault(p["uid"], []).append(p)
    return by_uid


async def enrich_own_threads(
    by_uid: dict[str, list[dict]], max_own_threads: int = 10
) -> dict[str, list[dict]]:
    """对每个 uid 补抓其「发的主题」(do=thread),把 uid 自己的主题帖楼层并入。"""
    async with _client() as c:
        for uid in list(by_uid.keys()):
            try:
                r = await c.get(
                    f"{_BASE}home.php?mod=space&uid={uid}&do=thread&view=me&page=1"
                )
                if r.status_code != 200:
                    continue
                threads = parse_index(r.text)
            except Exception:
                continue
            for t in threads[:max_own_threads]:
                try:
                    posts = await collect_thread(t["url"])
                except Exception:
                    continue
                for p in posts:
                    if p["uid"] == uid:
                        by_uid.setdefault(uid, []).append(p)
    return by_uid


# ── 档案合并层(镜像 plugins/s1profile._merge_posts/_save_profile)──
def _profile_path(profile_dir: str, uid: str) -> str:
    return os.path.join(profile_dir, f"{uid}.json")


def _load_profile(profile_dir: str, uid: str) -> dict:
    try:
        with open(_profile_path(profile_dir, uid), "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        return {"uid": uid, "username": None, "posts": [], "revisits": []}


def merge_into_profiles(by_uid: dict[str, list[dict]], profile_dir: str) -> dict[str, int]:
    """把收集到的 by_uid 并入已有档案,按 (uid,page,正文前40字) 去重,写回 JSON。

    返回 {uid: 新增条数}(镜像 s1profile.update_profiles 的收集+合并部分,不做评估)。
    """
    os.makedirs(profile_dir, exist_ok=True)
    result: dict[str, int] = {}
    for uid, posts in by_uid.items():
        prof = _load_profile(profile_dir, uid)
        # 去重键 (uid, page, content[:40])。踩坑:已有档案 posts[] 里并不存 uid 字段
        # (s1profile._merge_posts 存的是 {username,time,page,content,url}),故这里用档案自身的
        # uid 而不是 p["uid"](chatto 原实现第二次运行会 KeyError,这里已修正)。
        uid_key = prof.get("uid") or uid
        seen = {(uid_key, p.get("page"), p["content"][:40]) for p in prof["posts"]}
        added = 0
        for p in posts:
            key = (p["uid"], p.get("page"), p["content"][:40])
            if key in seen:
                continue
            seen.add(key)
            prof["posts"].append({
                "username": p.get("username"),
                "time": p.get("time"),
                "page": p.get("page"),
                "content": p["content"],
                "url": p.get("url"),
            })
            if p.get("username"):
                prof["username"] = p["username"]
            added += 1
        prof["posts"].sort(key=lambda x: x.get("time") or "")
        with open(_profile_path(profile_dir, uid), "w", encoding="utf-8") as f:
            json.dump(prof, f, ensure_ascii=False, indent=2)
        result[uid] = added
    return result


# ── vertex-edge edge script hooks ──────────────────────────────────
async def pre_process(data, settings):
    """按 settings["stage"] 分派收集阶段。``data`` 是该 edge source 顶点传来的值。

    - stage="collect"(e_collect):data 是版块 URL → 抓最新 N 帖 → by_uid
    - stage="enrich" (e_enrich) :data 是上一步的 by_uid → 每人主题帖补抓 → 最终 by_uid
    """
    stage = settings.get("stage", "collect")
    max_threads = int(settings.get("max_threads", 15))
    max_own = int(settings.get("max_own_threads", 10))
    if stage == "enrich":
        return await enrich_own_threads(data, max_own)
    # stage == "collect"(默认):data 是 URL
    return await collect_forum_posts(str(data).strip(), max_threads)


def post_process(data, settings):
    """stage="enrich"(最终边):把 by_uid 合并进档案并写回 JSON,返回摘要。其它阶段透传。"""
    if settings.get("stage") != "enrich":
        return data
    profile_dir = settings.get("profile_dir", "")
    if not profile_dir:
        profile_dir = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "data", "profiles"
        )
    by_uid = data if isinstance(data, dict) else {}
    added = merge_into_profiles(by_uid, profile_dir)
    users = []
    for uid, n in sorted(added.items(), key=lambda kv: -kv[1]):
        prof = _load_profile(profile_dir, uid)
        users.append({
            "uid": uid,
            "username": prof.get("username"),
            "added": n,
            "total": len(prof["posts"]),
        })
    return {
        "url": settings.get("url", ""),
        "collected_at": datetime.now().isoformat(timespec="seconds"),
        "profile_dir": profile_dir,
        "users_collected": len(by_uid),
        "users": users,
        "posts_added": sum(added.values()),
    }
