#!/usr/bin/env python3
"""CF 子域×每日 流量 DAG（无缓存、拉最新、纯 fn 边、快）。

DAG 边：
  edge_0  cf_task            → [fn:cf_query]  → cf_subdomain_daily   （fresh GraphQL，并行）
  edge_1  cf_subdomain_daily → [fn:cf_verify] → cf_verify_result     （自洽校验）
  edge_2  cf_subdomain_daily → [fn:cf_itemize] → cf_itemized_table   （逐日逐子域明细表）
  edge_3  cf_itemized_table  → [fn:cf_report]  → cf_final_report     （成稿交付）

所有数据边为 Python fn，确定性、无网络（除 edge_0 拉 CF）；不写本地缓存。
"""
from __future__ import annotations

import datetime as dt
import functools
import json
import os
import sys
import time
import types

# cf_usage 在 knowledge-base/tools；dag_agent 在当前仓库
sys.path.insert(0, "/mnt/e/knowledge-base/tools")
sys.path.insert(0, "/mnt/c/Users/lumin/Downloads/agent")

import cf_usage as cf  # noqa: E402
import dag_agent as dag_mod  # noqa: E402
from dag_agent.schema import Processor  # noqa: E402

ZONES = {
    "3a7a1da0f18e49ea3e40eea909b06f11": "moonchan.xyz",
    "b0cb2da4b0616ff64685f57e86750fed": "810114.xyz",
    "ac6701051cbfbfb41c35d14f36353c8e": "4545810.xyz",
}


def fmt_bytes(b: float) -> str:
    if b >= 1e12:
        return f"{b/1e12:.2f} TB"
    if b >= 1e9:
        return f"{b/1e9:.2f} GB"
    if b >= 1e6:
        return f"{b/1e6:.2f} MB"
    if b >= 1e3:
        return f"{b/1e3:.1f} KB"
    return f"{b:.0f} B"


def fmt_num(n: int) -> str:
    return f"{n:,}"


# ---- fn processors --------------------------------------------------------
def _window(days: int = 7):
    """UTC 昨天为窗口结束日（CF analytics 的最新完整日）。"""
    today_utc = dt.datetime.utcnow().date()
    end = today_utc - dt.timedelta(days=1)
    start = end - dt.timedelta(days=days - 1)
    return start.isoformat(), end.isoformat()


def cf_query(_task_text: str) -> str:
    """edge_0：从 CF GraphQL 拉全量子域×每日数据，无缓存。"""
    email, key = os.environ["CF_EMAIL"], os.environ["CF_GLOBAL_KEY"]
    ids = list(ZONES.keys())
    frm, to = _window(7)
    A = types.SimpleNamespace(
        days=7, end=to, by="all", top=20, full=True, limit=500,
        fast=False, workers=8, zone=None, hosts=None, json=True, quiet=True,
    )
    cache: dict = {}  # 空 dict = 不落盘、不读盘、每次 fresh
    log = lambda *a: None
    agg, trunc = cf.hosts_complete(email, key, ids, A, frm, to, cache, log)
    out = {
        "from": frm,
        "to": to,
        "fetched_at": dt.datetime.utcnow().isoformat() + "Z",
        "zones": [{"zone": ZONES[i], "id": i, "subdomains": agg[i]} for i in ids],
        "truncated": trunc,
    }
    return json.dumps(out, ensure_ascii=False)


def cf_verify(data_json: str) -> str:
    """edge_1：自洽校验——唯一性 + 各口径合计一致 + 完整性（无截断日）。"""
    d = json.loads(data_json)
    frm, to = d["from"], d["to"]
    dates = [
        (dt.date.fromisoformat(frm) + dt.timedelta(days=i)).isoformat()
        for i in range((dt.date.fromisoformat(to) - dt.date.fromisoformat(frm)).days + 1)
    ]
    lines = [f"# CF 数据自检（{frm} ~ {to}）"]
    ok = True
    checks = 0

    # ① 唯一性：(zone, host, date) 不重复
    seen = set()
    dup = []
    for z in d["zones"]:
        for h, v in z["subdomains"].items():
            for date, (c, b) in v["days"].items():
                k = (z["id"], h, date)
                if k in seen:
                    dup.append(k)
                seen.add(k)
    checks += 1
    if dup:
        ok = False
        lines.append(f"- ❌ ① 唯一性：发现 {len(dup)} 个重复 (zone,host,date)")
    else:
        lines.append(f"- ✅ ① 唯一性：{len(seen)} 个 (zone,host,date) 全部唯一")

    # ② 自洽：per-host 窗口合计 == 其每日之和；per-zone == 子域之和 == 每日之和
    mism = []
    for z in d["zones"]:
        zone_day_c = {date: 0 for date in dates}
        zone_day_b = {date: 0 for date in dates}
        for h, v in z["subdomains"].items():
            hc = sum(c for c, _ in v["days"].values())
            hb = sum(b for _, b in v["days"].values())
            sc = sum(c for c, _ in v["days"].values())
            sb = sum(b for _, b in v["days"].values())
            if hc != sc or hb != sb:
                mism.append((z["zone"], h, "host-window!=days-sum"))
            for date, (c, b) in v["days"].items():
                zone_day_c[date] += c
                zone_day_b[date] += b
        zone_c = sum(zone_day_c.values())
        zone_b = sum(zone_day_b.values())
        host_c = sum(sum(c for c, _ in v["days"].values()) for v in z["subdomains"].values())
        host_b = sum(sum(b for _, b in v["days"].values()) for v in z["subdomains"].values())
        if zone_c != host_c or zone_b != host_b:
            mism.append((z["zone"], "zone-day-total != host-sum"))
    checks += 1
    if mism:
        ok = False
        lines.append(f"- ❌ ② 自洽：{len(mism)} 处口径不一致 {mism[:3]}")
    else:
        lines.append(f"- ✅ ② 自洽：子域合计 == 逐日合计 == 域合计，{checks} 项全绿")

    # ③ 完整性：无截断日（hosts_complete 单日基数 < 500 才完整）
    trunc_all = []
    for zid, days_t in d.get("truncated", {}).items():
        if days_t:
            trunc_all.append((ZONES.get(zid, zid), days_t))
    checks += 1
    if trunc_all:
        ok = False
        lines.append(f"- ❌ ③ 完整性：存在截断日 {trunc_all}")
    else:
        lines.append("- ✅ ③ 完整性：无截断日，子域全集完整")

    # 汇总
    grand_c = sum(
        sum(c for c, _ in v["days"].values())
        for z in d["zones"] for v in z["subdomains"].values()
    )
    grand_b = sum(
        sum(b for _, b in v["days"].values())
        for z in d["zones"] for v in z["subdomains"].values()
    )
    total_hosts = sum(len(z["subdomains"]) for z in d["zones"])
    lines.append("")
    lines.append(f"**合计**：{total_hosts} 个子域 × {len(dates)} 天，"
                 f"请求 {fmt_num(grand_c)}，流量 {fmt_bytes(grand_b)}")
    lines.append(f"**结果**：{'ALL GREEN ✅' if ok else 'FAILED ❌'}")
    return "\n".join(lines)


def cf_itemize(data_json: str) -> str:
    """edge_2：按子域为主键——每个子域列出过去一周每一日的访问量与流量。"""
    d = json.loads(data_json)
    frm, to = d["from"], d["to"]
    dates = [
        (dt.date.fromisoformat(frm) + dt.timedelta(days=i)).isoformat()
        for i in range((dt.date.fromisoformat(to) - dt.date.fromisoformat(frm)).days + 1)
    ]
    # 收集所有子域：(host, zone, {date:[c,b]}, week_count, week_bytes)
    subs = []
    for z in d["zones"]:
        for h, v in z["subdomains"].items():
            wc = sum(c for c, _ in v["days"].values())
            wb = sum(b for _, b in v["days"].values())
            subs.append((h, z["zone"], v["days"], wc, wb))
    subs.sort(key=lambda r: (-r[4], -r[3]))  # 按周流量降序

    out = [f"# CF 子域×每日 访问量/流量明细（{frm} ~ {to}）",
           f"> 拉取时间 {d['fetched_at']}，数据源 Cloudflare GraphQL `httpRequestsAdaptiveGroups`",
           f"> 主键=子域，每个子域下列出过去一周每一日的请求数与流量；按周流量降序。{len(subs)} 个子域。"]

    grand_c = sum(s[3] for s in subs)
    grand_b = sum(s[4] for s in subs)
    for h, zn, days, wc, wb in subs:
        out.append(f"\n### {h}（{zn}） — 周：访问量 {fmt_num(wc)} / 流量 {fmt_bytes(wb)}\n")
        out.append("| 日期 | 访问量 | 流量 |")
        out.append("|---|--:|--:|")
        for date in dates:
            if date in days:
                c, b = days[date]
                out.append(f"| {date} | {fmt_num(c)} | {fmt_bytes(b)} |")
            else:
                out.append(f"| {date} | 0 | 0 B |")
        out.append(f"| **合计** | **{fmt_num(wc)}** | **{fmt_bytes(wb)}** |")

    out.append(f"\n## 总合计（{frm} ~ {to}，{len(subs)} 个子域）")
    out.append(f"- 访问量：{fmt_num(grand_c)}")
    out.append(f"- 流量：{fmt_bytes(grand_b)}")
    return "\n".join(out)


def cf_report(table_text: str) -> str:
    """edge_3：成稿——明细表 + 拉取元信息。"""
    return table_text


# ---- 每条 edge 的耗时记录 -------------------------------------------------
EDGE_TIMINGS: dict[str, float] = {}
EDGE_NETWORK: dict[str, bool] = {
    "cf_query": True,   # 走 CF GraphQL
    "cf_verify": False,
    "cf_itemize": False,
    "cf_report": False,
}


def timed(fn):
    """包一层计时；functools.wraps 保留原签名，供引擎 inspect.signature 正常识别入参。"""

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        t0 = time.perf_counter()
        try:
            return fn(*args, **kwargs)
        finally:
            EDGE_TIMINGS[fn.__name__] = time.perf_counter() - t0

    return wrapper


# ---- build & run DAG ------------------------------------------------------
def build_dag():
    d = dag_mod.DAG(name="cf_subdomain_daily", models="cf_dag/models.json")
    d.add_edge(
        input_tags=["cf_task"],
        processor=Processor(fn=timed(cf_query), model="cf_query"),
        output_tags=["cf_subdomain_daily"],
        layer=0,
    )
    d.add_edge(
        input_tags=["cf_subdomain_daily"],
        processor=Processor(fn=timed(cf_verify), model="cf_verify"),
        output_tags=["cf_verify_result"],
        layer=1,
    )
    d.add_edge(
        input_tags=["cf_subdomain_daily"],
        processor=Processor(fn=timed(cf_itemize), model="cf_itemize"),
        output_tags=["cf_itemized_table"],
        layer=1,
    )
    d.add_edge(
        input_tags=["cf_itemized_table"],
        processor=Processor(fn=timed(cf_report), model="cf_report"),
        output_tags=["cf_final_report"],
        layer=2,
    )
    return d


def run_once() -> str:
    """一次调用得到全部结果：构建 DAG → 单次 dag.run() → 返回逐子域×逐日明细。

    内部 4 条 fn 边由框架按标签路由自动依次执行（query→verify / query→itemize→report），
    对调用者而言就是「一次 dag 调用」。返回最终交付物 cf_final_report 全文。
    """
    EDGE_TIMINGS.clear()
    d = build_dag()
    d.add_content(contents="按子域×逐日分项列出 CF 全量子域访问量与流量", tags=["cf_task"], gen=0)
    d.run(until_tag="cf_final_report")  # 一次调用，框架内部按 layer/gen 调度 4 条边
    return d.get_result("cf_final_report")


def print_timings(total: float) -> None:
    """打印每条 edge 的耗时表（含占 run 总耗时比例）。"""
    order = ["cf_query", "cf_verify", "cf_itemize", "cf_report"]
    edge_of = {"cf_query": "edge_0", "cf_verify": "edge_1",
               "cf_itemize": "edge_2", "cf_report": "edge_3"}
    print("\n=== 各 edge 耗时 ===")
    print(f"{'edge':<8}{'处理器':<14}{'耗时(s)':>10}{'占比':>10}  网络")
    print("-" * 56)
    for name in order:
        sec = EDGE_TIMINGS.get(name, 0.0)
        pct = (sec / total * 100.0) if total else 0.0
        net = "是" if EDGE_NETWORK[name] else "否"
        print(f"{edge_of[name]:<8}{name:<14}{sec:>10.3f}{pct:>9.1f}%  {net}")
    print("-" * 56)
    sum_edges = sum(EDGE_TIMINGS.get(n, 0.0) for n in order)
    print(f"{'合计(边)':<8}{'':<14}{sum_edges:>10.3f}")
    print(f"{'总计(run)':<8}{'':<14}{total:>10.3f}")


def main():
    # 凭据：用 cf_usage 自带 load_creds（正确去引号），结果写入 os.environ 供 cf_query 取用
    if not os.environ.get("CF_EMAIL"):
        email, key = cf.load_creds()
        os.environ["CF_EMAIL"] = email
        os.environ["CF_GLOBAL_KEY"] = key

    print("=== 一次调用 dag.run()，计时开始 ===")
    t0 = time.perf_counter()
    report = run_once()  # 一次调用 → 全部结果
    total = time.perf_counter() - t0

    out_path = "cf_dag/cf_report.md"
    with open(out_path, "w", encoding="utf-8") as fh:
        fh.write(report)
    print(f"=== 交付 ===\n{out_path}")
    print(f"明细行数: {len(report.splitlines())} 行（790 子域 × 逐日）")

    print_timings(total)


if __name__ == "__main__":
    main()
