#!/usr/bin/env python3
"""Run the chatto「去 stage1st 收集帖子」task as a vertex-edge graph.

用法(在项目根目录):
    python examples/s1profile_collect/run.py [--url <forum-url>] [--profile-dir <dir>] [--max-threads N]

流程: forum(source, URL) ─e_collect─▶ posts_pool ─e_enrich─▶ profiles(sink)
  1. e_collect  抓版块最新 N 帖全部楼层,按 uid 归并
  2. e_enrich   补抓每人「发的主题」(do=thread),合并进 data/profiles/<uid>.json(去重)
全程透明透传,不调用 LLM(0 token)。

--profile-dir 默认写到 examples/s1profile_collect/data/profiles(不动 chatto-bot 真实档案);
想直接更新 chatto-bot 的真实档案可传 /mnt/d/WorkPlace/chatto-bot/data/profiles。
"""

import argparse
import asyncio
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from framework import Graph, Executor, MockAgent

HERE = os.path.dirname(os.path.abspath(__file__))


async def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default=None, help="覆盖版块 URL")
    ap.add_argument("--profile-dir", default=None, help="档案输出目录(默认 examples/s1profile_collect/data/profiles)")
    ap.add_argument("--max-threads", type=int, default=None, help="抓最新几个帖(默认 3)")
    args = ap.parse_args()

    config_path = os.path.join(HERE, "config.json")
    graph = Graph.from_json(config_path)

    if args.url:
        await graph.vertices["forum"].set_data("url", args.url)
    if args.profile_dir:
        # 两条边的 profile_dir / url 都同步覆盖
        for eid in ("e_collect", "e_enrich"):
            edge = graph.edges[eid]
            if args.url:
                edge.settings["url"] = args.url
            if args.profile_dir and eid == "e_enrich":
                edge.settings["profile_dir"] = args.profile_dir
        if args.url:
            edge = graph.edges["e_collect"]
    if args.max_threads:
        graph.edges["e_collect"].settings["max_threads"] = args.max_threads

    executor = Executor(graph, agents=MockAgent(), max_concurrency=4, timeout=300)
    result = await executor.run()
    print(result.summary())

    edge_res = result.edge_results.get("e_enrich")
    if result.success and edge_res is not None:
        import json

        print(f"\n# 收集结果摘要: {edge_res.get('users_collected')} 位用户, 新增 {edge_res.get('posts_added')} 帖")
        print(f"# 档案目录: {edge_res.get('profile_dir')}")
        for u in edge_res.get("users", [])[:10]:
            print(f"    uid={u['uid']} name={u['username']} added={u['added']} total={u['total']}")
        return 0
    print("# 执行未成功或 e_enrich 无输出", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
