"""python -m dag8 的命令行入口。

    python -m dag8 run configs/demo.json                 # 用配置里的 seed
    python -m dag8 run configs/demo.json --seed "新的任务"  # 覆盖 seed
    python -m dag8 run configs/demo.json --dry-run       # 不调 pi，只验配置与占位符
    python -m dag8 show <run_id>                         # 看一次运行的全部内容
    python -m dag8 runs                                  # 列出最近的运行
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from dag8 import run as run_dag
from dag8.store import Store


def _cmd_run(args: argparse.Namespace) -> int:
    seed = args.seed
    if args.seed_file:
        seed = Path(args.seed_file).read_text(encoding="utf-8")
    if args.check_model:
        return _check_model(args.config)
    if args.preflight:
        from dag8.config import load as load_config
        from dag8.doctor import _check_http

        try:
            cfg = load_config(args.config)
        except (ValueError, OSError) as exc:
            print(f"预检失败：{exc}")
            return 2
        check = _check_http(cfg.pi.provider, 5.0)
        stream = sys.stderr if args.json else sys.stdout
        if not check.ok:
            print(f"预检失败：{check.detail}", file=stream)
            if check.hint:
                print(f"建议：{check.hint}", file=stream)
            return 1
        print(f"预检通过：{check.detail}", file=stream)
    if args.lint:
        from dag8.config import load as load_config
        from dag8.lint import lint_config

        issues = lint_config(load_config(args.config))
        errors = [i for i in issues if i.level == "error"]
        for it in issues:
            mark = "✗" if it.level == "error" else "!"
            where = f"[{it.edge_id}] " if it.edge_id else ""
            print(f"{mark} {where}{it.message}")
        if errors:
            print(f"lint 发现 {len(errors)} 个错误，已中止")
            return 1
    try:
        res = run_dag(args.config, seed_content=seed, db_path=args.db,
                      dry_run=args.dry_run, max_workers=args.workers,
                      rpm=args.rpm, rate_window=args.rate_window,
                      max_seconds=args.max_seconds,
                      use_cache=args.cache, cache_dir=args.cache_dir,
                      refresh_cache=args.refresh, cache_ttl=args.cache_ttl,
                      resume=args.resume, resume_last=args.resume_last,
                      keep_going=args.keep_going, adaptive=args.adaptive,
                      max_cost=args.max_cost, auto_model=args.auto_model)
    except (ValueError, OSError) as exc:
        print(f"运行失败：{exc}")
        return 2
    if args.json:
        payload = {
            "run_id": res.run_id,
            "dag": res.dag_name,
            "status": res.status,
            "result_tag": res.result_tag,
            "result": res.result,
            "db": res.db_path,
            "workers_used": res.workers_used,
            "cost_estimate": res.cost_estimate,
            "edges": [
                {"edge": e.edge_id, "layer": e.layer, "status": e.status,
                 "elapsed": e.elapsed, "cached": e.cached, "error": e.error}
                for e in res.edges
            ],
        }
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return 0 if res.status == "ok" else 1
    print(res.summary())
    print(f"\ndb: {res.db_path}")
    if res.result:
        print("\n--- result ---")
        print(res.result)
    return 0 if res.status == "ok" else 1


def _check_model(config_path: str) -> int:
    """只验 provider/model 真能跑，不跑边。

    必要性：`pi` 对未注册的模型只打一行 Warning 就按 custom model id 透传，
    退出码仍是 0。模型名拼错时 v8 看不出异常，只能等跑到一半才炸。
    这里先发一次最小请求，让问题在开跑之前暴露。
    """
    from dag8.config import load as load_config
    from dag8.pi import PiAgent, PiError

    cfg = load_config(config_path)
    # 预检只试一次：坏模型的重试代价是 timeout×retries（实测 240s×3 要三分半），
    # 而预检的目的就是尽快暴露问题，不值得为它付重试的钱。
    cfg.pi.retries = 1
    agent = PiAgent(cfg.pi)
    target = f"{cfg.pi.provider}/{cfg.pi.model}"
    print(f"检查 {target} ...", flush=True)
    try:
        out = agent.run("reply with the single word ok", timeout=min(cfg.pi.timeout, 60))
    except PiError as exc:
        print(f"✗ {target} 不可用：{exc}")
        print("\n可用模型见 v8/MODELS.md")
        return 1
    print(f"✓ {target} 可用（返回 {len(out)} 字符）")
    return 0


def _cmd_show(args: argparse.Namespace) -> int:
    with Store(args.db) as store:
        try:
            info = store.run_info(args.run_id)
        except KeyError:
            print(f"run 不存在：{args.run_id}")
            return 1
        if args.json:
            payload = {
                **{k: info[k] for k in
                   ("run_id", "dag_name", "status", "started_at", "finished_at",
                    "result_tag", "error")},
                "edges": [
                    {"edge": e["edge_id"], "layer": e["layer"], "status": e["status"],
                     "model": e["model"], "elapsed": e["elapsed"],
                     "cached": bool(e["cached"]), "error": e["error"]}
                    for e in store.edge_records(args.run_id)
                ],
                "contents": [
                    {"id": it["id"], "tags": it["tags"], "gen": it["gen"],
                     "chars": len(it["content"]), "source_edge": it["source_edge"]}
                    for it in store.all_contents(args.run_id)
                ],
            }
            print(json.dumps(payload, ensure_ascii=False, indent=2))
            return 0
        print(json.dumps({k: info[k] for k in
                          ("run_id", "dag_name", "status", "started_at", "finished_at",
                           "result_tag", "error")}, ensure_ascii=False, indent=2))
        print("\n--- edges ---")
        for e in store.edge_records(args.run_id):
            print(f"  [{e['layer']}] {e['edge_id']}  {e['status']}  "
                  f"{e['model'] or '-'}  {e['elapsed'] or 0:.1f}s"
                  + (f"  ERR {e['error'][:100]}" if e["error"] else ""))
        print("\n--- contents ---")
        for it in store.all_contents(args.run_id):
            tags = ",".join(it["tags"])
            body = it["content"]
            print(f"  #{it['id']} gen={it['gen']} tags=[{tags}] via {it['source_edge'] or 'seed'}"
                  f"  {len(body)} 字符")
    return 0


def _cmd_runs(args: argparse.Namespace) -> int:
    with Store(args.db) as store:
        rows = store.conn.execute(
            "SELECT run_id, dag_name, status, started_at, "
            "(SELECT COUNT(*) FROM contents c WHERE c.run_id=r.run_id) AS n "
            "FROM runs r ORDER BY started_at DESC LIMIT ?",
            (args.limit,),
        ).fetchall()
        if args.json:
            print(json.dumps([dict(r) for r in rows], ensure_ascii=False, indent=2))
            return 0
        if not rows:
            print("还没有任何运行记录")
            return 0
        print(f"{'run_id':14} {'dag':20} {'status':10} {'contents':>8}")
        for r in rows:
            print(f"{r['run_id']:14} {r['dag_name'][:20]:20} {r['status']:10} {r['n']:>8}")
    return 0


def _cmd_scaffold(args: argparse.Namespace) -> int:
    from dag8.scaffold import (build_composed_config, build_composed_review_config,
                               build_config, build_review_config, changed_modules,
                               scan_modules, write_config)

    root = Path(args.root).resolve()
    if args.changed:
        modules = changed_modules(root)
    else:
        modules = scan_modules(root, depth=args.depth, mode=args.mode)
    if not modules:
        print(f"没有在 {root} 扫描到模块")
        return 1
    if args.compose:
        builder = (build_composed_review_config if args.review
                   else build_composed_config)
        cfg, subgraphs = builder(
            root, modules, provider=args.provider, model=args.model,
            subdir=args.graphs_dir, max_modules=args.max_modules)
        out_parent = Path(args.out).parent
        for name, sub in subgraphs.items():
            write_config(sub, out_parent / args.graphs_dir / name)
    elif args.review:
        cfg = build_review_config(root, modules, provider=args.provider,
                                  model=args.model, max_modules=args.max_modules)
    else:
        cfg = build_config(root, modules, inspector=args.inspector,
                           provider=args.provider, model=args.model,
                           max_modules=args.max_modules)
    out = write_config(cfg, args.out)
    print(f"模块 {len(cfg['edges']) - 1} 个：")
    for e in cfg["edges"][:-1]:
        desc = (e.get("command") or e.get("prompt")
                or (e.get("graph") or {}).get("path") or "")
        print(f"  - {e['output_tags'][0]:20s} {desc[:100]}")
    print(f"\n配置：{out}")
    print(f"跑法：python -m dag8 run {out} --workers 2 --rpm 3 --preflight")
    return 0


def _cmd_explain(args: argparse.Namespace) -> int:
    from dag8.explain import explain, format_report

    with Store(args.db) as store:
        try:
            data = explain(store, args.run_id)
        except KeyError:
            print(f"run 不存在：{args.run_id}")
            return 1
    if args.json:
        print(json.dumps(data, ensure_ascii=False, indent=2))
    else:
        print(format_report(data))
    return 0


def _cmd_diff(args: argparse.Namespace) -> int:
    from dag8.diff import diff_runs, format_report

    with Store(args.db) as store:
        try:
            data = diff_runs(store, args.run_a, args.run_b)
        except KeyError as exc:
            print(f"run 不存在：{exc}")
            return 1
    if args.json:
        print(json.dumps(data, ensure_ascii=False, indent=2))
    else:
        print(format_report(data))
    return 0


def _cmd_route(args: argparse.Namespace) -> int:
    from dag8.router import choose_model

    with Store(args.db) as store:
        best = choose_model(store, min_calls=args.min_calls,
                            price_weight=args.price_weight)
    if not best:
        print("没有足够的历史数据来路由")
        return 1
    if args.json:
        print(json.dumps(best, ensure_ascii=False, indent=2))
    else:
        print(f"推荐 {best['key']}：ok {best['ok_rate'] * 100:.0f}% "
              f"p50 {best['p50']:.1f}s n={best['n']} score={best['score']}")
    return 0


def _cmd_models(args: argparse.Namespace) -> int:
    from dag8.models import format_report, load_catalog, scorecard

    with Store(args.db) as store:
        rows = scorecard(store, load_catalog(args.providers), args.limit)
    if args.json:
        print(json.dumps(rows, ensure_ascii=False, indent=2))
    else:
        print(format_report(rows))
    return 0


def _cmd_stats(args: argparse.Namespace) -> int:
    from dag8.stats import collect, format_report

    with Store(args.db) as store:
        data = collect(store, args.limit)
    if args.json:
        print(json.dumps(data, ensure_ascii=False, indent=2))
    else:
        print(format_report(data))
    return 0


def _cmd_convert(args: argparse.Namespace) -> int:
    import json as _json

    from dag8.convert import convert_file, convert_v7_to_graphs

    if args.mode == "graph":
        try:
            data = _json.loads(Path(args.source).read_text(encoding="utf-8"))
            parent, subs = convert_v7_to_graphs(
                data, subdir=args.graphs_dir, provider=args.provider,
                result_tag=args.result_tag, name=args.name)
        except (OSError, ValueError) as exc:
            print(f"转换失败：{exc}")
            return 1
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        for name, sub in subs.items():
            sp = out.parent / args.graphs_dir / name
            sp.parent.mkdir(parents=True, exist_ok=True)
            sp.write_text(_json.dumps(sub, ensure_ascii=False, indent=2) + "\n",
                          encoding="utf-8")
        out.write_text(_json.dumps(parent, ensure_ascii=False, indent=2) + "\n",
                       encoding="utf-8")
        print(f"v7 图 → {out}（{len(subs)} 张单边子图，目录 {args.graphs_dir}）")
        return 0
    try:
        out = convert_file(args.source, args.out, provider=args.provider,
                           result_tag=args.result_tag, name=args.name)
    except (OSError, ValueError) as exc:
        print(f"转换失败：{exc}")
        return 1
    print(f"v7 图 → {out}")
    return 0


def _cmd_cost(args: argparse.Namespace) -> int:
    from dag8.cost import format_report, load_prices, run_costs

    prices = load_prices(args.providers)
    with Store(args.db) as store:
        rows = run_costs(store, prices, args.limit)
    if args.json:
        print(json.dumps(rows, ensure_ascii=False, indent=2))
    else:
        print(format_report(rows))
    return 0


def _cmd_demo(args: argparse.Namespace) -> int:
    import shutil

    from dag8.demo import run_demo

    res, tmp = run_demo(args.workers)
    print(res.summary())
    if res.result:
        print(f"\nresult: {res.result}")
    if args.keep:
        print(f"\n配置与 db 保留在：{tmp}")
    else:
        shutil.rmtree(tmp, ignore_errors=True)
    return 0 if res.status == "ok" else 1


def _cmd_abort(args: argparse.Namespace) -> int:
    with Store(args.db) as store:
        try:
            store.run_info(args.run_id)
        except KeyError:
            print(f"run 不存在：{args.run_id}")
            return 1
        changed = store.abort_run(args.run_id)
    if changed:
        print("已标记 interrupted；可用 --resume-last 续跑")
        return 0
    print("该 run 不是 running 状态，未改动")
    return 1


def _cmd_prune(args: argparse.Namespace) -> int:
    with Store(args.db) as store:
        n = store.prune(args.keep, dry_run=not args.apply)
    if args.apply:
        print(f"已删除 {n} 个旧 run（保留最近 {args.keep}）")
    else:
        print(f"将删除 {n} 个旧 run（保留最近 {args.keep}）；加 --apply 执行")
    return 0


def _cmd_cache(args: argparse.Namespace) -> int:
    from dag8.cache import NodeCache

    cache = NodeCache(args.dir)
    if args.clear:
        print(f"清空 {cache.clear()} 条缓存")
        return 0
    if args.evict_ttl is not None:
        print(f"按 TTL={args.evict_ttl}s 清理 {cache.evict(args.evict_ttl)} 条")
        return 0
    info = cache.info()
    if args.json:
        print(json.dumps(info, ensure_ascii=False, indent=2))
    else:
        print(f"缓存：{info['entries']} 条 / {info['bytes']} 字节  ({cache.dir})")
    return 0


def _cmd_lint(args: argparse.Namespace) -> int:
    from dag8.config import load as load_config
    from dag8.lint import lint_config

    try:
        issues = lint_config(load_config(args.config))
    except (ValueError, OSError) as exc:
        print(f"配置加载失败：{exc}")
        return 2
    errors = [i for i in issues if i.level == "error"]
    warnings = [i for i in issues if i.level == "warning"]
    if args.json:
        print(json.dumps({
            "errors": len(errors), "warnings": len(warnings),
            "issues": [{"level": i.level, "edge": i.edge_id, "message": i.message}
                       for i in issues],
        }, ensure_ascii=False, indent=2))
        return 1 if errors or (args.strict and warnings) else 0
    for it in issues:
        mark = "✗" if it.level == "error" else "!"
        where = f"[{it.edge_id}] " if it.edge_id else ""
        print(f"{mark} {where}{it.message}")
    if not issues:
        print("✓ 配置检查通过")
    else:
        print(f"\n{len(errors)} errors / {len(warnings)} warnings")
    return 1 if errors or (args.strict and warnings) else 0


def _cmd_doctor(args: argparse.Namespace) -> int:
    from dag8.doctor import check_all, format_report

    checks = check_all(args.provider, args.model, args.timeout,
                       skip_gateway=args.skip_gateway)
    if args.json:
        print(json.dumps([c.__dict__ for c in checks], ensure_ascii=False, indent=2))
    else:
        print(format_report(checks))
    return 0 if all(c.ok for c in checks) else 1


def _cmd_new(args: argparse.Namespace) -> int:
    from dag8.templates import build, list_templates

    if args.list:
        for name, desc in list_templates().items():
            print(f"  {name:14s} {desc}")
        return 0
    if not args.template or not args.out:
        print("需要模板名与 --out（或使用 --list 查看模板）")
        return 2
    try:
        cfg = build(args.template, provider=args.provider, model=args.model,
                    seed=args.seed, seed_tag=args.seed_tag)
    except ValueError as exc:
        print(exc)
        return 1
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(cfg, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"模板 {args.template} → {out}")
    print(f"跑法：python -m dag8 run {out}")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="dag8", description="JSON 配置 + pi agent + sqlite3")
    ap.add_argument("--db", default=None, help="sqlite3 路径（默认 v8/data/dag8.sqlite3）")
    sub = ap.add_subparsers(dest="cmd", required=True)

    rp = sub.add_parser("run", help="跑一张配置好的图")
    rp.add_argument("--db", default=argparse.SUPPRESS, help="sqlite 路径（等价于全局 --db）")
    rp.add_argument("config")
    rp.add_argument("--seed", default=None, help="覆盖配置里的 seed_content")
    rp.add_argument("--seed-file", default=None, help="从文件读 seed（长文本用，优先级高于 --seed）")
    rp.add_argument("--dry-run", action="store_true", help="不调 pi，把渲染后的 prompt 当输出")
    rp.add_argument("--workers", type=int, default=1,
                    help="同一波次的并发节点数（默认 1 = 串行；>1 时并行跑互不依赖的节点）")
    rp.add_argument("--rpm", type=int, default=None,
                    help="API 节点在窗口内的最大调用数（默认不限；用于躲 429）")
    rp.add_argument("--rate-window", type=float, default=60.0,
                    help="限速窗口秒数（默认 60）")
    rp.add_argument("--max-seconds", type=float, default=None,
                    help="整次运行的墙钟上限；超了停止调度，剩余边标 skipped")
    rp.add_argument("--lint", action="store_true",
                    help="跑之前先做配置静态检查；有 error 直接中止")
    rp.add_argument("--preflight", action="store_true",
                    help="开跑前先做一次快速 HTTP 预检；网关不可用直接退出")
    rp.add_argument("--cache", action="store_true",
                    help="启用节点结果缓存（按 节点+输入+模型 命中，命中则不调 pi）")
    rp.add_argument("--cache-dir", default=None, help="缓存目录（默认 v8/data/cache）")
    rp.add_argument("--refresh", action="store_true",
                    help="忽略已有缓存但仍写入（配合 --cache）")
    rp.add_argument("--cache-ttl", type=float, default=None,
                    help="缓存有效期（秒）；超时视为未命中")
    rp.add_argument("--resume", default=None,
                    help="续跑指定 run：跳过已成功的边，只补未完成的（需同一个 --db）")
    rp.add_argument("--resume-last", action="store_true",
                    help="自动续跑同一配置最近一个未成功的 run（崩溃/中断后）")
    rp.add_argument("--keep-going", action="store_true",
                    help="有边失败时继续跑其它不受影响的分支；结果状态为 partial/error")
    rp.add_argument("--auto-model", action="store_true",
                    help="按 db 历史 stats + providers.json 价格自动选模型（覆盖 pi.model）")
    rp.add_argument("--max-cost", type=float, default=None,
                    help="整次运行的估算成本上限（按 providers.json 价格）；超了停止调度")
    rp.add_argument("--adaptive", action="store_true",
                    help="自适应并发：某波次出现 429 就把下一波 worker 减半，成功后再加回")
    rp.add_argument("--check-model", action="store_true",
                    help="只验证 provider/model 能否真的调通，不跑边")
    rp.add_argument("--json", action="store_true",
                    help="只输出 JSON（run_id/status/result/edges），便于 CI/脚本")
    rp.set_defaults(func=_cmd_run)

    sp = sub.add_parser("show", help="查看一次运行")
    sp.add_argument("--db", default=argparse.SUPPRESS, help="sqlite 路径（等价于全局 --db）")
    sp.add_argument("run_id")
    sp.add_argument("--json", action="store_true", help="机器可读输出")
    sp.set_defaults(func=_cmd_show)

    lp = sub.add_parser("runs", help="列出最近的运行")
    lp.add_argument("--limit", type=int, default=20)
    lp.add_argument("--json", action="store_true", help="机器可读输出")
    lp.set_defaults(func=_cmd_runs)

    ep = sub.add_parser("explain", help="解释一次运行为什么失败/卡住，并给建议")
    ep.add_argument("--db", default=argparse.SUPPRESS, help="sqlite 路径（等价于全局 --db）")
    ep.add_argument("run_id")
    ep.add_argument("--json", action="store_true")
    ep.set_defaults(func=_cmd_explain)

    dp = sub.add_parser("diff", help="对比两次运行：边状态/时延/模型/结果长度")
    dp.add_argument("--db", default=argparse.SUPPRESS, help="sqlite 路径（等价于全局 --db）")
    dp.add_argument("run_a")
    dp.add_argument("run_b")
    dp.add_argument("--json", action="store_true")
    dp.set_defaults(func=_cmd_diff)

    rtp = sub.add_parser("route", help="按历史 stats + 价格推荐模型")
    rtp.add_argument("--db", default=argparse.SUPPRESS, help="sqlite 路径（等价于全局 --db）")
    rtp.add_argument("--min-calls", type=int, default=1)
    rtp.add_argument("--price-weight", type=float, default=0.0)
    rtp.add_argument("--json", action="store_true")
    rtp.set_defaults(func=_cmd_route)

    mp = sub.add_parser("models", help="模型记分卡：provider 价格 + 实测成功率/时延")
    mp.add_argument("--db", default=argparse.SUPPRESS, help="sqlite 路径（等价于全局 --db）")
    mp.add_argument("--providers", default=None)
    mp.add_argument("--limit", type=int, default=50)
    mp.add_argument("--json", action="store_true")
    mp.set_defaults(func=_cmd_models)

    sp = sub.add_parser("stats", help="历史运行统计：模型成功率/时延/缓存/错误 Top")
    sp.add_argument("--db", default=argparse.SUPPRESS, help="sqlite 路径（等价于全局 --db）")
    sp.add_argument("--limit", type=int, default=50)
    sp.add_argument("--json", action="store_true")
    sp.set_defaults(func=_cmd_stats)

    cv = sub.add_parser("convert", help="v7 图 → v8 配置（摊平为可波次并行的图）")
    cv.add_argument("source")
    cv.add_argument("--out", required=True)
    cv.add_argument("--provider", default="laptop")
    cv.add_argument("--result-tag", default=None)
    cv.add_argument("--name", default=None)
    cv.add_argument("--mode", choices=["pi", "graph"], default="pi",
                    help="pi = 每条边一个 pi 节点；graph = 每条边一张单边 v7 子图（v7 客户端+并行）")
    cv.add_argument("--graphs-dir", default="graphs")
    cv.set_defaults(func=_cmd_convert)

    cp = sub.add_parser("cost", help="按运行记录估算 tokens / 费用")
    cp.add_argument("--db", default=argparse.SUPPRESS, help="sqlite 路径（等价于全局 --db）")
    cp.add_argument("--providers", default=None, help="provider 价格文件（默认 configs/providers.json）")
    cp.add_argument("--limit", type=int, default=20)
    cp.add_argument("--json", action="store_true")
    cp.set_defaults(func=_cmd_cost)

    dmp = sub.add_parser("demo", help="离线 demo：command 节点跑多波次并行（不调模型）")
    dmp.add_argument("--workers", type=int, default=2)
    dmp.add_argument("--keep", action="store_true", help="保留临时配置与 db")
    dmp.set_defaults(func=_cmd_demo)

    abp = sub.add_parser("abort", help="把卡在 running 的 run 标为 interrupted")
    abp.add_argument("run_id")
    abp.set_defaults(func=_cmd_abort)

    prp = sub.add_parser("prune", help="删除旧运行（默认 dry-run）")
    prp.add_argument("--keep", type=int, default=20, help="保留最近 N 个 run")
    prp.add_argument("--apply", action="store_true", help="真正删除（默认只预览）")
    prp.set_defaults(func=_cmd_prune)

    chp = sub.add_parser("cache", help="缓存维护：统计 / 按 TTL 清理 / 清空")
    chp.add_argument("--dir", default=None, help="缓存目录（默认 v8/data/cache）")
    chp.add_argument("--evict-ttl", type=float, default=None, help="删除早于 N 秒的条目")
    chp.add_argument("--clear", action="store_true", help="清空全部缓存")
    chp.add_argument("--json", action="store_true")
    chp.set_defaults(func=_cmd_cache)

    lp = sub.add_parser("lint", help="静态检查配置：不可达输入/环/结果缺失/子图路径")
    lp.add_argument("config")
    lp.add_argument("--strict", action="store_true", help="警告也返回非 0")
    lp.add_argument("--json", action="store_true", help="机器可读输出")
    lp.set_defaults(func=_cmd_lint)

    dp = sub.add_parser("doctor", help="健康检查：pi / provider 注册表 / 网关")
    dp.add_argument("--provider", default="laptop")
    dp.add_argument("--model", default="deepseek-flash")
    dp.add_argument("--timeout", type=float, default=8.0)
    dp.add_argument("--skip-gateway", action="store_true", help="只查本地，不调网关")
    dp.add_argument("--json", action="store_true")
    dp.set_defaults(func=_cmd_doctor)

    np = sub.add_parser("new", help="从模板生成配置（summarize / extract_json / …）")
    np.add_argument("template", nargs="?", default=None)
    np.add_argument("--out", default=None)
    np.add_argument("--list", action="store_true", help="列出模板")
    np.add_argument("--seed", default=None)
    np.add_argument("--seed-tag", default=None)
    np.add_argument("--provider", default="laptop")
    np.add_argument("--model", default="deepseek-v4-flash")
    np.set_defaults(func=_cmd_new)

    gp = sub.add_parser("scaffold", help="从项目结构生成图：每个模块一个 command 节点")
    gp.add_argument("root", help="项目根目录")
    gp.add_argument("--out", required=True, help="生成的 v8 配置路径")
    gp.add_argument("--depth", type=int, default=1, help="模块扫描深度（1 = 顶层）")
    gp.add_argument("--mode", choices=["package", "dir"], default="package",
                    help="package = 含 __init__.py 的目录；dir = 所有顶层目录")
    gp.add_argument("--changed", action="store_true",
                    help="只给 git 变更（含未跟踪）涉及的顶层模块建图")
    gp.add_argument("--review", action="store_true",
                    help="每模块生成 git diff 节点 + 汇总评审，而不是结构报告")
    gp.add_argument("--compose", action="store_true",
                    help="每模块生成 v8 子图，父图用 engine=v8 并行组合")
    gp.add_argument("--graphs-dir", default="graphs",
                    help="--compose 时子图输出目录（相对 --out 的父目录）")
    gp.add_argument("--inspector", default=None, help="模块指纹脚本（默认 scripts/inspect_module.py）")
    gp.add_argument("--provider", default="laptop")
    gp.add_argument("--model", default="deepseek-v4-flash")
    gp.add_argument("--max-modules", type=int, default=12)
    gp.set_defaults(func=_cmd_scaffold)

    args = ap.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
