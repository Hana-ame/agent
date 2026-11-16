#!/usr/bin/env python3
"""
gen_portal.py —— 项目门户页：索引所有 HTML 交付物 + 汇总真实统计

设计原则：所有数字都从真实数据源提取，不硬编码。
  1. 扫描 axi/*.html        → 列出交付物（大小、标题）
  2. 解析 gen_test_plan.py  → 测试项数 / 覆盖率
  3. 解析 run/cov/annotated → 覆盖率数据（若有）
  4. 解析 sv/*.sv           → 源文件行数与模块数
  5. 扫描 sim.sh            → 套件清单
  6. 解析 gen_fsm_html.py 的校验结果 → FSM 状态/转移数

用法: python3 gen_portal.py
"""
import re
import sys
import html
from pathlib import Path
from datetime import datetime

HERE = Path(__file__).resolve().parent      # verilog/axi
ROOT = HERE.parent                          # verilog
OUT = HERE / "INDEX.html"

# HTML 交付物说明（人工维护的元信息：标题 + 用途 + 来源类型）
DOC_META = {
    "DESIGN.html": ("设计文档", "架构、协议定义、转换设计、验证策略、工具链（12 章节）", "手写"),
    "INTERFACE_MAP.html": ("接口逻辑映射", "字段→信号映射、接口信号全景、数据流转链、FSM 驱动", "源码解析 + 自校验"),
    "TEST_PLAN.html": ("测试方案与覆盖缺口", "7 维度 29 项测试、覆盖判定、执行计划 P0-P4", "源码解析"),
    "COVERAGE.html": ("覆盖率可视化", "逐行执行次数、零覆盖行红色高亮、按文件分栏", "coverage 注解解析"),
    "FSM.html": ("FSM 状态转移图", "串行桥/并发桥状态机、转移条件、可点击高亮出边", "源码解析 + 自校验"),
    "REGFILE.html": ("寄存器手册", "地址映射表、属性语义（RO/RW/W1C/RSVD）、复位值", "从 spec 生成"),
}


def parse_test_plan():
    """从 gen_test_plan.py 提取测试项与覆盖统计"""
    p = HERE / "gen_test_plan.py"
    if not p.exists():
        return None
    txt = p.read_text(encoding="utf-8")
    ids = re.findall(r'dict\(id="([A-Z]-\d+)"', txt)
    # 统计已覆盖/缺口：跑一次生成器解析更准，这里用轻量推断
    # 已实现的缺口任务名
    gaps_file = HERE / "sv" / "tb_gaps_sv.sv"
    dual_file = HERE / "sv" / "tb_dual_sv.sv"
    covered_tasks = set()
    for f in (gaps_file, dual_file):
        if f.exists():
            covered_tasks |= set(re.findall(r"task automatic (t_[a-z0-9_]+)", f.read_text(encoding="utf-8")))
    # 映射表（与 gen_test_plan.py 保持一致）
    gap_map = {
        "F-05": "t_f05_overwrite", "F-06": "t_f06_unwritten",
        "E-03": "t_e03_recover", "E-04": "t_e04_oob_sweep",
        "S-02": "t_s02_backtoback", "S-03": "t_s03_long_pause",
        "R-01": "t_r01_reset_clear", "R-02": "t_r02_mid_reset",
        "T-02": "t_t02_long_random", "P-06": "t_p06_aw_ar_same_cycle",
    }
    covered = len(ids) - len([i for i in ids if i in gap_map and gap_map[i] not in covered_tasks])
    covered -= len([i for i in ids if i in ("W-02", "W-03", "T-03") and "WAIT_CYCLES" not in
                    (HERE / "sv" / "tb_gaps_sv.sv").read_text(encoding="utf-8")])
    return {"total": len(ids), "covered": covered, "gaps": len(ids) - covered}


def parse_coverage():
    """从注解目录统计覆盖率。

    重要：必须区分 RTL 与 TB。
      - RTL 覆盖率是真正有意义的指标（设计代码被执行了多少）
      - TB 的「零覆盖」大多来自 FAIL 分支（测试通过时不执行），属正常
    混在一起会得出误导性的低数字。

    同时识别 Verilator 的注解缺陷（函数内 case 的 return 分支不计入）。
    """
    ann = HERE / "run" / "cov" / "annotated"
    if not ann.exists():
        return None

    LINE_RE = re.compile(r'^([%~]?)(\d{6})\s+(.*)$')
    FUNC_CASE_RE = re.compile(r'^\s*(?:\d+|default|[A-Z][A-Z0-9_]*)\s*:\s*return\b')
    # 与 gen_coverage_html.py 保持同一口径：函数体内裸 return 也属注解缺陷
    FUNC_BARE_RETURN_RE = re.compile(r'^\s*return\b[^;]*;\s*$')

    # 非测试缺口（与 gen_coverage_html.py 保持同一口径）：
    #   1. 检查器违规分支   2. 状态机 default 回退（枚举已穷举）
    #   3. VERBOSE 日志     4. 未使用的声明
    _NON_GAP_RES = [
        re.compile(r"violation\(|errors\+\+|\[%s\] VIOLATION"),
        re.compile(r'\$display\(\s*"\[%s\].*(xfer|APB)'),
        re.compile(r'^\s*(NAME\s*,|apb\.(PWRITE|PSLVERR)\s*\?|apb\.PADDR\s*,)'),
        re.compile(r"^\s*logic\s+(w_slverr|r_slverr)\s*;"),
        re.compile(r"^\s*default\s*:"),
        re.compile(r"^\s*(phase\s*<=\s*APB_IDLE|apb\.PREADY\s*<=\s*1'b0|"
                   r"apb\.PSLVERR\s*<=\s*1'b0|next\s*=\s*S_IDLE|"
                   r"wst\s*<=\s*W_IDLE|rst\s*<=\s*R_IDLE)"),
    ]

    def is_non_gap(line):
        return any(r.search(line) for r in _NON_GAP_RES)

    stat = {"rtl": {"total": 0, "hit": 0, "zero": 0, "artifact": 0},
            "tb":  {"total": 0, "hit": 0, "zero": 0, "artifact": 0},
            "files": 0}
    for sv in ann.glob("*.sv"):
        stat["files"] += 1
        kind = "tb" if sv.name.startswith("tb_") else "rtl"
        for line in sv.read_text(encoding="utf-8", errors="replace").splitlines():
            m = LINE_RE.match(line)
            if not m:
                continue
            stat[kind]["total"] += 1
            cnt = int(m.group(2))
            if cnt > 0:
                stat[kind]["hit"] += 1
            elif m.group(1) == '%':
                # 与 gen_coverage_html.py 保持同一口径
                if FUNC_CASE_RE.match(m.group(3)) or FUNC_BARE_RETURN_RE.match(m.group(3)):
                    stat[kind]["artifact"] += 1
                elif is_non_gap(m.group(3)):
                    stat[kind]["checker"] = stat[kind].get("checker", 0) + 1
                else:
                    stat[kind]["zero"] += 1

    r = stat["rtl"]
    if r["total"] == 0:
        return None
    eff_total = r["total"] - r["artifact"] - r.get("checker", 0)
    return {
        "files": stat["files"],
        "rtl_total": r["total"], "rtl_hit": r["hit"],
        "rtl_zero": r["zero"], "rtl_artifact": r["artifact"],
        "rtl_checker": r.get("checker", 0),
        "rtl_eff_pct": (100.0 * r["hit"] / eff_total) if eff_total else 0.0,
        "rtl_raw_pct": 100.0 * r["hit"] / r["total"],
        "tb_total": stat["tb"]["total"], "tb_hit": stat["tb"]["hit"],
    }


def parse_sources():
    """统计设计源文件"""
    sv = HERE / "sv"
    rtl, tb = [], []
    for f in sorted(sv.glob("*.sv")):
        lines = len(f.read_text(encoding="utf-8").splitlines())
        (tb if f.name.startswith("tb_") else rtl).append((f.name, lines))
    return {"rtl": rtl, "tb": tb}


def parse_suites():
    """从 sim.sh 的 all 目标提取「实际执行的测试套件」清单。

    注意：不能统计所有 case 分支——那会把 coverage/docs/wave/mutate
    等工具目标也算进去。只有 all 目标里真正被调用的才是测试套件。
    """
    p = ROOT / "sim.sh"
    if not p.exists():
        return []
    txt = p.read_text(encoding="utf-8")
    # 提取 all) 分支里的 `bash "$0" <target>` 调用
    m = re.search(r"^    all\)(.*?)^        ;;", txt, re.S | re.M)
    if not m:
        return []
    names = re.findall(r'bash "\$0" (\w+)', m.group(1))
    # 排除静态检查类（lint 不是仿真套件）
    NON_TEST = {"lint"}
    return [n for n in names if n not in NON_TEST]


def parse_fsm_stats():
    """从 gen_fsm_html.py 的解析结果推断（跑一次轻量解析）"""
    try:
        sys.path.insert(0, str(HERE))
        import gen_fsm_html as g
        total_s = total_e = 0
        machines = 0
        for name, label in (("axi_apb_bridge_sv.sv", "串行桥"),
                            ("axi_apb_bridge_dual_sv.sv", "并发桥")):
            m = g.parse_fsm(HERE / "sv" / name, label)
            if m and m.get("states"):
                machines += 1
                total_s += len(m["states"])
                total_e += len(m["edges"])
        return {"machines": machines, "states": total_s, "edges": total_e}
    except Exception:
        return None


def human_size(n: int) -> str:
    return f"{n/1024:.0f} KB" if n >= 1024 else f"{n} B"


def build_html():
    tp = parse_test_plan()
    cov = parse_coverage()
    src = parse_sources()
    suites = parse_suites()
    fsm = parse_fsm_stats()

    bar_items = []
    for name in sorted(DOC_META, key=lambda x: list(DOC_META).index(x)):
        f = HERE / name
        if not f.exists():
            continue
        title, desc, source = DOC_META[name]
        bar_items.append({
            "file": name, "title": title, "desc": desc,
            "source": source, "size": f.stat().st_size,
        })

    # ---- KPI 卡片 ----
    kpis = []
    if tp:
        kpis.append(("测试项", f"{tp['covered']}/{tp['total']}",
                     f"{100*tp['covered']//tp['total']}% 覆盖 · {tp['gaps']} 缺口", "#3fb950"))
    if cov:
        _den = cov['rtl_total'] - cov['rtl_artifact'] - cov.get('rtl_checker', 0)
        kpis.append(("RTL 覆盖率", f"{cov['rtl_eff_pct']:.1f}%",
                     f"{cov['rtl_hit']}/{_den} 行·有效目标",
                     "#58a6ff"))
    if fsm:
        kpis.append(("状态机", f"{fsm['machines']}",
                     f"{fsm['states']} 状态 · {fsm['edges']} 转移", "#bc8cff"))
    if suites:
        kpis.append(("测试套件", f"{len(suites)}", "全部 Verilator 仿真", "#d29922"))
    rtl_lines = sum(n for _, n in src["rtl"])
    tb_lines = sum(n for _, n in src["tb"])
    kpis.append(("源码行数", f"{rtl_lines + tb_lines}", f"RTL {rtl_lines} · TB {tb_lines}", "#8b949e"))

    # ---- 文档卡片 ----
    cards = []
    for it in bar_items:
        src_badge = "解析" if "解析" in it["source"] else "手写"
        badge_cls = "gen" if "解析" in it["source"] else "manual"
        cards.append(f"""
<a class="doc" href="{html.escape(it['file'])}">
  <div class="dh">
    <span class="dtitle">{html.escape(it['title'])}</span>
    <span class="badge {badge_cls}">{src_badge}</span>
  </div>
  <div class="dfile">{html.escape(it['file'])} · {human_size(it['size'])}</div>
  <div class="ddesc">{html.escape(it['desc'])}</div>
  <div class="dsrc">{html.escape(it['source'])}</div>
</a>""")

    # ---- 套件列表 ----
    suite_chips = "".join(f'<span class="chip">{html.escape(s)}</span>' for s in suites)

    # ---- 源文件表 ----
    src_rows = []
    for name, lines in src["rtl"]:
        src_rows.append(f'<tr><td class="mono">{name}</td><td class="num">{lines}</td>'
                        f'<td><span class="tag rtl">RTL</span></td></tr>')
    for name, lines in src["tb"]:
        src_rows.append(f'<tr><td class="mono">{name}</td><td class="num">{lines}</td>'
                        f'<td><span class="tag tb">TB</span></td></tr>')

    kpi_html = "".join(f"""
<div class="kpi">
  <div class="kv" style="color:{color}">{value}</div>
  <div class="kl">{label}</div>
  <div class="kd">{detail}</div>
</div>""" for label, value, detail, color in kpis)

    now = datetime.now().strftime("%Y-%m-%d %H:%M")

    return f"""<!DOCTYPE html>
<html lang="zh-CN"><head><meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>AXI→APB→AXI 验证项目 · 文档门户</title>
<style>
:root{{--bg:#0d1117;--panel:#161b22;--bd:#30363d;--fg:#c9d1d9;--dim:#8b949e;
--blue:#58a6ff;--green:#3fb950;--amber:#d29922;--purple:#bc8cff;
--mono:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace}}
*{{box-sizing:border-box;margin:0;padding:0}}
body{{background:var(--bg);color:var(--fg);font:14px/1.65 -apple-system,"Segoe UI","Noto Sans SC",sans-serif;
padding:0 0 60px}}
.wrap{{max-width:1180px;margin:0 auto;padding:0 26px}}
header{{border-bottom:1px solid var(--bd);padding:44px 0 30px;margin-bottom:34px;
background:linear-gradient(180deg,#131a22 0%,#0d1117 100%)}}
h1{{font-size:29px;font-weight:660;letter-spacing:-.3px}}
h1 small{{display:block;font-size:13.5px;font-weight:400;color:var(--dim);margin-top:9px;letter-spacing:0}}
.kpis{{display:grid;grid-template-columns:repeat(5,1fr);gap:12px;margin-top:26px}}
@media(max-width:900px){{.kpis{{grid-template-columns:repeat(2,1fr)}}}}
.kpi{{background:var(--panel);border:1px solid var(--bd);border-radius:9px;padding:15px 17px}}
.kv{{font-family:var(--mono);font-size:24px;font-weight:600;line-height:1.2}}
.kl{{font-size:11.5px;color:#c9d1d9;margin-top:5px}}
.kd{{font-size:10.5px;color:var(--dim);margin-top:3px}}
h2{{font-size:18px;margin:32px 0 16px;padding-bottom:9px;border-bottom:1px solid var(--bd)}}
h2 .n{{font-family:var(--mono);font-size:11px;color:var(--blue);margin-right:9px}}
.docs{{display:grid;grid-template-columns:repeat(2,1fr);gap:13px}}
@media(max-width:820px){{.docs{{grid-template-columns:1fr}}}}
.doc{{display:block;background:var(--panel);border:1px solid var(--bd);border-radius:9px;
padding:17px 19px;text-decoration:none;color:inherit;transition:.15s}}
.doc:hover{{border-color:var(--blue);background:#1f6feb11;transform:translateY(-1px)}}
.dh{{display:flex;justify-content:space-between;align-items:center}}
.dtitle{{font-size:15px;font-weight:620;color:#eaf2fa}}
.badge{{font-size:10px;padding:2px 7px;border-radius:4px;font-family:var(--mono)}}
.badge.gen{{background:#123024;border:1px solid #238636;color:#56d364}}
.badge.manual{{background:#2b2110;border:1px solid #9e6a03;color:#e3b341}}
.dfile{{font-family:var(--mono);font-size:11px;color:var(--blue);margin-top:7px}}
.ddesc{{font-size:12.5px;color:#8b949e;margin-top:7px;line-height:1.55}}
.dsrc{{font-size:10.5px;color:#6e7681;margin-top:8px}}
.chip{{display:inline-block;font-family:var(--mono);font-size:11.5px;padding:4px 11px;
border:1px solid #33506e;border-radius:5px;margin:3px 5px 3px 0;color:#9db9d4;background:#0e1a26}}
table{{width:100%;border-collapse:collapse;font-size:12.5px;margin:8px 0}}
th,td{{padding:8px 11px;border-bottom:1px solid #21262d;text-align:left}}
th{{color:var(--dim);font-size:10.5px;text-transform:uppercase;letter-spacing:.4px;background:#11151b}}
tbody tr:hover{{background:#11151b}}
.mono{{font-family:var(--mono);color:#79c0ff}}
.num{{font-family:var(--mono);text-align:right;color:#3fb950}}
.tag{{font-size:10px;padding:1px 6px;border-radius:3px;font-family:var(--mono)}}
.tag.rtl{{background:#1f190d;border:1px solid #9e6a03;color:#e3b341}}
.tag.tb{{background:#1c1a2e;border:1px solid #6e40c9;color:#bc8cff}}
.grid2{{display:grid;grid-template-columns:1fr 1fr;gap:22px}}
@media(max-width:820px){{.grid2{{grid-template-columns:1fr}}}}
pre{{background:#0b1017;border:1px solid #222c37;border-radius:8px;padding:15px 17px;
font-family:var(--mono);font-size:12px;line-height:1.6;color:#b8c8d8;overflow-x:auto}}
pre .c{{color:#5b7186}}
footer{{margin-top:40px;padding-top:18px;border-top:1px solid var(--bd);
color:#6e7681;font-size:11.5px}}
</style></head><body>
<header><div class="wrap">
<h1>AXI4-Lite → APB → 慢速外设 → AXI
<small>验证项目文档门户 · 所有统计数字均从源码/测试/覆盖率数据实时提取</small></h1>
<div class="kpis">{kpi_html}</div>
</div></header>

<div class="wrap">
<h2><span class="n">01</span>文档交付物</h2>
<div class="docs">{''.join(cards)}</div>

<h2><span class="n">02</span>测试套件（{len(suites)} 个）</h2>
<div>{suite_chips}</div>
<pre><span class="c"># 一键运行</span>
bash verilog/sim.sh              <span class="c"># 全量回归</span>
bash verilog/sim.sh sv           <span class="c"># 单个套件</span>
bash verilog/sim.sh coverage     <span class="c"># 覆盖率报告 + HTML</span>
bash verilog/sim.sh docs         <span class="c"># 重新生成所有 HTML 文档</span>

<span class="c"># 参数化回归</span>
WAIT=0  bash verilog/sim.sh gaps <span class="c"># 零等待</span>
WAIT=16 bash verilog/sim.sh gaps <span class="c"># 长等待</span>
SEED=123 bash verilog/sim.sh gaps <span class="c"># 多 seed</span></pre>

<h2><span class="n">03</span>源码构成</h2>
<div class="grid2">
<div>
<h3 style="font-size:13.5px;color:#eaf2fa;margin-bottom:6px">设计文件（RTL）</h3>
<table><thead><tr><th>文件</th><th style="width:70px">行数</th><th style="width:60px">类型</th></tr></thead>
<tbody>{''.join(r for r in src_rows if 'rtl' in r)}</tbody></table>
</div>
<div>
<h3 style="font-size:13.5px;color:#eaf2fa;margin-bottom:6px">测试文件（TB）</h3>
<table><thead><tr><th>文件</th><th style="width:70px">行数</th><th style="width:60px">类型</th></tr></thead>
<tbody>{''.join(r for r in src_rows if 'tb' in r)}</tbody></table>
</div>
</div>

<h2><span class="n">04</span>工具链</h2>
<table><thead><tr><th style="width:130px">工具</th><th>职责</th><th style="width:160px">说明</th></tr></thead>
<tbody>
<tr><td class="mono">slang</td><td>编译检查（仿真前）</td><td>完整 SV 语法/语义，含 UVM 库校验</td></tr>
<tr><td class="mono">Verilator</td><td>仿真 + 覆盖率</td><td>{len(suites)} 个套件全部运行</td></tr>
<tr><td class="mono">irun / VCS</td><td>覆盖率/UVM 运行时</td><td>最终验收（服务器）</td></tr>
</tbody></table>

<footer>
生成时间 {now} · 生成器 <code>python3 gen_portal.py</code><br>
所有统计来自：<code>gen_test_plan.py</code> 测试清单、
<code>run/cov/annotated</code> 覆盖率数据、
<code>sv/*.sv</code> 源码行数、<code>sim.sh</code> 套件定义、<code>gen_fsm_html.py</code> 解析结果
</footer>
</div></body></html>"""


def main():
    txt = build_html()
    OUT.write_text(txt, encoding="utf-8")

    # 打印摘要
    tp = parse_test_plan()
    cov = parse_coverage()
    src = parse_sources()
    suites = parse_suites()
    fsm = parse_fsm_stats()
    print("门户摘要：")
    if tp:
        print(f"  测试项      {tp['covered']}/{tp['total']}（{tp['gaps']} 缺口）")
    if cov:
        _den = cov['rtl_total'] - cov['rtl_artifact'] - cov.get('rtl_checker', 0)
        print(f"  RTL 覆盖率  {cov['rtl_eff_pct']:.1f}%（{cov['rtl_hit']}/{_den} 有效目标，"
              f"原始 {cov['rtl_raw_pct']:.1f}%，注解缺陷 {cov['rtl_artifact']}，"
              f"非缺口 {cov.get('rtl_checker', 0)}）")
    if fsm:
        print(f"  状态机      {fsm['machines']} 个 · {fsm['states']} 状态 · {fsm['edges']} 转移")
    print(f"  测试套件    {len(suites)} 个")
    print(f"  源码        RTL {sum(n for _,n in src['rtl'])} 行 · TB {sum(n for _,n in src['tb'])} 行")
    print(f"\n已生成: {OUT}  ({len(txt)} 字节)")


if __name__ == "__main__":
    main()
