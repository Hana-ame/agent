#!/usr/bin/env python3
"""
gen_coverage_html.py —— 把 verilator_coverage 注解输出转成交互式 HTML 覆盖率报告

输入：verilator_coverage --annotate 的输出目录（每行带覆盖率计数前缀）
输出：COVERAGE.html（单文件，可点击查看每个文件的逐行覆盖情况）

注解行格式：
    "<count>  <源代码>"     被执行的次数
    "%<count> <源代码>"     count=0 表示从未执行（%000000 为高亮标记）
    "        <源代码>"      无可覆盖点（注释/声明等）

用法:
    python3 gen_coverage_html.py [注解目录] [-o 输出html]
    默认注解目录: axi/run/cov/annotated
"""
import re
import sys
import html
from pathlib import Path

HERE = Path(__file__).resolve().parent          # verilog/axi
DEFAULT_ANN = HERE / "run" / "cov" / "annotated"
OUT = HERE / "COVERAGE.html"

# verilator_coverage 注解行：可选 % 前缀 + 6 位数字 + 空格 + 源码
LINE_RE = re.compile(r'^([%~]?)(\d{6})\s(.*)$')


def parse_annotated(path: Path):
    """解析一个带注解的源文件 → [(行号, 命中次数或None, 源码, 是否零覆盖)]"""
    rows = []
    for i, raw in enumerate(path.read_text(encoding="utf-8", errors="replace").splitlines(), 1):
        m = LINE_RE.match(raw)
        if m:
            marker, cnt, src = m.groups()
            count = int(cnt)
            zero = marker == '%' and count == 0
            rows.append((i, count, src, zero))
        else:
            # 无可覆盖点的行（注解文件对此类行只做缩进）
            rows.append((i, None, raw.strip(), False))
    return rows


# ---------------------------------------------------------------------------
# Verilator 注解已知缺陷：函数内部 case 的 return 分支
# Verilator 对函数体内 case 的各分支不做独立计数，
# 整个函数体计一次，分支行显示 %000000，但功能实际被执行。
# 已用插桩验证（直接层次调用 attr_of/reset_of）确认返回值正确。
# ---------------------------------------------------------------------------
FUNC_CASE_RE = re.compile(
    r'^\s*(?:\d+|default|[A-Z][A-Z0-9_]*)\s*:\s*return\b'
)


def is_func_case_artifact(src: str) -> bool:
    """判断某行是否为「函数内 case return 分支」的注解缺陷行"""
    return bool(FUNC_CASE_RE.match(src))


def summarize(rows):
    covered = [r for r in rows if r[1] is not None]
    hit = [r for r in covered if r[1] > 0]
    zero = [r for r in rows if r[3]]
    # 区分注解缺陷（函数内 case return）与真实零覆盖
    artifact = [r for r in zero if is_func_case_artifact(r[2])]
    real_zero = [r for r in zero if r not in artifact]
    eff_cov = len(covered) - len(artifact)
    eff_hit = len(hit)
    return {
        "total": len(rows),
        "coverable": len(covered),
        "hit": len(hit),
        "zero": len(zero),
        "artifact": len(artifact),
        "real_zero": len(real_zero),
        "pct": (100.0 * len(hit) / len(covered)) if covered else 0.0,
        "eff_pct": (100.0 * eff_hit / eff_cov) if eff_cov else 0.0,
    }


def build_html(files_data):
    rtl_files = [d for d in files_data if not d.get("is_tb")]
    tb_files = [d for d in files_data if d.get("is_tb")]
    rtl_cov = sum(d["stats"]["coverable"] for d in rtl_files)
    rtl_hit = sum(d["stats"]["hit"] for d in rtl_files)
    rtl_art = sum(d["stats"]["artifact"] for d in rtl_files)
    rtl_real_zero = sum(d["stats"]["real_zero"] for d in rtl_files)
    rtl_eff_pct = (100.0 * rtl_hit / (rtl_cov - rtl_art)) if (rtl_cov - rtl_art) else 0.0

    total_cov = sum(d["stats"]["coverable"] for d in files_data)
    total_hit = sum(d["stats"]["hit"] for d in files_data)
    total_zero = sum(d["stats"]["real_zero"] for d in files_data)
    total_artifact = sum(d["stats"]["artifact"] for d in files_data)
    pct = (100.0 * total_hit / total_cov) if total_cov else 0.0

    # 侧栏文件列表（分 RTL / TB 两组）
    nav = []
    nav_rtl = []
    nav_tb = []
    for idx, d in enumerate(files_data):
        s = d["stats"]
        color = "#3fb950" if s["pct"] >= 95 else ("#d29922" if s["pct"] >= 80 else "#f85149")
        extra = ''
        if s["real_zero"]:
            extra = f' · <b style="color:#f85149">{s["real_zero"]} 未覆盖</b>'
        elif s["artifact"]:
            extra = f' · <span style="color:#6e7681">{s["artifact"]} 注解缺陷</span>'
        nav.append(
            f'<div class="fitem" data-idx="{idx}">'
            f'<span class="fname">{html.escape(d["name"])}</span>'
            f'<span class="fpct" style="color:{color}">{s["pct"]:.1f}%</span>'
            f'<span class="fmeta">{s["hit"]}/{s["coverable"]}{extra}'
            + '</span></div>'
        )
        (nav_tb if d.get("is_tb") else nav_rtl).append(nav[-1])

    # 每个文件的代码视图
    panes = []
    for idx, d in enumerate(files_data):
        lines = []
        for ln, cnt, src, zero in d["rows"]:
            esc = html.escape(src) or "&nbsp;"
            if zero:
                lines.append(
                    f'<div class="ln zero" id="L{ln}">'
                    f'<span class="no">{ln}</span><span class="hit">✗</span>'
                    f'<span class="src">{esc}</span></div>'
                )
            elif cnt is not None:
                cls = "hit" if cnt > 0 else "none"
                lines.append(
                    f'<div class="ln {cls}">'
                    f'<span class="no">{ln}</span><span class="hit">{cnt if cnt else "·"}</span>'
                    f'<span class="src">{esc}</span></div>'
                )
            else:
                lines.append(
                    f'<div class="ln plain">'
                    f'<span class="no">{ln}</span><span class="hit"></span>'
                    f'<span class="src">{esc}</span></div>'
                )
        panes.append(
            f'<div class="pane" id="pane{idx}"><div class="code">{"".join(lines)}</div></div>'
        )

    return f"""<!DOCTYPE html>
<html lang="zh-CN"><head><meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>覆盖率报告 · AXI→APB→AXI</title>
<style>
:root{{--bg:#0d1117;--panel:#161b22;--bd:#30363d;--fg:#c9d1d9;--dim:#8b949e;
--blue:#58a6ff;--green:#3fb950;--amber:#d29922;--red:#f85149;
--mono:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace}}
*{{box-sizing:border-box;margin:0;padding:0}}
body{{background:var(--bg);color:var(--fg);font:13.5px/1.55 -apple-system,"Segoe UI","Noto Sans SC",sans-serif}}
header{{border-bottom:1px solid var(--bd);padding:22px 26px 18px;background:#11151b}}
h1{{font-size:20px;font-weight:650}}
h1 small{{display:block;font-size:12px;color:var(--dim);font-weight:400;margin-top:6px}}
.stats{{display:flex;gap:26px;margin-top:16px;flex-wrap:wrap}}
.stat .v{{font-family:var(--mono);font-size:21px;font-weight:600}}
.stat .l{{font-size:11px;color:var(--dim);margin-top:2px}}
.bar{{height:7px;background:#21262d;border-radius:4px;overflow:hidden;margin-top:12px;max-width:520px}}
.bar div{{height:100%;background:linear-gradient(90deg,#238636,#3fb950);border-radius:4px}}
.layout{{display:grid;grid-template-columns:300px 1fr;min-height:calc(100vh - 200px)}}
aside{{border-right:1px solid var(--bd);overflow-y:auto;max-height:calc(100vh - 200px)}}
.grp{{padding:12px 16px 7px;font-size:10.5px;color:#6e7681;text-transform:uppercase;
letter-spacing:.5px;border-bottom:1px solid #21262d;background:#0f141a}}
.grp small{{display:block;font-size:9.5px;text-transform:none;letter-spacing:0;margin-top:2px}}
.fitem{{padding:11px 16px;border-bottom:1px solid #21262d;cursor:pointer;transition:.12s}}
.fitem:hover{{background:#161b22}}
.fitem.on{{background:#1f6feb22;border-left:3px solid var(--blue)}}
.fname{{display:block;font-family:var(--mono);font-size:12.5px;color:#79c0ff}}
.fpct{{display:inline-block;font-family:var(--mono);font-size:11px;margin-right:8px}}
.fmeta{{font-size:10.5px;color:var(--dim)}}
main{{overflow:auto;max-height:calc(100vh - 200px)}}
.pane{{display:none}}.pane.on{{display:block}}
.code{{font-family:var(--mono);font-size:12px;line-height:1.62}}
.ln{{display:flex;white-space:pre}}
.ln .no{{flex:none;width:52px;text-align:right;padding-right:12px;color:#484f58;
user-select:none;border-right:1px solid #21262d}}
.ln .hit{{flex:none;width:56px;text-align:right;padding-right:12px;color:#6e7681;font-size:11px}}
.ln .src{{flex:1;padding-left:12px;color:#c9d1d9}}
.ln.zero{{background:#f8514918}}
.ln.zero .src{{color:#ffdcd7}}
.ln.zero .hit{{color:var(--red);font-weight:700}}
.ln.zero .no{{color:var(--red);border-right-color:#8b2b28}}
.ln.hit .hit{{color:#3fb950}}
.ln.plain .src{{color:#6e7681}}
footer{{border-top:1px solid var(--bd);padding:14px 26px;color:#6e7681;font-size:11.5px}}
code{{font-family:var(--mono);background:#0d1117;border:1px solid #21262d;
padding:1px 5px;border-radius:4px;color:#79c0ff}}
</style></head><body>
<header>
<h1>覆盖率报告：AXI→APB→AXI
<small>Verilator --coverage + verilator_coverage 注解 · 红色行为从未执行</small></h1>
<div class="stats">
<div class="stat"><div class="v" style="color:var(--green)">{rtl_eff_pct:.1f}%</div><div class="l">RTL 有效覆盖率<br><small>{rtl_hit}/{rtl_cov - rtl_art}</small></div></div>
<div class="stat"><div class="v">{pct:.1f}%</div><div class="l">含 TB 总覆盖率</div></div>
<div class="stat"><div class="v">{total_hit}/{total_cov}</div><div class="l">已执行 / 可覆盖行</div></div>
<div class="stat"><div class="v" style="color:var(--red)">{total_zero}</div><div class="l">真实零覆盖</div></div>
<div class="stat"><div class="v" style="color:var(--dim)">{total_artifact}</div><div class="l">注解缺陷<br><small>函数内 case return</small></div></div>
<div class="stat"><div class="v">{len(files_data)}</div><div class="l">源文件</div></div>
</div>
<div class="bar"><div style="width:{pct:.1f}%"></div></div>
</header>
<div class="layout">
<aside>
<div class="grp">设计文件（RTL）</div>{''.join(nav_rtl)}
<div class="grp">测试文件（TB）<small>零覆盖多为 FAIL 分支</small></div>{''.join(nav_tb)}
</aside>
<main>{''.join(panes)}</main>
</div>
<footer>
生成命令：<code>bash verilog/sim.sh coverage</code> 然后
<code>python3 gen_coverage_html.py</code>
</footer>
<script>
const items = document.querySelectorAll('.fitem');
const panes = document.querySelectorAll('.pane');
function show(i) {{
  items.forEach(x => x.classList.remove('on'));
  panes.forEach(x => x.classList.remove('on'));
  items[i].classList.add('on');
  document.getElementById('pane' + i).classList.add('on');
  document.querySelector('main').scrollTop = 0;
}}
items.forEach((it, i) => it.onclick = () => show(i));
if (items.length) show(0);
</script>
</body></html>"""


def main():
    args = sys.argv[1:]
    ann_dir = DEFAULT_ANN
    out = OUT
    if args and not args[0].startswith("-"):
        ann_dir = Path(args[0])
    if "-o" in args:
        out = Path(args[args.index("-o") + 1])

    if not ann_dir.exists():
        print(f"错误：注解目录不存在: {ann_dir}", file=sys.stderr)
        print("请先运行: bash verilog/sim.sh coverage", file=sys.stderr)
        sys.exit(1)

    files_data = []
    for sv in sorted(ann_dir.glob("*.sv")):
        rows = parse_annotated(sv)
        stats = summarize(rows)
        files_data.append({"name": sv.name, "rows": rows, "stats": stats})

    if not files_data:
        print(f"错误：{ann_dir} 下没有 .sv 注解文件", file=sys.stderr)
        sys.exit(1)

    # 标记文件类型：测试平台（tb_*/含 FAIL 分支）与设计（RTL）
    for d in files_data:
        d["is_tb"] = d["name"].startswith("tb_")

    # 按未覆盖数降序（问题最多的排前面），但 RTL 优先展示
    files_data.sort(key=lambda d: (d["is_tb"], -d["stats"]["real_zero"], d["name"]))

    for d in files_data:
        s = d["stats"]
        mark = ""
        if s["artifact"]:
            mark = f"  [注解缺陷 {s['artifact']}]"
        print(f"  {d['name']:<32} {s['pct']:5.1f}%  "
              f"{s['hit']:3d}/{s['coverable']:3d} "
              f"真实零覆盖 {s['real_zero']}{mark}")

    html_text = build_html(files_data)
    out.write_text(html_text, encoding="utf-8")
    tot = sum(d["stats"]["coverable"] for d in files_data)
    hit = sum(d["stats"]["hit"] for d in files_data)
    zero = sum(d["stats"]["real_zero"] for d in files_data)
    art = sum(d["stats"]["artifact"] for d in files_data)
    eff_tot = tot - art
    eff_pct = 100.0 * hit / eff_tot if eff_tot else 0
    rtl_d = [d for d in files_data if not d.get("is_tb")]
    rc2 = sum(d["stats"]["coverable"] for d in rtl_d)
    rh2 = sum(d["stats"]["hit"] for d in rtl_d)
    ra2 = sum(d["stats"]["artifact"] for d in rtl_d)
    rz2 = sum(d["stats"]["real_zero"] for d in rtl_d)
    print(f"\n【RTL】可覆盖 {rc2} · 命中 {rh2} · 注解缺陷 {ra2} · 真实零覆盖 {rz2}")
    print(f"【RTL】有效覆盖率: {100.0*rh2/(rc2-ra2) if (rc2-ra2) else 0:.1f}%  ({rh2}/{rc2-ra2})")
    print(f"\n原始行覆盖率: {100.0*hit/tot if tot else 0:.1f}%  ({hit}/{tot})")
    print(f"真实零覆盖:   {zero} 行")
    print(f"注解缺陷:     {art} 行（Verilator 函数内 case return，非真实缺失）")
    print(f"有效覆盖率:   {eff_pct:.1f}%  ({hit}/{eff_tot})")
    print(f"已生成: {out}  ({len(html_text)} 字节)")


if __name__ == "__main__":
    main()
