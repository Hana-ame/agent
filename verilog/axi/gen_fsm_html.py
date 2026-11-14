#!/usr/bin/env python3
"""
gen_fsm_html.py —— 从 SV 源码解析 FSM，生成交互式状态转移图 HTML

设计原则（同 gen_interface_map.py）：解析真实源码而非手绘，
状态名与转移条件不一致时立刻报错，避免文档与代码脱节。

解析内容：
  1. `typedef enum ... { S_XXX, ... } state_t;`  → 状态集合
  2. `case (state) ... endcase` 块内的转移          → 边与条件
  3. 每个状态在输出 case 中驱动的信号（可选）        → 状态行为

支持同时解析多个桥（单路径串行桥 + 并发双路径桥）。

用法: python3 gen_fsm_html.py
"""
import re
import sys
import html
from pathlib import Path

HERE = Path(__file__).resolve().parent      # verilog/axi
SV = HERE / "sv"
OUT = HERE / "FSM.html"

# 状态命名前缀（支持多种命名约定）
#   S_XXX  单状态机（串行桥）
#   W_XXX / R_XXX / A_XXX  并发桥的写/读/仲裁三个枚举
STATE_PREFIXES = ("S_", "W_", "R_", "A_")


def _is_state(tok: str) -> bool:
    return any(tok.startswith(pre) and len(tok) > len(pre)
               for pre in STATE_PREFIXES)


def parse_fsm(path: Path, machine_name: str):
    """解析一个 FSM：返回 {states: [...], edges: [(from,to,cond), ...]}"""
    txt = path.read_text(encoding="utf-8")

    # ---- 1. 状态枚举（支持多个 enum，任意 *_t 类型名）----
    states = []
    for em in re.finditer(r"typedef\s+enum\s+logic\s*\[[^\]]+\]\s*\{(.*?)\}\s*(\w+)\s*;", txt, re.S):
        body, tname = em.group(1), em.group(2)
        cand = [s.strip() for s in re.split(r"[,\s]+", body) if _is_state(s.strip())]
        if cand:
            states.extend(cand)
    if not states:
        return None

    # ---- 2. 找转移逻辑 ----
    # 串行桥：always_comb 里 next = S_XXX
    # 并发桥：always_ff 里 wst <= W_XXX / rst <= R_XXX
    trans_block = None
    blocks = re.findall(r"always_comb\s+begin(.*?)\n    end\b", txt, re.S)
    for b in blocks:
        if "next = S_" in b:
            trans_block = b
            break
    if trans_block is None:
        # 并发桥：取 always_ff 块（贪婪匹配到最后一个 "end"），<= 归一为 =
        ffm = re.search(r"always_ff[^\n]*begin(.*)\n    end\b", txt, re.S)
        if ffm:
            cand = ffm.group(1).replace("<=", "=")
            # 状态名在赋值右侧：wst = W_IDLE;
            if any(re.search(rf"=\s*{pre}\w+", cand) for pre in STATE_PREFIXES):
                trans_block = cand
    if trans_block is None:
        return {"states": states, "edges": [], "error": "未找到转移逻辑"}

    edges = []
    # 逐个状态匹配 case 分支
    for i, st in enumerate(states):
        # 从 "S_XXX:" 到下一个 "S_YYY:" 或 "default:" 或 endcase
        pat = re.compile(
            rf"\b{re.escape(st)}\s*:(.*?)(?=\n\s+(?:{'|'.join(re.escape(s) for s in states)}|default)\s*:|\n\s*endcase)",
            re.S)
        mm = pat.search(trans_block)
        if not mm:
            continue
        body = mm.group(1)

        # 找出该分支内所有 next = S_YYY，并尽量附带条件
        # 形式：if (cond) next = X;  else if (cond2) next = Y;  ...
        # 串行桥用 next =，并发桥用 wst = / rst =（已归一 <= 为 =）
        parts = re.split(r"\b(?:next|wst|rst)\s*=\s*", body)
        # parts[0] 是第一个赋值前的条件上下文
        cond_ctx = parts[0]
        for idx in range(1, len(parts)):
            rest = parts[idx]
            tm = re.match(r"([A-Z]_\w+)", rest.strip())
            if not tm:
                continue
            to = tm.group(1)
            # 条件：取该赋值前最近的 if/else if 的条件，或 else
            cond = extract_condition(cond_ctx, idx, body)
            edges.append((st, to, cond))

    return {"states": states, "edges": edges, "name": machine_name}


def extract_condition(pre_ctx: str, assign_idx: int, body: str):
    """从赋值前的上下文中提取 if/else 条件（尽力而为）"""
    text = pre_ctx.strip()
    # 去掉末尾多余
    if not text:
        return "无条件"
    # 找最后一个 if (...) / else if (...)
    conds = re.findall(r"(?:else\s+)?if\s*\(([^)]*(?:\([^)]*\)[^)]*)*)\)", text)
    if conds:
        c = conds[-1].strip()
        c = re.sub(r"\s+", " ", c)
        return c
    if "else" in text:
        return "else"
    return "无条件"


def build_html(machines):
    total_states = sum(len(m["states"]) for m in machines)
    total_edges = sum(len(m["edges"]) for m in machines)

    sections = []
    for mi, m in enumerate(machines):
        st_states = m["states"]
        edges = m["edges"]

        # 计算每个状态的入/出度，用于着色
        out_deg = {s: 0 for s in st_states}
        in_deg = {s: 0 for s in st_states}
        for (f, t, c) in edges:
            out_deg[f] = out_deg.get(f, 0) + 1
            in_deg[t] = in_deg.get(t, 0) + 1

        # 状态卡片
        cards = []
        for s in st_states:
            n_in = in_deg.get(s, 0)
            n_out = out_deg.get(s, 0)
            # 分类：
            #   start 起始态（无入边，有出边）
            #   end   终态（有入边，无出边）
            #   aux   辅助枚举（无入边且无出边，如仲裁器 owner 值，非真实 FSM 状态）
            kind = ""
            if n_in == 0 and n_out == 0:
                kind = "aux"
            elif n_in == 0:
                kind = "start"
            elif n_out == 0:
                kind = "end"
            cards.append(
                f'<div class="state {kind}" data-st="{s}" data-m="{mi}">'
                f'<div class="sn">{s}</div>'
                f'<div class="sd">入 {n_in} · 出 {n_out}</div></div>'
            )

        # 转移表
        rows = []
        for (f, t, c) in edges:
            rows.append(
                f'<tr data-m="{mi}" data-from="{f}" data-to="{t}">'
                f'<td class="mono">{f}</td>'
                f'<td class="arrow">→</td>'
                f'<td class="mono">{t}</td>'
                f'<td class="cond">{html.escape(c)}</td></tr>'
            )

        sections.append(f"""
<div class="machine" id="m{mi}">
  <h2>{html.escape(m.get('name', 'FSM'))}
    <small>{len(st_states)} 状态 · {len(edges)} 转移</small></h2>

  <div class="diagram">
    {''.join(cards)}
  </div>

  <table class="edges">
    <thead><tr><th style="width:150px">当前状态</th><th style="width:40px"></th>
    <th style="width:150px">下一状态</th><th>转移条件</th></tr></thead>
    <tbody>{''.join(rows)}</tbody>
  </table>
</div>""")

    return f"""<!DOCTYPE html>
<html lang="zh-CN"><head><meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>FSM 状态转移图 · AXI→APB→AXI</title>
<style>
:root{{--bg:#0d1117;--panel:#161b22;--bd:#30363d;--fg:#c9d1d9;--dim:#8b949e;
--blue:#58a6ff;--green:#3fb950;--amber:#d29922;--purple:#bc8cff;--red:#f85149;
--mono:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace}}
*{{box-sizing:border-box;margin:0;padding:0}}
body{{background:var(--bg);color:var(--fg);font:13.5px/1.6 -apple-system,"Segoe UI","Noto Sans SC",sans-serif;
padding:0 0 50px}}
.wrap{{max-width:1180px;margin:0 auto;padding:0 24px}}
header{{border-bottom:1px solid var(--bd);padding:30px 0 22px;margin-bottom:26px}}
h1{{font-size:23px;font-weight:650}}
h1 small{{display:block;font-size:12.5px;color:var(--dim);font-weight:400;margin-top:7px}}
.stats{{display:flex;gap:24px;margin-top:16px;flex-wrap:wrap}}
.stat .v{{font-family:var(--mono);font-size:20px;font-weight:600;color:var(--blue)}}
.stat .l{{font-size:11px;color:var(--dim)}}
h2{{font-size:17px;margin:30px 0 14px;padding-bottom:8px;border-bottom:1px solid var(--bd)}}
h2 small{{font-size:12px;color:var(--dim);font-weight:400;margin-left:8px}}
.diagram{{display:flex;flex-wrap:wrap;gap:9px;margin:16px 0 22px;
padding:18px;background:var(--panel);border:1px solid var(--bd);border-radius:9px}}
.state{{background:#0e1a26;border:1px solid #33506e;border-radius:8px;
padding:10px 14px;min-width:128px;cursor:pointer;transition:.13s}}
.state:hover{{border-color:var(--blue);background:#1f6feb22}}
.state.on{{border-color:var(--blue);background:#1f6feb33;box-shadow:0 0 0 1px var(--blue)}}
.state.start{{border-color:#2a5c4a;background:#0d1f19}}
.state.end{{border-color:#5c4a2a;background:#1f190d}}
.state.aux{{border-color:#3a3a3a;background:#161616;opacity:.62}}
.state.aux .sn{{color:var(--dim)}}
.sn{{font-family:var(--mono);font-size:12.5px;color:#9db9d4}}
.state.start .sn{{color:var(--green)}}
.state.end .sn{{color:var(--amber)}}
.sd{{font-size:10.5px;color:var(--dim);margin-top:3px}}
table{{width:100%;border-collapse:collapse;font-size:12.5px}}
th,td{{padding:8px 11px;border-bottom:1px solid #21262d;text-align:left}}
th{{color:var(--dim);font-size:10.5px;text-transform:uppercase;letter-spacing:.4px;background:#11151b}}
tbody tr:hover{{background:#11151b}}
tr.dim{{opacity:.32}}
tr.hl{{background:#1f6feb22}}
.mono{{font-family:var(--mono);color:#79c0ff}}
.arrow{{color:#3d5266;text-align:center}}
.cond{{font-family:var(--mono);font-size:11.5px;color:#a5c9a5}}
footer{{margin-top:36px;padding-top:16px;border-top:1px solid var(--bd);
color:#6e7681;font-size:11.5px}}
code{{font-family:var(--mono);background:#0d1117;border:1px solid #21262d;
padding:1px 5px;border-radius:4px;color:#79c0ff}}
</style></head><body><div class="wrap">
<header>
<h1>FSM 状态转移图：AXI→APB→AXI
<small>从 SystemVerilog 源码解析生成 · 状态与转移条件均取自源码，非手绘</small></h1>
<div class="stats">
<div class="stat"><div class="v">{len(machines)}</div><div class="l">状态机</div></div>
<div class="stat"><div class="v">{total_states}</div><div class="l">状态总数</div></div>
<div class="stat"><div class="v">{total_edges}</div><div class="l">转移总数</div></div>
</div>
</header>
{''.join(sections)}
<footer>
生成：<code>python3 gen_fsm_html.py</code> ·
源码：<code>sv/axi_apb_bridge_sv.sv</code>（串行桥）、
<code>sv/axi_apb_bridge_dual_sv.sv</code>（并发桥）<br>
提示：点击状态卡片可高亮该状态的所有出边；绿色=起始态，琥珀色=终态，
灰色=辅助枚举值（无转移，如仲裁器 owner）。
</footer>
</div>
<script>
let cur = null;
document.querySelectorAll('.state').forEach(el => el.onclick = () => {{
  const st = el.dataset.st, mi = el.dataset.m;
  if (cur === st + mi) {{
    cur = null;
    document.querySelectorAll('.state').forEach(x => x.classList.remove('on'));
    document.querySelectorAll('tr').forEach(x => x.classList.remove('dim','hl'));
    return;
  }}
  cur = st + mi;
  document.querySelectorAll('.state').forEach(x => x.classList.remove('on'));
  el.classList.add('on');
  document.querySelectorAll('tbody tr').forEach(tr => {{
    const same = tr.dataset.m === mi;
    if (!same) return;
    if (tr.dataset.from === st) {{
      tr.classList.remove('dim'); tr.classList.add('hl');
    }} else {{
      tr.classList.remove('hl'); tr.classList.add('dim');
    }}
  }});
}});
</script>
</body></html>"""


def main():
    machines = []

    # 串行桥
    m1 = parse_fsm(SV / "axi_apb_bridge_sv.sv", "串行桥 axi_apb_bridge_sv")
    if m1:
        machines.append(m1)
        print(f"串行桥: {len(m1['states'])} 状态, {len(m1['edges'])} 转移")
        if m1.get("error"):
            print(f"  警告: {m1['error']}", file=sys.stderr)
    else:
        print("警告: 未解析到串行桥 FSM", file=sys.stderr)

    # 并发桥
    m2 = parse_fsm(SV / "axi_apb_bridge_dual_sv.sv", "并发双路径桥 axi_apb_bridge_dual_sv")
    if m2:
        machines.append(m2)
        print(f"并发桥: {len(m2['states'])} 状态, {len(m2['edges'])} 转移")
        if m2.get("error"):
            print(f"  警告: {m2['error']}", file=sys.stderr)

    if not machines:
        print("错误: 未解析到任何 FSM", file=sys.stderr)
        sys.exit(1)

    # 健全性检查
    all_states = set()
    bad = []
    for m in machines:
        st = set(m["states"])
        all_states |= st
        for (f, t, c) in m["edges"]:
            if f not in st:
                bad.append(f"{m.get('name')}: 转移源 {f} 不在枚举中")
            if t not in st:
                bad.append(f"{m.get('name')}: 转移目标 {t} 不在枚举中")
    if bad:
        print("\n❌ 校验失败：", file=sys.stderr)
        for b in bad:
            print("   -", b, file=sys.stderr)
        sys.exit(1)
    print("✅ 校验通过：所有转移端点均在状态枚举内")

    txt = build_html(machines)
    OUT.write_text(txt, encoding="utf-8")
    print(f"\n已生成: {OUT}  ({len(txt)} 字节)")


if __name__ == "__main__":
    main()
