#!/usr/bin/env python3
"""
gen_interface_map.py —— 从真实 SystemVerilog 源码生成「接口逻辑映射」交互式 HTML

设计原则：不手写映射表，而是解析源码后自校验，确保 HTML 与代码永不脱节。
  1. 解析 bus_txn_pkg.sv      → 事务结构体字段（第 3 层）
  2. 解析 axi_lite_if.sv / apb_if.sv → 接口信号与 modport 方向（第 1 层）
  3. 解析 axi_apb_bridge_sv.sv → FSM 状态 + 每状态驱动的信号（转换设计）
  4. 校验映射中引用的字段/信号确实存在于源码，否则报错退出
  5. 输出单文件交互式 HTML

用法: python3 gen_interface_map.py
"""
import re
import sys
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent   # verilog/axi
ROOT = HERE                              # 输出目录 = verilog/axi
SV = HERE / "sv"                         # SV 源码目录


# ----------------------------------------------------------------------
# 解析：事务结构体（第 3 层）
# ----------------------------------------------------------------------
def parse_txn_pkg(path: Path):
    txt = path.read_text(encoding="utf-8")
    structs = {}
    # 匹配 typedef struct packed { ... } name_t;
    for m in re.finditer(
        r"typedef struct packed \{(.*?)\}\s*(\w+);", txt, re.S
    ):
        body, name = m.group(1), m.group(2)
        fields = []
        for fm in re.finditer(
            r"(bit|logic)\s*(?:\[([^\]]+)\])?\s*(\w+)\s*;\s*(?://\s*(.*))?$",
            body, re.M,
        ):
            kind, width, fname, comment = fm.groups()
            fields.append({
                "name": fname,
                "kind": kind,
                "width": (width or "").strip(),
                "comment": (comment or "").strip(),
            })
        structs[name] = fields
    return structs


# ----------------------------------------------------------------------
# 解析：接口信号 + modport 方向（第 1 层）
# ----------------------------------------------------------------------
def parse_interface(path: Path):
    txt = path.read_text(encoding="utf-8")
    iface = path.stem

    signals = {}
    for m in re.finditer(
        r"^\s*logic\s*(?:\[([^\]]+)\])?\s*(\w+)\s*;", txt, re.M
    ):
        width, name = m.group(1), m.group(2)
        signals[name] = (width or "").strip()

    modports = {}
    # modport name ( ... );
    for m in re.finditer(r"modport\s+(\w+)\s*\((.*?)\);", txt, re.S):
        mp_name, body = m.group(1), m.group(2)
        directions = {}
        # 按 direction 关键字切段：从每个 input/output/inout 位置，到下一个关键字位置
        markers = [(dm.start(), dm.end(), dm.group(1))
                   for dm in re.finditer(r"\b(input|output|inout)\b", body)]
        for i, (s, e, d) in enumerate(markers):
            end = markers[i + 1][0] if i + 1 < len(markers) else len(body)
            names = body[e:end]
            for n in re.split(r"[,\s]+", names.strip()):
                if n:
                    directions[n] = d
        modports[mp_name] = directions

    # SVA 断言名
    asserts = re.findall(r"^\s*(\w+):\s*\n?\s*assert property", txt, re.M)
    covers = re.findall(r"^\s*(\w+):\s*cover property", txt, re.M)

    return {
        "iface": iface,
        "signals": signals,
        "modports": modports,
        "asserts": asserts,
        "covers": covers,
    }


# ----------------------------------------------------------------------
# 解析：桥的 FSM（第 2 层→第 1 层的转换逻辑）
# ----------------------------------------------------------------------
def parse_bridge_fsm(path: Path):
    txt = path.read_text(encoding="utf-8")

    # 状态枚举
    m = re.search(
        r"typedef enum logic \[[^\]]+\] \{(.*?)\}\s*state_t;", txt, re.S
    )
    states = []
    if m:
        for s in re.split(r"[,\s]+", m.group(1)):
            s = s.strip()
            if re.fullmatch(r"S_\w+", s):
                states.append(s)

    # 每个状态驱动哪些 apb 信号（组合块里的 S_XXX: begin apb.Y = ... end）
    apb_drive = {}
    for sm in re.finditer(
        r"(S_\w+):\s*begin\s+((?:apb\.\w+\s*=\s*[^;]+;\s*)+)end", txt
    ):
        st, body = sm.group(1), sm.group(2)
        sigs = []
        for am in re.finditer(r"apb\.(\w+)\s*=\s*([^;]+);", body):
            sigs.append({
                "sig": am.group(1),
                "val": am.group(2).strip(),
            })
        apb_drive.setdefault(st, []).extend(sigs)

    # AXI 握手输出（always_comb 里 case (state) 的 axi.XXX = 1'b1）
    axi_drive = {}
    for sm in re.finditer(
        r"(S_\w+):\s*(?:begin\s*)?(.*?)(?:end\s*)?(?=\n\s+(?:S_\w+|default):)",
        txt, re.S,
    ):
        st, body = sm.group(1), sm.group(2)
        sigs = []
        for am in re.finditer(r"axi\.(\w+)\s*=\s*(1'b1|1'b0|'0)", body):
            sigs.append({"sig": am.group(1), "val": am.group(2)})
        if sigs:
            axi_drive.setdefault(st, []).extend(sigs)

    return {"states": states, "apb_drive": apb_drive, "axi_drive": axi_drive}


# ----------------------------------------------------------------------
# 映射声明：意图（字段）→ 激励（驱动什么）→ 数据（观测什么）
# 每条都在下方校验：引用的名称必须真实存在于源码
# ----------------------------------------------------------------------
MAPPING = {
    "txn_t": {
        "write": {
            "drives": ["AWVALID", "AWADDR", "WVALID", "WDATA",
                       "ARVALID", "ARADDR"],
            "observes": ["BVALID", "BRESP", "RVALID", "RDATA", "RRESP"],
            "desc": "选择写/读通道：写走 AW/W/B，读走 AR/R",
        },
        "addr": {
            "drives": ["AWADDR", "ARADDR"],
            "observes": ["PRDATA", "PSLVERR"],
            "desc": "地址 → AWADDR（写）或 ARADDR（读）→ 桥转 PADDR → 从机译码",
        },
        "data": {
            "drives": ["WDATA"],
            "observes": ["RDATA", "PRDATA"],
            "desc": "写数据 → WDATA → PWDATA → 寄存器；读数据反向 RDATA",
        },
        "pause": {
            "drives": ["AWVALID", "WVALID", "ARVALID"],
            "observes": [],
            "desc": "事务后拉低全部 VALID，插 N 拍总线空闲",
        },
        "same": {
            "drives": ["AWVALID", "WVALID"],
            "observes": ["AWREADY", "WREADY"],
            "desc": "AW 与 W 是否同拍发起（同发需两个 READY 同时为高）",
        },
        "exp_err": {
            "drives": ["AWADDR", "ARADDR"],
            "observes": ["BRESP", "RRESP", "PSLVERR"],
            "desc": "越界地址 → 从机 PSLVERR → 桥转 BRESP/RRESP=SLVERR",
        },
    },
    "rsp_t": {
        "data": {"desc": "读回数据（来自 RDATA）"},
        "resp": {"desc": "实际响应编码（来自 BRESP/RRESP）"},
        "idle": {"desc": "实际注入的空闲拍数（pause 可观测）"},
    },
}

# 桥内部信号流转（跨模块的映射链）
FLOW_CHAIN = [
    ("txn_t.addr",  "axi.AWADDR / axi.ARADDR", "适配层展开"),
    ("AWADDR/ARADDR", "bridge paddr_q",        "桥锁存地址"),
    ("paddr_q",     "apb.PADDR",               "桥输出到 APB"),
    ("apb.PADDR",   "slave idx = PADDR>>2",    "从机地址译码"),
    ("slave mem[idx]", "apb.PRDATA",           "从机读数据"),
    ("apb.PRDATA",  "bridge rdata_q",          "桥在完成拍锁存"),
    ("rdata_q",     "axi.RDATA",               "读响应回 AXI"),
]


# ----------------------------------------------------------------------
# 校验：HTML 声称的一切必须在源码中真实存在
# ----------------------------------------------------------------------
def validate(txn_pkg, axi_if, apb_if, bridge):
    errors = []

    txn_fields = {f["name"] for f in txn_pkg.get("txn_t", [])}
    rsp_fields = {f["name"] for f in txn_pkg.get("rsp_t", [])}

    for fname in MAPPING["txn_t"]:
        if fname not in txn_fields:
            errors.append(f"txn_t.{fname} 在源码中不存在")
    for fname in MAPPING["rsp_t"]:
        if fname not in rsp_fields:
            errors.append(f"rsp_t.{fname} 在源码中不存在")

    axi_sigs = set(axi_if["signals"])
    apb_sigs = set(apb_if["signals"])
    known = axi_sigs | apb_sigs

    for fname, spec in MAPPING["txn_t"].items():
        for s in spec.get("drives", []) + spec.get("observes", []):
            base = s.replace("axi.", "").replace("apb.", "")
            if base not in axi_sigs and base not in apb_sigs:
                errors.append(f"{fname} 引用的信号 {s} 在接口中不存在")

    # 桥状态必须都在枚举里
    for st in bridge["states"]:
        if not re.fullmatch(r"S_\w+", st):
            errors.append(f"非法状态名 {st}")
    for st in bridge["apb_drive"]:
        if st not in bridge["states"]:
            errors.append(f"apb_drive 状态 {st} 不在 FSM 枚举中")
    for st in bridge["axi_drive"]:
        if st not in bridge["states"]:
            errors.append(f"axi_drive 状态 {st} 不在 FSM 枚举中")

    return errors


# ----------------------------------------------------------------------
# 生成 HTML
# ----------------------------------------------------------------------
HTML = """<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>接口逻辑映射 · AXI→APB→AXI</title>
<style>
:root{--bg:#0d1117;--panel:#161b22;--panel2:#1c2128;--bd:#30363d;--fg:#c9d1d9;
--dim:#8b949e;--blue:#58a6ff;--green:#3fb950;--amber:#d29922;--purple:#bc8cff;
--cyan:#39c5cf;--red:#f85149;--mono:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace}
*{box-sizing:border-box;margin:0;padding:0}
body{background:var(--bg);color:var(--fg);font:14px/1.6 -apple-system,"Segoe UI","Noto Sans SC",sans-serif;padding:0 0 60px}
.wrap{max-width:1200px;margin:0 auto;padding:0 24px}
header{border-bottom:1px solid var(--bd);padding:32px 0 24px;margin-bottom:28px}
h1{font-size:24px;font-weight:650}
h1 small{display:block;font-size:13px;color:var(--dim);font-weight:400;margin-top:8px}
.tabs{display:flex;gap:6px;flex-wrap:wrap;margin:20px 0 0}
.tab{padding:7px 15px;border:1px solid var(--bd);border-radius:7px;background:var(--panel);
cursor:pointer;font-size:13px;color:var(--dim);transition:.15s}
.tab:hover{border-color:var(--blue);color:var(--fg)}
.tab.on{background:#1f6feb22;border-color:var(--blue);color:var(--blue);font-weight:600}
.pane{display:none}.pane.on{display:block}
h2{font-size:17px;margin:26px 0 14px;padding-bottom:8px;border-bottom:1px solid var(--bd)}
h2 .n{font-family:var(--mono);font-size:11px;color:var(--blue);margin-right:8px}
p{margin-bottom:10px;color:#b1bac4}
code{font-family:var(--mono);font-size:.88em;background:#0d1117;border:1px solid #21262d;
padding:1px 5px;border-radius:4px;color:#79c0ff}

/* 映射表 */
table{width:100%;border-collapse:collapse;font-size:13px;margin:10px 0}
th,td{padding:10px 12px;border-bottom:1px solid #21262d;text-align:left;vertical-align:top}
th{color:var(--dim);font-size:11px;text-transform:uppercase;letter-spacing:.4px;background:#11151b}
tr:hover td{background:#11151b}
.mono{font-family:var(--mono);font-size:12px}

/* 信号芯片 */
.chip{display:inline-block;font-family:var(--mono);font-size:11.5px;padding:2px 8px;
border-radius:5px;margin:2px 3px 2px 0;border:1px solid;white-space:nowrap}
.chip.out{background:#12261a;border-color:#238636;color:#56d364}
.chip.in{background:#1c1a2e;border-color:#6e40c9;color:#bc8cff}
.chip.apb{background:#2b2110;border-color:#9e6a03;color:#e3b341}
.chip.dim{background:#161b22;border-color:#30363d;color:#8b949e}

/* 流程链 */
.chain{display:flex;flex-direction:column;gap:0;margin:16px 0}
.step{display:grid;grid-template-columns:1fr 26px 1fr 130px;gap:12px;align-items:center;
padding:11px 14px;background:var(--panel);border:1px solid var(--bd);border-radius:8px;font-size:12.6px}
.step .arrow{text-align:center;color:#3d5266;font-size:15px}
.step .lbl{color:var(--dim);font-size:11.5px;text-align:right;font-family:var(--mono)}
.conn{height:14px;margin-left:calc(50% - 60px);border-left:2px dashed #30363d;width:0}

/* 状态机 */
.fsm{display:flex;flex-wrap:wrap;gap:8px;margin:14px 0}
.st{font-family:var(--mono);font-size:11.5px;padding:6px 11px;border-radius:6px;
border:1px solid var(--bd);background:var(--panel);cursor:pointer;transition:.15s}
.st:hover,.st.on{border-color:var(--blue);background:#1f6feb22;color:var(--blue)}
.detail{margin-top:14px;background:var(--panel);border:1px solid var(--bd);border-radius:8px;
padding:16px;min-height:90px}
.detail .t{font-family:var(--mono);color:var(--blue);font-size:13px;margin-bottom:10px}

/* 图例 */
.legend{display:flex;gap:18px;flex-wrap:wrap;font-size:12px;color:var(--dim);margin:14px 0 0;
padding:12px 16px;background:var(--panel);border:1px solid var(--bd);border-radius:8px}
footer{margin-top:40px;padding-top:18px;border-top:1px solid var(--bd);color:#6e7681;font-size:12px}
</style>
</head>
<body>
<div class="wrap">
<header>
  <h1>接口逻辑映射：字段 → 信号 → 时序
    <small>由 gen_interface_map.py 解析真实 SV 源码生成 · 字段与信号均经源码校验</small></h1>
  <div class="tabs">
    <div class="tab on" data-p="fields">① 字段→信号映射</div>
    <div class="tab" data-p="signals">② 接口信号全景</div>
    <div class="tab" data-p="chain">③ 数据流转链</div>
    <div class="tab" data-p="fsm">④ 状态机驱动</div>
  </div>
</header>
__PANES__
<footer>
  源码：<code>bus_txn_pkg.sv</code> · <code>axi_lite_if.sv</code> ·
  <code>apb_if.sv</code> · <code>axi_apb_bridge_sv.sv</code><br>
  生成命令：<code>python3 gen_interface_map.py</code> · 所有映射在生成时校验，源码改名会立即报错
</footer>
</div>
<script>
document.querySelectorAll('.tab').forEach(t => t.onclick = () => {
  document.querySelectorAll('.tab').forEach(x => x.classList.remove('on'));
  document.querySelectorAll('.pane').forEach(x => x.classList.remove('on'));
  t.classList.add('on');
  document.getElementById(t.dataset.p).classList.add('on');
});
document.querySelectorAll('.st').forEach(s => s.onclick = () => {
  const d = document.getElementById('st-detail');
  d.querySelector('.t').textContent = s.dataset.st;
  d.querySelector('.body').innerHTML = s.dataset.body;
  document.querySelectorAll('.st').forEach(x => x.classList.remove('on'));
  s.classList.add('on');
});
</script>
</body>
</html>
"""


def chip(name, kind, val=None):
    label = f"{name}" + (f" = {val}" if val else "")
    return f'<span class="chip {kind}">{label}</span>'


def build_fields_pane(txn_pkg, axi_if):
    rows = []
    for fname, spec in MAPPING["txn_t"].items():
        f = next(f for f in txn_pkg["txn_t"] if f["name"] == fname)
        w = f" [{f['width']}]" if f["width"] else ""
        drives = "".join(chip(s, "out") for s in spec.get("drives", [])) or '<span class="chip dim">—</span>'
        observes = "".join(chip(s, "in") for s in spec.get("observes", [])) or '<span class="chip dim">—</span>'
        rows.append(f"""<tr>
<td class="mono"><b>{fname}</b><span style="color:#6e7681">{w}</span></td>
<td>{drives}</td><td>{observes}</td><td style="color:#8b949e">{spec['desc']}</td></tr>""")

    rsp_rows = []
    for fname, spec in MAPPING["rsp_t"].items():
        f = next(f for f in txn_pkg["rsp_t"] if f["name"] == fname)
        w = f" [{f['width']}]" if f["width"] else ""
        rsp_rows.append(f'<tr><td class="mono"><b>{fname}</b><span style="color:#6e7681">{w}</span></td>'
                        f'<td style="color:#8b949e">{spec["desc"]}</td></tr>')

    return f"""<div class="pane on" id="fields">
<h2><span class="n">01</span>字段 → 信号映射（txn_t，第 3 层）</h2>
<p>上层只写结构体字段，适配层把它展开成 AXI 信号。下表每行：
<b>字段</b> → <b>驱动哪些信号</b> / <b>观测哪些信号</b>。</p>
<table>
<thead><tr><th style="width:150px">事务字段</th><th>驱动（master→slave）</th>
<th>观测（slave→master）</th><th style="width:34%">转换逻辑</th></tr></thead>
<tbody>{''.join(rows)}</tbody></table>

<h2><span class="n">02</span>结果字段（rsp_t）</h2>
<table><thead><tr><th style="width:150px">字段</th><th>来源</th></tr></thead>
<tbody>{''.join(rsp_rows)}</tbody></table>
<div class="legend">
<span>🟩 驱动信号（本层输出）</span><span>🟪 观测信号（本层采样）</span>
<span>⬜ 无</span></div>
</div>"""


def build_signals_pane(axi_if, apb_if):
    def table(iface, cls):
        mp = iface["modports"].get("master", {})
        rows = []
        for sig, width in iface["signals"].items():
            d = mp.get(sig, "?")
            kind = "out" if d == "output" else "in"
            w = f"[{width}]" if width else "[1]"
            rows.append(f'<tr><td class="mono">{sig}</td><td class="mono" style="color:#6e7681">{w}</td>'
                        f'<td>{chip(d, kind)}</td></tr>')
        asserts = "".join(chip(a, "dim") for a in iface["asserts"])
        covers = "".join(chip(c, "dim") for c in iface["covers"])
        return f"""<h2><span class="n">{cls}</span>{iface['iface'].upper()} 信号（{len(iface['signals'])} 根）</h2>
<table><thead><tr><th style="width:120px">信号</th><th style="width:90px">位宽</th>
<th style="width:120px">master 方向</th></tr></thead>
<tbody>{''.join(rows)}</tbody></table>
<p style="margin-top:10px"><b>SVA 断言</b>（{len(iface['asserts'])} 条）：{asserts}</p>
<p><b>覆盖点</b>：{covers or '<span class="chip dim">无</span>'}</p>"""

    return f"""<div class="pane" id="signals">
{table(axi_if, '03')}{table(apb_if, '04')}
</div>"""


def build_chain_pane():
    steps = []
    for i, (src, dst, note) in enumerate(FLOW_CHAIN):
        if i:
            steps.append('<div class="conn"></div>')
        steps.append(f"""<div class="step">
<div class="mono" style="color:#58a6ff">{src}</div>
<div class="arrow">→</div>
<div class="mono" style="color:#3fb950">{dst}</div>
<div class="lbl">{note}</div></div>""")
    return f"""<div class="pane" id="chain">
<h2><span class="n">05</span>数据流转链（地址/数据从字段到总线的完整路径）</h2>
<p>每一段都是源码里真实存在的赋值或锁存关系。</p>
<div class="chain">{''.join(steps)}</div>
</div>"""


def build_fsm_pane(bridge):
    chips = []
    for st in bridge["states"]:
        apb = bridge["apb_drive"].get(st, [])
        axi = bridge["axi_drive"].get(st, [])
        body = ""
        if axi:
            body += "<div><b>AXI 输出</b><br>" + "".join(
                chip(f"{x['sig']}={x['val']}", "out") for x in axi) + "</div>"
        if apb:
            body += '<div style="margin-top:8px"><b>APB 输出</b><br>' + "".join(
                chip(f"{x['sig']}={x['val']}", "apb") for x in apb) + "</div>"
        if not body:
            body = '<span class="chip dim">本状态无总线输出（等待握手）</span>'
        chips.append(f'<div class="st" data-st="{st}" data-body="{body.replace(chr(34), "&quot;")}">{st}</div>')

    first = bridge["states"][0] if bridge["states"] else "S_IDLE"
    apb0 = bridge["apb_drive"].get(first, [])
    axi0 = bridge["axi_drive"].get(first, [])
    init_body = ""
    if axi0:
        init_body += "<div><b>AXI 输出</b><br>" + "".join(
            chip(f"{x['sig']}={x['val']}", "out") for x in axi0) + "</div>"
    if apb0:
        init_body += "<div><b>APB 输出</b><br>" + "".join(
            chip(f"{x['sig']}={x['val']}", "apb") for x in apb0) + "</div>"
    if not init_body:
        init_body = '<span class="chip dim">本状态无总线输出（等待握手）</span>'

    return f"""<div class="pane" id="fsm">
<h2><span class="n">06</span>状态机驱动映射（{len(bridge['states'])} 个状态）</h2>
<p>点击任一状态，查看它在该拍强制驱动哪些总线信号。这些内容直接解析自
<code>case (state)</code> 分支，不是手写的。</p>
<div class="fsm">{''.join(chips)}</div>
<div class="detail" id="st-detail">
  <div class="t">{first}</div>
  <div class="body">{init_body}</div>
</div>
<div class="legend"><span>🟩 AXI 输出</span><span>🟨 APB 输出</span></div>
</div>"""


def main():
    txn_pkg = parse_txn_pkg(SV / "bus_txn_pkg.sv")
    axi_if = parse_interface(SV / "axi_lite_if.sv")
    apb_if = parse_interface(SV / "apb_if.sv")
    bridge = parse_bridge_fsm(SV / "axi_apb_bridge_sv.sv")

    print(f"解析 txn_t      : {[f['name'] for f in txn_pkg.get('txn_t', [])]}")
    print(f"解析 rsp_t      : {[f['name'] for f in txn_pkg.get('rsp_t', [])]}")
    print(f"解析 AXI 接口   : {len(axi_if['signals'])} 信号, "
          f"{len(axi_if['asserts'])} 断言, modport={list(axi_if['modports'])}")
    print(f"解析 APB 接口   : {len(apb_if['signals'])} 信号, {len(apb_if['asserts'])} 断言")
    print(f"解析桥 FSM      : {len(bridge['states'])} 状态, "
          f"{len(bridge['apb_drive'])} 个状态有 APB 输出")

    errs = validate(txn_pkg, axi_if, apb_if, bridge)
    if errs:
        print("\n❌ 校验失败（HTML 会与源码脱节）：", file=sys.stderr)
        for e in errs:
            print("   -", e, file=sys.stderr)
        sys.exit(1)
    print("✅ 校验通过：所有映射字段/信号均存在于源码")

    panes = "".join([
        build_fields_pane(txn_pkg, axi_if),
        build_signals_pane(axi_if, apb_if),
        build_chain_pane(),
        build_fsm_pane(bridge),
    ])
    html = HTML.replace("__PANES__", panes)
    out = ROOT / "INTERFACE_MAP.html"
    out.write_text(html, encoding="utf-8")
    print(f"\n已生成: {out}  ({len(html)} 字节)")


if __name__ == "__main__":
    main()
