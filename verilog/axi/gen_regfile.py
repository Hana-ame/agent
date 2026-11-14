#!/usr/bin/env python3
"""
gen_regfile.py —— 从 JSON 规格生成 APB 寄存器从机 RTL + 测试 + 文档

设计原则：
  1. 单一事实来源：regfile_spec.json 是唯一描述，RTL/测试/文档全部由它派生
  2. 生成前校验：地址对齐、重叠、属性合法性、位宽一致
  3. 生成物带头部标记，避免与手写文件混淆

生成产物：
  sv/apb_slave_regfile_gen.sv   寄存器从机 RTL
  sv/tb_regfile_gen.sv          属性验证 testbench
  REGFILE.html                  寄存器手册（地址映射表 + 属性说明）

用法:
  python3 gen_regfile.py
  python3 gen_regfile.py --spec other.json
"""
import json
import sys
import html
from pathlib import Path

HERE = Path(__file__).resolve().parent      # verilog/axi
SV = HERE / "sv"
DEFAULT_SPEC = HERE / "regfile_spec.json"

ATTR_CODES = {"RSVD": 0, "RW": 1, "RO": 2, "W1C": 3}
VALID_ATTRS = set(ATTR_CODES)


def load_spec(path: Path):
    if not path.exists():
        print(f"错误：规格文件不存在: {path}", file=sys.stderr)
        sys.exit(1)
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        print(f"错误：JSON 解析失败: {e}", file=sys.stderr)
        sys.exit(1)


def validate(spec):
    """校验规格：返回错误列表"""
    errs = []
    bus = spec.get("bus", {})
    aw = bus.get("addr_width", 8)
    dw = bus.get("data_width", 32)

    regs = spec.get("registers", [])
    if not regs:
        errs.append("registers 为空")

    word_bytes = dw // 8
    seen_offsets = {}
    seen_names = set()

    for i, r in enumerate(regs):
        name = r.get("name", f"<#{i}>")
        # 必填字段
        for k in ("name", "offset", "attr", "reset"):
            if k not in r:
                errs.append(f"{name}: 缺少字段 {k}")
        if not all(k in r for k in ("name", "offset", "attr", "reset")):
            continue

        # 属性合法性
        if r["attr"] not in VALID_ATTRS:
            errs.append(f"{name}: 非法属性 {r['attr']}（合法: {sorted(VALID_ATTRS)}）")

        # 名字唯一
        if r["name"] in seen_names:
            errs.append(f"{name}: 名字重复")
        seen_names.add(r["name"])

        # 地址解析与对齐
        try:
            off = int(str(r["offset"]), 0)
        except ValueError:
            errs.append(f"{name}: offset 无法解析: {r['offset']}")
            continue
        if off % word_bytes != 0:
            errs.append(f"{name}: offset 0x{off:X} 未按 {word_bytes} 字节对齐")
        if off >= (1 << aw):
            errs.append(f"{name}: offset 0x{off:X} 超出地址空间 (AW={aw})")

        # 地址重叠
        if off in seen_offsets:
            errs.append(f"{name}: 地址 0x{off:X} 与 {seen_offsets[off]} 冲突")
        seen_offsets[off] = name

        # 复位值范围
        try:
            rst = int(str(r["reset"]), 0)
        except ValueError:
            errs.append(f"{name}: reset 无法解析: {r['reset']}")
            continue
        if rst >= (1 << dw):
            errs.append(f"{name}: reset 0x{rst:X} 超出 {dw} 位范围")

        # 语义检查：RO 有固定值才有意义；W1C 复位值应可清零
        if r["attr"] == "RO" and rst == 0:
            # 允许但提示
            pass

    return errs


def gen_rtl(spec):
    """生成寄存器从机 RTL"""
    bus = spec["bus"]
    aw = bus.get("addr_width", 8)
    dw = bus.get("data_width", 32)
    wc = bus.get("wait_cycles", 2)
    regs = spec["registers"]
    nreg = len(regs)
    mod = spec.get("module", "apb_slave_regfile_gen")

    # 按 offset 排序生成 case 项
    sorted_regs = sorted(regs, key=lambda r: int(str(r["offset"]), 0))

    attr_cases = []
    for r in sorted_regs:
        idx = int(str(r["offset"]), 0) // (dw // 8)
        attr_cases.append(f"            {idx}: return A_{r['attr']};   // {r['name']} "
                          f"@0x{int(str(r['offset']),0):02X}")

    reset_cases = []
    for r in sorted_regs:
        idx = int(str(r["offset"]), 0) // (dw // 8)
        rst = int(str(r["reset"]), 0)
        reset_cases.append(f"            {idx}: return {dw}'h{rst:0{dw//4}X};")

    reg_table = "\n".join(
        f"    //   [{int(str(r['offset']),0)//(dw//8):2d}] 0x{int(str(r['offset']),0):02X}  "
        f"{r['name']:<12} {r['attr']:<5} reset=0x{int(str(r['reset']),0):08X}   {r.get('description','')}"
        for r in sorted_regs
    )

    return f"""// ============================================================================
// {mod}：APB 寄存器从机（由 gen_regfile.py 从 regfile_spec.json 生成）
// ============================================================================
// ⚠️ 本文件为自动生成，请勿手工编辑。修改 regfile_spec.json 后重新生成。
//
// 寄存器映射：
{reg_table}
//
// 寄存器属性语义：
//   RO    只读：写被忽略，读返回复位时的固定值
//   RW    读写：写什么读什么
//   W1C   写 1 清零：写 1 的位清零，写 0 的位保持
//   RSVD  保留：读恒 0，写忽略
// ============================================================================
`ifndef {mod.upper()}_SV
`define {mod.upper()}_SV
`timescale 1ns / 1ps

module {mod} #(
    parameter int AW          = {aw},
    parameter int DW          = {dw},
    parameter int NREG        = {nreg},
    parameter int WAIT_CYCLES = {wc}
)(
    input  logic PCLK,
    input  logic PRESETn,
    apb_if.slave apb
);

    import apb_defs_pkg::*;

    // ---- 寄存器属性编码 ----
    localparam int A_RSVD = 0;
    localparam int A_RW   = 1;
    localparam int A_RO   = 2;
    localparam int A_W1C  = 3;

    // ---- 寄存器名到索引（供 TB / 软件参考）----
{chr(10).join(f"    localparam int {r['name']:<12} = {int(str(r['offset']),0)//(dw//8)};   // 0x{int(str(r['offset']),0):02X}" for r in sorted_regs)}

    // ---- 属性查询（按索引）----
    function automatic int attr_of(input int i);
        case (i)
{chr(10).join(attr_cases)}
            default: return A_RSVD;   // 未定义地址视为保留
        endcase
    endfunction

    // ---- 复位值查询（按索引）----
    function automatic logic [DW-1:0] reset_of(input int i);
        case (i)
{chr(10).join(reset_cases)}
            default: return '0;
        endcase
    endfunction

    // ---- 寄存器堆 ----
    logic [DW-1:0] mem [0:NREG-1];

    // ---- 地址译码 ----
    logic [AW-1:0] idx;
    logic          in_range;
    assign idx = apb.PADDR >> 2;
    assign in_range = (idx < NREG);

    // ---- 写数据按属性变换 ----
    function automatic logic [DW-1:0] write_transform(
        input logic [DW-1:0] old_v,
        input logic [DW-1:0] wr_v,
        input int            attr
    );
        case (attr)
            A_RW:  return wr_v;
            A_W1C: return old_v & ~wr_v;
            default: return old_v;      // RO / RSVD 忽略写
        endcase
    endfunction

    // ---- 读数据按属性变换 ----
    function automatic logic [DW-1:0] read_transform(
        input logic [DW-1:0] cur_v,
        input int            attr
    );
        return (attr == A_RSVD) ? '0 : cur_v;
    endfunction

    // ---- 阶段状态机 ----
    apb_phase_t phase;
    logic [7:0] wait_cnt;

    always_ff @(posedge PCLK or negedge PRESETn) begin
        if (!PRESETn) begin
            phase       <= APB_IDLE;
            wait_cnt    <= '0;
            apb.PREADY  <= 1'b0;
            apb.PSLVERR <= 1'b0;
            apb.PRDATA  <= '0;
            for (int i = 0; i < NREG; i++)
                mem[i] <= reset_of(i);
        end else begin
            case (phase)
                APB_IDLE: begin
                    apb.PREADY  <= 1'b0;
                    apb.PSLVERR <= 1'b0;
                    if (apb.PSEL && !apb.PENABLE) begin
                        wait_cnt <= WAIT_CYCLES[7:0];
                        phase    <= APB_SETUP;
                    end
                end
                APB_SETUP: begin
                    apb.PREADY  <= 1'b0;
                    apb.PSLVERR <= 1'b0;
                    if (apb.PENABLE)
                        phase <= APB_ACC;
                end
                APB_ACC: begin
                    if (wait_cnt == 0) begin
                        apb.PREADY  <= 1'b1;
                        apb.PSLVERR <= in_range ? 1'b0 : 1'b1;
                        if (in_range && apb.PWRITE)
                            mem[idx] <= write_transform(mem[idx], apb.PWDATA, attr_of(idx));
                        if (!apb.PWRITE)
                            apb.PRDATA <= in_range ? read_transform(mem[idx], attr_of(idx)) : '0;
                        phase <= APB_IDLE;
                    end else begin
                        apb.PREADY  <= 1'b0;
                        wait_cnt    <= wait_cnt - 1'b1;
                    end
                end
                default: begin
                    phase       <= APB_IDLE;
                    apb.PREADY  <= 1'b0;
                    apb.PSLVERR <= 1'b0;
                end
            endcase
        end
    end

endmodule
`endif // {mod.upper()}_SV
"""


def gen_tb(spec):
    """生成属性验证 testbench"""
    bus = spec["bus"]
    aw = bus.get("addr_width", 8)
    dw = bus.get("data_width", 32)
    regs = sorted(spec["registers"], key=lambda r: int(str(r["offset"]), 0))
    nreg = len(regs)
    mod = spec.get("module", "apb_slave_regfile_gen")

    # 按属性分组生成测试调用
    ro_regs = [r for r in regs if r["attr"] == "RO"]
    w1c_regs = [r for r in regs if r["attr"] == "W1C"]
    rsvd_regs = [r for r in regs if r["attr"] == "RSVD"]
    rw_regs = [r for r in regs if r["attr"] == "RW"]

    def off(r):
        return int(str(r["offset"]), 0)

    def rst(r):
        return int(str(r["reset"]), 0)

    ro_body = "\n".join(
        f'        // {r["name"]}: 初始 0x{rst(r):08X}\n'
        f'        read_reg({aw}\'h{off(r):02X}, d);\n'
        f'        checks++;\n'
        f'        if (d !== {dw}\'h{rst(r):08X}) begin\n'
        f'            $display("  [RO] FAIL: {r["name"]} 初值=%h expect {rst(r):08X}", d);\n'
        f'            errors++;\n'
        f'        end\n'
        f'        write_reg({aw}\'h{off(r):02X}, {dw}\'hFFFF_FFFF);   // 应被忽略\n'
        f'        read_reg({aw}\'h{off(r):02X}, d);\n'
        f'        checks++;\n'
        f'        if (d !== {dw}\'h{rst(r):08X}) begin\n'
        f'            $display("  [RO] FAIL: {r["name"]} 被写改动=%h", d);\n'
        f'            errors++;\n'
        f'        end'
        for r in ro_regs
    ) or "        // （规格中无 RO 寄存器）"

    w1c_body = "\n".join(
        f'        // {r["name"]}: 写全 1 应清零\n'
        f'        write_reg({aw}\'h{off(r):02X}, {dw}\'hFFFF_FFFF);\n'
        f'        read_reg({aw}\'h{off(r):02X}, d);\n'
        f'        checks++;\n'
        f'        if (d !== \'0) begin\n'
        f'            $display("  [W1C] FAIL: {r["name"]} 写 1 后应 0，实际=%h", d);\n'
        f'            errors++;\n'
        f'        end'
        for r in w1c_regs
    ) or "        // （规格中无 W1C 寄存器）"

    rsvd_body = "\n".join(
        f'        // {r["name"]}: 读恒 0\n'
        f'        write_reg({aw}\'h{off(r):02X}, {dw}\'hDEAD_BEEF);\n'
        f'        read_reg({aw}\'h{off(r):02X}, d);\n'
        f'        checks++;\n'
        f'        if (d !== \'0) begin\n'
        f'            $display("  [RSVD] FAIL: {r["name"]} 读应 0，实际=%h", d);\n'
        f'            errors++;\n'
        f'        end'
        for r in rsvd_regs
    ) or "        // （规格中无 RSVD 寄存器）"

    rw_body = "\n".join(
        f'        // {r["name"]}: 写随机值读回一致\n'
        f'        wv = $urandom;\n'
        f'        write_reg({aw}\'h{off(r):02X}, wv);\n'
        f'        read_reg({aw}\'h{off(r):02X}, d);\n'
        f'        checks++;\n'
        f'        if (d !== wv) begin\n'
        f'            $display("  [RW] FAIL: {r["name"]} 写=%h 读=%h", wv, d);\n'
        f'            errors++;\n'
        f'        end'
        for r in rw_regs
    ) or "        // （规格中无 RW 寄存器）"

    reg_map = "\n".join(
        f"    //   0x{off(r):02X}  {r['name']:<12} {r['attr']:<5} reset=0x{rst(r):08X}"
        for r in regs
    )

    return f"""// ============================================================================
// tb_regfile_gen：属性验证（由 gen_regfile.py 从 regfile_spec.json 生成）
// ============================================================================
// ⚠️ 自动生成，勿手工编辑。
//
// 覆盖寄存器（{len(regs)} 个）：
{reg_map}
// ============================================================================
`ifndef TB_REGFILE_GEN_SV
`define TB_REGFILE_GEN_SV
`timescale 1ns / 1ps

module tb_regfile_gen #(
    parameter int AW   = {aw},
    parameter int DW   = {dw},
    parameter int NREG = {nreg}
);

    import bus_txn_pkg::*;
    import axi_defs_pkg::*;

    logic ACLK = 0, ARESETn = 0;
    always #5 ACLK = ~ACLK;

    axi_lite_if #(.AW(AW), .DW(DW)) axi(ACLK);
    apb_if      #(.AW(AW), .DW(DW)) apb(ACLK);

    axi_apb_bridge_sv #(.AW(AW), .DW(DW)) bridge (
        .ACLK(ACLK), .ARESETn(ARESETn), .axi(axi), .apb(apb)
    );
    {mod} #(.AW(AW), .DW(DW), .NREG(NREG), .WAIT_CYCLES({bus.get('wait_cycles',2)})) dut (
        .PCLK(ACLK), .PRESETn(ARESETn), .apb(apb)
    );
    axi_txn_host #(.AW(AW), .DW(DW)) host (
        .ACLK(ACLK), .ARESETn(ARESETn), .axi(axi)
    );

    int errors = 0;
    int checks = 0;

    task automatic do_reset();
        ARESETn = 0;
        repeat (4) @(posedge ACLK);
        ARESETn = 1;
        repeat (2) @(posedge ACLK);
    endtask

    function automatic txn_t mk(input bit wr, input [AW-1:0] a, input [DW-1:0] d);
        txn_t t;
        t = '0;
        t.write = wr; t.addr = a; t.data = d;
        return t;
    endfunction

    task automatic write_reg(input [AW-1:0] a, input [DW-1:0] d);
        txn_t t; rsp_t r;
        t = mk(1'b1, a, d);
        host.execute(t, r);
        checks++;
        if (r.resp !== RESP_OKAY) begin
            $display("  [TB] FAIL: 写 addr=%0h resp=%s", a, resp_name(r.resp));
            errors++;
        end
    endtask

    task automatic read_reg(input [AW-1:0] a, output [DW-1:0] d);
        txn_t t; rsp_t r;
        t = mk(1'b0, a, '0);
        host.execute(t, r);
        checks++;
        d = r.data;
        if (r.resp !== RESP_OKAY) begin
            $display("  [TB] FAIL: 读 addr=%0h resp=%s", a, resp_name(r.resp));
            errors++;
        end
    endtask

    logic [DW-1:0] d, wv;

    initial begin
        $display("======================================================");
        $display(" generated regfile test ({len(regs)} registers)");
        $display("======================================================");
        do_reset();

        $display("== RO 寄存器（只读）==");
{ro_body}

        $display("== W1C 寄存器（写 1 清零）==");
{w1c_body}

        $display("== RSVD 寄存器（保留）==");
{rsvd_body}

        $display("== RW 寄存器（读写）==");
{rw_body}

        $display("------------------------------------------------------");
        $display(" regfile-gen result: %0d checks, %0d errors", checks, errors);
        if (errors == 0) begin
            $display("PASS: generated regfile attributes");
            $finish;
        end else begin
            $display("FAIL: %0d errors", errors);
            $fatal(1, "generated regfile test failed");
        end
    end

    initial begin
        $dumpfile("tb_regfile_gen.vcd");
        $dumpvars(0, tb_regfile_gen);
    end

endmodule
`endif // TB_REGFILE_GEN_SV
"""


def gen_html(spec):
    """生成寄存器手册 HTML"""
    bus = spec["bus"]
    aw = bus.get("addr_width", 8)
    dw = bus.get("data_width", 32)
    regs = sorted(spec["registers"], key=lambda r: int(str(r["offset"]), 0))

    attr_color = {
        "RO": ("#58a6ff", "只读：写忽略，读固定值"),
        "RW": ("#3fb950", "读写：写什么读什么"),
        "W1C": ("#d29922", "写 1 清零：写 1 的位清 0，写 0 保持"),
        "RSVD": ("#8b949e", "保留：读 0，写忽略"),
    }

    rows = []
    for r in regs:
        off = int(str(r["offset"]), 0)
        rst = int(str(r["reset"]), 0)
        color, tip = attr_color[r["attr"]]
        rows.append(f"""<tr>
<td class="mono">0x{off:02X}</td>
<td class="mono">[{off//(dw//8)}]</td>
<td class="name">{html.escape(r['name'])}</td>
<td><span class="attr" style="color:{color};border-color:{color}44">{r['attr']}</span></td>
<td class="mono">0x{rst:08X}</td>
<td class="desc">{html.escape(r.get('description',''))}</td>
</tr>""")

    # 统计
    from collections import Counter
    cnt = Counter(r["attr"] for r in regs)
    stat_chips = "".join(
        f'<span class="chip" style="color:{attr_color[a][0]}">{a}: {n}</span>'
        for a, n in sorted(cnt.items())
    )

    attr_desc = "\n".join(
        f'<div class="attrrow"><span class="attr" style="color:{c};border-color:{c}44">{a}</span>'
        f'<span class="atip">{t}</span></div>'
        for a, (c, t) in attr_color.items()
    )

    return f"""<!DOCTYPE html>
<html lang="zh-CN"><head><meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>寄存器手册 · {html.escape(spec.get('module','regfile'))}</title>
<style>
:root{{--bg:#0d1117;--panel:#161b22;--bd:#30363d;--fg:#c9d1d9;--dim:#8b949e;
--blue:#58a6ff;--mono:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace}}
*{{box-sizing:border-box;margin:0;padding:0}}
body{{background:var(--bg);color:var(--fg);font:13.5px/1.6 -apple-system,"Segoe UI","Noto Sans SC",sans-serif;padding:0 0 50px}}
.wrap{{max-width:1000px;margin:0 auto;padding:0 24px}}
header{{border-bottom:1px solid var(--bd);padding:32px 0 22px;margin-bottom:26px}}
h1{{font-size:23px;font-weight:650}}
h1 small{{display:block;font-size:12.5px;color:var(--dim);font-weight:400;margin-top:8px}}
.meta{{display:flex;gap:20px;margin-top:16px;flex-wrap:wrap;font-family:var(--mono);font-size:12px;color:var(--dim)}}
.meta b{{color:var(--blue);font-weight:600}}
h2{{font-size:16.5px;margin:28px 0 12px;padding-bottom:8px;border-bottom:1px solid var(--bd)}}
table{{width:100%;border-collapse:collapse;font-size:12.5px}}
th,td{{padding:9px 11px;border-bottom:1px solid #21262d;text-align:left}}
th{{color:var(--dim);font-size:10.5px;text-transform:uppercase;letter-spacing:.4px;background:#11151b}}
tbody tr:hover{{background:#11151b}}
.mono{{font-family:var(--mono);color:#79c0ff}}
.name{{font-family:var(--mono);font-weight:600;color:#eaf2fa}}
.attr{{display:inline-block;font-family:var(--mono);font-size:10.5px;padding:1px 7px;
border:1px solid;border-radius:4px}}
.desc{{color:#8b949e;font-size:12px}}
.chip{{font-family:var(--mono);font-size:11.5px;padding:3px 10px;border:1px solid var(--bd);
border-radius:5px;margin-right:8px;background:var(--panel)}}
.attrrow{{display:flex;gap:14px;align-items:baseline;padding:8px 0;border-bottom:1px solid #21262d}}
.atip{{font-size:12.5px;color:#8b949e}}
pre{{background:#0b1017;border:1px solid #222c37;border-radius:8px;padding:14px 16px;
font-family:var(--mono);font-size:12px;color:#b8c8d8;overflow-x:auto}}
footer{{margin-top:34px;padding-top:16px;border-top:1px solid var(--bd);color:#6e7681;font-size:11.5px}}
code{{font-family:var(--mono);background:#0d1117;border:1px solid #21262d;padding:1px 5px;
border-radius:4px;color:#79c0ff}}
</style></head><body><div class="wrap">
<header>
<h1>{html.escape(spec.get('module','regfile'))} · 寄存器手册
<small>由 gen_regfile.py 从 regfile_spec.json 生成 · 规格是唯一事实来源</small></h1>
<div class="meta">
<span>地址位宽 <b>{aw}</b></span>
<span>数据位宽 <b>{dw}</b></span>
<span>寄存器数 <b>{len(regs)}</b></span>
<span>wait states <b>{bus.get('wait_cycles',2)}</b></span>
</div>
<div style="margin-top:14px">{stat_chips}</div>
</header>

<h2>地址映射表</h2>
<table><thead><tr>
<th style="width:80px">偏移</th><th style="width:55px">索引</th>
<th style="width:150px">名称</th><th style="width:80px">属性</th>
<th style="width:110px">复位值</th><th>说明</th>
</tr></thead><tbody>{''.join(rows)}</tbody></table>

<h2>寄存器属性语义</h2>
{attr_desc}

<h2>验证覆盖</h2>
<pre><span style="color:#5b7186"># 生成 RTL + TB + 本手册</span>
python3 verilog/axi/gen_regfile.py

<span style="color:#5b7186"># 运行生成的测试</span>
bash verilog/sim.sh regfilegen</pre>

<footer>
生成命令 <code>python3 gen_regfile.py</code> ·
规格 <code>regfile_spec.json</code> ·
生成物 <code>sv/apb_slave_regfile_gen.sv</code>、<code>sv/tb_regfile_gen.sv</code>
</footer>
</div></body></html>"""


def main():
    spec_path = DEFAULT_SPEC
    args = sys.argv[1:]
    if "--spec" in args:
        spec_path = Path(args[args.index("--spec") + 1])

    spec = load_spec(spec_path)
    print(f"读取规格: {spec_path.name}")

    errs = validate(spec)
    if errs:
        print("\n❌ 规格校验失败：", file=sys.stderr)
        for e in errs:
            print("   -", e, file=sys.stderr)
        sys.exit(1)
    regs = spec["registers"]
    print(f"✅ 规格校验通过：{len(regs)} 个寄存器")
    from collections import Counter
    cnt = Counter(r["attr"] for r in regs)
    print("   属性分布: " + " · ".join(f"{a}={n}" for a, n in sorted(cnt.items())))

    mod = spec.get("module", "apb_slave_regfile_gen")

    rtl_path = SV / f"{mod}.sv"
    rtl_path.write_text(gen_rtl(spec), encoding="utf-8")
    print(f"\n已生成 RTL: {rtl_path.name}  ({rtl_path.stat().st_size} 字节)")

    tb_path = SV / "tb_regfile_gen.sv"
    tb_path.write_text(gen_tb(spec), encoding="utf-8")
    print(f"已生成 TB : {tb_path.name}  ({tb_path.stat().st_size} 字节)")

    html_path = HERE / "REGFILE.html"
    txt = gen_html(spec)
    html_path.write_text(txt, encoding="utf-8")
    print(f"已生成手册: {html_path.name}  ({len(txt)} 字节)")


if __name__ == "__main__":
    main()
