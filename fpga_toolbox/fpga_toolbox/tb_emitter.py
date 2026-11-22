"""Tools 5 & 6: Contract validation and zero-touch SV wrapper / SVA+bind / cocotb codegen."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Optional

if __package__ in (None, ""):
    import sys as _sys
    from pathlib import Path as _Path

    _sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
    __package__ = "fpga_toolbox"

from .core.types import (
    OPTIONAL_AXIL_SIGNALS,
    REQUIRED_AXIL_SIGNALS,
    CompiledUnitIR,
    err_result,
    ok_result,
)
from .core.validator import build_and_validate_ir, load_raw_spec


def _emit_wrap(u: CompiledUnitIR) -> str:
    b, ax = u.boundary, u.boundary.axil
    pm = ax.port_map
    clk_p, rst_p = b.clock.port, b.reset.port
    params = " #(\n" + ",\n".join(f"    .{k}({v})" for k, v in b.parameters.items()) + "\n  )" if b.parameters else ""
    conns = [
        f"    .{clk_p}({clk_p})",
        f"    .{rst_p}({rst_p})",
    ] + [f"    .{pm[s]}(s_axil_{s})" for s in REQUIRED_AXIL_SIGNALS]
    conns += [f"    .{pm[s]}(s_axil_{s})" for s in OPTIONAL_AXIL_SIGNALS if s in pm]
    obs_decls = []
    # A port already wired through clock/reset/port_map must never be connected twice.
    connected = {clk_p, rst_p} | set(pm.values())
    for o in b.out_of_scope_ports:
        if o.port in connected:
            continue
        connected.add(o.port)
        if o.direction == "input":
            conns.append(f"    .{o.port}({o.tie_off})")
        else:
            w = f"[{o.width-1}:0] " if o.width > 1 else ""
            obs_decls.append(f"  logic {w}obs_{o.port};")
            conns.append(f"    .{o.port}(obs_{o.port})")
    obs_s = "\n".join(obs_decls) + ("\n\n" if obs_decls else "")
    # Pre-join: a backslash inside an f-string expression needs Python 3.12+.
    conn_s = ",\n".join(conns)
    return f"""`timescale 1ns/1ps
module {b.module_name}_axil_wrap #(parameter int ADDR_W={ax.addr_width}, DATA_W={ax.data_width}, STRB_W={ax.strb_width}) (
  input logic {clk_p}, input logic {rst_p},
  input logic [ADDR_W-1:0] s_axil_awaddr, input logic [2:0] s_axil_awprot, input logic s_axil_awvalid, output logic s_axil_awready,
  input logic [DATA_W-1:0] s_axil_wdata, input logic [STRB_W-1:0] s_axil_wstrb, input logic s_axil_wvalid, output logic s_axil_wready,
  output logic [1:0] s_axil_bresp, output logic s_axil_bvalid, input logic s_axil_bready,
  input logic [ADDR_W-1:0] s_axil_araddr, input logic [2:0] s_axil_arprot, input logic s_axil_arvalid, output logic s_axil_arready,
  output logic [DATA_W-1:0] s_axil_rdata, output logic [1:0] s_axil_rresp, output logic s_axil_rvalid, input logic s_axil_rready
);
{obs_s}  {b.module_name}{params} u_dut (
{conn_s}
  );
  {b.module_name}_axil_sva_checker #(.ADDR_W(ADDR_W), .DATA_W(DATA_W), .STRB_W(STRB_W), .MAX_RESP_CYCLES({u.test_plan.protocol_rules.max_resp_cycles})) u_chk (
    .clk({clk_p}), .rst_n({rst_p if b.reset.active_low else "~" + rst_p}),
    .awaddr(s_axil_awaddr), .awvalid(s_axil_awvalid), .awready(s_axil_awready),
    .wdata(s_axil_wdata), .wstrb(s_axil_wstrb), .wvalid(s_axil_wvalid), .wready(s_axil_wready),
    .bresp(s_axil_bresp), .bvalid(s_axil_bvalid), .bready(s_axil_bready),
    .araddr(s_axil_araddr), .arvalid(s_axil_arvalid), .arready(s_axil_arready),
    .rdata(s_axil_rdata), .rresp(s_axil_rresp), .rvalid(s_axil_rvalid), .rready(s_axil_rready)
  );
endmodule
"""


def _emit_sva(u: CompiledUnitIR) -> str:
    b, ax, pr = u.boundary, u.boundary.axil, u.test_plan.protocol_rules
    return f"""`timescale 1ns/1ps
module {b.module_name}_axil_sva_checker #(parameter int ADDR_W={ax.addr_width}, DATA_W={ax.data_width}, STRB_W={ax.strb_width}, MAX_RESP_CYCLES={pr.max_resp_cycles}) (
  input logic clk, input logic rst_n,
  input logic [ADDR_W-1:0] awaddr, input logic awvalid, input logic awready,
  input logic [DATA_W-1:0] wdata, input logic [STRB_W-1:0] wstrb, input logic wvalid, input logic wready,
  input logic [1:0] bresp, input logic bvalid, input logic bready,
  input logic [ADDR_W-1:0] araddr, input logic arvalid, input logic arready,
  input logic [DATA_W-1:0] rdata, input logic [1:0] rresp, input logic rvalid, input logic rready
);
  logic prev_awvalid, prev_awready, prev_wvalid, prev_wready, prev_arvalid, prev_arready, prev_bvalid, prev_bready, prev_rvalid, prev_rready;
  int unsigned wr_cnt, rd_cnt;
  logic wr_pend, rd_pend;
  always_ff @(posedge clk) begin
    if (!rst_n) begin
      {{prev_awvalid, prev_awready, prev_wvalid, prev_wready, prev_arvalid, prev_arready,
        prev_bvalid, prev_bready, prev_rvalid, prev_rready, wr_pend, rd_pend}} <= '0;
      wr_cnt <= 0; rd_cnt <= 0;
      assert (!bvalid && !rvalid) else $error("[RULE:P_002] Response valid high during reset");
    end else begin
      if (prev_awvalid && !prev_awready) assert (awvalid) else $error("[RULE:P_001] AWVALID dropped");
      if (prev_wvalid  && !prev_wready)  assert (wvalid)  else $error("[RULE:P_001] WVALID dropped");
      if (prev_arvalid && !prev_arready) assert (arvalid) else $error("[RULE:P_001] ARVALID dropped");
      if (prev_bvalid  && !prev_bready)  assert (bvalid)  else $error("[RULE:P_001] BVALID dropped");
      if (prev_rvalid  && !prev_rready)  assert (rvalid)  else $error("[RULE:P_001] RVALID dropped");
      prev_awvalid <= awvalid; prev_awready <= awready; prev_wvalid <= wvalid; prev_wready <= wready;
      prev_arvalid <= arvalid; prev_arready <= arready; prev_bvalid <= bvalid; prev_bready <= bready;
      prev_rvalid  <= rvalid;  prev_rready  <= rready;
      if (awvalid && awready) begin wr_pend <= 1'b1; wr_cnt <= 0; end
      else if (bvalid && bready) begin wr_pend <= 1'b0; wr_cnt <= 0; end
      else if (wr_pend) begin wr_cnt <= wr_cnt + 1; assert (wr_cnt < MAX_RESP_CYCLES) else $error("[RULE:P_003] BVALID timeout"); end
      if (arvalid && arready) begin rd_pend <= 1'b1; rd_cnt <= 0; end
      else if (rvalid && rready) begin rd_pend <= 1'b0; rd_cnt <= 0; end
      else if (rd_pend) begin rd_cnt <= rd_cnt + 1; assert (rd_cnt < MAX_RESP_CYCLES) else $error("[RULE:P_003] RVALID timeout"); end
    end
  end
endmodule
"""


def _emit_bind(u: CompiledUnitIR) -> str:
    b, ax = u.boundary, u.boundary.axil
    pm = ax.port_map
    tgt = b.instance_hierarchy or b.module_name
    rst = b.reset.port if b.reset.active_low else f"~{b.reset.port}"
    return f"""bind {tgt} {b.module_name}_axil_sva_checker #(.ADDR_W({ax.addr_width}), .DATA_W({ax.data_width}), .STRB_W({ax.strb_width}), .MAX_RESP_CYCLES({u.test_plan.protocol_rules.max_resp_cycles})) u_bind_chk (
  .clk({b.clock.port}), .rst_n({rst}),
  .awaddr({pm['awaddr']}), .awvalid({pm['awvalid']}), .awready({pm['awready']}),
  .wdata({pm['wdata']}), .wstrb({pm['wstrb']}), .wvalid({pm['wvalid']}), .wready({pm['wready']}),
  .bresp({pm['bresp']}), .bvalid({pm['bvalid']}), .bready({pm['bready']}),
  .araddr({pm['araddr']}), .arvalid({pm['arvalid']}), .arready({pm['arready']}),
  .rdata({pm['rdata']}), .rresp({pm['rresp']}), .rvalid({pm['rvalid']}), .rready({pm['rready']})
);
"""


def _emit_cocotb(u: CompiledUnitIR) -> str:
    b, ax, tp = u.boundary, u.boundary.axil, u.test_plan
    regs = json.dumps([asdict(r) for r in tp.register_rules], indent=2)
    ills = json.dumps([asdict(i) for i in tp.illegal_addr_rules], indent=2)
    rst_assert = 0 if b.reset.active_low else 1
    rst_idle = 1 - rst_assert
    return f'''import json, random, cocotb
from pathlib import Path
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge
from cocotbext.axi import AxiLiteBus, AxiLiteMaster, AxiResp

REGS = json.loads(r"""{regs}""")
ILLS = json.loads(r"""{ills}""")
STRESS = {{"random_backpressure": {tp.stress_profile.random_backpressure}, "iterations": {tp.stress_profile.iterations}, "seed": {tp.stress_profile.seed}}}
REPORT = {{"module": "{b.module_name}", "seed": {tp.stress_profile.seed}, "checks": []}}

def _rec(rid, cat, ok, detail):
    REPORT["checks"].append({{"rule_id": rid, "category": cat, "status": "PASS" if ok else "FAIL", "detail": detail}})
    Path("test_results_{b.module_name}.json").write_text(json.dumps(REPORT, indent=2), encoding="utf-8")

@cocotb.test()
async def test_axil_rules(dut):
    # Phase 1: startup (power-on reset & settle)
    rng = random.Random(STRESS["seed"])
    cocotb.start_soon(Clock(dut.{b.clock.port}, {round(1000.0 / b.clock.freq_mhz, 3)}, units="ns").start())
    dut.{b.reset.port}.value = {rst_assert}
    for _ in range(4):
        await RisingEdge(dut.{b.clock.port})
    dut.{b.reset.port}.value = {rst_idle}
    for _ in range(2):
        await RisingEdge(dut.{b.clock.port})
    m = AxiLiteMaster(AxiLiteBus.from_prefix(dut, "s_axil"), dut.{b.clock.port}, dut.{b.reset.port}, reset_active_level={not b.reset.active_low})

    # Phase 2: reset value + write/readback (honoring access mode & write_mask)
    for r in REGS:
        acc = r["access"]
        if acc != "WO":
            res = await m.read(r["offset"], {ax.strb_width})
            got = int.from_bytes(res.data, "little") & r["read_mask"]
            exp = r["reset_value"] & r["read_mask"]
            _rec(r["rule_id"], "reset_value", res.resp == AxiResp.OKAY and got == exp, f"got=0x{{got:X}} exp=0x{{exp:X}}")
        if acc == "RW" and r["write_mask"]:
            pat = 0xA5A55A5A & r["write_mask"]
            wres = await m.write(r["offset"], pat.to_bytes({ax.strb_width}, "little"))
            rres = await m.read(r["offset"], {ax.strb_width})
            got = int.from_bytes(rres.data, "little") & r["write_mask"] & r["read_mask"]
            exp = pat & r["read_mask"]
            _rec(r["rule_id"] + "_WR", "rw_write_readback", wres.resp == AxiResp.OKAY and rres.resp == AxiResp.OKAY and got == exp, f"got=0x{{got:X}} exp=0x{{exp:X}}")
        elif acc in ("RO", "RSVD"):
            await m.write(r["offset"], (0xFFFFFFFF & {ax.data_mask}).to_bytes({ax.strb_width}, "little"))
            rres = await m.read(r["offset"], {ax.strb_width})
            got = int.from_bytes(rres.data, "little") & r["read_mask"]
            exp = r["reset_value"] & r["read_mask"]
            _rec(r["rule_id"] + "_RO", "ro_immutable", rres.resp == AxiResp.OKAY and got == exp, f"got=0x{{got:X}} exp=0x{{exp:X}}")

    # Phase 3: illegal address checks
    for ill in ILLS:
        res = await m.read(ill["offset"], {ax.strb_width})
        _rec(ill["rule_id"], "illegal_addr", res.resp.name == ill["expected_resp"], f"got={{res.resp.name}} exp={{ill['expected_resp']}}")

    # Phase 4: stress iterations with optional random idle gaps
    rw_regs = [r for r in REGS if r["access"] == "RW" and r["write_mask"]]
    for it in range(min(STRESS["iterations"], 64) if rw_regs else 0):
        r = rng.choice(rw_regs)
        pat = rng.getrandbits({ax.data_width}) & r["write_mask"]
        await m.write(r["offset"], pat.to_bytes({ax.strb_width}, "little"))
        if STRESS["random_backpressure"]:
            for _ in range(rng.randint(0, 2)):
                await RisingEdge(dut.{b.clock.port})
        rres = await m.read(r["offset"], {ax.strb_width})
        got = int.from_bytes(rres.data, "little") & r["write_mask"] & r["read_mask"]
        if got != (pat & r["read_mask"]):
            _rec(f"STRESS_{{it}}", "stress_rw", False, f"reg={{r['name']}} got=0x{{got:X}} exp=0x{{pat & r['read_mask']:X}}")
            break
    else:
        if rw_regs and STRESS["iterations"] > 0:
            _rec("STRESS_SUMMARY", "stress_rw", True, f"completed={{min(STRESS['iterations'], 64)}}")

    # Phase 5: drain & finalize
    for _ in range(3):
        await RisingEdge(dut.{b.clock.port})
'''


def tool_validate_contract(spec_path: str, rtl_file: Optional[str] = None) -> Dict[str, Any]:
    sp = Path(spec_path)
    if not sp.is_file():
        return err_result("validate_contract", f"Spec not found: {spec_path}")
    unit, diags = build_and_validate_ir(load_raw_spec(sp), Path(rtl_file) if rtl_file else None)
    dd = [asdict(d) for d in diags]
    if unit is None:
        return err_result("validate_contract", "Semantic validation failed.", diagnostics=dd)
    return ok_result(
        "validate_contract",
        {
            "module": unit.boundary.module_name,
            "rtl_cross_checked": bool(rtl_file),
            "has_apb": unit.boundary.apb is not None,
            "register_count": len(unit.test_plan.register_rules),
        },
        diagnostics=dd,
    )


def tool_emit_testbench(spec_path: str, out_dir: str, rtl_file: Optional[str] = None) -> Dict[str, Any]:
    sp = Path(spec_path)
    if not sp.is_file():
        return err_result("emit_testbench", f"Spec not found: {spec_path}")
    unit, diags = build_and_validate_ir(load_raw_spec(sp), Path(rtl_file) if rtl_file else None)
    dd = [asdict(d) for d in diags]
    # 存在 ERROR 级诊断（如 E001 缺必需 port_map）时必须提前失败。
    # 否则会带着残缺的 IR 往下 emit，最终在生成 wrapper 时抛 KeyError，
    # 而报错只有 'awaddr' 这种单词、看不出根因（实测踩过）。
    hard_errors = [d for d in dd if str(d.get("severity", "")).upper() == "ERROR"]
    if hard_errors:
        summary = "; ".join(f"{d.get('code')}: {d.get('message')}" for d in hard_errors[:5])
        return err_result("emit_testbench",
                          f"Spec has {len(hard_errors)} error(s): {summary}",
                          diagnostics=dd)
    if unit is None:
        return err_result("emit_testbench", "Compilation failed.", diagnostics=dd)

    # emit_testbench 只生成 AXI4-Lite wrapper。若 spec 没有 AXI4-Lite boundary
    # （如纯 APB 寄存器文件），port_map 会是空 dict，往下 emit 必然 KeyError。
    # 这里给出明确错误而非让 'awaddr' 这种单词报错（实测踩过）。
    if not unit.boundary.axil.port_map:
        return err_result(
            "emit_testbench",
            "Spec has no AXI4-Lite boundary (boundary.axi4_lite.port_map is empty). "
            "emit_testbench generates an AXI4-Lite wrapper; provide an axi4_lite "
            "boundary, or generate one via `extract_interfaces --emit-draft-yaml`.",
            diagnostics=dd,
        )

    od = Path(out_dir)
    od.mkdir(parents=True, exist_ok=True)
    mod = unit.boundary.module_name
    artifacts = {
        "unit_isolation_wrapper": str((od / f"{mod}_axil_wrap.sv").resolve()),
        "sva_protocol_checker": str((od / f"{mod}_axil_sva_checker.sv").resolve()),
        "in_situ_bind_file": str((od / f"{mod}_axil_bind.sv").resolve()),
        "cocotb_testbench": str((od / f"test_{mod}_axil.py").resolve()),
    }
    Path(artifacts["unit_isolation_wrapper"]).write_text(_emit_wrap(unit), encoding="utf-8")
    Path(artifacts["sva_protocol_checker"]).write_text(_emit_sva(unit), encoding="utf-8")
    Path(artifacts["in_situ_bind_file"]).write_text(_emit_bind(unit), encoding="utf-8")
    Path(artifacts["cocotb_testbench"]).write_text(_emit_cocotb(unit), encoding="utf-8")
    return ok_result("emit_testbench", {"module": mod, "output_dir": str(od.resolve()), "artifacts": artifacts}, diagnostics=dd)


if __name__ == "__main__":
    import sys

    from .cli import run_single_tool

    # 传了 --out-dir 才生成测试台，否则只校验契约
    pick = "emit_testbench" if any(a.startswith("--out-dir") or a.startswith("--out_dir") for a in sys.argv[1:]) else "validate_contract"
    sys.exit(run_single_tool(pick))
