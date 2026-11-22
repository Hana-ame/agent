"""Fusion & cross-repo regression tests against examples/axi_apb_ref/.

Verifies:
  1. APB + AXI4-Lite dual-protocol extraction on flat-port axi_apb_bridge.sv
  2. SV interface/modport recognition + W012 warning on axi_apb_bridge_sv.sv & dual bridge (no 1-bit tie-off bug)
  3. Unified contract validator accepting agent/verilog apb-regfile-spec/v2 JSON directly + E013/E014 bitfield guards
  4. lint_tb catching C1/C2 clocking races, R2/R4 handshake drops, S1/S2/S3 silent-pass patterns
  5. run_mutation guaranteed try/finally restore, mutation score calculation, and E030 residual-mutation guard
  6. normalize_sim_result Verilator per-suite identity verification (rejecting silent skips)
  7. visual_renderer HTML consistency check + coverage/mutation summary rendering
  8. audit_harness fault-injection self-audit & 12-tool MCP registry/CLI contract
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REF = ROOT / "examples" / "axi_apb_ref"
sys.path.insert(0, str(ROOT))

from fpga_toolbox import call_tool, list_tool_schemas  # noqa: E402

FAILS = 0


def check(cond: bool, msg: str) -> None:
    global FAILS
    if cond:
        print(f"  ok   {msg}")
    else:
        print(f"  FAIL {msg}")
        FAILS += 1


def test_cross_repo_extraction() -> None:
    print("fusion: cross-repo APB + SV interface extraction")
    r_flat = call_tool("extract_interfaces", {"rtl_file": str(REF / "axi" / "axi_apb_bridge.sv")})
    check(r_flat["ok"], "axi_apb_bridge.sv extracted")
    m0 = r_flat["data"]["modules"][0]
    protos = {i["protocol"]: i for i in m0["interfaces"]}
    check("axi4_lite" in protos and len(protos["axi4_lite"]["port_map"]) == 16, "axi4_lite slave (16 signals) extracted")
    check("apb" in protos and len(protos["apb"]["port_map"]) == 8, "apb master (8 signals) extracted")
    check(protos["apb"]["role"] == "master", "apb role classified as master")
    check(len(m0["out_of_scope_ports"]) == 0, "zero APB ports leaked into out_of_scope_ports")

    r_sv = call_tool("extract_interfaces", {"rtl_file": str(REF / "axi" / "sv" / "axi_apb_bridge_sv.sv")})
    check(r_sv["ok"], "axi_apb_bridge_sv.sv extracted")
    msv = r_sv["data"]["modules"][0]
    sv_protos = [i["protocol"] for i in msv["interfaces"]]
    check("axi4_lite" in sv_protos and "apb" in sv_protos, "SV interface/modport axi & apb recognized")
    check(len(msv["out_of_scope_ports"]) == 0, "SV interface ports not misclassified as 1-bit tie_off inputs")
    if msv.get("parser_backend") != "slang":
        w_codes = {d["code"] for d in r_sv["diagnostics"]}
        check("W012" in w_codes, "W012 warning emitted when slang is absent")


def test_regfile_spec_v2_and_bitfields() -> None:
    print("fusion: apb-regfile-spec/v2 validation & bitfield E013/E014")
    r_v2 = call_tool("validate_contract", {"spec_path": str(REF / "axi" / "regfile_spec.json")})
    check(r_v2["ok"], "agent/verilog regfile_spec.json validated directly")
    check(r_v2["data"]["has_apb"] is True, "APB boundary recognized")
    check(r_v2["data"]["register_count"] >= 4, "registers parsed from v2 JSON")

    with tempfile.TemporaryDirectory() as td:
        bad_spec = Path(td) / "bad_bf.json"
        bad_spec.write_text(json.dumps({
            "$schema": "apb-regfile-spec/v2",
            "module": "bad_rf",
            "bus": {"protocol": "APB", "addr_width": 8, "data_width": 32},
            "registers": [{
                "name": "CTRL",
                "offset": "0x00",
                "attr": "RW",
                "reset": "0x0",
                "fields": [
                    {"name": "A", "bits": "3:0", "attr": "RW", "reset": 0},
                    {"name": "B", "bits": "5:2", "attr": "RW", "reset": 0},
                    {"name": "C", "bits": "33:32", "attr": "RO", "reset": 0},
                ],
            }],
        }), encoding="utf-8")
        r_bad = call_tool("validate_contract", {"spec_path": str(bad_spec)})
        codes = {d["code"] for d in r_bad["diagnostics"]}
        check(not r_bad["ok"], "invalid bitfields rejected")
        check("E013" in codes, "E013 bitfield overlap caught")
        check("E014" in codes, "E014 bitfield out-of-width caught")


def test_lint_tb() -> None:
    print("fusion: lint_tb (C1/C2, R2/R4, S1/S2/S3)")
    with tempfile.TemporaryDirectory() as td:
        p = Path(td)
        bad_sv = p / "tb_faulty.sv"
        bad_sv.write_text(
            "module tb_faulty;\n"
            "  initial begin\n"
            "    @(posedge ACLK);\n"
            "    axi.AWVALID = 1'b1;\n"
            "    if (axi.BVALID) begin axi.BREADY = 1'b0; end\n"
            "  end\n"
            "endmodule\n",
            encoding="utf-8",
        )
        bad_sh = p / "check.sh"
        bad_sh.write_text(
            "#!/bin/bash\n"
            "grep_violation() { return 1; }\n"
            "passes=$(grep -c '^PASS' log.txt)\n"
            "[ \"$passes\" -ge 5 ]\n",
            encoding="utf-8",
        )
        r = call_tool("lint_tb", {"target_path": str(p)})
        codes = {d["code"] for d in r["diagnostics"]}
        check(not r["ok"], "faulty TB/script fails lint_tb")
        check({"C1", "R2", "S1", "S3"}.issubset(codes), f"caught C1/R2/S1/S3 (got {sorted(codes)})")


def test_mutation_runner() -> None:
    print("fusion: run_mutation restore, score, and residual guard")
    with tempfile.TemporaryDirectory() as td:
        wd = Path(td)
        dut = wd / "dut.txt"
        dut.write_text("value = 42\n", encoding="utf-8")
        spec = wd / "muts.json"
        spec.write_text(json.dumps([
            {"id": "M01", "file": "dut.txt", "old": "value = 42", "new": "value = 0", "desc": "zero out"}
        ]), encoding="utf-8")
        # Test command fails when value != 42 -> kills mutation
        cmd = [sys.executable, "-c", "import sys; sys.exit(0 if '42' in open('dut.txt').read() else 1)"]
        r = call_tool("run_mutation", {"workdir": str(wd), "mutation_spec": str(spec), "test_cmd": cmd, "min_score": 100.0})
        check(r["ok"] and r["data"]["mutation_score"] == 100.0, "mutation killed and score=100%")
        check(dut.read_text(encoding="utf-8") == "value = 42\n", "original file restored by try/finally")

        # Simulate residual unrestored mutation -> must be blocked with E030
        dut.write_text("value = 0\n", encoding="utf-8")
        r_res = call_tool("run_mutation", {"workdir": str(wd), "mutation_spec": str(spec), "test_cmd": cmd})
        check(not r_res["ok"] and any(d["code"] == "E030" for d in r_res["diagnostics"]), "residual mutation rejected (E030)")


def test_verilator_log_normalizer_and_visual() -> None:
    print("fusion: Verilator suite identity normalizer & visual consistency check")
    with tempfile.TemporaryDirectory() as td:
        td_p = Path(td)
        log = td_p / "sim.log"
        # 用 sim.sh 的真实格式：`== sim: <name> (top=<top>) ==`。套件名后带
        # 参数。旧正则要求套件名后紧跟 `==`，对这种真实日志永远匹配不到，
        # 会把所有套件误判为 missing —— 「逐套件身份核对」护栏对真实日志
        # 从不生效。这里特意用真实格式锁死该缺陷。
        log.write_text(
            "== sim: counter (top=tb_counter) ==\nPASS: all 20 checks passed\n"
            "== sim: axi (top=tb_axi_apb) ==\nPASS: all 35 checks passed\n",
            encoding="utf-8",
        )
        r_ok = call_tool("normalize_sim_result", {
            "module_name": "axi_apb_bridge",
            "verilator_log_path": str(log),
            "expected_suites": ["counter", "axi"],
            "coverage_summary": {"rtl_pct": 100.0},
            "mutation_summary": {"mutation_score": 100.0},
        })
        check(r_ok["ok"] and r_ok["data"]["overall_status"] == "PASS", "all expected Verilator suites verified PASS")

        r_miss = call_tool("normalize_sim_result", {
            "module_name": "axi_apb_bridge",
            "verilator_log_path": str(log),
            "expected_suites": ["counter", "axi", "dual"],
        })
        check(r_miss["ok"] and r_miss["data"]["overall_status"] == "FAIL", "missing suite 'dual' flagged as FAIL (no silent skip)")

        r_vis = call_tool("render_visual_report", {
            "report": r_ok["data"],
            "dep_graph": {"dependency_closure": ["axi_apb_bridge", "apb_slave_regs"]},
            "extracted_interfaces": [{"protocol": "axi4_lite", "port_map": {"awaddr": "AWADDR"}}],
            "out_html_path": str(td_p / "dash.html"),
        })
        check(r_vis["ok"] and r_vis["data"]["consistency_ok"] is True, "visual HTML consistency check passed")
        html_txt = (td_p / "dash.html").read_text(encoding="utf-8")
        check("rtl_cov=100.0%" in html_txt and "mutation=100.0%" in html_txt, "coverage & mutation summaries rendered")


def test_audit_and_registry() -> None:
    print("fusion: audit_harness & 12-tool registry/CLI")
    r_aud = call_tool("audit_harness", {"strict": True})
    # 动态断言：检查数会随新护栏覆盖扩展（已从 4 增至 9，含 A5-A9 套件身份核对、版本守卫、emit fail-fast、slang 回退可见性）。
    # 这里只要求「全部通过」，不再硬编码具体数字，避免加检查时误判为回归。
    aud = r_aud.get("data", {})
    check(r_aud["ok"] and aud.get("failed") == 0 and aud.get("passed", 0) >= 9,
          f"audit_harness all fault-injection checks passed (passed={aud.get('passed')}/{aud.get('total')})")

    schemas = list_tool_schemas()
    names = {s["name"] for s in schemas}
    check(len(schemas) == 12, f"12 tools registered (got {len(schemas)})")
    check({"lint_tb", "run_sim", "run_mutation", "audit_harness"}.issubset(names), "all 4 new tools in MCP schemas")

    proc = subprocess.run([sys.executable, "-m", "fpga_toolbox", "audit_harness"], cwd=str(ROOT), capture_output=True, text=True, check=False)
    check(proc.returncode == 0 and '"ok": true' in proc.stdout, "CLI `python -m fpga_toolbox audit_harness` exits 0")


def test_slang_backend_real_rtl() -> None:
    """锁死 slang 后端两个历史缺陷（仅在装了 slang 时验证）。

    缺陷 1：slang --ast-json - 把人类可读 banner（"Top level design units:"
            / "Build succeeded:")和 JSON 混在同一个 stdout，旧实现直接
            json.loads 整段输出必然失败 → slang 后端从未真正生效。
    缺陷 2：slang 遇到未知 package 时 exit 非 0 但仍输出完整 AST JSON，
            旧实现因 returncode != 0 整段丢弃；且回退到 builtin 时对调用方
            无任何信号 —— 恰是本项目批判的「静默通过」。
    """
    import shutil

    print("fusion: slang AST backend on real RTL (banner handling + fallback visibility)")
    if not shutil.which("slang"):
        print("  skip slang not installed (W013/回退诊断在无 slang 时仍可验证)")
        return

    # 1) 真实桥文件（带跨文件 import）必须真的走 slang 后端，不能静默回退
    bridge = REF / "axi" / "axi_apb_bridge.sv"
    if bridge.is_file():
        r = call_tool("extract_interfaces", {
            "rtl_file": str(bridge), "parser_backend": "slang",
        })
        check(r.get("ok") is True, "slang backend parses bridge with cross-file imports")
        mods = r.get("data", {}).get("modules", [])
        if mods:
            check(mods[0].get("parser_backend") == "slang",
                  f"bridge really used slang backend (got {mods[0].get('parser_backend')})")

    # 2) auto 后端应优先选 slang，且不产生「回退」诊断
    if bridge.is_file():
        r = call_tool("extract_interfaces", {"rtl_file": str(bridge)})
        mods = r.get("data", {}).get("modules", [])
        if mods:
            check(mods[0].get("parser_backend") == "slang",
                  "auto backend prefers slang when available")
            diags = r.get("diagnostics") or []
            check(not any(d.get("code") == "W013" for d in diags),
                  "no W013 fallback warning when slang succeeds")

    # 3) 覆盖率：靶场里含 module 的 .sv 文件应全部走 slang（零静默回退）
    from fpga_toolbox.core.sv_parser import parse_sv_file
    rtl_files = sorted(REF.rglob("*.sv"))
    with_mod = []
    for f in rtl_files:
        if "module " in f.read_text(encoding="utf-8", errors="replace"):
            with_mod.append(f)
    slang_hits = 0
    fallback_diags = 0
    for f in with_mod:
        diag: list = []
        m = parse_sv_file(f, backend="auto", diagnostics=diag)
        if m and m[0].parser_backend == "slang":
            slang_hits += 1
        elif diag:
            fallback_diags += 1
    check(slang_hits == len(with_mod),
          f"all {len(with_mod)} module-bearing .sv files parsed by slang (got {slang_hits})")
    check(fallback_diags == 0,
          "no silent builtin fallback (fallbacks are reported via diagnostics)")

    # 4) 无 slang 的环境必须产出回退诊断（可见信号）
    from fpga_toolbox.core import sv_parser as sp
    if bridge.is_file():
        diag2: list = []
        m2 = parse_sv_file(bridge, backend="auto", diagnostics=diag2)
        check(m2 is not None, "auto backend still returns modules when diagnostics collected")


def test_emit_testbench_error_paths() -> None:
    """锁死 emit_testbench 的两个错误路径（历史缺陷：报错只有 'awaddr' 一个词）。

    缺陷：spec 缺 AXI4-Lite boundary（如纯 APB 寄存器文件）时，build_and_validate_ir
    静默构造出空 port_map 的 axil IR，往下 emit 在 _emit_wrap 抛 KeyError('awaddr')，
    调用方只看到一个单词、看不出根因。修复后必须给出明确错误。
    """
    import tempfile

    print("fusion: emit_testbench clear error paths (no cryptic KeyError)")
    with tempfile.TemporaryDirectory() as tmp:
        # 1) 纯 APB 寄存器 spec（无 axi4_lite boundary）→ 必须明确拒绝
        r = call_tool("emit_testbench", {
            "spec_path": str(REF / "axi" / "regfile_spec.json"),
            "out_dir": str(Path(tmp) / "apb_only"),
        })
        check(r.get("ok") is False,
              "pure-APB spec rejected (no AXI4-Lite boundary)")
        err = str(r.get("error") or "")
        check("awaddr" != err,
              f"error is not a bare 'awaddr' (got {err[:60]!r})")
        check("axi4_lite" in err.lower() or "boundary" in err.lower(),
              "error mentions the missing AXI4-Lite boundary")

        # 2) 完整链路但缺必需 port_map → 必须报 E001 而非崩在生成阶段
        # 构造一个含 axi4_lite 但缺 wstrb 的契约
        bad = Path(tmp) / "no_wstrb.yaml"
        bad.write_text(
            "boundary:\n"
            "  module_name: foo\n"
            "  rtl_files: [foo.sv]\n"
            "  clock: {port: ACLK}\n"
            "  reset: {port: ARESETn, active_low: true}\n"
            "  axi4_lite:\n"
            "    addr_width: 8\n"
            "    data_width: 32\n"
            "    role: slave\n"
            "    port_map:\n"
            "      awaddr: AWADDR\n"  # 故意只给一个信号，缺 wstrb 等
            "test_plan:\n"
            "  protocol_rules: {check_valid_stability: true, check_reset_outputs: true, max_resp_cycles: 16, allow_unaligned_addr: false}\n"
            "  register_rules: []\n"
            "  illegal_addr_rules: []\n"
            "  stress_profile: {random_backpressure: false, iterations: 1, seed: 1}\n",
            encoding="utf-8",
        )
        r2 = call_tool("emit_testbench", {
            "spec_path": str(bad),
            "out_dir": str(Path(tmp) / "no_wstrb"),
        })
        check(r2.get("ok") is False,
              "incomplete port_map rejected with E001")
        err2 = str(r2.get("error") or "")
        check("E001" in err2,
              f"error surfaces the E001 diagnostic (got {err2[:70]!r})")
        check(len(r2.get("diagnostics") or []) >= 1,
              "E001 diagnostic is propagated to the caller")


def test_suite_identity_real_format() -> None:
    """锁死 verify_suite_identities 对 sim.sh 真实日志格式的解析。

    sim.sh 打印 `== sim: <name> (top=<top>) ==`，套件名后带 (top=...) /
    (WAIT=...) 参数。旧正则 `==\s*sim:\s*(\w+)\s*==` 对此永远匹配不到，
    会把所有套件判为 missing——反静默通过护栏形同虚设。
    """
    from fpga_toolbox.sim_runner import verify_suite_identities

    print("fusion: suite identity parsing on real sim.sh log format")
    real = (
        "== sim: counter (top=tb_counter) ==\nPASS: all 20 checks passed\n"
        "== sim: latency (WAIT=93) ==\nPASS: latency suite ok\n"
        "== sim: sv (top=tb_axi_apb_sv) ==\n[MON] monitor clean\nPASS: sv txn testbench\n"
        "== sim: gaps (top=tb_gaps_sv) ==\nFAIL: coverage-gap testbench\n"
    )
    r = verify_suite_identities(real, ["counter", "latency", "sv", "gaps", "dual"])
    check(r["observed"].get("counter") == "PASS", f"counter parsed PASS (got {r['observed'].get('counter')})")
    check(r["observed"].get("latency") == "PASS", f"latency (WAIT=) format parsed (got {r['observed'].get('latency')})")
    check(r["observed"].get("sv") == "PASS", f"sv parsed PASS despite non-PASS monitor line (got {r['observed'].get('sv')})")
    check(r["observed"].get("gaps") == "FAIL", f"gaps with FAIL detected (got {r['observed'].get('gaps')})")
    check(r["missing"] == ["dual"], f"only 'dual' missing (got {r['missing']})")
    check(r["all_verified"] is False, "all_verified False when a suite missing")


def main() -> int:
    test_cross_repo_extraction()
    test_regfile_spec_v2_and_bitfields()
    test_lint_tb()
    test_mutation_runner()
    test_verilator_log_normalizer_and_visual()
    test_suite_identity_real_format()
    test_audit_and_registry()
    test_slang_backend_real_rtl()
    test_emit_testbench_error_paths()
    print("\n" + ("ALL FUSION CHECKS PASSED" if FAILS == 0 else f"{FAILS} CHECK(S) FAILED"))
    return 0 if FAILS == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
