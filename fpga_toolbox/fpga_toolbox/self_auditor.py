"""Tool 12: Self-audit tool verifying that verification guardrails fail loudly (no silent passes).

Ported in spirit from agent/verilog/audit.sh: injects synthetic defects into isolated
temp fixtures and confirms that lint_tb, run_mutation (residual guard), normalize_sim_result,
and copy_sandbox all reject invalid/broken inputs with ok=False.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any, Dict, List

if __package__ in (None, ""):
    import sys as _sys
    from pathlib import Path as _Path

    _sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
    __package__ = "fpga_toolbox"

from .core.types import err_result, ok_result
from .mutation_runner import tool_run_mutation
from .result_normalizer import tool_normalize_sim_result
from .sandbox_copier import tool_copy_sandbox
from .sim_runner import verify_suite_identities, tool_run_sim
from .tb_emitter import tool_emit_testbench
from .core import sv_parser
from .tb_linter import tool_lint_tb


def tool_audit_harness(strict: bool = True) -> Dict[str, Any]:
    checks: List[Dict[str, Any]] = []
    with tempfile.TemporaryDirectory(prefix="fpga_audit_") as td:
        root = Path(td)

        # A1: lint_tb must catch C1 clocking race & R2 handshake drop
        bad_tb = root / "tb_bad.sv"
        bad_tb.write_text(
            "module tb_bad;\n"
            "  initial begin\n"
            "    @(posedge ACLK);\n"
            "    axi.AWVALID = 1'b1;\n"
            "    if (axi.BVALID) begin axi.BREADY = 1'b0; end\n"
            "  end\n"
            "endmodule\n",
            encoding="utf-8",
        )
        r_lint = tool_lint_tb(str(bad_tb))
        codes = {d.get("code") for d in r_lint.get("diagnostics", [])}
        a1_ok = (not r_lint["ok"]) and ("C1" in codes) and ("R2" in codes)
        checks.append({"id": "A1_LINT_FAULT_CATCH", "passed": a1_ok, "detail": f"ok={r_lint['ok']} codes={sorted(codes)}"})

        # A2: run_mutation must refuse residual unrestored mutation (E030)
        mut_dir = root / "mut_proj"
        mut_dir.mkdir()
        (mut_dir / "dut.sv").write_text("assign y = a + b;\n", encoding="utf-8")
        spec = root / "muts.json"
        spec.write_text('[{"id":"M01","file":"dut.sv","old":"a - b","new":"a + b","desc":"residual"}]', encoding="utf-8")
        r_mut = tool_run_mutation(str(mut_dir), str(spec), ["python3", "-c", "pass"])
        m_codes = {d.get("code") for d in r_mut.get("diagnostics", [])}
        a2_ok = (not r_mut["ok"]) and ("E030" in m_codes)
        checks.append({"id": "A2_MUTATION_RESIDUAL_GUARD", "passed": a2_ok, "detail": f"ok={r_mut['ok']} codes={sorted(m_codes)}"})

        # A3: normalize_sim_result must reject unrecognized JSON instead of silent PASS
        junk = root / "junk.json"
        junk.write_text('{"foo": "bar"}', encoding="utf-8")
        r_norm = tool_normalize_sim_result("audit_mod", cocotb_results_json=str(junk))
        a3_ok = not r_norm["ok"]
        checks.append({"id": "A3_NORMALIZER_REJECT_JUNK", "passed": a3_ok, "detail": f"ok={r_norm['ok']}"})

        # A4: copy_sandbox must refuse targeting project root
        proj = root / "proj"
        proj.mkdir()
        (proj / "a.sv").write_text("module a; endmodule\n", encoding="utf-8")
        r_sb = tool_copy_sandbox(str(proj), ["a.sv"], str(proj))
        a4_ok = not r_sb["ok"]
        checks.append({"id": "A4_SANDBOX_ROOT_GUARD", "passed": a4_ok, "detail": f"ok={r_sb['ok']}"})

        # A5: verify_suite_identities 必须解析 sim.sh 真实格式日志，且漏跑被精确抓出。
        # 这是 run_sim / normalize 的守门逻辑。历史缺陷：正则要求套件名后紧跟 `==`，
        # 而真实日志是 `== sim: <name> (top=<top>) ==`，导致所有套件被误判 missing，
        # 反静默通过护栏形同虚设。此检查锁死：真实格式能解析、漏跑能识别。
        real_log = (
            "== sim: counter (top=tb_counter) ==\nPASS: all 20 checks passed\n"
            "== sim: sv (top=tb_axi_apb_sv) ==\nPASS: sv testbench\n"
        )
        a5 = verify_suite_identities(real_log, ["counter", "sv", "gaps"])
        a5_ok = (
            a5["observed"].get("counter") == "PASS"
            and a5["observed"].get("sv") == "PASS"
            and a5["missing"] == ["gaps"]
            and a5["all_verified"] is False
        )
        checks.append({
            "id": "A5_SUITE_IDENTITY_REAL_FORMAT",
            "passed": a5_ok,
            "detail": f"observed={a5['observed']} missing={a5['missing']}",
        })

        # A6: 全部预期套件齐且都 PASS 时必须 all_verified=True（不能假报警）
        a6 = verify_suite_identities(real_log, ["counter", "sv"])
        a6_ok = a6["all_verified"] is True and a6["missing"] == [] and a6["failed"] == []
        checks.append({
            "id": "A6_SUITE_IDENTITY_ALL_PASS",
            "passed": a6_ok,
            "detail": f"all_verified={a6['all_verified']} missing={a6['missing']}",
        })

        # A7: run_sim 必须在 Verilator 版本不达标时显式拒绝（E020），不能静默跑。
        # 伪造一个 meets_min_version=False 的 EDA 环境，确认 E020 守卫生效。
        import fpga_toolbox.sim_runner as _sr
        _orig_probe = _sr.probe_eda_env
        r_sim = None
        e020 = False

        def _fake_env():
            return {
                "verilator": {"path": "/nonexistent", "version": "5.020",
                              "meets_min_version": False, "message": "Verilator 5.020 < 5.040"},
                "slang": {"path": None, "available": False},
                "stdbuf": {"path": "/usr/bin/stdbuf", "available": True},
            }

        try:
            _sr.probe_eda_env = _fake_env
            _sim_dir = root / "simproj"
            _sim_dir.mkdir()
            _sim_script = _sim_dir / "sim.sh"
            _sim_script.write_text("#!/bin/bash\necho dummy\n", encoding="utf-8")
            r_sim = tool_run_sim(str(_sim_script), suite="counter", timeout_sec=10)
            e020 = any(d.get("code") == "E020" for d in (r_sim.get("diagnostics") or []))
            a7_ok = (not r_sim["ok"]) and e020
        finally:
            _sr.probe_eda_env = _orig_probe
        checks.append({
            "id": "A7_RUN_SIM_VERSION_GUARD",
            "passed": a7_ok,
            "detail": f"ok={r_sim['ok']} e020={e020}",
        })

        # A8: emit_testbench 在 spec 缺 AXI4-Lite boundary 时必须 fail-fast，
        # 而不是带着空 port_map 往下 emit 到 KeyError('awaddr')（历史缺陷）。
        apb_only = root / "apb_only.json"
        apb_only.write_text(
            '{"$schema":"apb-regfile-spec/v2","module":"m","bus":{"protocol":"APB",'
            '"addr_width":8,"data_width":32,"wait_cycles":2},"registers":[]}',
            encoding="utf-8",
        )
        r_emit = tool_emit_testbench(str(apb_only), str(root / "tb_out"))
        a8_ok = (not r_emit["ok"]) and ("awaddr" not in str(r_emit.get("error") or "")) \
            and ("axi4_lite" in str(r_emit.get("error") or "").lower()
                 or "boundary" in str(r_emit.get("error") or "").lower())
        checks.append({
            "id": "A8_EMIT_FAIL_FAST",
            "passed": a8_ok,
            "detail": f"ok={r_emit['ok']} error={str(r_emit.get('error') or '')[:60]}",
        })

        # A9: slang 不可用（或解析失败）时必须对调用方可见（W013 诊断），
        # 而不是静默退回 builtin——否则「可选 slang AST 后端」就是虚报。
        # 伪造 slang 不在 PATH，确认 builtin 回退会产出 W013。
        import fpga_toolbox.core.sv_parser as _sp
        _orig_which = _sp.shutil.which
        try:
            _sp.shutil.which = lambda name: None if name == "slang" else _orig_which(name)
            _rtl = root / "probe.sv"
            _rtl.write_text("module probe(input logic a, output logic y);\nassign y = a;\nendmodule\n",
                            encoding="utf-8")
            diags: list = []
            _sp.parse_sv_file(_rtl, backend="auto", diagnostics=diags)
            a9_ok = any("slang" in d for d in diags)
        finally:
            _sp.shutil.which = _orig_which
        checks.append({
            "id": "A9_SLANG_FALLBACK_VISIBLE",
            "passed": a9_ok,
            "detail": f"slang-missing diagnostics={len(diags)} visible={a9_ok}",
        })

    failed = [c for c in checks if not c["passed"]]
    data = {"total": len(checks), "passed": len(checks) - len(failed), "failed": len(failed), "checks": checks}
    if failed and strict:
        return err_result(
            "audit_harness",
            f"Self-audit failed on {len(failed)} check(s): {[c['id'] for c in failed]}",
            diagnostics=[{"severity": "ERROR", "code": "E040", "path": c["id"], "message": c["detail"]} for c in failed],
        )
    return ok_result("audit_harness", data)


if __name__ == "__main__":
    import sys

    from .cli import run_single_tool

    sys.exit(run_single_tool("audit_harness"))
