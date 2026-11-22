"""Dependency-free regression suite for fpga_toolbox.

Run from the repo root:
    python -m tests.test_regression        (or)  python tests/test_regression.py

Covers the failure modes found during the first audit:
  * copy_sandbox wiping the project when sandbox_dir == project_root
  * copy_sandbox flattening same-basename files
  * copy_sandbox cleaning an unrelated non-empty directory
  * aresetn / s_axi_aresetn not recognised as reset -> duplicated port connection
  * normalize_sim_result silently returning NO_RESULTS on unreadable input
"""

from __future__ import annotations

import json
import re
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fpga_toolbox import call_tool  # noqa: E402

RTL = """`include "defs.svh"
module reg_slave #(parameter int ADDR_W = 12, DATA_W = 32) (
    input  logic              {clk},
    input  logic              {rst},
    input  logic [ADDR_W-1:0] s_axi_awaddr,
    input  logic [2:0]        s_axi_awprot,
    input  logic              s_axi_awvalid,
    output logic              s_axi_awready,
    input  logic [DATA_W-1:0] s_axi_wdata,
    input  logic [3:0]        s_axi_wstrb,
    input  logic              s_axi_wvalid,
    output logic              s_axi_wready,
    output logic [1:0]        s_axi_bresp,
    output logic              s_axi_bvalid,
    input  logic              s_axi_bready,
    input  logic [ADDR_W-1:0] s_axi_araddr,
    input  logic [2:0]        s_axi_arprot,
    input  logic              s_axi_arvalid,
    output logic              s_axi_arready,
    output logic [DATA_W-1:0] s_axi_rdata,
    output logic [1:0]        s_axi_rresp,
    output logic              s_axi_rvalid,
    input  logic              s_axi_rready,
    output logic              irq_o
);
    assign s_axi_awready = 1'b0;
    assign s_axi_wready  = 1'b0;
    assign s_axi_bresp   = 2'b00;
    assign s_axi_bvalid  = 1'b0;
    assign s_axi_arready = 1'b0;
    assign s_axi_rdata   = '0;
    assign s_axi_rresp   = 2'b00;
    assign s_axi_rvalid  = 1'b0;
    assign irq_o         = 1'b0;
endmodule
"""

FAILURES: list[str] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    if cond:
        print(f"  ok   {name}")
    else:
        print(f"  FAIL {name}  {detail}")
        FAILURES.append(name)


def make_project(root: Path, rst: str = "aresetn", clk: str = "aclk") -> Path:
    """Create a small AXI4-Lite project and return its root directory."""
    (root / "rtl").mkdir(parents=True, exist_ok=True)
    (root / "inc").mkdir(parents=True, exist_ok=True)
    (root / "rtl" / "reg_slave.sv").write_text(RTL.format(clk=clk, rst=rst), encoding="utf-8")
    (root / "inc" / "defs.svh").write_text("`define FOO 1\n", encoding="utf-8")
    return root


def test_sandbox_guards(tmp: Path) -> None:
    print("copy_sandbox: sandbox_dir must never be the project root")
    proj = make_project(tmp / "guard")
    before = sorted(p.relative_to(proj).as_posix() for p in proj.rglob("*") if p.is_file())
    res = call_tool("copy_sandbox", {
        "project_root": str(proj),
        "relative_files": ["rtl/reg_slave.sv"],
        "sandbox_dir": str(proj),
        "clean": True,
    })
    after = sorted(p.relative_to(proj).as_posix() for p in proj.rglob("*") if p.is_file())
    check("rejected", res["ok"] is False, res.get("error") or "")
    check("project intact", before == after, f"{before} != {after}")

    print("copy_sandbox: ancestor of the project root is rejected")
    res = call_tool("copy_sandbox", {
        "project_root": str(proj),
        "relative_files": ["rtl/reg_slave.sv"],
        "sandbox_dir": str(tmp),
        "clean": True,
    })
    check("ancestor rejected", res["ok"] is False, res.get("error") or "")
    check("ancestor kept files", (proj / "rtl" / "reg_slave.sv").is_file())


def test_sandbox_layout(tmp: Path) -> None:
    print("copy_sandbox: same basename in different dirs must not collide")
    proj = tmp / "layout"
    (proj / "rtl" / "a").mkdir(parents=True)
    (proj / "rtl" / "b").mkdir(parents=True)
    (proj / "rtl" / "a" / "common.sv").write_text("module A; endmodule\n", encoding="utf-8")
    (proj / "rtl" / "b" / "common.sv").write_text("module B; endmodule\n", encoding="utf-8")
    sb = tmp / "layout_sb"
    res = call_tool("copy_sandbox", {
        "project_root": str(proj),
        "relative_files": ["rtl/a/common.sv", "rtl/b/common.sv"],
        "sandbox_dir": str(sb),
    })
    files = sorted(p.relative_to(sb).as_posix() for p in sb.rglob("*.sv"))
    check("tool ok", res["ok"] is True, res.get("error") or "")
    check("both files staged", files == ["rtl/a/common.sv", "rtl/b/common.sv"], str(files))
    check("contents distinct",
          "module A" in (sb / "rtl/a/common.sv").read_text(encoding="utf-8")
          and "module B" in (sb / "rtl/b/common.sv").read_text(encoding="utf-8"))

    print("copy_sandbox: non-empty unmarked directory is not wiped")
    foreign = tmp / "foreign"
    foreign.mkdir()
    (foreign / "keep.txt").write_text("important\n", encoding="utf-8")
    res = call_tool("copy_sandbox", {
        "project_root": str(proj),
        "relative_files": ["rtl/a/common.sv"],
        "sandbox_dir": str(foreign),
        "clean": True,
    })
    check("refused unmarked clean", res["ok"] is False, res.get("error") or "")
    check("foreign file survived", (foreign / "keep.txt").is_file())


def test_reset_detection(tmp: Path) -> None:
    for rst_name in ("aresetn", "resetn", "rst_n", "s_axi_aresetn", "nrst", "reset"):
        print(f"extract_interfaces: reset '{rst_name}' recognised")
        proj = make_project(tmp / f"rst_{rst_name}", rst=rst_name)
        rtl = proj / "rtl" / "reg_slave.sv"
        res = call_tool("extract_interfaces", {"rtl_file": str(rtl), "module_name": "reg_slave"})
        mod = res["data"]["modules"][0]
        check(f"{rst_name} detected", (mod["reset"] or {}).get("port") == rst_name, json.dumps(mod["reset"]))
        tie_offs = [p["port"] for p in mod["out_of_scope_ports"] if p.get("tie_off")]
        check(f"{rst_name} not tied off", rst_name not in tie_offs, str(tie_offs))
        active_low = bool((mod["reset"] or {}).get("active_low"))
        expected_low = rst_name not in ("reset", "rst")
        check(f"{rst_name} polarity", active_low is expected_low, f"active_low={active_low}")

        print(f"emit_testbench: no duplicate '.{rst_name}' connection")
        draft = tmp / f"rst_{rst_name}" / "contract.yaml"
        call_tool("extract_interfaces", {
            "rtl_file": str(rtl), "module_name": "reg_slave", "emit_draft_yaml": str(draft),
        })
        out = tmp / f"rst_{rst_name}" / "tb"
        res = call_tool("emit_testbench", {
            "spec_path": str(draft), "out_dir": str(out), "rtl_file": str(rtl),
        })
        check(f"{rst_name} emitter ok", res["ok"] is True, res.get("error") or str(res.get("diagnostics")))
        if res["ok"]:
            wrap = Path(res["data"]["artifacts"]["unit_isolation_wrapper"]).read_text(encoding="utf-8")
            conns = [ln for ln in wrap.splitlines() if ln.strip().startswith(f".{rst_name}(")]
            check(f"{rst_name} connected once", len(conns) == 1, str(conns))
            py = Path(res["data"]["artifacts"]["cocotb_testbench"]).read_text(encoding="utf-8")
            m = re.search(r"reset_active_level=(\w+)", py)
            check(f"{rst_name} reset_active_level", bool(m) and m.group(1) == str(not expected_low),
                  m.group(1) if m else "missing")


def test_normalizer(tmp: Path) -> None:
    print("normalize_sim_result: unrecognized input is an error, not an empty report")
    junk = tmp / "junk.json"
    junk.write_text('{"unrelated": 1}', encoding="utf-8")
    res = call_tool("normalize_sim_result", {"module_name": "m", "cocotb_results_json": str(junk)})
    check("junk rejected", res["ok"] is False, res.get("data"))

    print("normalize_sim_result: missing file is an error")
    res = call_tool("normalize_sim_result", {"module_name": "m", "cocotb_results_json": str(tmp / "nope.json")})
    check("missing rejected", res["ok"] is False)

    print("normalize_sim_result: cocotb JUnit XML is parsed")
    xml = tmp / "results.xml"
    xml.write_text(
        '<?xml version="1.0"?><testsuites><testsuite name="reg_slave">'
        '<testcase name="axil_write_read" classname="axil"/>'
        '<testcase name="axil_illegal_addr" classname="axil"><failure message="got OKAY">trace</failure></testcase>'
        "</testsuite></testsuites>",
        encoding="utf-8",
    )
    res = call_tool("normalize_sim_result", {"module_name": "reg_slave", "cocotb_results_json": str(xml)})
    data = res["data"]
    check("xml parsed", res["ok"] is True and data["summary"]["total"] == 2, json.dumps(data.get("summary")))
    check("xml verdict", data.get("overall_status") == "FAIL", data.get("overall_status"))


def test_full_chain(tmp: Path) -> None:
    print("full chain: index -> deps -> sandbox -> contract -> tb -> report -> visual")
    proj = make_project(tmp / "chain")
    rtl = proj / "rtl" / "reg_slave.sv"

    idx = call_tool("index_repo", {"project_root": str(proj)})
    check("index ok", idx["ok"] and idx["data"]["modules"], idx.get("error") or "")

    deps = call_tool("resolve_deps", {"target_module": "reg_slave", "repo_index": idx["data"]})
    check("deps ok", deps["ok"] and deps["data"]["dependency_closure"] == ["reg_slave"], deps.get("error") or "")

    sb = tmp / "chain_sb"
    cp = call_tool("copy_sandbox", {
        "project_root": str(proj),
        "relative_files": deps["data"]["all_files"],
        "sandbox_dir": str(sb),
    })
    check("sandbox ok", cp["ok"], cp.get("error") or "")

    draft = tmp / "chain" / "contract.yaml"
    ex = call_tool("extract_interfaces", {
        "rtl_file": str(rtl), "module_name": "reg_slave", "emit_draft_yaml": str(draft),
        "staged_rtl_files": ["rtl/reg_slave.sv", "inc/defs.svh"],
    })
    check("extract ok", ex["ok"], ex.get("error") or "")

    val = call_tool("validate_contract", {"spec_path": str(draft), "rtl_file": str(rtl)})
    check("validate ok", val["ok"], str(val.get("diagnostics")))

    tb = call_tool("emit_testbench", {"spec_path": str(draft), "out_dir": str(tmp / "chain" / "tb"), "rtl_file": str(rtl)})
    check("emit ok", tb["ok"], str(tb.get("diagnostics")))

    if tb["ok"]:
        wrap = Path(tb["data"]["artifacts"]["unit_isolation_wrapper"]).read_text(encoding="utf-8")
        dup = [ln for ln in wrap.splitlines() if ln.strip().startswith(".aresetn(")]
        check("wrapper has single aresetn", len(dup) == 1, str(dup))
        import py_compile
        py_compile.compile(tb["data"]["artifacts"]["cocotb_testbench"], doraise=True)
        check("cocotb testbench compiles", True)

    norm = call_tool("normalize_sim_result", {"module_name": "reg_slave", "cocotb_results_json": str(tmp / "results.xml")})
    check("normalize ok", norm["ok"], norm.get("error") or "")

    vis = call_tool("render_visual_report", {
        "report": norm["data"] if norm["ok"] else {"module": "reg_slave", "overall_status": "NO_RESULTS"},
        "dep_graph": deps["data"] if deps["ok"] else {"dependency_closure": []},
        "extracted_interfaces": ex["data"]["modules"][0]["interfaces"] if ex["ok"] else [],
        "out_html_path": str(tmp / "chain" / "visual.html"),
    })
    check("visual ok", vis["ok"], vis.get("error") or "")
    if vis["ok"]:
        svg = Path(vis["data"]["svg_path"])
        check("svg sidecar written", svg.is_file() and "<svg" in svg.read_text(encoding="utf-8"))
        check("html written", Path(vis["data"]["html_path"]).is_file())


def test_cli(tmp: Path) -> None:
    """The CLI surface is the part users actually touch, so drive it as a subprocess."""
    import subprocess

    root = Path(__file__).resolve().parents[1]
    proj = make_project(tmp / "cli")
    rtl = proj / "rtl" / "reg_slave.sv"

    def run(*args: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [sys.executable, *args],
            cwd=str(root),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )

    print("cli: list / schemas")
    res = run("-m", "fpga_toolbox", "list")
    check("python -m fpga_toolbox list", res.returncode == 0 and "index_repo" in res.stdout, res.stderr[-300:])
    res = run("-m", "fpga_toolbox", "schemas")
    check("python -m fpga_toolbox schemas", res.returncode == 0 and '"inputSchema"' in res.stdout, res.stderr[-300:])

    print("cli: native flags for a tool")
    res = run("-m", "fpga_toolbox", "index_repo", "--project-root", str(proj))
    check("native --project-root", res.returncode == 0 and '"reg_slave"' in res.stdout, res.stdout[-300:] + res.stderr[-300:])

    print("cli: single-file and module execution")
    res = run(str(root / "fpga_toolbox" / "iface_extractor.py"),
              "--rtl-file", str(rtl), "--module-name", "reg_slave")
    check("python fpga_toolbox/iface_extractor.py",
          res.returncode == 0 and '"extract_interfaces"' in res.stdout, res.stderr[-400:])
    res = run("-m", "fpga_toolbox.repo_indexer", "--project-root", str(proj))
    check("python -m fpga_toolbox.repo_indexer",
          res.returncode == 0 and '"index_repo"' in res.stdout, res.stderr[-400:])
    check("no runpy double-import warning", "RuntimeWarning" not in res.stderr, res.stderr[-300:])

    print("cli: @file.json for object parameters")
    report = tmp / "cli" / "report.json"
    report.write_text(json.dumps({
        "module": "reg_slave", "overall_status": "PASS",
        "summary": {"total": 1, "passed": 1, "failed": 0, "pass_rate": 1.0},
        "checks": [{"rule_id": "R1", "category": "c", "status": "PASS", "detail": "d"}],
    }), encoding="utf-8")
    deps = tmp / "cli" / "deps.json"
    deps.write_text(json.dumps({"dependency_closure": ["reg_slave"]}), encoding="utf-8")
    ifaces = tmp / "cli" / "ifaces.json"
    ifaces.write_text(json.dumps([{"protocol": "axi4_lite", "port_map": {"awaddr": "s_axi_awaddr"}}]), encoding="utf-8")
    out = tmp / "cli" / "visual.html"
    res = run("-m", "fpga_toolbox", "render_visual_report",
              "--report", f"@{report}", "--dep-graph", f"@{deps}",
              "--extracted-interfaces", f"@{ifaces}", "--out-html-path", str(out))
    check("@file.json objects", res.returncode == 0 and out.is_file(), res.stdout[-300:] + res.stderr[-300:])

    print("cli: boolean flags and failure exit code")
    sb = tmp / "cli_sb"
    res = run("-m", "fpga_toolbox", "copy_sandbox", "--project-root", str(proj),
              "--relative-files", "rtl/reg_slave.sv", "--sandbox-dir", str(sb), "--no-clean")
    check("--no-clean accepted", res.returncode == 0, res.stderr[-300:])
    res = run("-m", "fpga_toolbox", "index_repo", "--project-root", str(tmp / "missing"))
    check("failure exits 1", res.returncode == 1, str(res.returncode))


def main() -> int:
    with tempfile.TemporaryDirectory(prefix="fpga_tb_") as td:
        tmp = Path(td)
        test_sandbox_guards(tmp)
        test_sandbox_layout(tmp)
        test_reset_detection(tmp)
        test_normalizer(tmp)
        test_full_chain(tmp)
        test_cli(tmp)
    print()
    if FAILURES:
        print(f"FAILED ({len(FAILURES)}): " + ", ".join(FAILURES))
        return 1
    print("ALL CHECKS PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
