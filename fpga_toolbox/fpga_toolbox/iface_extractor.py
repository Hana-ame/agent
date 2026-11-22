"""Tool 4: Extract AXI4-Lite, AXI4-Stream, APB, SV interface, clk/rst, and sideband ports from RTL AST."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml

if __package__ in (None, ""):
    import sys as _sys
    from pathlib import Path as _Path

    _sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
    __package__ = "fpga_toolbox"

from .core.sv_parser import SVModuleAST, SVPortAST, parse_sv_file
from .core.types import REQUIRED_APB_SIGNALS, REQUIRED_AXIL_SIGNALS, err_result, ok_result

_AXIL_PATS: Tuple[Tuple[str, re.Pattern[str]], ...] = (
    ("awaddr",  re.compile(r"(?:^|_)aw_?addr$", re.I)),
    ("awprot",  re.compile(r"(?:^|_)aw_?prot$", re.I)),
    ("awvalid", re.compile(r"(?:^|_)aw_?(?:valid|vld)$", re.I)),
    ("awready", re.compile(r"(?:^|_)aw_?(?:ready|rdy)$", re.I)),
    ("wdata",   re.compile(r"(?:^|_)w_?data$", re.I)),
    ("wstrb",   re.compile(r"(?:^|_)w_?strb$", re.I)),
    ("wvalid",  re.compile(r"(?:^|_)w_?(?:valid|vld)$", re.I)),
    ("wready",  re.compile(r"(?:^|_)w_?(?:ready|rdy)$", re.I)),
    ("bresp",   re.compile(r"(?:^|_)b_?resp$", re.I)),
    ("bvalid",  re.compile(r"(?:^|_)b_?(?:valid|vld)$", re.I)),
    ("bready",  re.compile(r"(?:^|_)b_?(?:ready|rdy)$", re.I)),
    ("araddr",  re.compile(r"(?:^|_)ar_?addr$", re.I)),
    ("arprot",  re.compile(r"(?:^|_)ar_?prot$", re.I)),
    ("arvalid", re.compile(r"(?:^|_)ar_?(?:valid|vld)$", re.I)),
    ("arready", re.compile(r"(?:^|_)ar_?(?:ready|rdy)$", re.I)),
    ("rdata",   re.compile(r"(?:^|_)r_?data$", re.I)),
    ("rresp",   re.compile(r"(?:^|_)r_?resp$", re.I)),
    ("rvalid",  re.compile(r"(?:^|_)r_?(?:valid|vld)$", re.I)),
    ("rready",  re.compile(r"(?:^|_)r_?(?:ready|rdy)$", re.I)),
)

_AXIS_PATS: Tuple[Tuple[str, re.Pattern[str]], ...] = (
    ("tdata",  re.compile(r"(?:^|_)t_?data$", re.I)),
    ("tvalid", re.compile(r"(?:^|_)t_?(?:valid|vld)$", re.I)),
    ("tready", re.compile(r"(?:^|_)t_?(?:ready|rdy)$", re.I)),
    ("tlast",  re.compile(r"(?:^|_)t_?last$", re.I)),
    ("tkeep",  re.compile(r"(?:^|_)t_?keep$", re.I)),
)

_APB_PATS: Tuple[Tuple[str, re.Pattern[str]], ...] = (
    ("psel",    re.compile(r"(?:^|_)p_?sel\d*$", re.I)),
    ("penable", re.compile(r"(?:^|_)p_?enable$", re.I)),
    ("pwrite",  re.compile(r"(?:^|_)p_?write$", re.I)),
    ("paddr",   re.compile(r"(?:^|_)p_?addr$", re.I)),
    ("pwdata",  re.compile(r"(?:^|_)p_?wdata$", re.I)),
    ("pstrb",   re.compile(r"(?:^|_)p_?strb$", re.I)),
    ("pprot",   re.compile(r"(?:^|_)p_?prot$", re.I)),
    ("pready",  re.compile(r"(?:^|_)p_?ready$", re.I)),
    ("prdata",  re.compile(r"(?:^|_)p_?rdata$", re.I)),
    ("pslverr", re.compile(r"(?:^|_)p_?slverr$", re.I)),
)

# Clock / reset naming. Anchored, so ``reset_done`` or ``clk_div2`` never match,
# while every common prefix style (``s_axi_aresetn``, ``i_rst_n``, ``core_clk``) does.
_CLK_RE = re.compile(r"^(?:[a-z0-9]+_)*(?:[aps]?clk|clock)$", re.I)
_RST_RE = re.compile(r"^(?:[a-z0-9]+_)*(?:[aps]?reset|[aps]?rst|nreset|nrst)(?:_?[nb])?$", re.I)


def _norm_prefix(raw: str) -> str:
    return re.sub(r"^(?:i_|o_|in_|out_)", "", raw, flags=re.I).strip("_").lower() or "default"


def _is_clock(name: str) -> bool:
    return bool(_CLK_RE.match(name))


def _is_reset(name: str) -> bool:
    return bool(_RST_RE.match(name))


def _reset_active_low(name: str) -> bool:
    n = name.lower()
    if re.search(r"(?:^|_)n(?:rst|reset)$", n):
        return True
    if n.endswith(("_n", "_b")):
        return True
    return bool(re.match(r"^(?:[a-z0-9]+_)*[aps]?(?:reset|rst)[nb]$", n))


def _infer_sv_iface_protocol(iface_type: str, port_name: str) -> str:
    probe = f"{iface_type} {port_name}".lower()
    if "axi" in probe and ("stream" in probe or "axis" in probe):
        return "axi4_stream"
    if "axi" in probe:
        return "axi4_lite"
    if "apb" in probe:
        return "apb"
    return "sv_interface"


def classify_module(mod: SVModuleAST) -> Dict[str, Any]:
    claimed, clk, rst = set(), None, None
    for p in mod.ports:
        if p.is_interface or p.direction != "input" or p.width != 1:
            continue
        if clk is None and _is_clock(p.name):
            clk = p
            claimed.add(p.name)
        elif rst is None and _is_reset(p.name):
            rst = p
            claimed.add(p.name)

    ifaces: List[Dict[str, Any]] = []
    # First classify explicit SystemVerilog interface/modport ports so they are never misclassified as 1-bit tie-offs.
    for p in mod.ports:
        if not p.is_interface:
            continue
        claimed.add(p.name)
        proto = _infer_sv_iface_protocol(p.interface_type, p.name)
        mp = p.modport.lower()
        role = "slave" if "slave" in mp or "slv" in mp else ("master" if "master" in mp or "mst" in mp else "unknown")
        ifaces.append({
            "protocol": proto,
            "group_prefix": p.name,
            "role": role,
            "addr_width": mod.parameters.get("AW"),
            "data_width": mod.parameters.get("DW"),
            "port_map": {},
            "sv_interface": {"port": p.name, "interface_type": p.interface_type, "modport": p.modport},
            "missing_signals": [],
        })

    for proto, pats, min_cnt in (("axi4_lite", _AXIL_PATS, 6), ("axi4_stream", _AXIS_PATS, 2), ("apb", _APB_PATS, 4)):
        groups: Dict[str, Dict[str, SVPortAST]] = {}
        for p in mod.ports:
            if p.name in claimed:
                continue
            for canon, pat in pats:
                m = pat.search(p.name)
                if m:
                    groups.setdefault(_norm_prefix(p.name[:m.start()]), {})[canon] = p
                    break
        for grp, sigs in groups.items():
            if len(sigs) < min_cnt:
                continue
            claimed.update(p.name for p in sigs.values())
            if proto == "apb":
                v_port = sigs.get("psel") or sigs.get("penable")
                req = REQUIRED_APB_SIGNALS
                aw_p, dw_p = sigs.get("paddr"), sigs.get("pwdata") or sigs.get("prdata")
            elif proto == "axi4_lite":
                v_port = sigs.get("awvalid") or sigs.get("arvalid")
                req = REQUIRED_AXIL_SIGNALS
                aw_p, dw_p = sigs.get("awaddr") or sigs.get("araddr"), sigs.get("wdata") or sigs.get("rdata")
            else:
                v_port = sigs.get("tvalid")
                req = ("tdata", "tvalid", "tready")
                aw_p, dw_p = None, sigs.get("tdata")
            role = ("slave" if v_port.direction == "input" else "master") if v_port else "unknown"
            ifaces.append({
                "protocol": proto,
                "group_prefix": grp,
                "role": role,
                "addr_width": aw_p.width if aw_p else None,
                "data_width": dw_p.width if dw_p else None,
                "port_map": {k: v.name for k, v in sigs.items()},
                "missing_signals": [s for s in req if s not in sigs],
            })

    oos = []
    for p in mod.ports:
        if p.name in claimed:
            continue
        if p.direction == "input":
            item = {"port": p.name, "dir": "input", "width": p.width, "tie_off": f"{p.width}'b0"}
            if _is_clock(p.name) or _is_reset(p.name):
                item["warning"] = "clock/reset-like port left out of scope; verify tie-off is intended"
            oos.append(item)
        else:
            oos.append({"port": p.name, "dir": p.direction, "width": p.width, "observe_only": True})
    return {
        "module_name": mod.name,
        "rtl_file": mod.file_path,
        "parser_backend": mod.parser_backend,
        "parameters": mod.parameters,
        "total_ports": len(mod.ports),
        "clock": {"port": clk.name, "freq_mhz": 100.0} if clk else None,
        "reset": {"port": rst.name, "active_low": _reset_active_low(rst.name), "sync": True} if rst else None,
        "interfaces": ifaces,
        "out_of_scope_ports": oos,
    }


def tool_extract_interfaces(
    rtl_file: str,
    module_name: Optional[str] = None,
    emit_draft_yaml: Optional[str] = None,
    staged_rtl_files: Optional[List[str]] = None,
    parser_backend: str = "auto",
) -> Dict[str, Any]:
    p = Path(rtl_file)
    if not p.is_file():
        return err_result("extract_interfaces", f"RTL file not found: {rtl_file}")
    # 收集 slang 后端的回退原因：slang 装了却没生效时对调用方可见，
    # 而不是静默退回 builtin（否则「可选 slang AST 后端」就是虚报）。
    parse_diags: List[str] = []
    mods = parse_sv_file(p, backend=parser_backend, diagnostics=parse_diags)
    if not mods:
        return err_result("extract_interfaces", f"No modules found in {rtl_file}")
    selected = [m for m in mods if m.name == module_name] if module_name else mods
    if not selected:
        return err_result("extract_interfaces", f"Module '{module_name}' not found.")

    extracted = [classify_module(m) for m in selected]
    data: Dict[str, Any] = {"rtl_file": str(p.resolve()), "modules": extracted}

    diagnostics: List[Dict[str, Any]] = []
    # 把 slang 回退/解析诊断透传（W013），避免静默降级
    for pd in parse_diags:
        diagnostics.append({
            "severity": "WARNING",
            "code": "W013",
            "path": p.name,
            "message": f"Parser backend: {pd}",
        })
    for m in extracted:
        if m["clock"] is None:
            diagnostics.append({"severity": "WARNING", "code": "W010", "path": m["module_name"], "message": "No clock port detected; defaulting to aclk@100MHz."})
        if m["reset"] is None:
            diagnostics.append({"severity": "WARNING", "code": "W011", "path": m["module_name"], "message": "No reset port detected; defaulting to aresetn active-low."})
        for iface in m["interfaces"]:
            if "sv_interface" in iface and m.get("parser_backend") != "slang":
                diagnostics.append({
                    "severity": "WARNING",
                    "code": "W012",
                    "path": f"{m['module_name']}.{iface['sv_interface']['port']}",
                    "message": f"SV interface port '{iface['sv_interface']['interface_type']}.{iface['sv_interface']['modport']}' detected by builtin parser; install slang to expand member signals.",
                })

    if emit_draft_yaml:
        m0 = extracted[0]
        axil = next((i for i in m0["interfaces"] if i["protocol"] == "axi4_lite"), None)
        apb = next((i for i in m0["interfaces"] if i["protocol"] == "apb"), None)
        boundary: Dict[str, Any] = {
            "module_name": m0["module_name"],
            "rtl_files": staged_rtl_files or [m0["rtl_file"]],
            "parameters": m0["parameters"],
            "clock": m0["clock"] or {"port": "aclk", "freq_mhz": 100.0},
            "reset": m0["reset"] or {"port": "aresetn", "active_low": True, "sync": True},
            "axi4_lite": {
                "addr_width": (axil["addr_width"] or 12) if axil else 12,
                "data_width": (axil["data_width"] or 32) if axil else 32,
                "role": (axil["role"] if axil and axil["role"] != "unknown" else "slave"),
                "port_map": axil["port_map"] if axil else {},
            },
            "out_of_scope_ports": m0["out_of_scope_ports"],
        }
        if apb:
            boundary["apb"] = {
                "addr_width": apb["addr_width"] or 8,
                "data_width": apb["data_width"] or 32,
                "role": apb["role"] if apb["role"] != "unknown" else "slave",
                "port_map": apb["port_map"],
            }
        if m0.get("instance_hierarchy"):
            boundary["instance_hierarchy"] = m0["instance_hierarchy"]
        draft = {
            "boundary": boundary,
            "test_plan": {
                "protocol_rules": {"check_valid_stability": True, "check_reset_outputs": True, "max_resp_cycles": 16, "allow_unaligned_addr": False},
                "register_rules": [{"rule_id": "REG_001", "name": "CTRL_REG", "offset": 0, "reset_value": 0, "access": "RW", "write_mask": 0xFFFFFFFF, "read_mask": 0xFFFFFFFF}],
                "illegal_addr_rules": [{"rule_id": "ILL_001", "offset": 0xFC, "expected_resp": "SLVERR"}],
                "stress_profile": {"random_backpressure": True, "iterations": 32, "seed": 20261120},
            },
        }
        out_p = Path(emit_draft_yaml)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        out_p.write_text(yaml.safe_dump(draft, sort_keys=False, allow_unicode=True), encoding="utf-8")
        data["draft_yaml_path"] = str(out_p.resolve())

    return ok_result("extract_interfaces", data, diagnostics=diagnostics)


if __name__ == "__main__":
    import sys

    from .cli import run_single_tool

    sys.exit(run_single_tool("extract_interfaces"))
