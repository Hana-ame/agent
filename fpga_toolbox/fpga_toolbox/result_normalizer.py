"""Tool 7: Normalize cocotb / SVA / Verilator suite results into a strict VerificationReport schema.

Accepted inputs:
  * cocotb JUnit XML (``results.xml``)
  * JSON envelope carrying a ``checks`` or ``tests`` list, or a bare JSON list
  * SVA log containing ``[RULE:P_xxx]`` lines
  * Verilator multi-suite regression log (with per-suite identity verification)

Unrecognized input is reported as an error instead of silently yielding an
empty PASS/NO_RESULTS report.
"""

from __future__ import annotations

import json
import re
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

if __package__ in (None, ""):
    import sys as _sys
    from pathlib import Path as _Path

    _sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
    __package__ = "fpga_toolbox"

from .core.types import err_result, ok_result
from .sim_runner import verify_suite_identities

_SVA_ERR_RE = re.compile(r"\[RULE:(P_\d+)\]\s*(.+)")
_STATUS_MAP = {
    "PASS": "PASS", "PASSED": "PASS", "OK": "PASS",
    "FAIL": "FAIL", "FAILED": "FAIL", "ERROR": "FAIL",
    "SKIP": "SKIP", "SKIPPED": "SKIP",
}


def _norm_status(raw: Any) -> str:
    return _STATUS_MAP.get(str(raw).strip().upper(), "FAIL")


def _from_xml(path: Path) -> Tuple[List[Dict[str, Any]], Optional[int]]:
    root = ET.parse(path).getroot()
    checks: List[Dict[str, Any]] = []
    for tc in root.iter("testcase"):
        status, detail = "PASS", ""
        for child in tc:
            tag = child.tag.lower()
            if tag in ("failure", "error"):
                status = "FAIL"
                detail = (child.get("message") or child.text or "").strip()
                break
            if tag == "skipped":
                status = "SKIP"
                detail = (child.get("message") or "").strip()
        checks.append({
            "rule_id": tc.get("name") or "UNKNOWN",
            "category": tc.get("classname") or "cocotb",
            "status": status,
            "detail": detail.replace("\n", " ")[:400],
        })
    return checks, None


def _from_json(raw: Any) -> Tuple[List[Dict[str, Any]], Optional[int]]:
    if isinstance(raw, dict) and isinstance(raw.get("checks"), list):
        checks = [{
            "rule_id": str(c.get("rule_id", "UNKNOWN")),
            "category": str(c.get("category", "functional")),
            "status": _norm_status(c.get("status", "FAIL")),
            "detail": str(c.get("detail", "")),
        } for c in raw["checks"]]
        return checks, raw.get("seed")

    tests = None
    seed = None
    if isinstance(raw, dict) and isinstance(raw.get("tests"), list):
        tests, seed = raw["tests"], raw.get("seed")
    elif isinstance(raw, list):
        tests = raw
    if tests is not None:
        checks = [{
            "rule_id": str(t.get("name", t.get("rule_id", "UNKNOWN"))),
            "category": str(t.get("category", "cocotb")),
            "status": _norm_status(t.get("status", t.get("result", "FAIL"))),
            "detail": str(t.get("detail", "")),
        } for t in tests if isinstance(t, dict)]
        return checks, seed

    raise ValueError(
        "Unrecognized results format: expected cocotb JUnit XML, "
        "a JSON object with 'checks'/'tests', or a JSON list of tests."
    )


def tool_normalize_sim_result(
    module_name: str,
    cocotb_results_json: Optional[str] = None,
    sva_log_path: Optional[str] = None,
    verilator_log_path: Optional[str] = None,
    expected_suites: Optional[List[str]] = None,
    coverage_summary: Optional[Dict[str, Any]] = None,
    mutation_summary: Optional[Dict[str, Any]] = None,
    out_report_json: Optional[str] = None,
) -> Dict[str, Any]:
    if not cocotb_results_json and not sva_log_path and not verilator_log_path:
        return err_result("normalize_sim_result", "Provide cocotb_results_json, sva_log_path, and/or verilator_log_path.")

    checks: List[Dict[str, Any]] = []
    seed: Optional[int] = None
    sources: List[str] = []

    if cocotb_results_json:
        p = Path(cocotb_results_json)
        if not p.is_file():
            return err_result("normalize_sim_result", f"Results file not found: {cocotb_results_json}")
        try:
            if p.suffix.lower() == ".xml":
                parsed, seed = _from_xml(p)
            else:
                parsed, seed = _from_json(json.loads(p.read_text(encoding="utf-8")))
        except (ValueError, ET.ParseError, json.JSONDecodeError) as exc:
            return err_result("normalize_sim_result", f"Failed to parse '{p}': {exc}")
        checks.extend(parsed)
        sources.append(str(p))

    if sva_log_path:
        p = Path(sva_log_path)
        if not p.is_file():
            return err_result("normalize_sim_result", f"SVA log not found: {sva_log_path}")
        for line in p.read_text(encoding="utf-8", errors="ignore").splitlines():
            m = _SVA_ERR_RE.search(line)
            if m:
                checks.append({"rule_id": m.group(1), "category": "sva_protocol", "status": "FAIL", "detail": m.group(2).strip()})
        sources.append(str(p))

    if verilator_log_path:
        p = Path(verilator_log_path)
        if not p.is_file():
            return err_result("normalize_sim_result", f"Verilator log not found: {verilator_log_path}")
        log_txt = p.read_text(encoding="utf-8", errors="ignore")
        ident = verify_suite_identities(log_txt, expected_suites or [])
        for sname, st in ident["observed"].items():
            checks.append({
                "rule_id": f"SUITE_{sname}",
                "category": "verilator_suite",
                "status": "PASS" if st == "PASS" else "FAIL",
                "detail": f"suite={sname} status={st}",
            })
        for miss in ident["missing"]:
            checks.append({
                "rule_id": f"SUITE_{miss}",
                "category": "verilator_suite",
                "status": "FAIL",
                "detail": f"Expected suite '{miss}' missing from log (silent skip guard).",
            })
        sources.append(str(p))

    passed = sum(1 for c in checks if c["status"] == "PASS")
    failed = sum(1 for c in checks if c["status"] == "FAIL")
    skipped = sum(1 for c in checks if c["status"] == "SKIP")
    overall = "PASS" if (passed > 0 and failed == 0) else ("FAIL" if failed > 0 else "NO_RESULTS")
    report: Dict[str, Any] = {
        "schema_version": "1.1",
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "module": module_name,
        "seed": seed,
        "overall_status": overall,
        "summary": {
            "total": len(checks),
            "passed": passed,
            "failed": failed,
            "skipped": skipped,
            "pass_rate": round(passed / len(checks), 4) if checks else 0.0,
        },
        "sources": sources,
        "failed_rules": [c for c in checks if c["status"] == "FAIL"],
        "checks": checks,
    }
    if coverage_summary is not None:
        report["coverage_summary"] = coverage_summary
    if mutation_summary is not None:
        report["mutation_summary"] = mutation_summary

    if out_report_json:
        out = Path(out_report_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return ok_result("normalize_sim_result", report)


if __name__ == "__main__":
    import sys

    from .cli import run_single_tool

    sys.exit(run_single_tool("normalize_sim_result"))
