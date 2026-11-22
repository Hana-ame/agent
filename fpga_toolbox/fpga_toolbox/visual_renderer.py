"""Tool 8: Render standalone SVG dependency topology + rule audit HTML dashboard.

Writes both ``<out>.html`` (dashboard) and ``<out>.svg`` (standalone topology),
and runs a built-in HTML consistency check (inspired by agent/verilog/axi/docs_check.py).
"""

from __future__ import annotations

import html
import re
from pathlib import Path
from typing import Any, Dict, List

if __package__ in (None, ""):
    import sys as _sys
    from pathlib import Path as _Path

    _sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
    __package__ = "fpga_toolbox"

from .core.types import err_result, ok_result


def _build_svg(report: Dict[str, Any], dep_graph: Dict[str, Any], ifaces: List[Dict[str, Any]]) -> str:
    mod = html.escape(str(report.get("module", "unknown")))
    status = html.escape(str(report.get("overall_status", "UNKNOWN")))
    accent = "#065F46" if status == "PASS" else ("#1E3A8A" if status == "NO_RESULTS" else "#991B1B")
    deps = [m for m in dep_graph.get("dependency_closure", []) if m != report.get("module")]
    iface_txt = ", ".join(
        f"{i.get('protocol', '?')}({len(i.get('port_map') or {})})" for i in ifaces
    ) or "none"
    height = max(240, 220 + len(deps[:5]) * 34)
    boxes = "".join(
        f'<rect x="240" y="{206+i*32}" width="240" height="24" fill="#F8FAFC" stroke="#64748B"/>'
        f'<text x="360" y="{222+i*32}" text-anchor="middle" fill="#111827" font-family="monospace" font-size="12">{html.escape(str(d))}</text>'
        for i, d in enumerate(deps[:5])
    )
    return f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 720 {height}" width="100%">
  <rect x="200" y="20" width="320" height="160" fill="#FFFFFF" stroke="#1D4ED8" stroke-dasharray="5 4"/>
  <rect x="236" y="56" width="248" height="72" fill="#0F172A"/>
  <text x="360" y="88" text-anchor="middle" fill="#FFFFFF" font-family="monospace" font-size="14">{mod}</text>
  <text x="360" y="108" text-anchor="middle" fill="#CBD5E1" font-family="monospace" font-size="11">{html.escape(iface_txt)}</text>
  <text x="360" y="160" text-anchor="middle" fill="{accent}" font-family="sans-serif" font-size="12" font-weight="bold">[{status}]</text>
  {boxes}
</svg>"""


def _check_html_consistency(page: str, expected_checks: int) -> List[Dict[str, Any]]:
    diags: List[Dict[str, Any]] = []
    for tag in ("table", "tbody", "html", "body"):
        opens = len(re.findall(rf"<{tag}\b", page, re.I))
        closes = len(re.findall(rf"</{tag}>", page, re.I))
        if opens != closes:
            diags.append({
                "severity": "WARNING",
                "code": "W050",
                "path": f"html.{tag}",
                "message": f"Unbalanced <{tag}> tags (open={opens}, close={closes}).",
            })
    rendered_rows = len(re.findall(r"<tbody>(.*?)</tbody>", page, re.S)[0].split("<tr>")) - 1 if "<tbody>" in page else 0
    if expected_checks > 0 and rendered_rows != expected_checks:
        diags.append({
            "severity": "WARNING",
            "code": "W051",
            "path": "html.checks",
            "message": f"Rendered check rows ({rendered_rows}) != report checks ({expected_checks}).",
        })
    return diags


def tool_render_visual_report(
    report: Dict[str, Any],
    dep_graph: Dict[str, Any],
    extracted_interfaces: List[Dict[str, Any]],
    out_html_path: str,
) -> Dict[str, Any]:
    if not isinstance(report, dict) or not isinstance(dep_graph, dict):
        return err_result("render_visual_report", "report and dep_graph must be objects.")
    if not isinstance(extracted_interfaces, list):
        return err_result("render_visual_report", "extracted_interfaces must be a list.")

    mod = html.escape(str(report.get("module", "unknown")))
    status = html.escape(str(report.get("overall_status", "UNKNOWN")))
    color = "#065F46" if status == "PASS" else ("#1E3A8A" if status == "NO_RESULTS" else "#991B1B")
    svg = _build_svg(report, dep_graph, extracted_interfaces)

    checks_list = [c for c in report.get("checks", []) if isinstance(c, dict)]
    rows = "".join(
        f"<tr><td><code>{html.escape(str(c.get('rule_id', '')))}</code></td><td>{html.escape(str(c.get('category', '')))}</td>"
        f"<td><strong>{html.escape(str(c.get('status', '')))}</strong></td><td>{html.escape(str(c.get('detail', '')))}</td></tr>"
        for c in checks_list
    ) or '<tr><td colspan="4">no checks recorded</td></tr>'
    summary = report.get("summary", {}) if isinstance(report.get("summary"), dict) else {}

    extra_bits: List[str] = []
    if isinstance(report.get("coverage_summary"), dict):
        cov = report["coverage_summary"]
        extra_bits.append(f"rtl_cov={html.escape(str(cov.get('rtl_pct', 'n/a')))}%")
    if isinstance(report.get("mutation_summary"), dict):
        mut = report["mutation_summary"]
        extra_bits.append(f"mutation={html.escape(str(mut.get('mutation_score', 'n/a')))}%")
    extra_s = (" · " + " · ".join(extra_bits)) if extra_bits else ""

    page = f"""<!DOCTYPE html><html lang="zh-CN"><head><meta charset="UTF-8"><title>{mod} 验证视图</title>
<style>body{{font-family:sans-serif;max-width:1040px;margin:40px auto;padding:0 24px;color:#111827}}
table{{width:100%;border-collapse:collapse;margin-top:12px}}th,td{{padding:9px 10px;border-bottom:1px solid #E5E7EB;text-align:left}}
.st{{color:{color};font-weight:700}}code{{background:#F3F4F6;padding:1px 5px;border-radius:3px}}</style></head><body>
<h1>FPGA 模块验证视图：{mod} <span class="st">[{status}]</span></h1>
<p>checks={summary.get('total', 0)} · pass={summary.get('passed', 0)} · fail={summary.get('failed', 0)} · pass_rate={summary.get('pass_rate', 0)}{extra_s}</p>
{svg}
<table><thead><tr><th>Rule ID</th><th>Category</th><th>Status</th><th>Detail</th></tr></thead><tbody>{rows}</tbody></table>
</body></html>"""

    diags = _check_html_consistency(page, len(checks_list))
    out_p = Path(out_html_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    out_p.write_text(page, encoding="utf-8")
    svg_p = out_p.with_suffix(".svg")
    svg_p.write_text(svg, encoding="utf-8")
    return ok_result(
        "render_visual_report",
        {"html_path": str(out_p.resolve()), "svg_path": str(svg_p.resolve()), "consistency_ok": len(diags) == 0},
        diagnostics=diags,
    )


if __name__ == "__main__":
    import sys

    from .cli import run_single_tool

    sys.exit(run_single_tool("render_visual_report"))
