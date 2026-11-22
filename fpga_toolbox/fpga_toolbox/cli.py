"""
模块目的：统一 CLI 入口，把 TOOL_REGISTRY 里的每个原子工具自动映射为子命令与原生 --参数。
发生了什么：
  1. 支持 `python3 -m fpga_toolbox <tool_name> --param value` 与 `fpga-toolbox <tool_name> --param value`，
     不再需要 `.cli` 前缀和 `call --args '{...}'`。
  2. 提供 run_single_tool() 供 tools/*.py 单文件直接运行（`python3 fpga_toolbox/tools/iface_extractor.py --rtl-file ...`）。
  3. object/array 类型参数支持 `@file.json`（从文件读 JSON）、空格分隔多项或行内 JSON 字符串。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from .registry import TOOL_REGISTRY, call_tool, list_tool_schemas


def _parse_json_or_file(raw: str) -> Any:
    """支持 `@path.json` 从文件读取 JSON，或直接解析行内 JSON 字符串。"""
    if raw.startswith("@"):
        return json.loads(Path(raw[1:]).read_text(encoding="utf-8"))
    return json.loads(raw)


def _add_schema_args(parser: argparse.ArgumentParser, schema: Dict[str, Any]) -> None:
    """根据工具的 inputSchema 自动注册原生 --kebab-case 与 --snake_case 命令行参数。"""
    props = schema.get("properties", {})
    required = set(schema.get("required", []))
    for name, meta in props.items():
        ptype = meta.get("type", "string")
        kebab = "--" + name.replace("_", "-")
        flags = [kebab] if kebab == f"--{name}" else [kebab, f"--{name}"]
        is_req = name in required

        if ptype == "boolean":
            parser.add_argument(
                *flags,
                dest=name,
                action=argparse.BooleanOptionalAction,
                default=None,
                help=f"布尔参数 {name}",
            )
        elif ptype == "array":
            parser.add_argument(
                *flags,
                dest=name,
                nargs="+",
                required=is_req,
                default=None,
                help=f"列表参数 {name}（支持空格分隔多项或 @file.json）",
            )
        elif ptype == "object":
            parser.add_argument(
                *flags,
                dest=name,
                required=is_req,
                default=None,
                help=f"JSON 对象参数 {name}（支持 @file.json 或行内 JSON）",
            )
        else:
            parser.add_argument(
                *flags,
                dest=name,
                required=is_req,
                default=None,
                help=f"字符串参数 {name}",
            )


def _extract_params(ns: argparse.Namespace, schema: Dict[str, Any]) -> Dict[str, Any]:
    """将 argparse 解析结果按 inputSchema 还原为工具函数的关键字参数字典。"""
    props = schema.get("properties", {})
    params: Dict[str, Any] = {}
    for name, meta in props.items():
        val = getattr(ns, name, None)
        if val is None:
            continue
        ptype = meta.get("type", "string")
        if ptype == "object":
            params[name] = _parse_json_or_file(val)
        elif ptype == "array":
            # 若只传了单个 @file.json 或 '[...]' 字符串则按 JSON 解析，否则保留列表
            if len(val) == 1 and (val[0].startswith("@") or val[0].startswith("[")):
                params[name] = _parse_json_or_file(val[0])
            else:
                params[name] = val
        else:
            params[name] = val
    return params


def run_single_tool(tool_name: str, argv: Optional[List[str]] = None) -> int:
    """
    对外接口：供单个工具文件（tools/*.py）直接作为脚本运行时调用。
    示例：`python3 fpga_toolbox/tools/iface_extractor.py --rtl-file ./dut.sv`
    """
    meta = TOOL_REGISTRY[tool_name]
    p = argparse.ArgumentParser(prog=tool_name, description=meta["description"])
    _add_schema_args(p, meta["inputSchema"])
    ns = p.parse_args(argv)
    params = _extract_params(ns, meta["inputSchema"])
    res = call_tool(tool_name, params)
    print(json.dumps(res, indent=2, ensure_ascii=False))
    return 0 if res["ok"] else 1


def main(argv: Optional[List[str]] = None) -> int:
    """
    包级 CLI 主入口：支持 `python3 -m fpga_toolbox <tool_name> --参数 值`
    以及 `list` / `schemas` 辅助查询命令。
    """
    p = argparse.ArgumentParser(
        prog="fpga-toolbox",
        description="FPGA 原子化验证工具箱：直接用 `python3 -m fpga_toolbox <工具名> --参数 值` 调用。",
    )
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("list", help="列出全部可用原子工具")
    sub.add_parser("schemas", help="导出全部工具的 MCP / Function-Calling JSON Schema")

    # 每个工具同时注册为一级子命令，参数由 inputSchema 自动生成
    for tname, meta in TOOL_REGISTRY.items():
        tp = sub.add_parser(tname, help=meta["description"])
        _add_schema_args(tp, meta["inputSchema"])

    ns = p.parse_args(argv)
    if ns.cmd == "list":
        print(json.dumps(sorted(TOOL_REGISTRY), indent=2, ensure_ascii=False))
        return 0
    if ns.cmd == "schemas":
        print(json.dumps(list_tool_schemas(), indent=2, ensure_ascii=False))
        return 0

    meta = TOOL_REGISTRY[ns.cmd]
    params = _extract_params(ns, meta["inputSchema"])
    res = call_tool(ns.cmd, params)
    print(json.dumps(res, indent=2, ensure_ascii=False))
    return 0 if res["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
