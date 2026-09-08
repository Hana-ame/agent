"""Standalone CLI runner for V4 edges."""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, Optional

_REPO_ROOT = str(Path(__file__).resolve().parent.parent.parent)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from framework.edges.base import AgentProtocol, EdgeV4, MockAgentV4
from framework.vertex_v4 import VertexStateV4, VertexStoreV4

logger = logging.getLogger("vertex_edge_agent.edges.cli")


def parse_args() -> argparse.Namespace:
    """Parse standalone command line arguments."""
    parser = argparse.ArgumentParser(description="Standalone V4 Edge Runner")
    parser.add_argument("--config", "-c", default=None, help="Path to JSON file specifying all edge parameters")
    parser.add_argument("--dir", default=None, help="Base directory for resolving relative script and database paths")
    parser.add_argument("--db", default=None, help="SQLite database path")
    parser.add_argument("--session", default=None, help="Session identifier (required if not in config JSON)")
    parser.add_argument("--edge-id", default=None, help="Edge identifier")
    parser.add_argument(
        "--type",
        choices=["code", "llm", "llm_chat", "llm_generate", "llm_process", "llm_callable", "reflexive"],
        default=None,
        help="Edge type",
    )
    parser.add_argument("--input", default=None, help="Input vertex name")
    parser.add_argument("--output", default=None, help="Output vertex name")
    parser.add_argument("--script", default=None, help="Script path for code or recovery edge")
    parser.add_argument("--model", default=None, help="Model name for LLM edge")
    parser.add_argument("--settings", default=None, help="JSON settings string")
    parser.add_argument("--seed-input", default=None, help="Optional initial input string to seed into input vertex")
    parser.add_argument("--mock", action="store_true", default=None, help="Run with mock LLM agent for offline testing")
    return parser.parse_args()


def main() -> None:
    """Standalone runner entrypoint capable of executing from any working directory."""
    args = parse_args()

    config_json: Dict[str, Any] = {}
    if args.config:
        cfg_path = Path(args.config)
        if not cfg_path.is_absolute() and args.dir:
            cfg_path = Path(args.dir) / cfg_path
        if not cfg_path.exists() and os.path.exists(os.path.join(_REPO_ROOT, str(args.config))):
            cfg_path = Path(_REPO_ROOT) / args.config
        if not cfg_path.exists():
            sys.stderr.write(f"Error: Config file not found: {args.config}\n")
            sys.exit(1)
        try:
            with open(cfg_path, "r", encoding="utf-8") as f:
                config_json = json.load(f)
        except Exception as e:
            sys.stderr.write(f"Error reading JSON config file '{args.config}': {e}\n")
            sys.exit(1)

    settings_dict = dict(config_json.get("settings") or {})
    if args.settings is not None:
        try:
            cli_settings = json.loads(args.settings)
            settings_dict.update(cli_settings)
        except json.JSONDecodeError as e:
            print(f"Error: Invalid JSON for --settings: {e}", file=sys.stderr)
            sys.exit(1)

    base_dir = args.dir or config_json.get("dir")
    if args.config and not base_dir:
        base_dir = str(Path(args.config).resolve().parent)
    if base_dir:
        abs_base_dir = os.path.abspath(base_dir)
        if abs_base_dir not in sys.path:
            sys.path.insert(0, abs_base_dir)

    session = args.session or config_json.get("session") or config_json.get("session_id")
    if not session:
        sys.stderr.write("Error: --session is required (either via CLI argument or 'session' in JSON config)\n")
        sys.exit(1)

    db_path = args.db or config_json.get("db", ":memory:")
    if db_path != ":memory:" and not os.path.isabs(db_path) and base_dir:
        db_path = os.path.join(base_dir, db_path)

    seed_input = args.seed_input if args.seed_input is not None else config_json.get("seed_input")
    is_mock = args.mock if args.mock is not None else bool(config_json.get("mock", False))

    edge_config = dict(config_json)
    if args.edge_id:
        edge_config["id"] = args.edge_id
    elif "id" not in edge_config and "edge_id" not in edge_config:
        edge_config["id"] = "standalone_edge"

    if args.type:
        edge_config["type"] = args.type
    elif "type" not in edge_config and "edge_type" not in edge_config:
        edge_config["type"] = "code"

    if args.input:
        edge_config["input_vertex"] = args.input
    if args.output:
        edge_config["output_vertex"] = args.output
    if args.script:
        edge_config["script"] = args.script
    if args.model:
        edge_config["model"] = args.model
    edge_config["settings"] = settings_dict

    try:
        edge = EdgeV4.from_config(edge_config, base_dir=base_dir)
    except Exception as exc:
        sys.stderr.write(f"Error initializing edge: {exc}\n")
        sys.exit(1)

    store = VertexStoreV4(db_path)
    try:
        if seed_input is not None and edge.input_vertex:
            store.save_vertex(
                session_id=session,
                name=edge.input_vertex,
                content=seed_input,
                state=VertexStateV4.DATA_READY.value,
            )
            if edge.output_vertex and store.get_vertex(session, edge.output_vertex) is None:
                store.save_vertex(
                    session_id=session,
                    name=edge.output_vertex,
                    content="",
                    state=VertexStateV4.TODO.value,
                )

        agent: Optional[AgentProtocol] = None
        if is_mock:
            mock_resp = settings_dict.get(
                "mock_response",
                '{"mock": true, "status": "success", "content": "mock agent response"}',
            )
            agent = MockAgentV4(response_text=str(mock_resp))

        result = asyncio.run(edge.run(session, store, agent=agent))
        sys.stdout.write(json.dumps(result.to_dict(), indent=2) + "\n")
        if not result.success and not result.skipped:
            sys.exit(1)
    finally:
        store.close()


if __name__ == "__main__":
    main()
