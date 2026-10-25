"""V4 manifest runner — load, seed the store and execute in one call.

Compatibility note: ``examples/run.py`` is the **V1** runner and is left untouched.
This module is the V4 entry point and is installed as the ``vea-run-v4`` console
script (``framework.run_v4:main``), so a manifest can be executed right after
``pip install`` without touching the repository:

    vea-run-v4 examples/custom_edge/config.json --session demo

Programmatic one-step use (the two calls ``load_from_manifest`` +
``populate_store`` collapse into one):

    from framework.run_v4 import run_from_manifest
    result = run_from_manifest("examples/custom_edge/config.json", session_id="demo")
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

from framework.executor_v4 import ExecutionResultV4, ExecutorV4
from framework.graphs.loader import DiscreteGraphLoaderV4
from framework.graphs.core import GraphV4
from framework.vertex_v4 import VertexStoreV4

__all__ = [
    "load_v4_manifest",
    "run_manifest_async",
    "run_from_manifest",
    "parse_args",
    "main",
]


def _looks_like_v1_manifest(data: Dict[str, Any]) -> bool:
    """Best-effort check that ``data`` is a V1 manifest, not a V4 one.

    V4 vertices carry ``name`` / ``state`` / ``content``; V1 ones carry ``id`` /
    ``settings`` / ``initial_data``. Only manifests that unambiguously look V1 are
    flagged, so nothing that could still be V4 is rejected here.
    """
    if str(data.get("version", "")).startswith("4"):
        return False
    vertices = [v for v in (data.get("vertices") or []) if isinstance(v, dict)]
    if not vertices:
        return False
    return all("name" not in v and "state" not in v for v in vertices)


def load_v4_manifest(
    manifest_path: Union[str, Path],
    override_session_id: Optional[str] = None,
    script_roots: Optional[Sequence[Union[str, Path]]] = None,
) -> GraphV4:
    """Load a V4 master manifest, refusing V1 manifests with an actionable error.

    Raises:
        ValueError: ``manifest_path`` points at a V1 manifest (the legacy runner
            ``examples/run.py`` must be used instead).
        FileNotFoundError: The manifest does not exist.
    """
    path = Path(manifest_path).resolve()
    if not path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")
    with open(path, "r", encoding="utf-8") as handle:
        data = json.load(handle)

    if _looks_like_v1_manifest(data):
        raise ValueError(
            f"'{manifest_path}' is a V1 manifest, not a V4 one "
            "(V4 vertices must use 'name'/'state' instead of 'id'/'initial_data').\n"
            "  V4 manifests: vea-run-v4 <config.json>   (or `python -m framework.run_v4`)\n"
            "  V1 manifests: python examples/run.py <config.json>   (legacy runner, unchanged)"
        )

    return DiscreteGraphLoaderV4.load_from_manifest(
        str(path),
        override_session_id=override_session_id,
        script_roots=script_roots,
    )


async def run_manifest_async(
    manifest_path: Union[str, Path],
    *,
    session_id: Optional[str] = None,
    store: Optional[VertexStoreV4] = None,
    script_roots: Optional[Sequence[Union[str, Path]]] = None,
    max_concurrency: int = 4,
    timeout: float = 120.0,
    agent: Any = None,
    snapshot_dir: Optional[Union[str, Path]] = None,
    enable_snapshots: bool = False,
) -> ExecutionResultV4:
    """Load a V4 manifest, seed every vertex/edge into the store, and execute it.

    This is the single-call equivalent of
    ``load_from_manifest`` → ``populate_store`` → ``ExecutorV4(...).run()``, so a
    manifest is directly runnable without wiring the store by hand.
    """
    graph = load_v4_manifest(manifest_path, override_session_id=session_id, script_roots=script_roots)
    owns_store = store is None
    if owns_store:
        store = VertexStoreV4(":memory:")
    try:
        DiscreteGraphLoaderV4.populate_store(graph, store)
        executor = ExecutorV4(
            graph=graph,
            store=store,
            agent=agent,
            max_concurrency=max_concurrency,
            timeout=timeout,
            snapshot_dir=snapshot_dir,
            enable_snapshots=enable_snapshots,
        )
        return await executor.run()
    finally:
        if owns_store:
            close = getattr(store, "close", None)
            if close is not None:
                close()


def run_from_manifest(
    manifest_path: Union[str, Path],
    *,
    session_id: Optional[str] = None,
    store: Optional[VertexStoreV4] = None,
    script_roots: Optional[Sequence[Union[str, Path]]] = None,
    max_concurrency: int = 4,
    timeout: float = 120.0,
    agent: Any = None,
    snapshot_dir: Optional[Union[str, Path]] = None,
    enable_snapshots: bool = False,
) -> ExecutionResultV4:
    """Synchronous wrapper around :func:`run_manifest_async`.

    Use :func:`run_manifest_async` directly when you are already inside a running
    event loop (for example inside an ASGI request handler).
    """
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        pass
    else:
        raise RuntimeError(
            "run_from_manifest() cannot be called from a running event loop; "
            "await run_manifest_async(...) instead."
        )
    return asyncio.run(
        run_manifest_async(
            manifest_path,
            session_id=session_id,
            store=store,
            script_roots=script_roots,
            max_concurrency=max_concurrency,
            timeout=timeout,
            agent=agent,
            snapshot_dir=snapshot_dir,
            enable_snapshots=enable_snapshots,
        )
    )


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    """Parse ``vea-run-v4`` command line arguments."""
    parser = argparse.ArgumentParser(
        prog="vea-run-v4",
        description="Run a V4 manifest end to end: load, seed SQLite, execute.",
    )
    parser.add_argument("manifest", help="Path to a V4 master manifest JSON")
    parser.add_argument("--session", "-s", default=None, help="Session id (default: from the manifest)")
    parser.add_argument("--db", default=":memory:", help="SQLite database path (default: in-memory)")
    parser.add_argument(
        "--script-root",
        action="append",
        default=None,
        dest="script_roots",
        help="Directory that edge script specs may load from (repeatable). "
        "Default: the manifest directory, the repository root and VEA_SCRIPT_ROOTS.",
    )
    parser.add_argument("--concurrency", type=int, default=4, help="Max concurrent edges (default: 4)")
    parser.add_argument("--timeout", type=float, default=120.0, help="Per-run timeout in seconds (default: 120)")
    parser.add_argument("--snapshot-dir", default=None, help="Record graph snapshots into this directory")
    parser.add_argument("--json", action="store_true", help="Print the full result as JSON")
    parser.add_argument("--print-events", action="store_true", help="Stream execution events as they happen")
    parser.add_argument("--quiet", action="store_true", help="Print nothing except errors")
    return parser.parse_args(argv)


def _format_result(result: ExecutionResultV4) -> str:
    """Render an execution result as a short human-readable report."""
    lines: List[str] = [
        f"session:     {result.session_id}",
        f"success:     {result.success}",
        f"elapsed:     {result.execution_time:.3f}s",
        f"completed:   {', '.join(result.completed_edges) or '(none)'}",
    ]
    if result.vertex_contents:
        lines.append("vertex data:")
        for name, content in result.vertex_contents.items():
            preview = str(content).replace("\n", " ")
            if len(preview) > 120:
                preview = preview[:117] + "..."
            lines.append(f"  {name}: {preview}")
    if result.errors:
        lines.append(f"errors:      {len(result.errors)}")
        lines.extend(f"  - {err}" for err in result.errors[:10])
    return "\n".join(lines)


async def _run_with_events(manifest_path, args) -> ExecutionResultV4:
    """Execute a manifest while printing each observability event as it arrives."""
    graph = load_v4_manifest(
        manifest_path, override_session_id=args.session, script_roots=args.script_roots
    )
    store = VertexStoreV4(args.db)
    try:
        DiscreteGraphLoaderV4.populate_store(graph, store)
        executor = ExecutorV4(
            graph=graph,
            store=store,
            max_concurrency=args.concurrency,
            timeout=args.timeout,
            snapshot_dir=args.snapshot_dir,
        )
        async for event in executor.stream():
            print(
                f"{event.event_type:<12} "
                f"edge={event.edge_id or '-'} vertex={event.vertex_name or '-'}"
            )
        result = getattr(executor, "_result", None) or ExecutionResultV4(
            session_id=executor.session_id
        )
        if not args.quiet:
            print()
            print(_format_result(result))
        return result
    finally:
        close = getattr(store, "close", None)
        if close is not None:
            close()


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Command line entry point. Returns 0 on a successful run, 1 otherwise."""
    args = parse_args(argv)
    # A file database is owned by this process, so it must be closed here rather
    # than by the one-step helper (which only closes stores it created itself).
    store = VertexStoreV4(args.db) if args.db != ":memory:" else None
    result: Optional[ExecutionResultV4] = None
    try:
        if args.print_events:
            result = asyncio.run(_run_with_events(args.manifest, args))
        elif store is None:
            result = run_from_manifest(
                args.manifest,
                session_id=args.session,
                script_roots=args.script_roots,
                max_concurrency=args.concurrency,
                timeout=args.timeout,
                snapshot_dir=args.snapshot_dir,
            )
        else:
            result = asyncio.run(
                run_manifest_async(
                    args.manifest,
                    session_id=args.session,
                    store=store,
                    script_roots=args.script_roots,
                    max_concurrency=args.concurrency,
                    timeout=args.timeout,
                    snapshot_dir=args.snapshot_dir,
                )
            )
        if args.json:
            print(json.dumps(result.to_dict(), indent=2, ensure_ascii=False))
        elif not args.quiet:
            print(_format_result(result))
    except (ValueError, FileNotFoundError) as exc:
        sys.stderr.write(f"error: {exc}\n")
        return 1
    except Exception as exc:  # noqa: BLE001 - a runner must not leak a traceback
        sys.stderr.write(f"error: execution failed: {exc}\n")
        return 1
    finally:
        if store is not None:
            close = getattr(store, "close", None)
            if close is not None:
                close()
    return 0 if result and result.success else 1


if __name__ == "__main__":
    sys.exit(main())
