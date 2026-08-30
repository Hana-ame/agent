"""Single-edge driver — run any script edge (``file.py:ClassName``) standalone.

Drives one edge's full orchestration chain — ``pre_process -> compute ->
post_process`` (the same path ``Edge._run_compute`` uses inside the graph) —
without building a whole Graph. Useful for debugging a single edge in
isolation: verify the right class is loaded (script-loader by-name bug), feed
it arbitrary data, and see the result without wiring up vertices/executor.

This driver is generic over edges — it does NOT assume an LLM compute. A real
LLM call is only made when ``--base-url``/``--api-key`` are given; otherwise
you are expected to ``--skip-compute`` (offline, deterministic: the
``pre_process`` output flows straight into ``post_process``). Pure-data /
fetch edges are driven offline this way; LLM edges need the endpoint.

Usage::

    # Offline, deterministic — skip compute (generic non-LLM edge, no cost):
    python -m framework.utils.run_edge \\
        --dir examples/hn_ai_report \\
        --script hn_edges.py:FetchCommentsEdge \\
        --data '{"id":123,"url":"https://news.ycombinator.com/item?id=123"}' \\
        --skip-compute

    # Real LLM (e.g. sensenova), endpoint+key given explicitly:
    python -m framework.utils.run_edge \\
        --dir examples/hn_ai_report \\
        --script hn_edges.py:SummarizeEdge \\
        --data '{...}' \\
        --base-url https://token.sensenova.cn/v1/chat/completions \\
        --api-key "$SENSENOVA_API_KEY"

The ``--script`` path is resolved relative to ``--dir`` (default: CWD), and
``file.py:Class`` is loaded by explicit class name — mirroring how the
framework resolves MapEdge pipeline steps (relative to the config dir).

Return code: 0 on success, 1 when the edge itself failed, 2 on CLI misuse
(no LLM endpoint given and ``--skip-compute`` absent).
"""

import argparse
import asyncio
import json as _json
import os
import sys
import time
from typing import Any, Dict, Optional

# Allow running as ``python -m framework.utils.run_edge`` from the repo root
# or from an example dir: add framework root to sys.path when missing.
_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")
)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from framework.edge import Edge  # noqa: E402
from framework.agents import HttpLLMAgent  # noqa: E402
from framework.utils.script_loader import load_class_from_script  # noqa: E402


async def run_edge(
    dir_path: str,
    script: str,
    data: Any,
    skip_compute: bool = False,
    base_url: Optional[str] = None,
    api_key: Optional[str] = None,
) -> Dict[str, Any]:
    """Load and drive one edge end-to-end; return a result report dict.

    Args:
        dir_path: Base directory the ``script`` path resolves against.
        script:   ``file.py`` or ``file.py:ClassName`` (explicit name wins).
        data:     Input payload fed to the edge (pre_process target).
        skip_compute:  Skip the compute stage (pre_process output flows
            directly into post_process). General-edges offline path.
        base_url: Full chat-completions URL (real LLM; compute NOT skipped).
        api_key:  API key for the LLM call (real LLM).

    Returns:
        Dict with ``script``, ``class``, ``agent``, ``ok``, ``result``,
        ``latency_s`` and (for real LLM) ``usage``.  ``agent`` is ``None``
        when compute is skipped.
    """
    start = time.monotonic()
    if skip_compute and base_url:
        raise ValueError("--skip-compute and --base-url are mutually exclusive")
    if not skip_compute and not base_url:
        # No silent MockAgent fallback (framework Edge.compute would use one):
        # a compute run must have a real LLM endpoint, or skip compute.
        raise ValueError(
            "compute requires --base-url; use skip_compute=True for offline "
            "generic-edge runs (no mock fallback)"
        )

    edge = None
    agent = None
    ok = False
    res = None
    try:
        # Script loading is part of the driven chain: a missing file or a
        # wrong class name surfaces as a report failure (ok=False), not a
        # crash out of the driver.
        cls = _resolve_class(dir_path, script)

        # ``skip_compute`` is carried through the edge settings so graph and
        # standalone driver use the SAME mechanism (edge-level short-circuit in
        # ``_run_compute``), not two different code paths.
        edge = cls(
            edge_id="single_edge", source_id="src", destination_id="dst",
            settings={"skip_compute": skip_compute} if skip_compute else None,
        )

        if base_url:
            agent = HttpLLMAgent(api_key=api_key or "", base_url=base_url)  # type: ignore[call-arg]

        res = await edge._run_pre_process(data)
        res = await edge._run_compute(res, agent)
        res = await edge._run_post_process(res)
        ok = True
    except Exception as e:  # surfaced same way the executor reports edge failure
        res = f"{type(e).__name__}: {e}"
    finally:
        latency = time.monotonic() - start
        if isinstance(agent, HttpLLMAgent):
            await agent.close()

    class_name = cls.__name__ if edge is not None else "<load failed>"
    report: Dict[str, Any] = {
        "script": script,
        "class": class_name,
        "agent": None if skip_compute else ("HttpLLMAgent" if isinstance(agent, HttpLLMAgent) else "none"),
        "skip_compute": skip_compute,
        "ok": ok,
        "latency_s": round(latency, 2),
        "result": res,
    }
    if isinstance(agent, HttpLLMAgent) and hasattr(agent, "get_usage_summary"):
        report["usage"] = agent.get_usage_summary()
    return report


def _resolve_class(dir_path: str, script: str) -> type:
    """Resolve ``file.py`` / ``file.py:ClassName`` to the Edge subclass.

    The script file is resolved relative to ``dir_path`` (mirrors how graph.py
    resolves MapEdge pipeline steps relative to the config directory), and an
    explicit ``:ClassName`` wins over auto-discovery.
    """
    def _resolve(rel: str) -> str:
        return rel if os.path.isabs(rel) else os.path.join(dir_path, rel)

    if ":" in script:
        path_part, cls_name = script.split(":", 1)
        return load_class_from_script(_resolve(path_part), Edge, cls_name)
    return load_class_from_script(_resolve(script), Edge, Edge)


def _load_data(raw: Optional[str]) -> Any:
    if raw is None:
        return None
    try:
        return _json.loads(raw)
    except Exception:
        return raw  # not JSON → return as string


def _print_report(r: Dict[str, Any]) -> None:
    print(f"script        : {r['script']}")
    print(f"class         : {r['class']}  (loaded by name / auto-discovered)")
    print(f"skip_compute  : {r['skip_compute']}")
    print("agent         : " + (r["agent"] if r["agent"] else "— (compute skipped)"))
    print(f"ok            : {r['ok']}")
    print(f"latency       : {r['latency_s']}s")
    if r.get("usage"):
        print(f"usage         : {r['usage']}")
    print("-- result --")
    print(_json.dumps(r["result"], ensure_ascii=False, indent=2) if not isinstance(r["result"], str) else r["result"])


def main() -> int:
    ap = argparse.ArgumentParser(
        prog="run_edge",
        description="Drive one script edge (file.py:ClassName) standalone.",
    )
    ap.add_argument("--dir", default=".", help="Base dir for the script path (default: CWD)")
    ap.add_argument("--script", required=True, help="file.py or file.py:ClassName")
    ap.add_argument("--data", default=None, help="JSON payload (or raw string)")
    ap.add_argument("--skip-compute", action="store_true",
                    help="Skip the compute stage — pre_process output flows "
                         "straight to post_process (offline, generic edge).")
    ap.add_argument("--base-url", default=None, help="Full /chat/completions URL (real LLM)")
    ap.add_argument("--api-key", default=None, help="API key (real LLM)")
    args = ap.parse_args()

    if args.base_url and args.skip_compute:
        print("ERROR: --skip-compute and --base-url are mutually exclusive", file=sys.stderr)
        return 2
    if not args.base_url and not args.skip_compute:
        print(
            "ERROR: either give --base-url + --api-key (real LLM), or "
            "--skip-compute (offline, generic edge); no mock fallback",
            file=sys.stderr,
        )
        return 2
    if args.base_url and not args.api_key:
        print("ERROR: --api-key required for a real LLM call", file=sys.stderr)
        return 2

    report = asyncio.run(run_edge(
        dir_path=args.dir,
        script=args.script,
        data=_load_data(args.data),
        skip_compute=args.skip_compute,
        base_url=args.base_url,
        api_key=args.api_key,
    ))
    _print_report(report)
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())