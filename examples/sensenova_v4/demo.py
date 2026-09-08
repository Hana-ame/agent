"""SenseNova V4 Edge Demo — direct connection, no proxy.

Exercises the full V4 handshake, execution, and state-transition pipeline
against the SenseNova 6.8 Flash Lite endpoint.

Prerequisites::

    export SENSENOVA_API_KEY=sk-xxxxx

Run::

    python3 examples/sensenova_v4/demo.py

The script creates two vertices (user_input → sensenova_output), wires
them with a ``SenseNovaEdgeV4``, and reports the result.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys

from framework.edge_v4 import EdgeResultV4
from framework.sensenova_edge_v4 import SensenovaEdgeV4
from framework.vertex_v4 import (
    VertexAttributeV4,
    VertexStateV4,
    VertexStoreV4,
)


async def main() -> None:
    session_id = "sensenova_v4_demo"

    store = VertexStoreV4(":memory:")

    # ---- Create vertices ----
    store.save_vertex(
        session_id=session_id,
        name="user_input",
        content="Please write a short haiku about artificial intelligence.",
        attributes=[VertexAttributeV4.START, VertexAttributeV4.PLAIN_TEXT],
        state=VertexStateV4.DATA_READY,
    )
    store.save_vertex(
        session_id=session_id,
        name="sensenova_output",
        content="",
        attributes=[VertexAttributeV4.PLAIN_TEXT],
        state=VertexStateV4.TODO,
    )

    # ---- Build the edge ----
    edge = SensenovaEdgeV4(
        edge_id="e_sensenova",
        input_vertex="user_input",
        output_vertex="sensenova_output",
        settings={
            "prompt": "You are a creative poet.",
            "model": "sensenova-6.8-flash-lite",
            "temperature": 0.7,
        },
    )

    print(f"[SenseNova V4 Edge]  endpoint={edge.base_url}")
    print(f"                      model={edge.model}")
    print(f"                      session={session_id}")
    print()

    # ---- Execute ----
    try:
        result = await edge.run(session_id, store)
    finally:
        await edge.close_agent()
        store.close()

    # ---- Report ----
    out_v = None
    if result.success:
        print("✅ SUCCESS")
        print(f"  edge:     {result.edge_id}")
        print(f"  output:   {result.output[:200]}…")
        if "usage" in result.metadata:
            u = result.metadata["usage"]
            print(f"  tokens:   prompt={u.get('prompt_tokens', 0)}  "
                  f"completion={u.get('completion_tokens', 0)}  "
                  f"total={u.get('total_tokens', 0)}")
    else:
        print("❌ FAILED")
        print(f"  edge:     {result.edge_id}")
        print(f"  error:    {result.error}")
        print(f"  reason:   {result.reason}")

    sys.exit(0 if result.success else 1)


if __name__ == "__main__":
    asyncio.run(main())
