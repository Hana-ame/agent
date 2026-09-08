"""SenseNova 6.8 Flash Lite — V4 dedicated edge.

A V4 ``EdgeV4`` subclass purpose-built for the SenseNova free-tier inference
endpoint. Unlike the generic :class:`LLMEdgeV4`, this edge:

* **Owns its own** :class:`HttpLLMAgent` wired to the SenseNova base URL
  (``https://sensenova.moonchan.xyz/v1/chat/completions``), so no external
  agent injection is needed.
* **No API key required** — the free SenseNova endpoint is publicly
  accessible without authentication. The ``Authorization`` header is omitted
  when ``SENSENOVA_API_KEY`` is absent.
* **Defaults the model** to ``sensenova-6.8-flash-lite``; override via the
  ``model`` argument or ``settings["model"]``.
* **Supports SSE streaming** via ``run_stream()`` for long-form generation.
* **Validates JSON output** when the downstream vertex carries the
  ``VertexAttributeV4.JSON`` attribute.
* Fully complies with the V4 **two-sided handshake** protocol, SQLite staging,
  and reject-state error recovery.

Usage (in-framework)::

    from framework.sensenova_edge_v4 import SensenovaEdgeV4

    edge = SensenovaEdgeV4(
        edge_id="e_sensenova",
        input_vertex="user_input",
        output_vertex="sensenova_output",
        settings={"prompt": "You are a helpful assistant.", "temperature": 0.7},
    )

Usage (config.json — loaded by ``DiscreteGraphLoaderV4``)::

    {
      "type": "sensenova",
      "script": "sensenova_edge_v4.py:SenseNovaEdgeV4",
      "settings": {"prompt": "…", "model": "sensenova-6.8-flash-lite"}
    }

Standalone CLI::

    python3 -m framework.sensenova_edge_v4 \\
        --db workflow.db --session my_session \\
        --edge-id e_sensenova --input user_input --output sensenova_output \\
        --prompt "Summarise the input in one sentence."

Environment::

    export SENSENOVA_API_KEY=sk-xxxxx   # optional (free tier needs no key)
"""

from __future__ import annotations

import argparse
import asyncio
import inspect
import json
import logging
import os
import sys
from pathlib import Path  # noqa: F401 — used by EdgeV4.run_standalone signature
from typing import Any, AsyncGenerator, Dict, Optional, Union  # noqa: F401 — Union used by base class

from framework.edge_v4 import EdgeResultV4, EdgeV4
from framework.vertex_v4 import (
    VertexAttributeV4,
    VertexRecordV4,
    VertexStateV4,
    VertexStoreV4,
)

logger = logging.getLogger("vertex_edge_agent.sensenova_edge_v4")

#: Default endpoint for the SenseNova free-tier chat-completions API.
DEFAULT_BASE_URL = "https://sensenova.moonchan.xyz/v1/chat/completions"

#: Default model identifier.
DEFAULT_MODEL = "sensenova-6.8-flash-lite"

#: Environment variable name for the API key.
API_KEY_ENV = "SENSENOVA_API_KEY"


class SensenovaEdgeV4(EdgeV4):
    """V4 edge dedicated to SenseNova 6.8 Flash Lite inference.

    Parameters
    ----------
    edge_id:
        Unique identifier for this edge within the graph.
    input_vertex:
        Name of the upstream vertex that must be ``data ready``.
    output_vertex:
        Name of the downstream vertex that must be ``todo`` or
        ``todo urgent``.
    model:
        Model identifier passed to the SenseNova API.
        Defaults to ``sensenova-6.8-flash-lite``.
    prompt_template:
        Optional system-prompt template. ``{input}`` is replaced with the
        upstream vertex content at runtime.
    settings:
        Arbitrary key/value dict forwarded to the agent and used for
        parameters such as ``temperature``, ``top_p``, ``max_tokens``,
        ``stream``, ``json_mode``, ``proxy``, ``base_url``, ``model``,
        ``prompt``, etc.

    Raises
    ------
    None
        The edge always constructs successfully. The free SenseNova endpoint
        does not require an API key; the ``Authorization`` header is simply
        omitted when ``SENSENOVA_API_KEY`` is unset.
    """

    def __init__(
        self,
        edge_id: str,
        input_vertex: str,
        output_vertex: str,
        model: str = DEFAULT_MODEL,
        prompt_template: Optional[str] = None,
        settings: Optional[Dict[str, Any]] = None,
    ) -> None:
        edge_settings = dict(settings or {})
        edge_settings.setdefault("model", model)
        if prompt_template:
            edge_settings["prompt"] = prompt_template
        super().__init__(
            edge_id=edge_id,
            input_vertex=input_vertex,
            output_vertex=output_vertex,
            edge_type="sensenova",
            settings=edge_settings,
        )

        # ------------------------------------------------------------------
        # Resolve API key — optional. SenseNova free tier requires no key.
        # ------------------------------------------------------------------
        self.api_key = os.environ.get(API_KEY_ENV, "").strip()

        # Resolve base URL — settings override the default.
        self.base_url: str = self.settings.get("base_url", DEFAULT_BASE_URL)

        # Resolve model — settings override the default.
        self.model: str = self.settings.get("model", DEFAULT_MODEL)

        # Build the owned agent.  SenseNova is directly reachable — no
        # transport proxy needed by default, but ``settings["proxy"]`` is
        # honoured if provided. When ``api_key`` is empty the agent simply
        # omits the Authorization header (public endpoint).
        from framework.agents import HttpLLMAgent

        self._agent = HttpLLMAgent(
            base_url=self.base_url,
            api_key=self.api_key,
            max_retries=self.settings.get("max_retries", 3),
            timeout=float(self.settings.get("timeout", 300.0)),
            proxy=self.settings.get("proxy"),
            trust_env=bool(self.settings.get("trust_env", True)),
        )
        logger.debug(
            "[SenseNovaEdgeV4:%s] ready  base_url=%s  model=%s",
            self.id,
            self.base_url,
            self.model,
        )

    # ------------------------------------------------------------------
    # Agent lifecycle
    # ------------------------------------------------------------------

    async def close_agent(self) -> None:
        """Close the owned HTTP agent. Safe to call multiple times."""
        await self._agent.close()

    # ------------------------------------------------------------------
    # V4 execution
    # ------------------------------------------------------------------

    async def run(
        self,
        session_id: str,
        store: VertexStoreV4,
        agent: Optional[Any] = None,
        auto_transition: bool = True,
        **kwargs: Any,
    ) -> EdgeResultV4:
        """Execute the SenseNova edge following the V4 handshake contract.

        Parameters
        ----------
        session_id:
            The session namespace in the vertex store.
        store:
            The SQLite-backed vertex store.
        agent:
            Optional external agent override. When *None*, the edge uses its
            owned :class:`HttpLLMAgent` wired to SenseNova.
        auto_transition:
            Whether to transition the downstream vertex to DATA_READY upon completion.

        Returns
        -------
        EdgeResultV4
            Success or failure with staging-documented diagnostics.
        """
        satisfied, reason, in_v, out_v = self.check_handshake(session_id, store)
        if not satisfied or in_v is None or out_v is None:
            return EdgeResultV4(
                edge_id=self.id,
                success=False,
                skipped=True,
                reason=reason,
            )

        # Use the owned agent unless an external one is injected.
        active_agent = agent if agent is not None else self._agent

        prompt_template = self.settings.get("prompt", "{input}")
        rendered_prompt = prompt_template.replace("{input}", in_v.content)

        temperature = float(self.settings.get("temperature", 0.7))
        model = self.settings.get("model", self.model)

        try:
            # Stage intermediate prompt for observability.
            store.stage_output(
                session_id=session_id,
                edge_id=self.id,
                key="rendered_prompt",
                value=rendered_prompt,
                vertex_name=self.output_vertex,
            )

            # Invoke agent — try multiple call patterns for compatibility.
            res_future = self._invoke_agent(
                active_agent, in_v.content, rendered_prompt, model,
                temperature, self.settings,
            )

            if inspect.isawaitable(res_future):
                response = await res_future
            else:
                response = res_future

            output_str = str(response)

            # JSON validation when the downstream vertex expects it.
            output_str = self._validate_json_if_needed(output_str, out_v)

            # Success: update downstream content and transition to 'data ready' if auto_transition is True.
            if auto_transition:
                store.update_vertex_content(
                    session_id=session_id,
                    name=self.output_vertex,
                    content=output_str,
                    state=VertexStateV4.DATA_READY.value,
                    increment_count=True,
                )
            else:
                store.update_vertex_content(
                    session_id=session_id,
                    name=self.output_vertex,
                    content=output_str,
                )

            usage_raw = active_agent.get_usage_summary() if hasattr(active_agent, "get_usage_summary") else {}
            if inspect.isawaitable(usage_raw):
                usage_raw = await usage_raw
            usage = usage_raw if isinstance(usage_raw, dict) else {}

            return EdgeResultV4(
                edge_id=self.id,
                success=True,
                output=output_str,
                metadata={
                    "model": model,
                    "length": len(output_str),
                    "usage": usage,
                },
            )

        except Exception as exc:
            err_msg = str(exc)
            logger.error(
                "[SenseNovaEdgeV4:%s] Inference failed: %s", self.id, err_msg,
                exc_info=True,
            )
            store.stage_output(
                session_id=session_id,
                edge_id=self.id,
                key="error_feedback",
                value=err_msg,
                vertex_name=self.output_vertex,
                metadata={
                    "model": model,
                    "exception": type(exc).__name__,
                    "base_url": self.base_url,
                },
            )
            store.update_vertex_state(
                session_id, self.output_vertex, VertexStateV4.REJECT.value,
            )

            return EdgeResultV4(
                edge_id=self.id,
                success=False,
                error=err_msg,
                reason="SenseNova inference error",
            )

    async def run_stream(
        self,
        session_id: str,
        store: VertexStoreV4,
        agent: Optional[Any] = None,
        auto_transition: bool = True,
        **kwargs: Any,
    ) -> AsyncGenerator[str, None]:
        """Execute the edge with SSE streaming, yielding content deltas.

        Streams are not retried — a partial stream is not replayable. The
        full concatenated output is written to the downstream vertex only
        after all deltas are received.

        Yields
        ------
        str
            Individual content delta chunks from the model.

        Raises
        ------
        RuntimeError
            If the handshake is not satisfied (upstream not ready,
            downstream not demanded).
        """
        satisfied, reason, in_v, out_v = self.check_handshake(session_id, store)
        if not satisfied or in_v is None or out_v is None:
            raise RuntimeError(
                f"Handshake failed for {self.id}: {reason}"
            )

        active_agent = agent if agent is not None else self._agent

        prompt_template = self.settings.get("prompt", "{input}")
        rendered_prompt = prompt_template.replace("{input}", in_v.content)

        model = self.settings.get("model", self.model)
        temperature = float(self.settings.get("temperature", 0.7))

        store.stage_output(
            session_id=session_id,
            edge_id=self.id,
            key="rendered_prompt",
            value=rendered_prompt,
            vertex_name=self.output_vertex,
        )

        chunks: list[str] = []
        try:
            if hasattr(active_agent, "stream_process"):
                stream_settings = dict(self.settings)
                stream_settings["stream"] = True
                async for delta in active_agent.stream_process(
                    data=in_v.content,
                    prompt=rendered_prompt,
                    model=model,
                    settings=stream_settings,
                ):
                    chunks.append(delta)
                    yield delta
            else:
                # Fallback: non-streaming agent.
                res_future = self._invoke_agent(
                    active_agent, in_v.content, rendered_prompt, model,
                    temperature, self.settings,
                )
                if inspect.isawaitable(res_future):
                    response = await res_future
                else:
                    response = res_future
                chunks.append(str(response))
                yield str(response)

            output_str = "".join(chunks)
            output_str = self._validate_json_if_needed(output_str, out_v)

            if auto_transition:
                store.update_vertex_content(
                    session_id=session_id,
                    name=self.output_vertex,
                    content=output_str,
                    state=VertexStateV4.DATA_READY.value,
                    increment_count=True,
                )
            else:
                store.update_vertex_content(
                    session_id=session_id,
                    name=self.output_vertex,
                    content=output_str,
                )

        except Exception as exc:
            err_msg = str(exc)
            logger.error(
                "[SenseNovaEdgeV4:%s] Stream failed: %s", self.id, err_msg,
                exc_info=True,
            )
            partial = "".join(chunks)
            if partial:
                logger.warning(
                    "[SenseNovaEdgeV4:%s] Staging partial output (%d chars)",
                    self.id, len(partial),
                )
            store.stage_output(
                session_id=session_id,
                edge_id=self.id,
                key="error_feedback",
                value=err_msg,
                vertex_name=self.output_vertex,
                metadata={
                    "model": model,
                    "exception": type(exc).__name__,
                    "partial_chars": len(partial),
                },
            )
            store.update_vertex_state(
                session_id, self.output_vertex, VertexStateV4.REJECT.value,
            )
            raise

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _invoke_agent(
        agent: Any,
        data: Any,
        prompt: str,
        model: str,
        temperature: float,
        settings: Optional[Dict[str, Any]],
    ) -> Any:
        """Invoke an agent via the best available call pattern.

        Supports ``process()``, ``chat()``, ``generate()``, and plain
        ``callable()`` — matching the dispatch logic in
        :class:`LLMEdgeV4` for consistency.
        """
        if hasattr(agent, "process") and callable(agent.process):
            return agent.process(
                data, prompt, model=model, settings=settings,
            )
        if hasattr(agent, "chat") and callable(agent.chat):
            return agent.chat(
                [{"role": "user", "content": prompt}],
                model=model,
                temperature=temperature,
            )
        if hasattr(agent, "generate") and callable(agent.generate):
            return agent.generate(prompt, model=model, temperature=temperature)
        if callable(agent):
            return agent(prompt)
        raise TypeError(
            f"Agent {type(agent).__name__} has no callable "
            "process/chat/generate method"
        )

    @staticmethod
    def _validate_json_if_needed(
        output_str: str,
        out_v: VertexRecordV4,
    ) -> str:
        """Strip markdown fences and validate JSON when the downstream
        vertex carries the ``JSON`` attribute.
        """
        if not out_v.has_attribute(VertexAttributeV4.JSON):
            return output_str

        clean = output_str.strip()
        if clean.startswith("```json"):
            clean = clean[7:]
        elif clean.startswith("```"):
            clean = clean[3:]
        if clean.endswith("```"):
            clean = clean[:-3]

        try:
            json.loads(clean.strip())
        except Exception as json_err:
            raise ValueError(f"Malformed JSON output: {json_err}") from json_err
        return clean.strip()


# ------------------------------------------------------------------
# Standalone CLI Entrypoint
# ------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    """Parse standalone command-line arguments."""
    parser = argparse.ArgumentParser(
        description="SenseNova 6.8 Flash Lite — V4 standalone edge runner",
    )
    parser.add_argument("--db", default=":memory:", help="SQLite database path")
    parser.add_argument("--session", required=True, help="Session identifier")
    parser.add_argument("--edge-id", default="e_sensenova", help="Edge identifier")
    parser.add_argument("--input", required=True, help="Input vertex name")
    parser.add_argument("--output", required=True, help="Output vertex name")
    parser.add_argument(
        "--model", default=DEFAULT_MODEL,
        help=f"Model name (default: {DEFAULT_MODEL})",
    )
    parser.add_argument("--prompt", default="{input}", help="Prompt template")
    parser.add_argument("--settings", default="{}", help="JSON settings string")
    parser.add_argument(
        "--stream", action="store_true",
        help="Enable SSE streaming mode",
    )
    return parser.parse_args()


async def _cli_main() -> None:
    """Standalone CLI runner with optional streaming."""
    args = parse_args()
    try:
        settings_dict: Dict[str, Any] = json.loads(args.settings)
    except Exception:
        settings_dict = {}

    edge = SensenovaEdgeV4(
        edge_id=args.edge_id,
        input_vertex=args.input,
        output_vertex=args.output,
        model=args.model,
        prompt_template=args.prompt,
        settings=settings_dict,
    )

    store = VertexStoreV4(args.db)

    try:
        if args.stream:
            chunks: list[str] = []
            async for delta in edge.run_stream(args.session, store):
                sys.stdout.write(delta)
                sys.stdout.flush()
                chunks.append(delta)
            print()
            print(json.dumps({"success": True, "output": "".join(chunks)}, indent=2))
        else:
            result = await edge.run(args.session, store)
            sys.stdout.write(json.dumps(result.to_dict(), indent=2) + "\n")
            if not result.success and not result.skipped:
                sys.exit(1)
    finally:
        await edge.close_agent()
        store.close()


#: CamelCase alias for SenseNovaEdgeV4
SenseNovaEdgeV4 = SensenovaEdgeV4


def main() -> None:
    """Entry point for ``python -m framework.sensenova_edge_v4``."""
    asyncio.run(_cli_main())


if __name__ == "__main__":
    main()

