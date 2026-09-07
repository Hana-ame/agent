"""V4 Standalone Edge Execution Engine.

Implements two-sided handshake forward edges and reflexive self-loop recovery edges.
Each edge executes against strictly two vertex references (input and output) within a session.
Supports both in-framework orchestration and standalone CLI/process execution.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple, Union

from framework.vertex_v4 import (
    VertexAttributeV4,
    VertexRecordV4,
    VertexStateV4,
    VertexStoreV4,
)

logger = logging.getLogger("vertex_edge_agent.edge_v4")


@dataclass
class EdgeResultV4:
    """Result of an EdgeV4 execution."""

    edge_id: str
    success: bool
    skipped: bool = False
    output: Any = None
    error: Optional[str] = None
    reason: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize result to dictionary."""
        return {
            "edge_id": self.edge_id,
            "success": self.success,
            "skipped": self.skipped,
            "output": self.output,
            "error": self.error,
            "reason": self.reason,
            "metadata": self.metadata,
        }


class EdgeV4:
    """Base class for V4 executable edges."""

    def __init__(
        self,
        edge_id: str,
        input_vertex: str,
        output_vertex: str,
        edge_type: str = "code",
        settings: Optional[Dict[str, Any]] = None,
    ):
        self.id = edge_id
        self.input_vertex = str(input_vertex)
        self.output_vertex = str(output_vertex)
        self.type = edge_type
        self.settings = dict(settings or {})

    @property
    def is_reflexive(self) -> bool:
        """Return True if this edge connects a vertex to itself."""
        return self.input_vertex == self.output_vertex

    def check_handshake(
        self,
        session_id: str,
        store: VertexStoreV4,
    ) -> Tuple[bool, str, Optional[VertexRecordV4], Optional[VertexRecordV4]]:
        """Evaluate two-sided handshake contract against current vertex states.

        Returns:
            Tuple of (is_satisfied, reason, input_vertex_record, output_vertex_record)
        """
        in_v = store.get_vertex(session_id, self.input_vertex)
        if in_v is None:
            return False, f"Input vertex '{self.input_vertex}' not found in session '{session_id}'", None, None

        if self.is_reflexive:
            # Reflexive edge inspects a single vertex
            return True, "Reflexive edge target located", in_v, in_v

        out_v = store.get_vertex(session_id, self.output_vertex)
        if out_v is None:
            return False, f"Output vertex '{self.output_vertex}' not found in session '{session_id}'", in_v, None

        # Two-sided handshake requirement:
        # Upstream must be 'data ready' AND Downstream must be in ('todo', 'todo urgent')
        upstream_ready = in_v.state == VertexStateV4.DATA_READY.value
        downstream_demanded = out_v.state in (
            VertexStateV4.TODO.value,
            VertexStateV4.TODO_URGENT.value,
        )

        if not upstream_ready:
            return (
                False,
                f"Upstream '{self.input_vertex}' state is '{in_v.state}', required '{VertexStateV4.DATA_READY.value}'",
                in_v,
                out_v,
            )

        if not downstream_demanded:
            return (
                False,
                f"Downstream '{self.output_vertex}' state is '{out_v.state}', required 'todo' or 'todo urgent'",
                in_v,
                out_v,
            )

        return True, "Handshake satisfied", in_v, out_v

    async def run(
        self,
        session_id: str,
        store: VertexStoreV4,
        agent: Optional[Any] = None,
    ) -> EdgeResultV4:
        """Execute the edge against the storage store. Must be overridden by subclasses."""
        raise NotImplementedError("Subclasses must implement run()")

    async def run_standalone(
        self,
        session_id: str,
        store_or_db_path: Union[VertexStoreV4, str, Path],
        agent: Optional[Any] = None,
    ) -> EdgeResultV4:
        """Execute this edge as an autonomous standalone unit."""
        should_close = False
        if isinstance(store_or_db_path, VertexStoreV4):
            store = store_or_db_path
        else:
            store = VertexStoreV4(str(store_or_db_path))
            should_close = True

        try:
            return await self.run(session_id=session_id, store=store, agent=agent)
        finally:
            if should_close:
                store.close()


class CodeEdgeV4(EdgeV4):
    """Executes Python transformation logic or scripts between vertices."""

    def __init__(
        self,
        edge_id: str,
        input_vertex: str,
        output_vertex: str,
        script: Optional[Union[str, Callable]] = None,
        settings: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(
            edge_id=edge_id,
            input_vertex=input_vertex,
            output_vertex=output_vertex,
            edge_type="code",
            settings=settings,
        )
        self.script = script
        self._callable: Optional[Callable] = None
        if callable(script):
            self._callable = script

    def _resolve_callable(self) -> Callable:
        """Resolve script to callable function."""
        if self._callable is not None:
            return self._callable

        if isinstance(self.script, str):
            # Format: "path/to/script.py:function_name" or "path/to/script.py"
            parts = self.script.split(":")
            file_path = parts[0]
            func_name = parts[1] if len(parts) > 1 else None

            from framework.utils.script_loader import load_script
            mod = load_script(file_path)
            if func_name:
                fn = getattr(mod, func_name, None)
                if fn and callable(fn):
                    self._callable = fn
                    return fn
                raise AttributeError(f"Callable '{func_name}' not found in '{file_path}'")

            # Look for common function names
            for candidate in ("execute", "process", "run", "transform"):
                fn = getattr(mod, candidate, None)
                if fn and callable(fn):
                    self._callable = fn
                    return fn

            raise ValueError(f"No valid callable entrypoint found in '{file_path}'")

        # Default passthrough transformation
        return lambda content, settings, staging=None: content

    async def run(
        self,
        session_id: str,
        store: VertexStoreV4,
        agent: Optional[Any] = None,
    ) -> EdgeResultV4:
        """Execute code edge following the two-sided handshake contract."""
        satisfied, reason, in_v, out_v = self.check_handshake(session_id, store)
        if not satisfied or in_v is None or out_v is None:
            return EdgeResultV4(
                edge_id=self.id,
                success=False,
                skipped=True,
                reason=reason,
            )

        try:
            fn = self._resolve_callable()
            # Retrieve any latest staged records for context
            latest_staging = store.get_latest_staged_for_vertex(session_id, self.output_vertex)
            staging_dict = latest_staging.to_dict() if latest_staging else {}

            if asyncio.iscoroutinefunction(fn):
                res = await fn(in_v.content, self.settings, staging_dict)
            else:
                res = fn(in_v.content, self.settings, staging_dict)

            output_str = str(res) if res is not None else ""

            # Handshake fulfillment: Update downstream content, set to 'data ready', increment count
            store.update_vertex_content(
                session_id=session_id,
                name=self.output_vertex,
                content=output_str,
                state=VertexStateV4.DATA_READY.value,
                increment_count=True,
            )

            return EdgeResultV4(
                edge_id=self.id,
                success=True,
                output=output_str,
                metadata={"length": len(output_str)},
            )

        except Exception as exc:
            err_msg = str(exc)
            logger.error("[CodeEdgeV4:%s] Execution failed: %s", self.id, err_msg)
            # Stage failure diagnostic attributed to this edge
            store.stage_output(
                session_id=session_id,
                edge_id=self.id,
                key="error_feedback",
                value=err_msg,
                vertex_name=self.output_vertex,
                metadata={"exception_type": type(exc).__name__},
            )
            # Transition downstream state to 'reject'
            store.update_vertex_state(session_id, self.output_vertex, VertexStateV4.REJECT.value)

            return EdgeResultV4(
                edge_id=self.id,
                success=False,
                error=err_msg,
                reason="Execution threw exception",
            )


class LLMEdgeV4(EdgeV4):
    """Executes LLM inference between upstream input and downstream output."""

    def __init__(
        self,
        edge_id: str,
        input_vertex: str,
        output_vertex: str,
        model: str = "sensenova-6.8-flash-lite",
        prompt_template: Optional[str] = None,
        settings: Optional[Dict[str, Any]] = None,
    ):
        edge_settings = dict(settings or {})
        edge_settings["model"] = model
        if prompt_template:
            edge_settings["prompt"] = prompt_template
        super().__init__(
            edge_id=edge_id,
            input_vertex=input_vertex,
            output_vertex=output_vertex,
            edge_type="llm",
            settings=edge_settings,
        )

    async def run(
        self,
        session_id: str,
        store: VertexStoreV4,
        agent: Optional[Any] = None,
    ) -> EdgeResultV4:
        """Execute LLM edge with prompt construction and response delivery."""
        satisfied, reason, in_v, out_v = self.check_handshake(session_id, store)
        if not satisfied or in_v is None or out_v is None:
            return EdgeResultV4(
                edge_id=self.id,
                success=False,
                skipped=True,
                reason=reason,
            )

        if agent is None:
            from framework.agents import HttpLLMAgent
            agent = HttpLLMAgent()

        prompt_template = self.settings.get("prompt", "{input}")
        rendered_prompt = prompt_template.replace("{input}", in_v.content)

        model = self.settings.get("model", "sensenova-6.8-flash-lite")
        temperature = float(self.settings.get("temperature", 0.7))

        try:
            # Stage intermediate prompt draft
            store.stage_output(
                session_id=session_id,
                edge_id=self.id,
                key="rendered_prompt",
                value=rendered_prompt,
                vertex_name=self.output_vertex,
            )

            # Invoke agent - check for common method signatures
            if hasattr(agent, "chat") and asyncio.iscoroutinefunction(agent.chat):
                response = await agent.chat(
                    [{"role": "user", "content": rendered_prompt}],
                    model=model,
                    temperature=temperature,
                )
            elif hasattr(agent, "generate") and asyncio.iscoroutinefunction(agent.generate):
                response = await agent.generate(rendered_prompt, model=model, temperature=temperature)
            elif hasattr(agent, "process") and asyncio.iscoroutinefunction(agent.process):
                response = await agent.process(
                    in_v.content, rendered_prompt, model=model, settings=self.settings,
                )
            elif callable(agent):
                response = str(agent(rendered_prompt))
            else:
                raise TypeError(
                    f"Agent {type(agent).__name__} has no callable chat/generate/process method"
                )

            output_str = str(response)

            # If downstream vertex expects JSON, validate syntax
            if out_v.has_attribute(VertexAttributeV4.JSON):
                try:
                    # Strip markdown fence if present
                    clean_str = output_str.strip()
                    if clean_str.startswith("```json"):
                        clean_str = clean_str[7:]
                    elif clean_str.startswith("```"):
                        clean_str = clean_str[3:]
                    if clean_str.endswith("```"):
                        clean_str = clean_str[:-3]
                    json.loads(clean_str.strip())
                    output_str = clean_str.strip()
                except Exception as json_err:
                    raise ValueError(f"Malformed JSON output: {json_err}") from json_err

            # Success: update downstream content and transition to 'data ready'
            store.update_vertex_content(
                session_id=session_id,
                name=self.output_vertex,
                content=output_str,
                state=VertexStateV4.DATA_READY.value,
                increment_count=True,
            )

            return EdgeResultV4(
                edge_id=self.id,
                success=True,
                output=output_str,
                metadata={"model": model},
            )

        except Exception as exc:
            err_msg = str(exc)
            logger.error("[LLMEdgeV4:%s] Inference failed: %s", self.id, err_msg)
            store.stage_output(
                session_id=session_id,
                edge_id=self.id,
                key="error_feedback",
                value=err_msg,
                vertex_name=self.output_vertex,
                metadata={"model": model, "exception": type(exc).__name__},
            )
            store.update_vertex_state(session_id, self.output_vertex, VertexStateV4.REJECT.value)

            return EdgeResultV4(
                edge_id=self.id,
                success=False,
                error=err_msg,
                reason="LLM inference error",
            )


class ReflexiveEdgeV4(EdgeV4):
    """Reflexive recovery edge (input == output) designed for reset on reject."""

    def __init__(
        self,
        edge_id: str,
        vertex_name: str,
        trigger_state: str = VertexStateV4.REJECT.value,
        target_state: str = VertexStateV4.TODO_URGENT.value,
        max_retries: int = 3,
        script: Optional[Union[str, Callable]] = None,
        settings: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(
            edge_id=edge_id,
            input_vertex=vertex_name,
            output_vertex=vertex_name,
            edge_type="reflexive",
            settings=settings,
        )
        self.trigger_state = trigger_state
        self.target_state = target_state
        self.max_retries = int(self.settings.get("max_retries", max_retries))
        self.script = script
        self._callable: Optional[Callable] = None
        if callable(script):
            self._callable = script

    def _resolve_callable(self) -> Optional[Callable]:
        if self._callable is not None:
            return self._callable
        if isinstance(self.script, str):
            from framework.utils.script_loader import load_script
            parts = self.script.split(":")
            mod = load_script(parts[0])
            func_name = parts[1] if len(parts) > 1 else "recover"
            fn = getattr(mod, func_name, None)
            if fn and callable(fn):
                self._callable = fn
                return fn
        return None

    async def run(
        self,
        session_id: str,
        store: VertexStoreV4,
        agent: Optional[Any] = None,
    ) -> EdgeResultV4:
        """Execute reflexive recovery upon target vertex entering trigger state."""
        target_v = store.get_vertex(session_id, self.output_vertex)
        if target_v is None:
            return EdgeResultV4(
                edge_id=self.id,
                success=False,
                skipped=True,
                reason=f"Target vertex '{self.output_vertex}' not found",
            )

        if target_v.state != self.trigger_state:
            return EdgeResultV4(
                edge_id=self.id,
                success=False,
                skipped=True,
                reason=f"Target state is '{target_v.state}', waiting for '{self.trigger_state}'",
            )

        # Retrieve diagnostics from staging table
        latest_error = store.get_latest_staged_for_vertex(
            session_id=session_id,
            vertex_name=self.output_vertex,
            key="error_feedback",
        )
        error_info = latest_error.value if latest_error else "No staged error diagnostic"

        # Circuit breaker: Check if maximum iterations reached
        if target_v.processed_count >= self.max_retries:
            logger.warning(
                "[ReflexiveEdgeV4:%s] Max retries (%d) reached for vertex '%s'. Locking to 'forbidden'.",
                self.id,
                self.max_retries,
                self.output_vertex,
            )
            store.update_vertex_state(session_id, self.output_vertex, VertexStateV4.FORBIDDEN.value)
            store.stage_output(
                session_id=session_id,
                edge_id=self.id,
                key="retry_exhausted",
                value=f"Exceeded max retries: {self.max_retries}",
                vertex_name=self.output_vertex,
            )
            return EdgeResultV4(
                edge_id=self.id,
                success=False,
                error="Max retries exceeded",
                reason=f"Attempted {target_v.processed_count} retries out of {self.max_retries}",
            )

        # Apply optional recovery transformer
        new_content = target_v.content
        fn = self._resolve_callable()
        if fn:
            try:
                if asyncio.iscoroutinefunction(fn):
                    new_content = await fn(target_v.content, error_info, self.settings)
                else:
                    new_content = fn(target_v.content, error_info, self.settings)
            except Exception as rec_err:
                logger.error("[ReflexiveEdgeV4:%s] Recovery script failed: %s", self.id, rec_err)
                store.stage_output(
                    session_id=session_id,
                    edge_id=self.id,
                    key="error_feedback",
                    value=f"Recovery script failed: {rec_err}",
                    vertex_name=self.output_vertex,
                    metadata={"exception_type": type(rec_err).__name__},
                )
                return EdgeResultV4(
                    edge_id=self.id,
                    success=False,
                    error=f"Recovery script failed: {rec_err}",
                    reason="Recovery transform threw exception",
                )

        # Reset state back to target_state ('todo urgent' or 'todo') and increment processed_count
        store.update_vertex_content(
            session_id=session_id,
            name=self.output_vertex,
            content=str(new_content),
            state=self.target_state,
            increment_count=True,
        )

        store.stage_output(
            session_id=session_id,
            edge_id=self.id,
            key="reflexive_reset",
            value=f"Reset vertex to '{self.target_state}', attempt #{target_v.processed_count + 1}",
            vertex_name=self.output_vertex,
            metadata={"diagnostic": error_info},
        )

        return EdgeResultV4(
            edge_id=self.id,
            success=True,
            output=new_content,
            metadata={
                "attempt": target_v.processed_count + 1,
                "reset_to": self.target_state,
            },
        )


# ------------------------------------------------------------------
# Standalone CLI Entrypoint
# ------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    """Parse standalone command line arguments."""
    parser = argparse.ArgumentParser(description="Standalone V4 Edge Runner")
    parser.add_argument("--db", default=":memory:", help="SQLite database path")
    parser.add_argument("--session", required=True, help="Session identifier")
    parser.add_argument("--edge-id", default="standalone_edge", help="Edge identifier")
    parser.add_argument("--type", choices=["code", "llm", "reflexive"], default="code", help="Edge type")
    parser.add_argument("--input", required=True, help="Input vertex name")
    parser.add_argument("--output", required=True, help="Output vertex name")
    parser.add_argument("--script", default=None, help="Script path for code or recovery edge")
    parser.add_argument("--model", default="sensenova-6.8-flash-lite", help="Model name for LLM edge")
    parser.add_argument("--settings", default="{}", help="JSON settings string")
    return parser.parse_args()


def main() -> None:
    """Standalone runner entrypoint."""
    args = parse_args()
    try:
        settings_dict = json.loads(args.settings)
    except Exception:
        settings_dict = {}

    edge: EdgeV4
    if args.type == "code":
        edge = CodeEdgeV4(
            edge_id=args.edge_id,
            input_vertex=args.input,
            output_vertex=args.output,
            script=args.script,
            settings=settings_dict,
        )
    elif args.type == "llm":
        edge = LLMEdgeV4(
            edge_id=args.edge_id,
            input_vertex=args.input,
            output_vertex=args.output,
            model=args.model,
            settings=settings_dict,
        )
    elif args.type == "reflexive":
        edge = ReflexiveEdgeV4(
            edge_id=args.edge_id,
            vertex_name=args.output,
            script=args.script,
            settings=settings_dict,
        )
    else:
        sys.stderr.write(f"Unknown edge type: {args.type}\n")
        sys.exit(1)

    store = VertexStoreV4(args.db)
    result = asyncio.run(edge.run(args.session, store))
    sys.stdout.write(json.dumps(result.to_dict(), indent=2) + "\n")
    store.close()
    if not result.success and not result.skipped:
        sys.exit(1)


if __name__ == "__main__":
    main()
