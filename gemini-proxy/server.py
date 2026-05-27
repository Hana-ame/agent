"""
Google Gemini OpenAI-Compatible Proxy
- Transparently forwards OpenAI-format requests → Google AI Studio
- Extracts Gemma <thought> tags into reasoning_content so OpenCode can display them
"""

import json
import logging
import os
import re
import sys
import traceback
from pathlib import Path

import httpx
from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

# 加载 .env 文件（如存在）
_ENV_FILE = Path(__file__).resolve().parent.parent / ".env"
if _ENV_FILE.is_file():
    with open(_ENV_FILE) as _f:
        for _line in _f:
            _line = _line.strip()
            if _line and not _line.startswith("#") and "=" in _line:
                _k, _v = _line.split("=", 1)
                os.environ.setdefault(_k.strip(), _v.strip())

# ── logging ──────────────────────────────────────────────────────────
LOG = logging.getLogger("gemini-proxy")
LOG.setLevel(logging.DEBUG)
_h = logging.StreamHandler(sys.stderr)
_h.setFormatter(logging.Formatter("%(asctime)s  %(levelname)-7s  %(message)s"))
LOG.addHandler(_h)

API_KEY = os.environ.get(
    "GEMINI_API_KEY",
    "AIzaSyCdoEmNPRDQ-Gul29JNDJ9TyBCLzw2uFaM",
)
GOOGLE_BASE = "https://generativelanguage.googleapis.com/v1beta/openai"
STREAM_TIMEOUT = httpx.Timeout(600.0, connect=10.0, read=None)
NO_STREAM_TIMEOUT = httpx.Timeout(120.0, connect=10.0)
BUFFERED_MODE = os.environ.get("BUFFERED_MODE", "1").strip() == "1"

app = FastAPI(title="Gemini Proxy")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── thought → reasoning_content transformation ───────────────────────

THOUGHT_RE = re.compile(r"<thought>(.*?)</thought>", re.DOTALL)


def transform_nonstream(obj: dict) -> dict:
    """Extract <thought> from content → reasoning_content (non-streaming)."""
    for choice in obj.get("choices", []):
        msg = choice.get("message", {})
        content = msg.get("content", "")
        if not content:
            continue
        m = THOUGHT_RE.search(content)
        if m:
            msg["reasoning_content"] = m.group(1)
            msg["content"] = THOUGHT_RE.sub("", content).strip()
        msg.pop("extra_content", None)
    return obj


def transform_stream_chunk(line: str, buf: dict) -> str | None:
    """Transform one SSE `data: {...}` line. Returns transformed line or None if skip.

    buf tracks state across chunks: {"in_thought": bool}
    """
    if not line.startswith("data: "):
        return line
    if line.strip() == "data: [DONE]":
        return line

    try:
        obj = json.loads(line[6:])
    except json.JSONDecodeError:
        return line

    for choice in obj.get("choices", []):
        delta = choice.get("delta", {})
        if not delta:
            continue
        delta.pop("extra_content", None)
        content = delta.get("content", "")

        if buf.get("in_thought"):
            # we are inside a thought block
            if "</thought>" in content:
                parts = content.split("</thought>", 1)
                delta["reasoning_content"] = parts[0]
                delta["content"] = parts[1].strip() if len(parts) > 1 else ""
                buf["in_thought"] = False
            else:
                delta["reasoning_content"] = content
                delta["content"] = ""
        elif "<thought>" in content:
            if "</thought>" in content:
                # entire thought in one chunk
                before, rest = content.split("<thought>", 1)
                thought, after = rest.split("</thought>", 1)
                delta["reasoning_content"] = thought
                delta["content"] = (after.strip()) if after else ""
                if before.strip():
                    delta["content"] = before.strip() + (" " + delta["content"] if delta["content"] else "")
            else:
                # thought starts but doesn't end in this chunk
                parts = content.split("<thought>", 1)
                before = parts[0].strip()
                thought_start = parts[1]
                if before:
                    delta["content"] = before
                else:
                    delta["content"] = ""
                delta["reasoning_content"] = thought_start
                buf["in_thought"] = True

    return "data: " + json.dumps(obj, ensure_ascii=False)


async def sse_transform(response: httpx.Response, buf: dict):
    """Stream SSE response, transforming each chunk.
    Uses byte-level buffering so multi-byte UTF-8 chars are never split.
    Any exception is caught and logged so the stream never dies silently.
    """
    leftover = b""
    chunk_count = 0
    try:
        async for chunk in response.aiter_bytes():
            chunk_count += 1
            data = leftover + chunk
            parts = data.split(b"\n")
            leftover = parts.pop(-1)
            for part in parts:
                line_bytes = part.rstrip(b"\r")
                if not line_bytes:
                    yield "\n"
                    continue
                try:
                    line = line_bytes.decode("utf-8")
                    transformed = transform_stream_chunk(line, buf)
                    if transformed is not None:
                        yield transformed + "\n"
                except Exception:
                    LOG.warning("SSE transform error for line: %s", line_bytes[:200], exc_info=True)
    except Exception:
        LOG.error("SSE stream died after %d chunks", chunk_count, exc_info=True)


# ── buffered streaming: non-stream to Google, fake SSE to client ─────

async def buffered_sse_generator(obj: dict):
    """Convert a complete non-streaming response into SSE chunks."""
    import time

    model = obj.get("model", "gemma-4-31b-it")
    created = int(time.time())
    msg_id = obj.get("id", f"chatcmpl-{created}")

    for choice in obj.get("choices", []):
        msg = choice.get("message", {})
        reasoning = msg.get("reasoning_content", "")
        content = msg.get("content", "")

        # emit reasoning as delta chunks (split into ~100 char pieces)
        if reasoning:
            for i in range(0, len(reasoning), 80):
                piece = reasoning[i : i + 80]
                chunk = {
                    "id": msg_id,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": model,
                    "choices": [{
                        "index": 0,
                        "delta": {"role": "assistant", "reasoning_content": piece},
                    }],
                }
                yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"
                created += 1

        # emit content
        if content:
            chunk = {
                "id": msg_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": model,
                "choices": [{
                    "index": 0,
                    "delta": {"role": "assistant", "content": content},
                }],
            }
            yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"

    # finish
    finish_chunk = {
        "id": msg_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model,
        "choices": [{
            "index": 0,
            "delta": {},
            "finish_reason": "stop",
        }],
    }
    yield f"data: {json.dumps(finish_chunk, ensure_ascii=False)}\n\n"
    yield "data: [DONE]\n\n"


# ── routes ────────────────────────────────────────────────────────────

@app.get("/")
async def root():
    return {"status": "ok", "target": GOOGLE_BASE}


@app.get("/health")
async def health():
    return {"status": "healthy"}


@app.api_route(
    "/v1/{rest:path}",
    methods=["GET", "POST", "PUT", "DELETE", "PATCH", "OPTIONS"],
)
async def proxy(request: Request, rest: str):
    target_url = f"{GOOGLE_BASE}/{rest}"
    if request.url.query:
        target_url += f"?{request.url.query}"

    body = await request.body()
    body_json = None
    is_stream = False
    try:
        body_json = json.loads(body)
        is_stream = body_json.get("stream", False)
        # override max output to 256k
        body_json["max_tokens"] = 256000
        body_json.pop("max_completion_tokens", None)
        # strip thinking params that Gemma models don't support
        for k in ("reasoning_effort", "thinking", "thinking_level"):
            body_json.pop(k, None)
        body = json.dumps(body_json, ensure_ascii=False).encode()
    except Exception:
        pass

    # ── log incoming ──────────────────────────────────────────────────
    LOG.info(
        "<<< %s %s  client=%s  len=%d  stream=%s",
        request.method, target_url,
        request.client.host if request.client else "?",
        len(body), is_stream,
    )
    if body:
        LOG.debug("<<< body: %s", body[:3000].decode(errors="replace"))

    headers = {
        "authorization": f"Bearer {API_KEY}",
        "content-type": request.headers.get("content-type", "application/json"),
    }

    timeout = STREAM_TIMEOUT if is_stream else NO_STREAM_TIMEOUT

    # 流式场景下不进入 async with，避免 client 在 StreamingResponse
    # 消费 r.aiter_bytes() 之前就被关闭。
    client = httpx.AsyncClient(timeout=timeout)
    try:
        r = await client.send(
            client.build_request(
                method=request.method,
                url=target_url,
                headers=headers,
                content=body,
            ),
            stream=is_stream,
        )
    except Exception as exc:
        LOG.error("GOOGLE REQUEST FAILED: %s", exc)
        await client.aclose()
        return Response(
            content=json.dumps({"error": {"message": str(exc), "type": type(exc).__name__}}),
            status_code=502,
            media_type="application/json",
        )

    # ── buffered streaming: non-stream to Google, SSE to client ─────
    if BUFFERED_MODE and is_stream and body_json is not None and rest == "chat/completions":
        body_json["stream"] = False
        body_ns = json.dumps(body_json, ensure_ascii=False).encode()

        try:
            async with httpx.AsyncClient(timeout=NO_STREAM_TIMEOUT) as client:
                r_buf = await client.send(
                    client.build_request(
                        method=request.method,
                        url=target_url,
                        headers=headers,
                        content=body_ns,
                    ),
                    stream=False,
                )
        except Exception as exc:
            LOG.error("BUFFERED REQUEST FAILED: %s", exc)
            return Response(
                content=json.dumps({"error": {"message": str(exc), "type": type(exc).__name__}}),
                status_code=502,
                media_type="application/json",
            )

        raw = r_buf.content
        LOG.info(">>> Google (buffered) → %d  len=%d", r_buf.status_code, len(raw))

        if r_buf.status_code == 200:
            try:
                obj = json.loads(raw)
                obj = transform_nonstream(obj)
                return StreamingResponse(
                    buffered_sse_generator(obj),
                    status_code=200,
                    headers={"content-type": "text/event-stream"},
                )
            except Exception:
                LOG.error("BUFFERED transform failed", exc_info=True)
                return Response(content=raw, status_code=200, media_type="application/json")
        else:
            LOG.error(">>> ERROR: %s", raw[:1000].decode(errors="replace"))
            return Response(content=raw, status_code=r_buf.status_code, media_type="application/json")

    # ── streaming response ───────────────────────────────────────────
    if is_stream:
        if r.status_code == 200:
            LOG.info(">>> Google → %d (streaming)", r.status_code)
            resp_headers = {
                k: v
                for k, v in r.headers.items()
                if k.lower()
                not in ("transfer-encoding", "content-encoding", "content-length")
            }
            resp_headers["content-type"] = "text/event-stream"

            async def sse_with_cleanup():
                try:
                    async for chunk in sse_transform(r, {"in_thought": False}):
                        yield chunk
                finally:
                    await client.aclose()

            return StreamingResponse(
                sse_with_cleanup(),
                status_code=200,
                headers=resp_headers,
            )
        else:
            await r.aread()
            raw = r.content
            await client.aclose()
            LOG.info(">>> Google → %d (stream err)  len=%d", r.status_code, len(raw))
            LOG.error(">>> ERROR: %s", raw[:1000].decode(errors="replace"))
            return Response(
                content=raw,
                status_code=r.status_code,
                media_type="application/json",
            )

    # ── non-streaming response ───────────────────────────────────────
    raw = r.content
    await client.aclose()
    LOG.info(">>> Google → %d  len=%d", r.status_code, len(raw))

    if r.status_code == 200 and b"<thought>" in raw:
        try:
            obj = json.loads(raw)
            obj = transform_nonstream(obj)
            raw = json.dumps(obj, ensure_ascii=False).encode()
            LOG.debug(">>> transformed non-stream: thought extracted")
        except Exception:
            pass

    if r.status_code >= 400:
        LOG.error(">>> ERROR: %s", raw[:1000].decode(errors="replace"))

    return Response(
        content=raw,
        status_code=r.status_code,
        headers={
            k: v
            for k, v in r.headers.items()
            if k.lower() not in ("transfer-encoding", "content-encoding", "content-length")
        },
    )


if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("PORT", "8317"))
    LOG.info("Starting Gemini Proxy on 0.0.0.0:%d", port)
    uvicorn.run(app, host="0.0.0.0", port=port, log_config=None)
