"""
Google Gemini OpenAI-Compatible Proxy
Transparently forwards OpenAI-format requests to Google AI Studio.
"""

import json
import logging
import os
import sys
import traceback

import httpx
from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware

# ── logging ──────────────────────────────────────────────────────────
LOG = logging.getLogger("gemini-proxy")
LOG.setLevel(logging.DEBUG)
_h = logging.StreamHandler(sys.stderr)
_h.setFormatter(logging.Formatter("%(asctime)s  %(levelname)-7s  %(message)s"))
LOG.addHandler(_h)

API_KEY = os.environ.get(
    "GEMINI_API_KEY",
    "AIzaSyAi_QKOU0VBSfF1SR6GRz6V_k8_Hd7RNQ4",
)
GOOGLE_BASE = "https://generativelanguage.googleapis.com/v1beta/openai"
TIMEOUT = httpx.Timeout(600.0, connect=10.0)

app = FastAPI(title="Gemini Proxy")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


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

    # ── log incoming request ─────────────────────────────────────────
    LOG.info(
        "<<< %s %s  client=%s  content_type=%s  body_len=%d",
        request.method, target_url,
        request.client.host if request.client else "?",
        request.headers.get("content-type", "-"),
        len(body),
    )
    body_preview = body[:2000]
    if body_preview:
        LOG.debug("<<< body preview: %s", body_preview.decode(errors="replace"))

    LOG.debug(
        "<<< headers: %s",
        {k: v for k, v in request.headers.items()},
    )

    headers = {
        "authorization": f"Bearer {API_KEY}",
        "content-type": request.headers.get("content-type", "application/json"),
    }

    try:
        async with httpx.AsyncClient(timeout=TIMEOUT) as client:
            r = await client.request(
                method=request.method,
                url=target_url,
                headers=headers,
                content=body,
            )
    except Exception as exc:
        LOG.error("GOOGLE REQUEST FAILED: %s", exc)
        LOG.debug("traceback: %s", traceback.format_exc())
        return Response(
            content=json.dumps({"error": {"message": str(exc), "type": type(exc).__name__}}),
            status_code=502,
            media_type="application/json",
        )

    # ── log response ─────────────────────────────────────────────────
    LOG.info(
        ">>> Google → status=%d  content_type=%s  body_len=%d",
        r.status_code,
        r.headers.get("content-type", "-"),
        len(r.content),
    )
    body_text = r.content.decode(errors="replace")[:2000]
    LOG.debug(">>> body preview: %s", body_text)

    if r.status_code >= 400:
        LOG.error(">>> Google ERROR (%d): %s", r.status_code, body_text[:500])

    return Response(
        content=r.content,
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
