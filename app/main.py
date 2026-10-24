"""HTTP layer. Business logic lives in engine.py; this file only parses and routes."""

import json
import os
import threading

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from starlette.concurrency import run_in_threadpool

from . import engine
from .planner import LLMPlanner
from .store import Store

MAX_REQUEST_BYTES = 768 * 1024


def create_app(planner=None, data_dir=None) -> FastAPI:
    app = FastAPI(docs_url=None, redoc_url=None, openapi_url=None)
    store = Store(data_dir or os.environ.get("DATA_DIR", "/tmp/ga5-data"))
    active_planner = planner or LLMPlanner.from_env()
    lock = threading.RLock()

    @app.exception_handler(engine.ApiError)
    async def _api_error(_request, exc: engine.ApiError):
        # Error bodies carry a fixed code only. They never echo request content.
        return JSONResponse(status_code=exc.status, content={"error": exc.code})

    async def _read_body(request: Request):
        raw = await request.body()
        if len(raw) > MAX_REQUEST_BYTES:
            raise engine.ApiError(413, "body_too_large")
        try:
            return json.loads(raw)
        except ValueError:
            raise engine.ApiError(400, "invalid_json")

    def _locked(fn, *args):
        with lock:
            return fn(*args)

    @app.get("/healthz")
    def healthz():
        return {"ok": True}

    @app.post("/v2/incidents")
    async def create_incident(request: Request):
        body = await _read_body(request)
        return await run_in_threadpool(
            _locked,
            engine.create_run,
            store,
            active_planner,
            body,
            request.headers.get("traceparent"),
            request.headers.get("tracestate"),
        )

    @app.post("/v2/incidents/{run_id}/receipts")
    async def post_receipt(run_id: str, request: Request):
        body = await _read_body(request)
        return await run_in_threadpool(_locked, engine.apply_receipt, store, run_id, body)

    @app.get("/v2/incidents/{run_id}")
    async def get_incident(run_id: str):
        return await run_in_threadpool(_locked, engine.get_current, store, run_id)

    return app


app = create_app()
