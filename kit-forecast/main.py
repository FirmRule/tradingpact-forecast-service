"""KiT forecast service for the Desk chart: a LOCAL PREVIEW, not for deploy.

Same contract as apps/kronos (POST /forecast, GET /health, bearer token), so the
app uses it by pointing KRONOS_URL here. KiT is AGPL-3.0 (vendor/KiT/LICENSE):
offering it to traders over a network obliges publishing this service's source.

    KIT_API_TOKEN=... .venv/bin/python -I -m uvicorn main:app --host 127.0.0.1 --port 4007 --app-dir .
"""

from __future__ import annotations

import asyncio
import faulthandler
import hashlib
import signal
import hmac
import json
import logging
import os
import queue
import sys
import threading
import time
from collections import OrderedDict
from contextlib import asynccontextmanager
from typing import Annotated, Any

from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field, model_validator

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "kronos"))  # summarize/path_rows: the card's figures, model-agnostic
sys.path.insert(0, HERE)
from forecast import path_rows, summarize  # noqa: E402

from kit_engine import Cancelled, Candle, KitEngine, TooFewCandles, UnsupportedTimeframe  # noqa: E402

# `kill -USR1 <pid>` writes every thread's stack to the log, for a stuck forecast.
faulthandler.register(signal.SIGUSR1, all_threads=True)

log = logging.getLogger("kit")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

CKPT = os.getenv("KIT_CKPT", os.path.join(HERE, "ckpt", "step_market_284M.pt"))
DEVICE = os.getenv("KIT_DEVICE", "mps")
TOKEN = os.getenv("KIT_API_TOKEN", "")
MAX_BODY_BYTES = 1024 * 1024
CACHE_SIZE = 32

state: dict[str, Any] = {"engine": None}
inference_lock = threading.Lock()
cache: OrderedDict[str, dict[str, Any]] = OrderedDict()
cache_lock = threading.Lock()


@asynccontextmanager
async def lifespan(_: FastAPI):
    if not TOKEN:
        raise RuntimeError("Set KIT_API_TOKEN.")
    engine = KitEngine(CKPT, device=DEVICE)
    gpu = ""
    if DEVICE == "cuda":  # which GPU Modal gave this container: an L4, or the A10G fallback
        import torch  # already loaded by kit_engine

        gpu = f" ({torch.cuda.get_device_name(0)})"
    log.info("loaded KiT in %.1fs on %s%s", engine.load_seconds, DEVICE, gpu)
    state["engine"] = engine
    yield


app = FastAPI(title="TradingPact KiT preview", lifespan=lifespan, docs_url=None, redoc_url=None)


def _token_ok(authorization: str | None) -> bool:
    return bool(authorization) and hmac.compare_digest(authorization, f"Bearer {TOKEN}")


class GateMiddleware:
    """Token and body size checked before the body is read (as in apps/kronos)."""

    def __init__(self, inner):
        self.inner = inner

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http" or scope.get("path") == "/health":
            await self.inner(scope, receive, send)
            return
        headers = {k.decode("latin-1").lower(): v.decode("latin-1") for k, v in scope.get("headers", [])}
        if not _token_ok(headers.get("authorization")):
            await _plain(send, 401, b"Unauthorized")
            return
        declared = headers.get("content-length")
        try:
            too_big = declared is not None and int(declared) > MAX_BODY_BYTES
        except ValueError:
            too_big = True
        if too_big:
            await _plain(send, 413, b"Request too large")
            return
        await self.inner(scope, receive, send)


async def _plain(send, status: int, body: bytes) -> None:
    await send({"type": "http.response.start", "status": status,
                "headers": [(b"content-type", b"text/plain"), (b"content-length", str(len(body)).encode())]})
    await send({"type": "http.response.body", "body": body})


app.add_middleware(GateMiddleware)


def require_token(authorization: Annotated[str | None, Header()] = None) -> None:
    if not _token_ok(authorization):
        raise HTTPException(status_code=401, detail="Unauthorized")


class BarIn(BaseModel):
    time: int = Field(gt=0)
    open: float = Field(gt=0)
    high: float = Field(gt=0)
    low: float = Field(gt=0)
    close: float = Field(gt=0)
    volume: float | None = Field(default=None, ge=0)


class ForecastIn(BaseModel):
    bars: list[BarIn] = Field(min_length=64, max_length=2048)
    future_times: list[int] = Field(min_length=1, max_length=128)
    samples: int = Field(default=20, ge=2, le=64)
    # Accepted for the Kronos contract; KiT has no temperature or nucleus sampling.
    temperature: float = Field(default=1.0, gt=0, le=2)
    top_p: float = Field(default=0.95, gt=0, le=1)
    return_paths: bool = False

    @model_validator(mode="after")
    def check_times(self) -> ForecastIn:
        times = [bar.time for bar in self.bars]
        if any(later <= earlier for earlier, later in zip(times, times[1:])):
            raise ValueError("bars must be in time order without repeats")
        future = self.future_times
        if future[0] <= times[-1] or any(b <= a for a, b in zip(future, future[1:])):
            raise ValueError("future_times must follow the last bar, in order")
        return self


@app.get("/health")
def health() -> dict[str, Any]:
    return {"ok": state["engine"] is not None, "model": "KiT", "device": DEVICE}


@app.post("/forecast", dependencies=[Depends(require_token)])
def forecast(body: ForecastIn) -> dict[str, Any]:
    engine: KitEngine | None = state["engine"]
    if engine is None:
        raise HTTPException(status_code=503, detail="The model is still loading.")
    key = hashlib.sha256(json.dumps({"bars": [b.model_dump() for b in body.bars], "future": body.future_times,
                                     "samples": body.samples, "paths": body.return_paths},
                                    separators=(",", ":")).encode()).hexdigest()
    with cache_lock:
        if key in cache:
            cache.move_to_end(key)
            return {**cache[key], "cached": True}
    candles = [Candle(**b.model_dump()) for b in body.bars]
    started = time.time()
    try:
        with inference_lock:
            paths, info = engine.sample(candles, body.future_times, body.samples)
    except UnsupportedTimeframe as error:
        raise HTTPException(status_code=501, detail=str(error)) from error
    except TooFewCandles as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
    future = body.future_times[: info["horizon"]]
    result = {
        **summarize(paths, candles, future),
        **({"paths": path_rows(paths)} if body.return_paths else {}),
        "model": "KiT",
        "lookback": info["context"],
        "kit": info,
        "elapsedMs": int((time.time() - started) * 1000),
    }
    log.info("forecast %s ctx=%d steps=%d/%d samples=%d volume=%s in %dms", info["scale"], info["context"],
             info["horizon"], len(body.future_times), body.samples, info["volume"], result["elapsedMs"])
    with cache_lock:
        cache[key] = result
        while len(cache) > CACHE_SIZE:
            cache.popitem(last=False)
    return {**result, "cached": False}


@app.post("/forecast/stream", dependencies=[Depends(require_token)])
def forecast_stream(body: ForecastIn) -> StreamingResponse:
    """The same forecast as NDJSON: a draft of all paths every few refinement
    steps ({"kind": "draft", "step", "of", ...}), then {"kind": "final", ...},
    or {"kind": "error", "status", "detail"}. A cached forecast sends only its
    final line. When the caller goes away the forecast stops at its next step,
    so abandoned requests don't hold the model."""
    engine: KitEngine | None = state["engine"]
    if engine is None:
        raise HTTPException(status_code=503, detail="The model is still loading.")
    key = hashlib.sha256(json.dumps({"bars": [b.model_dump() for b in body.bars], "future": body.future_times,
                                     "samples": body.samples, "paths": body.return_paths},
                                    separators=(",", ":")).encode()).hexdigest()
    with cache_lock:
        hit = cache.get(key)
    if hit is not None:
        line = json.dumps({"kind": "final", **hit, "cached": True}) + "\n"
        return StreamingResponse(iter([line]), media_type="application/x-ndjson")

    candles = [Candle(**b.model_dump()) for b in body.bars]
    events: queue.Queue = queue.Queue()
    gone = threading.Event()
    started = time.time()
    try:
        scale, _ = engine.scale_for(candles)
        context = min(engine.cfg.data.window_for(scale)[0], len(candles) - 1)
    except UnsupportedTimeframe:
        context = 0  # work() reports it as the error line

    def figures(paths, lookback: int) -> dict[str, Any]:
        future = body.future_times[: paths.shape[1]]
        return {**summarize(paths, candles, future), **({"paths": path_rows(paths)} if body.return_paths else {}),
                "model": "KiT", "lookback": lookback, "elapsedMs": int((time.time() - started) * 1000)}

    def work() -> None:
        try:
            while not inference_lock.acquire(timeout=0.5):
                if gone.is_set():
                    return
            try:
                paths, info = engine.sample(
                    candles, body.future_times, body.samples,
                    on_draft=lambda step, of, p: events.put(("draft", step, of, p)),
                    should_stop=gone.is_set)
            finally:
                inference_lock.release()
            events.put(("final", paths, info))
        except Cancelled:
            log.info("stream forecast stopped: the caller went away")
        except UnsupportedTimeframe as error:
            events.put(("error", 501, str(error)))
        except TooFewCandles as error:
            events.put(("error", 422, str(error)))
        except Exception as error:  # noqa: BLE001 - reported to the caller as a failed forecast
            log.exception("stream forecast failed")
            events.put(("error", 500, type(error).__name__))
        finally:
            events.put(("end",))

    def next_event():
        # Wakes every half second so a gone caller is noticed.
        while not gone.is_set():
            try:
                return events.get(timeout=0.5)
            except queue.Empty:
                continue
        return ("end",)

    async def lines():
        threading.Thread(target=work, daemon=True).start()
        try:
            while True:
                event = await asyncio.to_thread(next_event)
                kind = event[0]
                if kind == "draft":
                    _, step, of, paths = event
                    yield json.dumps({"kind": "draft", "step": step, "of": of, **figures(paths, context)}) + "\n"
                    continue
                if kind == "final":
                    _, paths, info = event
                    result = {**figures(paths, info["context"]), "kit": info}
                    with cache_lock:
                        cache[key] = result
                        while len(cache) > CACHE_SIZE:
                            cache.popitem(last=False)
                    log.info("stream forecast %s ctx=%d steps=%d samples=%d in %dms", info["scale"], info["context"],
                             info["horizon"], body.samples, result["elapsedMs"])
                    yield json.dumps({"kind": "final", **result, "cached": False}) + "\n"
                elif kind == "error":
                    _, status, detail = event
                    yield json.dumps({"kind": "error", "status": status, "detail": detail}) + "\n"
                return
        finally:
            # Finished, failed, or the caller disconnected: stop the work.
            gone.set()

    return StreamingResponse(lines(), media_type="application/x-ndjson")
