"""Monte Carlo option pricing on one GPU (and, for comparison, on the replica's CPUs).

POST /price   {"options": [{kind, spot, strike, barrier, expiry_years, vol, rate, div_yield}, ...],
               "paths": 100000, "steps": 64, "seed": 7, "device": "cuda" | "cpu", "detail": false}
           -> {"results": [{price, stderr, delta, gamma, vega, analytic}], "compute_ms", ...}
GET  /status

With "detail", the first option also gets 40 sample paths and a terminal-price histogram. Requests
need the x-pricer-token header when PRICER_TOKEN is set.
"""

from __future__ import annotations

import asyncio
import os
import threading
import time
from collections import OrderedDict, deque
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator, Literal

import torch
from fastapi import FastAPI, Header, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, model_validator

from mc import KINDS, Option, black_scholes, normals, price, sample_paths, terminal_histogram

TOKEN = os.environ.get("PRICER_TOKEN", "")
CPU_THREADS = int(os.environ.get("CPU_THREADS", "4"))
# Replica-wide budgets: the gateway hides client addresses, so limits apply to all callers together.
GPU_PER_MINUTE = int(os.environ.get("GPU_PER_MINUTE", "1200"))
CPU_PER_MINUTE = int(os.environ.get("CPU_PER_MINUTE", "30"))
# The CPU path exists to show the speed-up, so it is kept small.
CPU_MAX_OPTIONS = 4
CPU_MAX_PATHS = 200_000

state: dict[str, Any] = {}


class Budget:
    def __init__(self, per_minute: int) -> None:
        super().__init__()
        self.per_minute = per_minute
        self.calls: deque[float] = deque()

    def take(self) -> None:
        now = time.monotonic()
        while self.calls and now - self.calls[0] > 60:
            self.calls.popleft()
        if len(self.calls) >= self.per_minute:
            raise HTTPException(429, "pricing budget exhausted; try again in a minute")
        self.calls.append(now)


class Normals:
    """The last few [paths, steps] normal draws per device: the same seed and shape reuse them."""

    def __init__(self, size: int) -> None:
        super().__init__()
        self.size = size
        self.cache: OrderedDict[tuple[int, int, int, str], torch.Tensor] = OrderedDict()

    def get(self, paths: int, steps: int, seed: int, device: torch.device) -> torch.Tensor:
        key = (paths, steps, seed, device.type)
        if key in self.cache:
            self.cache.move_to_end(key)
            return self.cache[key]
        w = normals(paths, steps, seed, device)
        self.cache[key] = w
        while len(self.cache) > self.size:
            self.cache.popitem(last=False)
        return w


class OptionIn(BaseModel):
    kind: Literal[KINDS]  # type: ignore[valid-type]
    spot: float = Field(gt=0, lt=1e6)
    strike: float = Field(gt=0, lt=1e6)
    barrier: float = Field(ge=0, lt=1e7)
    expiry_years: float = Field(gt=0, le=5)
    vol: float = Field(gt=0, le=3)
    rate: float = Field(ge=-0.05, le=0.25)
    div_yield: float = Field(ge=0, le=0.25)

    @model_validator(mode="after")
    def barrier_kinds_need_barrier(self) -> OptionIn:
        if self.kind in ("up_and_out_call", "down_and_in_put") and self.barrier <= 0:
            raise ValueError(f"{self.kind} needs a positive barrier")
        return self


class PriceRequest(BaseModel):
    options: list[OptionIn] = Field(min_length=1, max_length=4096)
    paths: int = Field(default=100_000, ge=2_000, le=1_000_000)
    steps: int = Field(default=64, ge=1, le=256)
    seed: int = Field(default=7, ge=0, le=2**31)
    device: Literal["cuda", "cpu"] = "cuda"
    detail: bool = False


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncIterator[None]:
    torch.set_num_threads(CPU_THREADS)
    gpu = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    state.update(
        gpu=gpu,
        gpu_lock=threading.Lock(),
        cpu_lock=threading.Lock(),
        normals=Normals(6),
        budgets={"cuda": Budget(GPU_PER_MINUTE), "cpu": Budget(CPU_PER_MINUTE)},
    )
    # Warm up kernels and the default draw, so the replica turns ready with no first-call penalty.
    warm = [Option("asian_call", 100, 100, 0, 1, 0.3, 0.04, 0.0)]
    w = await asyncio.to_thread(state["normals"].get, 100_000, 64, 7, gpu)
    for _ in range(3):
        await asyncio.to_thread(price, warm, w)
    yield


app = FastAPI(lifespan=lifespan)


def run(req: PriceRequest) -> dict[str, Any]:
    device = state["gpu"] if req.device == "cuda" else torch.device("cpu")
    opts = [Option(**o.model_dump()) for o in req.options]
    paths = req.paths - req.paths % 2
    with state["gpu_lock" if req.device == "cuda" else "cpu_lock"]:
        t0 = time.perf_counter()
        w = state["normals"].get(paths, req.steps, req.seed, device)
        t1 = time.perf_counter()
        results = price(opts, w)
        if device.type == "cuda":
            torch.cuda.synchronize()
        t2 = time.perf_counter()
        detail = None
        if req.detail:
            detail = {
                "paths": sample_paths(opts[0], w, 40, 65),
                "histogram": terminal_histogram(opts[0], w, 40),
            }
    return {
        "results": [{**r.__dict__, "analytic": black_scholes(o)} for o, r in zip(opts, results)],
        "device": str(device) if device.type == "cpu" else torch.cuda.get_device_name(device),
        "paths": paths,
        "steps": req.steps,
        "normals_ms": round((t1 - t0) * 1000, 1),
        "compute_ms": round((t2 - t1) * 1000, 1),
        "detail": detail,
    }


@app.get("/status")
async def status() -> JSONResponse:
    return JSONResponse({"ok": True, "device": str(state["gpu"]), "cpu_threads": CPU_THREADS})


@app.post("/price")
async def price_endpoint(req: PriceRequest, x_pricer_token: str | None = Header(default=None)) -> JSONResponse:
    if TOKEN and x_pricer_token != TOKEN:
        raise HTTPException(401, "missing or wrong x-pricer-token")
    if req.device == "cpu" and (len(req.options) > CPU_MAX_OPTIONS or req.paths > CPU_MAX_PATHS):
        raise HTTPException(422, f"CPU runs take at most {CPU_MAX_OPTIONS} options and {CPU_MAX_PATHS:,} paths")
    state["budgets"][req.device].take()
    return JSONResponse(await asyncio.to_thread(run, req))


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", "8080")))
