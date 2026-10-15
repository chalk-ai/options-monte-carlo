"""Options Monte Carlo demo app.

- POST /api/price   prices one option on the GPU pricer (or its CPUs, for comparison), with sample
                    paths and a terminal-price histogram.
- POST /api/stress  reprices the whole synthetic book under a spot/vol scenario with ONE ChalkSQL
                    query: catalog_call('function.option-mc-price', ...) per position, P&L against the
                    stored marks, sector totals by window function.
- GET  /api/universe the underlyings, the book's composition and the stress presets.

The book parquet is served from a Chalk Volume mounted at MEDIA_DIR; ChalkSQL reads it back over HTTPS.
"""

from __future__ import annotations

import asyncio
import base64
import hashlib
import io
import json
import os
import time
from collections import Counter, OrderedDict, deque
from pathlib import Path
from typing import Any, Literal

import httpx
import pyarrow.parquet as pq
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

WEB = Path(os.environ.get("WEB_DIR", "/web"))
MEDIA = Path(os.environ.get("MEDIA_DIR", "/media"))
API = os.environ["CHALK_API_SERVER"].rstrip("/")
ENV_ID = os.environ["CHALK_ENVIRONMENT_ID"]
PRICER_URL = os.environ["PRICER_URL"].rstrip("/")
PRICER_TOKEN = os.environ.get("PRICER_TOKEN", "")
STRESS_PATHS = 50_000  # the marks were computed at this path count and seed: scenario P&L uses common numbers
CACHE_SECONDS = 600
# Replica-wide budgets: the gateway hides client addresses, so limits apply to all visitors together.
LIMITS = {"gpu": 60, "cpu": 10, "stress_custom": 6, "stress": 30}


def versioned(url: str, path: Path) -> str:
    """`url` tagged with a hash of the file it serves: the engine caches read_parquet results by URL."""
    if not path.is_file():
        return url
    return f"{url}?v={hashlib.md5(path.read_bytes()).hexdigest()[:12]}"


BOOK_PATH = MEDIA / "data" / "book_marked.parquet"
BOOK_URL = versioned(os.environ["BOOK_PARQUET"], BOOK_PATH)
UNDERLYINGS = (
    {r["underlying"]: r for r in pq.read_table(MEDIA / "data" / "underlyings.parquet").to_pylist()}
    if (MEDIA / "data" / "underlyings.parquet").is_file()
    else {}
)
PRESETS = json.loads((WEB / "presets.json").read_text())

STRESS_SQL = """WITH priced AS (
    SELECT b.position_id, b.underlying, b.sector, b.kind, b.quantity, b.mark,
           catalog_call('function.option-mc-price',
                        b.kind, b.spot * {spot}, b.strike, b.barrier, b.expiry_years,
                        b.vol * {vol}, b.rate, b.div_yield, {paths}) AS r
    FROM read_parquet('{book}') AS b
)
SELECT position_id, underlying, sector, kind, quantity,
       quantity * 100 * (r[1] - mark) AS pnl,
       quantity * 100 * r[3] AS delta_shares,
       sum(quantity * 100 * (r[1] - mark)) OVER (PARTITION BY sector) AS sector_pnl
FROM priced
ORDER BY abs(quantity * 100 * (r[1] - mark)) DESC"""

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origin_regex=r"https://([a-z0-9-]+\.)*chalk\.(ai|dev)|http://localhost(:\d+)?",
    allow_methods=["GET", "POST"],
    allow_headers=["content-type"],
)
if (MEDIA / "data").is_dir():
    app.mount("/data", StaticFiles(directory=MEDIA / "data"), name="data")

_client = httpx.AsyncClient(timeout=300)
_cache: OrderedDict[str, tuple[float, dict[str, Any]]] = OrderedDict()
_calls: dict[str, deque[float]] = {k: deque() for k in LIMITS}
_sql_slots = asyncio.Semaphore(2)


def take(kind: str) -> None:
    now = time.monotonic()
    calls = _calls[kind]
    while calls and now - calls[0] > 60:
        calls.popleft()
    if len(calls) >= LIMITS[kind]:
        raise HTTPException(429, "The demo is busy right now; try a preset or wait a minute.")
    calls.append(now)


def cached(key: str) -> dict[str, Any] | None:
    hit = _cache.get(key)
    if hit and time.time() - hit[0] < CACHE_SECONDS:
        _cache.move_to_end(key)
        return {**hit[1], "cached": True}
    return None


def remember(key: str, value: dict[str, Any]) -> dict[str, Any]:
    _cache[key] = (time.time(), value)
    while len(_cache) > 256:
        _cache.popitem(last=False)
    return {**value, "cached": False}


async def token() -> str:
    # chalk_identity=True mounts a workload JWT and rotates it in place, so it is re-read each call.
    path = os.environ.get("CHALK_WEB_IDENTITY_TOKEN_FILE")
    if path and Path(path).exists():
        return Path(path).read_text().strip()
    r = await _client.post(
        f"{API}/v1/oauth/token",
        json={
            "client_id": os.environ["CHALK_CLIENT_ID"],
            "client_secret": os.environ["CHALK_CLIENT_SECRET"],
            "grant_type": "client_credentials",
        },
    )
    r.raise_for_status()
    return r.json()["access_token"]


class PriceRequest(BaseModel):
    underlying: str = Field(pattern=r"^SYN\d{2}$")
    kind: Literal["european_call", "european_put", "asian_call", "asian_put", "up_and_out_call", "down_and_in_put"]
    strike_pct: float = Field(ge=50, le=150)  # strike as % of spot
    barrier_pct: float = Field(default=0, ge=0, le=300)  # barrier as % of spot (barrier kinds only)
    expiry_months: int = Field(ge=1, le=36)
    paths: Literal[20_000, 100_000, 200_000, 1_000_000] = 100_000
    device: Literal["cuda", "cpu"] = "cuda"


class StressRequest(BaseModel):
    preset: str | None = None
    spot_shock_pct: float = Field(default=0, ge=-40, le=40)
    vol_shock_pct: float = Field(default=0, ge=-50, le=150)


@app.get("/")
async def index() -> FileResponse:
    return FileResponse(WEB / "index.html")


@app.get("/healthz")
async def healthz() -> dict[str, bool]:
    return {"ok": True}


@app.get("/api/universe")
async def universe() -> JSONResponse:
    book = pq.read_table(BOOK_PATH).to_pylist() if BOOK_PATH.is_file() else []
    return JSONResponse(
        {
            "underlyings": list(UNDERLYINGS.values()),
            "positions": len(book),
            "kinds": Counter(r["kind"] for r in book),
            "sectors": Counter(r["sector"] for r in book),
            "book_value": round(sum(r["quantity"] * 100 * r["mark"] for r in book)),
            "presets": PRESETS,
            "book_parquet": BOOK_URL,
        }
    )


@app.post("/api/price")
async def price(req: PriceRequest) -> JSONResponse:
    u = UNDERLYINGS.get(req.underlying)
    if u is None:
        raise HTTPException(404, "unknown underlying")
    barrier_kind = req.kind in ("up_and_out_call", "down_and_in_put")
    if req.device == "cpu" and req.paths > 200_000:
        raise HTTPException(422, "CPU runs are capped at 200,000 paths")
    if barrier_kind and req.barrier_pct <= 0:
        raise HTTPException(422, "barrier options need a barrier")
    key = json.dumps(req.model_dump(), sort_keys=True)
    if hit := cached(key):
        return JSONResponse(hit)
    take("gpu" if req.device == "cuda" else "cpu")
    option = {
        "kind": req.kind,
        "spot": u["spot"],
        "strike": round(u["spot"] * req.strike_pct / 100, 4),
        "barrier": round(u["spot"] * req.barrier_pct / 100, 4) if barrier_kind else 0.0,
        "expiry_years": req.expiry_months / 12,
        "vol": u["vol"],
        "rate": 0.04,
        "div_yield": u["div_yield"],
    }
    t0 = time.perf_counter()
    r = await _client.post(
        f"{PRICER_URL}/price",
        json={"options": [option], "paths": req.paths, "steps": 64, "seed": 7, "device": req.device, "detail": True},
        headers={"x-pricer-token": PRICER_TOKEN},
    )
    if r.status_code == 429:
        raise HTTPException(429, "The GPU is busy right now; try again in a minute.")
    r.raise_for_status()
    body = r.json()
    result = {
        "option": option,
        "underlying": u,
        **body["results"][0],
        "device": body["device"],
        "paths": body["paths"],
        "compute_ms": body["compute_ms"],
        "round_trip_ms": round((time.perf_counter() - t0) * 1000),
        "detail": body["detail"],
    }
    return JSONResponse(remember(key, result))


@app.post("/api/stress")
async def stress(req: StressRequest) -> JSONResponse:
    if req.preset is not None:
        preset = next((p for p in PRESETS if p["id"] == req.preset), None)
        if preset is None:
            raise HTTPException(404, "unknown preset")
        spot_pct, vol_pct = float(preset["spot_shock_pct"]), float(preset["vol_shock_pct"])
    else:
        spot_pct, vol_pct = round(req.spot_shock_pct), round(req.vol_shock_pct)
    key = f"stress:{spot_pct}:{vol_pct}"
    if hit := cached(key):
        return JSONResponse(hit)
    take("stress")
    if req.preset is None:
        take("stress_custom")
    # Both multipliers are formatted from validated floats; no visitor text reaches the SQL.
    sql = STRESS_SQL.format(
        spot=f"{1 + spot_pct / 100:.4f}", vol=f"{1 + vol_pct / 100:.4f}", paths=STRESS_PATHS, book=BOOK_URL
    )
    async with _sql_slots:
        t0 = time.perf_counter()
        r = await _client.post(
            f"{API}/chalk.protosql.v1.SqlService/ExecuteSqlQuery",
            headers={
                "Authorization": f"Bearer {await token()}",
                "X-Chalk-Env-Id": ENV_ID,
                "x-chalk-deployment-type": "engine-grpc",
            },
            json={"query": sql, "syncOptions": {}},
        )
        sql_ms = (time.perf_counter() - t0) * 1000
    body = r.json()
    if "parquet" not in body:
        raise HTTPException(502, f"ChalkSQL error: {json.dumps(body.get('errors', body))[:300]}")
    rows = pq.read_table(io.BytesIO(base64.b64decode(body["parquet"]))).to_pylist()
    sectors = sorted({(r["sector"], r["sector_pnl"]) for r in rows}, key=lambda s: s[1])
    result = {
        "spot_shock_pct": spot_pct,
        "vol_shock_pct": vol_pct,
        "positions": len(rows),
        "total_pnl": round(sum(r["pnl"] for r in rows)),
        "net_delta_shares": round(sum(r["delta_shares"] for r in rows)),
        "sectors": [{"sector": s, "pnl": round(p)} for s, p in sectors],
        "top": [
            {k: (round(v, 2) if isinstance(v, float) else v) for k, v in r.items() if k != "sector_pnl"}
            for r in rows[:10]
        ],
        "sql": sql,
        "sql_ms": round(sql_ms),
        "paths": STRESS_PATHS,
    }
    return JSONResponse(remember(key, result))


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", "8080")), proxy_headers=True)
