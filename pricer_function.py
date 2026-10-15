"""The Chalk function behind catalog_call('function.option-mc-price', ...).

It returns [price, stderr, delta, gamma, vega] for one option, but never prices one option at a time:
the function is async, so every row the engine hands one handler call runs as a coroutine on one
event loop, and a small coalescer turns all of them into a single request to the GPU pricer. A book
of 2,000 positions becomes a handful of GPU batches, not 2,000 round trips.

This module is copied into the function's container on its own, so it imports nothing local.
"""

from __future__ import annotations

import asyncio
import os
import weakref
from typing import Any

FIELDS = ("kind", "spot", "strike", "barrier", "expiry_years", "vol", "rate", "div_yield")
MAX_PER_REQUEST = 4096  # the pricer's limit per request


class _Coalescer:
    """Collects the rows submitted during one pass of the event loop and prices them together."""

    def __init__(self) -> None:
        super().__init__()
        self.pending: list[tuple[dict[str, Any], int, asyncio.Future[list[float]]]] = []
        self.scheduled = False

    def submit(self, option: dict[str, Any], paths: int) -> asyncio.Future[list[float]]:
        loop = asyncio.get_running_loop()
        fut: asyncio.Future[list[float]] = loop.create_future()
        self.pending.append((option, paths, fut))
        if not self.scheduled:
            # call_soon runs after every row coroutine already queued by asyncio.gather has
            # submitted, so one flush sees the whole handler call.
            self.scheduled = True
            loop.call_soon(lambda: asyncio.ensure_future(self.flush()))
        return fut

    async def flush(self) -> None:
        import httpx

        batch, self.pending, self.scheduled = self.pending, [], False
        by_paths: dict[int, list[tuple[dict[str, Any], asyncio.Future[list[float]]]]] = {}
        for option, paths, fut in batch:
            by_paths.setdefault(paths, []).append((option, fut))
        async with httpx.AsyncClient(timeout=300) as client:
            for paths, items in by_paths.items():
                for start in range(0, len(items), MAX_PER_REQUEST):
                    chunk = items[start : start + MAX_PER_REQUEST]
                    try:
                        r = await client.post(
                            os.environ["PRICER_URL"].rstrip("/") + "/price",
                            json={"options": [o for o, _ in chunk], "paths": paths, "steps": 64, "seed": 7},
                            headers={"x-pricer-token": os.environ.get("PRICER_TOKEN", "")},
                        )
                        r.raise_for_status()
                        for (_, fut), res in zip(chunk, r.json()["results"]):
                            fut.set_result([res["price"], res["stderr"], res["delta"], res["gamma"], res["vega"]])
                    except Exception as exc:  # every row in the chunk fails with the request
                        for _, fut in chunk:
                            if not fut.done():
                                fut.set_exception(exc)


# One coalescer per event loop: the runtime starts a fresh loop for each handler call.
_coalescers: weakref.WeakKeyDictionary[asyncio.AbstractEventLoop, _Coalescer] = weakref.WeakKeyDictionary()


async def option_mc_price(
    kind: str,
    spot: float,
    strike: float,
    barrier: float,
    expiry_years: float,
    vol: float,
    rate: float,
    div_yield: float,
    paths: int,
) -> list[float]:
    """Monte Carlo [price, stderr, delta, gamma, vega] of one option; vega is per vol point."""
    loop = asyncio.get_running_loop()
    coalescer = _coalescers.get(loop)
    if coalescer is None:
        coalescer = _coalescers[loop] = _Coalescer()
    option = dict(zip(FIELDS, (kind, spot, strike, barrier, expiry_years, vol, rate, div_yield)))
    return await coalescer.submit(option, int(paths))
