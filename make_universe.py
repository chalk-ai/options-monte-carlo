#!/usr/bin/env -S uv run --script
#
# /// script
# requires-python = ">=3.12"
# dependencies = ["pyarrow==21.0.0"]
# ///
"""Generate the synthetic universe and options book (fixed seed, so every run is identical).

    ./make_universe.py --out data/

writes data/underlyings.parquet (40 fictional underlyings SYN01..SYN40 in five sectors) and
data/book.parquet (2,000 option positions on them). Nothing here is market data.
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

RATE = 0.04
# sector: (volatility range, dividend yield range)
SECTORS = {
    "Utilities": ((0.15, 0.25), (0.02, 0.04)),
    "Consumer": ((0.20, 0.35), (0.01, 0.025)),
    "Industrials": ((0.25, 0.40), (0.01, 0.02)),
    "Technology": ((0.35, 0.60), (0.0, 0.01)),
    "Biotech": ((0.50, 0.85), (0.0, 0.0)),
}
KIND_WEIGHTS = {
    "european_call": 0.25,
    "european_put": 0.25,
    "asian_call": 0.12,
    "asian_put": 0.08,
    "up_and_out_call": 0.15,
    "down_and_in_put": 0.15,
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=Path("data"))
    parser.add_argument("--positions", type=int, default=2000)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    rng = random.Random(20261015)

    underlyings = []
    for i in range(40):
        sector = list(SECTORS)[i % len(SECTORS)]
        (v_lo, v_hi), (q_lo, q_hi) = SECTORS[sector]
        underlyings.append(
            {
                "underlying": f"SYN{i + 1:02d}",
                "sector": sector,
                "spot": round(rng.uniform(20, 400), 2),
                "vol": round(rng.uniform(v_lo, v_hi), 4),
                "div_yield": round(rng.uniform(q_lo, q_hi), 4),
            }
        )
    pq.write_table(pa.Table.from_pylist(underlyings), args.out / "underlyings.parquet")

    kinds, weights = zip(*KIND_WEIGHTS.items())
    book = []
    for pid in range(args.positions):
        u = rng.choice(underlyings)
        kind = rng.choices(kinds, weights)[0]
        strike = round(u["spot"] * rng.uniform(0.8, 1.2), 2)
        if kind == "up_and_out_call":
            barrier = round(max(strike, u["spot"]) * rng.uniform(1.15, 1.5), 2)
        elif kind == "down_and_in_put":
            barrier = round(min(strike, u["spot"]) * rng.uniform(0.65, 0.9), 2)
        else:
            barrier = 0.0
        book.append(
            {
                "position_id": pid + 1,
                "underlying": u["underlying"],
                "sector": u["sector"],
                "kind": kind,
                "spot": u["spot"],
                "strike": strike,
                "barrier": barrier,
                "expiry_years": round(rng.choice([1, 2, 3, 6, 9, 12, 18, 24]) / 12, 4),
                "vol": u["vol"],
                "rate": RATE,
                "div_yield": u["div_yield"],
                # Contracts of 100 units; negative is short.
                "quantity": rng.choice([-1, 1]) * rng.randint(1, 50),
            }
        )
    pq.write_table(pa.Table.from_pylist(book), args.out / "book.parquet")
    print(f"{len(underlyings)} underlyings, {len(book)} positions in {args.out}")


if __name__ == "__main__":
    main()
