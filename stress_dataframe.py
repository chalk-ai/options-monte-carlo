#!/usr/bin/env -S uv run --script
#
# /// script
# requires-python = ">=3.12,<3.14"
# dependencies = ["chalkpy[chalkdf]==2.162.5"]
# ///
"""The book stress test as a Chalk DataFrame: shock spot and vol, reprice every position on the GPU,
sum the P&L by sector.

    CHALK_ENVIRONMENT=<env id> ./stress_dataframe.py --spot -20 --vol 50 [--app https://<app>]
"""

from __future__ import annotations

import argparse
import json
import os
import urllib.request

APP = "https://tmnmc9beyujew-options-monte-carlo.staging.internal.gcp.chalk.dev"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--spot", type=float, default=-20, help="spot shock, percent")
    parser.add_argument("--vol", type=float, default=50, help="relative volatility shock, percent")
    parser.add_argument("--app", default=os.environ.get("OPTIONS_MC_APP", APP))
    args = parser.parse_args()
    # The app hands out the marked book's current versioned URL: the engine caches scans by URL.
    with urllib.request.urlopen(f"{args.app.rstrip('/')}/api/universe", timeout=30) as r:
        book_url = json.load(r)["book_parquet"]

    import pyarrow as pa
    from chalk import functions as F
    from chalk.features import _
    from chalkdf import DataFrame

    spot, vol = 1 + args.spot / 100, 1 + args.vol / 100
    price = F.catalog_call(
        "function.option-mc-price",
        _.kind,
        _.spot * spot,
        _.strike,
        _.barrier,
        _.expiry_years,
        _.vol * vol,
        _.rate,
        _.div_yield,
        50_000,
        output_type=pa.list_(pa.float64()),  # [price, stderr, delta, gamma, vega]
    )
    by_sector = (
        DataFrame.scan([book_url])
        .with_columns({"r": price})
        .with_columns({"pnl": _.quantity * 100 * (F.element_at(_.r, 0) - _.mark)})
        .group_by("sector")
        .agg(_.pnl.sum().alias("pnl"))
    )
    for row in sorted(by_sector.run(remote=True).to_arrow().to_pylist(), key=lambda r: r["pnl"]):
        print(f"{row['sector']:<12} {row['pnl']:>16,.0f}")


if __name__ == "__main__":
    main()
