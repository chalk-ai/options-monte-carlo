#!/usr/bin/env -S uv run --script
#
# /// script
# requires-python = ">=3.12,<3.14"
# dependencies = ["chalkcompute==2.13.5", "httpx==0.28.1", "pyarrow==21.0.0"]
# ///
"""Mark the whole book with one ChalkSQL query and publish the marks to the app.

    PRICER_TOKEN=... ./batch.py --env-id <environment> --pricer https://<pricer> --data data/

sql/mark_book.sql prices every position through function.option-mc-price (the GPU) and stores
price, standard error and Greeks as data/book_marked.parquet, which stress tests compare against.
The app serves the book back to ChalkSQL over HTTPS at a URL versioned by the file's hash, because
the engine caches read_parquet results by URL.
"""

from __future__ import annotations

import argparse
import hashlib
import subprocess
import time
from pathlib import Path

import httpx
import pyarrow.parquet as pq
from chalk_env import load_chalk_env
from chalk_sql import run_sql

HERE = Path(__file__).resolve().parent


def publish(data: Path, env_id: str, pricer: str, app: str, name: str) -> str:
    """Upload data/, redeploy the app, and return the file's versioned URL once the app serves it."""
    subprocess.run(
        [str(HERE / "deploy_app.py"), "--env-id", env_id, "--pricer", pricer, "--upload", str(data)], check=True
    )
    digest = hashlib.md5((data / name).read_bytes()).hexdigest()
    url = f"{app}/data/{name}"
    # The previous app revision keeps serving for a while after a redeploy; wait for the new bytes.
    for _ in range(60):
        r = httpx.get(url, timeout=120)
        if r.status_code == 200 and hashlib.md5(r.content).hexdigest() == digest:
            return f"{url}?v={digest[:12]}"
        time.sleep(10)
    raise RuntimeError(f"{url} never served the uploaded file")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--env-id", required=True)
    parser.add_argument("--pricer", required=True)
    parser.add_argument("--data", type=Path, default=Path("data"))
    args = parser.parse_args()
    load_chalk_env(args.env_id)
    app = f"https://{args.env_id}-options-monte-carlo.staging.internal.gcp.chalk.dev"

    book_url = publish(args.data, args.env_id, args.pricer, app, "book.parquet")
    marked, seconds = run_sql((HERE / "sql/mark_book.sql").read_text().replace("{book}", book_url))
    print(f"marked {marked.num_rows} positions in {seconds:.1f}s", flush=True)
    pq.write_table(marked, args.data / "book_marked.parquet")
    publish(args.data, args.env_id, args.pricer, app, "book_marked.parquet")


if __name__ == "__main__":
    main()
