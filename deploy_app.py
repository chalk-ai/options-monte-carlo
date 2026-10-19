#!/usr/bin/env -S uv run --script
#
# /// script
# requires-python = ">=3.12,<3.14"
# dependencies = ["chalkcompute==2.13.5"]
# ///
"""Upload the book to a Chalk Volume and deploy the demo app (app/) as a CPU ScalingGroup.

    PRICER_TOKEN=... ./deploy_app.py --env-id <environment> --pricer https://<pricer> [--upload data/]

The app runs book stress tests as ChalkSQL queries in the same environment, authenticated with the
ScalingGroup's own workload identity, so it holds no client secret. A ScalingGroup mounts the Volume
version current at deploy time, so upload and redeploy together.
"""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

from chalk_env import load_chalk_env

HERE = Path(__file__).resolve().parent
VOLUME = "options-monte-carlo-data"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", default="options-monte-carlo")
    parser.add_argument("--env-id", default=None)
    parser.add_argument("--pricer", required=True, help="web_url of the option-pricer ScalingGroup")
    parser.add_argument("--upload", type=Path, default=None, help="local data/ directory to upload first")
    args = parser.parse_args()
    load_chalk_env(args.env_id)
    from chalkcompute import Image, ReadinessProbe, ScalingGroup, Volume

    volume = Volume(VOLUME)
    if args.upload:
        t0 = time.monotonic()
        volume.put_dir(args.upload, "data")
        print(f"uploaded {args.upload}/ in {time.monotonic() - t0:.0f}s")

    image = (
        Image.debian_slim(python_version="3.12")
        .pip_install(
            ["chalkcompute==2.13.5", "httpx==0.28.1", "pyarrow==21.0.0",
             "fastapi==0.116.1", "uvicorn[standard]==0.35.0"]
        )
        .add_local_file(str(HERE / "app/server.py"), "/app/server.py", strategy="copy")
        .add_local_file(str(HERE / "app/index.html"), "/web/index.html", strategy="copy")
        .add_local_file(str(HERE / "app/presets.json"), "/web/presets.json", strategy="copy")
        .workdir("/app")
    )
    env_id = os.environ["CHALK_ENVIRONMENT_ID"]
    # The predicted public URL of this group; ChalkSQL reads the book from it.
    web_url = f"https://{env_id}-{args.name}.staging.internal.gcp.chalk.dev"
    group = ScalingGroup(
        image=image,
        name=args.name,
        cpu="2",
        memory="4Gi",
        env={
            "PORT": "8080",
            "WEB_DIR": "/web",
            "MEDIA_DIR": "/media",
            "CHALK_API_SERVER": os.environ["CHALK_API_SERVER"],
            "CHALK_ENVIRONMENT_ID": env_id,
            "BOOK_PARQUET": f"{web_url}/data/book_marked.parquet",
            "PRICER_URL": args.pricer,
            "PRICER_TOKEN": os.environ.get("PRICER_TOKEN", ""),
        },
        entrypoint=["python", "/app/server.py"],
        port=8080,
        readiness_probe=ReadinessProbe.http("/healthz", period_seconds=5),
        volumes=[volume.mount("/media")],
        chalk_identity=True,
        min_replicas=1,
        max_replicas=1,
    )
    t0 = time.monotonic()
    try:
        group.deploy(build_timeout=1800.0, ready_timeout=1200.0)
    finally:
        print(f"deploy returned after {time.monotonic() - t0:.0f}s")
    print("web_url:", group.web_url)


if __name__ == "__main__":
    main()
