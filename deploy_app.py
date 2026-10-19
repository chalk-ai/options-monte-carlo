#!/usr/bin/env -S uv run --script
#
# /// script
# requires-python = ">=3.12,<3.14"
# dependencies = ["chalkcompute==2.13.5"]
# ///
"""Upload the book to a Chalk Volume and deploy the demo app (app/) as a CPU ScalingGroup.

    PRICER_TOKEN=... ./deploy_app.py [--upload data/]

The app runs book stress tests as ChalkSQL queries in the same environment, authenticated with the
ScalingGroup's own workload identity, so it holds no client secret. A ScalingGroup mounts the Volume
version current at deploy time, so upload and redeploy together.
"""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path


HERE = Path(__file__).resolve().parent
VOLUME = "options-monte-carlo-data"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--upload", type=Path, default=None, help="local data/ directory to upload first")
    args = parser.parse_args()
    from chalkcompute import Image, ReadinessProbe, ScalingGroup, Volume

    service_url = ScalingGroup.from_name("option-pricer").web_url

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
    group = ScalingGroup(
        image=image,
        name='options-monte-carlo',
        cpu="2",
        memory="4Gi",
        env={
            "PORT": "8080",
            "WEB_DIR": "/web",
            "MEDIA_DIR": "/media",
            "PRICER_URL": service_url,
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
