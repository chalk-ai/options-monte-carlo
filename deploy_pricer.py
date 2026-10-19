#!/usr/bin/env -S uv run --script
#
# /// script
# requires-python = ">=3.12,<3.14"
# dependencies = ["chalkcompute==2.13.5"]
# ///
"""Deploy the Monte Carlo pricer (pricer/) as a GPU ScalingGroup.

    PRICER_TOKEN=... ./deploy_pricer.py

The Chalk function in deploy_functions.py calls it, so ChalkSQL and chalkdf price options with
catalog_call; the app calls it directly for the single-option explorer.
"""

from __future__ import annotations

import os
import time
from pathlib import Path


HERE = Path(__file__).resolve().parent


def main() -> None:
    from chalkcompute import Image, ReadinessProbe, ScalingGroup

    image = (
        # The CUDA torch stack comes with the base image; pip only adds the web server.
        Image.base("pytorch/pytorch:2.9.1-cuda12.6-cudnn9-runtime")
        .pip_install(["fastapi==0.116.1", "uvicorn[standard]==0.35.0"])
        .add_local_file(str(HERE / "pricer/mc.py"), "/app/mc.py", strategy="copy")
        .add_local_file(str(HERE / "pricer/server.py"), "/app/server.py", strategy="copy")
        .workdir("/app")
    )
    group = ScalingGroup(
        image=image,
        name='option-pricer',
        gpu='nvidia-l4',
        cpu="4",
        memory="16Gi",
        env={"PORT": "8080", "CPU_THREADS": "4", "PRICER_TOKEN": os.environ.get("PRICER_TOKEN", "")},
        entrypoint=["python", "/app/server.py"],
        port=8080,
        readiness_probe=ReadinessProbe.http("/status", period_seconds=5),
        min_replicas=1,
        max_replicas=1,
    )
    t0 = time.monotonic()
    try:
        group.deploy(build_timeout=2400.0, ready_timeout=1800.0)
    finally:
        print(f"deploy returned after {time.monotonic() - t0:.0f}s")
    print("web_url:", group.web_url)


if __name__ == "__main__":
    main()
