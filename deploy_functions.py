#!/usr/bin/env -S uv run --script
#
# /// script
# requires-python = ">=3.12,<3.14"
# dependencies = ["chalkcompute==2.13.5"]
# ///
"""Deploy the pricer as a Chalk function, so ChalkSQL and chalkdf can call it by name.

    PRICER_TOKEN=... ./deploy_functions.py --env-id <environment> --pricer https://<pricer web_url>

    SELECT catalog_call('function.option-mc-price',
                        'asian_call', 100.0, 100.0, 0.0, 1.0, 0.3, 0.04, 0.0, 50000)
    -- [price, stderr, delta, gamma, vega]

The function (pricer_function.py) coalesces the rows of each call into one GPU request.
"""

from __future__ import annotations

import argparse
import os

from chalk_env import load_chalk_env
from pricer_function import option_mc_price


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--env-id", default=None)
    parser.add_argument("--pricer", required=True, help="web_url of the option-pricer ScalingGroup")
    args = parser.parse_args()
    load_chalk_env(args.env_id)
    import chalkcompute
    from chalkcompute import Image, RemoteFunction

    name = "option-mc-price"
    try:
        chalkcompute.function(
            image=Image.debian_slim(python_version="3.12").pip_install(["httpx==0.28.1"]),
            name=name,
            cpu="2",
            memory="4Gi",
            env={"PRICER_URL": args.pricer, "PRICER_TOKEN": os.environ.get("PRICER_TOKEN", "")},
            min_replicas=1,
            max_replicas=2,
            # A handler call carries up to this many engine batches; their rows go to the GPU together.
            max_batching_size=64,
            max_buffer_duration=20,
        )(option_mc_price)
    except Exception as exc:
        # The decorator polls readiness for a fixed 300 s; a cold image pull can outlast it while the
        # rollout continues server-side.
        if "not ready" not in str(exc):
            raise
        print(f"{name}: {exc}; waiting longer")
        RemoteFunction.from_name(name).wait_ready(timeout=1200)
    print(f"deployed function.{name}")


if __name__ == "__main__":
    main()
