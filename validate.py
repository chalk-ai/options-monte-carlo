#!/usr/bin/env -S uv run --script
#
# /// script
# requires-python = ">=3.12"
# dependencies = ["torch==2.7.1", "numpy==2.3.3"]
# ///
"""Check the Monte Carlo pricer against closed forms and time it on the GPU and the CPU.

    ./validate.py [--paths 200000] [--cpu-threads 4]

1. 200 random European options: |MC - Black-Scholes| in standard errors (should look like |N(0,1)|).
2. Put-call parity across kinds on the same paths: in/out barrier parity is not used; instead an
   up-and-out call with an unreachable barrier must equal the European call exactly.
3. Throughput: options per second at the given path count, GPU vs CPU.
"""

from __future__ import annotations

import argparse
import random
import statistics
import time

import torch

from pricer.mc import Option, black_scholes, normals, price


def random_european(rng: random.Random) -> Option:
    s = rng.uniform(20, 400)
    return Option(
        kind=rng.choice(["european_call", "european_put"]),
        spot=s,
        strike=s * rng.uniform(0.8, 1.2),
        barrier=0.0,
        expiry_years=rng.uniform(1 / 12, 2),
        vol=rng.uniform(0.15, 0.8),
        rate=0.04,
        div_yield=rng.uniform(0, 0.03),
    )


def timed(opts: list[Option], paths: int, steps: int, device: torch.device) -> float:
    w = normals(paths, steps, 7, device)
    price(opts[:2], w)  # warm up kernels and the allocator
    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    w = normals(paths, steps, 7, device)
    price(opts, w)
    if device.type == "cuda":
        torch.cuda.synchronize()
    return time.perf_counter() - t0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--paths", type=int, default=200_000)
    parser.add_argument("--steps", type=int, default=64)
    parser.add_argument("--cpu-threads", type=int, default=4)
    args = parser.parse_args()
    gpu = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rng = random.Random(1)

    opts = [random_european(rng) for _ in range(200)]
    res = price(opts, normals(args.paths, args.steps, 11, gpu))
    z = [abs(r.price - black_scholes(o)) / r.stderr for o, r in zip(opts, res)]
    rel = [abs(r.price - black_scholes(o)) / black_scholes(o) for o, r in zip(opts, res)]
    print(f"European vs Black-Scholes, {len(opts)} options at {args.paths:,} paths:")
    print(f"  |error| / stderr: median {statistics.median(z):.2f}, 95th pct {sorted(z)[189]:.2f}, max {max(z):.2f}")
    print(f"  within 2 stderr: {sum(x <= 2 for x in z)}/{len(z)}; median relative error {statistics.median(rel):.4%}")

    o = opts[0]
    knock = Option("up_and_out_call", o.spot, o.strike, o.spot * 1e6, o.expiry_years, o.vol, o.rate, o.div_yield)
    eu = Option("european_call", o.spot, o.strike, 0.0, o.expiry_years, o.vol, o.rate, o.div_yield)
    a, b = price([knock, eu], normals(args.paths, args.steps, 3, gpu))
    print(f"unreachable up-and-out == European call: {a.price:.6f} vs {b.price:.6f}")

    book = [random_european(rng) for _ in range(256)]
    t_gpu = timed(book, args.paths, args.steps, gpu)
    torch.set_num_threads(args.cpu_threads)
    t_cpu = timed(book[:16], args.paths, args.steps, torch.device("cpu"))
    print(f"throughput at {args.paths:,} paths x {args.steps} steps, 5 pricings per option (price + 4 bumps):")
    print(f"  {gpu}: {len(book) / t_gpu:.1f} options/s ({t_gpu * 1000 / len(book):.1f} ms each)")
    print(f"  cpu x{args.cpu_threads}: {16 / t_cpu:.2f} options/s ({t_cpu * 1000 / 16:.0f} ms each)")


if __name__ == "__main__":
    main()
