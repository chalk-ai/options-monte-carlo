"""Monte Carlo pricing of European, Asian and barrier options under geometric Brownian motion.

Every option in a request is priced on the same standard normals (one draw per seed, path count and
step count), so prices, Greeks and scenario P&L all use common random numbers:

- Paths are antithetic: the second half of the paths mirrors the first, and the standard error is
  computed from the pair averages.
- An option's path statistics (terminal, average, maximum and minimum price) are computed once per
  volatility. Spot bumps rescale every statistic exactly, so delta and gamma cost nothing extra; vega
  reprices at sigma +/- 1 point on the same normals.
- Barriers and Asian averages are monitored at the discrete steps (64 by default), not continuously.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

KINDS = ("european_call", "european_put", "asian_call", "asian_put", "up_and_out_call", "down_and_in_put")
SPOT_BUMP = 0.01  # relative spot bump for delta and gamma
VOL_BUMP = 0.01  # absolute volatility bump for vega (one vol point)
# Elements of one [options, paths, steps] block; bounds GPU memory at about 1 GiB of float32 per block.
BLOCK_ELEMENTS = 1 << 28


@dataclass(frozen=True)
class Option:
    kind: str
    spot: float
    strike: float
    barrier: float  # 0 when the kind has none
    expiry_years: float
    vol: float
    rate: float
    div_yield: float


@dataclass(frozen=True)
class Result:
    price: float
    stderr: float
    delta: float
    gamma: float
    vega: float  # per one vol point


def normals(paths: int, steps: int, seed: int, device: torch.device) -> torch.Tensor:
    """Cumulative standard-normal increments for antithetic paths, shape [paths, steps]."""
    if paths % 2:
        raise ValueError("paths must be even (antithetic pairs)")
    gen = torch.Generator(device="cpu").manual_seed(seed)
    # Drawn on the CPU generator so the numbers are identical on every device.
    z = torch.randn(paths // 2, steps, generator=gen, dtype=torch.float32).to(device)
    w = torch.cumsum(z, dim=1)
    return torch.cat([w, -w], dim=0)


def _stats(opts: list[Option], w: torch.Tensor, vol_shift: float) -> tuple[torch.Tensor, ...]:
    """Terminal, average, max and min of each option's paths at unit spot: four [k, paths] tensors."""
    dev, steps = w.device, w.shape[1]
    t = torch.tensor([o.expiry_years for o in opts], device=dev)[:, None, None]
    sigma = torch.tensor([o.vol + vol_shift for o in opts], device=dev).clamp_min(1e-4)[:, None, None]
    carry = torch.tensor([o.rate - o.div_yield for o in opts], device=dev)[:, None, None]
    dt = t / steps
    k = torch.arange(1, steps + 1, device=dev, dtype=torch.float32)[None, None, :]
    rel = torch.exp((carry - 0.5 * sigma**2) * dt * k + sigma * torch.sqrt(dt) * w[None])
    return rel[..., -1], rel.mean(dim=2), rel.amax(dim=2), rel.amin(dim=2)


def _payoffs(opts: list[Option], stats: tuple[torch.Tensor, ...], spot_scale: float) -> torch.Tensor:
    """Undiscounted payoffs [k, paths] with every spot scaled by spot_scale."""
    term, avg, hi, lo = stats
    dev = term.device
    s0 = torch.tensor([o.spot * spot_scale for o in opts], device=dev)[:, None]
    strike = torch.tensor([o.strike for o in opts], device=dev)[:, None]
    barrier = torch.tensor([o.barrier for o in opts], device=dev)[:, None]
    out = torch.empty_like(term)
    for kind in KINDS:
        idx = [i for i, o in enumerate(opts) if o.kind == kind]
        if not idx:
            continue
        i = torch.tensor(idx, device=dev)
        s, k_, b = s0[i], strike[i], barrier[i]
        if kind == "european_call":
            p = (s * term[i] - k_).clamp_min(0)
        elif kind == "european_put":
            p = (k_ - s * term[i]).clamp_min(0)
        elif kind == "asian_call":
            p = (s * avg[i] - k_).clamp_min(0)
        elif kind == "asian_put":
            p = (k_ - s * avg[i]).clamp_min(0)
        elif kind == "up_and_out_call":
            p = (s * term[i] - k_).clamp_min(0) * (s * hi[i] < b)
        else:  # down_and_in_put
            p = (k_ - s * term[i]).clamp_min(0) * (s * lo[i] <= b)
        out[i] = p
    return out


def _discounted_mean(opts: list[Option], payoff: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    disc = torch.tensor([math.exp(-o.rate * o.expiry_years) for o in opts], device=payoff.device)
    half = payoff.shape[1] // 2
    pairs = 0.5 * (payoff[:, :half] + payoff[:, half:])
    mean = pairs.mean(dim=1) * disc
    stderr = pairs.std(dim=1) / math.sqrt(half) * disc
    return mean, stderr


def price(opts: list[Option], w: torch.Tensor) -> list[Result]:
    """Price `opts` on the cumulative normals `w` from normals()."""
    for o in opts:
        if o.kind not in KINDS:
            raise ValueError(f"unknown option kind {o.kind!r}")
    paths, steps = w.shape
    per_block = max(1, BLOCK_ELEMENTS // (paths * steps))
    results: list[Result] = []
    for start in range(0, len(opts), per_block):
        block = opts[start : start + per_block]
        base = _stats(block, w, 0.0)
        p0, se = _discounted_mean(block, _payoffs(block, base, 1.0))
        up, _ = _discounted_mean(block, _payoffs(block, base, 1 + SPOT_BUMP))
        dn, _ = _discounted_mean(block, _payoffs(block, base, 1 - SPOT_BUMP))
        del base
        v_up, _ = _discounted_mean(block, _payoffs(block, _stats(block, w, VOL_BUMP), 1.0))
        v_dn, _ = _discounted_mean(block, _payoffs(block, _stats(block, w, -VOL_BUMP), 1.0))
        spots = torch.tensor([o.spot for o in block], device=w.device)
        h = SPOT_BUMP * spots
        delta = (up - dn) / (2 * h)
        gamma = (up - 2 * p0 + dn) / h**2
        vega = (v_up - v_dn) / 2  # per vol point: the bump is one point each way
        for row in torch.stack([p0, se, delta, gamma, vega], dim=1).cpu().tolist():
            results.append(Result(*row))
    return results


def sample_paths(opt: Option, w: torch.Tensor, count: int, points: int) -> list[list[float]]:
    """The first `count` price paths of `opt`, each thinned to at most `points` points (spot first)."""
    steps = w.shape[1]
    dt = opt.expiry_years / steps
    k = torch.arange(1, steps + 1, device=w.device, dtype=torch.float32)[None, :]
    drift = (opt.rate - opt.div_yield - 0.5 * opt.vol**2) * dt
    rel = torch.exp(drift * k + opt.vol * math.sqrt(dt) * w[:count])
    paths = torch.cat([torch.ones(count, 1, device=w.device), rel], dim=1) * opt.spot
    stride = max(1, math.ceil(paths.shape[1] / points))
    return paths[:, ::stride].cpu().tolist()


def terminal_histogram(opt: Option, w: torch.Tensor, bins: int) -> dict[str, object]:
    """Histogram of the terminal price over all paths, and the share of paths that pay out."""
    term, avg, hi, lo = (s[0] for s in _stats([opt], w, 0.0))
    final = term * opt.spot
    lo_edge, hi_edge = torch.quantile(final[:200_000], torch.tensor([0.001, 0.999], device=w.device)).tolist()
    counts = torch.histc(final.clamp(lo_edge, hi_edge), bins=bins, min=lo_edge, max=hi_edge)
    payoff = _payoffs([opt], (term[None], avg[None], hi[None], lo[None]), 1.0)[0]
    return {
        "edges": [lo_edge + (hi_edge - lo_edge) * i / bins for i in range(bins + 1)],
        "counts": counts.cpu().tolist(),
        "paying_share": float((payoff > 0).float().mean()),
    }


def black_scholes(opt: Option) -> float | None:
    """Closed-form price for the European kinds (None otherwise), with a continuous dividend yield."""
    if opt.kind not in ("european_call", "european_put"):
        return None
    s, k, t, v, r, q = opt.spot, opt.strike, opt.expiry_years, opt.vol, opt.rate, opt.div_yield
    d1 = (math.log(s / k) + (r - q + 0.5 * v * v) * t) / (v * math.sqrt(t))
    d2 = d1 - v * math.sqrt(t)
    n = lambda x: 0.5 * (1 + math.erf(x / math.sqrt(2)))  # noqa: E731
    if opt.kind == "european_call":
        return s * math.exp(-q * t) * n(d1) - k * math.exp(-r * t) * n(d2)
    return k * math.exp(-r * t) * n(-d2) - s * math.exp(-q * t) * n(-d1)
