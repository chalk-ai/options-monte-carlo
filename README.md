# Monte Carlo options pricing on GPUs, on Chalk

A PyTorch Monte Carlo pricer for European, Asian (average-price) and barrier options runs on a GPU
ScalingGroup. A Chalk function exposes it to ChalkSQL and Chalk DataFrames, so one query can mark a
whole options book, or reprice every position under a market shock and roll the P&L up by sector.

Live demo: <https://chalk.ai/demos/options-monte-carlo>

```sql
WITH priced AS (
    SELECT b.position_id, b.sector, b.quantity, b.mark,
           catalog_call('function.option-mc-price',
                        b.kind, b.spot * 0.80, b.strike, b.barrier, b.expiry_years,
                        b.vol * 1.50, b.rate, b.div_yield, 50000) AS r   -- [price, stderr, delta, gamma, vega]
    FROM read_parquet('https://<app>/data/book_marked.parquet') AS b
)
SELECT sector, sum(quantity * 100 * (r[1] - mark)) AS pnl
FROM priced
GROUP BY sector
```

**Everything here is synthetic.** The 40 underlyings (`SYN01` to `SYN40`), their prices, volatilities
and dividend yields, and the 2,000-position book are generated from a fixed seed by
`make_universe.py`. None of it is market data, and none of it is advice.

## The model

- Geometric Brownian motion with constant volatility, a 4% rate and a continuous dividend yield.
- 64 time steps per path. Asian averages and barriers are monitored at those 64 dates, not
  continuously, so barrier prices differ slightly from continuous-monitoring closed forms.
- Antithetic paths; the standard error comes from the pair averages.
- **Common random numbers.** Every option in a request is priced on the same normals (one draw per
  seed, path count and step count). Spot bumps rescale each path's terminal, average, maximum and
  minimum exactly, so delta and gamma cost almost nothing; vega reprices at sigma +/- 1 point on the
  same normals. Stress scenarios reuse the marks' seed and path count, so a zero shock gives a P&L of
  exactly 0 and a shock moves prices, not simulation noise.
- The normals are drawn on the CPU generator and moved to the device, so the CPU and the GPU return
  the same prices for the same request.

## How it fits together

```
make_universe.py ─► data/{underlyings,book}.parquet ─► Chalk Volume "options-monte-carlo-data"
                                                          │ mounted at /media
                                                          ▼
          options-monte-carlo app (CPU ScalingGroup): serves the book, runs stress tests as ChalkSQL
                 │ /api/price (explorer)                  │ ChalkSQL
                 ▼                                        ▼
   option-pricer (GPU ScalingGroup, PyTorch) ◄── catalog_call('function.option-mc-price', ...)
                                                  a Chalk function that coalesces the rows of each
                                                  call into one GPU request (pricer_function.py)
```

## Files

| File | What it does |
| --- | --- |
| `pricer/mc.py` | The Monte Carlo engine: paths, payoffs, Greeks, sample paths, histogram, Black–Scholes |
| `pricer/server.py` | FastAPI server for the GPU ScalingGroup: `POST /price`, `GET /status` |
| `validate.py` | Monte Carlo vs Black–Scholes on random European options, and GPU vs CPU timing |
| `make_universe.py` | The synthetic underlyings and book |
| `deploy_pricer.py` | Deploys the pricer on a GPU |
| `pricer_function.py`, `deploy_functions.py` | The async Chalk function and its deploy script |
| `deploy_app.py`, `app/` | The demo app: explorer proxy, stress tests, Volume-backed data |
| `sql/mark_book.sql` | Batch job: mark every position (price, stderr, Greeks) |
| `sql/stress.sql` | One stress scenario |
| `batch.py` | Runs the marking query and publishes the marks |
| `stress_dataframe.py` | The stress test as a Chalk DataFrame |

## Run it

Needs [uv](https://docs.astral.sh/uv/) and your existing `chalk login`. The pricer
only answers requests that carry `PRICER_TOKEN`.

```bash
export PRICER_TOKEN=$(openssl rand -hex 16)
./validate.py                                      # needs a CUDA GPU locally for the GPU half
./make_universe.py --out data/
./deploy_pricer.py # prints the pricer's web_url
./deploy_functions.py
./deploy_app.py --upload data/
./batch.py
./stress_dataframe.py --spot -20 --vol 50
```

The query engine reads its function catalog when the environment is deployed: after the first
`deploy_functions.py`, deploy (or re-activate) the environment once so `catalog_call` can see the new
name. Redeploying an existing function needs no such step.

## Measured

On Chalk staging, 2026-10-15, one NVIDIA L4 for the pricer (4 vCPU on the same replica for the CPU
comparison):

| Step | Measured |
| --- | --- |
| One option with Greeks, 200,000 paths x 64 steps | GPU median 7.8 ms; 4 CPU cores median 108 ms (14x) |
| Mark the book: 2,000 positions at 50,000 paths, `sql/mark_book.sql` | 4.0 s end to end (2.6 s of GPU compute) |
| One stress scenario through the app (`/api/stress`) | about 3.6 s of ChalkSQL |
| European positions in the book vs Black–Scholes | 986 of 991 within 2 standard errors; median relative error 0.05% |
| Same request on two GPUs (L4, H100) | bit-identical marks |
| Zero-shock scenario | P&L exactly 0 |

`./validate.py` on an A100 at 200,000 paths: 200 random European options all within 2 standard
errors of Black–Scholes (median relative error 0.22%), and an up-and-out call with an unreachable
barrier equal to the European call to the last digit.

## Notes

- **Limits.** The gateway hides client addresses, so limits are replica-wide. The app allows 60 GPU
  explorer runs, 10 CPU runs, 30 stress tests and 6 custom stress tests per minute, with identical
  requests cached for ten minutes; the pricer caps requests at 4,096 options and 1,000,000 paths, and
  CPU runs at 4 options and 200,000 paths.
- **Coalescing.** The function is async: every row of a handler call runs as a coroutine on one
  event loop, and the coalescer sends all of them to the GPU in one request.
- **Versioned parquet URLs.** The engine caches `read_parquet` results by URL, so the app and
  `batch.py` add `?v=<content hash>` whenever the book changes.
- **Delete it:** `ScalingGroup.from_name("option-pricer").delete()` (and `options-monte-carlo`,
  `option-mc-price`) with the same `CHALK_*` environment.

## License

Apache-2.0 (see `LICENSE`). The universe and book are synthetic and generated by this code.

## Local authentication

The SDK reads credentials, API host, and the selected environment from your existing
`chalk login` in `~/.chalk.yml`, including project-scoped and default logins.
For unattended runs, `CHALK_CLIENT_ID` and `CHALK_CLIENT_SECRET` remain supported.
The examples use the current environment selected by your Chalk login.
