-- Batch job: mark every position in the book on the GPU, through the Chalk function.
-- r = [price, stderr, delta, gamma, vega]; 50,000 antithetic paths, 64 steps, seed 7.
-- Stress tests reprice at the same path count and seed, so scenario P&L uses common random numbers.
WITH priced AS (
    SELECT b.*,
           catalog_call('function.option-mc-price',
                        b.kind, b.spot, b.strike, b.barrier, b.expiry_years,
                        b.vol, b.rate, b.div_yield, 50000) AS r
    FROM read_parquet('{book}') AS b
)
SELECT position_id, underlying, sector, kind, spot, strike, barrier, expiry_years, vol, rate,
       div_yield, quantity,
       r[1] AS mark, r[2] AS mark_stderr, r[3] AS delta, r[4] AS gamma, r[5] AS vega
FROM priced
