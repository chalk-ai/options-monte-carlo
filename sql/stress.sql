-- One stress test, as the app runs it: spot -20%, volatility x1.5, every position repriced on the GPU.
WITH priced AS (
    SELECT b.position_id, b.underlying, b.sector, b.kind, b.quantity, b.mark,
           catalog_call('function.option-mc-price',
                        b.kind, b.spot * 0.80, b.strike, b.barrier, b.expiry_years,
                        b.vol * 1.50, b.rate, b.div_yield, 50000) AS r
    FROM read_parquet('{book}') AS b
)
SELECT position_id, underlying, sector, kind, quantity,
       quantity * 100 * (r[1] - mark) AS pnl,
       quantity * 100 * r[3] AS delta_shares,
       sum(quantity * 100 * (r[1] - mark)) OVER (PARTITION BY sector) AS sector_pnl
FROM priced
ORDER BY abs(quantity * 100 * (r[1] - mark)) DESC
