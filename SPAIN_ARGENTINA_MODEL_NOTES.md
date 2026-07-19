# Spain vs Argentina predictor

## Repository model run
- Spain win: 39.7%
- Draw: 24.5%
- Argentina win: 35.8%
- Pick: Spain (toss-up)

The repository fixture file still used knockout placeholders, so the final fixture was locally resolved to Spain vs Argentina.
The model code itself was unchanged mathematically. `n_jobs` was limited to 4 to avoid CPU oversubscription in this runtime.

## Improved model
- Spain win: 40.4%
- Draw: 27.7%
- Argentina win: 32.0%
- Expected goals: Spain 1.42, Argentina 1.18
- Most likely score: Spain 1-1 Argentina

The improved version uses leakage-safe rolling form, dynamic Elo, recency weighting, an XGBoost outcome classifier,
a Dixon-Coles-adjusted Poisson score model, temporal validation, probability blending, temperature calibration,
and symmetric neutral-venue prediction.
