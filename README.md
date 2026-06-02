# iren-lev-breakeven

IREN 2x leverage vs underlying (1x) breakeven analyzer. Computes the
break-even realized volatility `σ_be(K, T)` at which a daily-rebalanced 2x
position ties buy-and-hold IREN for target prices K and holding periods T,
and contrasts it with IREN's empirical low / base / high volatility regimes.

> Personal analysis tool. Not investment advice. Drift is intentionally
> *not* estimated — all simulation statistics are conditioned on reaching K.

## Quickstart

```bash
uv sync
uv run ilb fetch                                # populate data cache
uv run ilb breakeven --targets 150,450,700      # closed-form σ_be table
uv run ilb run --targets 150,450,700 --horizon 1.0
```

See `config.yaml` for defaults and the spec for the full model
(`docs/SPEC.md` once vendored — for now refer to the v3 brief).

## Key identities

For continuous rebalancing at leverage β,
`L_T/L_0 = (S_T/S_0)^β · exp(−½(β²−β)·QV − (β−1)cT − φT)`
where `QV = ∫σ_t² dt` is the path's *realized* integrated variance.

Breakeven (β=2, c=φ=0): `σ_be(K, T) = √(ln(K/S_0) / T)`.

Discrete daily rebalancing in `simulate.py` is the ground truth; the
closed-form is a continuous-time approximation used for the interactive
breakeven panel and for sanity-checking the simulator.

## Regime diagnostic (BTC decoupling)

IREN was a bitcoin miner (Iris Energy) before pivoting to AI cloud / HPC, so a
`σ_base` / percentile band estimated over the *full* history blends two different
volatility regimes and distorts the breakeven verdict near `σ_be(165)`. This is a
**diagnostic only** — it does not touch the breakeven math.

`btc_regime.py` (pure-compute, no I/O):
- aligns IREN and BTC daily log-returns on IREN's trading days;
- rolling 60/120d OLS `β_t = Cov/Var` and Pearson `ρ_t` of IREN on BTC;
- structural-break detection on `β_60` via `ruptures` PELT (`PELT_PENALTY`,
  `PELT_MODEL` are documented tunables), with a CUSUM fallback if `ruptures` is
  unavailable, and a `CATALYST_OVERRIDE` constant to pin the split by hand;
- re-estimates `σ_low/base/high` on the post-break sample (reusing the existing
  `regime_vols(rolling_sigma(...))` pipeline) so the numbers compare to the site.

Two figures are added to the build (`btc_decoupling.png`, `regime_sigma_compare.png`)
and a new section on the page after "Volatility regimes over time". Caveat carried
in the captions: the post-pivot sample is short (large SE) — use the post-pivot
regime for the *central* σ but keep the high-vol miner-era history for the *tails*.

BTC bars come from `load_prices("BTC-USD")` (yfinance live → parquet cache →
committed `data_snapshot/btc-usd_daily.csv` fallback, mirroring IREN). `ruptures`
is now a runtime dependency; after editing `pyproject.toml` run `uv lock` (or just
`uv sync`, which re-resolves) to refresh `uv.lock`.
