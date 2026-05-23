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
