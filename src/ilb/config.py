"""Configuration models and YAML loader (spec §6, §8)."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field, field_validator


class SiteConfig(BaseModel):
    base_path: str = "/iren-lev-breakeven"
    title: str = "IREN 2x Leverage Breakeven"
    data_source: Literal["snapshot", "live"] = "snapshot"
    refresh_cron: str = "0 21 * * 1-5"
    interactive_panel: bool = True


class Config(BaseModel):
    ticker: str = "IREN"
    spot: float | None = None
    targets: list[float] = Field(default_factory=lambda: [150.0, 450.0, 700.0])
    leverage: float = 2.0
    horizon_years: float = 1.0
    windows: list[int] = Field(default_factory=lambda: [20, 60, 120])
    ewma_lambda: float = 0.94
    generator: Literal["bootstrap", "gbm"] = "bootstrap"
    block_size: int = 15
    regime: Literal["low", "base", "high"] = "base"
    drift_scenario: float = 0.0  # annual log drift — SCENARIO knob, not an estimate
    financing_rate: float = 0.0
    expense_ratio: float = 0.0
    mc_sims: int = 20_000
    mc_days: int = 252
    band_pct: float = 0.05
    impair_threshold: float = 0.05
    seed: int = 42
    theme: Literal["dark", "light"] = "dark"
    site: SiteConfig = Field(default_factory=SiteConfig)

    @field_validator("ewma_lambda")
    @classmethod
    def _lambda_unit_interval(cls, v: float) -> float:
        if not 0.0 < v < 1.0:
            raise ValueError("ewma_lambda must lie in (0, 1)")
        return v

    @field_validator("leverage")
    @classmethod
    def _leverage_positive(cls, v: float) -> float:
        if v <= 0:
            raise ValueError("leverage must be > 0")
        return v

    @field_validator("targets")
    @classmethod
    def _targets_positive(cls, v: list[float]) -> list[float]:
        if not v or any(t <= 0 for t in v):
            raise ValueError("targets must be a non-empty list of positive prices")
        return v

    @field_validator("band_pct", "impair_threshold")
    @classmethod
    def _fraction(cls, v: float) -> float:
        if not 0.0 < v < 1.0:
            raise ValueError("must lie in (0, 1)")
        return v


def load_config(path: str | Path | None = None) -> Config:
    """Load config.yaml from `path` (defaults to ./config.yaml).

    Missing file → all defaults. Unknown keys raise (pydantic strict on extras).
    """
    cfg_path = Path(path) if path else Path("config.yaml")
    if not cfg_path.exists():
        return Config()
    with cfg_path.open("r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}
    return Config(**raw)
