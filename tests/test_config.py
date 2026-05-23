"""Config defaults, YAML round-trip, and validation."""

from __future__ import annotations

from pathlib import Path

import pytest

from ilb.config import Config, load_config


def test_defaults():
    c = Config()
    assert c.ticker == "IREN"
    assert c.targets == [150.0, 450.0, 700.0]
    assert c.leverage == 2.0
    assert c.drift_scenario == 0.0
    assert c.generator == "bootstrap"
    assert c.regime == "base"
    assert c.site.base_path == "/iren-lev-breakeven"


def test_load_config_file_roundtrip(tmp_path: Path):
    cfg_text = """
ticker: TEST
targets: [10, 20]
horizon_years: 0.5
generator: gbm
regime: high
site:
  base_path: "/"
  data_source: live
"""
    p = tmp_path / "config.yaml"
    p.write_text(cfg_text)
    c = load_config(p)
    assert c.ticker == "TEST"
    assert c.targets == [10.0, 20.0]
    assert c.horizon_years == 0.5
    assert c.generator == "gbm"
    assert c.regime == "high"
    assert c.site.base_path == "/"
    assert c.site.data_source == "live"


def test_missing_file_returns_defaults(tmp_path: Path):
    c = load_config(tmp_path / "does_not_exist.yaml")
    assert c.ticker == "IREN"


def test_invalid_lambda_rejected():
    with pytest.raises(ValueError):
        Config(ewma_lambda=1.5)


def test_invalid_targets_rejected():
    with pytest.raises(ValueError):
        Config(targets=[])
    with pytest.raises(ValueError):
        Config(targets=[10.0, -5.0])
