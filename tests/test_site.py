"""Site builder + JS/Python parity for σ_be."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ilb import data as data_mod
from ilb import site as site_mod
from ilb.config import Config
from ilb.leverage import breakeven_sigma


def _js_breakeven_sigma(K: float, S0: float, T: float, c: float, phi: float,
                        beta: float) -> float:
    """Pure-Python port of the JS in web/breakeven.js. Used for the parity test
    — if you change one, change both, and this test will catch the drift."""
    if T <= 0:
        return float("nan")
    log_g = math.log(K / S0)
    num = (beta - 1) * log_g - (beta - 1) * c * T - phi * T
    qv = max(0.0, (2 / (beta * beta - beta)) * num)
    return math.sqrt(qv / T)


@pytest.mark.parametrize("K", [80, 150, 300, 450, 700])
@pytest.mark.parametrize("T", [0.25, 0.5, 1.0, 2.0])
@pytest.mark.parametrize("c,phi,beta", [(0.0, 0.0, 2.0), (0.03, 0.01, 2.0),
                                        (0.0, 0.0, 2.5)])
def test_js_python_parity(K, T, c, phi, beta):
    """Spec §10/§12.5: JS port and Python reference must agree within ±1e-6."""
    js = _js_breakeven_sigma(K, 58, T, c, phi, beta)
    py = float(breakeven_sigma(K, 58, T, financing_rate=c, expense_ratio=phi,
                               leverage=beta))
    assert abs(js - py) < 1e-6


def test_js_source_uses_same_formula():
    """Brittle but cheap: check the JS source contains the canonical formula."""
    js = Path("web/breakeven.js").read_text(encoding="utf-8")
    assert "(2 / (beta * beta - beta))" in js
    assert "Math.log(K / S0)" in js


def test_index_template_has_required_placeholders():
    tpl = Path("web/index.html").read_text(encoding="utf-8")
    for token in ("{{TITLE}}", "{{ASOF}}", "{{SPOT}}", "{{REGIME_BASE}}",
                  "{{BE_SAMPLE_JSON}}"):
        assert token in tpl, f"missing template token: {token}"


def test_inputs_json_shape(tmp_path: Path, monkeypatch):
    """Build site against a synthetic price frame and verify the inputs shape."""
    rng = np.random.default_rng(0)
    n = 600
    idx = pd.bdate_range("2022-01-03", periods=n)
    r = rng.normal(0.0, 0.06, size=n)
    close = 58 * np.exp(np.cumsum(r))
    log_return = np.concatenate([[np.nan], np.diff(np.log(close))])
    df = pd.DataFrame({"close": close, "log_return": log_return}, index=idx)

    monkeypatch.setattr(data_mod, "load_prices", lambda *a, **k: df)
    monkeypatch.setattr(site_mod, "load_prices", lambda *a, **k: df)
    monkeypatch.setattr(site_mod, "PLOTS_DIR", tmp_path / "plots")

    cfg = Config(mc_sims=200)
    out = site_mod.build(cfg, tmp_path / "site")

    inputs = json.loads((out / "inputs.json").read_text())
    assert {"meta", "spot", "regimes", "targets", "horizons", "target_grid"} <= inputs.keys()
    assert inputs["spot"] > 0
    assert {"low", "base", "high"} <= inputs["regimes"].keys()
    assert (out / "index.html").exists()
    assert (out / ".nojekyll").exists()
    assert (out / "assets" / "breakeven.js").exists()
    assert (out / "assets" / "styles.css").exists()
    # At least the 5 PNGs landed in assets
    pngs = list((out / "assets").glob("*.png"))
    assert len(pngs) == 5
