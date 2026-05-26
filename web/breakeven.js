// Drift-free closed-form breakeven recomputation.
// Mirror of src/ilb/leverage.py — must agree to ±1e-6 (parity test).
//
// Continuous-time identity:
//   L_T/L_0 = (S_T/S_0)^β · exp(-½(β²-β)·QV - (β-1)·c·T - φ·T)
// Breakeven: σ_be(K, T) = √( max(0, [ln(K/S_0) - (c+φ)·T] · 2 / ((β²-β)·T)) · ((β-1)) / 1 )
// For β=2 collapse: σ_be = √(max(0, ln(K/S_0) - (c+φ)·T) / T).

(function () {
  function breakevenSigma(K, S0, T, c, phi, beta) {
    if (T <= 0) return NaN;
    const logG = Math.log(K / S0);
    // generalized: QV_be = (2/(β²-β)) · [(β-1)·logG - (β-1)·c·T - φ·T]
    const num = (beta - 1) * logG - (beta - 1) * c * T - phi * T;
    const qv = Math.max(0, (2 / (beta * beta - beta)) * num);
    return Math.sqrt(qv / T);
  }

  function leveredTerminal(grossST, qv, T, c, phi, beta) {
    const drag = 0.5 * (beta * beta - beta) * qv + (beta - 1) * c * T + phi * T;
    return Math.pow(grossST, beta) * Math.exp(-drag);
  }

  const ids = ["K", "T", "sigma", "c", "phi", "beta"];
  const els = {};
  ids.forEach(id => {
    els[id] = document.getElementById("ctl-" + id);
    els["out-" + id] = document.getElementById("out-" + id);
  });

  async function loadInputs() {
    try {
      const r = await fetch("./inputs.json");
      if (!r.ok) return null;
      return await r.json();
    } catch (_) { return null; }
  }

  function fmtPct(x) { return (100 * x).toFixed(2) + "%"; }

  function recompute(state) {
    const K = +els.K.value, T = +els.T.value, sigma = +els.sigma.value,
          c = +els.c.value, phi = +els.phi.value, beta = +els.beta.value;
    els["out-K"].textContent = K;
    els["out-T"].textContent = T.toFixed(2);
    els["out-sigma"].textContent = sigma.toFixed(2);
    els["out-c"].textContent = c.toFixed(3);
    els["out-phi"].textContent = phi.toFixed(3);
    els["out-beta"].textContent = beta.toFixed(1);

    const S0 = state.spot;
    const sigmaBe = breakevenSigma(K, S0, T, c, phi, beta);
    const qv = sigma * sigma * T;
    const LT = leveredTerminal(K / S0, qv, T, c, phi, beta);
    const ST = K / S0;
    document.getElementById("rd-sigma-be").textContent = sigmaBe.toFixed(4);
    document.getElementById("rd-LT").textContent = LT.toFixed(3);
    document.getElementById("rd-ST").textContent = ST.toFixed(3);
    const verdict = LT > ST
      ? `2x wins by ${fmtPct(LT / ST - 1)}`
      : (LT < ST ? `1x wins by ${fmtPct(ST / LT - 1)}` : "tie");
    document.getElementById("rd-verdict").textContent = verdict;

    drawCurve(state, T, c, phi, beta);
  }

  function drawCurve(state, T, c, phi, beta) {
    const S0 = state.spot;
    const targets = (state.targets && state.targets.length) ? state.targets : [150, 450, 700];
    const sigmas = [];
    for (let s = 0.01; s <= 2.5 + 1e-9; s += 0.02) sigmas.push(s);
    const traces = targets.map(K => {
      const gross = K / S0;
      const ratio = sigmas.map(sig => {
        const qv = sig * sig * T;
        const LT = leveredTerminal(gross, qv, T, c, phi, beta);
        return LT / gross;
      });
      const sigmaBe = breakevenSigma(K, S0, T, c, phi, beta);
      return {
        x: sigmas, y: ratio, mode: "lines", name: `K=$${K} σ_be=${sigmaBe.toFixed(2)}`,
      };
    });
    traces.push({
      x: [0, 2.5], y: [1, 1], mode: "lines", line: {color: "#7a8696", dash: "dot"},
      name: "tie", hoverinfo: "skip",
    });
    const reg = state.regimes || {};
    const shapes = [];
    if (reg.low && reg.high) {
      shapes.push({type: "rect", xref: "x", yref: "paper",
                   x0: reg.low, x1: reg.high, y0: 0, y1: 1,
                   fillcolor: "rgba(124,224,164,0.08)", line: {width: 0}});
    }
    if (reg.base) {
      shapes.push({type: "line", xref: "x", yref: "paper",
                   x0: reg.base, x1: reg.base, y0: 0, y1: 1,
                   line: {color: "#7ce0a4", width: 1, dash: "dash"}});
    }
    Plotly.react("plot-curve", traces, {
      title: {text: `Breakeven curves at T=${T.toFixed(2)}y, β=${beta}`,
              font: {size: 14, color: "#c2c9d4"}},
      xaxis: {title: "realized annualized σ", gridcolor: "#1f2731", zerolinecolor: "#2a3441"},
      yaxis: {title: "L_T/L_0  ÷  S_T/S_0", gridcolor: "#1f2731", zerolinecolor: "#2a3441"},
      shapes: shapes,
      template: "plotly_dark",
      margin: {t: 40, l: 60, r: 20, b: 50},
      paper_bgcolor: "rgba(0,0,0,0)",
      plot_bgcolor: "rgba(0,0,0,0)",
      font: {color: "#c2c9d4", family: "Inter, sans-serif"},
      legend: {bgcolor: "rgba(0,0,0,0)", bordercolor: "#1f2731", borderwidth: 1},
      showlegend: true,
    }, {responsive: true, displaylogo: false});
  }

  async function init() {
    const state = (await loadInputs()) || {spot: 58, targets: [150, 450, 700]};
    if (state.spot) {
      // Re-anchor the K slider around the current spot
      els.K.max = Math.max(800, Math.round(state.spot * 15));
      els.K.value = Math.round(state.spot * 2.5);
      els["out-K"].textContent = els.K.value;
    }
    ids.forEach(id => els[id].addEventListener("input", () => recompute(state)));
    recompute(state);
  }
  document.addEventListener("DOMContentLoaded", init);

  // Exported for the parity test runner (if loaded under jsdom/node).
  if (typeof module !== "undefined") {
    module.exports = { breakevenSigma, leveredTerminal };
  }
})();
