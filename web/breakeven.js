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

  function resolveEls() {
    ids.forEach(id => {
      els[id] = document.getElementById("ctl-" + id);
      els["out-" + id] = document.getElementById("out-" + id);
    });
  }

  async function loadInputs() {
    try {
      const r = await fetch("./inputs.json");
      if (!r.ok) return null;
      return await r.json();
    } catch (_) { return null; }
  }

  function fmtPct(x) { return (100 * x).toFixed(2) + "%"; }

  // Portfolio panel — currency state and formatters.
  // Currency is a display unit; numbers are used as-is (no FX conversion).
  let currency = "USD";
  function parseAmount(s) {
    if (s == null) return 0;
    const cleaned = String(s).replace(/[^0-9.]/g, "");
    const n = parseFloat(cleaned);
    return Number.isFinite(n) && n >= 0 ? n : 0;
  }
  function fmtCurrency(v) {
    if (!Number.isFinite(v)) return "—";
    if (currency === "KRW") return "₩" + Math.round(v).toLocaleString("ko-KR");
    return "$" + v.toLocaleString("en-US", {maximumFractionDigits: 0});
  }
  function fmtCurrencySigned(v) {
    if (!Number.isFinite(v)) return "—";
    const sign = v >= 0 ? "+" : "−";
    return sign + fmtCurrency(Math.abs(v));
  }

  const I18N = {
    en: {
      verdict2x: pct => `2x wins by ${pct}`,
      verdict1x: pct => `1x wins by ${pct}`,
      tie: "tie",
      xAxis: "realized annualized σ",
      yAxis: "L_T/L_0  ÷  S_T/S_0",
      todayMarker: (w, sig, K) => `today σ_${w}d=${sig.toFixed(2)} · K=$${K}`,
      tieTrace: "tie",
      pfTimeAxis: "holding period t (years)",
      pfValueAxis: "portfolio value",
      pf1xName: "1x position",
      pf2xName: "2x position",
      pfStartLine: "start",
      shTimeAxis: "date",
      shValueAxis: "annualized σ",
      shWin: w => `σ·${w}d`,
      shEwma: "EWMA(0.94)",
    },
    ko: {
      verdict2x: pct => `2x 우세 ${pct}`,
      verdict1x: pct => `1x 우세 ${pct}`,
      tie: "동률",
      xAxis: "실현 연율 σ",
      yAxis: "L_T/L_0  ÷  S_T/S_0",
      todayMarker: (w, sig, K) => `오늘 σ_${w}일=${sig.toFixed(2)} · K=$${K}`,
      tieTrace: "동률",
      pfTimeAxis: "보유기간 t (년)",
      pfValueAxis: "포트폴리오 평가액",
      pf1xName: "1x 포지션",
      pf2xName: "2x 포지션",
      pfStartLine: "시작",
      shTimeAxis: "날짜",
      shValueAxis: "연율 σ",
      shWin: w => `σ·${w}일`,
      shEwma: "EWMA(0.94)",
    },
  };
  function t() { return I18N[document.documentElement.lang === "ko" ? "ko" : "en"]; }

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
    const L = t();
    let verdict, state2x;
    if (LT > ST)      { verdict = L.verdict2x(fmtPct(LT / ST - 1)); state2x = "2x"; }
    else if (LT < ST) { verdict = L.verdict1x(fmtPct(ST / LT - 1)); state2x = "1x"; }
    else              { verdict = L.tie;                            state2x = "tie"; }
    document.getElementById("rd-verdict").textContent = verdict;
    const pill = document.getElementById("verdict-pill");
    if (pill) pill.setAttribute("data-state", state2x);

    drawCurve(state, T, c, phi, beta);
    drawPortfolio(state, K, T, sigma, c, phi, beta);
  }

  function drawPortfolio(state, K, T, sigma, c, phi, beta) {
    const assetEl = document.getElementById("ctl-asset");
    if (!assetEl) return;
    const asset = parseAmount(assetEl.value);
    const S0 = state.spot;
    if (!(S0 > 0) || !(T > 0)) return;

    const gross = K / S0;
    const N = 80;
    const times = new Array(N + 1);
    const v1x = new Array(N + 1);
    const v2x = new Array(N + 1);
    for (let i = 0; i <= N; i++) {
      const tau = i / N;
      const tHere = tau * T;
      const stock = Math.pow(gross, tau);
      const qv = sigma * sigma * tHere;
      const lev = Math.pow(gross, beta * tau) * Math.exp(
        -0.5 * (beta * beta - beta) * qv
        - (beta - 1) * c * tHere
        - phi * tHere
      );
      times[i] = tHere;
      v1x[i]   = asset * stock;
      v2x[i]   = asset * lev;
    }

    const end1x = v1x[N];
    const end2x = v2x[N];
    const edge = end2x - end1x;
    const set = (id, val) => { const el = document.getElementById(id); if (el) el.textContent = val; };
    set("pr-start", fmtCurrency(asset));
    set("pr-1x",    fmtCurrency(end1x));
    set("pr-2x",    fmtCurrency(end2x));
    set("pr-edge",  fmtCurrencySigned(edge));
    const edgeEl = document.getElementById("pr-edge");
    if (edgeEl) {
      edgeEl.classList.toggle("positive", edge > 0);
      edgeEl.classList.toggle("negative", edge < 0);
    }

    const L = t();
    const hoverFmt = currency === "KRW" ? "₩%{y:,.0f}" : "$%{y:,.0f}";
    Plotly.react("plot-portfolio", [
      {
        x: [0, T], y: [asset, asset], mode: "lines",
        line: {color: "#2a3441", dash: "dot", width: 1},
        name: L.pfStartLine, hoverinfo: "skip",
      },
      {
        x: times, y: v1x, mode: "lines", name: L.pf1xName,
        line: {color: "#7a8696", width: 2},
        hovertemplate: `t=%{x:.2f}y<br>1x=${hoverFmt}<extra></extra>`,
      },
      {
        x: times, y: v2x, mode: "lines", name: L.pf2xName,
        line: {color: "#7ce0a4", width: 3},
        hovertemplate: `t=%{x:.2f}y<br>2x=${hoverFmt}<extra></extra>`,
      },
    ], {
      xaxis: {title: L.pfTimeAxis, gridcolor: "#1f2731", zerolinecolor: "#2a3441"},
      yaxis: {title: L.pfValueAxis, gridcolor: "#1f2731", zerolinecolor: "#2a3441",
              tickformat: ",", tickprefix: currency === "KRW" ? "₩" : "$"},
      template: "plotly_dark",
      margin: {t: 18, l: 90, r: 20, b: 50},
      paper_bgcolor: "rgba(0,0,0,0)",
      plot_bgcolor: "rgba(0,0,0,0)",
      font: {color: "#c2c9d4", family: "Inter, sans-serif"},
      legend: {bgcolor: "rgba(0,0,0,0)", bordercolor: "#1f2731", borderwidth: 1,
               x: 0.02, y: 0.98},
      showlegend: true,
    }, {responsive: true, displayModeBar: false});
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
    const L = t();
    traces.push({
      x: [0, 2.5], y: [1, 1], mode: "lines", line: {color: "#7a8696", dash: "dot"},
      name: L.tieTrace, hoverinfo: "skip",
    });

    // Snapshot marker: σ_60d (or σ at the largest available window),
    // evaluated at the current K (slider) so the dot lives on its curve.
    const sigNow = state.sigma_now || {};
    const anchor = sigNow.d60 || sigNow.d120 || sigNow.d20;
    if (anchor && typeof anchor.sigma === "number") {
      const Kcur = +els.K.value;
      const grossCur = Kcur / S0;
      const qvNow = anchor.sigma * anchor.sigma * T;
      const yNow = leveredTerminal(grossCur, qvNow, T, c, phi, beta) / grossCur;
      traces.push({
        x: [anchor.sigma], y: [yNow], mode: "markers",
        marker: {size: 12, color: "#7ce0a4", line: {color: "#0a0e14", width: 2},
                 symbol: "circle"},
        name: L.todayMarker(anchor.window, anchor.sigma, Kcur),
        hovertemplate: `σ=%{x:.3f}<br>L_T/L_0 ÷ S_T/S_0=%{y:.3f}<extra></extra>`,
      });
    }

    const reg = state.regimes || {};
    const shapes = [];
    if (anchor && typeof anchor.sigma === "number") {
      shapes.push({type: "line", xref: "x", yref: "paper",
                   x0: anchor.sigma, x1: anchor.sigma, y0: 0, y1: 1,
                   line: {color: "#7ce0a4", width: 1.5, dash: "solid"}});
    }
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
      // Title intentionally omitted — slider outputs above already show T & β.
      xaxis: {title: L.xAxis, gridcolor: "#1f2731", zerolinecolor: "#2a3441"},
      yaxis: {title: L.yAxis, gridcolor: "#1f2731", zerolinecolor: "#2a3441"},
      shapes: shapes,
      template: "plotly_dark",
      margin: {t: 18, l: 60, r: 20, b: 50},
      paper_bgcolor: "rgba(0,0,0,0)",
      plot_bgcolor: "rgba(0,0,0,0)",
      font: {color: "#c2c9d4", family: "Inter, sans-serif"},
      legend: {bgcolor: "rgba(0,0,0,0)", bordercolor: "#1f2731", borderwidth: 1},
      showlegend: true,
    }, {responsive: true, displayModeBar: false});
  }

  function fmtSig(v) { return (v == null || !Number.isFinite(v)) ? "—" : v.toFixed(3); }

  function paintSigmaHistory(state) {
    const hist = state.sigma_history;
    if (!hist) return;
    const L = t();
    const windows = hist.windows || [20, 60, 120];

    // --- distribution stats table (one row per window) ---
    const statsBody = document.getElementById("sigma-stats-body");
    if (statsBody && hist.stats) {
      const cols = ["latest", "mean", "min", "p10", "p25", "p50", "p75", "p90", "max"];
      statsBody.innerHTML = "";
      windows.forEach(w => {
        const d = hist.stats["d" + w];
        if (!d) return;
        const tr = document.createElement("tr");
        const cells = [L.shWin(w),
          ...cols.map(k => fmtSig(d[k])),
          (d.n != null ? String(d.n) : "—")];
        cells.forEach((txt, i) => {
          const td = document.createElement("td");
          td.textContent = txt;
          if (i === 0) td.className = "sig-win";
          if (i === 1) td.className = "sig-latest";   // latest column emphasized
          tr.appendChild(td);
        });
        statsBody.appendChild(tr);
      });
    }

    // --- month-end numeric table ---
    const monBody = document.getElementById("sigma-monthly-body");
    if (monBody && hist.monthly) {
      const m = hist.monthly;
      const s = m.series || {};
      monBody.innerHTML = "";
      // newest first so the most recent months are visible without scrolling
      for (let i = (m.dates || []).length - 1; i >= 0; i--) {
        const tr = document.createElement("tr");
        const vals = [m.dates[i],
          fmtSig((s.d20 || [])[i]), fmtSig((s.d60 || [])[i]),
          fmtSig((s.d120 || [])[i]), fmtSig((s.ewma || [])[i])];
        vals.forEach((txt, j) => {
          const td = document.createElement("td");
          td.textContent = txt;
          if (j === 0) td.className = "sig-win";
          tr.appendChild(td);
        });
        monBody.appendChild(tr);
      }
    }

    // --- interactive hover-exact daily chart ---
    const div = document.getElementById("plot-sigma-history");
    if (div && hist.dates && hist.series) {
      const x = hist.dates;
      const colors = {d20: "#7a8696", d60: "#7ce0a4", d120: "#6aa9ff", ewma: "#e0b87c"};
      const widths = {d20: 1, d60: 2.2, d120: 1.6, ewma: 1.4};
      const dashes = {ewma: "dot"};
      const traces = [];
      windows.forEach(w => {
        const key = "d" + w;
        if (!hist.series[key]) return;
        traces.push({
          x, y: hist.series[key], mode: "lines", name: L.shWin(w),
          line: {color: colors[key] || "#7a8696", width: widths[key] || 1.5},
          connectgaps: false,
          hovertemplate: `%{x}<br>${L.shWin(w)}=%{y:.3f}<extra></extra>`,
        });
      });
      if (hist.series.ewma) {
        traces.push({
          x, y: hist.series.ewma, mode: "lines", name: L.shEwma,
          line: {color: colors.ewma, width: widths.ewma, dash: dashes.ewma},
          connectgaps: false,
          hovertemplate: `%{x}<br>${L.shEwma}=%{y:.3f}<extra></extra>`,
        });
      }
      // regime bands for visual reference (same low/base/high as the curve panel)
      const reg = state.regimes || {};
      const shapes = [];
      if (reg.low && reg.high) {
        shapes.push({type: "rect", xref: "paper", yref: "y",
                     x0: 0, x1: 1, y0: reg.low, y1: reg.high,
                     fillcolor: "rgba(124,224,164,0.06)", line: {width: 0}});
      }
      if (reg.base) {
        shapes.push({type: "line", xref: "paper", yref: "y",
                     x0: 0, x1: 1, y0: reg.base, y1: reg.base,
                     line: {color: "#7ce0a4", width: 1, dash: "dash"}});
      }
      Plotly.react("plot-sigma-history", traces, {
        xaxis: {title: L.shTimeAxis, gridcolor: "#1f2731", zerolinecolor: "#2a3441",
                type: "date"},
        yaxis: {title: L.shValueAxis, gridcolor: "#1f2731", zerolinecolor: "#2a3441",
                rangemode: "tozero"},
        shapes,
        hovermode: "x unified",
        template: "plotly_dark",
        margin: {t: 18, l: 60, r: 20, b: 50},
        paper_bgcolor: "rgba(0,0,0,0)",
        plot_bgcolor: "rgba(0,0,0,0)",
        font: {color: "#c2c9d4", family: "Inter, sans-serif"},
        legend: {bgcolor: "rgba(0,0,0,0)", bordercolor: "#1f2731", borderwidth: 1,
                 orientation: "h", x: 0, y: 1.08},
        showlegend: true,
      }, {responsive: true, displayModeBar: false});
    }
  }

  function paintSnapshot(state) {
    // Spot lives in the header (build-time template substitution), not here.
    const sn = state.sigma_now || {};
    [20, 60, 120].forEach(w => {
      const d = sn["d" + w];
      const sig = document.getElementById("snap-sig" + w);
      const pct = document.getElementById("snap-pct" + w);
      const bar = document.getElementById("snap-bar" + w);
      if (!d) {
        if (sig) sig.textContent = "—";
        if (pct) pct.textContent = "—";
        return;
      }
      if (sig) sig.textContent = d.sigma.toFixed(2);
      if (pct) pct.textContent = Math.round(d.pct * 100) + "th";
      if (bar) bar.style.width = Math.max(2, Math.round(d.pct * 100)) + "%";
    });
  }

  async function init() {
    resolveEls();
    const state = (await loadInputs()) || {spot: 58, targets: [150, 450, 700]};
    if (state.spot) {
      // Re-anchor the K slider around the current spot
      els.K.max = Math.max(800, Math.round(state.spot * 15));
      els.K.value = Math.round(state.spot * 2.5);
      els["out-K"].textContent = els.K.value;
    }
    paintSnapshot(state);
    paintSigmaHistory(state);
    ids.forEach(id => els[id].addEventListener("input", () => recompute(state)));

    // Portfolio panel inputs: amount + currency toggle. Both feed into recompute
    // (cheap — only redraws plot-portfolio, not the full surface).
    const assetEl = document.getElementById("ctl-asset");
    if (assetEl) assetEl.addEventListener("input", () => recompute(state));
    document.querySelectorAll(".currency-toggle button").forEach(b => {
      b.addEventListener("click", () => {
        currency = b.dataset.cur === "KRW" ? "KRW" : "USD";
        document.querySelectorAll(".currency-toggle button").forEach(o => {
          o.classList.toggle("active", o.dataset.cur === currency);
        });
        recompute(state);
      });
    });

    recompute(state);

    // i18n swaps innerHTML of containers, which can recreate child nodes (the
    // snap-pct spans live inside translated snap-subs). Re-paint + recompute on
    // every lang change so percentile values, verdict text, and Plotly labels
    // all stay current.
    document.addEventListener("langchange", () => {
      paintSnapshot(state);
      paintSigmaHistory(state);
      recompute(state);
    });
  }
  document.addEventListener("DOMContentLoaded", init);

  // Exported for the parity test runner (if loaded under jsdom/node).
  if (typeof module !== "undefined") {
    module.exports = { breakevenSigma, leveredTerminal };
  }
})();
