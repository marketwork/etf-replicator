'use strict';

// ── Constants ─────────────────────────────────────────────────────────────────
const ETF_CLR  = '#4A90D9';
const REP_CLR  = '#E86B3A';
const GRID     = 'rgba(200,200,200,0.08)';
const AX_COLOR = '#8b949e';

const BASE = {
  paper_bgcolor: 'transparent',
  plot_bgcolor:  'transparent',
  font:   { color: '#e6edf3', size: 12 },
  xaxis:  { showgrid: true, gridcolor: GRID, zeroline: false, color: AX_COLOR, linecolor: '#30363d' },
  yaxis:  { showgrid: true, gridcolor: GRID, zeroline: false, color: AX_COLOR, linecolor: '#30363d' },
  legend: { orientation: 'h', yanchor: 'bottom', y: 1.02, xanchor: 'right', x: 1, bgcolor: 'transparent' },
  margin: { l: 52, r: 20, t: 32, b: 44 },
};
const CFG = { responsive: true, displayModeBar: false };

const PIE_COLORS = [
  '#4A90D9','#E86B3A','#50C878','#FFD700','#C678DD',
  '#56B6C2','#E06C75','#98C379','#D19A66','#ABB2BF','#61AFEF',
];

let _result = null;
let _mode   = 'optimise';

// ── Init ──────────────────────────────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', () => {
  const today = new Date();
  const t3y   = new Date(today); t3y.setFullYear(today.getFullYear() - 3);
  $('end').value   = fmt(today);
  $('start').value = fmt(t3y);

  bindSlider('n-stocks',  'n-stocks-val');
  bindSlider('train-pct', 'train-pct-val');

  $('run-btn').addEventListener('click', runOptimisation);
  $('export-btn').addEventListener('click', exportCSV);

  document.querySelectorAll('.tab-btn').forEach(btn =>
    btn.addEventListener('click', () => switchTab(btn.dataset.tab))
  );

  // Mode switcher
  document.querySelectorAll('.mode-btn').forEach(btn =>
    btn.addEventListener('click', () => setMode(btn.dataset.mode))
  );

  // Live portfolio preview
  $('portfolio-input').addEventListener('input', updatePortfolioPreview);
});

function fmt(d) { return d.toISOString().split('T')[0]; }
function $(id)  { return document.getElementById(id); }

function bindSlider(sliderId, labelId) {
  const sl = $(sliderId), lb = $(labelId);
  sl.addEventListener('input', () => { lb.textContent = sl.value; });
}

// ── Mode ──────────────────────────────────────────────────────────────────────
function setMode(mode) {
  _mode = mode;
  document.querySelectorAll('.mode-btn').forEach(b =>
    b.classList.toggle('active', b.dataset.mode === mode)
  );
  document.querySelectorAll('.mode-section').forEach(s => {
    s.style.display = s.dataset.for === mode ? 'flex' : 'none';
  });
  $('run-label').textContent = mode === 'evaluate' ? 'Evaluate portfolio' : 'Run optimisation';
  showView('landing');
}

// ── Views ─────────────────────────────────────────────────────────────────────
function showView(name) {
  document.querySelectorAll('.view').forEach(v => v.classList.remove('active'));
  $(`view-${name}`).classList.add('active');
}

function setProgress(pct, text) {
  $('progress-fill').style.width = pct + '%';
  $('progress-text').textContent  = text || '';
}

function showError(msg) {
  $('error-msg').textContent = msg;
  showView('error');
  setRunning(false);
}

function setRunning(on) {
  $('run-btn').disabled     = on;
  $('run-label').textContent = on ? 'Running…' : 'Run optimisation';
  $('run-spinner').style.display = on ? 'inline-block' : 'none';
}

// ── Tabs ──────────────────────────────────────────────────────────────────────
function switchTab(tabId) {
  document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
  document.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));
  document.querySelector(`[data-tab="${tabId}"]`).classList.add('active');
  $(tabId).classList.add('active');
  // Reflow Plotly charts now that they're visible
  setTimeout(() => {
    $(tabId).querySelectorAll('.js-plotly-plot').forEach(el => Plotly.Plots.resize(el));
  }, 30);
}

// ── Run ───────────────────────────────────────────────────────────────────────
function runOptimisation() {
  if (_mode === 'evaluate') { runEvaluate(); return; }

  const etf = $('etf').value.trim().toUpperCase();
  if (!etf) { alert('Enter an ETF ticker.'); return; }

  const customRaw = $('custom-universe').value.trim();
  const custom    = customRaw
    ? customRaw.split(/[\s,\n]+/).map(s => s.trim().toUpperCase()).filter(Boolean)
    : [];

  setRunning(true);
  showView('loading');
  setProgress(0, 'Initialising…');

  streamSSE('/api/optimize', {
    etf,
    start:           $('start').value,
    end:             $('end').value,
    n_stocks:        +$('n-stocks').value,
    method:          $('method').value,
    train_frac:      +$('train-pct').value / 100,
    min_coverage:    0.80,
    custom_universe: custom,
  });
}

function runEvaluate() {
  const etf  = $('etf').value.trim().toUpperCase();
  const text = $('portfolio-input').value.trim();
  if (!etf)  { alert('Enter an ETF ticker.'); return; }
  if (!text) { alert('Enter at least one portfolio ticker.'); return; }

  const portfolio = parsePortfolio(text);
  if (!Object.keys(portfolio).length) {
    alert('Could not parse any tickers. Check the format and try again.');
    return;
  }

  setRunning(true);
  showView('loading');
  setProgress(0, 'Downloading prices…');

  streamSSE('/api/evaluate', {
    etf,
    portfolio,
    start:      $('start').value,
    end:        $('end').value,
    train_frac: +$('train-pct').value / 100,
  });
}

function streamSSE(url, body) {
  fetch(url, {
    method:  'POST',
    headers: { 'Content-Type': 'application/json' },
    body:    JSON.stringify(body),
  })
  .then(res => {
    const reader  = res.body.getReader();
    const decoder = new TextDecoder();
    let buf = '';
    (function pump() {
      reader.read().then(({ done, value }) => {
        if (done) return;
        buf += decoder.decode(value, { stream: true });
        let nl;
        while ((nl = buf.indexOf('\n')) !== -1) {
          const line = buf.slice(0, nl).trim();
          buf = buf.slice(nl + 1);
          if (line.startsWith('data: ')) {
            try { onMsg(JSON.parse(line.slice(6))); } catch (_) {}
          }
        }
        pump();
      }).catch(e => showError('Connection lost: ' + e.message));
    })();
  })
  .catch(e => showError(e.message));
}

// ── Portfolio parser ──────────────────────────────────────────────────────────
function parsePortfolio(text) {
  const lines = text.split(/\n/).map(s => s.trim()).filter(Boolean);
  const entries = [];

  for (const line of lines) {
    const parts = line.replace(/,/g, ' ').trim().split(/\s+/);
    if (!parts[0]) continue;
    const ticker = parts[0].toUpperCase();
    let weight = null;
    if (parts.length >= 2) {
      const raw = parts[1].replace('%', '');
      const v   = parseFloat(raw);
      if (!isNaN(v)) weight = parts[1].includes('%') ? v / 100 : v;
    }
    entries.push({ ticker, weight });
  }

  if (!entries.length) return {};

  const hasWeights = entries.some(e => e.weight !== null);
  const result = {};
  if (hasWeights) {
    const total = entries.reduce((s, e) => s + (e.weight ?? 0), 0) || 1;
    entries.forEach(e => { result[e.ticker] = (e.weight ?? 0) / total; });
  } else {
    entries.forEach(e => { result[e.ticker] = 1 / entries.length; });
  }
  return result;
}

function updatePortfolioPreview() {
  const portfolio = parsePortfolio($('portfolio-input').value);
  const wrap = $('portfolio-preview');
  const keys = Object.keys(portfolio);
  if (!keys.length) { wrap.style.display = 'none'; return; }

  wrap.style.display = 'flex';
  wrap.innerHTML = keys.map(t =>
    `<span class="ptag">${t} <b>${p1(portfolio[t])}</b></span>`
  ).join('');
}

function onMsg(msg) {
  if (msg.type === 'progress') {
    setProgress(msg.pct, msg.text);
  } else if (msg.type === 'done') {
    _result = msg.result;
    renderResults(msg.result);
    setRunning(false);
  } else if (msg.type === 'error') {
    showError(msg.message);
  }
}

// ── Render ────────────────────────────────────────────────────────────────────
function renderResults(d) {
  setProgress(100, 'Done!');

  const isEval = d.mode === 'evaluate';
  $('results-title').textContent = isEval
    ? `Portfolio vs ${d.etf} — ${d.n_stocks} stock${d.n_stocks !== 1 ? 's' : ''}`
    : `${d.etf} Replica — ${d.n_stocks} stock${d.n_stocks !== 1 ? 's' : ''}`;
  $('results-caption').textContent =
    `${isEval ? 'Portfolio' : 'Universe'}: ${d.universe_label} · Train: ${d.dates[0]} → ${d.train_cutoff} · OOS: ${d.train_cutoff} → ${d.dates.at(-1)}`;

  // Show missing tickers warning for evaluate mode
  const existingWarn = document.querySelector('.missing-warn');
  if (existingWarn) existingWarn.remove();
  if (isEval && d.missing && d.missing.length) {
    const warn = document.createElement('div');
    warn.className = 'missing-warn';
    warn.textContent = `No price data found for: ${d.missing.join(', ')}. These were excluded.`;
    $('results-title').insertAdjacentElement('afterend', warn);
  }

  renderMetrics(d);

  // Switch to first tab, render all charts
  switchTab('tab-perf');
  renderPerf(d);
  renderHoldings(d);
  renderAnalysis(d);

  showView('results');
}

// ── Metrics ───────────────────────────────────────────────────────────────────
function renderMetrics(d) {
  const delta  = d.oos_tracking_error - d.tracking_error;
  const dclass = delta >  0.005 ? 'delta-bad'
               : delta < -0.005 ? 'delta-good'
               : 'delta-neu';
  const dtext  = (delta >= 0 ? '+' : '') + p2(delta) + ' vs IS';

  $('metrics-row').innerHTML = [
    { label: 'Stocks selected',      value: d.n_stocks,                         delta: '' },
    { label: 'Tracking error (IS)',  value: p2(d.tracking_error),               delta: '' },
    { label: 'Tracking error (OOS)', value: p2(d.oos_tracking_error),           delta: dtext, dc: dclass },
    { label: 'R² in-sample',         value: d.r_squared.toFixed(4),             delta: '' },
    { label: 'Correlation (IS)',      value: d.correlation.toFixed(4),           delta: '' },
    { label: 'Correlation (OOS)',     value: d.oos_correlation.toFixed(4),       delta: '' },
  ].map(c => `
    <div class="metric-card">
      <div class="metric-label">${c.label}</div>
      <div class="metric-value">${c.value}</div>
      ${c.delta ? `<div class="metric-delta ${c.dc}">${c.delta}</div>` : ''}
    </div>`).join('');
}

// ── Performance charts ────────────────────────────────────────────────────────
function renderPerf(d) {
  const cutDate = d.train_cutoff;

  Plotly.newPlot('chart-cum', [
    { x: d.dates, y: d.etf_cum,  name: d.etf,
      type: 'scatter', mode: 'lines', line: { color: ETF_CLR, width: 2 } },
    { x: d.dates, y: d.port_cum, name: `Replica (${d.n_stocks})`,
      type: 'scatter', mode: 'lines', line: { color: REP_CLR, width: 2 } },
  ], {
    ...BASE,
    height: 400,
    shapes: [{ type: 'line', x0: cutDate, x1: cutDate,
               yref: 'paper', y0: 0, y1: 1,
               line: { color: '#555', dash: 'dot', width: 1 } }],
    annotations: [{ x: cutDate, yref: 'paper', y: 0.98,
                    text: 'OOS ▶', showarrow: false,
                    font: { color: '#666', size: 11 }, xanchor: 'left' }],
    yaxis: { ...BASE.yaxis, title: 'Growth of $1' },
  }, CFG);

  const rte = d.rolling_te.map(v => v > 0 ? +(v * 100).toFixed(3) : null);
  Plotly.newPlot('chart-rte', [
    { x: d.dates, y: rte,
      type: 'scatter', mode: 'lines', fill: 'tozeroy',
      fillcolor: 'rgba(232,107,58,0.12)', line: { color: REP_CLR, width: 1.5 },
      connectgaps: false, name: 'Rolling TE (%)' },
  ], {
    ...BASE,
    height: 260,
    shapes: [{ type: 'line', xref: 'paper', x0: 0, x1: 1,
               y0: d.tracking_error * 100, y1: d.tracking_error * 100,
               line: { color: ETF_CLR, dash: 'dash', width: 1 } }],
    yaxis: { ...BASE.yaxis, title: 'TE (%)' },
  }, CFG);
}

// ── Holdings ──────────────────────────────────────────────────────────────────
function renderHoldings(d) {
  const rows = d.tickers
    .map((t, i) => ({ t, w: d.weights[i], m: d.meta[t] || { name: t, sector: '—', market_cap_B: 0 } }))
    .sort((a, b) => b.w - a.w);

  const maxW = rows[0].w;

  $('hold-title').textContent = `${d.n_stocks} selected stocks`;

  $('holdings-table').innerHTML = `
    <table class="htable">
      <thead><tr>
        <th>Ticker</th><th>Name</th><th>Sector</th><th style="text-align:right">Weight</th>
      </tr></thead>
      <tbody>
      ${rows.map(({ t, w, m }) => `
        <tr>
          <td><span class="ticker">${t}</span></td>
          <td style="color:var(--muted)">${m.name}</td>
          <td style="color:var(--muted)">${m.sector}</td>
          <td style="text-align:right;min-width:80px">
            <div>${p2(w)}</div>
            <div class="wbar-bg"><div class="wbar-fill" style="width:${(w/maxW*100).toFixed(1)}%"></div></div>
          </td>
        </tr>`).join('')}
      </tbody>
    </table>`;

  // Weights bar
  Plotly.newPlot('chart-weights', [{
    x: rows.map(r => +(r.w * 100).toFixed(2)),
    y: rows.map(r => r.t),
    type: 'bar', orientation: 'h',
    marker: { color: REP_CLR },
    text: rows.map(r => p1(r.w)),
    textposition: 'outside',
    textfont: { color: '#e6edf3', size: 11 },
  }], {
    ...BASE,
    height: Math.max(380, rows.length * 30),
    xaxis: { ...BASE.xaxis, title: 'Weight (%)' },
    yaxis: { ...BASE.yaxis, autorange: 'reversed', showgrid: false },
    margin: { ...BASE.margin, l: 60 },
  }, CFG);

  // Sector pie
  const sec = {};
  rows.forEach(({ w, m }) => { sec[m.sector] = (sec[m.sector] || 0) + w; });
  const secArr = Object.entries(sec).sort((a, b) => b[1] - a[1]);

  Plotly.newPlot('chart-sector', [{
    labels: secArr.map(([s]) => s),
    values: secArr.map(([, v]) => +(v * 100).toFixed(2)),
    type: 'pie', hole: 0.42,
    textinfo: 'label+percent',
    textfont: { color: '#e6edf3', size: 11 },
    marker: { colors: PIE_COLORS },
  }], {
    ...BASE,
    height: 280,
    showlegend: false,
    margin: { l: 10, r: 10, t: 10, b: 10 },
  }, CFG);
}

// ── Analysis ──────────────────────────────────────────────────────────────────
function renderAnalysis(d) {
  const ep = d.etf_returns.map(v => +(v * 100).toFixed(4));
  const pp = d.port_returns.map(v => +(v * 100).toFixed(4));
  const df = d.port_returns.map((v, i) => +((v - d.etf_returns[i]) * 100).toFixed(4));

  const rng = Math.max(Math.abs(Math.min(...ep)), Math.abs(Math.max(...ep))) * 1.1;

  Plotly.newPlot('chart-scatter', [
    { x: [-rng, rng], y: [-rng, rng],
      type: 'scatter', mode: 'lines',
      line: { color: '#555', dash: 'dot', width: 1 }, showlegend: false },
    { x: ep, y: pp,
      type: 'scatter', mode: 'markers',
      marker: { size: 3, color: REP_CLR, opacity: 0.45 }, name: 'Daily returns' },
  ], {
    ...BASE,
    height: 340,
    xaxis: { ...BASE.xaxis, title: `${d.etf} return (%)` },
    yaxis: { ...BASE.yaxis, title: 'Replica return (%)' },
  }, CFG);

  Plotly.newPlot('chart-diff', [{
    x: d.dates, y: df,
    type: 'bar',
    marker: { color: df.map(v => v >= 0 ? 'rgba(63,185,80,0.65)' : 'rgba(248,81,73,0.65)') },
    name: 'Daily difference',
  }], {
    ...BASE,
    height: 340,
    yaxis: { ...BASE.yaxis, title: 'Difference (% pts)' },
    bargap: 0,
  }, CFG);

  // Performance table
  const p = d.perf;
  const sh = (r, v) => v > 0 ? (r / v).toFixed(2) : '—';

  $('perf-table').innerHTML = `
    <div class="ptable-wrap">
      <table class="ptable">
        <thead><tr>
          <th></th>
          <th>Ann. Return</th>
          <th>Ann. Volatility</th>
          <th>Max Drawdown</th>
          <th>Sharpe (0% rf)</th>
        </tr></thead>
        <tbody>
          <tr>
            <td>${d.etf}</td>
            <td>${colored(p.etf.ann_ret)}</td>
            <td>${p2(p.etf.ann_vol)}</td>
            <td>${colored(p.etf.max_dd)}</td>
            <td>${sh(p.etf.ann_ret, p.etf.ann_vol)}</td>
          </tr>
          <tr>
            <td>Replica (${d.n_stocks})</td>
            <td>${colored(p.port.ann_ret)}</td>
            <td>${p2(p.port.ann_vol)}</td>
            <td>${colored(p.port.max_dd)}</td>
            <td>${sh(p.port.ann_ret, p.port.ann_vol)}</td>
          </tr>
        </tbody>
      </table>
    </div>`;
}

// ── Export ────────────────────────────────────────────────────────────────────
function exportCSV() {
  if (!_result) return;
  const d = _result;
  const rows = [['Ticker','Weight','Weight_%','Name','Sector','MarketCap_B']];
  d.tickers.forEach((t, i) => {
    const m = d.meta[t] || {};
    rows.push([t, d.weights[i].toFixed(6), (d.weights[i]*100).toFixed(2),
               m.name||t, m.sector||'—', m.market_cap_B||0]);
  });
  const csv  = rows.map(r => r.join(',')).join('\n');
  const blob = new Blob([csv], { type: 'text/csv' });
  const a    = Object.assign(document.createElement('a'), {
    href:     URL.createObjectURL(blob),
    download: `${d.etf}_replica_${d.n_stocks}stocks.csv`,
  });
  a.click();
  URL.revokeObjectURL(a.href);
}

// ── Formatting helpers ────────────────────────────────────────────────────────
function p2(v) { return (v * 100).toFixed(2) + '%'; }
function p1(v) { return (v * 100).toFixed(1) + '%'; }
function colored(v) {
  const cls = v >= 0 ? 'pos' : 'neg';
  return `<span class="${cls}">${p2(v)}</span>`;
}
