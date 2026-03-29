/* ML Drift Monitor - Dashboard JS
 * Fetches data from the API and renders Plotly charts.
 */

// ---------------------------------------------------------------------------
// Plotly base layout
// ---------------------------------------------------------------------------
function norrLayout(overrides) {
  return Object.assign({
    paper_bgcolor: '#ffffff',
    plot_bgcolor: '#fff9f3',
    font: { family: "'Satoshi', system-ui, sans-serif", color: '#1d1d1d', size: 12 },
    margin: { t: 24, b: 48, l: 56, r: 24 },
    xaxis: {
      gridcolor: 'rgba(29,29,29,0.07)',
      linecolor: 'rgba(29,29,29,0.18)',
      tickfont: { size: 11, color: 'rgba(29,29,29,0.55)' },
    },
    yaxis: {
      gridcolor: 'rgba(29,29,29,0.07)',
      linecolor: 'rgba(29,29,29,0.18)',
      tickfont: { size: 11, color: 'rgba(29,29,29,0.55)' },
    },
    legend: {
      bgcolor: 'transparent',
      font: { size: 11 },
      orientation: 'h',
      yanchor: 'bottom', y: 1.04,
      xanchor: 'left', x: 0,
    },
  }, overrides || {});
}

const plotlyConfig = { responsive: true, displayModeBar: false };

// ---------------------------------------------------------------------------
// State
// ---------------------------------------------------------------------------
let allMetrics = [];
let allAlerts = {};
let currentWeek = null;
let currentFeature = null;
let featureList = [];

// ---------------------------------------------------------------------------
// Init
// ---------------------------------------------------------------------------
async function init() {
  try {
    const [metrics, alerts] = await Promise.all([
      fetch('/metrics').then(r => r.json()),
      fetch('/alerts').then(r => r.json()),
    ]);

    allMetrics = metrics;
    allAlerts = alerts;

    featureList = Object.keys(metrics[0].feature_drift_scores)
      .filter(f => f !== 'target' && f !== 'prediction')
      .sort();

    currentWeek = metrics[metrics.length - 1].week;
    currentFeature = featureList.includes('Amount') ? 'Amount' : featureList[0];

    renderKPIs();
    renderF1Chart();
    renderHeatmap();
    buildSelectors();
    await renderDistribution();
    renderAlerts();

    updateSidebarCounts();
  } catch (err) {
    document.getElementById('loading-overlay').innerHTML =
      '<div style="color:#cf5c36;padding:2rem">Failed to load metrics. Is the API running?</div>';
    console.error(err);
  } finally {
    document.getElementById('loading-overlay').style.display = 'none';
  }
}

// ---------------------------------------------------------------------------
// KPIs
// ---------------------------------------------------------------------------
function renderKPIs() {
  const latest = allMetrics[allMetrics.length - 1];
  const baseline = allMetrics[0].classification.f1;
  const f1 = latest.classification.f1;
  const delta = f1 - baseline;
  const alertCount = allAlerts.alert_weeks?.length ?? 0;
  const share = latest.dataset_drift.share_drifted_features;

  set('kpi-weeks', allMetrics.length);
  set('kpi-f1', f1.toFixed(3));
  const deltaEl = document.getElementById('kpi-f1-delta');
  if (deltaEl) {
    deltaEl.textContent = (delta >= 0 ? '+' : '') + delta.toFixed(3) + ' vs baseline';
    deltaEl.style.color = delta < 0 ? 'var(--color-error)' : 'var(--color-success)';
  }
  set('kpi-alerts', alertCount);
  set('kpi-alerts-sub', alertCount + ' of ' + allMetrics.length + ' weeks');
  set('kpi-drift', (share * 100).toFixed(1) + '%');
  set('kpi-drift-sub', 'latest week');
}

// ---------------------------------------------------------------------------
// F1 Timeline
// ---------------------------------------------------------------------------
function renderF1Chart() {
  const weeks = allMetrics.map(m => m.week);
  const f1s = allMetrics.map(m => m.classification.f1);
  const baseline = f1s[0];
  const threshold = baseline * 0.9;
  const colors = f1s.map(v => v < threshold ? '#cf5c36' : '#6eb4d1');

  const layout = norrLayout({
    height: 300,
    xaxis: { title: { text: 'Week', font: { size: 12 } } },
    yaxis: { title: { text: 'F1-Score', font: { size: 12 } }, range: [0, 1.05] },
    shapes: [{
      type: 'line',
      x0: weeks[0], x1: weeks[weeks.length - 1],
      y0: threshold, y1: threshold,
      line: { color: '#cf5c36', width: 1.5, dash: 'dot' }
    }],
    annotations: [{
      x: weeks[weeks.length - 1], y: threshold,
      text: 'Alert threshold (' + threshold.toFixed(3) + ')',
      showarrow: false, xanchor: 'right', yanchor: 'bottom',
      font: { color: '#cf5c36', size: 11 }
    }]
  });

  Plotly.newPlot('f1-chart', [
    {
      x: weeks, y: f1s,
      fill: 'tozeroy', fillcolor: 'rgba(110,180,209,0.07)',
      line: { color: 'transparent' },
      showlegend: false, hoverinfo: 'skip', type: 'scatter'
    },
    {
      x: weeks, y: f1s,
      mode: 'lines+markers', name: 'F1-Score',
      line: { color: '#6eb4d1', width: 2.5 },
      marker: { size: 9, color: colors, line: { width: 2, color: '#fff' } },
      hovertemplate: 'Week %{x}<br>F1: %{y:.4f}<extra></extra>',
      type: 'scatter'
    }
  ], layout, plotlyConfig);
}

// ---------------------------------------------------------------------------
// Heatmap
// ---------------------------------------------------------------------------
function renderHeatmap() {
  const weeks = allMetrics.map(m => 'W' + m.week);
  const z = featureList.map(feat =>
    allMetrics.map(m => parseFloat((m.feature_drift_scores[feat] ?? 0).toFixed(4)))
  );
  const text = z.map(row => row.map(v => v.toFixed(2)));

  Plotly.newPlot('heatmap', [{
    type: 'heatmap',
    z, x: weeks, y: featureList,
    colorscale: [[0, '#fff9f3'], [0.15, '#6eb4d1'], [0.4, '#ffcf99'], [1, '#cf5c36']],
    zmin: 0, zmax: 0.5,
    text, texttemplate: '%{text}',
    textfont: { size: 10, color: '#1d1d1d' },
    colorbar: {
      title: { text: 'PSI', font: { size: 11 } },
      tickfont: { size: 10 }, thickness: 12, len: 0.8
    }
  }], norrLayout({ height: 400 }), plotlyConfig);
}

// ---------------------------------------------------------------------------
// Distribution selectors + chart
// ---------------------------------------------------------------------------
function buildSelectors() {
  const weekSel = document.getElementById('dist-week');
  const featSel = document.getElementById('dist-feature');

  allMetrics.forEach(m => {
    const o = new Option('Week ' + m.week, m.week);
    if (m.week === currentWeek) o.selected = true;
    weekSel.add(o);
  });

  featureList.forEach(f => {
    const o = new Option(f, f);
    if (f === currentFeature) o.selected = true;
    featSel.add(o);
  });

  weekSel.addEventListener('change', () => { currentWeek = +weekSel.value; renderDistribution(); });
  featSel.addEventListener('change', () => { currentFeature = featSel.value; renderDistribution(); });
}

async function renderDistribution() {
  const el = document.getElementById('dist-chart');
  el.style.minHeight = '240px';

  try {
    const data = await fetch('/data/distribution/' + currentWeek + '/' + currentFeature).then(r => r.json());
    Plotly.newPlot('dist-chart', [
      { type: 'bar', x: data.bins, y: data.reference, name: 'Reference', opacity: 0.75, marker: { color: '#6eb4d1' } },
      { type: 'bar', x: data.bins, y: data.batch, name: 'Week ' + currentWeek, opacity: 0.75, marker: { color: '#cf5c36' } }
    ], norrLayout({
      height: 260,
      barmode: 'overlay',
      xaxis: { title: { text: currentFeature, font: { size: 12 } } },
      yaxis: { title: { text: 'Count', font: { size: 12 } } },
    }), plotlyConfig);
  } catch (err) {
    el.innerHTML = '<p class="chart-error">Error loading distribution data.</p>';
  }
}

// ---------------------------------------------------------------------------
// Alerts
// ---------------------------------------------------------------------------
function renderAlerts() {
  const container = document.getElementById('alerts-container');
  const alertWeeks = allAlerts.alert_weeks ?? [];

  if (alertWeeks.length === 0) {
    container.innerHTML = '<div class="alert-ok">No alerts detected across all monitored weeks.</div>';
    return;
  }

  const baseline = allMetrics[0].classification.f1;
  const threshold = baseline * 0.9;
  container.innerHTML = '';

  [...alertWeeks].reverse().forEach((weekNum, idx) => {
    const m = allMetrics.find(x => x.week === weekNum);
    if (!m) return;

    const f1 = m.classification.f1;
    const severity = f1 < threshold ? 'Critical' : 'Warning';
    const share = (m.dataset_drift.share_drifted_features * 100).toFixed(1);

    const item = document.createElement('div');
    item.className = 'alert-item' + (idx === 0 ? ' open' : '');
    item.innerHTML =
      '<div class="alert-item__header" onclick="this.parentElement.classList.toggle(\'open\')">' +
        '<div class="alert-item__summary">' +
          '<span class="severity-badge severity-' + severity.toLowerCase() + '">' + severity + '</span>' +
          '<span class="alert-item__title">Week ' + weekNum + '</span>' +
          '<span class="alert-item__meta">F1 ' + f1.toFixed(3) + ' &middot; ' + share + '% drifted</span>' +
        '</div>' +
        '<svg class="chevron" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polyline points="6 9 12 15 18 9"/></svg>' +
      '</div>' +
      '<div class="alert-item__body">' +
        '<div class="alert-metrics">' +
          metricBox('F1-Score', f1.toFixed(4)) +
          metricBox('Precision', m.classification.precision.toFixed(4)) +
          metricBox('Recall', m.classification.recall.toFixed(4)) +
        '</div>' +
        (m.alerts || []).map(a => '<div class="alert-msg">' + a + '</div>').join('') +
        '<a href="/report/' + weekNum + '" target="_blank" class="report-link">View Evidently report &rarr;</a>' +
      '</div>';

    container.appendChild(item);
  });
}

// ---------------------------------------------------------------------------
// Sidebar nav active state + counts
// ---------------------------------------------------------------------------
function updateSidebarCounts() {
  const alertCount = allAlerts.alert_weeks?.length ?? 0;
  const badge = document.getElementById('sidebar-alert-count');
  if (badge) badge.textContent = alertCount;
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------
function set(id, value) {
  const el = document.getElementById(id);
  if (el) el.textContent = value;
}

function metricBox(label, value) {
  return '<div class="alert-metric"><div class="alert-metric__label">' + label + '</div>' +
    '<div class="alert-metric__value">' + value + '</div></div>';
}

// ---------------------------------------------------------------------------
// Smooth scroll for sidebar links
// ---------------------------------------------------------------------------
document.addEventListener('DOMContentLoaded', () => {
  document.querySelectorAll('.sidebar-link[href^="#"]').forEach(link => {
    link.addEventListener('click', e => {
      e.preventDefault();
      document.querySelectorAll('.sidebar-link').forEach(l => l.classList.remove('active'));
      link.classList.add('active');
      const target = document.querySelector(link.getAttribute('href'));
      if (target) target.scrollIntoView({ behavior: 'smooth', block: 'start' });
    });
  });

  init();
});
