"""
infrastructure/api.py - FastAPI application (thin routing layer).

Serves pre-computed metrics and reports from the filesystem.
Dashboard is a static HTML/JS app served from /dashboard.

Run with:
  uvicorn src.infrastructure.api:app --host 0.0.0.0 --port 8000
  # or: make api
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("api")

# ---------------------------------------------------------------------------
# Config bootstrap
# ---------------------------------------------------------------------------

_CONFIG_PATHS = ["config.yaml", "../config.yaml"]


def _load_config() -> dict[str, Any]:
    for p in _CONFIG_PATHS:
        if Path(p).exists():
            with open(p) as fh:
                return yaml.safe_load(fh)
    raise FileNotFoundError("config.yaml not found")


_cfg = _load_config()
_METRICS_DIR = Path(_cfg["monitoring"]["metrics_dir"])
_REPORTS_DIR = Path(_cfg["monitoring"]["reports_dir"])
_REF_PATH = Path(_cfg["data"]["reference_output"])
_SIM_DIR = Path(_cfg["data"]["simulated_dir"])

# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(
    title="ML Drift Monitor API",
    description=(
        "Exposes weekly drift metrics and Evidently reports for the "
        "fraud detection model. Dashboard at /dashboard."
    ),
    version="1.0.0",
)

# Static assets (logo, icon)
_assets_path = Path("dashboard/assets")
if not _assets_path.exists():
    _assets_path = Path("../dashboard/assets")
if _assets_path.exists():
    app.mount("/static", StaticFiles(directory=str(_assets_path)), name="static")

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _all_metric_files() -> list[Path]:
    return sorted(_METRICS_DIR.glob("week_*.json"))


def _load_week_metrics(week: int) -> dict[str, Any]:
    path = _METRICS_DIR / f"week_{week:02d}.json"
    if not path.exists():
        raise HTTPException(
            status_code=404,
            detail=f"Metrics for week {week} not found. Run `make monitor` first.",
        )
    with open(path) as fh:
        return json.load(fh)


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


_HUB_TEMPLATE = (
    "<!DOCTYPE html>\n"
    '<html lang="pt">\n'
    "<head>\n"
    '  <meta charset="UTF-8">\n'
    '  <meta name="viewport" content="width=device-width, initial-scale=1.0">\n'
    "  <title>ML Drift Monitor</title>\n"
    '  <link href="https://api.fontshare.com/v2/css?f[]=satoshi@400,500&f[]=rubik@300&display=swap" rel="stylesheet">\n'
    "  <style>\n"
    "    *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }\n"
    "    :root {\n"
    "      --ink: #1d1d1d; --cream: #fff9f3; --surface: #ffffff;\n"
    "      --sunken: #f0e8df; --terracotta: #cf5c36; --sky: #6eb4d1;\n"
    "      --success: #4a9b6f; --border: rgba(29,29,29,0.12);\n"
    "      --text2: rgba(29,29,29,0.55);\n"
    "      --shadow-sm: 0 1px 3px rgba(207,92,54,0.08),0 1px 2px rgba(29,29,29,0.06);\n"
    "      --shadow-lg: 0 8px 24px rgba(207,92,54,0.12),0 4px 10px rgba(29,29,29,0.08);\n"
    "    }\n"
    "    body { font-family:'Satoshi',system-ui,sans-serif; background:var(--cream); color:var(--ink); min-height:100vh; padding:clamp(2rem,5vw,4rem) clamp(1.5rem,8vw,6rem); }\n"
    "    .header { display:flex; align-items:center; gap:1.25rem; margin-bottom:2.5rem; }\n"
    "    .header img { width:56px; height:auto; }\n"
    "    .header-text { flex:1; }\n"
    "    .header h1 { font-family:'Rubik',system-ui,sans-serif; font-weight:300; font-size:clamp(1.6rem,3vw,2.2rem); letter-spacing:-0.025em; }\n"
    "    .header p { font-size:0.875rem; color:var(--text2); margin-top:0.2rem; }\n"
    "    .lang-toggle { display:flex; gap:2px; background:rgba(29,29,29,0.06); border-radius:9999px; padding:3px; margin-left:auto; }\n"
    "    .lang-toggle button { background:transparent; border:none; cursor:pointer; font-size:0.72rem; font-weight:500; padding:3px 10px; border-radius:9999px; color:var(--text2); font-family:'Satoshi',system-ui,sans-serif; transition:all 120ms; }\n"
    "    .lang-toggle button.active { background:var(--surface); color:var(--terracotta); box-shadow:0 1px 3px rgba(29,29,29,0.1); }\n"
    "    .status-badge { display:inline-flex; align-items:center; gap:0.5rem; background:var(--surface); border:1px solid var(--border); border-radius:9999px; padding:0.35rem 1rem; font-size:0.8rem; font-weight:500; box-shadow:var(--shadow-sm); margin-bottom:2rem; }\n"
    "    .status-dot { width:8px; height:8px; border-radius:50%; background:__STATUS_COLOR__; }\n"
    "    .metrics-strip { display:flex; gap:1rem; flex-wrap:wrap; margin-bottom:2.5rem; }\n"
    "    .metric-pill { background:var(--surface); border:1px solid var(--border); border-radius:9999px; padding:0.4rem 1rem; font-size:0.8rem; box-shadow:var(--shadow-sm); }\n"
    "    .metric-pill span { font-weight:500; color:var(--terracotta); }\n"
    "    .grid { display:grid; grid-template-columns:repeat(auto-fit,minmax(280px,1fr)); gap:1.5rem; margin-bottom:3rem; }\n"
    "    .card { background:var(--surface); border:1px solid var(--border); border-radius:16px; padding:2rem; box-shadow:var(--shadow-sm); transition:box-shadow 200ms,transform 200ms; text-decoration:none; color:inherit; display:block; }\n"
    "    .card:hover { box-shadow:var(--shadow-lg); transform:translateY(-2px); }\n"
    "    .card-tag { font-size:0.7rem; font-weight:500; letter-spacing:0.08em; text-transform:uppercase; color:var(--text2); margin-bottom:0.75rem; }\n"
    "    .card h2 { font-family:'Rubik',system-ui,sans-serif; font-weight:300; font-size:1.4rem; letter-spacing:-0.02em; margin-bottom:0.5rem; }\n"
    "    .card p { font-size:0.9rem; color:var(--text2); line-height:1.6; margin-bottom:1.5rem; }\n"
    "    .btn { display:inline-block; background:var(--terracotta); color:#fff; font-size:0.85rem; font-weight:500; padding:0.55rem 1.25rem; border-radius:8px; text-decoration:none; transition:background 150ms; }\n"
    "    .btn:hover { background:#b84e2c; }\n"
    "    .btn.outline { background:transparent; color:var(--terracotta); border:1px solid var(--terracotta); }\n"
    "    .btn.outline:hover { background:rgba(207,92,54,0.06); }\n"
    "    hr { border:none; border-top:1px solid var(--border); margin:2.5rem 0; }\n"
    "    .instructions { display:grid; grid-template-columns:1fr 1fr; gap:2rem; }\n"
    "    @media(max-width:720px) { .instructions { grid-template-columns:1fr; } }\n"
    "    .block { background:var(--surface); border:1px solid var(--border); border-radius:16px; padding:1.75rem; box-shadow:var(--shadow-sm); }\n"
    "    .block h3 { font-size:0.7rem; font-weight:500; letter-spacing:0.08em; text-transform:uppercase; color:var(--text2); margin-bottom:1rem; }\n"
    "    .block h4 { font-family:'Rubik',system-ui,sans-serif; font-weight:300; font-size:1.15rem; letter-spacing:-0.02em; margin-bottom:1.25rem; }\n"
    "    .step { display:flex; gap:0.9rem; margin-bottom:1rem; align-items:flex-start; }\n"
    "    .step-num { width:22px; height:22px; background:var(--sunken); border-radius:50%; display:flex; align-items:center; justify-content:center; font-size:0.72rem; font-weight:500; flex-shrink:0; margin-top:2px; color:var(--terracotta); }\n"
    "    .step p { font-size:0.875rem; line-height:1.65; color:var(--ink); margin:0; }\n"
    "    code { font-family:'SF Mono','Fira Code',monospace; font-size:0.78rem; background:var(--sunken); padding:0.1rem 0.35rem; border-radius:4px; color:var(--terracotta); }\n"
    "    footer { margin-top:3rem; font-size:0.78rem; color:var(--text2); }\n"
    "  </style>\n"
    "</head>\n"
    "<body>\n"
    '  <div class="header">\n'
    '    <img src="/static/norr-logo.png" alt="Norr">\n'
    '    <div class="header-text">\n'
    "      <h1>ML Drift Monitor</h1>\n"
    '      <p><span data-i18n="subtitle">Modelo de Detec\u00e7\u00e3o de Fraude</span></p>\n'
    "    </div>\n"
    '    <div class="lang-toggle">\n'
    "      <button id=\"hub-btn-pt\" onclick=\"setLang('pt')\" class=\"active\">PT</button>\n"
    "      <button id=\"hub-btn-en\" onclick=\"setLang('en')\">EN</button>\n"
    "    </div>\n"
    "  </div>\n"
    "\n"
    '  <div class="status-badge">\n'
    '    <div class="status-dot"></div>\n'
    '    __WEEKS__ <span data-i18n="weeks-monitored">semanas monitoradas</span> &nbsp; __STATUS_LABEL__\n'
    "  </div>\n"
    "\n"
    '  <div class="metrics-strip">\n'
    '    <div class="metric-pill"><span data-i18n="pill-weeks">Semanas monitoradas</span>: <span>__WEEKS__</span></div>\n'
    '    <div class="metric-pill"><span data-i18n="pill-alerts">Semanas com alerta</span>: <span>__ALERTS__</span></div>\n'
    '    <div class="metric-pill">Stack: <span>XGBoost &middot; Evidently &middot; FastAPI</span></div>\n'
    "  </div>\n"
    "\n"
    '  <div class="grid">\n'
    '    <a class="card" href="/dashboard/">\n'
    '      <div class="card-tag" data-i18n="card-dash-tag">DASHBOARD</div>\n'
    "      <h2>Visual Monitor</h2>\n"
    '      <p data-i18n="card-dash-desc">Gr\u00e1ficos interativos de F1-Score, heatmap de drift, distribui\u00e7\u00f5es e painel de alertas.</p>\n'
    '      <span class="btn" data-i18n="card-dash-btn">Abrir Dashboard</span>\n'
    "    </a>\n"
    '    <a class="card" href="/docs" target="_blank">\n'
    '      <div class="card-tag">API &mdash; SWAGGER UI</div>\n'
    "      <h2>API Reference</h2>\n"
    '      <p data-i18n="card-api-desc">Endpoints REST para consultar m\u00e9tricas, alertas e relat\u00f3rios Evidently por semana.</p>\n'
    '      <span class="btn outline" data-i18n="card-api-btn">Abrir API Docs</span>\n'
    "    </a>\n"
    "  </div>\n"
    "\n"
    "  <hr>\n"
    "\n"
    '  <div class="instructions">\n'
    '    <div class="block">\n'
    '      <h3 data-i18n="block1-tag">PARA QUEM N\u00c3O \u00c9 DA \u00c1REA</h3>\n'
    '      <h4 data-i18n="block1-title">Como usar o monitor</h4>\n'
    '      <div class="step"><div class="step-num">1</div><p data-i18n="step1-1">Clique em <strong>Abrir Dashboard</strong>. O painel visual abre na mesma aba.</p></div>\n'
    '      <div class="step"><div class="step-num">2</div><p data-i18n="step1-2">Olhe o gr\u00e1fico. Se a linha azul cair abaixo da vermelha, o modelo est\u00e1 errando mais.</p></div>\n'
    '      <div class="step"><div class="step-num">3</div><p data-i18n="step1-3">Role at\u00e9 <strong>Painel de Alertas</strong>. Semana <em>Critical</em> = dados mudaram significativamente.</p></div>\n'
    '      <div class="step"><div class="step-num">4</div><p data-i18n="step1-4">Alertas recorrentes? Avise o time t\u00e9cnico \u2014 hora de retreinar o modelo.</p></div>\n'
    "    </div>\n"
    '    <div class="block">\n'
    '      <h3 data-i18n="block2-tag">PARA DEVS E DATA SCIENTISTS</h3>\n'
    '      <h4 data-i18n="block2-title">API endpoints</h4>\n'
    '      <div class="step"><div class="step-num">1</div><p><code>GET /metrics</code> <span data-i18n="api-step1">\u2014 JSON com F1, PSI e share drifted de todas as semanas.</span></p></div>\n'
    '      <div class="step"><div class="step-num">2</div><p><code>GET /alerts</code> <span data-i18n="api-step2">\u2014 semanas com drift acima do threshold.</span></p></div>\n'
    '      <div class="step"><div class="step-num">3</div><p><code>GET /report/{week}</code> <span data-i18n="api-step3">\u2014 relat\u00f3rio Evidently HTML da semana N.</span></p></div>\n'
    '      <div class="step"><div class="step-num">4</div><p data-i18n="api-step4">Para retreinar: <code>make train &amp;&amp; make simulate &amp;&amp; make monitor</code>.</p></div>\n'
    "    </div>\n"
    "  </div>\n"
    "\n"
    "  <footer>ML Drift Monitor v1.0</footer>\n"
    "\n"
    "<script>\n"
    "(function() {\n"
    "  var T = {\n"
    "    pt: {\n"
    "      'subtitle': 'Modelo de Detec\u00e7\u00e3o de Fraude',\n"
    "      'weeks-monitored': 'semanas monitoradas',\n"
    "      'pill-weeks': 'Semanas monitoradas',\n"
    "      'pill-alerts': 'Semanas com alerta',\n"
    "      'card-dash-tag': 'DASHBOARD',\n"
    "      'card-dash-desc': 'Gr\u00e1ficos interativos de F1-Score, heatmap de drift, distribui\u00e7\u00f5es e painel de alertas.',\n"
    "      'card-dash-btn': 'Abrir Dashboard',\n"
    "      'card-api-desc': 'Endpoints REST para consultar m\u00e9tricas, alertas e relat\u00f3rios Evidently por semana.',\n"
    "      'card-api-btn': 'Abrir API Docs',\n"
    "      'block1-tag': 'PARA QUEM N\u00c3O \u00c9 DA \u00c1REA',\n"
    "      'block1-title': 'Como usar o monitor',\n"
    "      'step1-1': 'Clique em <strong>Abrir Dashboard</strong>. O painel visual abre na mesma aba.',\n"
    "      'step1-2': 'Olhe o gr\u00e1fico. Se a linha azul cair abaixo da vermelha, o modelo est\u00e1 errando mais.',\n"
    "      'step1-3': 'Role at\u00e9 <strong>Painel de Alertas</strong>. Semana <em>Critical</em> = dados mudaram significativamente.',\n"
    "      'step1-4': 'Alertas recorrentes? Avise o time t\u00e9cnico \u2014 hora de retreinar o modelo.',\n"
    "      'block2-tag': 'PARA DEVS E DATA SCIENTISTS',\n"
    "      'block2-title': 'API endpoints',\n"
    "      'api-step1': '\u2014 JSON com F1, PSI e share drifted de todas as semanas.',\n"
    "      'api-step2': '\u2014 semanas com drift acima do threshold.',\n"
    "      'api-step3': '\u2014 relat\u00f3rio Evidently HTML da semana N.',\n"
    "      'api-step4': 'Para retreinar: <code>make train &amp;&amp; make simulate &amp;&amp; make monitor</code>.',\n"
    "    },\n"
    "    en: {\n"
    "      'subtitle': 'Fraud Detection Model',\n"
    "      'weeks-monitored': 'weeks monitored',\n"
    "      'pill-weeks': 'Weeks monitored',\n"
    "      'pill-alerts': 'Alert weeks',\n"
    "      'card-dash-tag': 'DASHBOARD',\n"
    "      'card-dash-desc': 'Interactive charts for F1-Score, drift heatmap, distribution comparison, and alerts panel.',\n"
    "      'card-dash-btn': 'Open Dashboard',\n"
    "      'card-api-desc': 'REST endpoints for metrics, alerts, and weekly Evidently reports. Test in the browser.',\n"
    "      'card-api-btn': 'Open API Docs',\n"
    "      'block1-tag': 'FOR NON-TECHNICAL USERS',\n"
    "      'block1-title': 'How to use the monitor',\n"
    "      'step1-1': 'Click <strong>Open Dashboard</strong>. The visual panel opens in the same tab.',\n"
    "      'step1-2': 'Check the chart. If the blue line falls below the red dashed line, the model is making more errors.',\n"
    "      'step1-3': 'Scroll to <strong>Alerts Panel</strong>. <em>Critical</em> week = data changed significantly.',\n"
    "      'step1-4': 'Recurring alerts? Notify the tech team \u2014 time to retrain the model.',\n"
    "      'block2-tag': 'FOR DEVS &amp; DATA SCIENTISTS',\n"
    "      'block2-title': 'API endpoints',\n"
    "      'api-step1': '\u2014 JSON array with F1, PSI and share drifted for all weeks.',\n"
    "      'api-step2': '\u2014 weeks with drift above the threshold.',\n"
    "      'api-step3': '\u2014 full Evidently HTML report for week N.',\n"
    "      'api-step4': 'To retrain: <code>make train &amp;&amp; make simulate &amp;&amp; make monitor</code>.',\n"
    "    }\n"
    "  };\n"
    "  function setLang(lang) {\n"
    "    localStorage.setItem('norr_lang', lang);\n"
    "    document.querySelectorAll('[data-i18n]').forEach(function(el) {\n"
    "      var key = el.getAttribute('data-i18n');\n"
    "      if (T[lang] && T[lang][key] !== undefined) el.innerHTML = T[lang][key];\n"
    "    });\n"
    "    document.getElementById('hub-btn-pt').classList.toggle('active', lang === 'pt');\n"
    "    document.getElementById('hub-btn-en').classList.toggle('active', lang === 'en');\n"
    "  }\n"
    "  window.setLang = setLang;\n"
    "  var saved = localStorage.getItem('norr_lang') || 'pt';\n"
    "  document.addEventListener('DOMContentLoaded', function() { setLang(saved); });\n"
    "})();\n"
    "</script>\n"
    "</body>\n"
    "</html>\n"
)


@app.get("/", response_class=HTMLResponse, include_in_schema=False)
def hub() -> HTMLResponse:
    weeks_ready = len(_all_metric_files())
    alerts_count = sum(
        1 for f in _all_metric_files()
        if json.loads(f.read_text()).get("has_alert")
    )
    status_color = "#cf5c36" if alerts_count > 0 else "#4a9b6f"
    status_label = (
        f"{alerts_count} semana(s) com alerta" if alerts_count > 0 else "Tudo certo"
    )
    html = (
        _HUB_TEMPLATE
        .replace("__WEEKS__", str(weeks_ready))
        .replace("__ALERTS__", str(alerts_count))
        .replace("__STATUS_COLOR__", status_color)
        .replace("__STATUS_LABEL__", status_label)
    )
    return HTMLResponse(content=html)



@app.get("/health", tags=["status"])
def health() -> dict[str, str]:
    weeks_ready = len(_all_metric_files())
    return {"status": "ok", "weeks_monitored": str(weeks_ready)}


@app.get("/metrics", tags=["metrics"])
def get_all_metrics() -> JSONResponse:
    files = _all_metric_files()
    if not files:
        raise HTTPException(
            status_code=404,
            detail="No metrics found. Run `make monitor` first.",
        )
    all_metrics: list[dict[str, Any]] = []
    for f in files:
        with open(f) as fh:
            all_metrics.append(json.load(fh))
    return JSONResponse(content=all_metrics)


@app.get("/metrics/{week}", tags=["metrics"])
def get_week_metrics(week: int) -> JSONResponse:
    if week < 1 or week > 52:
        raise HTTPException(status_code=422, detail="Week must be between 1 and 52.")
    return JSONResponse(content=_load_week_metrics(week))


@app.get("/alerts", tags=["alerts"])
def get_alerts() -> JSONResponse:
    files = _all_metric_files()
    alerts: list[dict[str, Any]] = []
    for f in files:
        with open(f) as fh:
            m = json.load(fh)
        if m.get("has_alert"):
            alerts.append({
                "week": m["week"],
                "alerts": m["alerts"],
                "f1": m["classification"].get("f1"),
                "share_drifted": m["dataset_drift"].get("share_drifted_features"),
            })
    return JSONResponse(content={
        "total_weeks_monitored": len(files),
        "alert_weeks": [a["week"] for a in alerts],
        "details": alerts,
    })


_NORR_REPORT_INJECT = """
<link href="https://api.fontshare.com/v2/css?f[]=satoshi@400,500&f[]=rubik@300&display=swap" rel="stylesheet">
<style>
  /* ── Norr overrides: keep class="dark" but flip colors ── */
  html.dark, html.dark body {
    background: #fff9f3 !important;
    color: #1d1d1d !important;
    font-family: 'Satoshi', system-ui, sans-serif !important;
    color-scheme: light !important;
  }

  /* Hide dark-mode toggle (no aria-label in some Evidently versions) */
  button[aria-label="Toggle color scheme"],
  button.MuiIconButton-root:not([aria-label]):not([aria-expanded]) { display: none !important; }

  /* MUI Paper / Cards */
  html.dark .MuiPaper-root, html.dark .MuiCard-root {
    background-color: #ffffff !important;
    border: 1px solid rgba(29,29,29,0.10) !important;
    box-shadow: 0 2px 8px rgba(207,92,54,0.07) !important;
    border-radius: 12px !important;
    color: #1d1d1d !important;
  }

  /* MUI text */
  html.dark .MuiTypography-root { color: #1d1d1d !important; font-family: 'Satoshi', system-ui, sans-serif !important; }
  html.dark .MuiTypography-h1, html.dark .MuiTypography-h2,
  html.dark .MuiTypography-h3, html.dark .MuiTypography-h4,
  html.dark .MuiTypography-h5, html.dark .MuiTypography-h6 {
    font-family: 'Rubik', system-ui, sans-serif !important;
    font-weight: 300 !important; color: #1d1d1d !important;
  }

  /* Chips */
  html.dark .MuiChip-root { border-radius: 9999px !important; }
  html.dark .MuiChip-colorSuccess { background: rgba(74,155,111,0.12) !important; color: #4a9b6f !important; }
  html.dark .MuiChip-colorError,
  html.dark .MuiChip-colorSecondary { background: rgba(207,92,54,0.10) !important; color: #cf5c36 !important; }
  html.dark .MuiChip-label { color: inherit !important; }

  /* Table */
  html.dark .MuiTableCell-root { color: #1d1d1d !important; border-color: rgba(29,29,29,0.07) !important; }
  html.dark .MuiTableHead-root .MuiTableCell-root {
    background: #f0e8df !important;
    font-weight: 500 !important; font-size: 0.75rem !important;
    letter-spacing: 0.07em !important; text-transform: uppercase !important;
    color: rgba(29,29,29,0.55) !important;
    border-bottom: 1px solid rgba(29,29,29,0.12) !important;
  }
  html.dark .MuiTableBody-root .MuiTableRow-root:hover .MuiTableCell-root {
    background: rgba(207,92,54,0.04) !important;
  }

  /* Inputs */
  html.dark .MuiInputBase-root { background: #ffffff !important; color: #1d1d1d !important; }
  html.dark .MuiOutlinedInput-notchedOutline { border-color: rgba(29,29,29,0.18) !important; }

  /* Buttons */
  html.dark .MuiToggleButton-root { border-color: rgba(29,29,29,0.14) !important; color: rgba(29,29,29,0.6) !important; }
  html.dark .MuiToggleButton-root.Mui-selected { background: rgba(207,92,54,0.10) !important; color: #cf5c36 !important; }
  html.dark .MuiIconButton-root { color: rgba(29,29,29,0.5) !important; }
  html.dark .MuiDivider-root { border-color: rgba(29,29,29,0.09) !important; }

  /* Plotly: override dark SVG and container backgrounds */
  html.dark .js-plotly-plot svg[style*="background"] { background: #ffffff !important; }
  html.dark .js-plotly-plot .svg-container { background: #ffffff !important; }
  html.dark .js-plotly-plot .bg { fill: #fff9f3 !important; }
  html.dark .js-plotly-plot .gridlayer path { stroke: rgba(29,29,29,0.08) !important; }
  html.dark .js-plotly-plot .zerolinelayer path { stroke: rgba(29,29,29,0.12) !important; }

  /* Top nav bar */
  #norr-report-header {
    position: sticky; top: 0; z-index: 9999;
    background: #ffffff;
    border-bottom: 1px solid rgba(29,29,29,0.10);
    padding: 0.7rem 2rem;
    display: flex; align-items: center; gap: 1.25rem;
    box-shadow: 0 1px 4px rgba(207,92,54,0.08);
    font-family: 'Satoshi', system-ui, sans-serif;
  }
  #norr-report-header a {
    font-size: 0.8rem; color: rgba(29,29,29,0.45);
    text-decoration: none; display: flex; align-items: center; gap: 0.35rem;
    transition: color 150ms; font-weight: 500;
  }
  #norr-report-header a:hover { color: #cf5c36; }
  #norr-report-header .sep { color: rgba(29,29,29,0.2); font-size: 0.9rem; }
  #norr-report-header .title {
    font-family: 'Rubik', system-ui, sans-serif;
    font-weight: 300; font-size: 0.95rem;
    color: #1d1d1d; letter-spacing: -0.02em;
  }
  #norr-report-header .week-badge {
    background: #cf5c36; color: #fff;
    font-size: 0.7rem; font-weight: 500;
    padding: 2px 10px; border-radius: 9999px;
    letter-spacing: 0.04em; margin-left: auto;
  }
  #norr-report-header .stack-tag {
    font-size: 0.7rem; color: rgba(29,29,29,0.35);
    background: #f0e8df; padding: 2px 8px; border-radius: 9999px;
    margin-left: auto;
  }
</style>
<script>
(function() {
  function fireResize() { window.dispatchEvent(new Event('resize')); }

  function fixInlineColors() {
    // Fix Plotly paper_bgcolor: rendered as inline style="background: rgb(17,17,17)" on main SVG
    document.querySelectorAll('.js-plotly-plot svg').forEach(function(svg) {
      var bg = svg.style.background;
      if (bg && bg !== '' && bg !== 'rgba(0, 0, 0, 0)' && bg !== 'transparent') {
        svg.style.background = '#ffffff';
      }
    });
    // Also fix any dark inline backgrounds on divs
    document.querySelectorAll('.js-plotly-plot .svg-container').forEach(function(el) {
      el.style.background = '#ffffff';
    });
    // Catch-all: any element with rgb(1x,1x,1x) inline background
    ['rgb(17', 'rgb(18', 'rgb(16'].forEach(function(prefix) {
      document.querySelectorAll('[style*="background: ' + prefix + '"], [style*="background-color: ' + prefix + '"]').forEach(function(el) {
        el.style.background = '#ffffff';
        el.style.backgroundColor = '#ffffff';
      });
    });
  }

  // IntersectionObserver: when a chart enters viewport, fire resize
  // then fix inline colors after Plotly re-renders
  var io = new IntersectionObserver(function(entries) {
    entries.forEach(function(e) {
      if (e.isIntersecting) {
        setTimeout(fireResize, 50);
        setTimeout(fireResize, 300);
        setTimeout(fixInlineColors, 600);
        setTimeout(fixInlineColors, 1200);
        io.unobserve(e.target);
      }
    });
  }, { threshold: 0.1 });

  function observeCharts() {
    document.querySelectorAll('.js-plotly-plot').forEach(function(el) { io.observe(el); });
  }

  // Hide theme toggle
  function hideThemeToggle() {
    document.querySelectorAll('button.MuiIconButton-root:not([aria-label]):not([aria-expanded])').forEach(function(b) {
      b.style.setProperty('display', 'none', 'important');
    });
  }

  // Watch for charts and re-hide toggle when React re-renders
  new MutationObserver(function(mutations) {
    var needsToggleHide = false;
    mutations.forEach(function(m) {
      m.addedNodes.forEach(function(n) {
        if (n.nodeType !== 1) return;
        if (n.classList && n.classList.contains('js-plotly-plot')) io.observe(n);
        (n.querySelectorAll ? n.querySelectorAll('.js-plotly-plot') : []).forEach(function(c) { io.observe(c); });
        needsToggleHide = true;
      });
    });
    if (needsToggleHide) hideThemeToggle();
  }).observe(document.body, { childList: true, subtree: true });
  [0, 100, 500, 1500, 3000, 6000].forEach(function(ms) {
    setTimeout(function() { observeCharts(); fixInlineColors(); hideThemeToggle(); }, ms);
  });
  // Keep hiding toggle as React re-renders (clears after 15s)
  var _toggleInterval = setInterval(hideThemeToggle, 300);
  setTimeout(function() { clearInterval(_toggleInterval); }, 15000);
})();
</script>
"""


@app.get("/report/{week}", tags=["reports"])
def get_report(week: int) -> HTMLResponse:
    if week < 1 or week > 52:
        raise HTTPException(status_code=422, detail="Week must be between 1 and 52.")
    report_path = _REPORTS_DIR / f"week_{week:02d}.html"
    if not report_path.exists():
        raise HTTPException(
            status_code=404,
            detail=f"Report for week {week} not found. Run `make monitor` first.",
        )
    html = report_path.read_text(encoding="utf-8")

    # Keep class="dark" (switching to "light" breaks chart layout dimensions)
    # Instead override via high-specificity CSS below

    # Inject Norr CSS + JS overrides before </head>
    html = html.replace("</head>", f"{_NORR_REPORT_INJECT}</head>", 1)

    # Inject sticky header bar after <body>
    header = (
        f'<div id="norr-report-header">'
        f'<a href="/dashboard/#alerts">&#8592; Dashboard</a>'
        f'<span class="sep">/</span>'
        f'<span class="title">Evidently Report</span>'
        f'<span class="stack-tag">Evidently AI</span>'
        f'<span class="week-badge">Week {week:02d}</span>'
        f'</div>'
    )
    html = html.replace("<body>", f"<body>{header}", 1)

    return HTMLResponse(content=html)


@app.get("/data/distribution/{week}/{feature}", tags=["data"])
def get_distribution(week: int, feature: str) -> JSONResponse:
    """Histogram data for reference vs. batch distribution comparison."""
    if week < 1 or week > 52:
        raise HTTPException(status_code=422, detail="Week must be between 1 and 52.")

    ref_path = _REF_PATH
    batch_path = _SIM_DIR / f"week_{week:02d}.parquet"

    if not ref_path.exists():
        raise HTTPException(status_code=404, detail="Reference data not found.")
    if not batch_path.exists():
        raise HTTPException(status_code=404, detail=f"Batch data for week {week} not found.")

    ref_df = pd.read_parquet(ref_path)
    batch_df = pd.read_parquet(batch_path)

    if feature not in ref_df.columns:
        raise HTTPException(status_code=404, detail=f"Feature '{feature}' not found.")

    ref_vals = ref_df[feature].dropna().values
    batch_vals = batch_df[feature].dropna().values if feature in batch_df.columns else np.array([])

    combined = np.concatenate([ref_vals, batch_vals])
    bins = np.linspace(float(combined.min()), float(combined.max()), 51)

    ref_counts, _ = np.histogram(ref_vals, bins=bins)
    batch_counts, _ = np.histogram(batch_vals, bins=bins)

    return JSONResponse(content={
        "feature": feature,
        "week": week,
        "bins": [round(b, 6) for b in bins[:-1].tolist()],
        "reference": ref_counts.tolist(),
        "batch": batch_counts.tolist(),
    })


# ---------------------------------------------------------------------------
# Dashboard static files (must be mounted after API routes)
# ---------------------------------------------------------------------------

_dashboard_path = Path("dashboard")
if not _dashboard_path.exists():
    _dashboard_path = Path("../dashboard")
if _dashboard_path.exists():
    app.mount("/dashboard", StaticFiles(directory=str(_dashboard_path), html=True), name="dashboard")
