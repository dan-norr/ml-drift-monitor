"""
app.py - ML Drift Monitor Dashboard (Norr Design System)

Visualises:
  1. F1-Score over 12 weeks (line chart with threshold)
  2. Feature drift heatmap (drift score per feature per week)
  3. Distribution comparison (reference vs. selected batch)
  4. Alert panel with per-week details

Run with:  python -m streamlit run dashboard/app.py
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st
import yaml
from PIL import Image

# ─────────────────────────────────────────────────────────────
# Norr Design Tokens
# ─────────────────────────────────────────────────────────────
NORR = {
    # Primitives
    "ink":        "#1d1d1d",
    "white":      "#ffffff",
    "cream":      "#fff9f3",
    "amber":      "#ffcf99",
    "sky":        "#6eb4d1",
    "terracotta": "#cf5c36",
    # Semantic
    "bg_primary":   "#fff9f3",
    "bg_secondary": "#f5ede3",
    "surface":      "#ffffff",
    "surface_sunken":"#f0e8df",
    "text_primary": "#1d1d1d",
    "text_secondary":"rgba(29,29,29,0.55)",
    "border":       "rgba(29,29,29,0.12)",
    "border_strong":"rgba(29,29,29,0.28)",
    "success":      "#4a9b6f",
    "success_bg":   "rgba(74,155,111,0.10)",
    "error":        "#cf5c36",
    "error_bg":     "rgba(207,92,54,0.10)",
    "warning":      "#ffcf99",
    "warning_bg":   "rgba(255,207,153,0.25)",
    "info":         "#6eb4d1",
    "info_bg":      "rgba(110,180,209,0.12)",
    # Shadows
    "shadow_sm": "0 1px 3px rgba(207,92,54,0.08),0 1px 2px rgba(29,29,29,0.06)",
    "shadow_md": "0 4px 12px rgba(207,92,54,0.10),0 2px 6px rgba(29,29,29,0.06)",
    "shadow_lg": "0 8px 24px rgba(207,92,54,0.12),0 4px 10px rgba(29,29,29,0.08)",
}

NORR_CSS = f"""
<style>
@import url('https://api.fontshare.com/v2/css?f[]=satoshi@400,500&f[]=rubik@300&display=swap');

/* ── Reset Streamlit defaults ── */
html, body, [class*="css"] {{
    font-family: 'Satoshi', system-ui, sans-serif !important;
    color: {NORR["text_primary"]} !important;
}}

.stApp {{
    background-color: {NORR["bg_primary"]} !important;
}}

/* ── Hide sidebar entirely ── */
[data-testid="stSidebar"],
[data-testid="collapsedControl"],
button[kind="header"] {{
    display: none !important;
}}

/* ── Main container padding ── */
[data-testid="stAppViewContainer"] > .main > .block-container {{
    padding: 2rem 2.5rem !important;
    max-width: 1200px !important;
}}

/* ── Title ── */
h1 {{
    font-family: 'Rubik', system-ui, sans-serif !important;
    font-weight: 300 !important;
    font-size: clamp(1.8rem, 2.5vw, 2.5rem) !important;
    color: {NORR["ink"]} !important;
    letter-spacing: -0.02em !important;
    line-height: 1.2 !important;
    margin-bottom: 0.25rem !important;
}}

/* ── Subheadings ── */
h2, h3 {{
    font-family: 'Satoshi', system-ui, sans-serif !important;
    font-weight: 500 !important;
    color: {NORR["ink"]} !important;
    letter-spacing: -0.015em !important;
}}

/* ── Caption / secondary text ── */
.caption, [data-testid="stCaptionContainer"] p {{
    color: {NORR["text_secondary"]} !important;
    font-size: 0.875rem !important;
}}

/* ── Custom KPI cards (replaces st.metric to avoid truncation) ── */
.kpi-grid {{
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 1rem;
    margin-bottom: 2rem;
}}
.kpi-card {{
    background: {NORR["surface"]};
    border: 1px solid {NORR["border"]};
    border-radius: 12px;
    padding: 1.25rem 1.5rem;
    box-shadow: {NORR["shadow_sm"]};
    transition: box-shadow 250ms ease-out;
}}
.kpi-card:hover {{ box-shadow: {NORR["shadow_md"]}; }}
.kpi-label {{
    font-size: 0.7rem;
    font-weight: 500;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: {NORR["text_secondary"]};
    margin-bottom: 0.4rem;
}}
.kpi-value {{
    font-family: 'Rubik', system-ui, sans-serif;
    font-weight: 300;
    font-size: 2rem;
    letter-spacing: -0.025em;
    color: {NORR["ink"]};
    line-height: 1;
}}
.kpi-delta {{
    font-size: 0.78rem;
    font-weight: 500;
    margin-top: 0.35rem;
}}
.kpi-delta.down {{ color: {NORR["terracotta"]}; }}
.kpi-delta.up   {{ color: {NORR["success"]}; }}
.kpi-delta.warn {{ color: {NORR["terracotta"]}; }}

/* ── Header ── */
.norr-header {{
    display: flex;
    align-items: center;
    gap: 1rem;
    margin-bottom: 1.5rem;
    padding-bottom: 1.25rem;
    border-bottom: 1px solid {NORR["border"]};
}}
.norr-header img {{ width: 44px; height: auto; flex-shrink: 0; }}
.norr-header h1 {{
    font-family: 'Rubik', system-ui, sans-serif !important;
    font-weight: 300 !important;
    font-size: 1.6rem !important;
    letter-spacing: -0.025em !important;
    line-height: 1.1 !important;
    margin: 0 !important;
}}
.norr-header p {{
    font-size: 0.8rem;
    color: {NORR["text_secondary"]};
    margin: 0.15rem 0 0;
}}
.norr-header a {{
    margin-left: auto;
    font-size: 0.78rem;
    color: {NORR["text_secondary"]};
    text-decoration: none;
    border: 1px solid {NORR["border"]};
    border-radius: 9999px;
    padding: 0.3rem 0.9rem;
    transition: border-color 150ms, color 150ms;
    white-space: nowrap;
}}
.norr-header a:hover {{ border-color: {NORR["terracotta"]}; color: {NORR["terracotta"]}; }}

/* ── Divider ── */
hr {{
    border: none !important;
    border-top: 1px solid {NORR["border"]} !important;
    margin: 2rem 0 !important;
}}

/* ── Expanders (alert panels) ── */
[data-testid="stExpander"] {{
    background: {NORR["surface"]} !important;
    border: 1px solid {NORR["border"]} !important;
    border-radius: 12px !important;
    box-shadow: {NORR["shadow_sm"]} !important;
    margin-bottom: 0.75rem !important;
    overflow: hidden !important;
}}

[data-testid="stExpander"] summary {{
    font-weight: 500 !important;
    padding: 1rem 1.25rem !important;
}}

/* ── Alert/error/success/info boxes ── */
[data-testid="stAlert"] {{
    border-radius: 8px !important;
    font-size: 0.875rem !important;
    border: none !important;
}}

/* ── Selectbox / slider ── */
[data-testid="stSelectbox"] > div > div,
[data-testid="stSlider"] {{
    border-radius: 8px !important;
}}

/* ── Slider accent ── */
[data-testid="stSlider"] [data-baseweb="slider"] [role="slider"] {{
    background-color: {NORR["terracotta"]} !important;
    border-color: {NORR["terracotta"]} !important;
}}

/* ── Sidebar section label ── */
[data-testid="stSidebar"] h2 {{
    font-size: 0.75rem !important;
    text-transform: uppercase !important;
    letter-spacing: 0.08em !important;
    color: {NORR["text_secondary"]} !important;
    margin-bottom: 0.75rem !important;
}}

/* ── Plotly chart containers ── */
[data-testid="stPlotlyChart"] {{
    background: {NORR["surface"]} !important;
    border-radius: 12px !important;
    padding: 1rem !important;
    border: 1px solid {NORR["border"]} !important;
    box-shadow: {NORR["shadow_sm"]} !important;
}}

/* ── Hide Streamlit footer & main menu ── */
#MainMenu, footer, [data-testid="stToolbar"] {{
    display: none !important;
}}

/* ── Success / OK state ── */
.norr-ok {{
    background: {NORR["success_bg"]} !important;
    border: 1px solid {NORR["success"]} !important;
    border-radius: 8px !important;
    color: {NORR["success"]} !important;
    padding: 1rem 1.25rem !important;
    font-weight: 500 !important;
}}
</style>
"""

# ─────────────────────────────────────────────────────────────
# Plotly base layout (Norr themed)
# ─────────────────────────────────────────────────────────────

def norr_layout(height: int = 360, **kwargs) -> dict:
    """Return a Plotly layout dict with Norr styling."""
    return dict(
        height=height,
        paper_bgcolor=NORR["surface"],
        plot_bgcolor=NORR["bg_primary"],
        font=dict(family="Satoshi, system-ui, sans-serif", color=NORR["text_primary"], size=12),
        margin=dict(t=24, b=40, l=48, r=24),
        xaxis=dict(
            gridcolor=NORR["border"],
            linecolor=NORR["border_strong"],
            tickfont=dict(size=11, color=NORR["text_secondary"]),
            title_font=dict(size=12, color=NORR["text_secondary"]),
        ),
        yaxis=dict(
            gridcolor=NORR["border"],
            linecolor=NORR["border_strong"],
            tickfont=dict(size=11, color=NORR["text_secondary"]),
            title_font=dict(size=12, color=NORR["text_secondary"]),
        ),
        legend=dict(
            bgcolor="rgba(0,0,0,0)",
            bordercolor=NORR["border"],
            font=dict(size=11),
            orientation="h",
            yanchor="bottom",
            y=1.04,
            xanchor="left",
            x=0,
        ),
        **kwargs,
    )


# ─────────────────────────────────────────────────────────────
# Data loading
# ─────────────────────────────────────────────────────────────

_CONFIG_PATHS = ["config.yaml", "../config.yaml"]


@st.cache_data
def load_config() -> dict[str, Any]:
    for p in _CONFIG_PATHS:
        if Path(p).exists():
            with open(p) as fh:
                return yaml.safe_load(fh)
    raise FileNotFoundError("config.yaml not found")


@st.cache_data
def load_all_metrics(metrics_dir: str) -> list[dict[str, Any]]:
    files = sorted(Path(metrics_dir).glob("week_*.json"))
    result: list[dict[str, Any]] = []
    for f in files:
        with open(f) as fh:
            result.append(json.load(fh))
    return result


@st.cache_data
def load_reference(ref_path: str) -> pd.DataFrame:
    return pd.read_parquet(ref_path) if Path(ref_path).exists() else pd.DataFrame()


@st.cache_data
def load_batch(simulated_dir: str, week: int) -> pd.DataFrame:
    path = Path(simulated_dir) / f"week_{week:02d}.parquet"
    return pd.read_parquet(path) if path.exists() else pd.DataFrame()


# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────

def main() -> None:
    cfg = load_config()
    metrics_dir    = cfg["monitoring"]["metrics_dir"]
    reports_dir    = cfg["monitoring"]["reports_dir"]
    simulated_dir  = cfg["data"]["simulated_dir"]
    ref_path       = cfg["data"]["reference_output"]

    _logo_path = Path("dashboard/assets/norr-logo.png")
    if not _logo_path.exists():
        _logo_path = Path("assets/norr-logo.png")
    _logo = Image.open(_logo_path) if _logo_path.exists() else None

    # Page icon: square crop of logo (avoids stretching in browser tab)
    _icon_path = Path("dashboard/assets/norr-icon.png")
    if not _icon_path.exists():
        _icon_path = Path("assets/norr-icon.png")
    _icon = Image.open(_icon_path) if _icon_path.exists() else _logo

    st.set_page_config(
        page_title="ML Drift Monitor",
        page_icon=_icon,
        layout="wide",
        initial_sidebar_state="collapsed",
    )

    # Inject Norr CSS
    st.markdown(NORR_CSS, unsafe_allow_html=True)

    # ── Controls (inline, no sidebar) ────────────────────────
    ctrl1, ctrl2, ctrl3 = st.columns([2, 2, 3])
    with ctrl1:
        psi_threshold = st.slider(
            "PSI Alert Threshold",
            min_value=0.05, max_value=0.5,
            value=float(cfg["monitoring"]["alert_thresholds"]["psi"]),
            step=0.05,
            help="PSI > threshold = data drift alert",
        )
    with ctrl2:
        f1_drop_threshold = st.slider(
            "F1 Drop Alert Threshold",
            min_value=0.01, max_value=0.3,
            value=float(cfg["monitoring"]["alert_thresholds"]["f1_drop"]),
            step=0.01,
            help="Relative F1 drop that triggers alert",
        )
    with ctrl3:
        c_text = NORR["text_secondary"]
        st.markdown(
            f"<p style='font-size:0.78rem;color:{c_text};padding-top:1.9rem;'>"
            "XGBoost, Evidently, FastAPI, Streamlit &nbsp; v1.0</p>",
            unsafe_allow_html=True,
        )

    # ── Load data ────────────────────────────────────────────
    all_metrics = load_all_metrics(metrics_dir)
    if not all_metrics:
        st.error("No metrics found. Run `make monitor` first.")
        return

    rows: list[dict[str, Any]] = []
    for m in all_metrics:
        row: dict[str, Any] = {"week": m["week"]}
        row.update(m.get("classification", {}))
        row.update({
            "share_drifted": m.get("dataset_drift", {}).get("share_drifted_features", 0.0),
            "has_alert": m.get("has_alert", False),
        })
        rows.append(row)

    df = pd.DataFrame(rows).sort_values("week")
    baseline_f1   = df["f1"].iloc[0] if len(df) > 0 else 1.0
    latest        = df.iloc[-1]
    f1_alert_line = baseline_f1 * (1 - f1_drop_threshold)
    n_alerts      = int(df["has_alert"].sum())

    # ── KPI cards ────────────────────────────────────────────
    f1_latest  = latest.get("f1", 0)
    f1_delta   = f1_latest - baseline_f1
    share_val  = latest.get("share_drifted", 0)

    delta_cls  = "down" if f1_delta < 0 else "up"
    delta_sign = f"{f1_delta:+.3f}"
    share_cls  = "warn" if share_val > 0.5 else ("down" if share_val > 0.2 else "up")

    st.markdown(f"""
<div class="kpi-grid">
  <div class="kpi-card">
    <div class="kpi-label">Weeks Monitored</div>
    <div class="kpi-value">{len(df)}</div>
  </div>
  <div class="kpi-card">
    <div class="kpi-label">Latest F1</div>
    <div class="kpi-value">{f1_latest:.3f}</div>
    <div class="kpi-delta {delta_cls}">{delta_sign} vs baseline</div>
  </div>
  <div class="kpi-card">
    <div class="kpi-label">Alert Weeks</div>
    <div class="kpi-value">{n_alerts}</div>
    <div class="kpi-delta {'down' if n_alerts > 0 else 'up'}">{n_alerts} of {len(df)} weeks</div>
  </div>
  <div class="kpi-card">
    <div class="kpi-label">Share Drifted</div>
    <div class="kpi-value">{share_val:.1%}</div>
    <div class="kpi-delta {share_cls}">latest week</div>
  </div>
</div>
""", unsafe_allow_html=True)

    st.divider()

    # ── 1. F1-Score Timeline ─────────────────────────────────
    st.subheader("Model Performance Over Time")

    point_colors = [
        NORR["terracotta"] if v < f1_alert_line else NORR["sky"]
        for v in df["f1"]
    ]

    fig_f1 = go.Figure()
    # Area fill for context
    fig_f1.add_trace(go.Scatter(
        x=df["week"], y=df["f1"],
        fill="tozeroy",
        fillcolor=f"rgba(110,180,209,0.08)",
        line=dict(color="rgba(0,0,0,0)"),
        showlegend=False, hoverinfo="skip",
    ))
    # Main line
    fig_f1.add_trace(go.Scatter(
        x=df["week"], y=df["f1"],
        mode="lines+markers",
        name="F1-Score",
        line=dict(color=NORR["sky"], width=2.5),
        marker=dict(size=9, color=point_colors, line=dict(width=2, color=NORR["white"])),
        hovertemplate="Week %{x}<br>F1: %{y:.4f}<extra></extra>",
    ))
    # Threshold line
    fig_f1.add_hline(
        y=f1_alert_line,
        line_dash="dot",
        line_color=NORR["terracotta"],
        line_width=1.5,
        annotation_text=f"Alert threshold ({f1_alert_line:.3f})",
        annotation_font_color=NORR["terracotta"],
        annotation_font_size=11,
    )
    fig_f1.update_layout(
        **norr_layout(height=320),
        xaxis_title="Week",
        yaxis_title="F1-Score",
        yaxis_range=[0, 1.05],
    )
    st.plotly_chart(fig_f1, use_container_width=True)

    st.divider()

    # ── 2. Feature Drift Heatmap ─────────────────────────────
    st.subheader("Feature Drift Heatmap")
    st.markdown(
        f"<p style='font-size:0.875rem;color:{NORR['text_secondary']};margin-top:-0.75rem;margin-bottom:1rem;'>"
        "Drift score por feature por semana. Acima de 0.20 = alerta.</p>",
        unsafe_allow_html=True,
    )

    drift_rows: list[dict[str, Any]] = []
    for m in all_metrics:
        for feat, score in m.get("feature_drift_scores", {}).items():
            drift_rows.append({"week": m["week"], "feature": feat, "drift_score": round(score, 4)})

    if drift_rows:
        df_drift  = pd.DataFrame(drift_rows)
        df_pivot  = df_drift.pivot(index="feature", columns="week", values="drift_score").fillna(0)

        # Custom colorscale: cream → sky → terracotta
        colorscale = [
            [0.0,  NORR["bg_primary"]],
            [0.15, NORR["sky"]],
            [0.4,  NORR["amber"]],
            [1.0,  NORR["terracotta"]],
        ]

        fig_heat = go.Figure(go.Heatmap(
            z=df_pivot.values,
            x=[f"W{c}" for c in df_pivot.columns],
            y=df_pivot.index.tolist(),
            colorscale=colorscale,
            zmin=0, zmax=0.5,
            text=[[f"{v:.2f}" for v in row] for row in df_pivot.values],
            texttemplate="%{text}",
            textfont=dict(size=10, color=NORR["ink"]),
            hoverongaps=False,
            colorbar=dict(
                title=dict(text="Drift Score", font=dict(size=11, color=NORR["text_secondary"])),
                tickfont=dict(size=10, color=NORR["text_secondary"]),
                thickness=12,
                len=0.8,
            ),
        ))
        fig_heat.update_layout(**norr_layout(height=340))
        st.plotly_chart(fig_heat, use_container_width=True)
    else:
        st.info("Sem dados de drift por feature.")

    st.divider()

    # ── 3. Distribution Comparison ───────────────────────────
    st.subheader("Distribuição: Referência vs. Batch")

    col_a, col_b = st.columns([1, 3])
    with col_a:
        selected_week = st.selectbox(
            "Semana",
            options=df["week"].tolist(),
            index=len(df) - 1,
        )
        drift_features = cfg["drift_simulation"]["drift_features"]
        selected_feature = st.selectbox("Feature", options=drift_features)

    ref_df   = load_reference(ref_path)
    batch_df = load_batch(simulated_dir, selected_week)

    if not ref_df.empty and not batch_df.empty and selected_feature in ref_df.columns:
        with col_b:
            fig_dist = go.Figure()
            fig_dist.add_trace(go.Histogram(
                x=ref_df[selected_feature],
                name="Referência",
                opacity=0.75,
                nbinsx=50,
                marker_color=NORR["sky"],
            ))
            fig_dist.add_trace(go.Histogram(
                x=batch_df[selected_feature],
                name=f"Semana {selected_week}",
                opacity=0.75,
                nbinsx=50,
                marker_color=NORR["terracotta"],
            ))
            fig_dist.update_layout(
                **norr_layout(height=300),
                barmode="overlay",
                xaxis_title=selected_feature,
                yaxis_title="Count",
            )
            st.plotly_chart(fig_dist, use_container_width=True)
    else:
        st.info("Dados não disponíveis. Rode `make train` e `make simulate` primeiro.")

    st.divider()

    # ── 4. Alert Panel ───────────────────────────────────────
    st.subheader("Painel de Alertas")

    alert_df = df[df["has_alert"]]
    if len(alert_df) == 0:
        st.markdown(
            f"<div class='norr-ok'>Nenhum alerta detectado nas {len(df)} semanas monitoradas.</div>",
            unsafe_allow_html=True,
        )
    else:
        for _, row in alert_df.iterrows():
            week_num   = int(row["week"])
            week_data  = all_metrics[week_num - 1]
            f1_val     = row.get("f1", 0)
            share_val  = row.get("share_drifted", 0)

            severity   = "Critical" if f1_val < f1_alert_line else "Warning"
            label      = f"Semana {week_num} / {severity} / F1 {f1_val:.3f} / Share drifted {share_val:.1%}"

            with st.expander(label, expanded=(week_num == int(alert_df.iloc[-1]["week"]))):
                for alert_msg in week_data.get("alerts", []):
                    st.error(alert_msg)

                m1, m2, m3 = st.columns(3)
                m1.metric("F1-Score",      f"{f1_val:.4f}")
                m2.metric("Precisão",      f"{row.get('precision', 0):.4f}")
                m3.metric("Recall",        f"{row.get('recall', 0):.4f}")

                report_path = Path(reports_dir) / f"week_{week_num:02d}.html"
                if report_path.exists():
                    c_secondary = NORR["text_secondary"]
                    c_sky = NORR["sky"]
                    st.markdown(
                        f"<p style='font-size:0.8rem;color:{c_secondary};margin-top:0.5rem;'>"
                        f"Relatório completo Evidently: "
                        f"<a href='http://localhost:8000/report/{week_num}' target='_blank' "
                        f"style='color:{c_sky};'>abrir via API →</a></p>",
                        unsafe_allow_html=True,
                    )


if __name__ == "__main__":
    main()
