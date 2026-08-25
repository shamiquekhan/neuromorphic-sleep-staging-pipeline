"""Reusable Streamlit UI components — Swiss Minimalist Design System.

Grid-based, typography-first, monochrome + accent, sharp corners, 1px borders.
"""

import streamlit as st

from sleep_staging.config import STAGE_COLORS

# ═══════════════════════════════════════════════════════════════════════════
#  SWISS DESIGN SYSTEM — CSS
# ═══════════════════════════════════════════════════════════════════════════

SWISS_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap');

/* ── Reset ────────────────────────────────────────────────────────────── */
.stApp {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    color: #111;
    background: #fff;
}
#MainMenu, footer, header { visibility: hidden; }

/* ── Type Scale ───────────────────────────────────────────────────────── */
.sz-hero {
    font-size: 4rem;
    font-weight: 900;
    letter-spacing: -0.04em;
    line-height: 0.92;
    text-transform: uppercase;
    color: #111;
}
.sz-display {
    font-size: 2.5rem;
    font-weight: 800;
    letter-spacing: -0.03em;
    line-height: 1.0;
    color: #111;
}
.sz-title {
    font-size: 1.1rem;
    font-weight: 700;
    letter-spacing: -0.01em;
    color: #111;
    margin: 0;
}
.sz-label {
    font-size: 0.6rem;
    font-weight: 700;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    color: #999;
}
.sz-mono {
    font-family: 'SF Mono', 'Fira Code', 'Consolas', monospace;
    font-size: 0.8rem;
    font-weight: 500;
    color: #555;
}
.sz-body {
    font-size: 0.85rem;
    font-weight: 400;
    line-height: 1.6;
    color: #333;
}
.sz-caption {
    font-size: 0.7rem;
    font-weight: 500;
    color: #999;
    letter-spacing: 0.02em;
}
.sz-accent {
    color: #E63946;
}

/* ── Layout Grid ──────────────────────────────────────────────────────── */
.swiss-grid-2 {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 1px;
    background: #e0e0e0;
    border: 1px solid #e0e0e0;
}
.swiss-grid-2 > div { background: #fff; padding: 1.5rem; }

.swiss-grid-3 {
    display: grid;
    grid-template-columns: 1fr 1fr 1fr;
    gap: 1px;
    background: #e0e0e0;
    border: 1px solid #e0e0e0;
}
.swiss-grid-3 > div { background: #fff; padding: 1.25rem; }

.swiss-grid-4 {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 1px;
    background: #e0e0e0;
    border: 1px solid #e0e0e0;
}
.swiss-grid-4 > div { background: #fff; padding: 1.25rem 1rem; }

/* ── Dividers ─────────────────────────────────────────────────────────── */
.divider-thick { border: none; border-top: 2px solid #111; margin: 2rem 0; }
.divider { border: none; border-top: 1px solid #e0e0e0; margin: 1.5rem 0; }
.divider-light { border: none; border-top: 1px solid #f2f2f2; margin: 1rem 0; }

/* ── Cards ────────────────────────────────────────────────────────────── */
.swiss-card {
    border: 1px solid #e0e0e0;
    padding: 1.5rem;
    background: #fff;
}
.swiss-card-accent {
    border: 1px solid #E63946;
    border-left: 3px solid #E63946;
    padding: 1.5rem;
    background: #fff;
}

/* ── Prediction Block ─────────────────────────────────────────────────── */
.prediction-stage {
    font-size: 6rem;
    font-weight: 900;
    letter-spacing: -0.05em;
    line-height: 0.9;
    text-align: center;
    margin: 1rem 0;
}
.prediction-meta {
    font-size: 0.8rem;
    color: #999;
    text-align: center;
    letter-spacing: 0.04em;
}

/* ── Probability Bars ─────────────────────────────────────────────────── */
.prob-row {
    display: flex;
    align-items: center;
    gap: 0.75rem;
    margin-bottom: 0.6rem;
}
.prob-label {
    width: 3rem;
    font-size: 0.7rem;
    font-weight: 700;
    text-align: right;
    color: #111;
}
.prob-track {
    flex: 1;
    height: 4px;
    background: #f2f2f2;
}
.prob-fill {
    height: 100%;
    transition: width 0.3s ease;
}
.prob-value {
    width: 3.5rem;
    font-size: 0.7rem;
    font-weight: 500;
    color: #999;
    text-align: right;
}

/* ── Architecture Diagram ─────────────────────────────────────────────── */
.arch-diagram {
    font-family: 'SF Mono', 'Fira Code', monospace;
    font-size: 0.7rem;
    line-height: 1.9;
    color: #555;
    background: #fafafa;
    border: 1px solid #e0e0e0;
    padding: 1.25rem;
    white-space: pre;
    overflow-x: auto;
}

/* ── Table ─────────────────────────────────────────────────────────────── */
.swiss-table {
    width: 100%;
    border-collapse: collapse;
    font-size: 0.8rem;
}
.swiss-table th {
    font-size: 0.6rem;
    font-weight: 700;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: #999;
    text-align: left;
    padding: 0.6rem 0.75rem;
    border-bottom: 2px solid #111;
}
.swiss-table td {
    padding: 0.6rem 0.75rem;
    border-bottom: 1px solid #f2f2f2;
    color: #111;
}
.swiss-table td:last-child { font-weight: 600; }

/* ── Buttons ──────────────────────────────────────────────────────────── */
.stButton > button {
    font-family: 'Inter', sans-serif;
    font-size: 0.65rem;
    font-weight: 700;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    border-radius: 0;
    border: 1px solid #111;
    background: #111;
    color: #fff;
    padding: 0.65rem 2rem;
    transition: all 0.12s ease;
}
.stButton > button:hover {
    background: #E63946;
    border-color: #E63946;
}
.stButton > button[kind="secondary"] {
    background: #fff;
    color: #111;
}
.stButton > button[kind="secondary"]:hover {
    background: #f2f2f2;
    border-color: #ccc;
}

/* ── Sidebar ──────────────────────────────────────────────────────────── */
section[data-testid="stSidebar"] {
    background: #fafafa;
    border-right: 1px solid #e0e0e0;
}

/* ── Streamlit Overrides ──────────────────────────────────────────────── */
div[data-testid="stVerticalBlock"] > div { gap: 0; }
.stTabs [data-baseweb="tab-list"] {
    gap: 0;
    border-bottom: 1px solid #e0e0e0;
}
.stTabs [data-baseweb="tab"] {
    font-family: 'Inter', sans-serif;
    font-size: 0.65rem;
    font-weight: 600;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    border-radius: 0;
    padding: 0.75rem 1.5rem;
}
.stTabs [aria-selected="true"] {
    border-bottom: 2px solid #E63946;
    color: #E63946;
}

/* ── Scrollbar ────────────────────────────────────────────────────────── */
::-webkit-scrollbar { width: 3px; }
::-webkit-scrollbar-track { background: #f8f8f8; }
::-webkit-scrollbar-thumb { background: #ddd; }
</style>
"""


# ═══════════════════════════════════════════════════════════════════════════
#  PUBLIC API
# ═══════════════════════════════════════════════════════════════════════════

def inject_swiss_css() -> None:
    st.markdown(SWISS_CSS, unsafe_allow_html=True)


def header() -> None:
    st.markdown(
        '<div class="sz-hero">NEUROSLEEP</div>'
        '<div class="sz-caption" style="margin-top:0.5rem;">Neuromorphic Sleep Stage Scoring</div>',
        unsafe_allow_html=True,
    )


def section_title(text: str) -> None:
    st.markdown(
        f'<div class="sz-label" style="margin-bottom:0.75rem; padding-bottom:0.5rem; '
        f'border-bottom:1px solid #e0e0e0;">{text}</div>',
        unsafe_allow_html=True,
    )


def model_status_grid(predictor) -> None:
    info = predictor.model_info
    st.markdown(
        f'<div class="swiss-grid-4">'
        f'<div><div class="sz-label">Model</div><div class="sz-title">{info["name"]}</div></div>'
        f'<div><div class="sz-label">Parameters</div><div class="sz-title">{info["parameters"]:,}</div></div>'
        f'<div><div class="sz-label">Device</div><div class="sz-title">{info["device"].upper()}</div></div>'
        f'<div><div class="sz-label">Status</div><div class="sz-title sz-accent">READY</div></div>'
        f'</div>',
        unsafe_allow_html=True,
    )


def prediction_block(result) -> None:
    color = STAGE_COLORS.get(result.stage, "#111")
    st.markdown(
        f'<div style="border:1px solid #e0e0e0; padding:2rem; text-align:center; margin:1.5rem 0;">'
        f'<div class="sz-label" style="margin-bottom:0.5rem;">Predicted Stage</div>'
        f'<div class="prediction-stage" style="color:{color};">{result.stage}</div>'
        f'<div class="prediction-meta">'
        f'{result.confidence:.1%} confidence &nbsp;&middot;&nbsp; {result.latency_ms:.1f} ms'
        f'</div></div>',
        unsafe_allow_html=True,
    )


def probability_bars(probabilities: dict[str, float]) -> None:
    max_prob = max(probabilities.values()) if probabilities else 1.0
    stage_hex = {
        "Wake": "#FF6B6B", "N1": "#FFA07A", "N2": "#4ECDC4",
        "N3": "#2C73D2", "REM": "#9B59B6",
    }
    html = '<div style="margin-top:0.5rem;">'
    for stage, prob in probabilities.items():
        color = stage_hex.get(stage, "#888")
        w = (prob / max_prob) * 100 if max_prob > 0 else 0
        html += (
            f'<div class="prob-row">'
            f'<div class="prob-label">{stage}</div>'
            f'<div class="prob-track"><div class="prob-fill" style="width:{w}%;background:{color};"></div></div>'
            f'<div class="prob-value">{prob:.1%}</div></div>'
        )
    html += "</div>"
    st.markdown(html, unsafe_allow_html=True)


def qc_panel(qc_results: list) -> None:
    all_pass = all(r.passed for r in qc_results)
    status = "PASS" if all_pass else "WARN"
    status_cls = "sz-accent" if not all_pass else ""

    rows = ""
    for r in qc_results:
        cls = "sz-accent" if not r.passed else ""
        label = "PASS" if r.passed else "FAIL"
        rows += (
            f'<div style="display:flex;justify-content:space-between;padding:0.4rem 0;'
            f'border-bottom:1px solid #f8f8f8;">'
            f'<span style="font-size:0.8rem;font-weight:500;">{r.channel_name}</span>'
            f'<span style="font-size:0.65rem;font-weight:700;letter-spacing:0.1em;text-transform:uppercase;'
            f'color:{"#E63946" if not r.passed else "#2d8a4e"};">{label}</span></div>'
        )
    st.markdown(
        f'<div style="margin-bottom:1.5rem;">'
        f'<div class="sz-label" style="margin-bottom:0.5rem;">Signal Quality</div>'
        f'{rows}'
        f'<div style="margin-top:0.5rem;padding-top:0.5rem;border-top:1px solid #f2f2f2;">'
        f'<span class="sz-caption">QC </span>'
        f'<span class="{status_cls}" style="font-size:0.65rem;font-weight:700;letter-spacing:0.1em;">{status}</span>'
        f'</div></div>',
        unsafe_allow_html=True,
    )


def stage_tag(stage: str) -> str:
    color = STAGE_COLORS.get(stage, "#888")
    return (
        f'<span style="display:inline-block;padding:0.1rem 0.4rem;font-size:0.6rem;'
        f'font-weight:700;letter-spacing:0.08em;text-transform:uppercase;color:{color};'
        f'border:1px solid {color};">{stage}</span>'
    )
