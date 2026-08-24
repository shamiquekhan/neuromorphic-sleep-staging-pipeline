"""Reusable Streamlit UI components — Swiss Style Design System."""

import streamlit as st

from sleep_staging.config import STAGE_COLORS

# ── Swiss Design System ──────────────────────────────────────────────────

SWISS_CSS = """
<style>
/* ── Import grotesque sans-serif ──────────────────────────────────────── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap');

/* ── Root Variables ───────────────────────────────────────────────────── */
:root {
    --swiss-black: #111111;
    --swiss-dark: #1a1a1a;
    --swiss-gray-900: #222222;
    --swiss-gray-800: #333333;
    --swiss-gray-700: #555555;
    --swiss-gray-600: #777777;
    --swiss-gray-500: #999999;
    --swiss-gray-400: #bbbbbb;
    --swiss-gray-300: #dddddd;
    --swiss-gray-200: #e8e8e8;
    --swiss-gray-100: #f2f2f2;
    --swiss-gray-50: #f8f8f8;
    --swiss-white: #ffffff;
    --swiss-accent: #E63946;
    --swiss-accent-dark: #c5303c;
    --swiss-border: #e0e0e0;
    --swiss-border-strong: #cccccc;
    --swiss-radius: 0px;
    --swiss-shadow: none;
}

/* ── Global Reset ────────────────────────────────────────────────────── */
.stApp {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Helvetica Neue', sans-serif;
    color: var(--swiss-black);
    background: var(--swiss-white);
}

/* ── Hide Streamlit branding ─────────────────────────────────────────── */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}

/* ── Typography Scale ────────────────────────────────────────────────── */
.swiss-title {
    font-family: 'Inter', sans-serif;
    font-size: 3.2rem;
    font-weight: 900;
    letter-spacing: -0.03em;
    line-height: 1.0;
    color: var(--swiss-black);
    margin: 0;
    text-transform: uppercase;
}

.swiss-subtitle {
    font-family: 'Inter', sans-serif;
    font-size: 0.85rem;
    font-weight: 400;
    letter-spacing: 0.12em;
    color: var(--swiss-gray-600);
    text-transform: uppercase;
    margin-top: 0.5rem;
}

.swiss-section-title {
    font-family: 'Inter', sans-serif;
    font-size: 0.7rem;
    font-weight: 700;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    color: var(--swiss-gray-500);
    margin-bottom: 0.75rem;
    padding-bottom: 0.5rem;
    border-bottom: 1px solid var(--swiss-border);
}

.swiss-label {
    font-family: 'Inter', sans-serif;
    font-size: 0.65rem;
    font-weight: 600;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: var(--swiss-gray-500);
    margin-bottom: 0.25rem;
}

.swiss-value {
    font-family: 'Inter', sans-serif;
    font-size: 1.1rem;
    font-weight: 600;
    color: var(--swiss-black);
    margin-top: 0;
}

.swiss-value-large {
    font-family: 'Inter', sans-serif;
    font-size: 3.5rem;
    font-weight: 900;
    letter-spacing: -0.03em;
    line-height: 1.0;
    color: var(--swiss-black);
}

.swiss-value-accent {
    font-family: 'Inter', sans-serif;
    font-size: 3.5rem;
    font-weight: 900;
    letter-spacing: -0.03em;
    line-height: 1.0;
    color: var(--swiss-accent);
}

/* ── Swiss Card System ───────────────────────────────────────────────── */
.swiss-card {
    border: 1px solid var(--swiss-border);
    border-radius: var(--swiss-radius);
    padding: 1.5rem;
    background: var(--swiss-white);
    margin-bottom: 1rem;
}

.swiss-card-accent {
    border: 1px solid var(--swiss-accent);
    border-left: 3px solid var(--swiss-accent);
    border-radius: var(--swiss-radius);
    padding: 1.5rem;
    background: var(--swiss-white);
    margin-bottom: 1rem;
}

/* ── Status Cards ────────────────────────────────────────────────────── */
.swiss-status-grid {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 1px;
    background: var(--swiss-border);
    border: 1px solid var(--swiss-border);
    margin-bottom: 2rem;
}

.swiss-status-item {
    background: var(--swiss-white);
    padding: 1.25rem 1rem;
    text-align: left;
}

.swiss-status-label {
    font-family: 'Inter', sans-serif;
    font-size: 0.6rem;
    font-weight: 700;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    color: var(--swiss-gray-500);
    margin-bottom: 0.5rem;
}

.swiss-status-value {
    font-family: 'Inter', sans-serif;
    font-size: 1.0rem;
    font-weight: 700;
    color: var(--swiss-black);
}

.swiss-status-value--accent {
    color: var(--swiss-accent);
}

/* ── Prediction Block ────────────────────────────────────────────────── */
.swiss-prediction {
    border: 1px solid var(--swiss-border);
    border-radius: var(--swiss-radius);
    padding: 2.5rem 2rem;
    background: var(--swiss-white);
    text-align: center;
    margin: 1.5rem 0;
}

.swiss-prediction-label {
    font-family: 'Inter', sans-serif;
    font-size: 0.65rem;
    font-weight: 700;
    letter-spacing: 0.2em;
    text-transform: uppercase;
    color: var(--swiss-gray-500);
    margin-bottom: 0.75rem;
}

.swiss-prediction-stage {
    font-family: 'Inter', sans-serif;
    font-size: 5rem;
    font-weight: 900;
    letter-spacing: -0.04em;
    line-height: 1.0;
    margin: 0;
}

.swiss-prediction-meta {
    font-family: 'Inter', sans-serif;
    font-size: 0.8rem;
    font-weight: 400;
    color: var(--swiss-gray-600);
    margin-top: 0.75rem;
    letter-spacing: 0.02em;
}

/* ── QC Panel ────────────────────────────────────────────────────────── */
.swiss-qc-row {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 0.5rem 0;
    border-bottom: 1px solid var(--swiss-gray-100);
    font-family: 'Inter', sans-serif;
    font-size: 0.85rem;
}

.swiss-qc-channel {
    font-weight: 500;
    color: var(--swiss-black);
}

.swiss-qc-pass {
    font-weight: 700;
    font-size: 0.7rem;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: #2d8a4e;
}

.swiss-qc-fail {
    font-weight: 700;
    font-size: 0.7rem;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: var(--swiss-accent);
}

/* ── Divider ─────────────────────────────────────────────────────────── */
.swiss-divider {
    border: none;
    border-top: 1px solid var(--swiss-border);
    margin: 2rem 0;
}

.swiss-divider-thick {
    border: none;
    border-top: 2px solid var(--swiss-black);
    margin: 2rem 0;
}

/* ── Probability bars ────────────────────────────────────────────────── */
.swiss-prob-row {
    display: flex;
    align-items: center;
    gap: 0.75rem;
    margin-bottom: 0.5rem;
    font-family: 'Inter', sans-serif;
}

.swiss-prob-label {
    width: 3rem;
    font-size: 0.75rem;
    font-weight: 600;
    color: var(--swiss-black);
    text-align: right;
}

.swiss-prob-bar-bg {
    flex: 1;
    height: 6px;
    background: var(--swiss-gray-200);
    border-radius: 0;
}

.swiss-prob-bar {
    height: 100%;
    border-radius: 0;
    transition: width 0.3s ease;
}

.swiss-prob-value {
    width: 3.5rem;
    font-size: 0.75rem;
    font-weight: 500;
    color: var(--swiss-gray-600);
    text-align: right;
}

/* ── Architecture Diagram ────────────────────────────────────────────── */
.swiss-arch {
    font-family: 'Inter', monospace;
    font-size: 0.75rem;
    line-height: 1.8;
    color: var(--swiss-gray-700);
    background: var(--swiss-gray-50);
    border: 1px solid var(--swiss-border);
    padding: 1.5rem;
    white-space: pre;
    overflow-x: auto;
}

/* ── Table ────────────────────────────────────────────────────────────── */
.swiss-table {
    width: 100%;
    border-collapse: collapse;
    font-family: 'Inter', sans-serif;
    font-size: 0.85rem;
}

.swiss-table th {
    font-size: 0.65rem;
    font-weight: 700;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: var(--swiss-gray-500);
    text-align: left;
    padding: 0.75rem 1rem;
    border-bottom: 2px solid var(--swiss-black);
}

.swiss-table td {
    padding: 0.75rem 1rem;
    border-bottom: 1px solid var(--swiss-gray-200);
    color: var(--swiss-black);
}

.swiss-table td:last-child {
    font-weight: 600;
}

/* ── Sidebar ─────────────────────────────────────────────────────────── */
section[data-testid="stSidebar"] {
    background: var(--swiss-gray-50);
    border-right: 1px solid var(--swiss-border);
}

section[data-testid="stSidebar"] .stRadio label,
section[data-testid="stSidebar"] .stSelectbox label {
    font-family: 'Inter', sans-serif;
    font-size: 0.75rem;
    font-weight: 600;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: var(--swiss-gray-600);
}

/* ── Buttons ─────────────────────────────────────────────────────────── */
.stButton > button {
    font-family: 'Inter', sans-serif;
    font-size: 0.75rem;
    font-weight: 700;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    border-radius: 0;
    border: 1px solid var(--swiss-black);
    background: var(--swiss-black);
    color: var(--swiss-white);
    padding: 0.6rem 2rem;
    transition: all 0.15s ease;
}

.stButton > button:hover {
    background: var(--swiss-accent);
    border-color: var(--swiss-accent);
    color: var(--swiss-white);
}

.stButton > button[kind="secondary"] {
    background: var(--swiss-white);
    color: var(--swiss-black);
}

.stButton > button[kind="secondary"]:hover {
    background: var(--swiss-gray-100);
    border-color: var(--swiss-gray-400);
}

/* ── Inputs ──────────────────────────────────────────────────────────── */
.stSelectbox > div > div,
.stSlider > div > div > div,
.stRadio > div {
    font-family: 'Inter', sans-serif;
    font-size: 0.85rem;
}

/* ── Expander ────────────────────────────────────────────────────────── */
.streamlit-expanderHeader {
    font-family: 'Inter', sans-serif;
    font-size: 0.7rem;
    font-weight: 700;
    letter-spacing: 0.12em;
    text-transform: uppercase;
}

/* ── Grid Utilities ──────────────────────────────────────────────────── */
.swiss-grid-2 {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 1px;
    background: var(--swiss-border);
    border: 1px solid var(--swiss-border);
}

.swiss-grid-2 > div {
    background: var(--swiss-white);
    padding: 1.5rem;
}

.swiss-grid-3 {
    display: grid;
    grid-template-columns: 1fr 1fr 1fr;
    gap: 1px;
    background: var(--swiss-border);
    border: 1px solid var(--swiss-border);
}

.swiss-grid-3 > div {
    background: var(--swiss-white);
    padding: 1.25rem;
}

/* ── Scrollbar ───────────────────────────────────────────────────────── */
::-webkit-scrollbar {
    width: 4px;
}
::-webkit-scrollbar-track {
    background: var(--swiss-gray-100);
}
::-webkit-scrollbar-thumb {
    background: var(--swiss-gray-400);
}

/* ── Streamlit overrides for Swiss look ──────────────────────────────── */
div[data-testid="stVerticalBlock"] > div {
    gap: 0;
}

.stTabs [data-baseweb="tab-list"] {
    gap: 0;
    border-bottom: 1px solid var(--swiss-border);
}

.stTabs [data-baseweb="tab"] {
    font-family: 'Inter', sans-serif;
    font-size: 0.7rem;
    font-weight: 600;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    border-radius: 0;
    padding: 0.75rem 1.5rem;
}

.stTabs [aria-selected="true"] {
    border-bottom: 2px solid var(--swiss-accent);
    color: var(--swiss-accent);
}
</style>
"""


def inject_swiss_css() -> None:
    """Inject the complete Swiss Style CSS into the page."""
    st.markdown(SWISS_CSS, unsafe_allow_html=True)


def header() -> None:
    """Render the Swiss Style application header."""
    st.markdown(
        """
        <div style="padding: 1rem 0 2rem 0; border-bottom: 2px solid #111; margin-bottom: 2rem;">
            <div class="swiss-title">NEUROSLEEP</div>
            <div class="swiss-subtitle">Neuromorphic Sleep Stage Scoring &mdash; Sleep-EDF Research Demonstrator</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def model_status_cards(predictor) -> None:
    """Display model status in a Swiss grid."""
    info = predictor.model_info
    st.markdown(
        f"""
        <div class="swiss-status-grid">
            <div class="swiss-status-item">
                <div class="swiss-status-label">Model</div>
                <div class="swiss-status-value">{info['name']}</div>
            </div>
            <div class="swiss-status-item">
                <div class="swiss-status-label">Parameters</div>
                <div class="swiss-status-value">{info['parameters']:,}</div>
            </div>
            <div class="swiss-status-item">
                <div class="swiss-status-label">Device</div>
                <div class="swiss-status-value">{info['device'].upper()}</div>
            </div>
            <div class="swiss-status-item">
                <div class="swiss-status-label">Status</div>
                <div class="swiss-status-value swiss-status-value--accent">READY</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def prediction_block(result) -> None:
    """Display the current prediction in Swiss Style."""
    color = STAGE_COLORS.get(result.stage, "#111")
    st.markdown(
        f"""
        <div class="swiss-prediction">
            <div class="swiss-prediction-label">Current Sleep Stage</div>
            <div class="swiss-prediction-stage" style="color: {color};">{result.stage}</div>
            <div class="swiss-prediction-meta">
                Confidence {result.confidence:.1%} &nbsp;&middot;&nbsp; Latency {result.latency_ms:.1f} ms
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def probability_bars(probabilities: dict[str, float]) -> None:
    """Render probability distribution as Swiss-style horizontal bars."""
    max_prob = max(probabilities.values()) if probabilities else 1.0
    stage_colors_hex = {
        "Wake": "#FF6B6B",
        "N1": "#FFA07A",
        "N2": "#4ECDC4",
        "N3": "#2C73D2",
        "REM": "#9B59B6",
    }
    bars_html = '<div style="margin-top: 0.5rem;">'
    for stage, prob in probabilities.items():
        bar_color = stage_colors_hex.get(stage, "#888")
        width_pct = (prob / max_prob) * 100 if max_prob > 0 else 0
        bars_html += f"""
        <div class="swiss-prob-row">
            <div class="swiss-prob-label">{stage}</div>
            <div class="swiss-prob-bar-bg">
                <div class="swiss-prob-bar" style="width: {width_pct}%; background: {bar_color};"></div>
            </div>
            <div class="swiss-prob-value">{prob:.1%}</div>
        </div>
        """
    bars_html += "</div>"
    st.markdown(bars_html, unsafe_allow_html=True)


def qc_panel(qc_results: list) -> None:
    """Render signal quality status in Swiss Style."""
    all_pass = all(r.passed for r in qc_results)
    status = "READY FOR INFERENCE" if all_pass else "QC WARNING"
    status_class = "swiss-qc-pass" if all_pass else "swiss-qc-fail"

    rows_html = ""
    for r in qc_results:
        cls = "swiss-qc-pass" if r.passed else "swiss-qc-fail"
        label = "PASS" if r.passed else "FAIL"
        rows_html += f"""
        <div class="swiss-qc-row">
            <span class="swiss-qc-channel">{r.channel_name}</span>
            <span class="{cls}">{label}</span>
        </div>
        """

    st.markdown(
        f"""
        <div style="margin-bottom: 1.5rem;">
            <div class="swiss-section-title">Signal Quality</div>
            {rows_html}
            <div style="margin-top: 0.75rem; padding-top: 0.75rem; border-top: 1px solid #e8e8e8;">
                <span class="swiss-label">QC Status</span>
                <span class="{status_class}" style="margin-left: 0.5rem;">{status}</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def stage_tag(stage: str) -> str:
    """Return an inline HTML badge for a sleep stage."""
    color = STAGE_COLORS.get(stage, "#888")
    return (
        f'<span style="display:inline-block; padding:0.15rem 0.5rem; '
        f'font-size:0.7rem; font-weight:700; letter-spacing:0.08em; '
        f'text-transform:uppercase; color:{color}; border:1px solid {color}; '
        f'border-radius:0;">{stage}</span>'
    )
