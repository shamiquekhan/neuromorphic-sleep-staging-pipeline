"""NeuroSleep — Streamlit Application.

Swiss Minimalist Design. Light theme. Run locally first.

    streamlit run app/streamlit_app.py
"""

import sys
from pathlib import Path

_repo = Path(__file__).resolve().parents[1]
if str(_repo) not in sys.path:
    sys.path.insert(0, str(_repo))
if str(_repo / "src") not in sys.path:
    sys.path.insert(0, str(_repo / "src"))

import streamlit as st

st.set_page_config(page_title="NeuroSleep", page_icon=None, layout="wide")

from app.components import inject_swiss_css, header, model_status_grid, section_title
from app.state import get_predictor, init_session, load_final_metrics

inject_swiss_css()
header()

predictor = get_predictor()
init_session()

model_status_grid(predictor)

st.markdown('<div class="divider-thick"></div>', unsafe_allow_html=True)

col_main, col_side = st.columns([3, 1])

with col_main:
    section_title("How to Use")
    st.markdown(
        '<div class="sz-body">'
        "<strong>1.</strong> Open <strong>Signal Viewer</strong> to browse PSG recordings.<br>"
        "<strong>2.</strong> Open <strong>Night Explorer</strong> for full-night hypnograms.<br>"
        "<strong>3.</strong> Open <strong>Model Information</strong> for architecture and results."
        "</div>",
        unsafe_allow_html=True,
    )

with col_side:
    section_title("System")
    st.markdown(
        '<table class="swiss-table">'
        "<tr><td>Dataset</td><td>Sleep-EDF (92 subjects)</td></tr>"
        "<tr><td>Window</td><td>10 &times; 30 s</td></tr>"
        "<tr><td>Sampling</td><td>100 Hz</td></tr>"
        "<tr><td>Model</td><td>Improved Student</td></tr>"
        "<tr><td>Parameters</td><td>99,477</td></tr>"
        "<tr><td>Evaluation</td><td>10-fold CV, 3 seeds</td></tr>"
        "</table>",
        unsafe_allow_html=True,
    )

# ── Final Metrics Banner ────────────────────────────────────────────────
import json as _json
_agg_path = Path(__file__).resolve().parents[2] / "results" / "100_subject_adaptation" / "final" / "aggregate_metrics.json"
metrics_100 = {}
if _agg_path.exists():
    with open(_agg_path) as _f:
        metrics_100 = _json.load(_f)

if metrics_100 and "full_finetune" in metrics_100:
    ft = metrics_100["full_finetune"]["overall"]
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    section_title("Final Results (100-Subject Benchmark)")
    st.markdown(
        f'<div class="swiss-grid-4">'
        f'<div><div class="sz-label">Accuracy</div><div class="sz-display">{ft["accuracy"]["mean"]:.1%}</div>'
        f'<div class="sz-caption">&plusmn; {ft["accuracy"]["std"]:.1%}</div></div>'
        f'<div><div class="sz-label">Cohen&rsquo;s &kappa;</div><div class="sz-display">{ft["kappa"]["mean"]:.3f}</div>'
        f'<div class="sz-caption">&plusmn; {ft["kappa"]["std"]:.3f}</div></div>'
        f'<div><div class="sz-label">Macro F1</div><div class="sz-display">{ft["macro_f1"]["mean"]:.3f}</div>'
        f'<div class="sz-caption">&plusmn; {ft["macro_f1"]["std"]:.3f}</div></div>'
        f'<div><div class="sz-label">Parameters</div><div class="sz-display">99,477</div></div>'
        f'</div>',
        unsafe_allow_html=True,
    )
else:
    metrics = load_final_metrics()
    if metrics:
        acc = metrics.get("accuracy", {})
        kappa = metrics.get("cohen_kappa", {})
        macro = metrics.get("macro_f1", {})
        st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
        section_title("Final Results")
        st.markdown(
            f'<div class="swiss-grid-4">'
            f'<div><div class="sz-label">Accuracy</div><div class="sz-display">{acc.get("mean",0):.1%}</div></div>'
            f'<div><div class="sz-label">Cohen&rsquo;s &kappa;</div><div class="sz-display">{kappa.get("mean",0):.3f}</div></div>'
            f'<div><div class="sz-label">Macro F1</div><div class="sz-display">{macro.get("mean",0):.3f}</div></div>'
            f'<div><div class="sz-label">Parameters</div><div class="sz-display">99,477</div></div>'
            f'</div>',
            unsafe_allow_html=True,
        )

st.markdown(
    '<div style="border-top:1px solid #e0e0e0;margin-top:3rem;padding:1rem 0;font-size:0.6rem;'
    'letter-spacing:0.12em;text-transform:uppercase;color:#bbb;text-align:center;">'
    "VIT Bhopal University &middot; NeuroSleep</div>",
    unsafe_allow_html=True,
)
