"""Sleep Night Explorer — Full-night hypnogram and prediction timeline."""

import sys
from pathlib import Path

_repo = Path(__file__).resolve().parents[2]
if str(_repo) not in sys.path:
    sys.path.insert(0, str(_repo))
if str(_repo / "src") not in sys.path:
    sys.path.insert(0, str(_repo / "src"))

import numpy as np
import streamlit as st

from app.components import inject_swiss_css, header, section_title
from app.state import get_predictor, init_session
from sleep_staging.config import STAGE_NAMES
from sleep_staging.data import available_subjects, get_contiguous_sequence, load_cached_subject
from sleep_staging.visualization import create_hypnogram

st.set_page_config(page_title="NeuroSleep — Night Explorer", page_icon=None, layout="wide")
inject_swiss_css()
header()
init_session()
predictor = get_predictor()

st.markdown('<div class="divider-thick"></div>', unsafe_allow_html=True)

subjects = available_subjects()
if not subjects:
    st.warning("No cached data found. Run Notebook 02 first.")
    st.stop()

subject = st.selectbox("Subject", subjects)
data = load_cached_subject(subject)
n_epochs = data["epochs"].shape[0]

st.markdown(
    f'<div class="sz-body" style="margin-bottom:1.5rem;">'
    f"<strong>{subject}</strong> &nbsp;&middot;&nbsp; {n_epochs} epochs "
    f"&nbsp;&middot;&nbsp; {n_epochs * 30 / 60:.0f} minutes"
    f"</div>",
    unsafe_allow_html=True,
)

# ── Batch Prediction ────────────────────────────────────────────────────
if st.button("Predict Full Night", type="primary", use_container_width=True):
    with st.spinner("Running inference across all epochs..."):
        all_preds = []
        step = 10
        for start in range(0, n_epochs - 9, step):
            seq = get_contiguous_sequence(data["epochs"], start, seq_len=10)
            result = predictor.predict(seq, target_epoch=9)
            all_preds.append(result.stage_index)
        while len(all_preds) < n_epochs:
            all_preds.append(all_preds[-1])

    st.session_state["night_predictions"] = all_preds

    fig = create_hypnogram(all_preds, STAGE_NAMES, title=f"Predicted Hypnogram — {subject}")
    st.plotly_chart(fig, use_container_width=True)

    # Stage Distribution
    from collections import Counter
    counts = Counter(all_preds)
    total = len(all_preds)

    section_title("Stage Distribution")

    rows = ""
    for idx in sorted(counts):
        name = STAGE_NAMES.get(idx, "?")
        pct = counts[idx] / total * 100
        rows += (
            f'<div style="display:flex;justify-content:space-between;padding:0.5rem 0;'
            f'border-bottom:1px solid #f2f2f2;font-size:0.8rem;">'
            f'<span style="font-weight:600;">{name}</span>'
            f'<span style="color:#999;">{counts[idx]} epochs ({pct:.1f}%)</span></div>'
        )
    st.markdown(rows, unsafe_allow_html=True)

elif "night_predictions" in st.session_state:
    fig = create_hypnogram(
        st.session_state["night_predictions"], STAGE_NAMES,
        title=f"Predicted Hypnogram — {subject}",
    )
    st.plotly_chart(fig, use_container_width=True)
