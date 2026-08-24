"""Sleep Night Explorer page — full-night hypnogram and prediction timeline."""

import sys
from pathlib import Path

_repo = Path(__file__).resolve().parents[2]
if str(_repo / "src") not in sys.path:
    sys.path.insert(0, str(_repo / "src"))

import numpy as np
import streamlit as st

from app.components import header
from app.state import get_predictor, init_session
from sleep_staging.config import STAGE_NAMES
from sleep_staging.data import available_subjects, get_contiguous_sequence, load_cached_subject
from sleep_staging.visualization import create_hypnogram

st.set_page_config(page_title="NeuroSleep — Night Explorer", page_icon="🧠", layout="wide")
header()
init_session()
predictor = get_predictor()

st.markdown("---")

subjects = available_subjects()
if not subjects:
    st.warning("No cached data found. Run Notebook 02 first.")
    st.stop()

subject = st.selectbox("Subject", subjects)
data = load_cached_subject(subject)
n_epochs = data["epochs"].shape[0]

st.markdown(f"**{subject}** — {n_epochs} epochs ({n_epochs * 30 / 60:.0f} minutes)")

# ── Batch prediction for full night ────────────────────────────────────
if st.button("PREDICT FULL NIGHT", type="primary", use_container_width=True):
    with st.spinner("Running inference across all epochs..."):
        all_preds = []
        step = 10
        for start in range(0, n_epochs - 9, step):
            seq = get_contiguous_sequence(data["epochs"], start, seq_len=10)
            result = predictor.predict(seq, target_epoch=9)
            all_preds.append(result.stage_index)
        # fill remaining
        while len(all_preds) < n_epochs:
            all_preds.append(all_preds[-1])

    st.session_state["night_predictions"] = all_preds

    fig = create_hypnogram(all_preds, STAGE_NAMES, title=f"Predicted Hypnogram — {subject}")
    st.plotly_chart(fig, use_container_width=True)

    # Stage distribution
    from collections import Counter
    counts = Counter(all_preds)
    total = len(all_preds)
    st.markdown("### Stage Distribution")
    for idx in sorted(counts):
        name = STAGE_NAMES.get(idx, "?")
        pct = counts[idx] / total * 100
        st.text(f"  {name:>4s}: {counts[idx]:>5d} epochs ({pct:.1f}%)")

elif "night_predictions" in st.session_state:
    fig = create_hypnogram(
        st.session_state["night_predictions"], STAGE_NAMES,
        title=f"Predicted Hypnogram — {subject}",
    )
    st.plotly_chart(fig, use_container_width=True)
