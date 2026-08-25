"""Signal Viewer — Interactive PSG waveform display."""

import sys
from pathlib import Path

_repo = Path(__file__).resolve().parents[2]
if str(_repo) not in sys.path:
    sys.path.insert(0, str(_repo))
if str(_repo / "src") not in sys.path:
    sys.path.insert(0, str(_repo / "src"))

import streamlit as st

from app.components import (
    inject_swiss_css, header, section_title,
    prediction_block, probability_bars, qc_panel,
)
from app.state import get_predictor, init_session
from sleep_staging.config import CACHE_DIR
from sleep_staging.data import available_subjects, get_contiguous_sequence, load_cached_subject
from sleep_staging.preprocessing import check_epoch_quality
from sleep_staging.visualization import CHANNEL_NAMES, create_signal_figure

st.set_page_config(page_title="NeuroSleep — Signal Viewer", page_icon=None, layout="wide")
inject_swiss_css()
header()
init_session()
predictor = get_predictor()

st.markdown('<div class="divider-thick"></div>', unsafe_allow_html=True)

subjects = available_subjects()
if not subjects:
    st.warning("No cached data found. Run Notebook 02 first.")
    st.stop()

section_title("Signal Selection")

c1, c2, c3 = st.columns(3)
with c1:
    subject = st.selectbox("Subject", subjects)
with c2:
    data = load_cached_subject(subject)
    n_epochs = data["epochs"].shape[0]
    epoch_idx = st.slider("Epoch", 0, n_epochs - 1, value=min(50, n_epochs - 1))
with c3:
    channel = st.selectbox("Channel", CHANNEL_NAMES, index=0)

# ── Waveform ────────────────────────────────────────────────────────────
epoch_data = data["epochs"][epoch_idx]
fig = create_signal_figure(epoch_data, CHANNEL_NAMES, title=f"{subject} — Epoch {epoch_idx}")
st.plotly_chart(fig, use_container_width=True)

# ── QC ──────────────────────────────────────────────────────────────────
qc = check_epoch_quality(epoch_data, CHANNEL_NAMES)
qc_panel(qc)

# ── Prediction ──────────────────────────────────────────────────────────
st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

if st.button("Run Prediction", type="primary", use_container_width=True):
    try:
        seq = get_contiguous_sequence(data["epochs"], epoch_idx, seq_len=10)
        result = predictor.predict(seq, target_epoch=9)
        st.session_state.current_prediction = result

        prediction_block(result)

        section_title("Probability Distribution")
        probability_bars(result.probabilities)

    except Exception as exc:
        st.error(f"Prediction failed: {exc}")
