"""Signal Viewer page — interactive PSG waveform display."""

import sys
from pathlib import Path

_repo = Path(__file__).resolve().parents[2]
if str(_repo / "src") not in sys.path:
    sys.path.insert(0, str(_repo / "src"))

import streamlit as st

from app.components import header, prediction_block
from app.state import get_predictor, init_session
from sleep_staging.config import CACHE_DIR
from sleep_staging.data import available_subjects, get_contiguous_sequence, load_cached_subject
from sleep_staging.preprocessing import check_epoch_quality
from sleep_staging.visualization import CHANNEL_NAMES, create_signal_figure

st.set_page_config(page_title="NeuroSleep — Signal Viewer", page_icon="🧠", layout="wide")
header()
init_session()
predictor = get_predictor()

st.markdown("---")

# ── Controls ───────────────────────────────────────────────────────────
subjects = available_subjects()
if not subjects:
    st.warning("No cached data found. Run Notebook 02 first.")
    st.stop()

col1, col2, col3 = st.columns(3)
with col1:
    subject = st.selectbox("Subject", subjects)
with col2:
    data = load_cached_subject(subject)
    n_epochs = data["epochs"].shape[0]
    epoch_idx = st.slider("Epoch", 0, n_epochs - 1, value=min(50, n_epochs - 1))
with col3:
    channel = st.selectbox("Channel", CHANNEL_NAMES, index=0)

# ── Display the selected epoch waveform ────────────────────────────────
epoch_data = data["epochs"][epoch_idx]
fig = create_signal_figure(epoch_data, CHANNEL_NAMES, title=f"Subject {subject} — Epoch {epoch_idx}")
st.plotly_chart(fig, use_container_width=True)

# ── QC ─────────────────────────────────────────────────────────────────
qc = check_epoch_quality(epoch_data, CHANNEL_NAMES)
from app.components import qc_panel
qc_panel(qc)

# ── Run prediction on this epoch ──────────────────────────────────────
if st.button("RUN PREDICTION", type="primary", use_container_width=True):
    try:
        seq = get_contiguous_sequence(data["epochs"], epoch_idx, seq_len=10)
        result = predictor.predict(seq, target_epoch=9)
        st.session_state.current_prediction = result
        prediction_block(result)

        from sleep_staging.visualization import create_probability_figure
        st.plotly_chart(
            create_probability_figure(result.probabilities),
            use_container_width=True,
        )
    except Exception as e:
        st.error(f"Prediction failed: {e}")
