"""Dashboard page — main exhibition view with LoRA toggle."""

import sys
from pathlib import Path

_repo = Path(__file__).resolve().parents[2]
if str(_repo / "src") not in sys.path:
    sys.path.insert(0, str(_repo / "src"))

import numpy as np
import streamlit as st

from app.components import header, model_status_cards, prediction_block
from app.state import get_predictor, init_session
from sleep_staging.config import CACHE_DIR, STAGE_NAMES
from sleep_staging.data import available_subjects, get_contiguous_sequence, load_cached_subject
from sleep_staging.preprocessing import check_epoch_quality
from sleep_staging.visualization import CHANNEL_NAMES, create_probability_figure

st.set_page_config(page_title="NeuroSleep — Dashboard", page_icon="🧠", layout="wide")
header()
init_session()

# ── Model Mode Selection ──────────────────────────────────────────────
st.markdown("---")
col_mode1, col_mode2 = st.columns([1, 3])
with col_mode1:
    mode = st.radio(
        "Model Mode",
        ["Base", "LoRA Adapted"],
        index=0 if st.session_state.model_mode == "base" else 1,
        key="mode_radio",
    )
    st.session_state.model_mode = "base" if mode == "Base" else "lora"

adapter_path = None
if st.session_state.model_mode == "lora":
    with col_mode2:
        adapter_options = []
        lora_dir = Path(_repo) / "artifacts" / "lora"
        if lora_dir.exists():
            adapter_options = [str(p) for p in lora_dir.iterdir() if p.is_dir()]
        if adapter_options:
            chosen = st.selectbox("Select adapter", adapter_options)
            adapter_path = chosen
        else:
            st.warning("No LoRA adapters found. Run `scripts/train_lora_adapter.py` first.")

predictor = get_predictor(adapter_path=adapter_path)
init_session()
model_status_cards(predictor)

st.markdown("---")

# ── Exhibition Demo Mode ──────────────────────────────────────────────
if st.button("START EXHIBITION DEMO", type="primary", use_container_width=True):
    st.session_state.demo_mode = True
    st.session_state.run_inference = True

# ── Dataset Selection ──────────────────────────────────────────────────
subjects = available_subjects()
if not subjects:
    st.warning("No cached data found. Run Notebook 02 first.")
    st.stop()

col1, col2, col3 = st.columns(3)
with col1:
    subject = st.selectbox("Subject", subjects, key="subject_select")
    st.session_state.selected_subject = subject

with col2:
    data = load_cached_subject(subject)
    n_epochs = data["epochs"].shape[0]
    epoch_idx = st.slider(
        "Epoch", 0, n_epochs - 1,
        value=min(st.session_state.selected_epoch, n_epochs - 1),
        key="epoch_slider",
    )
    st.session_state.selected_epoch = epoch_idx

with col3:
    target = st.selectbox("Target epoch in window", list(range(10)), index=9)
    st.session_state.selected_target_epoch = target

# ── Run Prediction ────────────────────────────────────────────────────
run = st.button("RUN PREDICTION", type="secondary", use_container_width=True)
if run or st.session_state.demo_mode:
    try:
        seq = get_contiguous_sequence(data["epochs"], epoch_idx, seq_len=10)
        result = predictor.predict(seq, target_epoch=st.session_state.selected_target_epoch)
        st.session_state.current_prediction = result
        st.session_state.current_sequence = seq

        # QC
        target_epoch_data = seq[0, st.session_state.selected_target_epoch]
        qc = check_epoch_quality(target_epoch_data, CHANNEL_NAMES)
        from app.components import qc_panel
        qc_panel(qc)

        prediction_block(result)

        st.plotly_chart(
            create_probability_figure(result.probabilities),
            use_container_width=True,
        )

    except Exception as e:
        st.error(f"Prediction failed: {e}")
    finally:
        st.session_state.demo_mode = False

# ── Show last result if available ─────────────────────────────────────
elif st.session_state.current_prediction is not None:
    prediction_block(st.session_state.current_prediction)
    st.plotly_chart(
        create_probability_figure(st.session_state.current_prediction.probabilities),
        use_container_width=True,
    )
