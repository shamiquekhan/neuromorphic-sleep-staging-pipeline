"""Dashboard — Exhibition view with prediction demo."""

import sys
from pathlib import Path

_repo = Path(__file__).resolve().parents[2]
if str(_repo) not in sys.path:
    sys.path.insert(0, str(_repo))
if str(_repo / "src") not in sys.path:
    sys.path.insert(0, str(_repo / "src"))

import numpy as np
import streamlit as st
from scipy.signal import resample

from app.components import (
    inject_swiss_css, header, model_status_grid, section_title,
    prediction_block, probability_bars, qc_panel,
)
from app.state import get_predictor, init_session
from sleep_staging.config import CACHE_DIR, STAGE_NAMES, PreprocessingConfig
from sleep_staging.data import available_subjects, get_contiguous_sequence, load_cached_subject
from sleep_staging.preprocessing import check_epoch_quality, bandpass_filter, notch_filter, zscore_normalize
from sleep_staging.visualization import CHANNEL_NAMES

st.set_page_config(page_title="NeuroSleep — Dashboard", page_icon=None, layout="wide")
inject_swiss_css()
header()

predictor = get_predictor()
init_session()

model_status_grid(predictor)

st.markdown('<div class="divider-thick"></div>', unsafe_allow_html=True)

# ── Demo Trigger ────────────────────────────────────────────────────────
col_demo, col_space = st.columns([1, 3])
with col_demo:
    if st.button("Start Demo", use_container_width=True):
        st.session_state.demo_mode = True
        st.session_state.run_inference = True

# ── Data Selection ──────────────────────────────────────────────────────
subjects = available_subjects()
has_cached_data = len(subjects) > 0

if not has_cached_data:
    st.info("No cached PSG data found. You can still use the **Upload Your ECG / PSG Data** section below.")

if has_cached_data:
    section_title("Data Selection")

    c1, c2, c3 = st.columns(3)
    with c1:
        subject = st.selectbox("Subject", subjects)
    with c2:
        data = load_cached_subject(subject)
        n_epochs = data["epochs"].shape[0]
        epoch_idx = st.slider("Epoch", 0, n_epochs - 1, value=min(50, n_epochs - 1))
    with c3:
        target = st.selectbox("Target epoch", list(range(10)), index=9)

    # ── Run Prediction ──────────────────────────────────────────────────
    run = st.button("Run Prediction", type="primary", use_container_width=True)
    if run or st.session_state.demo_mode:
        try:
            seq = get_contiguous_sequence(data["epochs"], epoch_idx, seq_len=10)
            result = predictor.predict(seq, target_epoch=target)
            st.session_state.current_prediction = result
            st.session_state.current_sequence = seq

            # QC
            target_epoch_data = seq[0, target]
            qc = check_epoch_quality(target_epoch_data, CHANNEL_NAMES)
            qc_panel(qc)

            # Prediction
            prediction_block(result)

            section_title("Probability Distribution")
            probability_bars(result.probabilities)

        except Exception as exc:
            st.error(f"Prediction failed: {exc}")
        finally:
            st.session_state.demo_mode = False

    elif st.session_state.current_prediction is not None:
        prediction_block(st.session_state.current_prediction)
        section_title("Probability Distribution")
        probability_bars(st.session_state.current_prediction.probabilities)

# ═══════════════════════════════════════════════════════════════════════════
#  USER ECG UPLOAD
# ═══════════════════════════════════════════════════════════════════════════
st.markdown('<div class="divider-thick"></div>', unsafe_allow_html=True)
section_title("Upload Your ECG / PSG Data")

st.markdown(
    '<div class="sz-body" style="margin-bottom:1rem;">'
    "Upload a <strong>CSV</strong> or <strong>.npy</strong> file with your signal data. "
    "The model expects <strong>4 channels</strong> (EEG Fpz-Cz, EEG Pz-Oz, EOG, EMG) "
    "sampled at <strong>100 Hz</strong>. If your data has a different sampling rate or "
    "channel count, it will be resampled/padded automatically."
    "</div>",
    unsafe_allow_html=True,
)

uploaded_file = st.file_uploader(
    "Choose a file",
    type=["csv", "npy"],
    help="CSV: rows = samples, columns = channels (or flat for single channel). "
         "NPY: numpy array with shape [samples, channels] or [channels, samples].",
)

if uploaded_file is not None:
    try:
        if uploaded_file.name.endswith(".npy"):
            raw_data = np.load(uploaded_file)
        else:
            import pandas as pd
            df = pd.read_csv(uploaded_file)
            raw_data = df.values.astype(np.float32)

        st.markdown(
            f'<div class="sz-caption" style="margin-bottom:0.5rem;">'
            f"Loaded: <strong>{uploaded_file.name}</strong> &nbsp;|&nbsp; "
            f"Shape: <strong>{raw_data.shape}</strong> &nbsp;|&nbsp; "
            f"Range: [{raw_data.min():.3f}, {raw_data.max():.3f}]"
            f"</div>",
            unsafe_allow_html=True,
        )

        # ── Normalize shape to [samples, channels] ──────────────────────
        if raw_data.ndim == 1:
            raw_data = raw_data.reshape(-1, 1)
        if raw_data.ndim == 3:
            # Could be [channels, samples] or [batch, channels, samples]
            if raw_data.shape[0] <= raw_data.shape[1] and raw_data.shape[0] <= raw_data.shape[2]:
                raw_data = raw_data.transpose(1, 0)  # [channels, samples] -> [samples, channels]
            else:
                raw_data = raw_data.reshape(-1, raw_data.shape[-1])

        n_samples, n_channels_raw = raw_data.shape

        # ── Channel mapping ─────────────────────────────────────────────
        cfg = PreprocessingConfig()
        target_channels = 4
        channel_names_upload = ["EEG Fpz-Cz", "EEG Pz-Oz", "EOG", "EMG"]

        if n_channels_raw < target_channels:
            # Pad with zeros for missing channels
            pad = np.zeros((n_samples, target_channels - n_channels_raw), dtype=raw_data.dtype)
            raw_data = np.concatenate([raw_data, pad], axis=1)
            st.info(f"Padded from {n_channels_raw} to {target_channels} channels (zero-filled).")
        elif n_channels_raw > target_channels:
            raw_data = raw_data[:, :target_channels]
            st.info(f"Trimmed from {n_channels_raw} to {target_channels} channels.")

        # ── Resample to 100 Hz if needed ────────────────────────────────
        st.divider()
        source_fs = st.number_input(
            "Source sampling rate (Hz)", min_value=1, max_value=1000, value=100, step=1,
        )

        if source_fs != cfg.sampling_rate:
            target_samples = int(n_samples * cfg.sampling_rate / source_fs)
            raw_data = resample(raw_data, target_samples, axis=0).astype(np.float32)
            st.info(f"Resampled from {source_fs} Hz to {cfg.sampling_rate} Hz ({target_samples} samples).")

        # ── Epoch the signal ────────────────────────────────────────────
        samples_per_epoch = cfg.samples_per_epoch
        total_samples = raw_data.shape[0]
        n_full_epochs = total_samples // samples_per_epoch

        if n_full_epochs < 1:
            st.error(
                f"Not enough data for one epoch. Need {samples_per_epoch} samples "
                f"at {cfg.sampling_rate} Hz ({cfg.epoch_seconds}s), got {total_samples}."
            )
            st.stop()

        # Trim to full epochs and reshape to [n_epochs, channels, samples_per_epoch]
        trimmed = raw_data[: n_full_epochs * samples_per_epoch]
        epochs = trimmed.reshape(n_full_epochs, target_channels, samples_per_epoch)

        # ── Preprocess each epoch ───────────────────────────────────────
        processed_epochs = np.zeros_like(epochs)
        for i in range(n_full_epochs):
            epoch = epochs[i].copy()
            for c in range(target_channels):
                epoch[c] = bandpass_filter(epoch[c], cfg.bandpass_low, cfg.bandpass_high, cfg.sampling_rate)
                epoch[c] = notch_filter(epoch[c], cfg.notch_freq, cfg.notch_quality, cfg.sampling_rate)
                epoch[c] = zscore_normalize(epoch[c])
            processed_epochs[i] = epoch

        st.markdown(
            f'<div class="sz-caption" style="margin-bottom:0.5rem;">'
            f"Processed: <strong>{n_full_epochs} epochs</strong> "
            f"({n_full_epochs * cfg.epoch_seconds}s of data)"
            f"</div>",
            unsafe_allow_html=True,
        )

        # ── Select sequence and target ──────────────────────────────────
        col_e1, col_e2 = st.columns(2)
        with col_e1:
            max_start = max(0, n_full_epochs - 10)
            start_epoch = st.slider(
                "Start epoch for 10-epoch sequence", 0, max_start, value=min(0, max_start),
            )
        with col_e2:
            target_upload = st.slider("Target epoch within sequence", 0, 9, value=9)

        # ── Build sequence [1, 10, 4, 3000] ────────────────────────────
        seq_len = 10
        end_epoch = min(start_epoch + seq_len, n_full_epochs)
        sequence = processed_epochs[start_epoch:end_epoch]

        if sequence.shape[0] < seq_len:
            # Pad with last epoch if sequence is too short
            pad_count = seq_len - sequence.shape[0]
            sequence = np.concatenate([sequence, np.tile(sequence[-1:], (pad_count, 1, 1))], axis=0)

        sequence = sequence[np.newaxis, ...]  # [1, 10, 4, 3000]

        # ── Run prediction ──────────────────────────────────────────────
        if st.button("Predict Sleep Stage", type="primary", use_container_width=True, key="predict_upload"):
            try:
                result = predictor.predict(sequence, target_epoch=target_upload)
                st.session_state.upload_prediction = result

                # QC on target epoch
                target_data = sequence[0, target_upload]
                qc = check_epoch_quality(target_data, CHANNEL_NAMES)
                qc_panel(qc)

                prediction_block(result)

                section_title("Probability Distribution")
                probability_bars(result.probabilities)

            except Exception as exc:
                st.error(f"Prediction failed: {exc}")

        elif st.session_state.get("upload_prediction") is not None:
            prediction_block(st.session_state.upload_prediction)
            section_title("Probability Distribution")
            probability_bars(st.session_state.upload_prediction.probabilities)

    except Exception as exc:
        st.error(f"Error processing file: {exc}")
