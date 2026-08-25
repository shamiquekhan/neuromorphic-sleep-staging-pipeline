"""Streamlit demo dashboard for Neuromorphic Sleep Stage Scoring."""

import sys
from pathlib import Path

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.inference.predict import load_model, run_inference, format_prediction
from src.models.improved_student import count_parameters

STAGE_NAMES = ["Wake", "N1", "N2", "N3", "REM"]
STAGE_COLORS = ["#FF6B6B", "#4ECDC4", "#45B7D1", "#96CEB4", "#FFEAA7"]

st.set_page_config(
    page_title="Neuromorphic Sleep Stage Scoring",
    page_icon="🧠",
    layout="wide",
)


@st.cache_resource
def load_final_model():
    checkpoint = PROJECT_ROOT / "artifacts" / "student_improved_best.pt"
    if not checkpoint.exists():
        return None
    return load_model(checkpoint)


def generate_demo_sequence(n_epochs=10):
    """Generate synthetic PSG-like data for demonstration."""
    fs = 100
    seq = np.zeros((n_epochs, 4, 30 * fs), dtype=np.float32)
    for i in range(n_epochs):
        t = np.linspace(0, 30, 30 * fs)
        stage = np.random.choice([0, 1, 2, 3, 4], p=[0.1, 0.05, 0.45, 0.25, 0.15])

        freq_map = {0: 15, 1: 7, 2: 12, 3: 2, 4: 10}
        amp_map = {0: 30, 1: 20, 2: 25, 3: 40, 4: 15}

        f = freq_map[stage]
        a = amp_map[stage]

        seq[i, 0, :] = a * np.sin(2 * np.pi * f * t) + np.random.randn(30 * fs) * 5
        seq[i, 1, :] = a * 0.8 * np.sin(2 * np.pi * f * t + 0.3) + np.random.randn(30 * fs) * 5
        seq[i, 2, :] = a * 0.5 * np.sin(2 * np.pi * 2 * t) + np.random.randn(30 * fs) * 3
        seq[i, 3, :] = a * 0.3 * np.sin(2 * np.pi * 1 * t) + np.random.randn(30 * fs) * 2

    return seq


def plot_psg_waveforms(epochs, title="PSG Waveforms"):
    """Plot multi-channel PSG waveforms."""
    fig, axes = plt.subplots(4, 1, figsize=(12, 6), sharex=True)
    channel_names = ["EEG Fpz-Cz", "EEG Pz-Oz", "EOG", "EMG"]
    fs = 100

    for ch in range(4):
        signal = epochs[:, ch, :].flatten()
        t = np.arange(len(signal)) / fs
        axes[ch].plot(t, signal, linewidth=0.5, color=STAGE_COLORS[ch])
        axes[ch].set_ylabel(channel_names[ch])
        axes[ch].grid(True, alpha=0.3)

    axes[-1].set_xlabel("Time (s)")
    axes[0].set_title(title)
    plt.tight_layout()
    return fig


def plot_probabilities(probs, preds):
    """Plot class probabilities as a stacked bar chart."""
    fig, ax = plt.subplots(figsize=(10, 4))
    x = np.arange(len(preds))
    bottom = np.zeros(len(preds))

    for i, stage in enumerate(STAGE_NAMES):
        ax.bar(x, probs[:, i], bottom=bottom, label=stage, color=STAGE_COLORS[i], alpha=0.8)
        bottom += probs[:, i]

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Probability")
    ax.set_title("Class Probabilities per Epoch")
    ax.legend(loc="upper right")
    ax.set_xticks(x)
    ax.set_xticklabels([f"E{i+1}" for i in range(len(preds))])
    plt.tight_layout()
    return fig


# ── Main App ──────────────────────────────────────────────────────────────────

st.title("🧠 Neuromorphic Sleep Stage Scoring")
st.markdown("**A Compact Deep Learning System for Five-Stage Sleep Classification**")

col1, col2 = st.columns([2, 1])

with col2:
    st.markdown("### Model Info")
    model = load_final_model()
    if model is not None:
        st.success("Final checkpoint loaded")
        st.metric("Parameters", f"{count_parameters(model):,}")
    else:
        st.error("Checkpoint not found at `artifacts/student_improved_best.pt`")

    st.markdown("---")
    st.markdown("**Target Classes:**")
    for name, color in zip(STAGE_NAMES, STAGE_COLORS):
        st.markdown(f" :{color.replace('#', '')}[{name}]")

with col1:
    tab1, tab2, tab3 = st.tabs(["Live Inference", "PSG Waveforms", "About"])

    with tab1:
        st.markdown("### Run Inference")

        if st.button("Load Sample Sequence"):
            st.session_state["demo_epochs"] = generate_demo_sequence()

        uploaded_file = st.file_uploader("Or upload a .npz file", type=["npz"])

        if uploaded_file is not None:
            data = np.load(uploaded_file)
            st.session_state["demo_epochs"] = data["epochs"]

        if "demo_epochs" in st.session_state and model is not None:
            epochs = st.session_state["demo_epochs"]
            st.info(f"Sequence shape: {epochs.shape} ({epochs.shape[0]} epochs)")

            if st.button("Run Inference"):
                probs, preds = run_inference(model, epochs)
                results = format_prediction(preds, probs)

                st.success("Inference complete")

                st.markdown("### Predicted Sleep Stages")
                hypnogram = " | ".join(f"**{r['predicted']}**" for r in results)
                st.markdown(hypnogram)

                st.markdown("### Per-Epoch Results")
                results_df = pd.DataFrame(results)
                st.dataframe(results_df.style.format({
                    "confidence": "{:.2%}",
                    **{f"prob_{s}": "{:.3f}" for s in STAGE_NAMES},
                }))

                fig = plot_probabilities(probs, preds)
                st.pyplot(fig)

    with tab2:
        st.markdown("### Multi-Channel PSG Waveforms")
        if "demo_epochs" in st.session_state:
            fig = plot_psg_waveforms(st.session_state["demo_epochs"])
            st.pyplot(fig)
        else:
            st.info("Load a sample sequence first (see Live Inference tab)")

    with tab3:
        st.markdown("""
        ## About This Project

        **Neuromorphic Sleep Stage Scoring** is an end-to-end deep-learning system for five-stage sleep classification from EEG, EOG, and EMG signals.

        ### Architecture
        - Multi-Resolution Stem
        - Depthwise-Separable CNN
        - Parametric Gabor Feature Extraction Block
        - 2-Layer GRU (300s context)
        - 5-Class Softmax

        ### Final Result
        | Metric | Value |
        |--------|-------|
        | Test Accuracy | 87.5% ± 3.2% |
        | Cohen's Kappa | 0.763 ± 0.043 |
        | Macro F1 | 0.721 ± 0.050 |
        | Parameters | 99,477 |

        ### Team
        - **Param Kaushik** — Dataset & Data Governance
        - **Suha Vora** — Signal Preprocessing
        - **Shailendra Bhatt** — Exploratory Data Analysis
        - **Shamique Khan** — Model Development & Training
        - **Aasir Jaffer Lone** — Evaluation & Performance
        """)


st.markdown("---")
st.caption("VIT Bhopal University — Neuromorphic Sleep Stage Scoring Project")
