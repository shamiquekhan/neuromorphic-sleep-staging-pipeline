"""Model Information page — architecture, results, LoRA, and reproducibility."""

import sys
from pathlib import Path

_repo = Path(__file__).resolve().parents[2]
if str(_repo / "src") not in sys.path:
    sys.path.insert(0, str(_repo / "src"))

import streamlit as st

from app.components import inject_swiss_css, header
from app.state import get_predictor
from sleep_staging.config import CHECKPOINT_PATH

st.set_page_config(page_title="NeuroSleep — Model Information", page_icon=None, layout="wide")
inject_swiss_css()
header()

predictor = get_predictor()
info = predictor.model_info

st.markdown('<div class="swiss-divider-thick"></div>', unsafe_allow_html=True)

# ── Model Card ───────────────────────────────────────────────────────────
st.markdown(
    """
    <div style="display:grid; grid-template-columns:1fr 1fr; gap:1px; background:#e0e0e0;
                border:1px solid #e0e0e0; margin-bottom:2rem;">
        <div style="background:#fff; padding:2rem;">
            <div class="swiss-section-title">Model Properties</div>
            <table class="swiss-table">
                <tr><td>Parameters</td><td>99,477</td></tr>
                <tr><td>Classes</td><td>5 (Wake, N1, N2, N3, REM)</td></tr>
                <tr><td>Context</td><td>10 &times; 30 s = 300 s</td></tr>
                <tr><td>Sampling Rate</td><td>100 Hz</td></tr>
                <tr><td>Device</td><td>{device}</td></tr>
                <tr><td>Checkpoint</td><td>{checkpoint}</td></tr>
                <tr><td>Adapter</td><td>{adapter}</td></tr>
            </table>
        </div>
        <div style="background:#fff; padding:2rem;">
            <div class="swiss-section-title">Architecture</div>
            <div class="swiss-arch">4-channel PSG input
      |
      v
Lite Multi-Resolution Stem
      |
      v
Depthwise-Separable CNN Encoder
      |
      +--------------------------+
      |                          |
      v                          v
Temporal CNN feature     Parametric Gabor FEB
      |                          |
      +------------+-------------+
                   v
             Feature fusion
                   |
                   v
              2-layer GRU
                   |
                   v
             5-class head
                   |
                   v
      Wake / N1 / N2 / N3 / REM</div>
        </div>
    </div>
    """.format(
        device=info["device"].upper(),
        checkpoint=CHECKPOINT_PATH.name,
        adapter=info.get("adapter") or "None",
    ),
    unsafe_allow_html=True,
)

# ── LoRA Adaptation ─────────────────────────────────────────────────────
st.markdown(
    """
    <div class="swiss-section-title">LoRA Adaptation</div>
    <div style="font-family:'Inter',sans-serif; font-size:0.9rem; line-height:1.7; color:#333; margin-bottom:1rem;">
        Low-Rank Adaptation (LoRA) enables parameter-efficient fine-tuning
        by freezing the base model and training small adapter matrices in the
        classification head. Only the adapter weights are updated; base
        weights remain frozen.
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <table class="swiss-table" style="max-width:400px;">
        <tr><td>Target modules</td><td>head (Linear 64 &rarr; 5)</td></tr>
        <tr><td>Rank</td><td>8</td></tr>
        <tr><td>Alpha</td><td>16.0</td></tr>
        <tr><td>Scaling</td><td>2.0</td></tr>
        <tr><td>Dropout</td><td>0.05</td></tr>
        <tr><td>Trainable params</td><td>552 (0.55%)</td></tr>
    </table>
    """,
    unsafe_allow_html=True,
)

lora_dir = Path(_repo) / "artifacts" / "lora"
if lora_dir.exists():
    adapters = [p.name for p in lora_dir.iterdir() if p.is_dir()]
    if adapters:
        st.markdown(
            f"""
            <div style="margin-top:1rem; font-family:'Inter',sans-serif; font-size:0.85rem; color:#555;">
                <strong>Available adapters:</strong> {', '.join(adapters)}
            </div>
            """,
            unsafe_allow_html=True,
        )

st.markdown(
    """
    <div class="swiss-section-title">Selective LoRA Experiment (15 Subjects, 4-Fold CV)</div>
    <table class="swiss-table" style="max-width:600px;">
        <tr><th>Config</th><th>Params</th><th>Accuracy</th><th>Macro F1</th><th>N1 F1</th><th>REM F1</th></tr>
        <tr><td>Frozen Base</td><td>0</td><td>83.0%</td><td>0.644</td><td><strong>0.586</strong></td><td>0.825</td></tr>
        <tr><td>LoRA Head</td><td>552</td><td>89.1%</td><td>0.686</td><td>0.340</td><td>0.883</td></tr>
        <tr><td>LoRA Gabor+Head</td><td>744</td><td>89.0%</td><td>0.686</td><td>0.343</td><td>0.893</td></tr>
        <tr><td>LoRA CNN+Head</td><td>552</td><td>89.1%</td><td>0.692</td><td>0.375</td><td>0.889</td></tr>
        <tr><td>LoRA All+Head</td><td>744</td><td>89.0%</td><td>0.689</td><td>0.362</td><td>0.890</td></tr>
        <tr><td><strong>Full FT</strong></td><td><strong>99,477</strong></td><td><strong>89.7%</strong></td><td><strong>0.783</strong></td><td><strong>0.817</strong></td><td><strong>0.922</strong></td></tr>
    </table>
    <div style="margin-top:0.5rem; font-family:'Inter',sans-serif; font-size:0.8rem; color:#888;">
        LoRA is ACTIVE DESTRUCTIVE to N1. Frozen base (0.586) outperforms all LoRA configs (0.340-0.375). Full FT is required for strong N1.
    </div>
    """,
    unsafe_allow_html=True,
)

# ── Official Results ────────────────────────────────────────────────────
st.markdown('<div class="swiss-divider"></div>', unsafe_allow_html=True)
st.markdown(
    """
    <div class="swiss-section-title">Official Results (15-Subject Expanded Dataset)</div>
    <table class="swiss-table" style="max-width:500px;">
        <tr><td>Test Accuracy</td><td><strong>87.5% &plusmn; 3.2%</strong></td></tr>
        <tr><td>Cohen's &kappa;</td><td><strong>0.763 &plusmn; 0.043</strong></td></tr>
        <tr><td>Macro F1</td><td>0.721 &plusmn; 0.050</td></tr>
        <tr><td>Parameters</td><td>99,477</td></tr>
        <tr><td>CPU Latency</td><td>8.5 ms/batch</td></tr>
    </table>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="swiss-section-title">Per-Class F1 (15-Subject 4-Fold CV)</div>
    <table class="swiss-table" style="max-width:400px;">
        <tr><td>Wake</td><td>0.961 &plusmn; 0.007</td></tr>
        <tr><td>N1</td><td><strong>0.720 &plusmn; 0.086</strong></td></tr>
        <tr><td>N2</td><td>0.845 &plusmn; 0.050</td></tr>
        <tr><td>N3</td><td>0.955 &plusmn; 0.017</td></tr>
        <tr><td>REM</td><td>0.918 &plusmn; 0.058</td></tr>
    </table>
    """,
    unsafe_allow_html=True,
)

# ── Reproducibility ─────────────────────────────────────────────────────
st.markdown('<div class="swiss-divider"></div>', unsafe_allow_html=True)
st.markdown(
    '<div class="swiss-section-title">Reproducibility</div>',
    unsafe_allow_html=True,
)

try:
    import torch
    import mne
    import streamlit as stlib

    st.markdown(
        f"""
        <table class="swiss-table" style="max-width:400px;">
            <tr><td>PyTorch</td><td>{torch.__version__}</td></tr>
            <tr><td>MNE</td><td>{mne.__version__}</td></tr>
            <tr><td>Streamlit</td><td>{stlib.__version__}</td></tr>
        </table>
        """,
        unsafe_allow_html=True,
    )
except ImportError:
    st.info("Install all dependencies for full version info.")

st.markdown(
    """
    <div style="margin-top:1rem; font-family:'Inter',sans-serif; font-size:0.85rem; line-height:1.7; color:#555;">
        <strong>Dataset:</strong> Sleep-EDF Expanded, 15 subjects (PhysioNet)<br>
        <strong>Training window:</strong> 10 &times; 30 s epochs (300 s context)<br>
        <strong>Final model:</strong> Improved Student (student_improved_best.pt)<br>
        <strong>Training:</strong> All-position supervision + N1/REM class weighting<br>
        <strong>Cross-validation:</strong> 4-fold subject-level CV
    </div>
    """,
    unsafe_allow_html=True,
)
