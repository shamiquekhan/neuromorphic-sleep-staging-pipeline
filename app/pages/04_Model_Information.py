"""Model Information page — architecture, results, LoRA, and reproducibility."""

import sys
from pathlib import Path

_repo = Path(__file__).resolve().parents[2]
if str(_repo / "src") not in sys.path:
    sys.path.insert(0, str(_repo / "src"))

import streamlit as st

from app.state import get_predictor
from sleep_staging.config import CHECKPOINT_PATH

st.set_page_config(page_title="NeuroSleep — Model Information", page_icon="🧠", layout="wide")

st.markdown(
    """
    <div style="text-align:center; padding: 0.5rem 0 1rem 0;">
        <h1 style="margin:0; font-size:2.2rem; letter-spacing:0.05em; color:#1a1a2e;">
            NEUROSLEEP
        </h1>
        <p style="margin:0; font-size:0.95rem; color:#666;">
            Model Information &amp; Reproducibility
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

predictor = get_predictor()
info = predictor.model_info

# ── Model Card ─────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("## Improved Student")

col1, col2 = st.columns(2)
with col1:
    st.markdown(f"""
    | Property | Value |
    |---|---|
    | Parameters | **{info['parameters']:,}** |
    | Classes | {info['n_classes']} |
    | Context | {info['seq_len']} x 30 s = {info['seq_len'] * 30} s |
    | Sampling rate | 100 Hz |
    | Device | {info['device'].upper()} |
    | Checkpoint | {CHECKPOINT_PATH.name} |
    | Adapter | {info.get('adapter') or 'None'} |
    """)

with col2:
    st.markdown("""
    ### Architecture

    ```
    4-channel PSG input
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
          Wake / N1 / N2 / N3 / REM
    ```
    """)

# ── LoRA Adaptation ───────────────────────────────────────────────────
st.markdown("---")
st.markdown("## LoRA Adaptation")

st.markdown("""
LoRA (Low-Rank Adaptation) enables parameter-efficient fine-tuning
by freezing the base model and training small adapter matrices.

| Property | Default |
|---|---|
| Target modules | `head` (classification) |
| Rank | 4 |
| Alpha | 8.0 |
| Scaling | 2.0 |
| Dropout | 0.05 |

**Train only the adapter** — base weights remain frozen.
""")

lora_dir = Path(_repo) / "artifacts" / "lora"
if lora_dir.exists():
    adapters = [p.name for p in lora_dir.iterdir() if p.is_dir()]
    if adapters:
        st.markdown("**Available adapters:**")
        for a in adapters:
            st.text(f"  - {a}")
    else:
        st.info("No trained adapters yet. Run `scripts/train_lora_adapter.py`.")
else:
    st.info("No LoRA artifacts directory found.")

# ── Official Results ──────────────────────────────────────────────────
st.markdown("---")
st.markdown("## Official Results")
st.markdown("""
| Metric | Value |
|---|---|
| Test Accuracy | **87.34%** |
| Cohen's κ | **0.7551** |
| Macro F1 | 0.6259 |
| Weighted F1 | 0.8653 |
| MGm | 0.5371 |
| CPU Latency | 8.5 ms/batch |
""")

# ── Reproducibility ───────────────────────────────────────────────────
st.markdown("---")
st.markdown("## Reproducibility")

try:
    import torch
    import mne
    import streamlit as stlib
    st.markdown(f"""
    | Component | Version |
    |---|---|
    | PyTorch | {torch.__version__} |
    | MNE | {mne.__version__} |
    | Streamlit | {stlib.__version__} |
    """)
except ImportError:
    st.info("Install all dependencies for full version info.")

st.markdown("""
**Dataset:** Sleep-EDF Expanded (PhysioNet)
**Training window:** 10 x 30 s epochs (300 s context)
**Final model:** Improved Student (`student_improved_best.pt`)
**Training:** Knowledge distillation from a larger teacher (research only)
""")
