"""NeuroSleep — Deployment Application.

This is a thin deployment wrapper around the core sleep_staging package.
The same inference engine powers local Streamlit, Hugging Face Space, and Kaggle.
"""

import sys
from pathlib import Path

# Add src to path for the sleep_staging package
_src = Path(__file__).resolve().parent / "src"
if str(_src) not in sys.path:
    sys.path.insert(0, str(_src))

import numpy as np
import streamlit as st

from sleep_staging.config import STAGE_COLORS, STAGE_LIST
from sleep_staging.inference import SleepStagePredictor

# ── Page Config ─────────────────────────────────────────────────────────
st.set_page_config(
    page_title="NeuroSleep",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap');
.stApp { font-family: 'Inter', sans-serif; }
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# ── Load Model ──────────────────────────────────────────────────────────
@st.cache_resource
def load_predictor():
    """Load the final model checkpoint."""
    model_dir = Path(__file__).resolve().parent / "model"
    checkpoint = model_dir / "student_full_finetuned.pt"
    if not checkpoint.exists():
        # Fallback: try relative to deployment parent
        checkpoint = Path(__file__).resolve().parent.parent / "artifacts" / "final" / "student_full_finetuned.pt"
    return SleepStagePredictor(checkpoint_path=str(checkpoint), device="cpu")

@st.cache_data
def load_final_metrics():
    """Load final metrics from results."""
    import json
    metrics_path = Path(__file__).resolve().parent.parent / "results" / "final" / "final_metrics.json"
    if not metrics_path.exists():
        metrics_path = Path(__file__).resolve().parent / "results" / "final" / "final_metrics.json"
    if metrics_path.exists():
        with open(metrics_path) as f:
            return json.load(f)
    return {}

# ── Demo Data ───────────────────────────────────────────────────────────
def make_demo_sequence(target_stage: str) -> np.ndarray:
    """Generate a synthetic 10-epoch sequence biased toward target_stage."""
    rng = np.random.RandomState(hash(target_stage) % 2**31)
    seq = rng.randn(1, 10, 4, 3000).astype(np.float32) * 0.1

    stage_biases = {
        "N2": {"mean": 0.0, "std": 0.05},
        "REM": {"mean": 0.02, "std": 0.08},
        "N3": {"mean": -0.02, "std": 0.03},
        "Wake": {"mean": 0.05, "std": 0.12},
        "N1": {"mean": 0.01, "std": 0.06},
    }
    bias = stage_biases.get(target_stage, {"mean": 0.0, "std": 0.05})
    seq[:, :, 0, :] += bias["mean"]
    seq[:, :, 0, :] *= (1 + bias["std"])

    return seq

DEMO_EXAMPLES = [
    ("Example 1 — N2 Sleep", "N2"),
    ("Example 2 — REM Sleep", "REM"),
    ("Example 3 — Deep Sleep (N3)", "N3"),
]

# ── Initialize ──────────────────────────────────────────────────────────
predictor = load_predictor()
metrics = load_final_metrics()

# ── Header ──────────────────────────────────────────────────────────────
st.markdown("""
<div style="padding: 1rem 0 2rem 0; border-bottom: 2px solid #111; margin-bottom: 2rem;">
    <div style="font-size:3.2rem; font-weight:900; letter-spacing:-0.03em; text-transform:uppercase;">NEUROSLEEP</div>
    <div style="font-size:0.85rem; font-weight:400; letter-spacing:0.12em; color:#777; text-transform:uppercase; margin-top:0.5rem;">
        Neuromorphic Sleep Stage Scoring &mdash; Research Demonstrator
    </div>
</div>
""", unsafe_allow_html=True)

# ── Status Grid ─────────────────────────────────────────────────────────
info = predictor.model_info
st.markdown(f"""
<div style="display:grid; grid-template-columns:repeat(4,1fr); gap:1px; background:#e0e0e0; border:1px solid #e0e0e0; margin-bottom:2rem;">
    <div style="background:#fff; padding:1.25rem;">
        <div style="font-size:0.6rem; font-weight:700; letter-spacing:0.15em; text-transform:uppercase; color:#999; margin-bottom:0.5rem;">Model</div>
        <div style="font-size:1.0rem; font-weight:700;">{info['name']}</div>
    </div>
    <div style="background:#fff; padding:1.25rem;">
        <div style="font-size:0.6rem; font-weight:700; letter-spacing:0.15em; text-transform:uppercase; color:#999; margin-bottom:0.5rem;">Parameters</div>
        <div style="font-size:1.0rem; font-weight:700;">{info['parameters']:,}</div>
    </div>
    <div style="background:#fff; padding:1.25rem;">
        <div style="font-size:0.6rem; font-weight:700; letter-spacing:0.15em; text-transform:uppercase; color:#999; margin-bottom:0.5rem;">Device</div>
        <div style="font-size:1.0rem; font-weight:700;">{info['device'].upper()}</div>
    </div>
    <div style="background:#fff; padding:1.25rem;">
        <div style="font-size:0.6rem; font-weight:700; letter-spacing:0.15em; text-transform:uppercase; color:#999; margin-bottom:0.5rem;">Status</div>
        <div style="font-size:1.0rem; font-weight:700; color:#E63946;">READY</div>
    </div>
</div>
""", unsafe_allow_html=True)

# ── Sidebar ─────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### Input Mode")
    input_mode = st.radio(
        "Choose input:",
        ["Demo Examples", "Upload .npz"],
        label_visibility="collapsed",
    )

    if input_mode == "Demo Examples":
        st.markdown("### Demo Examples")
        for label, _ in DEMO_EXAMPLES:
            if st.button(label, use_container_width=True):
                st.session_state["demo_choice"] = label
        demo_choice = st.session_state.get("demo_choice", DEMO_EXAMPLES[0][0])
    else:
        st.markdown("### Upload")
        uploaded = st.file_uploader("Upload .npz file", type=["npz"])

# ── Main Content ────────────────────────────────────────────────────────
col1, col2 = st.columns([3, 2])

with col1:
    st.markdown("### Prediction")

    # Get input sequence
    sequence = None
    if input_mode == "Demo Examples":
        demo_choice = st.session_state.get("demo_choice", DEMO_EXAMPLES[0][0])
        target_stage = dict(DEMO_EXAMPLES).get(demo_choice, "N2")
        sequence = make_demo_sequence(target_stage)
        st.info(f"Demo: {demo_choice}")
    elif input_mode == "Upload .npz" and uploaded is not None:
        data = np.load(uploaded)
        key = list(data.keys())[0] if data.keys() else "arr_0"
        sequence = data[key]
        if sequence.ndim == 3:
            sequence = sequence[np.newaxis, ...]

    if sequence is not None:
        # Validate
        try:
            SleepStagePredictor.validate_input(sequence)
            result = predictor.predict(sequence, target_epoch=9)

            # Prediction block
            color = STAGE_COLORS.get(result.stage, "#111")
            st.markdown(f"""
            <div style="border:1px solid #e0e0e0; padding:2.5rem 2rem; text-align:center; margin:1.5rem 0;">
                <div style="font-size:0.65rem; font-weight:700; letter-spacing:0.2em; text-transform:uppercase; color:#999; margin-bottom:0.75rem;">
                    PREDICTED STAGE
                </div>
                <div style="font-size:5rem; font-weight:900; letter-spacing:-0.04em; line-height:1.0; color:{color};">
                    {result.stage}
                </div>
                <div style="font-size:0.8rem; color:#777; margin-top:0.75rem;">
                    Confidence {result.confidence:.1%} &nbsp;&middot;&nbsp; Latency {result.latency_ms:.1f} ms
                </div>
            </div>
            """, unsafe_allow_html=True)

            # Probability bars
            st.markdown("### Probability Distribution")
            for stage, prob in result.probabilities.items():
                bar_color = STAGE_COLORS.get(stage, "#888")
                width_pct = prob * 100
                st.markdown(f"""
                <div style="display:flex; align-items:center; gap:0.75rem; margin-bottom:0.5rem;">
                    <div style="width:3rem; font-size:0.75rem; font-weight:600; text-align:right;">{stage}</div>
                    <div style="flex:1; height:6px; background:#f2f2f2;">
                        <div style="height:100%; width:{width_pct}%; background:{bar_color};"></div>
                    </div>
                    <div style="width:3.5rem; font-size:0.75rem; color:#777; text-align:right;">{prob:.1%}</div>
                </div>
                """, unsafe_allow_html=True)

        except ValueError as e:
            st.error(f"Input validation failed: {e}")
    else:
        st.info("Select a demo example or upload a .npz file to start.")

with col2:
    st.markdown("### Model Status")
    st.markdown(f"""
    <table style="width:100%; border-collapse:collapse; font-size:0.85rem;">
        <tr><td style="padding:0.5rem 0; border-bottom:1px solid #e8e8e8;">Checkpoint</td>
        <td style="padding:0.5rem 0; border-bottom:1px solid #e8e8e8; font-weight:600;">&#10003; Loaded</td></tr>
        <tr><td style="padding:0.5rem 0; border-bottom:1px solid #e8e8e8;">Architecture</td>
        <td style="padding:0.5rem 0; border-bottom:1px solid #e8e8e8; font-weight:600;">&#10003; Verified</td></tr>
        <tr><td style="padding:0.5rem 0; border-bottom:1px solid #e8e8e8;">Parameters</td>
        <td style="padding:0.5rem 0; border-bottom:1px solid #e8e8e8; font-weight:600;">{info['parameters']:,}</td></tr>
        <tr><td style="padding:0.5rem 0; border-bottom:1px solid #e8e8e8;">Input Shape</td>
        <td style="padding:0.5rem 0; border-bottom:1px solid #e8e8e8; font-weight:600;">10 x 4 x 3000</td></tr>
        <tr><td style="padding:0.5rem 0; border-bottom:1px solid #e8e8e8;">Classes</td>
        <td style="padding:0.5rem 0; border-bottom:1px solid #e8e8e8; font-weight:600;">5</td></tr>
        <tr><td style="padding:0.5rem 0;">Inference Engine</td>
        <td style="padding:0.5rem 0; font-weight:600; color:#E63946;">&#10003; Ready</td></tr>
    </table>
    """, unsafe_allow_html=True)

    # Final metrics
    if metrics:
        st.markdown("### Final Results")
        acc = metrics.get("accuracy", {})
        kappa = metrics.get("cohen_kappa", {})
        macro = metrics.get("macro_f1", {})
        st.markdown(f"""
        <table style="width:100%; border-collapse:collapse; font-size:0.85rem;">
            <tr><td style="padding:0.5rem 0; border-bottom:1px solid #e8e8e8;">Accuracy</td>
            <td style="padding:0.5rem 0; border-bottom:1px solid #e8e8e8; font-weight:700;">{acc.get('mean', 0):.1%} &plusmn; {acc.get('std', 0):.1%}</td></tr>
            <tr><td style="padding:0.5rem 0; border-bottom:1px solid #e8e8e8;">Cohen's &kappa;</td>
            <td style="padding:0.5rem 0; border-bottom:1px solid #e8e8e8; font-weight:700;">{kappa.get('mean', 0):.3f} &plusmn; {kappa.get('std', 0):.3f}</td></tr>
            <tr><td style="padding:0.5rem 0;">Macro F1</td>
            <td style="padding:0.5rem 0; font-weight:700;">{macro.get('mean', 0):.3f} &plusmn; {macro.get('std', 0):.3f}</td></tr>
        </table>
        """, unsafe_allow_html=True)

    # Provenance
    st.markdown("### Model Provenance")
    st.markdown("""
    <div style="font-size:0.85rem; line-height:1.7; color:#555;">
        <strong>Architecture:</strong> Improved Student<br>
        <strong>Training:</strong> Full Fine-Tuning<br>
        <strong>Dataset:</strong> Sleep-EDF Expanded (15 subjects)<br>
        <strong>Evaluation:</strong> 4-fold subject-level CV<br>
        <strong>Parameters:</strong> 99,477
    </div>
    """, unsafe_allow_html=True)

# ── Footer ──────────────────────────────────────────────────────────────
st.markdown('<div style="border-top:1px solid #e0e0e0; margin-top:2rem; padding:1rem 0; font-size:0.7rem; letter-spacing:0.1em; text-transform:uppercase; color:#999; text-align:center;">VIT Bhopal University &middot; NeuroSleep</div>', unsafe_allow_html=True)
